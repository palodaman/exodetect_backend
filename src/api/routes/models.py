"""
Model version management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from auth.dependencies import get_current_active_user, require_admin
from database.schema import User, ModelVersion, TrainingJob
from scheduler import training_scheduler


router = APIRouter(prefix="/api/models", tags=["models"])


class ModelVersionResponse(BaseModel):
    version: str
    model_name: str
    model_type: str
    training_date: datetime
    training_samples: int
    metrics: Dict[str, Any]
    status: str
    is_active: bool
    expires_at: datetime


class TrainingJobResponse(BaseModel):
    id: str
    model_name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    model_version: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]


@router.get("/versions", response_model=List[ModelVersionResponse])
async def list_model_versions(
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    days: int = Query(default=7, ge=1, le=90),
    current_user: User = Depends(get_current_active_user)
):
    """
    List available model versions
    Users can only see their organization's models
    """
    # Build filter
    filter_conditions = []

    # Date filter (last N days)
    start_date = datetime.utcnow() - timedelta(days=days)
    filter_conditions.append(ModelVersion.training_date >= start_date)

    # Organization filter
    if current_user.organization_id:
        filter_conditions.append(ModelVersion.organization_id == current_user.organization_id)

    # Model name filter
    if model_name:
        filter_conditions.append(ModelVersion.model_name == model_name)

    # Status filter
    if status:
        filter_conditions.append(ModelVersion.status == status)

    # Execute query
    versions = await ModelVersion.find(*filter_conditions).sort("-training_date").to_list()

    # Convert to response
    return [
        ModelVersionResponse(
            version=v.version,
            model_name=v.model_name,
            model_type=v.model_type,
            training_date=v.training_date,
            training_samples=v.training_samples,
            metrics=v.metrics,
            status=v.status,
            is_active=v.is_active,
            expires_at=v.expires_at
        )
        for v in versions
    ]


@router.get("/versions/{version}", response_model=ModelVersionResponse)
async def get_model_version(
    version: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get details of a specific model version"""
    model = await ModelVersion.find_one(ModelVersion.version == version)

    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Check organization access
    if current_user.organization_id and model.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return ModelVersionResponse(
        version=model.version,
        model_name=model.model_name,
        model_type=model.model_type,
        training_date=model.training_date,
        training_samples=model.training_samples,
        metrics=model.metrics,
        status=model.status,
        is_active=model.is_active,
        expires_at=model.expires_at
    )


@router.post("/versions/{version}/activate")
async def activate_model_version(
    version: str,
    current_user: User = Depends(require_admin)
):
    """
    Activate a model version (Admin only)
    Makes this version the active one for predictions
    """
    model = await ModelVersion.find_one(ModelVersion.version == version)

    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Check organization access
    if current_user.organization_id and model.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Deactivate other versions of the same model
    other_versions = await ModelVersion.find(
        ModelVersion.model_name == model.model_name,
        ModelVersion.version != version
    ).to_list()

    for other in other_versions:
        other.is_active = False
        await other.save()

    # Activate this version
    model.is_active = True
    model.status = "active"
    await model.save()

    return {
        "message": f"Model version {version} activated",
        "version": version,
        "model_name": model.model_name
    }


@router.delete("/versions/{version}")
async def delete_model_version(
    version: str,
    current_user: User = Depends(require_admin)
):
    """Delete a model version (Admin only)"""
    import os

    model = await ModelVersion.find_one(ModelVersion.version == version)

    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Check organization access
    if current_user.organization_id and model.organization_id != current_user.organization_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Don't delete active models
    if model.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active model version. Activate another version first."
        )

    # Delete file if exists
    if model.model_path and os.path.exists(model.model_path):
        os.remove(model.model_path)

    # Delete from database
    await model.delete()

    return {
        "message": f"Model version {version} deleted",
        "version": version
    }


@router.get("/training-jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    days: int = Query(default=7, ge=1, le=90),
    current_user: User = Depends(get_current_active_user)
):
    """List training jobs"""
    # Build filter
    filter_conditions = []

    # Date filter
    start_date = datetime.utcnow() - timedelta(days=days)
    filter_conditions.append(TrainingJob.started_at >= start_date)

    # Organization filter
    if current_user.organization_id:
        filter_conditions.append(TrainingJob.organization_id == current_user.organization_id)

    # Status filter
    if status:
        filter_conditions.append(TrainingJob.status == status)

    # Execute query
    jobs = await TrainingJob.find(*filter_conditions).sort("-started_at").to_list()

    return [
        TrainingJobResponse(
            id=str(job.id),
            model_name=job.model_name,
            status=job.status,
            started_at=job.started_at,
            completed_at=job.completed_at,
            model_version=job.model_version,
            metrics=job.metrics,
            error_message=job.error_message
        )
        for job in jobs
    ]


@router.post("/train")
async def trigger_training(
    model_name: str,
    days: int = Query(default=7, ge=1, le=90),
    incremental: bool = True,
    current_user: User = Depends(require_admin)
):
    """
    Manually trigger model training (Admin only)
    """
    import sys
    import os

    # Add scripts to path
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'scripts'
    ))

    from retrain_model import IncrementalModelTrainer

    # Create trainer
    trainer = IncrementalModelTrainer()

    # Trigger training in background
    import asyncio

    async def train():
        try:
            version = await trainer.retrain(
                model_name=model_name,
                days=days,
                min_samples=100,
                organization_id=current_user.organization_id,
                incremental=incremental
            )
            return version
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None

    # Start training task
    task = asyncio.create_task(train())

    return {
        "message": f"Training started for {model_name}",
        "model_name": model_name,
        "incremental": incremental,
        "days": days
    }


@router.get("/scheduler/status")
async def get_scheduler_status(current_user: User = Depends(require_admin)):
    """Get training scheduler status (Admin only)"""
    jobs = training_scheduler.get_jobs()

    return {
        "running": training_scheduler.is_running,
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None
            }
            for job in jobs
        ]
    }


@router.post("/scheduler/trigger")
async def trigger_scheduler_now(current_user: User = Depends(require_admin)):
    """Manually trigger scheduled retraining now (Admin only)"""
    training_scheduler.trigger_now()

    return {
        "message": "Retraining triggered",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/compare")
async def compare_model_versions(
    model_name: str,
    version1: str,
    version2: str,
    current_user: User = Depends(get_current_active_user)
):
    """Compare two model versions"""
    # Get both versions
    v1 = await ModelVersion.find_one(
        ModelVersion.model_name == model_name,
        ModelVersion.version == version1
    )
    v2 = await ModelVersion.find_one(
        ModelVersion.model_name == model_name,
        ModelVersion.version == version2
    )

    if not v1 or not v2:
        raise HTTPException(status_code=404, detail="One or both versions not found")

    # Check access
    if current_user.organization_id:
        if v1.organization_id != current_user.organization_id or v2.organization_id != current_user.organization_id:
            raise HTTPException(status_code=403, detail="Access denied")

    # Compare metrics
    comparison = {
        "model_name": model_name,
        "version1": {
            "version": v1.version,
            "training_date": v1.training_date.isoformat(),
            "training_samples": v1.training_samples,
            "metrics": v1.metrics,
            "status": v1.status,
            "is_active": v1.is_active
        },
        "version2": {
            "version": v2.version,
            "training_date": v2.training_date.isoformat(),
            "training_samples": v2.training_samples,
            "metrics": v2.metrics,
            "status": v2.status,
            "is_active": v2.is_active
        },
        "differences": {
            "sample_difference": v2.training_samples - v1.training_samples,
            "days_difference": (v2.training_date - v1.training_date).days
        }
    }

    return comparison

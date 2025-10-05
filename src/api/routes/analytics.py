"""
Analytics endpoints for query statistics and model performance
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from auth.dependencies import get_current_active_user, require_admin
from database.schema import User, QueryType
from query_logging.query_logger import query_logger


router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class QueryStatsResponse(BaseModel):
    total_queries: int
    avg_processing_time_ms: float
    queries_by_type: Dict[str, int]
    models_used: List[str]


class QueryListResponse(BaseModel):
    queries: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


@router.get("/stats", response_model=QueryStatsResponse)
async def get_query_statistics(
    organization_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get query statistics for the current user's organization
    Admin users can query any organization
    """
    # Determine which organization to query
    if organization_id and current_user.role != "admin":
        # Non-admin users can only query their own organization
        organization_id = current_user.organization_id
    elif not organization_id:
        organization_id = current_user.organization_id

    stats = await query_logger.get_query_stats(organization_id=organization_id)

    return QueryStatsResponse(**stats)


@router.get("/queries", response_model=QueryListResponse)
async def get_queries(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    query_type: Optional[QueryType] = None,
    limit: int = Query(default=100, le=1000),
    page: int = Query(default=1, ge=1),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get logged queries with pagination
    Users can only see their organization's queries
    """
    # Calculate skip offset
    skip = (page - 1) * limit

    # Get queries
    queries = await query_logger.get_queries_for_training(
        start_date=start_date,
        end_date=end_date,
        query_type=query_type,
        organization_id=current_user.organization_id,
        limit=limit
    )

    # Convert to dict
    query_list = [q.dict() for q in queries]

    return QueryListResponse(
        queries=query_list,
        total=len(query_list),
        page=page,
        page_size=limit
    )


@router.get("/export/training-data")
async def export_training_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    format: str = Query(default="json", regex="^(json|csv)$"),
    current_user: User = Depends(require_admin)
):
    """
    Export logged queries as training data (Admin only)
    Returns queries in a format suitable for model retraining
    """
    import json
    import csv
    from io import StringIO
    from fastapi.responses import StreamingResponse

    # Get queries
    queries = await query_logger.get_queries_for_training(
        start_date=start_date,
        end_date=end_date,
        organization_id=current_user.organization_id,
        limit=None  # Get all queries
    )

    if format == "json":
        # Export as JSON
        training_data = []
        for query in queries:
            training_data.append({
                "input": query.input_data,
                "output": query.prediction_result,
                "timestamp": query.created_at.isoformat(),
                "model_version": query.model_version
            })

        content = json.dumps(training_data, indent=2)
        media_type = "application/json"
        filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    else:  # CSV
        # Export as CSV
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "timestamp", "query_type", "model_version",
            "model_probability", "model_label",
            "processing_time_ms"
        ])

        # Write data
        for query in queries:
            writer.writerow([
                query.created_at.isoformat(),
                query.query_type,
                query.model_version,
                query.prediction_result.get("model_probability_candidate", ""),
                query.prediction_result.get("model_label", ""),
                query.processing_time_ms
            ])

        content = output.getvalue()
        media_type = "text/csv"
        filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/model-performance")
async def get_model_performance(
    model_version: Optional[str] = None,
    days: int = Query(default=7, ge=1, le=90),
    current_user: User = Depends(require_admin)
):
    """
    Get model performance metrics (Admin only)
    Shows prediction distribution and processing times by model version
    """
    from database.schema import Query as QueryModel

    # Calculate date range
    start_date = datetime.utcnow() - timedelta(days=days)

    # Build query
    filter_conditions = [
        QueryModel.created_at >= start_date,
        QueryModel.organization_id == current_user.organization_id
    ]

    if model_version:
        filter_conditions.append(QueryModel.model_version == model_version)

    # Get queries
    queries = await QueryModel.find(*filter_conditions).to_list()

    # Calculate metrics
    total_queries = len(queries)
    avg_processing_time = sum(q.processing_time_ms for q in queries) / total_queries if total_queries > 0 else 0

    # Count predictions by label
    label_counts = {}
    for q in queries:
        label = q.prediction_result.get("model_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    # Group by model version
    model_stats = {}
    for q in queries:
        version = q.model_version
        if version not in model_stats:
            model_stats[version] = {
                "count": 0,
                "avg_processing_time": 0,
                "total_time": 0
            }

        model_stats[version]["count"] += 1
        model_stats[version]["total_time"] += q.processing_time_ms

    # Calculate averages
    for version in model_stats:
        count = model_stats[version]["count"]
        model_stats[version]["avg_processing_time"] = model_stats[version]["total_time"] / count

    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "days": days
        },
        "summary": {
            "total_queries": total_queries,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "prediction_distribution": label_counts
        },
        "by_model": model_stats
    }

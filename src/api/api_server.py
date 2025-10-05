from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import json
import os
import io
import tempfile
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.exoplanet_predictor import ExoplanetPredictor
from ml.light_curve_processor import LightCurveProcessor
from utils.archive_fetcher import ArchiveFetcher
from database import init_db, close_db
from api.routes import auth as auth_router
from logging import query_logger
from logging.middleware import QueryLoggingMiddleware
from scheduler import training_scheduler

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ExoDetect API",
    description="Exoplanet Detection and Classification API with MongoDB & Authentication",
    version="3.0.0"
)


# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection, query logger, and scheduler on startup"""
    try:
        await init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        print("⚠️  Running without database - authentication and logging disabled")

    # Start query logger
    try:
        await query_logger.start()
    except Exception as e:
        print(f"⚠️  Failed to start query logger: {e}")

    # Start training scheduler (only in production)
    enable_scheduler = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
    if enable_scheduler:
        try:
            training_scheduler.start()
        except Exception as e:
            print(f"⚠️  Failed to start training scheduler: {e}")
    else:
        print("ℹ️  Training scheduler disabled (set ENABLE_SCHEDULER=true to enable)")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection, query logger, and scheduler on shutdown"""
    # Stop training scheduler
    try:
        training_scheduler.stop()
    except Exception as e:
        print(f"⚠️  Failed to stop training scheduler: {e}")

    # Stop query logger
    try:
        await query_logger.stop()
    except Exception as e:
        print(f"⚠️  Failed to stop query logger: {e}")

    await close_db()
    print("Database connection closed")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add query logging middleware
app.add_middleware(QueryLoggingMiddleware)

# Include authentication routes
app.include_router(auth_router.router)

# Include analytics routes
from api.routes import analytics as analytics_router
app.include_router(analytics_router.router)

# Include model management routes
from api.routes import models as models_router
app.include_router(models_router.router)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Initialize predictors and archive fetcher (lazy loading)
predictors = {}
archive_fetcher = None

def get_predictor(model_name="xgboost_enhanced"):
    """Get or create predictor instance"""
    if model_name not in predictors:
        predictors[model_name] = ExoplanetPredictor(model_name=model_name)
    return predictors[model_name]

def get_archive_fetcher():
    """Get or create archive fetcher instance"""
    global archive_fetcher
    if archive_fetcher is None:
        archive_fetcher = ArchiveFetcher()
    return archive_fetcher


# Request/Response Models
class LightCurveData(BaseModel):
    time: List[float] = Field(..., description="Time values in days")
    flux: List[float] = Field(..., description="Flux measurements")
    flux_err: Optional[List[float]] = Field(None, description="Flux uncertainties")

    class Config:
        schema_extra = {
            "example": {
                "time": [0.0, 0.1, 0.2],
                "flux": [1.0001, 0.9999, 1.0002],
                "flux_err": [0.0001, 0.0001, 0.0001]
            }
        }


class FeatureData(BaseModel):
    period: float = Field(..., gt=0, description="Orbital period in days")
    duration: float = Field(..., gt=0, description="Transit duration in hours")
    depth: float = Field(..., ge=0, description="Transit depth in ppm")
    snr: float = Field(..., gt=0, description="Signal-to-noise ratio")
    impact: Optional[float] = Field(None, ge=0, le=1.5, description="Impact parameter")
    star_teff: Optional[float] = Field(None, gt=2500, lt=10000, description="Stellar temperature")
    star_logg: Optional[float] = Field(None, gt=3, lt=5, description="Stellar log(g)")
    star_radius: Optional[float] = Field(None, gt=0.1, lt=20, description="Stellar radius")

    class Config:
        schema_extra = {
            "example": {
                "period": 10.5,
                "duration": 3.2,
                "depth": 500,
                "snr": 15.0,
                "impact": 0.5,
                "star_teff": 5778,
                "star_logg": 4.44,
                "star_radius": 1.0
            }
        }


class ArchiveQuery(BaseModel):
    identifier: str = Field(..., description="KOI/TOI/KIC/TIC identifier")
    mission: str = Field("Kepler", description="Mission name")
    include_light_curve: bool = Field(False, description="Fetch light curve data")

    class Config:
        schema_extra = {
            "example": {
                "identifier": "K02365.01",
                "mission": "Kepler",
                "include_light_curve": False
            }
        }


class PredictionResponse(BaseModel):
    target: str
    model_probability_candidate: float
    model_label: str
    confidence: str
    transit_params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    top_features: Optional[List[Dict[str, float]]] = None
    reasoning: Optional[List[str]] = None
    archive_snapshot: Optional[Dict[str, Any]] = None
    model_name: str
    model_version: str
    processing_time: float


class BatchPredictionRequest(BaseModel):
    targets: List[FeatureData]
    model_name: Optional[str] = "xgboost_enhanced"


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "ExoDetect API",
        "version": "2.0.0",
        "endpoints": {
            "predict_light_curve": "/api/predict/light-curve",
            "predict_features": "/api/predict/features",
            "predict_archive": "/api/predict/archive",
            "upload_light_curve": "/api/predict/upload",
            "batch_predict": "/api/predict/batch",
            "models": "/api/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are accessible
        model_dir = "models_enhanced" if os.path.exists("models_enhanced") else "models"
        models = [f.replace("_model.pkl", "") for f in os.listdir(model_dir) if f.endswith("_model.pkl")]

        return {
            "status": "healthy",
            "available_models": models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    model_dir = "models_enhanced" if os.path.exists("models_enhanced") else "models"
    models = []

    for f in os.listdir(model_dir):
        if f.endswith("_model.pkl"):
            model_name = f.replace("_model.pkl", "")
            models.append({
                "name": model_name,
                "type": model_name.split("_")[0],
                "enhanced": "enhanced" in model_name
            })

    return {"models": models}


@app.post("/api/predict/light-curve", response_model=PredictionResponse)
async def predict_from_light_curve(data: LightCurveData, model_name: str = "xgboost_enhanced"):
    """Predict exoplanet from raw light curve data"""
    start_time = datetime.now()

    try:
        predictor = get_predictor(model_name)

        # Convert to numpy arrays
        time = np.array(data.time)
        flux = np.array(data.flux)
        flux_err = np.array(data.flux_err) if data.flux_err else None

        # Run prediction in thread pool (CPU-intensive)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            predictor.predict_from_light_curve,
            time, flux, flux_err, True
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            target="uploaded_light_curve",
            model_probability_candidate=results["model_probability_candidate"],
            model_label=results["model_label"],
            confidence=results["confidence"],
            transit_params=results.get("transit_params"),
            features=results.get("features"),
            top_features=results.get("top_features"),
            reasoning=results.get("reasoning"),
            model_name=results["model_name"],
            model_version=results.get("model_version", "2.0"),
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/features", response_model=PredictionResponse)
async def predict_from_features(data: FeatureData, model_name: str = "xgboost_enhanced"):
    """Predict exoplanet from pre-extracted features"""
    start_time = datetime.now()

    try:
        predictor = get_predictor(model_name)

        # Convert to dict
        features_dict = data.dict(exclude_none=True)

        # Run prediction
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            predictor.predict_from_features,
            features_dict
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            target="feature_input",
            model_probability_candidate=results["model_probability_candidate"],
            model_label=results["model_label"],
            confidence=results["confidence"],
            features=features_dict,
            model_name=results["model_name"],
            model_version="2.0",
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/upload")
async def predict_from_upload(
    file: UploadFile = File(...),
    model_name: str = "xgboost_enhanced"
):
    """Upload and predict from CSV or FITS file"""
    start_time = datetime.now()

    if not file.filename.endswith(('.csv', '.fits')):
        raise HTTPException(status_code=400, detail="Only CSV and FITS files are supported")

    try:
        predictor = get_predictor(model_name)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-4:]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Process file
        file_format = 'fits' if file.filename.endswith('.fits') else 'csv'

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            predictor.predict_from_file,
            tmp_file_path, file_format
        )

        # Clean up
        os.unlink(tmp_file_path)

        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            target=file.filename,
            model_probability_candidate=results["model_probability_candidate"],
            model_label=results["model_label"],
            confidence=results["confidence"],
            transit_params=results.get("transit_params"),
            features=results.get("features"),
            model_name=model_name,
            model_version="2.0",
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.post("/api/predict/archive", response_model=PredictionResponse)
async def predict_from_archive(query: ArchiveQuery, model_name: str = "xgboost_enhanced"):
    """Predict from archive object (KOI/TOI/KIC/TIC/EPIC)"""
    start_time = datetime.now()

    try:
        predictor = get_predictor(model_name)
        fetcher = get_archive_fetcher()

        # Fetch archive data
        loop = asyncio.get_event_loop()

        # Determine if this is a KOI or TOI
        identifier_upper = query.identifier.upper()
        archive_data = None

        if 'KOI' in identifier_upper or 'K' == identifier_upper[0]:
            # Fetch KOI data
            archive_data = await loop.run_in_executor(
                executor,
                fetcher.fetch_koi_data,
                query.identifier
            )
        elif 'TOI' in identifier_upper:
            # Fetch TOI data
            archive_data = await loop.run_in_executor(
                executor,
                fetcher.fetch_toi_data,
                query.identifier
            )
        else:
            # Generic identifier (KIC, TIC, EPIC)
            archive_data = {"identifier": query.identifier}

        # Fetch light curve if requested
        time_arr = None
        flux_arr = None
        flux_err_arr = None

        if query.include_light_curve:
            try:
                time_arr, flux_arr, flux_err_arr = await loop.run_in_executor(
                    executor,
                    fetcher.fetch_light_curve,
                    query.identifier,
                    query.mission
                )
            except Exception as lc_error:
                print(f"Warning: Could not fetch light curve: {lc_error}")
                # Continue without light curve

        # Run prediction
        if time_arr is not None and flux_arr is not None:
            # Predict from light curve
            results = await loop.run_in_executor(
                executor,
                predictor.predict_from_light_curve,
                time_arr, flux_arr, flux_err_arr, True
            )
        elif archive_data and ('koi_period' in archive_data or 'toi_period' in archive_data):
            # Predict from archive features
            features = {}

            # Map KOI features
            if 'koi_period' in archive_data:
                features['period'] = archive_data.get('koi_period', 0)
                features['duration'] = archive_data.get('koi_duration', 0)
                features['depth'] = archive_data.get('koi_depth', 0) * 1e6  # Convert to ppm
                features['star_teff'] = archive_data.get('koi_steff', 5778)
                features['star_logg'] = archive_data.get('koi_slogg', 4.4)
                features['star_radius'] = archive_data.get('koi_srad', 1.0)
            # Map TOI features
            elif 'toi_period' in archive_data:
                features['period'] = archive_data.get('toi_period', 0)
                features['duration'] = archive_data.get('toi_duration', 0)
                features['depth'] = archive_data.get('toi_depth', 0) * 1e6  # Convert to ppm
                features['star_teff'] = archive_data.get('st_teff', 5778)
                features['star_logg'] = archive_data.get('st_logg', 4.4)
                features['star_radius'] = archive_data.get('st_rad', 1.0)

            # Estimate SNR (simplified)
            if features.get('depth', 0) > 0:
                features['snr'] = features['depth'] / 100.0  # Rough estimate
            else:
                features['snr'] = 7.0

            results = await loop.run_in_executor(
                executor,
                predictor.predict_from_features,
                features
            )
            results['features'] = features
        else:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data to make prediction. Archive data incomplete."
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            target=query.identifier,
            model_probability_candidate=results["model_probability_candidate"],
            model_label=results["model_label"],
            confidence=results.get("confidence", "Unknown"),
            transit_params=results.get("transit_params"),
            features=results.get("features"),
            top_features=results.get("top_features"),
            reasoning=results.get("reasoning"),
            archive_snapshot=archive_data,
            model_name=results.get("model_name", model_name),
            model_version=results.get("model_version", "2.0"),
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive prediction failed: {str(e)}")


@app.post("/api/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple targets"""
    results = []

    for target in request.targets:
        try:
            prediction = await predict_from_features(target, request.model_name)
            results.append(prediction.dict())
        except Exception as e:
            results.append({"error": str(e), "target": target.dict()})

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn

    # For local development
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
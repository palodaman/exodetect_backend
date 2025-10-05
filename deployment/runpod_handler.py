"""
RunPod serverless handler for ExoDetect API
"""
import runpod
import numpy as np
from exoplanet_predictor import ExoplanetPredictor
import json

# Initialize predictors
predictors = {}

def get_predictor(model_name="xgboost_enhanced"):
    """Get or create predictor instance"""
    if model_name not in predictors:
        predictors[model_name] = ExoplanetPredictor(model_name=model_name)
    return predictors[model_name]


def handler(job):
    """
    RunPod handler function

    Input format:
    {
        "input": {
            "type": "light_curve" | "features" | "file",
            "model_name": "xgboost_enhanced",
            "data": {...}
        }
    }
    """
    job_input = job["input"]

    try:
        prediction_type = job_input.get("type", "features")
        model_name = job_input.get("model_name", "xgboost_enhanced")
        data = job_input.get("data", {})

        predictor = get_predictor(model_name)

        if prediction_type == "light_curve":
            # Process light curve
            time = np.array(data["time"])
            flux = np.array(data["flux"])
            flux_err = np.array(data.get("flux_err")) if "flux_err" in data else None

            result = predictor.predict_from_light_curve(time, flux, flux_err)

        elif prediction_type == "features":
            # Process features
            result = predictor.predict_from_features(data)

        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        result = convert_types(result)

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
import numpy as np
import pandas as pd
import joblib
import json
import os

class ExoplanetPredictor:
    def __init__(self, model_name='xgboost'):
        """Initialize the predictor with a trained model"""
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load the trained model, scaler, and feature names"""
        model_path = f"models/{self.model_name}_model.pkl"
        scaler_path = "models/scaler.pkl"
        feature_names_path = "models/feature_names.json"

        if not os.path.exists(model_path):
            available_models = [f.replace('_model.pkl', '')
                              for f in os.listdir('models')
                              if f.endswith('_model.pkl')]
            raise ValueError(f"Model {self.model_name} not found. Available models: {available_models}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        print(f"Loaded {self.model_name} model successfully")

    def prepare_features(self, data):
        """Prepare features from input data"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        if 'period_to_duration_ratio' not in df.columns:
            if 'koi_period' in df.columns and 'koi_duration' in df.columns:
                df['period_to_duration_ratio'] = df['koi_period'] / (df['koi_duration'] / 24)

        if 'transit_depth_snr' not in df.columns:
            if 'koi_depth' in df.columns and 'koi_model_snr' in df.columns:
                df['transit_depth_snr'] = df['koi_depth'] * df['koi_model_snr']

        if 'stellar_flux' not in df.columns:
            if 'koi_insol' in df.columns and 'koi_srad' in df.columns:
                df['stellar_flux'] = df['koi_insol'] * df['koi_srad'] ** 2

        if 'log_period' not in df.columns and 'koi_period' in df.columns:
            df['log_period'] = np.log1p(df['koi_period'])

        if 'log_depth' not in df.columns and 'koi_depth' in df.columns:
            df['log_depth'] = np.log1p(df['koi_depth'])

        if 'log_snr' not in df.columns and 'koi_model_snr' in df.columns:
            df['log_snr'] = np.log1p(df['koi_model_snr'])

        for col in self.feature_names:
            if col not in df.columns:
                if col.startswith('koi_fpflag'):
                    df[col] = 0
                else:
                    df[col] = np.nan

        return df[self.feature_names]

    def predict(self, data, return_probability=True):
        """Make predictions on input data"""
        features = self.prepare_features(data)

        features_scaled = self.scaler.transform(features)

        if return_probability:
            probabilities = self.model.predict_proba(features_scaled)
            predictions = probabilities[:, 1]
        else:
            predictions = self.model.predict(features_scaled)

        return predictions

    def predict_with_explanation(self, data, threshold=0.5):
        """Make prediction with detailed explanation"""
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)

        probability = self.model.predict_proba(features_scaled)[0, 1]
        prediction = "Likely Candidate" if probability >= threshold else "Likely False Positive"

        result = {
            'model_probability_candidate': float(probability),
            'model_label': prediction,
            'threshold_used': threshold,
            'model_name': self.model_name,
            'features_used': self.feature_names
        }

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            result['top_features'] = [
                {'feature': name, 'importance': float(imp)}
                for name, imp in feature_importance
            ]

        return result


def predict_from_csv(csv_path, model_name='xgboost', output_path=None):
    """Make predictions on a CSV file"""
    predictor = ExoplanetPredictor(model_name=model_name)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    probabilities = predictor.predict(df)

    df['predicted_probability'] = probabilities
    df['predicted_label'] = df['predicted_probability'].apply(
        lambda p: "Likely Candidate" if p >= 0.5 else "Likely False Positive"
    )

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return df


def predict_single_target(target_features, model_name='xgboost'):
    """Make prediction for a single target"""
    predictor = ExoplanetPredictor(model_name=model_name)

    result = predictor.predict_with_explanation(target_features)

    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Model: {result['model_name']}")
    print(f"Probability (Candidate): {result['model_probability_candidate']:.4f}")
    print(f"Classification: {result['model_label']}")

    if 'top_features' in result:
        print("\nTop Contributing Features:")
        for feat in result['top_features']:
            print(f"  - {feat['feature']}: {feat['importance']:.4f}")

    return result


if __name__ == "__main__":
    example_target = {
        'koi_period': 9.488036,
        'koi_impact': 0.586,
        'koi_duration': 2.95750,
        'koi_depth': 874.8,
        'koi_prad': 2.26,
        'koi_teq': 793,
        'koi_insol': 9.11,
        'koi_model_snr': 35.8,
        'koi_steff': 5455,
        'koi_slogg': 4.467,
        'koi_srad': 0.927,
        'koi_kepmag': 15.347,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0
    }

    result = predict_single_target(example_target)

    print("\nFull result object:")
    print(json.dumps(result, indent=2))
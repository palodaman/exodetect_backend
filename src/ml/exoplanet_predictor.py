import numpy as np
import pandas as pd
import joblib
import json
import os
from .light_curve_processor import LightCurveProcessor


class ExoplanetPredictor:
    """End-to-end prediction system from raw light curves to exoplanet classification"""

    def __init__(self, model_name='xgboost_enhanced', model_dir='models_enhanced'):
        """Initialize predictor with specified model"""
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.light_curve_processor = LightCurveProcessor()

        # Get absolute path to models directory
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(backend_root, model_dir)

        # Try enhanced models first, fallback to original
        if not os.path.exists(self.model_dir):
            print(f"Enhanced models not found at {self.model_dir}, using original models...")
            self.model_dir = os.path.join(backend_root, 'models')
            self.model_name = model_name.replace('_enhanced', '')

        self.load_model()

    def load_model(self):
        """Load trained model, scaler, and feature names"""
        model_path = f"{self.model_dir}/{self.model_name}_model.pkl"
        scaler_path = f"{self.model_dir}/scaler.pkl"
        feature_names_path = f"{self.model_dir}/feature_names.json"

        if not os.path.exists(model_path):
            # Debug output
            print(f"❌ Model not found at: {model_path}")
            print(f"   Model dir exists: {os.path.exists(self.model_dir)}")
            print(f"   Working directory: {os.getcwd()}")
            print(f"   __file__ location: {__file__}")
            print(f"   Backend root: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

            # List available models if directory exists
            available_models = []
            if os.path.exists(self.model_dir):
                print(f"   Files in {self.model_dir}:")
                for f in os.listdir(self.model_dir):
                    print(f"      - {f}")

                available_models = [f.replace('_model.pkl', '')
                                  for f in os.listdir(self.model_dir)
                                  if f.endswith('_model.pkl')]

            # Also check if models directory exists
            alt_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
            if os.path.exists(alt_model_dir):
                print(f"   Files in {alt_model_dir}:")
                for f in os.listdir(alt_model_dir):
                    if f.endswith('.pkl'):
                        print(f"      - {f}")

            raise ValueError(f"Model {self.model_name} not found. Available: {available_models}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)

        print(f"Loaded {self.model_name} model successfully")

    def predict_from_light_curve(self, time, flux, flux_err=None, return_details=True):
        """
        Predict exoplanet probability from raw light curve data

        Parameters:
        -----------
        time : array-like
            Time values (in days)
        flux : array-like
            Flux measurements
        flux_err : array-like, optional
            Flux uncertainties
        return_details : bool
            Whether to return detailed results

        Returns:
        --------
        dict : Prediction results with probability and classification
        """
        # Process light curve
        self.light_curve_processor.load_light_curve(time, flux, flux_err)
        self.light_curve_processor.clean_outliers()
        self.light_curve_processor.detrend()

        # Find transits
        transit_params = self.light_curve_processor.find_transits_bls()

        # Extract features
        features = self.light_curve_processor.extract_transit_features(transit_params)

        # Prepare features for model
        feature_vector = self._prepare_features(features)

        # Make prediction
        probability = self._predict_probability(feature_vector)

        # Prepare results
        results = {
            'model_probability_candidate': float(probability),
            'model_label': 'Likely Candidate' if probability >= 0.5 else 'Likely False Positive',
            'confidence': self._calculate_confidence(probability),
            'model_name': self.model_name,
            'model_version': 'enhanced_v2.0'
        }

        if return_details:
            results['transit_params'] = {
                'period_days': transit_params['period'],
                'duration_hours': transit_params['duration'],
                'depth_ppm': transit_params['depth'],
                'snr': transit_params['snr'],
                'epoch': transit_params['epoch'],
                'transit_count': transit_params.get('transit_count', 0)
            }

            results['features'] = features

            # Add feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = sorted(
                    zip(self.feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                results['top_features'] = [
                    {name: float(imp)}
                    for name, imp in feature_importance
                ]

            # Classification reasoning
            results['reasoning'] = self._generate_reasoning(features, probability)

        return results

    def predict_from_file(self, filepath, file_format='csv'):
        """
        Predict from a light curve file

        Parameters:
        -----------
        filepath : str
            Path to light curve file
        file_format : str
            Format of file ('csv' or 'fits')

        Returns:
        --------
        dict : Prediction results
        """
        features, transit_params = self.light_curve_processor.process_light_curve_file(
            filepath, file_format
        )

        # Prepare features for model
        feature_vector = self._prepare_features(features)

        # Make prediction
        probability = self._predict_probability(feature_vector)

        results = {
            'filename': os.path.basename(filepath),
            'model_probability_candidate': float(probability),
            'model_label': 'Likely Candidate' if probability >= 0.5 else 'Likely False Positive',
            'confidence': self._calculate_confidence(probability),
            'transit_params': transit_params,
            'features': features,
            'model_name': self.model_name,
            'model_version': 'enhanced_v2.0'
        }

        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            results['top_features'] = [
                {name: float(imp)}
                for name, imp in feature_importance
            ]

        # Add reasoning
        results['reasoning'] = self._generate_reasoning(features, probability)

        return results

    def predict_from_features(self, features_dict):
        """
        Predict from pre-extracted features

        Parameters:
        -----------
        features_dict : dict
            Dictionary of feature values

        Returns:
        --------
        dict : Prediction results
        """
        feature_vector = self._prepare_features(features_dict)
        probability = self._predict_probability(feature_vector)

        return {
            'model_probability_candidate': float(probability),
            'model_label': 'Likely Candidate' if probability >= 0.5 else 'Likely False Positive',
            'confidence': self._calculate_confidence(probability),
            'model_name': self.model_name
        }

    def _prepare_features(self, features_dict):
        """Prepare feature vector for model input"""
        # Create DataFrame with single row
        df = pd.DataFrame([features_dict])

        # Add missing features with default values
        for feature in self.feature_names:
            if feature not in df.columns:
                # Handle different feature types
                if 'flag_' in feature:
                    df[feature] = 0
                elif 'temp_' in feature or 'size_' in feature:
                    df[feature] = 0
                else:
                    df[feature] = np.nan

        # Select only required features
        df = df[self.feature_names]

        # Fill NaN values
        for col in df.columns:
            if df[col].isna().any():
                if 'flag_' in col:
                    df[col] = 0
                else:
                    # Use median from training (stored in scaler)
                    df[col] = 0  # Simplified; in production would store medians

        return df

    def _predict_probability(self, feature_vector):
        """Make probability prediction"""
        # Scale features
        features_scaled = self.scaler.transform(feature_vector)

        # Predict probability
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features_scaled)[0, 1]
        else:
            # Fallback for models without predict_proba
            prediction = self.model.predict(features_scaled)[0]
            probability = float(prediction)

        return probability

    def _calculate_confidence(self, probability):
        """Calculate confidence level from probability"""
        distance_from_boundary = abs(probability - 0.5)

        if distance_from_boundary > 0.4:
            return 'Very High'
        elif distance_from_boundary > 0.3:
            return 'High'
        elif distance_from_boundary > 0.2:
            return 'Medium'
        elif distance_from_boundary > 0.1:
            return 'Low'
        else:
            return 'Very Low'

    def _generate_reasoning(self, features, probability):
        """Generate human-readable reasoning for prediction"""
        reasons = []

        # SNR-based reasoning
        if features.get('snr', 0) > 15:
            reasons.append("Strong transit signal detected (high SNR)")
        elif features.get('snr', 0) < 7:
            reasons.append("Weak transit signal (low SNR)")

        # Depth-based reasoning
        if features.get('depth', 0) > 1000:
            reasons.append("Deep transit consistent with planet")
        elif features.get('depth', 0) < 100:
            reasons.append("Very shallow transit depth")

        # Period-based reasoning
        if 0.5 < features.get('period', 0) < 500:
            reasons.append("Orbital period within typical range")
        elif features.get('period', 0) > 500:
            reasons.append("Unusually long orbital period")

        # Vetting flags
        if features.get('flag_stellar_eclipse', 0):
            reasons.append("Warning: Possible stellar eclipse (odd-even mismatch)")
        if features.get('flag_not_transit', 0):
            reasons.append("Warning: May not be a transit signal")

        # Overall assessment
        if probability > 0.8:
            reasons.append("Strong candidate for exoplanet")
        elif probability > 0.6:
            reasons.append("Good candidate requiring follow-up")
        elif probability > 0.4:
            reasons.append("Marginal candidate, additional vetting needed")
        else:
            reasons.append("Unlikely to be an exoplanet")

        return reasons


def demo_prediction():
    """Demo function showing how to use the predictor"""
    print("="*50)
    print("Exoplanet Prediction Demo")
    print("="*50)

    # Initialize predictor
    predictor = ExoplanetPredictor()

    # Simulate a light curve
    from light_curve_processor import simulate_light_curve
    time, flux = simulate_light_curve(
        period=12.5,
        depth=0.008,
        duration=0.12,
        noise_level=0.0005
    )

    print("\nSimulated light curve parameters:")
    print("  Period: 12.5 days")
    print("  Transit depth: 0.8%")
    print("  Duration: 2.88 hours")

    # Make prediction
    results = predictor.predict_from_light_curve(time, flux)

    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"Classification: {results['model_label']}")
    print(f"Probability: {results['model_probability_candidate']:.4f}")
    print(f"Confidence: {results['confidence']}")

    if 'transit_params' in results:
        print("\nDetected Transit Parameters:")
        for key, value in results['transit_params'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if 'reasoning' in results:
        print("\nReasoning:")
        for reason in results['reasoning']:
            print(f"  - {reason}")

    return results


if __name__ == "__main__":
    # Run demo
    results = demo_prediction()

    # Save results
    import json
    with open('demo_prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nFull results saved to demo_prediction_results.json")
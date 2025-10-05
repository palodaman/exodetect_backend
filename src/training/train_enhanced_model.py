import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.enhanced_data_loader import EnhancedExoplanetDataLoader


class EnhancedExoplanetTrainer:
    def __init__(self):
        self.models = {
            'xgboost_enhanced': XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=1,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm_enhanced': LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest_enhanced': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost_enhanced': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        }

        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = None

    def prepare_data(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Scale features and apply SMOTE"""
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with median
        for col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_smote:
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE - Training samples: {len(y_train)}")
            print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        return metrics

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        self.feature_names = X_train.columns.tolist()

        X_train_scaled, X_test_scaled, y_train_balanced, y_test = self.prepare_data(
            X_train, X_test, y_train, y_test, use_smote=True
        )

        print("\n" + "="*50)
        print("Training Enhanced Models...")
        print("="*50)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            try:
                model.fit(X_train_scaled, y_train_balanced)

                metrics = self.evaluate_model(model, X_test_scaled, y_test, name)
                self.results[name] = metrics

                print(f"Results for {name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

                # Save model
                os.makedirs('models_enhanced', exist_ok=True)
                model_path = f"models_enhanced/{name}_model.pkl"
                joblib.dump(model, model_path)
                print(f"  Model saved to {model_path}")

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        # Save scaler and feature names
        scaler_path = "models_enhanced/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"\nScaler saved to {scaler_path}")

        feature_names_path = "models_enhanced/feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Feature names saved to {feature_names_path}")

        return self.results

    def get_best_model(self):
        """Find the best performing model"""
        if not self.results:
            return None, None

        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0], best_model[1]

    def save_results(self):
        """Save training results"""
        os.makedirs('results', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/enhanced_training_results_{timestamp}.json"

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {results_path}")

        # Create comparison DataFrame
        summary_df = pd.DataFrame({
            name: {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            }
            for name, metrics in self.results.items()
        }).T

        summary_path = f"results/enhanced_model_comparison_{timestamp}.csv"
        summary_df.to_csv(summary_path)
        print(f"Model comparison saved to {summary_path}")

        return summary_df


def main():
    """Main training pipeline with enhanced data"""
    print("="*50)
    print("Enhanced Exoplanet Classifier Training")
    print("="*50)

    # Load enhanced data from multiple sources
    data_loader = EnhancedExoplanetDataLoader()

    print("\nLoading and processing enhanced dataset...")
    try:
        X_train, X_test, y_train, y_test, feature_names = data_loader.get_training_data(
            test_size=0.2, random_state=42
        )

        print(f"\nDataset loaded successfully!")
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

        # Train models
        trainer = EnhancedExoplanetTrainer()
        results = trainer.train_all_models(X_train, X_test, y_train, y_test)

        # Get best model
        best_model_name, best_metrics = trainer.get_best_model()
        if best_model_name:
            print("\n" + "="*50)
            print(f"Best Model: {best_model_name}")
            print(f"F1 Score: {best_metrics['f1_score']:.4f}")
            print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
            print("="*50)

        # Save results
        summary_df = trainer.save_results()
        print("\nEnhanced Model Comparison Summary:")
        print(summary_df.round(4))

    except Exception as e:
        print(f"Error during training: {e}")
        print("Falling back to original KOI data only...")

        # Fallback to original data loader
        from data_loader import ExoplanetDataLoader
        original_loader = ExoplanetDataLoader()
        X_train, X_test, y_train, y_test, feature_names = original_loader.get_train_test_data()

        trainer = EnhancedExoplanetTrainer()
        results = trainer.train_all_models(X_train, X_test, y_train, y_test)

    print("\nTraining complete!")
    print("Enhanced models saved in 'models_enhanced/' directory")
    print("Results saved in 'results/' directory")


if __name__ == "__main__":
    main()
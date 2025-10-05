import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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

from data_loader import ExoplanetDataLoader

class ExoplanetClassifierTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            ),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }

        self.scaler = StandardScaler()
        self.results = {}

    def prepare_data(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Scale features and optionally apply SMOTE for class imbalance"""
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
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names):
        """Train and evaluate all models"""
        X_train_scaled, X_test_scaled, y_train_balanced, y_test = self.prepare_data(
            X_train, X_test, y_train, y_test, use_smote=True
        )

        print("\n" + "="*50)
        print("Training and evaluating models...")
        print("="*50)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            model.fit(X_train_scaled, y_train_balanced)

            metrics = self.evaluate_model(model, X_test_scaled, y_test, name)
            self.results[name] = metrics

            print(f"Results for {name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

            os.makedirs('models', exist_ok=True)
            model_path = f"models/{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"  Model saved to {model_path}")

        scaler_path = "models/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"\nScaler saved to {scaler_path}")

        feature_names_path = "models/feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"Feature names saved to {feature_names_path}")

        return self.results

    def get_best_model(self):
        """Find the best performing model based on F1 score"""
        if not self.results:
            return None

        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0], best_model[1]

    def save_results(self):
        """Save training results to file"""
        os.makedirs('results', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/training_results_{timestamp}.json"

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {results_path}")

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

        summary_path = f"results/model_comparison_{timestamp}.csv"
        summary_df.to_csv(summary_path)
        print(f"Model comparison saved to {summary_path}")

        return summary_df


def main():
    """Main training pipeline"""
    print("="*50)
    print("Exoplanet Classifier Training Pipeline")
    print("="*50)

    data_loader = ExoplanetDataLoader()

    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = data_loader.get_train_test_data(
        test_size=0.2, random_state=42
    )

    trainer = ExoplanetClassifierTrainer()

    results = trainer.train_all_models(X_train, X_test, y_train, y_test, feature_names)

    best_model_name, best_metrics = trainer.get_best_model()
    print("\n" + "="*50)
    print(f"Best Model: {best_model_name}")
    print(f"F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
    print("="*50)

    summary_df = trainer.save_results()
    print("\nModel Comparison Summary:")
    print(summary_df.round(4))

    print("\nTraining complete!")
    print("Models saved in 'models/' directory")
    print("Results saved in 'results/' directory")


if __name__ == "__main__":
    main()
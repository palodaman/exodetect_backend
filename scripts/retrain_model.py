#!/usr/bin/env python3
"""
Incremental model retraining using logged queries
Retrains models with new data while preserving existing knowledge
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from database import init_db, close_db
from database.schema import ModelVersion, TrainingJob, TrainingStatus
from export_training_data import TrainingDataExporter
from dotenv import load_dotenv


class IncrementalModelTrainer:
    """Incremental model training with versioning"""

    def __init__(self, model_dir: str = "models_enhanced"):
        # Use absolute path
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(backend_root, model_dir)
        self.models_dir_versioned = os.path.join(backend_root, "models_versioned")

        # Create versioned models directory
        os.makedirs(self.models_dir_versioned, exist_ok=True)

    async def prepare_training_data(
        self,
        days: int = 7,
        min_samples: int = 100,
        organization_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Prepare training data from logged queries

        Args:
            days: Number of days to fetch queries from
            min_samples: Minimum number of samples required
            organization_id: Filter by organization

        Returns:
            DataFrame with training data, or None if insufficient data
        """
        print(f"📊 Fetching queries from last {days} days...")

        # Create exporter
        exporter = TrainingDataExporter()

        # Fetch queries
        start_date = datetime.utcnow() - timedelta(days=days)
        queries = await exporter.fetch_queries(
            start_date=start_date,
            organization_id=organization_id,
            min_confidence="Medium"  # Only use medium/high confidence
        )

        print(f"✅ Found {len(queries)} queries")

        if len(queries) < min_samples:
            print(f"⚠️  Insufficient data ({len(queries)} < {min_samples})")
            return None

        # Convert to DataFrame
        df = exporter.export_to_dataframe()

        # Show statistics
        stats = exporter.get_statistics()
        print(f"\n📈 Training Data Statistics:")
        print(f"   Total samples: {stats['total_queries']}")
        print(f"   Candidates: {stats['candidates']} ({stats['candidates']/stats['total_queries']*100:.1f}%)")
        print(f"   Non-candidates: {stats['non_candidates']} ({stats['non_candidates']/stats['total_queries']*100:.1f}%)")

        return df

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model_path: Optional[str] = None
    ) -> Any:
        """
        Train XGBoost model incrementally

        Args:
            X_train: Training features
            y_train: Training labels
            base_model_path: Path to existing model (for incremental training)

        Returns:
            Trained model
        """
        import xgboost as xgb

        print("🤖 Training XGBoost model...")

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Parameters
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'seed': 42
        }

        # Load base model if exists
        base_model = None
        if base_model_path and os.path.exists(base_model_path):
            print(f"📚 Loading base model from {base_model_path}")
            base_model = xgb.Booster()
            base_model.load_model(base_model_path)
            params['process_type'] = 'update'
            params['updater'] = 'refresh'

        # Train model
        num_rounds = 100 if base_model is None else 50  # Fewer rounds for incremental

        model = xgb.train(
            params,
            dtrain,
            num_rounds,
            xgb_model=base_model,
            verbose_eval=10
        )

        print("✅ Training complete")
        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model_path: Optional[str] = None
    ) -> Any:
        """
        Train LightGBM model incrementally

        Args:
            X_train: Training features
            y_train: Training labels
            base_model_path: Path to existing model (for incremental training)

        Returns:
            Trained model
        """
        import lightgbm as lgb

        print("🤖 Training LightGBM model...")

        # Create dataset
        dtrain = lgb.Dataset(X_train, label=y_train)

        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'seed': 42
        }

        # Load base model if exists
        init_model = None
        if base_model_path and os.path.exists(base_model_path):
            print(f"📚 Loading base model from {base_model_path}")
            init_model = base_model_path

        # Train model
        num_rounds = 100 if init_model is None else 50

        model = lgb.train(
            params,
            dtrain,
            num_rounds,
            init_model=init_model,
            verbose_eval=10
        )

        print("✅ Training complete")
        return model

    async def save_model_version(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        training_samples: int,
        metrics: Dict[str, float],
        organization_id: Optional[str] = None
    ) -> ModelVersion:
        """
        Save model with version information to MongoDB

        Args:
            model: Trained model
            model_name: Name of the model
            model_type: Type (xgboost, lightgbm, etc.)
            training_samples: Number of training samples
            metrics: Training metrics
            organization_id: Organization ID

        Returns:
            ModelVersion document
        """
        # Generate version string
        version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Save model file
        model_filename = f"{model_name}_{version}.pkl"
        model_path = os.path.join(self.models_dir_versioned, model_filename)

        joblib.dump(model, model_path)
        print(f"💾 Saved model to {model_path}")

        # Calculate expiration (7 days from now)
        expires_at = datetime.utcnow() + timedelta(days=7)

        # Create ModelVersion document
        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            training_date=datetime.utcnow(),
            training_samples=training_samples,
            metrics=metrics,
            status="active",
            organization_id=organization_id,
            expires_at=expires_at
        )

        await model_version.insert()
        print(f"✅ Saved model version: {version}")

        return model_version

    async def cleanup_old_versions(self, keep_days: int = 7):
        """
        Remove model versions older than keep_days

        Args:
            keep_days: Number of days to keep models
        """
        print(f"\n🧹 Cleaning up models older than {keep_days} days...")

        cutoff_date = datetime.utcnow() - timedelta(days=keep_days)

        # Find expired models
        expired_models = await ModelVersion.find(
            ModelVersion.training_date < cutoff_date
        ).to_list()

        for model in expired_models:
            # Delete file
            if os.path.exists(model.model_path):
                os.remove(model.model_path)
                print(f"   Deleted {model.model_path}")

            # Update status or delete from DB
            model.status = "expired"
            await model.save()

        print(f"✅ Cleaned up {len(expired_models)} old models")

    async def retrain(
        self,
        model_name: str = "xgboost_enhanced",
        days: int = 7,
        min_samples: int = 100,
        organization_id: Optional[str] = None,
        incremental: bool = True
    ) -> Optional[str]:
        """
        Main retraining function

        Args:
            model_name: Name of model to retrain
            days: Days of data to use
            min_samples: Minimum samples required
            organization_id: Organization ID
            incremental: Use incremental learning from existing model

        Returns:
            New model version string, or None if failed
        """
        print(f"\n{'='*60}")
        print(f"ExoDetect Model Retraining - {model_name}")
        print(f"{'='*60}\n")

        # Create training job record
        training_job = TrainingJob(
            model_name=model_name,
            status=TrainingStatus.RUNNING,
            started_at=datetime.utcnow(),
            organization_id=organization_id,
            config={
                "days": days,
                "min_samples": min_samples,
                "incremental": incremental
            }
        )
        await training_job.insert()

        try:
            # Prepare training data
            df = await self.prepare_training_data(
                days=days,
                min_samples=min_samples,
                organization_id=organization_id
            )

            if df is None:
                training_job.status = TrainingStatus.FAILED
                training_job.completed_at = datetime.utcnow()
                training_job.error_message = "Insufficient training data"
                await training_job.save()
                return None

            # Extract features and labels
            feature_cols = [
                'period', 'duration', 'depth', 'snr',
                'impact', 'star_teff', 'star_logg', 'star_radius'
            ]

            # Only use columns that exist
            available_features = [col for col in feature_cols if col in df.columns]

            if len(available_features) < 4:
                print("⚠️  Insufficient features in data")
                training_job.status = TrainingStatus.FAILED
                training_job.error_message = "Insufficient features"
                await training_job.save()
                return None

            X_train = df[available_features].fillna(0).values
            y_train = df['label'].values

            print(f"\n📊 Training data shape: {X_train.shape}")
            print(f"   Features: {available_features}")

            # Determine model type and train
            model_type = model_name.split('_')[0]  # e.g., 'xgboost' from 'xgboost_enhanced'

            # Find base model for incremental training
            base_model_path = None
            if incremental:
                base_model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")

            # Train model
            if model_type == 'xgboost':
                model = self.train_xgboost(X_train, y_train, base_model_path if incremental else None)
            elif model_type == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train, base_model_path if incremental else None)
            else:
                print(f"❌ Unsupported model type: {model_type}")
                training_job.status = TrainingStatus.FAILED
                training_job.error_message = f"Unsupported model type: {model_type}"
                await training_job.save()
                return None

            # Calculate basic metrics (in production, use validation set)
            metrics = {
                "training_samples": int(len(y_train)),
                "positive_samples": int(y_train.sum()),
                "negative_samples": int(len(y_train) - y_train.sum()),
                "feature_count": len(available_features)
            }

            # Save versioned model
            model_version = await self.save_model_version(
                model=model,
                model_name=model_name,
                model_type=model_type,
                training_samples=len(y_train),
                metrics=metrics,
                organization_id=organization_id
            )

            # Update training job
            training_job.status = TrainingStatus.COMPLETED
            training_job.completed_at = datetime.utcnow()
            training_job.model_version = model_version.version
            training_job.metrics = metrics
            await training_job.save()

            # Cleanup old versions
            await self.cleanup_old_versions(keep_days=7)

            print(f"\n{'='*60}")
            print(f"✅ Retraining complete!")
            print(f"   Version: {model_version.version}")
            print(f"   Samples: {metrics['training_samples']}")
            print(f"{'='*60}\n")

            return model_version.version

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            training_job.status = TrainingStatus.FAILED
            training_job.completed_at = datetime.utcnow()
            training_job.error_message = str(e)
            await training_job.save()
            raise


async def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrain ExoDetect models with new data"
    )
    parser.add_argument(
        "--model",
        default="xgboost_enhanced",
        help="Model name to retrain (default: xgboost_enhanced)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of data to use (default: 7)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required (default: 100)"
    )
    parser.add_argument(
        "--organization-id",
        help="Filter by organization ID"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full retraining (not incremental)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Initialize database
    await init_db()

    # Create trainer
    trainer = IncrementalModelTrainer()

    # Retrain
    try:
        version = await trainer.retrain(
            model_name=args.model,
            days=args.days,
            min_samples=args.min_samples,
            organization_id=args.organization_id,
            incremental=not args.full
        )

        if version:
            print(f"\n✅ Success! New model version: {version}")
        else:
            print(f"\n⚠️  Retraining skipped (insufficient data)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

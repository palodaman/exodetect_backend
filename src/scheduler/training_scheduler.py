"""
APScheduler for automated model retraining
Runs retraining jobs at scheduled times
"""

import asyncio
import os
import sys
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime


class TrainingScheduler:
    """Scheduler for automated model retraining"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False

        # Models to retrain
        self.models = [
            "xgboost_enhanced",
            "lightgbm_enhanced"
        ]

        # Training configuration
        self.config = {
            "days": int(os.getenv("RETRAIN_DAYS", "7")),
            "min_samples": int(os.getenv("RETRAIN_MIN_SAMPLES", "100")),
            "incremental": os.getenv("RETRAIN_INCREMENTAL", "true").lower() == "true"
        }

    async def retrain_all_models(self):
        """Retrain all configured models"""
        print(f"\n{'='*70}")
        print(f"🤖 Automated Retraining Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # Import here to avoid module load issues
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_path)
        from retrain_model import IncrementalModelTrainer

        trainer = IncrementalModelTrainer()

        results = {}

        for model_name in self.models:
            try:
                print(f"\n🔄 Retraining {model_name}...")

                version = await trainer.retrain(
                    model_name=model_name,
                    days=self.config["days"],
                    min_samples=self.config["min_samples"],
                    incremental=self.config["incremental"]
                )

                if version:
                    results[model_name] = {
                        "status": "success",
                        "version": version
                    }
                    print(f"✅ {model_name} retrained successfully: {version}")
                else:
                    results[model_name] = {
                        "status": "skipped",
                        "reason": "insufficient_data"
                    }
                    print(f"⚠️  {model_name} retraining skipped (insufficient data)")

            except Exception as e:
                results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"❌ {model_name} retraining failed: {e}")

        print(f"\n{'='*70}")
        print(f"✅ Automated Retraining Complete")
        print(f"   Results: {results}")
        print(f"{'='*70}\n")

        return results

    def start(self):
        """Start the scheduler"""
        if self.is_running:
            print("⚠️  Scheduler already running")
            return

        # Schedule midnight retraining (00:00 every day)
        self.scheduler.add_job(
            self.retrain_all_models,
            trigger=CronTrigger(hour=0, minute=0),
            id="midnight_retraining",
            name="Midnight Model Retraining",
            replace_existing=True
        )

        # Optional: Add weekly full retraining (Sunday 02:00)
        self.scheduler.add_job(
            lambda: self.retrain_all_models_full(),
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
            id="weekly_full_retraining",
            name="Weekly Full Retraining",
            replace_existing=True
        )

        # Start scheduler
        self.scheduler.start()
        self.is_running = True

        print("✅ Training scheduler started")
        print(f"   Midnight retraining: Every day at 00:00")
        print(f"   Weekly full retraining: Sundays at 02:00")
        print(f"   Models: {', '.join(self.models)}")

    async def retrain_all_models_full(self):
        """Full retraining (not incremental)"""
        print(f"\n{'='*70}")
        print(f"🔄 Weekly Full Retraining - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # Import here to avoid module load issues
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts')
        sys.path.insert(0, scripts_path)
        from retrain_model import IncrementalModelTrainer

        trainer = IncrementalModelTrainer()

        for model_name in self.models:
            try:
                version = await trainer.retrain(
                    model_name=model_name,
                    days=30,  # Use 30 days for full retraining
                    min_samples=self.config["min_samples"],
                    incremental=False  # Full retraining
                )

                if version:
                    print(f"✅ {model_name} full retrain: {version}")

            except Exception as e:
                print(f"❌ {model_name} full retrain failed: {e}")

    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return

        self.scheduler.shutdown()
        self.is_running = False
        print("✅ Training scheduler stopped")

    def trigger_now(self):
        """Manually trigger retraining now (for testing)"""
        if not self.is_running:
            print("⚠️  Scheduler not running. Starting it first...")
            self.start()

        print("🔄 Triggering retraining now...")
        self.scheduler.add_job(
            self.retrain_all_models,
            id="manual_retraining",
            replace_existing=True
        )

    def get_jobs(self):
        """Get all scheduled jobs"""
        return self.scheduler.get_jobs()


# Global scheduler instance
training_scheduler = TrainingScheduler()

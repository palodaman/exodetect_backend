"""
Automated task scheduler for ExoDetect
Handles midnight model retraining and other scheduled tasks
"""

from .training_scheduler import training_scheduler

__all__ = ["training_scheduler"]

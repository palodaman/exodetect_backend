"""
Query logging system for ExoDetect
Logs all prediction queries to MongoDB for model retraining
"""

from .query_logger import QueryLogger
from .middleware import QueryLoggingMiddleware

__all__ = ["QueryLogger", "QueryLoggingMiddleware"]

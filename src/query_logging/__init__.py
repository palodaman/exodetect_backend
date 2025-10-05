"""
Query logging system for ExoDetect
Logs all prediction queries to MongoDB for model retraining
"""

from .query_logger import QueryLogger, query_logger
from .middleware import QueryLoggingMiddleware

__all__ = ["QueryLogger", "query_logger", "QueryLoggingMiddleware"]

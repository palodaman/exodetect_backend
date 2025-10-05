"""
Query logger for ExoDetect predictions
Asynchronously logs prediction queries to MongoDB
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from database.schema import Query, QueryType


class QueryLogger:
    """Async query logger for prediction requests"""

    def __init__(self):
        self._log_queue = asyncio.Queue()
        self._running = False
        self._worker_task = None

    async def start(self):
        """Start the background logging worker"""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._log_worker())
        print("✅ Query logger started")

    async def stop(self):
        """Stop the background logging worker"""
        self._running = False

        if self._worker_task:
            # Wait for remaining logs to be processed
            await self._log_queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        print("✅ Query logger stopped")

    async def _log_worker(self):
        """Background worker that processes log queue"""
        while self._running:
            try:
                # Get log entry from queue with timeout
                log_entry = await asyncio.wait_for(
                    self._log_queue.get(),
                    timeout=1.0
                )

                # Save to MongoDB
                try:
                    query = Query(**log_entry)
                    await query.insert()
                except Exception as e:
                    print(f"❌ Failed to log query: {e}")
                finally:
                    self._log_queue.task_done()

            except asyncio.TimeoutError:
                # No items in queue, continue waiting
                continue
            except Exception as e:
                print(f"❌ Log worker error: {e}")

    async def log_prediction(
        self,
        query_type: QueryType,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        model_version: str,
        processing_time_ms: float,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a prediction query (non-blocking)

        Args:
            query_type: Type of query (archive, upload, light_curve, features)
            input_data: Input parameters/data
            prediction_result: Prediction output
            model_version: Model version used
            processing_time_ms: Time taken in milliseconds
            user_id: Optional user ID (if authenticated)
            organization_id: Optional organization ID
            metadata: Additional metadata
        """
        log_entry = {
            "query_type": query_type,
            "input_data": input_data,
            "prediction_result": prediction_result,
            "model_version": model_version,
            "processing_time_ms": processing_time_ms,
            "user_id": user_id,
            "organization_id": organization_id,
            "created_at": datetime.utcnow(),
            "metadata": metadata or {}
        }

        # Add to queue (non-blocking)
        await self._log_queue.put(log_entry)

    async def get_queries_for_training(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        query_type: Optional[QueryType] = None,
        organization_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[Query]:
        """
        Retrieve logged queries for model training

        Args:
            start_date: Filter from this date
            end_date: Filter to this date
            query_type: Filter by query type
            organization_id: Filter by organization
            limit: Maximum number of queries to return

        Returns:
            List of Query documents
        """
        # Build query filter
        filter_conditions = []

        if start_date:
            filter_conditions.append(Query.created_at >= start_date)

        if end_date:
            filter_conditions.append(Query.created_at <= end_date)

        if query_type:
            filter_conditions.append(Query.query_type == query_type)

        if organization_id:
            filter_conditions.append(Query.organization_id == organization_id)

        # Execute query
        query_builder = Query.find()

        if filter_conditions:
            query_builder = query_builder.find(*filter_conditions)

        query_builder = query_builder.sort("-created_at")

        if limit:
            query_builder = query_builder.limit(limit)

        return await query_builder.to_list()

    async def get_query_stats(
        self,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about logged queries

        Returns:
            Dictionary with query statistics
        """
        from beanie import PydanticObjectId

        # Build match stage
        match_stage = {}
        if organization_id:
            match_stage["organization_id"] = organization_id

        # Aggregate statistics
        pipeline = []

        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.extend([
            {
                "$group": {
                    "_id": None,
                    "total_queries": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time_ms"},
                    "queries_by_type": {
                        "$push": "$query_type"
                    },
                    "models_used": {
                        "$addToSet": "$model_version"
                    }
                }
            }
        ])

        result = await Query.aggregate(pipeline).to_list()

        if not result:
            return {
                "total_queries": 0,
                "avg_processing_time_ms": 0,
                "queries_by_type": {},
                "models_used": []
            }

        stats = result[0]

        # Count queries by type
        query_type_counts = {}
        for qtype in stats.get("queries_by_type", []):
            query_type_counts[qtype] = query_type_counts.get(qtype, 0) + 1

        return {
            "total_queries": stats.get("total_queries", 0),
            "avg_processing_time_ms": round(stats.get("avg_processing_time", 0), 2),
            "queries_by_type": query_type_counts,
            "models_used": stats.get("models_used", [])
        }


# Global logger instance
query_logger = QueryLogger()

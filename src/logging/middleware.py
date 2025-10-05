"""
FastAPI middleware for automatic query logging
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from database.schema import QueryType
from .query_logger import query_logger


class QueryLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically log prediction requests
    Only logs /api/predict/* endpoints
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.prediction_endpoints = {
            "/api/predict/archive": QueryType.ARCHIVE,
            "/api/predict/upload": QueryType.UPLOAD,
            "/api/predict/light-curve": QueryType.LIGHT_CURVE,
            "/api/predict/features": QueryType.FEATURES
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log if it's a prediction endpoint"""

        # Check if this is a prediction endpoint
        path = request.url.path
        query_type = self.prediction_endpoints.get(path)

        if not query_type:
            # Not a prediction endpoint, pass through
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Capture request data
        try:
            # Clone request body
            body = await request.body()

            # Parse input data
            try:
                input_data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                input_data = {"raw_body": body.decode("utf-8", errors="ignore")}

            # Reset body for downstream handlers
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

        except Exception as e:
            input_data = {"error": f"Failed to capture input: {str(e)}"}

        # Get user info if authenticated
        user_id = None
        organization_id = None

        try:
            # Check if user is authenticated
            if hasattr(request.state, "user"):
                user = request.state.user
                user_id = str(user.id) if user else None
                organization_id = user.organization_id if user else None
        except:
            pass

        # Process request
        response = await call_next(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Capture response (only if successful)
        if response.status_code == 200:
            try:
                # Read response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                # Parse prediction result
                try:
                    prediction_result = json.loads(response_body)
                except json.JSONDecodeError:
                    prediction_result = {"raw_response": response_body.decode("utf-8", errors="ignore")}

                # Get model version from response
                model_version = prediction_result.get("model_version", "unknown")

                # Log the query (non-blocking)
                await query_logger.log_prediction(
                    query_type=query_type,
                    input_data=input_data,
                    prediction_result=prediction_result,
                    model_version=model_version,
                    processing_time_ms=processing_time_ms,
                    user_id=user_id,
                    organization_id=organization_id,
                    metadata={
                        "user_agent": request.headers.get("user-agent"),
                        "ip_address": request.client.host if request.client else None
                    }
                )

                # Recreate response with same body
                from starlette.responses import Response
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )

            except Exception as e:
                print(f"⚠️ Failed to log query: {e}")

        return response

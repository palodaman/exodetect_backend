"""
MongoDB connection management using Beanie ODM
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import Optional
import logging

from .schema import DOCUMENT_MODELS

logger = logging.getLogger(__name__)

# Global MongoDB client
mongodb_client: Optional[AsyncIOMotorClient] = None


async def init_db():
    """Initialize MongoDB connection and Beanie ODM"""
    global mongodb_client

    # Get MongoDB URL from environment
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "exodetect")

    logger.info(f"Connecting to MongoDB at {mongodb_url}")

    try:
        # Create Motor client
        mongodb_client = AsyncIOMotorClient(mongodb_url)

        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info("MongoDB connection successful")

        # Initialize Beanie with document models
        database = mongodb_client[database_name]
        await init_beanie(database=database, document_models=DOCUMENT_MODELS)

        logger.info(f"Beanie initialized with database: {database_name}")
        logger.info(f"Registered models: {[model.__name__ for model in DOCUMENT_MODELS]}")

        return mongodb_client

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_db():
    """Close MongoDB connection"""
    global mongodb_client

    if mongodb_client is not None:
        mongodb_client.close()
        logger.info("MongoDB connection closed")
        mongodb_client = None


def get_database():
    """Get database instance (for direct access if needed)"""
    if mongodb_client is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    database_name = os.getenv("MONGODB_DATABASE", "exodetect")
    return mongodb_client[database_name]

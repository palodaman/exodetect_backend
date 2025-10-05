"""
MongoDB Schema Definitions for ExoDetect
Using Beanie ODM for async MongoDB operations
"""

from beanie import Document, Indexed
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid


# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class QueryType(str, Enum):
    ARCHIVE = "archive"
    UPLOAD = "upload"
    LIGHT_CURVE = "light_curve"
    FEATURES = "features"
    BATCH = "batch"


class ModelStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnnotationLabel(str, Enum):
    CONFIRMED_CANDIDATE = "confirmed_candidate"
    FALSE_POSITIVE = "false_positive"
    NEEDS_REVIEW = "needs_review"
    CONFIRMED_PLANET = "confirmed_planet"


# Sub-documents (embedded models)
class VettingFlags(BaseModel):
    ntl: bool = False  # Not transit-like
    ss: bool = False   # Stellar eclipse
    co: bool = False   # Centroid offset
    em: bool = False   # Ephemeris match


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    training_samples: int
    validation_samples: int
    test_samples: Optional[int] = None


class OrganizationSettings(BaseModel):
    max_users: int = 50
    max_queries_per_day: int = 1000
    enable_collaboration: bool = True
    enable_auto_retraining: bool = True
    retention_days: int = 7


# Main Documents

class Organization(Document):
    """Organization/Team document"""
    name: Indexed(str, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    settings: OrganizationSettings = Field(default_factory=OrganizationSettings)
    active: bool = True

    class Settings:
        name = "organizations"


class User(Document):
    """User document with authentication"""
    email: Indexed(EmailStr, unique=True)
    username: Indexed(str, unique=True)
    password_hash: str
    organization_id: Optional[str] = None
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True

    class Settings:
        name = "users"


class Session(Document):
    """User session for authentication tracking"""
    user_id: Indexed(str)
    session_token: Indexed(str, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True

    class Settings:
        name = "sessions"

    @staticmethod
    def create_expiration(hours: int = 24) -> datetime:
        return datetime.utcnow() + timedelta(hours=hours)


class Query(Document):
    """Logged user query for model predictions"""
    user_id: Indexed(str)
    session_id: Optional[str] = None
    organization_id: Optional[Indexed(str)] = None
    query_type: QueryType
    target_id: Optional[str] = None
    mission: Optional[str] = None
    input_data: Dict[str, Any] = {}
    prediction_result: Dict[str, Any] = {}
    model_version: Indexed(str)
    processing_time: float
    created_at: Indexed(datetime) = Field(default_factory=datetime.utcnow)
    used_for_training: bool = False
    feedback_score: Optional[int] = None  # 1-5 stars
    feedback_comment: Optional[str] = None

    class Settings:
        name = "queries"
        indexes = [
            [("created_at", -1)],  # For time-based queries
            [("model_version", 1), ("created_at", -1)],  # For model analysis
            [("organization_id", 1), ("created_at", -1)],  # For org queries
        ]


class ModelVersion(Document):
    """Model version tracking with metadata"""
    version: Indexed(str, unique=True)  # e.g., "2024-10-05_v1"
    model_name: str  # e.g., "xgboost_enhanced"
    file_path: str  # Relative path from backend root
    training_date: datetime
    metrics: ModelMetrics
    status: ModelStatus = ModelStatus.INACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # Auto-delete after 7 days
    training_data_count: int = 0
    parent_version: Optional[str] = None  # For incremental training tracking

    class Settings:
        name = "model_versions"
        indexes = [
            [("training_date", -1)],  # Latest models first
            [("status", 1)],
            [("expires_at", 1)],  # For cleanup
        ]

    @staticmethod
    def create_expiration(days: int = 7) -> datetime:
        return datetime.utcnow() + timedelta(days=days)

    @staticmethod
    def generate_version() -> str:
        """Generate version string: YYYY-MM-DD_v1"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"{today}_v{uuid.uuid4().hex[:6]}"


class CanvasSession(Document):
    """Collaborative canvas session"""
    organization_id: Indexed(str)
    name: str
    canvas_data: Dict[str, Any] = {}  # Store canvas state (JSON)
    created_by: str  # user_id
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True
    participants: List[str] = []  # List of user_ids

    class Settings:
        name = "canvas_sessions"


class QueryAnnotation(Document):
    """User annotations on queries for collaborative labeling"""
    query_id: Indexed(str)
    user_id: str
    annotation: AnnotationLabel
    confidence: float = 1.0  # 0.0 to 1.0
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "query_annotations"
        indexes = [
            [("query_id", 1), ("user_id", 1)],  # Unique per user per query
        ]


class TrainingJob(Document):
    """Track automated training jobs"""
    model_name: str
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    model_version: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    organization_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    class Settings:
        name = "training_jobs"
        indexes = [
            [("created_at", -1)],
            [("status", 1)],
        ]


# List of all document models for initialization
DOCUMENT_MODELS = [
    Organization,
    User,
    Session,
    Query,
    ModelVersion,
    CanvasSession,
    QueryAnnotation,
    TrainingJob,
]

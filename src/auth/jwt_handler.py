"""
JWT token generation and validation
Supports both internal JWT and future OAuth integration
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token

    Args:
        data: Payload data to encode (typically user_id, email, etc.)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a refresh token with longer expiration

    Args:
        data: Payload data to encode

    Returns:
        Encoded refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last 7 days

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def verify_token(token: str) -> bool:
    """
    Verify if a token is valid

    Args:
        token: JWT token string

    Returns:
        True if valid, False otherwise
    """
    payload = decode_token(token)
    return payload is not None


def create_oauth_compatible_token(
    user_id: str,
    email: str,
    provider: str = "internal",
    oauth_data: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Create tokens compatible with OAuth flow
    Placeholder for future OAuth integration

    Args:
        user_id: User's unique ID
        email: User's email
        provider: OAuth provider ('internal', 'google', 'github', etc.)
        oauth_data: Additional OAuth provider data

    Returns:
        Dictionary with access_token, refresh_token, and token_type
    """
    token_data = {
        "sub": user_id,
        "email": email,
        "provider": provider
    }

    # Add OAuth-specific data if provided
    if oauth_data:
        token_data["oauth"] = oauth_data

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": user_id, "provider": provider})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


def extract_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Extract user information from token

    Args:
        token: JWT token string

    Returns:
        User data from token payload
    """
    payload = decode_token(token)
    if not payload:
        return None

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "provider": payload.get("provider", "internal"),
        "oauth_data": payload.get("oauth")
    }

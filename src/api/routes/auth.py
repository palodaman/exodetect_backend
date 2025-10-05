"""
Authentication API routes
Supports both internal authentication and OAuth placeholder
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import User, Session as DBSession, Organization, UserRole
from auth import (
    create_access_token,
    hash_password,
    verify_password,
    get_current_user,
    get_current_active_user,
    decode_token,
    create_oauth_compatible_token
)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


# Request/Response Models
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str
    organization_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
    organization_id: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# OAuth Placeholder Models
class OAuthCallbackRequest(BaseModel):
    """Placeholder for future OAuth implementation"""
    provider: str  # 'google', 'github', etc.
    code: str
    state: Optional[str] = None


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user

    Creates user account and optionally a new organization
    """
    # Check if user already exists
    existing_user = await User.find_one(User.email == request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    existing_username = await User.find_one(User.username == request.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # Create organization if provided
    organization_id = None
    if request.organization_name:
        organization = Organization(name=request.organization_name)
        await organization.insert()
        organization_id = str(organization.id)

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password),
        organization_id=organization_id,
        role=UserRole.ADMIN if organization_id else UserRole.USER,
        last_login=datetime.utcnow()
    )
    await user.insert()

    # Generate tokens
    tokens = create_oauth_compatible_token(
        user_id=str(user.id),
        email=user.email,
        provider="internal"
    )

    # Create session
    session = DBSession(
        user_id=str(user.id),
        session_token=tokens["access_token"],
        expires_at=DBSession.create_expiration()
    )
    await session.insert()

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        user={
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "organization_id": user.organization_id
        }
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Login with email and password

    Returns JWT tokens for authentication
    """
    # Find user
    user = await User.find_one(User.email == request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    # Update last login
    user.last_login = datetime.utcnow()
    await user.save()

    # Generate tokens
    tokens = create_oauth_compatible_token(
        user_id=str(user.id),
        email=user.email,
        provider="internal"
    )

    # Create session
    session = DBSession(
        user_id=str(user.id),
        session_token=tokens["access_token"],
        expires_at=DBSession.create_expiration(),
        ip_address=http_request.client.host if http_request.client else None,
        user_agent=http_request.headers.get("user-agent")
    )
    await session.insert()

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        user={
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "organization_id": user.organization_id
        }
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout current user

    Invalidates all active sessions for the user
    """
    # Invalidate all user sessions
    sessions = await DBSession.find(DBSession.user_id == str(current_user.id)).to_list()
    for session in sessions:
        session.is_valid = False
        await session.save()

    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    """
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        role=current_user.role,
        organization_id=current_user.organization_id,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    """
    payload = decode_token(request.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    # Fetch user
    user = await User.get(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Generate new tokens
    tokens = create_oauth_compatible_token(
        user_id=str(user.id),
        email=user.email,
        provider=payload.get("provider", "internal")
    )

    # Create new session
    session = DBSession(
        user_id=str(user.id),
        session_token=tokens["access_token"],
        expires_at=DBSession.create_expiration()
    )
    await session.insert()

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
        user={
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "organization_id": user.organization_id
        }
    )


# OAuth Placeholder Endpoint
@router.post("/oauth/callback", response_model=TokenResponse)
async def oauth_callback(request: OAuthCallbackRequest):
    """
    OAuth callback endpoint (PLACEHOLDER)

    This endpoint is a placeholder for future OAuth integration.
    When you set up OAuth on your provider (Google, GitHub, etc.),
    you'll handle the callback here and exchange the code for user info.

    Current implementation returns error - to be implemented with actual OAuth provider.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"OAuth provider '{request.provider}' not yet configured. Please set up OAuth provider integration."
    )


@router.get("/oauth/providers")
async def get_oauth_providers():
    """
    Get list of available OAuth providers

    Returns list of providers that can be configured
    """
    return {
        "available_providers": [
            {
                "name": "google",
                "enabled": False,
                "description": "Sign in with Google"
            },
            {
                "name": "github",
                "enabled": False,
                "description": "Sign in with GitHub"
            },
            {
                "name": "microsoft",
                "enabled": False,
                "description": "Sign in with Microsoft"
            }
        ],
        "message": "OAuth providers not yet configured. Using internal authentication."
    }

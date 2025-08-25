"""
Authentication API Endpoints for SAP BTP Integration
Provides REST API endpoints for user authentication, session management, and user preferences
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .rbacService import RBACService, UserInfo, get_rbac_service, require_permission
from .sessionService import SessionService, SessionInfo, get_session_service
from .destinationService import DestinationService, get_destination_service

logger = logging.getLogger(__name__)


async def authenticate_user_credentials(
    username: str,
    password: str,
    rbac_service: RBACService
) -> Optional[UserInfo]:
    """Authenticate user against database or LDAP"""
    try:
        # In production, integrate with:
        # 1. Corporate LDAP/Active Directory
        # 2. SAP Cloud Identity Services
        # 3. Database with properly hashed passwords

        # Development mode authentication (use environment variables)
        import os
        if os.environ.get('DEVELOPMENT_MODE', 'false').lower() == 'true':
            logger.warning("Development mode authentication enabled - not for production use")

            # Get dev credentials from environment variables
            dev_username = os.environ.get('DEV_AUTH_USERNAME')
            dev_password = os.environ.get('DEV_AUTH_PASSWORD')

            if not dev_username or not dev_password:
                raise HTTPException(
                    status_code=500,
                    detail="Development mode enabled but DEV_AUTH_USERNAME/DEV_AUTH_PASSWORD not set"
                )

            if username == dev_username and password == dev_password:
                return UserInfo(
                    user_id="DEV_USER_001",
                    user_name="Development User",
                    email=dev_username,
                    roles=[UserRole.DEVELOPER],
                    scopes=["read:projects", "write:projects"],
                    tenant_id="dev-tenant"
                )

        # Production: implement real authentication here
        # Example: LDAP authentication
        # ldap_conn = ldap.initialize(os.environ.get('LDAP_SERVER'))
        # ldap_conn.simple_bind_s(f"uid={username},{os.environ.get('LDAP_BASE_DN')}", password)

        return None

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

# Security scheme
security = HTTPBearer()

# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model"""
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None  # For XSUAA token login


class LoginResponse(BaseModel):
    """Login response model"""
    success: bool
    message: str
    session_id: Optional[str] = None
    user_info: Optional[UserInfo] = None
    expires_at: Optional[datetime] = None


class SessionResponse(BaseModel):
    """Session response model"""
    sessions: List[SessionInfo]
    statistics: Dict[str, Any]


class UserPreferences(BaseModel):
    """User preferences model"""
    theme: str = "sap_fiori_3"
    language: str = "en"
    timezone: str = "UTC"
    notifications: Dict[str, bool] = Field(default_factory=lambda: {
        "email": True,
        "push": True,
        "deployment": True,
        "security": True
    })


class ChangePasswordRequest(BaseModel):
    """Change password request model"""
    current_password: str
    new_password: str


class TerminateSessionsResponse(BaseModel):
    """Terminate sessions response model"""
    terminated_count: int
    message: str


# Create router
auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    rbac_service: RBACService = Depends(get_rbac_service)
) -> UserInfo:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        user_info = await rbac_service.validate_token(token)

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return user_info

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    rbac_service: RBACService = Depends(get_rbac_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    User login endpoint
    In SAP BTP, this would typically be handled by the Application Router and XSUAA
    """
    try:
        # In a real SAP BTP environment, the Application Router would handle authentication
        # and provide the JWT token. For development, we simulate this process.

        if request.token:
            # Token-based login (XSUAA JWT)
            user_info = await rbac_service.validate_token(request.token)
            if not user_info:
                return LoginResponse(
                    success=False,
                    message="Invalid token"
                )

            # Create session
            session_info = await session_service.create_session(
                user_info=user_info,
                request=http_request,
                access_token=request.token
            )

            return LoginResponse(
                success=True,
                message="Login successful",
                session_id=session_info.session_id,
                user_info=user_info,
                expires_at=session_info.expires_at
            )

        elif request.username and request.password:
            # Username/password login (for development only)
            # In production, this would redirect to XSUAA

            # Authenticate against user database
            user_info = await authenticate_user_credentials(
                username=request.username,
                password=request.password,
                rbac_service=rbac_service
            )

            if not user_info:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )

            # Generate production token
            token = await rbac_service.generate_token(user_info)

            # Create session
            session_info = await session_service.create_session(
                user_info=user_info,
                request=http_request,
                access_token=token
            )

            return LoginResponse(
                success=True,
                message="Login successful",
                session_id=session_info.session_id,
                user_info=user_info,
                expires_at=session_info.expires_at
            )

        else:
            return LoginResponse(
                success=False,
                message="Username/password or token required"
            )

    except Exception as e:
        logger.error(f"Login error: {e}")
        return LoginResponse(
            success=False,
            message="Login failed"
        )


@auth_router.post("/logout")
async def logout(
    http_request: Request,
    current_user: UserInfo = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service)
):
    """User logout endpoint"""
    try:
        # Get session ID from request headers or cookies
        session_id = http_request.headers.get("X-Session-ID")

        if session_id:
            await session_service.terminate_session(session_id)

        return {"success": True, "message": "Logout successful"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"success": False, "message": "Logout failed"}


@auth_router.get("/sessions", response_model=SessionResponse)
async def get_user_sessions(
    current_user: UserInfo = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service)
):
    """Get user's active sessions"""
    try:
        sessions = await session_service.get_user_sessions(current_user.user_id)
        statistics = await session_service.get_session_statistics()

        return SessionResponse(
            sessions=sessions,
            statistics=statistics
        )

    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@auth_router.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service)
):
    """Terminate a specific session"""
    try:
        success = await session_service.terminate_session(session_id)

        if success:
            return {"success": True, "message": "Session terminated"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Terminate session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to terminate session")


@auth_router.post("/sessions/terminate-all", response_model=TerminateSessionsResponse)
async def terminate_all_sessions(
    http_request: Request,
    current_user: UserInfo = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service)
):
    """Terminate all user sessions except current one"""
    try:
        current_session_id = http_request.headers.get("X-Session-ID")

        terminated_count = await session_service.terminate_user_sessions(
            current_user.user_id,
            exclude_session=current_session_id
        )

        return TerminateSessionsResponse(
            terminated_count=terminated_count,
            message=f"Terminated {terminated_count} sessions"
        )

    except Exception as e:
        logger.error(f"Terminate all sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to terminate sessions")


@auth_router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(
    current_user: UserInfo = Depends(get_current_user)
):
    """Get user preferences"""
    try:
        # In a real implementation, this would fetch from a user preferences database
        # For now, return default preferences
        return UserPreferences()

    except Exception as e:
        logger.error(f"Get preferences error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")


@auth_router.put("/preferences")
async def update_user_preferences(
    preferences: UserPreferences,
    current_user: UserInfo = Depends(get_current_user)
):
    """Update user preferences"""
    try:
        # In a real implementation, this would save to a user preferences database
        logger.info(f"Updated preferences for user {current_user.user_id}: {preferences.dict()}")

        return {"success": True, "message": "Preferences updated"}

    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@auth_router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Change user password"""
    try:
        # In SAP BTP environments, password changes are typically handled by:
        # 1. SAP Cloud Identity Services
        # 2. Corporate Identity Provider (LDAP/Active Directory)
        # 3. External Identity Provider (OAuth/SAML)

        # For now, return not implemented for production security
        raise HTTPException(
            status_code=501,
            detail="Password change must be handled by the configured Identity Provider"
        )

        # Basic validation for future implementation
        if len(request.new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="New password must be at least 8 characters long"
            )

        # In a real implementation, this would:
        # 1. Validate current password with Identity Provider
        # 2. Update password through Identity Provider API
        # 3. Invalidate all existing sessions

        logger.info(f"Password change requested for user {current_user.user_id}")

        return {"success": True, "message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


@auth_router.get("/user-info", response_model=UserInfo)
async def get_user_info(
    current_user: UserInfo = Depends(get_current_user)
):
    """Get current user information"""
    return current_user


@auth_router.get("/permissions")
async def get_user_permissions(
    current_user: UserInfo = Depends(get_current_user)
):
    """Get current user permissions"""
    return {
        "user_id": current_user.user_id,
        "roles": current_user.roles,
        "permissions": current_user.permissions
    }


@auth_router.post("/refresh-token")
async def refresh_token(
    http_request: Request,
    current_user: UserInfo = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service)
):
    """Refresh authentication token"""
    try:
        session_id = http_request.headers.get("X-Session-ID")

        if session_id:
            session_info = await session_service.refresh_session(session_id)

            if session_info:
                return {
                    "success": True,
                    "message": "Token refreshed",
                    "expires_at": session_info.expires_at
                }

        raise HTTPException(status_code=401, detail="Session not found or expired")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")


# Protected endpoints that require specific permissions
@auth_router.get("/admin/users")
async def list_all_users(
    current_user: UserInfo = Depends(require_permission("admin:users")),
    session_service: SessionService = Depends(get_session_service)
):
    """List all users (admin only)"""
    try:
        statistics = await session_service.get_session_statistics()
        return {
            "total_users": statistics.get("unique_users", 0),
            "active_sessions": statistics.get("active_sessions", 0),
            "most_active_users": statistics.get("most_active_users", [])
        }

    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user list")


@auth_router.get("/admin/sessions")
async def list_all_sessions(
    current_user: UserInfo = Depends(require_permission("admin:sessions")),
    session_service: SessionService = Depends(get_session_service)
):
    """List all active sessions (admin only)"""
    try:
        statistics = await session_service.get_session_statistics()
        return statistics

    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session list")


# Health check endpoint
@auth_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "authentication"
    }

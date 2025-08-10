"""
Authentication and User Management API
Comprehensive endpoints for user authentication, registration, and management
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from ...core.rbac import (
    get_auth_service, UserModel, UserRole, Permission, 
    AuthenticationError, AuthorizationError, ValidationError
)
from ...core.errorHandling import SecurityError
from ...core.sessionManagement import get_session_manager
from ...core.auditTrail import audit_log, AuditEventType
from ..middleware.auth import (
    get_current_user, require_admin, require_super_admin, 
    require_permission, get_optional_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)


# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]
    session_id: str


class UserRegistrationRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    mfa_enabled: bool


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, request: Request):
    """
    Authenticate user and create secure session
    
    Returns:
        Access token, refresh token, and session information
    """
    try:
        auth_service = get_auth_service()
        session_manager = get_session_manager()
        
        # Authenticate user
        user = auth_service.authenticate_user(
            username=login_data.username,
            password=login_data.password
        )
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extract device fingerprint if provided
        device_fingerprint = request.headers.get("x-device-fingerprint")
        
        # Create session with tokens
        access_token, refresh_token, session_info = await session_manager.create_session(
            user_id=user.user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint
        )
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Log security event
        logger.info(f"User login successful: {user.username} from {client_ip}")
        
        # Audit log successful login
        await audit_log(
            event_type=AuditEventType.USER_LOGIN,
            user_id=user.user_id,
            session_id=session_info.session_id,
            ip_address=client_ip,
            user_agent=user_agent,
            outcome="success",
            details={
                "username": user.username,
                "login_method": "password"
            }
        )
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=session_manager.access_token_expire_minutes * 60,
            session_id=session_info.session_id,
            user={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        )
        
    except AuthenticationError as e:
        # Log failed login attempt
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Failed login attempt for {login_data.username} from {client_ip}: {e}")
        
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service unavailable"
        )


@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Logout current user by revoking their token
    """
    try:
        auth_service = get_auth_service()
        session_id = current_user.get("session_id")
        
        if session_id:
            auth_service.revoke_token(session_id)
            logger.info(f"User logged out: {current_user.get('username')}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user information
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(current_user["user_id"])
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            mfa_enabled=user.mfa_enabled
        )
        
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user information")


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Change current user's password
    """
    try:
        auth_service = get_auth_service()
        auth_service.change_password(
            user_id=current_user["user_id"],
            old_password=password_data.current_password,
            new_password=password_data.new_password
        )
        
        logger.info(f"Password changed for user: {current_user.get('username')}")
        return {"message": "Password changed successfully"}
        
    except (AuthenticationError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.post("/create-api-key")
async def create_api_key(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Create API key for current user
    """
    try:
        auth_service = get_auth_service()
        api_key = auth_service.create_api_key(current_user["user_id"])
        
        logger.info(f"API key created for user: {current_user.get('username')}")
        
        return {
            "api_key": api_key,
            "message": "API key created successfully",
            "warning": "Store this key securely - it will not be shown again"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")


# Admin-only endpoints
@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserRegistrationRequest,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Register new user (admin only)
    """
    try:
        auth_service = get_auth_service()
        
        # Create user
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role,
            full_name=user_data.full_name
        )
        
        logger.info(f"User registered by admin {admin_user.get('username')}: {user.username}")
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            mfa_enabled=user.mfa_enabled
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    admin_user: Dict[str, Any] = Depends(require_admin),
    active_only: bool = True
):
    """
    List all users (admin only)
    """
    try:
        auth_service = get_auth_service()
        
        users = []
        for user in auth_service.users.values():
            if not active_only or user.is_active:
                users.append(UserResponse(
                    user_id=user.user_id,
                    username=user.username,
                    email=user.email,
                    full_name=user.full_name,
                    role=user.role.value,
                    is_active=user.is_active,
                    created_at=user.created_at,
                    last_login=user.last_login,
                    mfa_enabled=user.mfa_enabled
                ))
        
        return users
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Get specific user by ID (admin only)
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            mfa_enabled=user.mfa_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UserUpdateRequest,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Update user information (admin only)
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update user fields
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
        
        if update_data.email is not None:
            user.email = update_data.email
        
        if update_data.role is not None:
            # Only super admin can change roles
            if admin_user.get("role") != UserRole.SUPER_ADMIN.value:
                raise HTTPException(status_code=403, detail="Only super admin can change user roles")
            user.role = update_data.role
        
        if update_data.is_active is not None:
            user.is_active = update_data.is_active
        
        logger.info(f"User updated by admin {admin_user.get('username')}: {user.username}")
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            mfa_enabled=user.mfa_enabled
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/users/{user_id}")
async def deactivate_user(
    user_id: str,
    super_admin: Dict[str, Any] = Depends(require_super_admin)
):
    """
    Deactivate user (super admin only)
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent super admin from deactivating themselves
        if user_id == super_admin["user_id"]:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        
        user.is_active = False
        
        # Revoke all user sessions
        for session_id, session in auth_service.sessions.items():
            if session["user_id"] == user_id:
                auth_service.revoke_token(session_id)
        
        logger.warning(f"User deactivated by super admin {super_admin.get('username')}: {user.username}")
        
        return {"message": f"User {user.username} has been deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate user")


@router.get("/stats")
async def get_user_stats(admin_user: Dict[str, Any] = Depends(require_admin)):
    """
    Get user statistics (admin only)
    """
    try:
        auth_service = get_auth_service()
        stats = auth_service.get_user_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get user stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user statistics")


@router.get("/sessions")
async def list_active_sessions(admin_user: Dict[str, Any] = Depends(require_admin)):
    """
    List all active sessions (admin only)
    """
    try:
        auth_service = get_auth_service()
        
        sessions = []
        for session_id, session_data in auth_service.sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_id": session_data["user_id"],
                "username": session_data["username"],
                "role": session_data["role"],
                "created_at": session_data["created_at"].isoformat(),
                "expires_at": session_data["expires_at"].isoformat(),
                "ip_address": session_data.get("ip_address"),
                "user_agent": session_data.get("user_agent")
            })
        
        return {
            "success": True,
            "data": sessions,
            "total_sessions": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active sessions")


@router.post("/revoke-session/{session_id}")
async def revoke_session(
    session_id: str,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Revoke specific session (admin only)
    """
    try:
        auth_service = get_auth_service()
        
        if session_id not in auth_service.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = auth_service.sessions[session_id]
        auth_service.revoke_token(session_id)
        
        logger.info(f"Session revoked by admin {admin_user.get('username')}: {session_data['username']}")
        
        return {"message": f"Session for user {session_data['username']} has been revoked"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revoke session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke session")


# Export router
__all__ = ["router"]
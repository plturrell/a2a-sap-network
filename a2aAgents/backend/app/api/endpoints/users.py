"""
User Management API
Comprehensive endpoints for user profile management and user operations
"""
import email

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ...core.rbac import (
    get_auth_service, UserModel, UserRole, Permission, 
    AuthenticationError, AuthorizationError, ValidationError
)
from ...core.auditTrail import audit_log, AuditEventType
from ..middleware.auth import (
    get_current_user, require_admin, require_super_admin
)

router = APIRouter(prefix="/users", tags=["User Management"])
logger = logging.getLogger(__name__)


# Request/Response models
class UserProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


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


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total_count: int
    page: int
    page_size: int


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user's profile information
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
        logger.error(f"Get user profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    profile_data: UserProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update current user's profile information
    """
    try:
        auth_service = get_auth_service()
        user_id = current_user["user_id"]
        user = auth_service.users.get(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Track changes for audit
        changes = {}
        
        # Update full name if provided
        if profile_data.full_name is not None:
            old_name = user.full_name
            user.full_name = profile_data.full_name
            changes["full_name"] = {"old": old_name, "new": profile_data.full_name}
        
        # Update email if provided
        if profile_data.email is not None:
            # Check if email already exists
            if any(u.email == profile_data.email and u.user_id != user_id 
                   for u in auth_service.users.values()):
                raise HTTPException(status_code=400, detail="Email already in use")
            
            old_email = user.email
            user.email = profile_data.email
            changes["email"] = {"old": old_email, "new": profile_data.email}
        
        if changes:
            # Log profile update
            logger.info(f"User profile updated: {user.username}")
            
            # Audit log profile update
            await audit_log(
                event_type=AuditEventType.USER_UPDATED,
                user_id=user.user_id,
                outcome="success",
                details={
                    "action": "profile_update",
                    "username": user.username,
                    "changes": changes
                }
            )
        
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
        logger.error(f"Update user profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile")


@router.get("/search", response_model=UserListResponse)
async def search_users(
    admin_user: Dict[str, Any] = Depends(require_admin),
    username: Optional[str] = Query(None, description="Filter by username"),
    email: Optional[str] = Query(None, description="Filter by email"),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    active_only: bool = Query(True, description="Show only active users"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    Search and filter users (admin only)
    """
    try:
        auth_service = get_auth_service()
        
        # Get all users
        all_users = list(auth_service.users.values())
        
        # Apply filters
        filtered_users = []
        for user in all_users:
            # Active filter
            if active_only and not user.is_active:
                continue
                
            # Username filter
            if username and username.lower() not in user.username.lower():
                continue
                
            # Email filter
            if email and email.lower() not in user.email.lower():
                continue
                
            # Role filter
            if role and user.role != role:
                continue
                
            filtered_users.append(user)
        
        # Pagination
        total_count = len(filtered_users)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_users = filtered_users[start_idx:end_idx]
        
        # Convert to response format
        user_responses = [
            UserResponse(
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
            for user in paginated_users
        ]
        
        return UserListResponse(
            users=user_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Search users error: {e}")
        raise HTTPException(status_code=500, detail="Failed to search users")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
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
        logger.error(f"Get user by ID error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.delete("/{user_id}")
async def deactivate_user(
    user_id: str,
    super_admin: Dict[str, Any] = Depends(require_super_admin)
):
    """
    Deactivate user account (super admin only)
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent super admin from deactivating themselves
        if user_id == super_admin["user_id"]:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        
        # Deactivate user
        user.is_active = False
        
        # Revoke all user sessions
        sessions_to_revoke = []
        for session_id, session in auth_service.sessions.items():
            if session["user_id"] == user_id:
                sessions_to_revoke.append(session_id)
        
        for session_id in sessions_to_revoke:
            auth_service.revoke_token(session_id)
        
        logger.warning(f"User deactivated by super admin {super_admin.get('username')}: {user.username}")
        
        # Audit log user deactivation
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "user_deactivated",
                "username": user.username,
                "deactivated_by": super_admin.get("username"),
                "sessions_revoked": len(sessions_to_revoke)
            }
        )
        
        return {
            "message": f"User {user.username} has been deactivated",
            "sessions_revoked": len(sessions_to_revoke)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to deactivate user")


# Export router
__all__ = ["router"]
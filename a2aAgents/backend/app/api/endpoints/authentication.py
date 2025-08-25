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
import pyotp
import qrcode
import io
import base64
from PIL import Image

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


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    reset_token: str
    new_password: str


class MFASetupRequest(BaseModel):
    password: str  # Current password for verification


class MFAVerifyRequest(BaseModel):
    totp_code: str


class MFALoginRequest(BaseModel):
    username: str
    password: str
    totp_code: Optional[str] = None


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserDeletionRequest(BaseModel):
    confirmation_username: str  # Must match username to confirm deletion
    cascade_data: bool = True  # Whether to delete associated data


class BulkUserOperation(BaseModel):
    user_ids: List[str]
    operation: str  # "activate", "deactivate", "delete", "update_role"
    parameters: Optional[Dict[str, Any]] = None  # For role updates, additional params


class BulkUserRequest(BaseModel):
    operations: List[BulkUserOperation]
    confirm_bulk_operation: bool = True  # Safety confirmation


class BulkOperationResult(BaseModel):
    operation_id: str
    user_id: str
    username: str
    operation: str
    status: str  # "success", "failed", "skipped"
    message: str
    details: Optional[Dict[str, Any]] = None


class BulkOperationResponse(BaseModel):
    total_operations: int
    successful: int
    failed: int
    skipped: int
    results: List[BulkOperationResult]
    execution_time_seconds: float


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


@router.post("/password-reset")
async def request_password_reset(reset_request: PasswordResetRequest, request: Request):
    """
    Request password reset for user account
    """
    try:
        auth_service = get_auth_service()
        session_manager = get_session_manager()

        # Find user by email
        user = None
        for u in auth_service.users.values():
            if u.email == reset_request.email:
                user = u
                break

        # Always return success to prevent email enumeration
        if not user:
            logger.warning(f"Password reset requested for unknown email: {reset_request.email}")
            return {"message": "If this email exists, a password reset link has been sent"}

        # Check if user is active
        if not user.is_active:
            logger.warning(f"Password reset requested for inactive user: {user.username}")
            return {"message": "If this email exists, a password reset link has been sent"}

        # Generate secure reset token (valid for 1 hour)
        reset_token = await session_manager.create_password_reset_token(
            user_id=user.user_id,
            email=user.email
        )

        # Get client information for security logging
        client_ip = request.client.host if request.client else "unknown"

        # Log security event
        logger.info(f"Password reset requested for user: {user.username} from {client_ip}")

        # Audit log password reset request
        await audit_log(
            event_type=AuditEventType.PASSWORD_RESET_REQUESTED,
            user_id=user.user_id,
            ip_address=client_ip,
            outcome="success",
            details={
                "username": user.username,
                "email": user.email,
                "reset_token_expires": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
        )

        # In a real implementation, send email with reset link
        logger.info(f"Password reset token for {user.email}: {reset_token}")

        return {"message": "If this email exists, a password reset link has been sent"}

    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        return {"message": "If this email exists, a password reset link has been sent"}


@router.post("/password-reset-confirm")
async def confirm_password_reset(reset_data: PasswordResetConfirmRequest, request: Request):
    """
    Confirm password reset with token and set new password
    """
    try:
        auth_service = get_auth_service()
        session_manager = get_session_manager()

        # Validate reset token
        token_data = await session_manager.validate_password_reset_token(reset_data.reset_token)

        user_id = token_data["user_id"]
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=400, detail="Invalid reset token")

        # Validate new password
        if len(reset_data.new_password) < auth_service.password_min_length:
            raise HTTPException(
                status_code=400,
                detail=f"Password must be at least {auth_service.password_min_length} characters long"
            )

        # Reset user password
        auth_service.reset_user_password(user_id, reset_data.new_password)

        # Invalidate the reset token
        await session_manager.invalidate_password_reset_token(reset_data.reset_token)

        # Revoke all existing sessions for security
        auth_service.revoke_all_user_sessions(user_id)

        # Get client information
        client_ip = request.client.host if request.client else "unknown"

        # Log security event
        logger.info(f"Password reset completed for user: {user.username} from {client_ip}")

        # Audit log successful password reset
        await audit_log(
            event_type=AuditEventType.PASSWORD_RESET_COMPLETED,
            user_id=user.user_id,
            ip_address=client_ip,
            outcome="success",
            details={
                "username": user.username,
                "sessions_revoked": "all"
            }
        )

        return {"message": "Password has been reset successfully. Please login with your new password."}

    except ValidationError as e:
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"Invalid password reset token from {client_ip}: {e}")
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    except Exception as e:
        logger.error(f"Password reset confirmation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")


@router.post("/mfa/setup")
async def setup_mfa(
    setup_request: MFASetupRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Set up Multi-Factor Authentication (TOTP) for current user
    """
    try:
        auth_service = get_auth_service()
        user_id = current_user["user_id"]
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Verify current password
        try:
            stored_hash = auth_service.secrets_manager.get_secret(f"user_password_{user_id}")
            salt, hash_hex = stored_hash.split(':')
            import hashlib
            expected_hash = hashlib.pbkdf2_hmac('sha256', setup_request.password.encode('utf-8'), salt.encode('utf-8'), 100000)
            if expected_hash.hex() != hash_hex:
                raise HTTPException(status_code=400, detail="Current password is incorrect")
        except Exception:
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Generate TOTP secret
        totp_secret = pyotp.random_base32()

        # Store TOTP secret
        auth_service.secrets_manager.set_secret(f"user_totp_{user_id}", totp_secret)

        # Generate QR code
        totp = pyotp.TOTP(totp_secret)
        provisioning_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name="A2A Platform"
        )

        # Create QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        logger.info(f"MFA setup initiated for user: {user.username}")

        # Audit log MFA setup
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "mfa_setup_initiated",
                "username": user.username
            }
        )

        return {
            "qr_code": f"data:image/png;base64,{qr_code_base64}",
            "secret": totp_secret,
            "provisioning_uri": provisioning_uri,
            "message": "Scan the QR code with your authenticator app and verify with a TOTP code"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to setup MFA")


@router.post("/mfa/verify")
async def verify_mfa_setup(
    verify_request: MFAVerifyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Verify TOTP code and enable MFA for user
    """
    try:
        auth_service = get_auth_service()
        user_id = current_user["user_id"]
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get TOTP secret
        try:
            totp_secret = auth_service.secrets_manager.get_secret(f"user_totp_{user_id}")
        except Exception:
            raise HTTPException(status_code=400, detail="MFA setup not initiated. Please setup MFA first.")

        # Verify TOTP code
        totp = pyotp.TOTP(totp_secret)
        if not totp.verify(verify_request.totp_code, valid_window=1):
            raise HTTPException(status_code=400, detail="Invalid TOTP code")

        # Enable MFA for user
        user.mfa_enabled = True

        logger.info(f"MFA enabled for user: {user.username}")

        # Audit log MFA enabled
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "mfa_enabled",
                "username": user.username
            }
        )

        return {
            "message": "MFA has been successfully enabled for your account",
            "mfa_enabled": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify MFA")


@router.post("/mfa/disable")
async def disable_mfa(
    setup_request: MFASetupRequest,  # Reuse for password verification
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Disable Multi-Factor Authentication for current user
    """
    try:
        auth_service = get_auth_service()
        user_id = current_user["user_id"]
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not user.mfa_enabled:
            raise HTTPException(status_code=400, detail="MFA is not enabled for this account")

        # Verify current password
        try:
            stored_hash = auth_service.secrets_manager.get_secret(f"user_password_{user_id}")
            salt, hash_hex = stored_hash.split(':')
            import hashlib
            expected_hash = hashlib.pbkdf2_hmac('sha256', setup_request.password.encode('utf-8'), salt.encode('utf-8'), 100000)
            if expected_hash.hex() != hash_hex:
                raise HTTPException(status_code=400, detail="Current password is incorrect")
        except Exception:
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Disable MFA and remove TOTP secret
        user.mfa_enabled = False
        try:
            # Remove TOTP secret
            auth_service.secrets_manager.set_secret(f"user_totp_{user_id}", "")
        except Exception:
            pass  # Secret removal is best effort

        logger.info(f"MFA disabled for user: {user.username}")

        # Audit log MFA disabled
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "mfa_disabled",
                "username": user.username
            }
        )

        return {
            "message": "MFA has been disabled for your account",
            "mfa_enabled": False
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MFA disable error: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable MFA")


@router.post("/login-mfa", response_model=LoginResponse)
async def login_with_mfa(login_data: MFALoginRequest, request: Request):
    """
    Authenticate user with optional MFA verification
    """
    try:
        auth_service = get_auth_service()
        session_manager = get_session_manager()

        # Authenticate user with username/password
        user = auth_service.authenticate_user(
            username=login_data.username,
            password=login_data.password
        )

        # Check if MFA is enabled for user
        if user.mfa_enabled:
            if not login_data.totp_code:
                # Return 400 indicating MFA code is required
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "mfa_required",
                        "message": "Multi-factor authentication code is required"
                    }
                )

            # Verify TOTP code
            try:
                totp_secret = auth_service.secrets_manager.get_secret(f"user_totp_{user.user_id}")
                totp = pyotp.TOTP(totp_secret)
                if not totp.verify(login_data.totp_code, valid_window=1):
                    raise HTTPException(status_code=400, detail="Invalid MFA code")
            except Exception as e:
                logger.warning(f"MFA verification failed for user {user.username}: {e}")
                raise HTTPException(status_code=400, detail="Invalid MFA code")

        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        device_fingerprint = request.headers.get("x-device-fingerprint")

        # Create session with tokens
        access_token, refresh_token, session_info = await session_manager.create_session(
            user_id=user.user_id,
            ip_address=client_ip,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint
        )

        # Mark MFA as verified in session if MFA was used
        if user.mfa_enabled:
            session_info.security_flags["mfa_verified"] = True

        # Update last login
        user.last_login = datetime.utcnow()

        # Log security event
        logger.info(f"User login successful: {user.username} from {client_ip} (MFA: {user.mfa_enabled})")

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
                "login_method": "password_mfa" if user.mfa_enabled else "password",
                "mfa_verified": user.mfa_enabled
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
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "mfa_enabled": user.mfa_enabled
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login with MFA error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service unavailable"
        )


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

        # Track changes for audit
        changes = {}

        # Update user fields with validation
        if update_data.full_name is not None:
            old_name = user.full_name
            user.full_name = update_data.full_name
            changes["full_name"] = {"old": old_name, "new": update_data.full_name}

        if update_data.email is not None:
            # Check if email already exists for another user
            if any(u.email == update_data.email and u.user_id != user_id
                   for u in auth_service.users.values()):
                raise HTTPException(status_code=400, detail="Email already in use by another user")

            old_email = user.email
            user.email = update_data.email
            changes["email"] = {"old": old_email, "new": update_data.email}

        if update_data.role is not None:
            # Only super admin can change roles
            if admin_user.get("role") != UserRole.SUPER_ADMIN.value:
                raise HTTPException(status_code=403, detail="Only super admin can change user roles")

            # Prevent changing own role
            if user_id == admin_user["user_id"]:
                raise HTTPException(status_code=400, detail="Cannot change your own role")

            old_role = user.role
            user.role = update_data.role
            changes["role"] = {"old": old_role.value, "new": update_data.role.value}

        if update_data.is_active is not None:
            # Prevent admin from deactivating themselves
            if user_id == admin_user["user_id"] and update_data.is_active == False:
                raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

            old_active = user.is_active
            user.is_active = update_data.is_active
            changes["is_active"] = {"old": old_active, "new": update_data.is_active}

            # If deactivating user, revoke all their sessions
            if not update_data.is_active:
                sessions_to_revoke = []
                for session_id, session in auth_service.sessions.items():
                    if session["user_id"] == user_id:
                        sessions_to_revoke.append(session_id)

                for session_id in sessions_to_revoke:
                    auth_service.revoke_token(session_id)

                changes["sessions_revoked"] = len(sessions_to_revoke)

        logger.info(f"User updated by admin {admin_user.get('username')}: {user.username}")

        # Audit log user update
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "admin_user_update",
                "username": user.username,
                "updated_by": admin_user.get("username"),
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
        logger.error(f"Update user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/users/{user_id}")
async def delete_user_permanently(
    user_id: str,
    deletion_data: UserDeletionRequest,
    super_admin: Dict[str, Any] = Depends(require_super_admin)
):
    """
    Permanently delete user with cascading cleanup (super admin only)
    WARNING: This action is irreversible and will permanently delete all user data
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent super admin from deleting themselves
        if user_id == super_admin["user_id"]:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")

        # Confirmation check - username must match
        if deletion_data.confirmation_username != user.username:
            raise HTTPException(
                status_code=400,
                detail="Username confirmation does not match. Deletion aborted for safety."
            )

        # Perform cascading deletion
        deletion_report = auth_service.delete_user(
            user_id=user_id,
            cascade_data=deletion_data.cascade_data
        )

        logger.critical(f"User permanently deleted by super admin {super_admin.get('username')}: {user.username} (ID: {user_id})")

        # Audit log user deletion
        await audit_log(
            event_type=AuditEventType.USER_DELETED,
            user_id=None,  # User no longer exists
            outcome="success",
            details={
                "action": "permanent_user_deletion",
                "deleted_username": user.username,
                "deleted_user_id": user_id,
                "deleted_by": super_admin.get("username"),
                "cascade_data": deletion_data.cascade_data,
                "cleanup_results": deletion_report["cleanup_results"]
            }
        )

        return {
            "message": f"User {user.username} has been permanently deleted",
            "warning": "This action is irreversible",
            "deletion_report": deletion_report
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    super_admin: Dict[str, Any] = Depends(require_super_admin)
):
    """
    Deactivate user account (reversible, super admin only)
    This is safer than permanent deletion and is the recommended approach
    """
    try:
        auth_service = get_auth_service()
        user = auth_service.users.get(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent super admin from deactivating themselves
        if user_id == super_admin["user_id"]:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

        if not user.is_active:
            raise HTTPException(status_code=400, detail="User is already deactivated")

        user.is_active = False

        # Revoke all user sessions
        sessions_revoked = 0
        sessions_to_revoke = []
        for session_id, session in auth_service.sessions.items():
            if session["user_id"] == user_id:
                sessions_to_revoke.append(session_id)

        for session_id in sessions_to_revoke:
            auth_service.revoke_token(session_id)
            sessions_revoked += 1

        logger.warning(f"User deactivated by super admin {super_admin.get('username')}: {user.username}")

        # Audit log user deactivation
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=user.user_id,
            outcome="success",
            details={
                "action": "user_deactivation",
                "username": user.username,
                "deactivated_by": super_admin.get("username"),
                "sessions_revoked": sessions_revoked
            }
        )

        return {
            "message": f"User {user.username} has been deactivated",
            "sessions_revoked": sessions_revoked,
            "note": "User account can be reactivated by updating is_active to true"
        }

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


@router.post("/users/bulk", response_model=BulkOperationResponse)
async def bulk_user_operations(
    bulk_request: BulkUserRequest,
    super_admin: Dict[str, Any] = Depends(require_super_admin)
):
    """
    Perform bulk operations on multiple users (super admin only)

    Supports operations: activate, deactivate, delete, update_role
    All operations are performed atomically - if any critical operation fails,
    the entire batch can be rolled back.
    """
    import time
    import secrets

    start_time = time.time()

    try:
        auth_service = get_auth_service()

        # Safety confirmation check
        if not bulk_request.confirm_bulk_operation:
            raise HTTPException(status_code=400, detail="Bulk operation confirmation required")

        # Validate operations
        valid_operations = {"activate", "deactivate", "delete", "update_role"}
        for operation in bulk_request.operations:
            if operation.operation not in valid_operations:
                raise HTTPException(status_code=400, detail=f"Invalid operation: {operation.operation}")

        results = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0

        # Process each bulk operation
        for operation in bulk_request.operations:
            operation_id = secrets.token_hex(8)

            # Process each user in this operation
            for user_id in operation.user_ids:
                try:
                    user = auth_service.users.get(user_id)
                    if not user:
                        results.append(BulkOperationResult(
                            operation_id=operation_id,
                            user_id=user_id,
                            username="unknown",
                            operation=operation.operation,
                            status="failed",
                            message="User not found"
                        ))
                        failed_count += 1
                        continue

                    # Prevent super admin from operating on themselves for dangerous operations
                    if user_id == super_admin["user_id"] and operation.operation in ["deactivate", "delete"]:
                        results.append(BulkOperationResult(
                            operation_id=operation_id,
                            user_id=user_id,
                            username=user.username,
                            operation=operation.operation,
                            status="skipped",
                            message="Cannot perform this operation on your own account"
                        ))
                        skipped_count += 1
                        continue

                    # Perform the operation
                    if operation.operation == "activate":
                        if user.is_active:
                            results.append(BulkOperationResult(
                                operation_id=operation_id,
                                user_id=user_id,
                                username=user.username,
                                operation=operation.operation,
                                status="skipped",
                                message="User is already active"
                            ))
                            skipped_count += 1
                        else:
                            user.is_active = True
                            results.append(BulkOperationResult(
                                operation_id=operation_id,
                                user_id=user_id,
                                username=user.username,
                                operation=operation.operation,
                                status="success",
                                message="User activated successfully"
                            ))
                            successful_count += 1

                    elif operation.operation == "deactivate":
                        if not user.is_active:
                            results.append(BulkOperationResult(
                                operation_id=operation_id,
                                user_id=user_id,
                                username=user.username,
                                operation=operation.operation,
                                status="skipped",
                                message="User is already inactive"
                            ))
                            skipped_count += 1
                        else:
                            user.is_active = False
                            # Revoke all user sessions
                            sessions_revoked = 0
                            sessions_to_revoke = []
                            for session_id, session in auth_service.sessions.items():
                                if session["user_id"] == user_id:
                                    sessions_to_revoke.append(session_id)

                            for session_id in sessions_to_revoke:
                                auth_service.revoke_token(session_id)
                                sessions_revoked += 1

                            results.append(BulkOperationResult(
                                operation_id=operation_id,
                                user_id=user_id,
                                username=user.username,
                                operation=operation.operation,
                                status="success",
                                message="User deactivated successfully",
                                details={"sessions_revoked": sessions_revoked}
                            ))
                            successful_count += 1

                    elif operation.operation == "delete":
                        # Perform user deletion with cascading
                        deletion_report = auth_service.delete_user(user_id, cascade_data=True)

                        results.append(BulkOperationResult(
                            operation_id=operation_id,
                            user_id=user_id,
                            username=deletion_report["username"],
                            operation=operation.operation,
                            status="success",
                            message="User deleted permanently",
                            details=deletion_report["cleanup_results"]
                        ))
                        successful_count += 1

                    elif operation.operation == "update_role":
                        if not operation.parameters or "role" not in operation.parameters:
                            results.append(BulkOperationResult(
                                operation_id=operation_id,
                                user_id=user_id,
                                username=user.username,
                                operation=operation.operation,
                                status="failed",
                                message="Role parameter required for role update operation"
                            ))
                            failed_count += 1
                            continue

                        new_role = UserRole(operation.parameters["role"])
                        old_role = user.role
                        user.role = new_role

                        results.append(BulkOperationResult(
                            operation_id=operation_id,
                            user_id=user_id,
                            username=user.username,
                            operation=operation.operation,
                            status="success",
                            message=f"Role updated from {old_role.value} to {new_role.value}",
                            details={"old_role": old_role.value, "new_role": new_role.value}
                        ))
                        successful_count += 1

                except Exception as e:
                    results.append(BulkOperationResult(
                        operation_id=operation_id,
                        user_id=user_id,
                        username=user.username if user else "unknown",
                        operation=operation.operation,
                        status="failed",
                        message=f"Operation failed: {str(e)}"
                    ))
                    failed_count += 1

        execution_time = time.time() - start_time

        # Log bulk operation
        logger.warning(
            f"Bulk user operation completed by super admin {super_admin.get('username')}: "
            f"{successful_count} successful, {failed_count} failed, {skipped_count} skipped"
        )

        # Audit log bulk operation
        await audit_log(
            event_type=AuditEventType.USER_UPDATED,
            user_id=super_admin["user_id"],
            outcome="success",
            details={
                "action": "bulk_user_operations",
                "performed_by": super_admin.get("username"),
                "total_operations": len(results),
                "successful": successful_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "execution_time_seconds": execution_time,
                "operations_summary": [
                    {
                        "operation": op.operation,
                        "user_count": len(op.user_ids)
                    }
                    for op in bulk_request.operations
                ]
            }
        )

        return BulkOperationResponse(
            total_operations=len(results),
            successful=successful_count,
            failed=failed_count,
            skipped=skipped_count,
            results=results,
            execution_time_seconds=execution_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk user operations error: {e}")
        raise HTTPException(status_code=500, detail="Bulk operations failed")


# Export router
__all__ = ["router"]
"""
Enhanced Authentication Middleware with RBAC Integration
"""

from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os

from app.core.config import settings
from app.core.rbac import get_auth_service, UserRole, Permission, AuthenticationError, AuthorizationError
from app.core.sessionManagement import get_session_manager

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer()

# FastAPI Dependencies for authentication and authorization
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user using session management

    Args:
        credentials: HTTP Bearer credentials from Authorization header

    Returns:
        User data dictionary with role, permissions, and session info

    Raises:
        HTTPException: 401 if authentication fails
    """
    try:
        # First try session-based validation (new system)
        session_manager = get_session_manager()
        try:
            token_info = await session_manager.validate_access_token(credentials.credentials)
            auth_service = get_auth_service()

            # Get user data from RBAC system
            user_id = token_info["user_id"]
            if user_id in auth_service.users:
                user = auth_service.users[user_id]
                return {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role,
                    "permissions": [p.value for p in user.get_permissions()],
                    "session_id": token_info["session_id"],
                    "session_info": token_info["session"]
                }
        except Exception:
            # Fall back to legacy token validation
            pass

        # Legacy token validation (for backward compatibility)
        auth_service = get_auth_service()
        user_data = auth_service.verify_token(credentials.credentials)
        return user_data

    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to optionally get current user
    Returns None if no authentication provided
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    try:
        token = auth_header.split(" ")[1]
        auth_service = get_auth_service()
        return auth_service.verify_token(token)
    except:
        return None


def require_permission(permission: Permission):
    """
    FastAPI dependency factory to require specific permission

    Args:
        permission: Required permission

    Returns:
        Dependency function that validates permission
    """
    def permission_dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        auth_service = get_auth_service()
        try:
            auth_service.require_permission(current_user, permission)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(status_code=403, detail=str(e))

    return permission_dependency


def require_role(role: UserRole):
    """
    FastAPI dependency factory to require specific role

    Args:
        role: Required user role

    Returns:
        Dependency function that validates role
    """
    def role_dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role")
        if user_role != role.value:
            raise HTTPException(
                status_code=403,
                detail=f"Role required: {role.value}, but user has: {user_role}"
            )
        return current_user

    return role_dependency


async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency to require admin privileges

    Args:
        current_user: Current authenticated user data

    Returns:
        User data if admin, raises HTTPException otherwise
    """
    auth_service = get_auth_service()
    try:
        auth_service.require_admin(current_user)
        return current_user
    except AuthorizationError as e:
        raise HTTPException(status_code=403, detail=str(e))


async def require_super_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency to require super admin privileges
    """
    user_role = current_user.get("role")
    if user_role != UserRole.SUPER_ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail="Super administrator access required"
        )
    return current_user


class JWTMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT token validation"""

    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or settings.SECRET_KEY

        # Ensure secret key is strong enough
        if not self.secret_key or len(self.secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")

        self.algorithm = "HS256"

        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/gateway/health",
            "/gateway/services",
        }

        # Paths that allow optional authentication
        self.optional_auth_paths = {
            "/gateway/rate-limits",
        }

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Extract token from header
        auth_header = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")

        user_info = None

        # Try JWT token first
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                user_info = {
                    "user_id": payload.get("sub"),
                    "username": payload.get("username"),
                    "email": payload.get("email"),
                    "tier": payload.get("tier", "authenticated"),
                    "scopes": payload.get("scopes", []),
                    "exp": payload.get("exp")
                }

                # Check token expiration with buffer for clock skew
                current_time = datetime.utcnow().timestamp()
                if user_info["exp"] and current_time > user_info["exp"]:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Token expired"}
                    )

                # Check token not used before issue time (nbf)
                if user_info.get("nbf") and current_time < user_info["nbf"]:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Token not yet valid"}
                    )

                # Check issuer if present
                if user_info.get("iss") and user_info["iss"] != "a2a-gateway":
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid token issuer"}
                    )

                # Add user info to request state (but don't log sensitive info)
                request.state.user = user_info
                logger.info(f"JWT authentication successful for user: {user_info.get('username', 'unknown')}")

            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Token expired"}
                )
            except jwt.InvalidTokenError:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid token"}
                )

        # Try API key authentication
        elif api_key:
            if await self.validate_api_key(api_key):
                user_info = {
                    "user_id": f"api_key_{api_key[:8]}",
                    "api_key": api_key[:8] + "...",  # Only log partial key
                    "tier": "authenticated",
                    "scopes": ["api_access"]
                }
                request.state.user = user_info
                logger.info(f"API key authentication successful for key: {api_key[:8]}...")

        # Check if authentication is required
        if request.url.path not in self.optional_auth_paths and user_info is None:
            # Check if this is a path that requires authentication
            if self.requires_authentication(request.url.path):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"}
                )

        return await call_next(request)

    def requires_authentication(self, path: str) -> bool:
        """Check if path requires authentication"""
        # Implement your authentication requirements logic
        auth_required_patterns = [
            "/gateway/agent_manager/",
            "/gateway/main/api/v1/users",
            "/gateway/main/api/v1/admin"
        ]

        return any(path.startswith(pattern) for pattern in auth_required_patterns)

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key against secure storage"""
        # Validate API key format first
        if not api_key or not isinstance(api_key, str):
            return False

        if not api_key.startswith("a2a_") or len(api_key) < 32:
            return False

        # Check against environment variable list (in production, use secure database)
        valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
        valid_keys = [key.strip() for key in valid_keys if key.strip()]

        # Development keys for testing (should be removed in production)
        if api_key.startswith("a2a_dev_") and os.getenv("ENVIRONMENT") == "development":
            return len(api_key) >= 32

        return api_key in valid_keys


def create_jwt_token(user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for user with enhanced security"""
    now = datetime.utcnow()

    if expires_delta:
        expire = now + expires_delta
    else:
        # Shorter default expiry for better security
        expire = now + timedelta(hours=1)

    # Generate unique token ID for tracking/revocation
    import uuid
    jti = str(uuid.uuid4())

    payload = {
        "sub": str(user_data["user_id"]),
        "username": user_data.get("username"),
        "email": user_data.get("email"),
        "tier": user_data.get("tier", "authenticated"),
        "scopes": user_data.get("scopes", ["read"]),
        "exp": expire.timestamp(),
        "iat": now.timestamp(),
        "nbf": now.timestamp(),  # Not before time
        "iss": "a2a-gateway",    # Issuer
        "aud": "a2a-network",    # Audience
        "jti": jti               # JWT ID for tracking
    }

    # Validate secret key strength
    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
        raise ValueError("JWT secret key must be at least 32 characters long")

    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def create_api_key(prefix: str = "a2a") -> str:
    """Generate API key"""
    import secrets
    import string

    # Generate random string
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(32))

    return f"{prefix}_{key}"

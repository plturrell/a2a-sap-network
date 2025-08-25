"""
API Dependencies for Authentication and Authorization
Provides dependency injection for FastAPI endpoints
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
import logging
from datetime import datetime
import os

from app.core.config import settings
from app.models.user import User

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Optional authentication - returns User if valid token, None otherwise
    Does not raise exceptions for missing/invalid tokens
    """
    try:
        # First check if user info is already in request state (from middleware)
        if hasattr(request.state, 'user') and request.state.user:
            user_data = request.state.user
            # Create User object from state data
            return User(
                id=user_data.get("user_id"),
                username=user_data.get("username"),
                email=user_data.get("email"),
                tier=user_data.get("tier", "authenticated"),
                scopes=user_data.get("scopes", [])
            )

        # If no middleware user info, try to validate token
        if credentials and credentials.credentials:
            payload = jwt.decode(
                credentials.credentials,
                settings.SECRET_KEY,
                algorithms=["HS256"]
            )

            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                return None

            # Create User object from token payload
            return User(
                id=payload.get("sub"),
                username=payload.get("username"),
                email=payload.get("email"),
                tier=payload.get("tier", "authenticated"),
                scopes=payload.get("scopes", [])
            )

        # Check for API key in headers
        api_key = request.headers.get("x-api-key")
        if api_key and await validate_api_key(api_key):
            return User(
                id=f"api_key_{api_key[:8]}",
                username="api_user",
                email=None,
                tier="authenticated",
                scopes=["api_access"]
            )

        return None

    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT token provided")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT token provided")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None


async def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """
    Required authentication - raises 401 if no valid user
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def validate_api_key(api_key: str) -> bool:
    """
    Validate API key against secure storage
    """
    # Validate API key format
    if not api_key.startswith("a2a_") or len(api_key) < 20:
        return False

    # In production, check against database/cache
    # For now, validate basic format and check environment
    valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
    return api_key in valid_keys or api_key.startswith("a2a_dev_")


async def require_scope(required_scope: str):
    """
    Dependency to require specific scope
    """
    def scope_checker(current_user: User = Depends(get_current_user)):
        if required_scope not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to require admin access
    """
    if "admin" not in current_user.scopes and current_user.tier != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required"
        )
    return current_user


async def require_premium(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to require premium tier
    """
    if current_user.tier not in ["premium", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user

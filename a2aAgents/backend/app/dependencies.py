"""
Common dependencies for FastAPI endpoints
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Simple bearer token security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """
    Get current user from bearer token.
    For development, this is a simple implementation.
    """
    if credentials:
        # In development, just return a dummy user
        return "dev_user"
    return None

async def require_auth(current_user: Optional[str] = Depends(get_current_user)) -> str:
    """
    Require authentication for protected endpoints
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

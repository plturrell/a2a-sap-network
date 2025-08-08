"""
JWT Authentication Middleware for API Gateway
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


class JWTMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT token validation"""
    
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or settings.SECRET_KEY
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
                
                # Check token expiration
                if user_info["exp"] and datetime.utcnow().timestamp() > user_info["exp"]:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Token expired"}
                    )
                
                # Add user info to request state
                request.state.user = user_info
                
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
            # Validate API key (implement your API key validation logic)
            if await self.validate_api_key(api_key):
                user_info = {
                    "api_key": api_key[:8] + "...",
                    "tier": "authenticated",
                    "scopes": ["api_access"]
                }
                request.state.user = user_info
        
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
        """Validate API key"""
        # Implement your API key validation logic
        # This could check against a database, cache, etc.
        
        # For demo purposes, accept any key that starts with "a2a_"
        return api_key.startswith("a2a_") and len(api_key) >= 20


def create_jwt_token(user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for user"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    payload = {
        "sub": str(user_data["user_id"]),
        "username": user_data.get("username"),
        "email": user_data.get("email"),
        "tier": user_data.get("tier", "authenticated"),
        "scopes": user_data.get("scopes", ["read"]),
        "exp": expire.timestamp(),
        "iat": datetime.utcnow().timestamp(),
        "iss": "a2a-gateway"
    }
    
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def create_api_key(prefix: str = "a2a") -> str:
    """Generate API key"""
    import secrets
    import string
    
    # Generate random string
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(32))
    
    return f"{prefix}_{key}"
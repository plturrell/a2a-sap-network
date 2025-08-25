"""
A2A Security Middleware
Provides authentication, authorization, rate limiting, and input validation
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import jwt

logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Security configuration settings"""

    # JWT Configuration
    jwt_secret: str = os.getenv('JWT_SECRET', '')
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    rate_limit_burst: int = 10

    # Input Validation
    max_payload_size: int = 1024 * 1024  # 1MB
    max_string_length: int = 10000
    max_array_length: int = 1000

    # Authentication
    auth_enabled: bool = True
    api_key_header: str = "X-API-Key"

    # Security Headers
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]

    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if not v and os.getenv('NODE_ENV') == 'production':
            raise ValueError("JWT_SECRET must be set in production")
        return v


class RateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst = burst
        self.tokens = defaultdict(lambda: burst)
        self.last_update = defaultdict(time.time)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update[identifier]

            # Add tokens based on elapsed time
            self.tokens[identifier] = min(
                self.burst,
                self.tokens[identifier] + elapsed * self.rate
            )
            self.last_update[identifier] = now

            if self.tokens[identifier] >= 1:
                self.tokens[identifier] -= 1
                return True
            return False

    def get_retry_after(self, identifier: str) -> int:
        """Get seconds until next available token"""
        tokens_needed = 1 - self.tokens[identifier]
        if tokens_needed <= 0:
            return 0
        return int(tokens_needed / self.rate) + 1


class InputValidator:
    """Comprehensive input validation"""

    # Patterns for common injection attacks
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript|eval)\b)",
        re.IGNORECASE
    )

    PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.[/\\]")

    SCRIPT_INJECTION_PATTERN = re.compile(
        r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
        re.IGNORECASE | re.DOTALL
    )

    @staticmethod
    def validate_string(value: str, field_name: str, max_length: int = 10000) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        if len(value) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length}")

        # Check for null bytes
        if '\x00' in value:
            raise ValueError(f"{field_name} contains null bytes")

        # Check for control characters (except newline and tab)
        if any(ord(char) < 32 and char not in '\n\t\r' for char in value):
            raise ValueError(f"{field_name} contains invalid control characters")

        return value

    @staticmethod
    def validate_path(path: str) -> str:
        """Validate file paths to prevent traversal attacks"""
        # Remove null bytes
        path = path.replace('\x00', '')

        # Check for path traversal
        if InputValidator.PATH_TRAVERSAL_PATTERN.search(path):
            raise ValueError("Path traversal detected")

        # Normalize path
        normalized = os.path.normpath(path)

        # Ensure path doesn't escape base directory
        if normalized.startswith('..'):
            raise ValueError("Path escapes base directory")

        return normalized

    @staticmethod
    def validate_sql_safe(value: str, field_name: str) -> str:
        """Check for potential SQL injection patterns"""
        if InputValidator.SQL_INJECTION_PATTERN.search(value):
            logger.warning(f"Potential SQL injection in {field_name}: {value[:100]}")
            raise ValueError(f"{field_name} contains potentially unsafe SQL patterns")
        return value

    @staticmethod
    def validate_no_scripts(value: str, field_name: str) -> str:
        """Check for script injection"""
        if InputValidator.SCRIPT_INJECTION_PATTERN.search(value):
            raise ValueError(f"{field_name} contains script tags")
        return value

    @staticmethod
    def validate_array(value: list, field_name: str, max_length: int = 1000) -> list:
        """Validate array input"""
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be an array")

        if len(value) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length}")

        return value

    @staticmethod
    def sanitize_for_logging(data: Any, max_length: int = 1000) -> str:
        """Sanitize data for safe logging"""
        # List of sensitive keys to mask
        SENSITIVE_KEYS = {
            'password', 'secret', 'token', 'api_key', 'apikey',
            'authorization', 'auth', 'credential', 'private_key'
        }

        def mask_sensitive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: '***MASKED***' if k.lower() in SENSITIVE_KEYS else mask_sensitive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_sensitive(item) for item in obj]
            elif isinstance(obj, str) and len(obj) > max_length:
                return obj[:max_length] + '...[truncated]'
            return obj

        masked = mask_sensitive(data)
        return json.dumps(masked, default=str)


class AuthenticationMiddleware:
    """JWT and API key authentication"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.bearer_scheme = HTTPBearer(auto_error=False)

    def create_token(self, user_id: str, agent_id: Optional[str] = None) -> str:
        """Create JWT token"""
        payload = {
            'user_id': user_id,
            'agent_id': agent_id,
            'exp': datetime.utcnow() + timedelta(minutes=self.config.jwt_expiry_minutes),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    async def authenticate(self, request: Request) -> Dict[str, Any]:
        """Authenticate request using JWT or API key"""
        # Check for API key
        api_key = request.headers.get(self.config.api_key_header)
        if api_key:
            # Validate API key (implement your logic)
            return self._validate_api_key(api_key)

        # Check for JWT token
        credentials: HTTPAuthorizationCredentials = await self.bearer_scheme(request)
        if credentials:
            return self.verify_token(credentials.credentials)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No valid authentication provided"
        )

    def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key (implement your logic)"""
        # This is a placeholder - implement your API key validation
        valid_keys = {
            os.getenv('AGENT_API_KEY', ''): {'user_id': 'agent_system', 'agent_id': 'system'}
        }

        if api_key in valid_keys:
            return valid_keys[api_key]

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


class SecurityMiddleware:
    """Combined security middleware"""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.rate_limit_requests,
            burst=self.config.rate_limit_burst
        )
        self.auth = AuthenticationMiddleware(self.config)
        self.validator = InputValidator()

    async def process_request(self, request: Request) -> Dict[str, Any]:
        """Process incoming request through security checks"""
        # 1. Check rate limiting
        if self.config.rate_limit_enabled:
            client_id = self._get_client_identifier(request)
            if not await self.rate_limiter.check_rate_limit(client_id):
                retry_after = self.rate_limiter.get_retry_after(client_id)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)}
                )

        # 2. Authenticate request
        auth_info = {}
        if self.config.auth_enabled:
            auth_info = await self.auth.authenticate(request)

        # 3. Validate content size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.config.max_payload_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Payload too large. Maximum size: {self.config.max_payload_size} bytes"
            )

        return auth_info

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""
        # Try to get authenticated user ID
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.get('user_id', 'unknown')}"

        # Fall back to IP address
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.client.host if request.client else 'unknown'

        return f"ip:{client_ip}"

    def validate_request_data(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> Any:
        """Validate request data"""
        if isinstance(data, str):
            # Validate string input
            data = self.validator.validate_string(data, "input", self.config.max_string_length)
            data = self.validator.validate_no_scripts(data, "input")

        elif isinstance(data, list):
            # Validate array input
            data = self.validator.validate_array(data, "input", self.config.max_array_length)
            # Recursively validate array items
            data = [self.validate_request_data(item) for item in data]

        elif isinstance(data, dict):
            # Recursively validate dictionary values
            validated = {}
            for key, value in data.items():
                # Validate key
                key = self.validator.validate_string(key, "key", 100)
                # Validate value
                validated[key] = self.validate_request_data(value)
            data = validated

        return data


def require_auth(permissions: Optional[List[str]] = None):
    """Decorator for endpoints requiring authentication"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get security middleware from app state
            security: SecurityMiddleware = request.app.state.security

            # Process security checks
            auth_info = await security.process_request(request)

            # Check permissions if specified
            if permissions:
                user_permissions = auth_info.get('permissions', [])
                if not any(perm in user_permissions for perm in permissions):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )

            # Add auth info to request state
            request.state.auth = auth_info

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def validate_input(schema: Optional[Dict[str, Any]] = None):
    """Decorator for input validation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request object in args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request and hasattr(request, 'json'):
                # Get request data
                try:
                    data = await request.json()
                except Exception:
                    data = {}

                # Get security middleware
                security: SecurityMiddleware = request.app.state.security

                # Validate data
                validated_data = security.validate_request_data(data, schema)

                # Replace request data with validated data
                request._json = validated_data

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Logging sanitizer
class SecureLogger:
    """Logger that sanitizes sensitive information"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validator = InputValidator()

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal log method with sanitization"""
        # Sanitize message
        msg = str(msg)

        # Sanitize args
        if args:
            args = tuple(self.validator.sanitize_for_logging(arg) for arg in args)

        # Sanitize kwargs
        if kwargs:
            kwargs = {k: self.validator.sanitize_for_logging(v) for k, v in kwargs.items()}

        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)


# Create secure logger instance
def get_secure_logger(name: str) -> SecureLogger:
    """Get a secure logger instance"""
    return SecureLogger(logging.getLogger(name))

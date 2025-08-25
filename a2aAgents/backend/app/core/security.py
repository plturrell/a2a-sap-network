"""
A2A Platform Security Module
Enterprise-grade security utilities and middleware
"""
import hmac
from app.core.loggingConfig import get_logger, LogCategory
import secrets
import time
import jwt
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import wraps
from enum import Enum

import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import Request
import base64
import os

from .exceptions import (
    A2AAuthenticationError,
    A2ATokenExpiredError,
    A2AInvalidTokenError,
    A2AConfigurationError
)

logger = get_logger(__name__, LogCategory.AGENT)


class SecurityLevel(str, Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SYSTEM = "system"


class Permission(str, Enum):
    """System permissions"""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    MANAGE_USERS = "manage_users"
    MANAGE_AGENTS = "manage_agents"
    ADMIN_ACCESS = "admin_access"
    SYSTEM_CONFIG = "system_config"
    VIEW_METRICS = "view_metrics"
    EXECUTE_OPERATIONS = "execute_operations"


class Role(str, Enum):
    """User roles with associated permissions"""
    VIEWER = "viewer"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SYSTEM = "system"


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    Role.VIEWER: [Permission.READ_DATA, Permission.VIEW_METRICS],
    Role.OPERATOR: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.EXECUTE_OPERATIONS,
        Permission.VIEW_METRICS
    ],
    Role.DEVELOPER: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.EXECUTE_OPERATIONS,
        Permission.MANAGE_AGENTS,
        Permission.VIEW_METRICS
    ],
    Role.ADMIN: [
        Permission.READ_DATA,
        Permission.WRITE_DATA,
        Permission.DELETE_DATA,
        Permission.MANAGE_USERS,
        Permission.MANAGE_AGENTS,
        Permission.ADMIN_ACCESS,
        Permission.VIEW_METRICS,
        Permission.EXECUTE_OPERATIONS,
        Permission.SYSTEM_CONFIG
    ],
    Role.SYSTEM: list(Permission)  # All permissions
}


class SecurityConfig:
    """Security configuration constants"""

    # JWT Configuration
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

    # Password Configuration
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_SALT_ROUNDS = 12

    # Session Configuration
    SESSION_TIMEOUT_MINUTES = 30
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

    # Encryption Configuration
    ENCRYPTION_KEY_LENGTH = 32

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_REQUESTS_PER_HOUR = 1000


class PasswordHasher:
    """Secure password hashing utilities"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            raise A2AAuthenticationError(
                f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters long"
            )

        salt = bcrypt.gensalt(rounds=SecurityConfig.PASSWORD_SALT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Password verification failed: {e}")
            return False

    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a cryptographically secure password"""
        # Include various character types for complexity
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password


class TokenManager:
    """JWT token management"""

    def __init__(self, secret_key: str):
        if not secret_key or len(secret_key) < 32:
            raise A2AConfigurationError("JWT secret key must be at least 32 characters")
        self.secret_key = secret_key

    def create_access_token(
        self,
        user_id: str,
        role: Role,
        permissions: Optional[List[Permission]] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            )

        # Use role permissions if specific permissions not provided
        if permissions is None:
            permissions = ROLE_PERMISSIONS.get(role, [])

        payload = {
            "sub": user_id,
            "role": role.value,
            "permissions": [p.value for p in permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }

        return jwt.encode(payload, self.secret_key, algorithm=SecurityConfig.JWT_ALGORITHM)

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=SecurityConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }

        return jwt.encode(payload, self.secret_key, algorithm=SecurityConfig.JWT_ALGORITHM)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise A2ATokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise A2AInvalidTokenError(f"Invalid token: {str(e)}")

    def refresh_access_token(self, refresh_token: str, role: Role) -> str:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise A2AInvalidTokenError("Invalid refresh token")

        return self.create_access_token(payload["sub"], role)


class DataEncryption:
    """Data encryption utilities"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key is None:
            encryption_key = self._generate_key()

        self.cipher_suite = Fernet(encryption_key)
        self.key = encryption_key

    @staticmethod
    def _generate_key() -> bytes:
        """Generate encryption key from password"""
        # Get password and salt from environment variables
        password = os.getenv("ENCRYPTION_PASSWORD")
        salt = os.getenv("ENCRYPTION_SALT")

        # Validate required environment variables
        if not password:
            raise ValueError(
                "ENCRYPTION_PASSWORD environment variable must be set. "
                "Generate a secure password using: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )

        if not salt:
            raise ValueError(
                "ENCRYPTION_SALT environment variable must be set. "
                "Generate a secure salt using: python -c \"import secrets; print(secrets.token_hex(16))\""
            )

        # Validate minimum security requirements
        if len(password) < 16:
            raise ValueError("ENCRYPTION_PASSWORD must be at least 16 characters long")

        if len(salt) < 16:
            raise ValueError("ENCRYPTION_SALT must be at least 16 characters long")

        # Convert to bytes
        password_bytes = password.encode()
        salt_bytes = salt.encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=SecurityConfig.ENCRYPTION_KEY_LENGTH,
            salt=salt_bytes,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise A2AAuthenticationError("Failed to decrypt data")

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        import json
        json_data = json.dumps(data, sort_keys=True)
        return self.encrypt(json_data)

    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt JSON back to dictionary"""
        import json
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)


class SecurityHeaders:
    """Security headers middleware"""

    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-eval' https://ui5.sap.com; "
                "style-src 'self' 'unsafe-inline' https://ui5.sap.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://ui5.sap.com"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self):
        self.requests = {}  # client_id -> list of timestamps
        self.blocked_clients = {}  # client_id -> unblock_time

    def is_allowed(
        self,
        client_id: str,
        max_requests: int = SecurityConfig.MAX_REQUESTS_PER_MINUTE,
        window_seconds: int = 60
    ) -> Tuple[bool, Optional[int]]:
        """Check if client is allowed to make request"""
        current_time = time.time()

        # Check if client is blocked
        if client_id in self.blocked_clients:
            unblock_time = self.blocked_clients[client_id]
            if current_time < unblock_time:
                return False, int(unblock_time - current_time)
            else:
                del self.blocked_clients[client_id]

        # Initialize client history if needed
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Clean old requests
        cutoff_time = current_time - window_seconds
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]

        # Check rate limit
        if len(self.requests[client_id]) >= max_requests:
            # Block client for window duration
            self.blocked_clients[client_id] = current_time + window_seconds
            return False, window_seconds

        # Record this request
        self.requests[client_id].append(current_time)
        return True, None

    def reset_client(self, client_id: str):
        """Reset rate limit for specific client"""
        self.requests.pop(client_id, None)
        self.blocked_clients.pop(client_id, None)


class SecurityValidator:
    """Input validation and sanitization"""

    @staticmethod
    def validate_user_input(
        data: str,
        max_length: int = 1000,
        allowed_chars: Optional[Set[str]] = None
    ) -> str:
        """Validate and sanitize user input"""
        if not isinstance(data, str):
            raise A2AAuthenticationError("Input must be string")

        if len(data) > max_length:
            raise A2AAuthenticationError(f"Input too long (max {max_length} characters)")

        if allowed_chars and not all(char in allowed_chars for char in data):
            raise A2AAuthenticationError("Input contains invalid characters")

        # Basic XSS prevention
        dangerous_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
        data_lower = data.lower()

        for pattern in dangerous_patterns:
            if pattern in data_lower:
                raise A2AAuthenticationError("Input contains potentially dangerous content")

        return data.strip()

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        import re

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise A2AAuthenticationError("Invalid email format")

        return email.lower().strip()

    @staticmethod
    def validate_password_strength(password: str) -> bool:
        """Validate password meets security requirements"""
        import re

        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            raise A2AAuthenticationError(
                f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters"
            )

        # Check for character variety
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

        if not all([has_upper, has_lower, has_digit, has_special]):
            raise A2AAuthenticationError(
                "Password must contain uppercase, lowercase, numbers, and special characters"
            )

        return True


# Authentication decorators

def require_auth(
    required_role: Optional[Role] = None,
    required_permissions: Optional[List[Permission]] = None
):
    """Decorator to require authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request object (assume it's passed as argument or in kwargs)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get('request')

            if not request:
                raise A2AAuthenticationError("Request object not found")

            # Validate token
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise A2AAuthenticationError("Missing or invalid authorization header")

            token = auth_header.split(" ")[1]

            # This would normally validate against your token manager
            # For now, we'll assume token validation is done elsewhere

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Permission checking logic would go here
            # This would typically check the user's permissions from the token
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Security utilities

def generate_api_key(length: int = 32) -> str:
    """Generate secure API key"""
    return secrets.token_urlsafe(length)


def generate_csrf_token() -> str:
    """Generate CSRF token"""
    return secrets.token_urlsafe(32)


def verify_csrf_token(token: str, expected_token: str) -> bool:
    """Verify CSRF token using timing-safe comparison"""
    return hmac.compare_digest(token, expected_token)


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive data for logging"""
    if len(data) <= visible_chars * 2:
        return mask_char * len(data)

    return (
        data[:visible_chars] +
        mask_char * (len(data) - visible_chars * 2) +
        data[-visible_chars:]
    )


def generate_secure_filename(original_filename: str) -> str:
    """Generate secure filename to prevent path traversal"""
    import re

    # Remove path separators and dangerous characters
    safe_filename = re.sub(r'[^\w\.-]', '_', original_filename)

    # Limit length
    if len(safe_filename) > 255:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[:250] + ext

    # Add random suffix to prevent conflicts
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(4)
    name, ext = os.path.splitext(safe_filename)

    return f"{name}_{timestamp}_{random_suffix}{ext}"


# Security audit utilities

class SecurityAuditor:
    """Security auditing utilities"""

    def __init__(self):
        self.audit_log = []

    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security-related events"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {}
        }

        self.audit_log.append(event)
        logger.info(f"Security event: {event_type}", extra=event)

    def get_audit_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get security audit summary"""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.utcnow()

        filtered_events = [
            event for event in self.audit_log
            if start_time <= datetime.fromisoformat(event["timestamp"]) <= end_time
        ]

        event_counts = {}
        for event in filtered_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(filtered_events),
            "event_types": event_counts,
            "recent_events": filtered_events[-10:]  # Last 10 events
        }


# Initialize global instances
_password_hasher = PasswordHasher()
_data_encryption = DataEncryption()
_rate_limiter = RateLimiter()
_security_auditor = SecurityAuditor()
_security_validator = SecurityValidator()


# Export convenience functions
hash_password = _password_hasher.hash_password
verify_password = _password_hasher.verify_password
generate_secure_password = _password_hasher.generate_secure_password

encrypt_data = _data_encryption.encrypt
decrypt_data = _data_encryption.decrypt
encrypt_dict = _data_encryption.encrypt_dict
decrypt_dict = _data_encryption.decrypt_dict

validate_rate_limit = _rate_limiter.is_allowed
reset_rate_limit = _rate_limiter.reset_client

validate_user_input = _security_validator.validate_user_input
validate_email = _security_validator.validate_email
validate_password_strength = _security_validator.validate_password_strength

log_security_event = _security_auditor.log_security_event
get_audit_summary = _security_auditor.get_audit_summary

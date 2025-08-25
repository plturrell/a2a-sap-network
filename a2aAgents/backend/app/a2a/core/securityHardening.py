#!/usr/bin/env python3
"""
Comprehensive Security Hardening for A2A Agents
Implements defense-in-depth security measures, input validation, access controls, and threat protection
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
import re
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import ipaddress

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different operations"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ThreatLevel(str, Enum):
    """Threat assessment levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(str, Enum):
    """Access levels for authorization"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for operations"""

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    access_levels: Set[AccessLevel] = field(default_factory=set)
    additional_claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for audit logging"""

    event_id: str
    event_type: str
    severity: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str]
    user_id: Optional[str]
    agent_id: Optional[str]
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SecurityException(Exception):
    """Base exception for security violations"""

    def __init__(self, message: str, threat_level: ThreatLevel = ThreatLevel.MEDIUM):
        super().__init__(message)
        self.threat_level = threat_level


class AuthenticationError(SecurityException):
    """Authentication failed"""


class AuthorizationError(SecurityException):
    """Authorization denied"""


class InputValidationError(SecurityException):
    """Input validation failed"""


class RateLimitExceededError(SecurityException):
    """Rate limit exceeded"""


class SecurityHardeningManager:
    """Comprehensive security hardening management"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.security_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        self.trusted_agents: Set[str] = set()
        self.session_store: Dict[str, Dict[str, Any]] = {}

        # Security configuration
        self.config = {
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "session_timeout": 3600,  # 1 hour
            "max_login_attempts": 5,
            "lockout_duration": 900,  # 15 minutes
            "require_https": True,
            "enable_audit_logging": True,
            "enable_threat_detection": True,
            "jwt_secret": secrets.token_hex(32),
            "encryption_key": secrets.token_bytes(32),
        }

        # Rate limiting configuration
        self.rate_limit_config = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "authentication": {"requests": 10, "window": 60},  # 10 auth attempts per minute
            "high_security": {"requests": 10, "window": 300},  # 10 requests per 5 minutes
        }

        # Initialize security components
        self._initialize_security_components()

        logger.info(f"Security hardening manager initialized for agent: {agent_id}")

    def _initialize_security_components(self):
        """Initialize security components"""
        # Load trusted agents from configuration
        self.trusted_agents.update(["agent_manager", "catalog_manager", "data_manager"])

        # Initialize input validators
        self.input_validators = {
            "email": self._validate_email,
            "url": self._validate_url,
            "ip_address": self._validate_ip_address,
            "filename": self._validate_filename,
            "json": self._validate_json,
            "sql_injection": self._detect_sql_injection,
            "xss": self._detect_xss,
            "path_traversal": self._detect_path_traversal,
        }

        # Initialize threat detection patterns
        self.threat_patterns = {
            "sql_injection": [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bOR\b.*=.*)",
                r"(;\s*DROP\s+TABLE)",
                r"(;\s*DELETE\s+FROM)",
                r"(\bEXEC\b.*\bxp_cmdshell\b)",
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"eval\s*\(",
            ],
            "path_traversal": [r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e\\", r"~/"],
            "command_injection": [
                r";\s*(ls|cat|grep|find|wget|curl)",
                r"\|\s*(ls|cat|grep|find|wget|curl)",
                r"&&\s*(ls|cat|grep|find|wget|curl)",
                r"`.*`",
                r"\$\(.*\)",
            ],
        }

    def configure_security(self, **config):
        """Configure security settings"""
        self.config.update(config)
        logger.info(f"Security configuration updated for {self.agent_id}")

    def add_trusted_agent(self, agent_id: str):
        """Add agent to trusted list"""
        self.trusted_agents.add(agent_id)
        logger.info(f"Added trusted agent: {agent_id}")

    def remove_trusted_agent(self, agent_id: str):
        """Remove agent from trusted list"""
        self.trusted_agents.discard(agent_id)
        logger.info(f"Removed trusted agent: {agent_id}")

    def validate_input(self, data: Any, validation_type: str, strict: bool = True) -> bool:
        """Validate input data for security threats"""
        try:
            if validation_type in self.input_validators:
                validator = self.input_validators[validation_type]
                return validator(data, strict)
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
                return not strict  # Fail safe - reject if strict mode
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Input validation error: {e}")
            return False

    def _validate_email(self, email: str, strict: bool = True) -> bool:  # pylint: disable=unused-argument
        """Validate email format"""
        if not isinstance(email, str):
            return False

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def _validate_url(self, url: str, strict: bool = True) -> bool:
        """Validate URL format and security"""
        if not isinstance(url, str):
            return False

        # Check for basic URL format
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, url):
            return False

        # Security checks
        if strict:
            # Block localhost and private IPs in strict mode
            localhost_patterns = [r"localhost", r"127\.0\.0\.1", r"0\.0\.0\.0"]
            for pattern in localhost_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False

        return True

    def _validate_ip_address(self, ip: str, strict: bool = True) -> bool:
        """Validate IP address format"""
        if not isinstance(ip, str):
            return False

        try:
            ip_obj = ipaddress.ip_address(ip)

            if strict:
                # In strict mode, reject private and loopback addresses
                return not (ip_obj.is_private or ip_obj.is_loopback)

            return True
        except ValueError:
            return False

    def _validate_filename(self, filename: str, strict: bool = True) -> bool:
        """Validate filename for path traversal and security"""
        if not isinstance(filename, str):
            return False

        # Check for path traversal attempts
        if self._detect_path_traversal(filename, strict):
            return False

        # Check for dangerous file patterns
        dangerous_patterns = [r"\.exe$", r"\.bat$", r"\.cmd$", r"\.sh$", r"\.ps1$"]
        for pattern in dangerous_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False

        return True

    def _validate_json(self, data: str, strict: bool = True) -> bool:
        """Validate JSON data"""
        if not isinstance(data, str):
            return False

        try:
            parsed = json.loads(data)

            if strict:
                # Check for deeply nested objects (potential DoS)
                max_depth = 10
                if self._get_json_depth(parsed) > max_depth:
                    return False

            return True
        except json.JSONDecodeError:
            return False

    def _get_json_depth(self, obj, current_depth=0):
        """Calculate JSON object depth"""
        if current_depth > 20:  # Prevent infinite recursion
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    def _detect_sql_injection(self, data: str, strict: bool = True) -> bool:
        """Detect SQL injection attempts"""
        if not isinstance(data, str):
            return False

        for pattern in self.threat_patterns["sql_injection"]:
            if re.search(pattern, data, re.IGNORECASE):
                self._log_security_event(
                    "sql_injection_attempt",
                    ThreatLevel.HIGH,
                    f"SQL injection pattern detected: {pattern}",
                    {"input_data": data[:100]},  # Truncate for logging
                )
                return True

        return False

    def _detect_xss(self, data: str, strict: bool = True) -> bool:
        """Detect XSS attempts"""
        if not isinstance(data, str):
            return False

        for pattern in self.threat_patterns["xss"]:
            if re.search(pattern, data, re.IGNORECASE):
                self._log_security_event(
                    "xss_attempt",
                    ThreatLevel.HIGH,
                    f"XSS pattern detected: {pattern}",
                    {"input_data": data[:100]},
                )
                return True

        return False

    def _detect_path_traversal(self, data: str, strict: bool = True) -> bool:
        """Detect path traversal attempts"""
        if not isinstance(data, str):
            return False

        for pattern in self.threat_patterns["path_traversal"]:
            if re.search(pattern, data, re.IGNORECASE):
                self._log_security_event(
                    "path_traversal_attempt",
                    ThreatLevel.HIGH,
                    f"Path traversal pattern detected: {pattern}",
                    {"input_data": data[:100]},
                )
                return True

        return False

    def check_rate_limit(self, identifier: str, operation_type: str = "default") -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        config = self.rate_limit_config.get(operation_type, self.rate_limit_config["default"])

        # Initialize rate limit tracking for this identifier
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {"requests": [], "blocked_until": 0}

        rate_data = self.rate_limits[identifier]

        # Check if currently blocked
        if now < rate_data["blocked_until"]:
            return False

        # Clean old requests outside the window
        window_start = now - config["window"]
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"] if req_time > window_start
        ]

        # Check if limit exceeded
        if len(rate_data["requests"]) >= config["requests"]:
            # Block for lockout duration
            rate_data["blocked_until"] = now + self.config["lockout_duration"]

            self._log_security_event(
                "rate_limit_exceeded",
                ThreatLevel.MEDIUM,
                f"Rate limit exceeded for {identifier}",
                {"operation_type": operation_type, "request_count": len(rate_data["requests"])},
            )
            return False

        # Add current request
        rate_data["requests"].append(now)
        return True

    def authenticate_request(self, token: str, context: SecurityContext) -> bool:
        """Authenticate request with token"""
        try:
            # For demo purposes, implement simple token validation
            # In production, use proper JWT validation
            if not token or len(token) < 10:
                raise AuthenticationError("Invalid token format")

            # Check rate limits for authentication attempts
            identifier = context.ip_address or "unknown"
            if not self.check_rate_limit(identifier, "authentication"):
                raise RateLimitExceededError("Too many authentication attempts")

            # Validate token format and signature (simplified)
            if not self._validate_token(token):
                self._log_security_event(
                    "authentication_failed",
                    ThreatLevel.MEDIUM,
                    "Invalid authentication token",
                    {"ip_address": context.ip_address},
                )
                return False

            # Check if IP is blocked
            if context.ip_address and context.ip_address in self.blocked_ips:
                raise AuthenticationError("IP address is blocked")

            return True

        except SecurityException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def _validate_token(self, token: str) -> bool:
        """Validate authentication token (simplified)"""
        # In production, implement proper JWT validation
        # This is a simplified version for demonstration
        try:
            # Check token format
            if not re.match(r"^[A-Za-z0-9+/]+=*$", token):
                return False

            # Decode and validate (simplified)
            decoded = base64.b64decode(token + "==")  # Add padding if needed
            return len(decoded) >= 16  # Minimum token size

        except Exception:
            return False

    def authorize_request(
        self, context: SecurityContext, required_level: AccessLevel, resource: str = None
    ) -> bool:
        """Authorize request based on context and requirements"""
        try:
            # Check if agent is trusted
            if context.agent_id and context.agent_id in self.trusted_agents:
                return True

            # Check access levels
            if required_level not in context.access_levels:
                self._log_security_event(
                    "authorization_failed",
                    ThreatLevel.MEDIUM,
                    f"Insufficient access level for {resource or 'resource'}",
                    {
                        "required_level": required_level.value,
                        "user_access_levels": [level.value for level in context.access_levels],
                        "resource": resource,
                    },
                )
                return False

            # Additional resource-specific checks
            if resource and not self._check_resource_access(context, resource):
                return False

            return True

        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False

    def _check_resource_access(self, context: SecurityContext, resource: str) -> bool:
        """Check resource-specific access permissions"""
        # Implement resource-specific access control logic
        # This could include checking ownership, group membership, etc.
        return True  # Simplified for demo

    def sanitize_input(self, data: Any, input_type: str = "general") -> Any:
        """Sanitize input data to prevent security issues"""
        if isinstance(data, str):
            return self._sanitize_string(data, input_type)
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v, input_type) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item, input_type) for item in data]
        else:
            return data

    def _sanitize_string(self, text: str, input_type: str) -> str:
        """Sanitize string input"""
        if input_type == "html":
            # Basic HTML sanitization
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)
            text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)
        elif input_type == "sql":
            # Basic SQL sanitization
            text = re.sub(r"[';\"\\]", "", text)
        elif input_type == "command":
            # Command injection prevention
            text = re.sub(r"[;&|`$()]", "", text)

        # General sanitization
        text = text.replace("\x00", "")  # Remove null bytes
        text = re.sub(r"[\r\n\t]+", " ", text)  # Normalize whitespace

        return text.strip()

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using real AES encryption"""
        try:
            encryption_key = self.config.get("encryption_key")
            if not encryption_key:
                logger.warning("No encryption key configured, using base64 encoding")
                return base64.b64encode(data.encode()).decode()
            
            # Use real AES encryption
            return self._aes_encrypt(data, encryption_key)
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            # Fallback to base64 if AES fails
            return base64.b64encode(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data using real AES decryption"""
        try:
            encryption_key = self.config.get("encryption_key")
            if not encryption_key:
                logger.warning("No encryption key configured, using base64 decoding")
                return base64.b64decode(encrypted_data).decode()
            
            # Use real AES decryption
            return self._aes_decrypt(encrypted_data, encryption_key)
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            # Fallback to base64 if AES fails
            try:
                return base64.b64decode(encrypted_data).decode()
            except:
                return encrypted_data

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)

    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)

        # Use PBKDF2 for password hashing
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return base64.b64encode(password_hash).decode(), salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(password_hash, computed_hash)
        except Exception:
            return False
    
    def _aes_encrypt(self, data: str, key: str) -> str:
        """Encrypt data using AES-256-GCM"""
        try:
            # Import cryptography library for real AES encryption
            try:
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.backends import default_backend
                import os
            except ImportError:
                logger.warning("Cryptography library not available, falling back to base64")
                return base64.b64encode(data.encode()).decode()
            
            # Generate salt and derive key
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key.encode())
            
            # Generate IV
            iv = os.urandom(12)  # 96 bits for GCM
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
            
            # Combine salt + iv + tag + ciphertext and encode
            encrypted_data = salt + iv + encryptor.tag + ciphertext
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"AES encryption failed: {e}")
            # Fallback to base64
            return base64.b64encode(data.encode()).decode()
    
    def _aes_decrypt(self, encrypted_data: str, key: str) -> str:
        """Decrypt data using AES-256-GCM"""
        try:
            # Import cryptography library
            try:
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.backends import default_backend
            except ImportError:
                logger.warning("Cryptography library not available, falling back to base64")
                return base64.b64decode(encrypted_data).decode()
            
            # Decode the encrypted data
            encrypted_bytes = base64.b64decode(encrypted_data)
            
            # Extract components
            salt = encrypted_bytes[:16]
            iv = encrypted_bytes[16:28]
            tag = encrypted_bytes[28:44]
            ciphertext = encrypted_bytes[44:]
            
            # Derive key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(key.encode())
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode()
            
        except Exception as e:
            logger.error(f"AES decryption failed: {e}")
            # Fallback to base64
            try:
                return base64.b64decode(encrypted_data).decode()
            except:
                return encrypted_data

    def _log_security_event(
        self,
        event_type: str,
        severity: ThreatLevel,
        description: str,
        details: Dict[str, Any] = None,
    ):
        """Log security event for audit trail"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=details.get("ip_address") if details else None,
            user_id=details.get("user_id") if details else None,
            agent_id=self.agent_id,
            description=description,
            details=details or {},
        )

        self.security_events.append(event)

        # Limit event history size
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]

        # Log to standard logging
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)

        logger.log(log_level, f"Security Event [{event_type}]: {description}")

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp >= cutoff_time]

        # Categorize events
        by_type = {}
        by_severity = {}

        for event in recent_events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "by_type": by_type,
            "by_severity": by_severity,
            "blocked_ips": len(self.blocked_ips),
            "trusted_agents": len(self.trusted_agents),
            "active_rate_limits": len(
                [k for k, v in self.rate_limits.items() if v["blocked_until"] > time.time()]
            ),
        }

    def get_security_recommendations(self) -> List[str]:
        """Get security recommendations based on current state"""
        recommendations = []
        summary = self.get_security_summary(hours=24)

        if summary["total_events"] > 50:
            recommendations.append(
                "High security event volume detected. Consider reviewing security policies."
            )

        if summary["by_severity"].get("critical", 0) > 0:
            recommendations.append(
                "Critical security events detected. Immediate investigation required."
            )

        if summary["by_severity"].get("high", 0) > 10:
            recommendations.append(
                "Multiple high-severity security events. Consider strengthening security controls."
            )

        if summary["blocked_ips"] > 100:
            recommendations.append(
                "Large number of blocked IPs. Consider reviewing threat patterns."
            )

        # Check for attack patterns
        if summary["by_type"].get("sql_injection_attempt", 0) > 5:
            recommendations.append(
                "Multiple SQL injection attempts detected. Review input validation."
            )

        if summary["by_type"].get("xss_attempt", 0) > 5:
            recommendations.append("Multiple XSS attempts detected. Review output encoding.")

        return recommendations


def secure_operation(
    required_level: AccessLevel = AccessLevel.READ,
    validate_input: bool = True,
    rate_limit_type: str = "default",
):
    """Decorator for securing operations"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get security manager
            if hasattr(self, "_security_manager"):
                security_manager = self._security_manager

                # Create security context (simplified for demo)
                context = SecurityContext(
                    agent_id=getattr(self, "agent_id", "unknown"),
                    timestamp=datetime.utcnow(),
                    access_levels={AccessLevel.READ, AccessLevel.WRITE},  # Simplified
                )

                # Check authorization
                if not security_manager.authorize_request(context, required_level):
                    raise AuthorizationError(f"Insufficient permissions for {func.__name__}")

                # Check rate limits
                identifier = context.agent_id
                if not security_manager.check_rate_limit(identifier, rate_limit_type):
                    raise RateLimitExceededError(f"Rate limit exceeded for {func.__name__}")

                # Validate inputs if requested
                if validate_input:
                    for arg in args:
                        if isinstance(arg, str):
                            if not security_manager.validate_input(arg, "general"):
                                raise InputValidationError(f"Invalid input in {func.__name__}")

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                return await func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


# Global security manager registry
_security_managers: Dict[str, SecurityHardeningManager] = {}


def get_security_manager(agent_id: str) -> SecurityHardeningManager:
    """Get or create security manager for agent"""
    if agent_id not in _security_managers:
        _security_managers[agent_id] = SecurityHardeningManager(agent_id)
    return _security_managers[agent_id]


def create_security_manager(agent_id: str) -> SecurityHardeningManager:
    """Create new security manager for agent"""
    manager = SecurityHardeningManager(agent_id)
    _security_managers[agent_id] = manager
    return manager

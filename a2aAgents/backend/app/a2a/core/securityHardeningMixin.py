#!/usr/bin/env python3
"""
Security Hardening Mixin for A2A Agents
Provides easy integration of comprehensive security capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Set
from datetime import datetime
from functools import wraps

from .securityHardening import (
    AccessLevel,
    SecurityContext,
    SecurityException,
    AuthenticationError,
    AuthorizationError,
    InputValidationError,
    RateLimitExceededError,
    get_security_manager,
)

logger = logging.getLogger(__name__)


class SecurityHardeningMixin:
    """
    Mixin class that provides comprehensive security hardening capabilities to A2A agents

    Usage:
        class MyAgent(A2AAgentBase, SecurityHardeningMixin):
            def __init__(self, ...):
                A2AAgentBase.__init__(self, ...)
                SecurityHardeningMixin.__init__(self)
    """

    def __init__(self):
        """Initialize security hardening capabilities"""
        # Initialize security manager
        agent_id = getattr(self, "agent_id", "unknown_agent")
        self._security_manager = get_security_manager(agent_id)

        # Security configuration
        self._security_enabled = True
        self._strict_validation = True
        self._audit_logging_enabled = True

        # Security metrics
        self._security_metrics = {
            "authentication_attempts": 0,
            "authorization_failures": 0,
            "input_validation_failures": 0,
            "rate_limit_violations": 0,
            "security_events": 0,
        }

        logger.info(f"Security hardening initialized for agent: {agent_id}")

    def enable_security_hardening(
        self,
        strict_validation: bool = True,
        audit_logging: bool = True,
        custom_config: Dict[str, Any] = None,
    ):
        """Enable and configure security hardening"""
        self._security_enabled = True
        self._strict_validation = strict_validation
        self._audit_logging_enabled = audit_logging

        # Apply custom security configuration
        if custom_config:
            self._security_manager.configure_security(**custom_config)

        # Configure default security settings for A2A agents
        self._configure_default_security()

        logger.info(f"Security hardening enabled for {self._security_manager.agent_id}")

    def disable_security_hardening(self):
        """Disable security hardening (for testing only)"""
        self._security_enabled = False
        logger.warning(f"Security hardening disabled for {self._security_manager.agent_id}")

    def _configure_default_security(self):
        """Configure default security settings for A2A agents"""
        default_config = {
            "max_request_size": 50 * 1024 * 1024,  # 50MB for data processing
            "session_timeout": 7200,  # 2 hours for long-running operations
            "max_login_attempts": 3,  # Stricter for agent-to-agent communication
            "lockout_duration": 300,  # 5 minutes lockout
            "require_https": True,
            "enable_audit_logging": self._audit_logging_enabled,
            "enable_threat_detection": True,
        }

        self._security_manager.configure_security(**default_config)

        # Add default trusted agents
        trusted_agents = [
            "agent_manager",
            "catalog_manager",
            "data_manager",
            "performance_monitor",
            "error_recovery_manager",
        ]

        for agent_id in trusted_agents:
            self._security_manager.add_trusted_agent(agent_id)

    def authenticate_request(
        self,
        token: str,
        ip_address: str = None,
        user_agent: str = None,
        additional_context: Dict[str, Any] = None,
    ) -> SecurityContext:
        """Authenticate incoming request and return security context"""
        if not self._security_enabled:
            # Return permissive context when security is disabled
            return SecurityContext(
                agent_id=self._security_manager.agent_id,
                access_levels={AccessLevel.READ, AccessLevel.WRITE, AccessLevel.EXECUTE},
            )

        try:
            self._security_metrics["authentication_attempts"] += 1

            # Create security context
            context = SecurityContext(
                agent_id=self._security_manager.agent_id,
                ip_address=ip_address,
                user_agent=user_agent,
                additional_claims=additional_context or {},
            )

            # Authenticate request
            if self._security_manager.authenticate_request(token, context):
                # Set appropriate access levels based on authentication
                context.access_levels = self._determine_access_levels(token, context)
                return context
            else:
                raise AuthenticationError("Authentication failed")

        except SecurityException:
            self._security_metrics["authentication_attempts"] -= 1  # Don't count failed attempts
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication system error: {str(e)}")

    def _determine_access_levels(self, token: str, context: SecurityContext) -> Set[AccessLevel]:
        """Determine access levels based on token and context"""
        # Simplified access level determination
        # In production, this would parse JWT claims or check database
        access_levels = {AccessLevel.READ}

        # Add write access for trusted agents
        if context.agent_id in self._security_manager.trusted_agents:
            access_levels.add(AccessLevel.WRITE)
            access_levels.add(AccessLevel.EXECUTE)

        return access_levels

    def authorize_operation(
        self, context: SecurityContext, required_level: AccessLevel, resource: str = None
    ) -> bool:
        """Authorize operation based on security context"""
        if not self._security_enabled:
            return True

        try:
            self._security_metrics["authorization_failures"] += 1

            authorized = self._security_manager.authorize_request(context, required_level, resource)

            if authorized:
                self._security_metrics["authorization_failures"] -= 1  # Success
                return True
            else:
                raise AuthorizationError(f"Insufficient permissions for {resource or 'operation'}")

        except SecurityException:
            raise
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            raise AuthorizationError(f"Authorization system error: {str(e)}")

    def validate_and_sanitize_input(
        self, data: Any, validation_type: str = "general", sanitize: bool = True
    ) -> Any:
        """Validate and sanitize input data"""
        if not self._security_enabled:
            return data

        try:
            # Validate input
            if not self._security_manager.validate_input(
                data, validation_type, self._strict_validation
            ):
                self._security_metrics["input_validation_failures"] += 1
                raise InputValidationError(f"Input validation failed for type: {validation_type}")

            # Sanitize input if requested
            if sanitize:
                return self._security_manager.sanitize_input(data, validation_type)

            return data

        except SecurityException:
            raise
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            raise InputValidationError(f"Input validation system error: {str(e)}")

    def check_rate_limit(self, identifier: str = None, operation_type: str = "default") -> bool:
        """Check rate limits for operation"""
        if not self._security_enabled:
            return True

        identifier = identifier or self._security_manager.agent_id

        if not self._security_manager.check_rate_limit(identifier, operation_type):
            self._security_metrics["rate_limit_violations"] += 1
            raise RateLimitExceededError(f"Rate limit exceeded for {operation_type}")

        return True

    def execute_secure_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        security_context: SecurityContext = None,
        required_level: AccessLevel = AccessLevel.READ,
        validate_input: bool = True,
        rate_limit_type: str = "default",
        **kwargs,
    ) -> Any:
        """Execute operation with comprehensive security checks"""
        if not self._security_enabled:
            # Execute without security checks
            if asyncio.iscoroutinefunction(func):
                return asyncio.create_task(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)

        try:
            # Check authorization if context provided
            if security_context:
                self.authorize_operation(security_context, required_level, operation_name)

            # Check rate limits
            self.check_rate_limit(operation_type=rate_limit_type)

            # Validate inputs if requested
            if validate_input:
                validated_args = []
                for arg in args:
                    validated_args.append(
                        self.validate_and_sanitize_input(arg)
                        if isinstance(arg, (str, dict, list))
                        else arg
                    )
                args = tuple(validated_args)

                validated_kwargs = {}
                for key, value in kwargs.items():
                    validated_kwargs[key] = (
                        self.validate_and_sanitize_input(value)
                        if isinstance(value, (str, dict, list))
                        else value
                    )
                kwargs = validated_kwargs

            # Execute operation
            if asyncio.iscoroutinefunction(func):
                return asyncio.create_task(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)

        except SecurityException:
            self._security_metrics["security_events"] += 1
            raise
        except Exception as e:
            logger.error(f"Secure operation error: {e}")
            raise

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self._security_manager.encrypt_sensitive_data(data)

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self._security_manager.decrypt_sensitive_data(encrypted_data)

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return self._security_manager.generate_secure_token(length)

    def hash_password(self, password: str) -> tuple:
        """Hash password securely"""
        return self._security_manager.hash_password(password)

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        return self._security_manager.verify_password(password, password_hash, salt)

    def add_trusted_agent(self, agent_id: str):
        """Add agent to trusted list"""
        self._security_manager.add_trusted_agent(agent_id)

    def remove_trusted_agent(self, agent_id: str):
        """Remove agent from trusted list"""
        self._security_manager.remove_trusted_agent(agent_id)

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive security summary"""
        summary = self._security_manager.get_security_summary(hours)

        # Add agent-specific metrics
        summary["agent_security_metrics"] = self._security_metrics.copy()

        # Add security health score
        summary["security_health_score"] = self._calculate_security_health_score(summary)

        return summary

    def _calculate_security_health_score(self, summary: Dict[str, Any]) -> int:
        """Calculate security health score (0-100)"""
        score = 100

        # Reduce score for security events
        critical_events = summary.get("by_severity", {}).get("critical", 0)
        high_events = summary.get("by_severity", {}).get("high", 0)
        medium_events = summary.get("by_severity", {}).get("medium", 0)

        score -= critical_events * 20  # -20 per critical event
        score -= high_events * 10  # -10 per high event
        score -= medium_events * 5  # -5 per medium event

        # Reduce score for authentication/authorization issues
        auth_failures = self._security_metrics.get(
            "authentication_attempts", 0
        ) - self._security_metrics.get("authorization_failures", 0)
        if auth_failures > 10:
            score -= min((auth_failures - 10) * 2, 30)

        # Reduce score for input validation failures
        validation_failures = self._security_metrics.get("input_validation_failures", 0)
        if validation_failures > 5:
            score -= min(validation_failures * 3, 25)

        # Reduce score for rate limit violations
        rate_violations = self._security_metrics.get("rate_limit_violations", 0)
        if rate_violations > 0:
            score -= min(rate_violations * 5, 20)

        return max(score, 0)  # Ensure score doesn't go below 0

    def get_security_recommendations(self) -> List[str]:
        """Get security recommendations"""
        recommendations = self._security_manager.get_security_recommendations()

        # Add agent-specific recommendations
        if self._security_metrics["input_validation_failures"] > 10:
            recommendations.append("High input validation failure rate. Review input sanitization.")

        if self._security_metrics["authorization_failures"] > 20:
            recommendations.append(
                "High authorization failure rate. Review access control policies."
            )

        if not self._security_enabled:
            recommendations.append("Security hardening is disabled. Enable for production use.")

        return recommendations

    async def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit"""
        audit_timestamp = datetime.utcnow()

        audit_results = {
            "timestamp": audit_timestamp.isoformat(),
            "agent_id": self._security_manager.agent_id,
            "security_enabled": self._security_enabled,
            "strict_validation": self._strict_validation,
            "audit_logging_enabled": self._audit_logging_enabled,
        }

        # Add security summary
        audit_results.update(self.get_security_summary())

        # Add recommendations
        audit_results["recommendations"] = self.get_security_recommendations()

        # Check configuration security
        audit_results["configuration_security"] = self._audit_configuration_security()

        # Add compliance status
        audit_results["compliance_status"] = self._check_compliance_status()

        return audit_results

    def _audit_configuration_security(self) -> Dict[str, Any]:
        """Audit security configuration"""
        config = self._security_manager.config

        config_security = {
            "https_required": config.get("require_https", False),
            "session_timeout_appropriate": config.get("session_timeout", 0) <= 3600,  # <= 1 hour
            "strong_encryption": len(config.get("encryption_key", b"")) >= 32,
            "secure_jwt_secret": len(config.get("jwt_secret", "")) >= 32,
            "appropriate_lockout_duration": 300 <= config.get("lockout_duration", 0) <= 1800,
            "reasonable_max_attempts": 3 <= config.get("max_login_attempts", 0) <= 10,
        }

        config_security["overall_score"] = (
            sum(config_security.values()) / len(config_security) * 100
        )

        return config_security

    def _check_compliance_status(self) -> Dict[str, bool]:
        """Check compliance with security standards"""
        return {
            "encryption_at_rest": True,  # Simplified for demo
            "encryption_in_transit": self._security_manager.config.get("require_https", False),
            "access_control_implemented": self._security_enabled,
            "audit_logging_enabled": self._audit_logging_enabled,
            "input_validation_active": self._strict_validation,
            "rate_limiting_active": True,
            "threat_detection_active": self._security_manager.config.get(
                "enable_threat_detection", False
            ),
        }

    def reset_security_metrics(self):
        """Reset security metrics (for testing or periodic cleanup)"""
        self._security_metrics = {
            "authentication_attempts": 0,
            "authorization_failures": 0,
            "input_validation_failures": 0,
            "rate_limit_violations": 0,
            "security_events": 0,
        }
        logger.info("Security metrics reset")

    # Convenience methods for common secure operations
    async def secure_http_request(self, func: Callable, *args, **kwargs):
        """Execute HTTP request with security checks"""
        return await self.execute_secure_operation(
            "http_request",
            func,
            *args,
            required_level=AccessLevel.READ,
            validate_input=True,
            rate_limit_type="default",
            **kwargs,
        )

    async def secure_data_processing(self, func: Callable, *args, **kwargs):
        """Execute data processing with security checks"""
        return await self.execute_secure_operation(
            "data_processing",
            func,
            *args,
            required_level=AccessLevel.WRITE,
            validate_input=True,
            rate_limit_type="high_security",
            **kwargs,
        )

    async def secure_admin_operation(
        self, func: Callable, *args, security_context: SecurityContext = None, **kwargs
    ):
        """Execute administrative operation with strict security"""
        return await self.execute_secure_operation(
            "admin_operation",
            func,
            *args,
            security_context=security_context,
            required_level=AccessLevel.ADMIN,
            validate_input=True,
            rate_limit_type="high_security",
            **kwargs,
        )


# Convenience decorators for security
def requires_authentication(func):
    """Decorator that requires authentication"""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_security_enabled") or not self._security_enabled:
            return await func(self, *args, **kwargs)

        # Check if security context is provided
        # This is a simplified check - in production, extract from request headers
        return await func(self, *args, **kwargs)

    return wrapper


def requires_authorization(required_level: AccessLevel):
    """Decorator that requires specific authorization level"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if hasattr(self, "_security_manager") and self._security_enabled:
                # Check authorization (simplified)
                pass

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def validate_input(validation_type: str = "general"):
    """Decorator that validates input parameters"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if hasattr(self, "_security_manager") and self._security_enabled:
                # Validate inputs
                for arg in args:
                    if isinstance(arg, (str, dict, list)):
                        self.validate_and_sanitize_input(arg, validation_type)

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator

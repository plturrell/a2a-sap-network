"""
A2A Platform Custom Exceptions
Comprehensive exception hierarchy for enterprise-grade error handling
"""

from app.core.loggingConfig import get_logger, LogCategory
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime
import logging


logger = get_logger(__name__, LogCategory.AGENT)


class ErrorCategory(str, Enum):
    """Error categories for classification and handling"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    NETWORK = "network"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    AGENT_COMMUNICATION = "agent_communication"


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base Exception Classes

class A2ABaseException(Exception):
    """Base exception for all A2A platform errors"""

    def __init__(
        self,
        message: str,
        error_code: str = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

        # Log the exception
        log_level = self._get_log_level()
        logger._log(log_level, f"A2A Exception: {self.error_code} - {message}", extra={
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context
        })

    def _get_log_level(self) -> int:
        """Get logging level based on severity"""
        severity_to_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_to_level.get(self.severity, logging.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


class A2AConfigurationError(A2ABaseException):
    """Configuration-related errors"""

    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context={"config_key": config_key} if config_key else None,
            **kwargs
        )


class A2AValidationError(A2ABaseException):
    """Data validation errors"""

    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context={"field_errors": field_errors} if field_errors else None,
            **kwargs
        )


# Authentication & Authorization Exceptions

class A2AAuthenticationError(A2ABaseException):
    """Authentication failures"""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class A2AAuthorizationError(A2ABaseException):
    """Authorization failures"""

    def __init__(self, message: str = "Access denied", required_permissions: List[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            context={"required_permissions": required_permissions} if required_permissions else None,
            **kwargs
        )


class A2ATokenExpiredError(A2AAuthenticationError):
    """JWT token expiration"""

    def __init__(self, message: str = "Authentication token has expired", **kwargs):
        super().__init__(message, error_code="TOKEN_EXPIRED", **kwargs)


class A2AInvalidTokenError(A2AAuthenticationError):
    """Invalid JWT token"""

    def __init__(self, message: str = "Invalid authentication token", **kwargs):
        super().__init__(message, error_code="INVALID_TOKEN", **kwargs)


# Database Exceptions

class A2ADatabaseError(A2ABaseException):
    """Database operation errors"""

    def __init__(self, message: str, operation: str = None, table: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context={"operation": operation, "table": table} if operation or table else None,
            **kwargs
        )


class A2AConnectionError(A2ADatabaseError):
    """Database connection errors"""

    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(message, error_code="DB_CONNECTION_FAILED", **kwargs)


class A2AConstraintViolationError(A2ADatabaseError):
    """Database constraint violations"""

    def __init__(self, message: str, constraint_name: str = None, **kwargs):
        super().__init__(
            message,
            error_code="CONSTRAINT_VIOLATION",
            context={"constraint_name": constraint_name} if constraint_name else None,
            **kwargs
        )


# Network & External Service Exceptions

class A2ANetworkError(A2ABaseException):
    """Network communication errors"""

    def __init__(self, message: str, endpoint: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            context={"endpoint": endpoint} if endpoint else None,
            **kwargs
        )


class A2AExternalServiceError(A2ABaseException):
    """External service integration errors"""

    def __init__(
        self,
        message: str,
        service_name: str = None,
        service_response: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            context={
                "service_name": service_name,
                "service_response": service_response
            } if service_name or service_response else None,
            **kwargs
        )


class A2ATimeoutError(A2ANetworkError):
    """Request timeout errors"""

    def __init__(self, message: str = "Request timeout", timeout_duration: float = None, **kwargs):
        super().__init__(
            message,
            error_code="REQUEST_TIMEOUT",
            context={"timeout_duration": timeout_duration} if timeout_duration else None,
            **kwargs
        )


class A2ARateLimitError(A2AExternalServiceError):
    """Rate limiting errors"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="RATE_LIMIT_EXCEEDED",
            context={"retry_after": retry_after} if retry_after else None,
            **kwargs
        )


# Agent Communication Exceptions

class A2AAgentError(A2ABaseException):
    """Agent-specific errors"""

    def __init__(self, message: str, agent_id: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AGENT_COMMUNICATION,
            severity=ErrorSeverity.MEDIUM,
            context={"agent_id": agent_id} if agent_id else None,
            **kwargs
        )


class A2AAgentUnavailableError(A2AAgentError):
    """Agent unavailable for processing"""

    def __init__(self, message: str = "Agent is unavailable", agent_id: str = None, **kwargs):
        super().__init__(
            message,
            error_code="AGENT_UNAVAILABLE",
            agent_id=agent_id,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class A2AAgentTimeoutError(A2AAgentError):
    """Agent processing timeout"""

    def __init__(
        self,
        message: str = "Agent processing timeout",
        agent_id: str = None,
        timeout_duration: float = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="AGENT_TIMEOUT",
            agent_id=agent_id,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if timeout_duration:
            self.context["timeout_duration"] = timeout_duration


class A2AAgentCommunicationError(A2AAgentError):
    """Inter-agent communication failures"""

    def __init__(
        self,
        message: str,
        source_agent: str = None,
        target_agent: str = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="AGENT_COMMUNICATION_FAILED",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if source_agent or target_agent:
            self.context.update({
                "source_agent": source_agent,
                "target_agent": target_agent
            })


# Business Logic Exceptions

class A2ABusinessLogicError(A2ABaseException):
    """Business rule violations"""

    def __init__(self, message: str, rule_name: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            context={"rule_name": rule_name} if rule_name else None,
            **kwargs
        )


class A2AResourceNotFoundError(A2ABusinessLogicError):
    """Resource not found"""

    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, **kwargs):
        super().__init__(
            message,
            error_code="RESOURCE_NOT_FOUND",
            context={
                "resource_type": resource_type,
                "resource_id": resource_id
            } if resource_type or resource_id else None,
            **kwargs
        )


class A2AResourceConflictError(A2ABusinessLogicError):
    """Resource conflict (duplicate, concurrent modification, etc.)"""

    def __init__(self, message: str, conflict_type: str = None, **kwargs):
        super().__init__(
            message,
            error_code="RESOURCE_CONFLICT",
            context={"conflict_type": conflict_type} if conflict_type else None,
            **kwargs
        )


class A2AConcurrencyError(A2AResourceConflictError):
    """Concurrency-related errors (e.g., circuit breaker open)"""

    def __init__(self, message: str, operation_id: str = None, **kwargs):
        super().__init__(
            message,
            conflict_type="concurrency",
            error_code="CONCURRENCY_ERROR",
            context={"operation_id": operation_id} if operation_id else None,
            **kwargs
        )


class A2AResourceExhaustionError(A2AResourceConflictError):
    """Resource exhaustion errors (e.g., connection pool empty)"""

    def __init__(self, message: str, resource_name: str = None, **kwargs):
        super().__init__(
            message,
            conflict_type="resource_exhaustion",
            error_code="RESOURCE_EXHAUSTION",
            context={"resource_name": resource_name} if resource_name else None,
            **kwargs
        )


# SAP Integration Exceptions

class A2ASAPIntegrationError(A2AExternalServiceError):
    """SAP system integration errors"""

    def __init__(
        self,
        message: str,
        sap_system: str = None,
        sap_error_code: str = None,
        **kwargs
    ):
        super().__init__(
            message,
            service_name="SAP",
            error_code="SAP_INTEGRATION_ERROR",
            **kwargs
        )
        if sap_system or sap_error_code:
            self.context.update({
                "sap_system": sap_system,
                "sap_error_code": sap_error_code
            })


class A2AHANAError(A2ASAPIntegrationError):
    """SAP HANA specific errors"""

    def __init__(self, message: str, hana_error_code: str = None, **kwargs):
        super().__init__(
            message,
            sap_system="HANA",
            sap_error_code=hana_error_code,
            error_code="HANA_ERROR",
            **kwargs
        )


class A2ABTPError(A2ASAPIntegrationError):
    """SAP BTP specific errors"""

    def __init__(self, message: str, btp_service: str = None, **kwargs):
        super().__init__(
            message,
            sap_system="BTP",
            error_code="BTP_ERROR",
            **kwargs
        )
        if btp_service:
            self.context["btp_service"] = btp_service


# Utility Functions

def create_error_response(
    exception: A2ABaseException,
    include_context: bool = False,
    include_stack_trace: bool = False
) -> Dict[str, Any]:
    """Create standardized error response from exception"""
    response = exception.to_dict()

    if not include_context:
        response.pop("context", None)

    if include_stack_trace and exception.original_error:
        import traceback
        response["stack_trace"] = traceback.format_exception(
            type(exception.original_error),
            exception.original_error,
            exception.original_error.__traceback__
        )

    return response


def handle_exception_chain(exception: Exception) -> A2ABaseException:
    """Convert generic exceptions to A2A exceptions with proper categorization"""

    if isinstance(exception, A2ABaseException):
        return exception

    # Map common Python exceptions to A2A exceptions
    exception_mapping = {
        ConnectionError: A2ANetworkError,
        TimeoutError: A2ATimeoutError,
        ValueError: A2AValidationError,
        KeyError: A2AConfigurationError,
        FileNotFoundError: A2AResourceNotFoundError,
        PermissionError: A2AAuthorizationError,
    }

    exception_class = exception_mapping.get(type(exception), A2ABaseException)

    return exception_class(
        message=str(exception),
        original_error=exception,
        error_code=f"WRAPPED_{type(exception).__name__.upper()}"
    )


# Exception Registry for API Documentation
EXCEPTION_REGISTRY = {
    # Authentication & Authorization
    "AUTH_001": A2AAuthenticationError,
    "AUTH_002": A2AAuthorizationError,
    "AUTH_003": A2ATokenExpiredError,
    "AUTH_004": A2AInvalidTokenError,

    # Database
    "DB_001": A2ADatabaseError,
    "DB_002": A2AConnectionError,
    "DB_003": A2AConstraintViolationError,

    # Network & External Services
    "NET_001": A2ANetworkError,
    "NET_002": A2ATimeoutError,
    "EXT_001": A2AExternalServiceError,
    "EXT_002": A2ARateLimitError,

    # Agents
    "AGT_001": A2AAgentError,
    "AGT_002": A2AAgentUnavailableError,
    "AGT_003": A2AAgentTimeoutError,
    "AGT_004": A2AAgentCommunicationError,

    # Business Logic
    "BIZ_001": A2ABusinessLogicError,
    "BIZ_002": A2AResourceNotFoundError,
    "BIZ_003": A2AResourceConflictError,

    # SAP Integration
    "SAP_001": A2ASAPIntegrationError,
    "SAP_002": A2AHANAError,
    "SAP_003": A2ABTPError,

    # System
    "SYS_001": A2AConfigurationError,
    "SYS_002": A2AValidationError,
}
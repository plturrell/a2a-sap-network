"""
import time
Secure Error Handling Framework
Prevents information disclosure while maintaining useful logging
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
import os

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .exceptions import ErrorSeverity


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ErrorContext:
    """Context information for errors"""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_id: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class SecureError(Exception):
    """Base secure error class with controlled information disclosure"""

    def __init__(self,
                 message: str,
                 user_message: Optional[str] = None,
                 error_code: Optional[str] = None,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)

        self.message = message  # Internal message (logged)
        self.user_message = user_message or "An error occurred. Please try again."  # User-facing message
        self.error_code = error_code or f"ERR_{uuid.uuid4().hex[:8].upper()}"
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.timestamp = datetime.utcnow()


class AuthenticationError(SecureError):
    """Authentication-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Authentication failed. Please check your credentials.",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AuthorizationError(SecureError):
    """Authorization-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Access denied. You don't have permission to perform this action.",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ValidationError(SecureError):
    """Input validation errors"""

    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        user_msg = f"Invalid input for field: {field_name}" if field_name else "Invalid input provided."
        super().__init__(
            message=message,
            user_message=user_msg,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class BusinessLogicError(SecureError):
    """Business logic errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Operation cannot be completed due to business rules.",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ExternalServiceError(SecureError):
    """External service communication errors"""

    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        user_msg = f"Service temporarily unavailable: {service_name}" if service_name else "External service unavailable."
        super().__init__(
            message=message,
            user_message=user_msg,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class SecurityError(SecureError):
    """Security-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Security violation detected. This incident has been logged.",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class SystemError(SecureError):
    """System-level errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            user_message="Internal system error. Please contact support if this persists.",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: bool = True
    message: str
    error_code: str
    timestamp: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SecureErrorHandler:
    """Secure error handler that prevents information disclosure"""

    def __init__(self,
                 logger_name: str = "a2a.error_handler",
                 include_stack_traces: bool = False,
                 mask_sensitive_data: bool = True):
        self.logger = logging.getLogger(logger_name)
        self.include_stack_traces = include_stack_traces or os.getenv("DEBUG", "false").lower() == "true"
        self.mask_sensitive_data = mask_sensitive_data

        # Sensitive field patterns to mask
        self.sensitive_patterns = {
            'password', 'secret', 'key', 'token', 'credential',
            'auth', 'session', 'cookie', 'private'
        }

    def handle_error(self,
                    error: Exception,
                    request: Optional[Request] = None,
                    context: Optional[ErrorContext] = None) -> ErrorResponse:
        """
        Handle error with secure logging and user-safe response

        Args:
            error: Exception that occurred
            request: FastAPI request object (optional)
            context: Additional context information

        Returns:
            Sanitized error response for user
        """
        # Generate unique error ID for correlation
        error_id = str(uuid.uuid4())

        # Extract request information if available
        request_info = {}
        if request:
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "remote_addr": request.client.host if request.client else "unknown"
            }

        # Handle SecureError instances
        if isinstance(error, SecureError):
            self._log_secure_error(error, error_id, request_info, context)

            return ErrorResponse(
                message=error.user_message,
                error_code=error.error_code,
                timestamp=error.timestamp.isoformat(),
                request_id=context.request_id if context else None
            )

        # Handle standard HTTP exceptions
        elif isinstance(error, HTTPException):
            self._log_http_exception(error, error_id, request_info, context)

            # Sanitize HTTP error messages
            safe_message = self._sanitize_http_message(error.detail, error.status_code)

            return ErrorResponse(
                message=safe_message,
                error_code=f"HTTP_{error.status_code}",
                timestamp=datetime.utcnow().isoformat(),
                request_id=context.request_id if context else None
            )

        # Handle unexpected exceptions
        else:
            self._log_unexpected_error(error, error_id, request_info, context)

            return ErrorResponse(
                message="An unexpected error occurred. Please contact support.",
                error_code=f"SYS_{error_id[:8]}",
                timestamp=datetime.utcnow().isoformat(),
                request_id=context.request_id if context else None
            )

    def _log_secure_error(self,
                         error: SecureError,
                         error_id: str,
                         request_info: Dict[str, Any],
                         context: Optional[ErrorContext]):
        """Log SecureError with appropriate detail level"""
        log_data = {
            "error_id": error_id,
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "timestamp": error.timestamp.isoformat(),
            **request_info
        }

        if context:
            if context.user_id:
                log_data["user_id"] = context.user_id
            if context.agent_id:
                log_data["agent_id"] = context.agent_id
            if context.operation:
                log_data["operation"] = context.operation
            if context.additional_data:
                log_data["context"] = self._sanitize_log_data(context.additional_data)

        if error.cause:
            log_data["cause"] = str(error.cause)

        # Log based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("CRITICAL ERROR", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("HIGH SEVERITY ERROR", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("MEDIUM SEVERITY ERROR", extra=log_data)
        else:
            self.logger.info("LOW SEVERITY ERROR", extra=log_data)

    def _log_http_exception(self,
                           error: HTTPException,
                           error_id: str,
                           request_info: Dict[str, Any],
                           context: Optional[ErrorContext]):
        """Log HTTP exception"""
        log_data = {
            "error_id": error_id,
            "status_code": error.status_code,
            "detail": str(error.detail),
            "timestamp": datetime.utcnow().isoformat(),
            **request_info
        }

        if context and context.user_id:
            log_data["user_id"] = context.user_id

        if error.status_code >= 500:
            self.logger.error("HTTP 5XX ERROR", extra=log_data)
        else:
            self.logger.warning("HTTP CLIENT ERROR", extra=log_data)

    def _log_unexpected_error(self,
                             error: Exception,
                             error_id: str,
                             request_info: Dict[str, Any],
                             context: Optional[ErrorContext]):
        """Log unexpected exception"""
        log_data = {
            "error_id": error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            **request_info
        }

        if context and context.user_id:
            log_data["user_id"] = context.user_id

        # Include stack trace if enabled
        if self.include_stack_traces:
            log_data["stack_trace"] = traceback.format_exc()

        self.logger.error("UNEXPECTED ERROR", extra=log_data)

    def _sanitize_http_message(self, detail: str, status_code: int) -> str:
        """Sanitize HTTP error messages to prevent information disclosure"""
        # Generic messages for common status codes
        generic_messages = {
            400: "Bad request. Please check your input.",
            401: "Authentication required.",
            403: "Access denied.",
            404: "Resource not found.",
            405: "Method not allowed.",
            409: "Conflict with current resource state.",
            422: "Invalid input data provided.",
            429: "Too many requests. Please try again later.",
            500: "Internal server error.",
            502: "Service temporarily unavailable.",
            503: "Service unavailable.",
            504: "Request timeout."
        }

        # Use generic message if available, otherwise sanitize
        if status_code in generic_messages:
            return generic_messages[status_code]

        # For other status codes, provide generic error message
        if status_code >= 500:
            return "Server error occurred."
        elif status_code >= 400:
            return "Client error occurred."
        else:
            return "An error occurred."

    def _sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize log data to prevent sensitive information logging"""
        if not self.mask_sensitive_data:
            return data

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if key contains sensitive patterns
            if any(pattern in key_lower for pattern in self.sensitive_patterns):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_log_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_log_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    @contextmanager
    def error_context(self, operation: str, **context_kwargs):
        """Context manager for error handling"""
        context = ErrorContext(
            operation=operation,
            additional_data=context_kwargs
        )

        try:
            yield context
        except Exception as e:
            # Re-raise with context
            if isinstance(e, SecureError):
                e.context = context
            raise


# Global error handler instance
_error_handler: Optional[SecureErrorHandler] = None

def get_error_handler() -> SecureErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = SecureErrorHandler()
    return _error_handler

def handle_error(error: Exception, **kwargs) -> ErrorResponse:
    """Convenience function to handle errors"""
    return get_error_handler().handle_error(error, **kwargs)


# FastAPI exception handler
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global FastAPI exception handler"""
    error_response = handle_error(exc, request=request)

    # Determine HTTP status code
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    elif isinstance(exc, AuthenticationError):
        status_code = 401
    elif isinstance(exc, AuthorizationError):
        status_code = 403
    elif isinstance(exc, ValidationError):
        status_code = 400
    elif isinstance(exc, BusinessLogicError):
        status_code = 422
    elif isinstance(exc, ExternalServiceError):
        status_code = 502
    elif isinstance(exc, SecurityError):
        status_code = 403
    else:
        status_code = 500

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


# Export main classes and functions
__all__ = [
    'SecureError',
    'AuthenticationError',
    'AuthorizationError',
    'ValidationError',
    'BusinessLogicError',
    'ExternalServiceError',
    'SecurityError',
    'SystemError',
    'ErrorResponse',
    'ErrorContext',
    'ErrorSeverity',
    'ErrorCategory',
    'SecureErrorHandler',
    'get_error_handler',
    'handle_error',
    'global_exception_handler'
]

"""
A2A Platform Logging Middleware
Automatic request tracking, correlation IDs, and performance logging for FastAPI
"""

import time
import uuid
from typing import Callable, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from app.core.loggingConfig import (
    get_logger,
    LogCategory,
    LoggingContext,
    correlation_id,
    request_id,
    user_id
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request logging and correlation tracking
    """

    def __init__(
        self,
        app: FastAPI,
        log_requests: bool = True,
        log_responses: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/favicon.ico"]

        self.logger = get_logger("a2a.api.middleware", LogCategory.API)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch function"""

        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Generate correlation and request IDs
        correlation_id_val = str(uuid.uuid4())
        request_id_val = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{correlation_id_val[:8]}"

        # Extract user ID from request if available
        user_id_val = self._extract_user_id(request)

        # Start request timing
        start_time = time.time()

        # Set up logging context
        async with LoggingContext(
            correlation_id_val=correlation_id_val,
            request_id_val=request_id_val,
            user_id_val=user_id_val
        ):
            # Log incoming request
            if self.log_requests:
                await self._log_request(request, request_id_val)

            # Process request
            response = await call_next(request)

            # Calculate request duration
            duration = time.time() - start_time

            # Log outgoing response
            if self.log_responses:
                self._log_response(request, response, duration, request_id_val)

            # Add correlation headers to response
            self._add_correlation_headers(response, correlation_id_val, request_id_val)

            return response

    async def _log_request(self, request: Request, request_id_val: str):
        """Log incoming HTTP request"""

        # Extract request details
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        content_type = request.headers.get("content-type", "unknown")
        content_length = request.headers.get("content-length", "0")

        # Sanitize headers to avoid logging sensitive information
        sanitized_headers = self._sanitize_headers(dict(request.headers))

        # Base request info
        request_info = {
            "request_id": request_id_val,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": content_type,
            "content_length": content_length,
            "headers": sanitized_headers
        }

        # Log request body if enabled and size is reasonable
        if (self.log_request_body and
            request.method in ["POST", "PUT", "PATCH"] and
            int(content_length or 0) <= self.max_body_size):

            try:
                body = await request.body()
                if body:
                    request_info["body"] = body.decode("utf-8", errors="replace")[:self.max_body_size]
            except Exception as e:
                request_info["body_error"] = str(e)

        self.logger.info(
            f"HTTP {request.method} {request.url.path}",
            category=LogCategory.API,
            **request_info
        )

    def _log_response(
        self,
        request: Request,
        response: StarletteResponse,
        duration: float,
        request_id_val: str
    ):
        """Log outgoing HTTP response"""

        # Sanitize response headers
        sanitized_response_headers = self._sanitize_headers(dict(response.headers))

        # Response details
        response_info = {
            "request_id": request_id_val,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 4),
            "content_type": response.headers.get("content-type", "unknown"),
            "content_length": response.headers.get("content-length", "unknown"),
            "response_headers": sanitized_response_headers
        }

        # Log response body if enabled and it's a reasonable size
        if (self.log_response_body and
            hasattr(response, 'body') and
            len(response.body or b"") <= self.max_body_size):
            try:
                response_info["body"] = response.body.decode("utf-8", errors="replace")[:self.max_body_size]
            except Exception as e:
                response_info["body_error"] = str(e)

        # Use appropriate log level based on status code
        if response.status_code >= 500:
            self.logger.error(
                f"HTTP {request.method} {request.url.path} -> {response.status_code}",
                category=LogCategory.API,
                **response_info
            )
        elif response.status_code >= 400:
            self.logger.warning(
                f"HTTP {request.method} {request.url.path} -> {response.status_code}",
                category=LogCategory.API,
                **response_info
            )
        else:
            self.logger.info(
                f"HTTP {request.method} {request.url.path} -> {response.status_code}",
                category=LogCategory.API,
                **response_info
            )

        # Log performance metrics for slow requests
        if duration > 1.0:  # Requests taking more than 1 second
            self.logger.log_performance(
                f"Slow API request: {request.method} {request.url.path}",
                duration,
                status_code=response.status_code,
                slow_request=True,
                performance_threshold=1.0
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""

        # Check common proxy headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP if multiple are listed
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to client host
        if hasattr(request, 'client') and request.client:
            return request.client.host

        return "unknown"

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request (JWT token, session, etc.)"""

        # Try to extract from Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                # This would need to be implemented with actual JWT decoding
                # For now, just extract a placeholder
                token = auth_header[7:]
                # In real implementation, decode JWT and extract user_id
                return f"user_from_token_{token[:8]}"
            except Exception:
                pass

        # Try to extract from session or cookies
        user_id_cookie = request.cookies.get("user_id")
        if user_id_cookie:
            return user_id_cookie

        # Could also check for API key headers, etc.
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key_user_{api_key[:8]}"

        return None

    def _sanitize_headers(self, headers: dict) -> dict:
        """Remove sensitive information from headers before logging"""
        sensitive_headers = {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
            "x-auth-token",
            "proxy-authorization",
            "x-access-token",
            "x-csrf-token",
            "x-secret-key"
        }

        sanitized = {}
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key in sensitive_headers:
                # Keep only first few characters for debugging
                sanitized[key] = f"[REDACTED:{value[:8]}...]" if len(value) > 8 else "[REDACTED]"
            elif any(sensitive_word in lower_key for sensitive_word in ["password", "secret", "key", "token"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value

        return sanitized

    def _add_correlation_headers(
        self,
        response: StarletteResponse,
        correlation_id_val: str,
        request_id_val: str
    ):
        """Add correlation tracking headers to response"""
        response.headers["X-Correlation-ID"] = correlation_id_val
        response.headers["X-Request-ID"] = request_id_val
        response.headers["X-API-Version"] = "v1"


class StructuredAPILogger:
    """Utility class for structured API-specific logging"""

    def __init__(self, name: str):
        self.logger = get_logger(name, LogCategory.API)

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        **context
    ):
        """Log API call with standardized structure"""
        self.logger.log_api_request(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration,
            user_id=user_id,
            **context
        )

    def log_auth_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        success: bool = True,
        **context
    ):
        """Log authentication/authorization events"""
        level = "info" if success else "warning"
        getattr(self.logger, level)(
            f"Auth event: {event_type}",
            category=LogCategory.SECURITY,
            event_type=event_type,
            user_id=user_id,
            success=success,
            **context
        )

    def log_business_operation(
        self,
        operation: str,
        entity_type: str,
        entity_id: str,
        success: bool = True,
        duration: Optional[float] = None,
        **context
    ):
        """Log business operation (CRUD, workflow, etc.)"""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            f"Business operation: {operation}",
            category=LogCategory.BUSINESS,
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            success=success,
            duration_seconds=duration,
            **context
        )

    def log_external_service_call(
        self,
        service_name: str,
        operation: str,
        success: bool = True,
        duration: Optional[float] = None,
        **context
    ):
        """Log external service interactions"""
        level = "info" if success else "warning"
        getattr(self.logger, level)(
            f"External service call: {service_name}.{operation}",
            category=LogCategory.INTEGRATION,
            service_name=service_name,
            operation=operation,
            success=success,
            duration_seconds=duration,
            **context
        )

    def log_data_access(
        self,
        table_name: str,
        operation: str,
        record_count: int,
        duration: Optional[float] = None,
        **context
    ):
        """Log database access operations"""
        self.logger.info(
            f"Data access: {operation} on {table_name}",
            category=LogCategory.DATABASE,
            table_name=table_name,
            operation=operation,
            record_count=record_count,
            duration_seconds=duration,
            **context
        )


def create_logging_middleware(
    app: FastAPI,
    log_requests: bool = True,
    log_responses: bool = True,
    log_bodies: bool = False,
    exclude_paths: Optional[list] = None
) -> LoggingMiddleware:
    """Factory function to create and configure logging middleware"""

    middleware = LoggingMiddleware(
        app=app,
        log_requests=log_requests,
        log_responses=log_responses,
        log_request_body=log_bodies,
        log_response_body=log_bodies,
        exclude_paths=exclude_paths or []
    )

    app.add_middleware(LoggingMiddleware)

    return middleware


# Export for easy import
__all__ = [
    "LoggingMiddleware",
    "StructuredAPILogger",
    "create_logging_middleware"
]
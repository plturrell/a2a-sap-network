"""
Production-Ready Logging Configuration for MCP Servers
Implements structured logging with correlation IDs
"""

import logging
import json
import sys
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with additional context"""
        # Base log structure
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data['correlation_id'] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Performance metrics
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        # Add service context
        if hasattr(record, 'service_name'):
            log_data['service_name'] = record.service_name
        if hasattr(record, 'service_port'):
            log_data['service_port'] = record.service_port

        return json.dumps(log_data)

class CorrelationFilter(logging.Filter):
    """Add correlation ID to all log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record"""
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.correlation_id = correlation_id
        return True

class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_operation(self, operation: str, duration_ms: float,
                     success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Log operation performance"""
        extra_fields = {
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'performance_metric': True
        }

        if metadata:
            extra_fields.update(metadata)

        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Operation {operation} completed in {duration_ms:.2f}ms",
            extra={'extra_fields': extra_fields}
        )

def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = True
) -> logging.Logger:
    """Configure production logging for MCP service"""

    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Configure formatter
    if enable_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(CorrelationFilter())
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationFilter())
        logger.addHandler(file_handler)

    # Add service context
    class ServiceContextFilter(logging.Filter):
        def filter(self, record):
            record.service_name = service_name
            return True

    for handler in logger.handlers:
        handler.addFilter(ServiceContextFilter())

    return logger

def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()

class LoggingMiddleware:
    """FastAPI middleware for request logging"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.perf_logger = PerformanceLogger(logger)

    async def __call__(self, request, call_next):
        """Log requests with correlation ID"""
        # Generate or extract correlation ID
        correlation_id = request.headers.get('X-Correlation-ID')
        correlation_id = set_correlation_id(correlation_id)

        # Log request
        start_time = time.time()
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={'extra_fields': {
                'method': request.method,
                'path': request.url.path,
                'client_host': request.client.host if request.client else None
            }}
        )

        try:
            # Process request
            response = await call_next(request)

            # Log response
            duration_ms = (time.time() - start_time) * 1000
            self.perf_logger.log_operation(
                f"{request.method} {request.url.path}",
                duration_ms,
                success=response.status_code < 400,
                metadata={
                    'status_code': response.status_code,
                    'method': request.method,
                    'path': request.url.path
                }
            )

            # Add correlation ID to response
            response.headers['X-Correlation-ID'] = correlation_id

            return response

        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                exc_info=True,
                extra={'extra_fields': {
                    'method': request.method,
                    'path': request.url.path,
                    'duration_ms': duration_ms,
                    'error': str(e)
                }}
            )
            raise

# Audit logger for security events
class SecurityAuditLogger:
    """Logger for security audit events"""

    def __init__(self, service_name: str):
        self.logger = logging.getLogger(f"{service_name}.security")
        self.logger.setLevel(logging.INFO)

    def log_auth_attempt(self, client_id: str, success: bool, method: str, reason: Optional[str] = None):
        """Log authentication attempt"""
        self.logger.info(
            f"Authentication {'succeeded' if success else 'failed'} for {client_id}",
            extra={'extra_fields': {
                'event_type': 'auth_attempt',
                'client_id': client_id,
                'success': success,
                'method': method,
                'reason': reason,
                'security_event': True
            }}
        )

    def log_rate_limit(self, client_id: str, limit: int, window: int):
        """Log rate limit exceeded"""
        self.logger.warning(
            f"Rate limit exceeded for {client_id}",
            extra={'extra_fields': {
                'event_type': 'rate_limit_exceeded',
                'client_id': client_id,
                'limit': limit,
                'window': window,
                'security_event': True
            }}
        )

    def log_permission_denied(self, client_id: str, resource: str, permission: str):
        """Log permission denied"""
        self.logger.warning(
            f"Permission denied for {client_id} on {resource}",
            extra={'extra_fields': {
                'event_type': 'permission_denied',
                'client_id': client_id,
                'resource': resource,
                'permission': permission,
                'security_event': True
            }}
        )

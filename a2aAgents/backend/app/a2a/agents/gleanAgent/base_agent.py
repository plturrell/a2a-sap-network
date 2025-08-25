"""
Base Agent Class with Enhanced Error Handling, Logging, and Performance Tracking
Provides foundation for Glean Agent with enterprise-grade capabilities
"""

import uuid
import time
import functools
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from enum import Enum
import logging
import json
from contextlib import contextmanager

# Type variable for generic decorator
T = TypeVar('T', bound=Callable[..., Any])


class ErrorCode(Enum):
    """A2A Platform Error Codes - Python implementation matching CAP error codes"""
    # Agent Errors (1000-1999)
    AGENT_NOT_FOUND = 'A2A_1001'
    AGENT_UNAVAILABLE = 'A2A_1002'
    AGENT_TIMEOUT = 'A2A_1003'
    AGENT_CONFIGURATION_ERROR = 'A2A_1004'
    AGENT_VALIDATION_FAILED = 'A2A_1005'
    AGENT_PROCESSING_ERROR = 'A2A_1006'
    AGENT_AUTHENTICATION_FAILED = 'A2A_1007'

    # Network Errors (2000-2999)
    NETWORK_CONNECTION_ERROR = 'A2A_2001'
    NETWORK_TIMEOUT = 'A2A_2002'
    NETWORK_SERVICE_UNAVAILABLE = 'A2A_2003'
    NETWORK_RATE_LIMIT_EXCEEDED = 'A2A_2004'

    # Data Errors (3000-3999)
    DATA_VALIDATION_ERROR = 'A2A_3001'
    DATA_NOT_FOUND = 'A2A_3002'
    DATA_CORRUPTION_DETECTED = 'A2A_3003'
    DATA_FORMAT_ERROR = 'A2A_3004'
    DATA_SIZE_LIMIT_EXCEEDED = 'A2A_3005'
    DATA_QUALITY_CHECK_FAILED = 'A2A_3006'
    DATA_ACCESS_DENIED = 'A2A_3007'

    # System Errors (9000-9999)
    INTERNAL_SERVER_ERROR = 'A2A_9001'
    SERVICE_UNAVAILABLE = 'A2A_9002'
    DATABASE_ERROR = 'A2A_9003'
    CONFIGURATION_ERROR = 'A2A_9004'
    RESOURCE_EXHAUSTED = 'A2A_9005'


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class ErrorCategory(Enum):
    """Error categories"""
    BUSINESS_LOGIC = 'business_logic'
    TECHNICAL = 'technical'
    SECURITY = 'security'
    PERFORMANCE = 'performance'
    INFRASTRUCTURE = 'infrastructure'


class A2AError(Exception):
    """Custom error class for A2A platform"""

    def __init__(self,
                 code: ErrorCode,
                 message: str,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        self.severity = self._determine_severity()
        self.category = self._determine_category()
        self.retryable = self._is_retryable()

    def _determine_severity(self) -> ErrorSeverity:
        """Determine error severity based on code"""
        critical_codes = [
            ErrorCode.DATA_CORRUPTION_DETECTED,
            ErrorCode.AGENT_AUTHENTICATION_FAILED
        ]
        high_codes = [
            ErrorCode.DATABASE_ERROR,
            ErrorCode.DATA_ACCESS_DENIED,
            ErrorCode.AGENT_PROCESSING_ERROR
        ]
        medium_codes = [
            ErrorCode.AGENT_UNAVAILABLE,
            ErrorCode.NETWORK_CONNECTION_ERROR,
            ErrorCode.DATA_VALIDATION_ERROR
        ]

        if self.code in critical_codes:
            return ErrorSeverity.CRITICAL
        elif self.code in high_codes:
            return ErrorSeverity.HIGH
        elif self.code in medium_codes:
            return ErrorSeverity.MEDIUM
        return ErrorSeverity.LOW

    def _determine_category(self) -> ErrorCategory:
        """Determine error category based on code"""
        code_value = self.code.value
        if code_value.startswith('A2A_1') or code_value.startswith('A2A_3'):
            return ErrorCategory.BUSINESS_LOGIC
        elif code_value.startswith('A2A_2') or code_value.startswith('A2A_9'):
            return ErrorCategory.TECHNICAL
        elif self.code == ErrorCode.AGENT_AUTHENTICATION_FAILED:
            return ErrorCategory.SECURITY
        return ErrorCategory.TECHNICAL

    def _is_retryable(self) -> bool:
        """Check if error is retryable"""
        retryable_codes = [
            ErrorCode.AGENT_TIMEOUT,
            ErrorCode.AGENT_UNAVAILABLE,
            ErrorCode.NETWORK_CONNECTION_ERROR,
            ErrorCode.NETWORK_TIMEOUT,
            ErrorCode.NETWORK_SERVICE_UNAVAILABLE,
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.RESOURCE_EXHAUSTED
        ]
        return self.code in retryable_codes

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary"""
        return {
            'code': self.code.value,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'retryable': self.retryable,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


class StructuredLogger:
    """Structured logger with correlation ID support"""

    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create console handler with JSON formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.context: Dict[str, Any] = {}

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Create child logger with additional context"""
        child = StructuredLogger(self.logger.name, self.logger.level)
        child.context = {**self.context, **kwargs}
        return child

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with context enrichment"""
        log_data = {
            **self.context,
            **kwargs,
            'timestamp': datetime.utcnow().isoformat(),
            'message': message
        }

        # Use appropriate log level
        log_method = getattr(self.logger, level)
        log_method(json.dumps(log_data))

    def debug(self, message: str, **kwargs):
        self._log('debug', message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log('warning', message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log('error', message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log('critical', message, **kwargs)


class PerformanceTracker:
    """Performance tracking for operations"""

    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, list] = {}

    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operation performance"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        self.operations[operation_id] = {
            'name': operation_name,
            'start_time': start_time,
            'metadata': metadata or {}
        }

        try:
            yield operation_id
        finally:
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to ms

            operation = self.operations.pop(operation_id)

            # Store metrics
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []

            self.metrics[operation_name].append({
                'duration': duration,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': operation['metadata']
            })

            # Clean up old metrics (keep last 1000)
            if len(self.metrics[operation_name]) > 1000:
                self.metrics[operation_name] = self.metrics[operation_name][-1000:]

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if operation_name:
            if operation_name not in self.metrics:
                return {}

            durations = [m['duration'] for m in self.metrics[operation_name]]
            if not durations:
                return {}

            return {
                'count': len(durations),
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'p50': self._percentile(durations, 50),
                'p95': self._percentile(durations, 95),
                'p99': self._percentile(durations, 99)
            }

        # Return all metrics
        result = {}
        for op_name in self.metrics:
            result[op_name] = self.get_metrics(op_name)
        return result

    def _percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]


class BaseAgent:
    """Base class for A2A agents with enhanced capabilities"""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.correlation_id: Optional[str] = None

        # Initialize logger
        self.logger = StructuredLogger(
            f'a2a.agent.{agent_id}',
            level=self.config.get('log_level', 'INFO')
        ).with_context(agent_id=agent_id)

        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()

        # Initialize error counts
        self.error_counts: Dict[str, int] = {}

        self.logger.info('Agent initialized', config=self.config)

    def with_correlation_id(self, correlation_id: str) -> 'BaseAgent':
        """Set correlation ID for tracking"""
        self.correlation_id = correlation_id
        self.logger = self.logger.with_context(correlation_id=correlation_id)
        return self

    def track_performance(self, operation_name: str):
        """Decorator for tracking method performance"""
        def decorator(func: T) -> T:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                metadata = {
                    'method': func.__name__,
                    'correlation_id': self.correlation_id
                }

                with self.performance_tracker.track_operation(operation_name, metadata) as op_id:
                    self.logger.debug(f'Starting operation: {operation_name}', operation_id=op_id)
                    try:
                        result = func(*args, **kwargs)
                        self.logger.debug(f'Completed operation: {operation_name}', operation_id=op_id)
                        return result
                    except Exception as e:
                        self.logger.error(
                            f'Operation failed: {operation_name}',
                            operation_id=op_id,
                            error=str(e)
                        )
                        raise
            return wrapper
        return decorator

    def handle_error(self, error: Union[Exception, A2AError], operation: str) -> A2AError:
        """Handle and transform errors"""
        if isinstance(error, A2AError):
            a2a_error = error
        else:
            # Transform to A2A error
            a2a_error = A2AError(
                code=ErrorCode.AGENT_PROCESSING_ERROR,
                message=str(error),
                details={
                    'operation': operation,
                    'agent_id': self.agent_id,
                    'correlation_id': self.correlation_id,
                    'traceback': traceback.format_exc()
                },
                cause=error
            )

        # Track error counts
        error_key = f"{a2a_error.code.value}:{operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Log error
        self.logger.error(
            'Agent error occurred',
            error_code=a2a_error.code.value,
            severity=a2a_error.severity.value,
            category=a2a_error.category.value,
            operation=operation,
            error_details=a2a_error.to_dict()
        )

        return a2a_error

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            'agent_id': self.agent_id,
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': self.performance_tracker.get_metrics(),
            'error_counts': self.error_counts,
            'config': {
                'log_level': self.config.get('log_level', 'INFO')
            }
        }

    def validate_input(self, data: Any, schema: Dict[str, Any]) -> None:
        """Validate input data against schema"""
        # This is a simplified validation - in production, use jsonschema or similar
        if not isinstance(data, dict):
            raise A2AError(
                code=ErrorCode.DATA_VALIDATION_ERROR,
                message='Input must be a dictionary',
                details={'received_type': type(data).__name__}
            )

        # Check required fields
        required_fields = schema.get('required', [])
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise A2AError(
                code=ErrorCode.DATA_VALIDATION_ERROR,
                message='Missing required fields',
                details={'missing_fields': missing_fields}
            )

    def __enter__(self):
        """Context manager entry"""
        self.correlation_id = str(uuid.uuid4())
        self.logger = self.logger.with_context(correlation_id=self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            self.handle_error(exc_val, 'context_exit')
        return False
#!/usr/bin/env python3
"""
Comprehensive Error Handling and Recovery Mechanisms for A2A Agents
Provides robust error handling, circuit breakers, retry logic, and recovery strategies
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import time
import traceback
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

def get_error_count(x):
    """Get error count for sorting error patterns"""
    return x[1]


from ...core.exceptions import ErrorSeverity


class ErrorCategory(str, Enum):
    """Error categories for classification"""

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    PROCESSING = "processing"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Available recovery actions"""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Context information for error tracking"""

    error_id: str
    timestamp: datetime
    agent_id: str
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_message: str
    stack_trace: str
    request_context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""

    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0  # 5 minutes
    fallback_function: Optional[Callable] = None
    escalation_threshold: int = 10


class CircuitState(str, Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation for error handling"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        success_threshold: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_change_time = datetime.utcnow()

        logger.info(f"Circuit breaker '{name}' initialized")

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.state_change_time = datetime.utcnow()
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        return (datetime.utcnow() - self.state_change_time).total_seconds() >= self.recovery_timeout

    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.state_change_time = datetime.utcnow()
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.utcnow()
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
            )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.state_change_time = datetime.utcnow()
            logger.warning(f"Circuit breaker '{self.name}' reopened during recovery attempt")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""


class ErrorRecoveryManager:
    """Comprehensive error handling and recovery management"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.global_error_count = 0
        self.error_patterns: Dict[str, int] = {}

        # Default recovery strategies
        self._initialize_default_strategies()

        logger.info(f"Error recovery manager initialized for agent: {agent_id}")

    def _initialize_default_strategies(self):
        """Initialize default recovery strategies"""
        self.recovery_strategies.update(
            {
                "network": RecoveryStrategy(
                    max_retries=3,
                    retry_delay=2.0,
                    exponential_backoff=True,
                    circuit_breaker_threshold=5,
                ),
                "external_service": RecoveryStrategy(
                    max_retries=5,
                    retry_delay=1.0,
                    exponential_backoff=True,
                    circuit_breaker_threshold=10,
                ),
                "processing": RecoveryStrategy(
                    max_retries=2,
                    retry_delay=0.5,
                    exponential_backoff=False,
                    circuit_breaker_threshold=3,
                ),
                "validation": RecoveryStrategy(
                    max_retries=1, retry_delay=0.1, exponential_backoff=False
                ),
                "default": RecoveryStrategy(),
            }
        )

    def register_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Register a new circuit breaker"""
        breaker = CircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = breaker
        return breaker

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)

    async def execute_with_recovery(
        self,
        operation_name: str,
        func: Callable,
        *args,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Any:
        """Execute function with comprehensive error handling and recovery"""
        context = context or {}
        strategy = self.recovery_strategies.get(category.value, self.recovery_strategies["default"])

        # Get or create circuit breaker for this operation
        breaker_name = f"{operation_name}_{category.value}"
        if breaker_name not in self.circuit_breakers:
            self.register_circuit_breaker(
                breaker_name, failure_threshold=strategy.circuit_breaker_threshold
            )

        circuit_breaker = self.circuit_breakers[breaker_name]
        last_exception = None

        for attempt in range(strategy.max_retries + 1):
            try:
                # Execute with circuit breaker protection
                result = await circuit_breaker.call(func, *args, **kwargs)

                # Log successful recovery if this was a retry
                if attempt > 0:
                    logger.info(f"Operation '{operation_name}' recovered after {attempt} attempts")

                return result

            except CircuitBreakerOpenError as e:
                logger.warning(f"Circuit breaker open for '{operation_name}': {e}")
                if strategy.fallback_function:
                    logger.info(f"Executing fallback for '{operation_name}'")
                    return await self._execute_fallback(strategy.fallback_function, *args, **kwargs)
                raise

            except Exception as e:
                last_exception = e

                # Create error context
                error_context = self._create_error_context(
                    operation_name, e, category, context, attempt
                )

                # Record error
                self._record_error(error_context)

                # Check if we should retry
                if attempt < strategy.max_retries:
                    retry_delay = self._calculate_retry_delay(strategy, attempt)
                    logger.warning(
                        f"Operation '{operation_name}' failed (attempt {attempt + 1}/{strategy.max_retries + 1}). "
                        f"Retrying in {retry_delay:.2f}s. Error: {str(e)}"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Operation '{operation_name}' failed after {strategy.max_retries + 1} attempts"
                    )

        # All retries exhausted, try fallback
        if strategy.fallback_function:
            logger.info(f"Executing fallback for failed operation '{operation_name}'")
            try:
                return await self._execute_fallback(strategy.fallback_function, *args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for '{operation_name}': {fallback_error}")

        # Record final failure
        self._handle_final_failure(operation_name, last_exception, category, context)
        raise last_exception

    def _create_error_context(
        self,
        operation: str,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any],
        attempt: int,
    ) -> ErrorContext:
        """Create error context for tracking"""
        error_id = self._generate_error_id(operation, error, attempt)

        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            agent_id=self.agent_id,
            operation=operation,
            severity=self._categorize_severity(error, category),
            category=category,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            request_context=context,
            recovery_attempts=attempt,
        )

    def _generate_error_id(self, operation: str, error: Exception, attempt: int) -> str:
        """Generate unique error ID"""
        content = f"{operation}_{str(error)}_{attempt}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _categorize_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Categorize error severity based on type and category"""
        error_type = type(error).__name__

        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION]:
            return ErrorSeverity.HIGH

        if error_type in ["ConnectionError", "TimeoutError", "PermissionError"]:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.MEDIUM

        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _record_error(self, error_context: ErrorContext):
        """Record error in history and update patterns"""
        self.error_history.append(error_context)
        self.global_error_count += 1

        # Track error patterns
        pattern_key = f"{error_context.category}_{type(error_context.error_message).__name__}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1

        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]  # Keep latest 500

        logger.debug(
            f"Recorded error {error_context.error_id} for operation {error_context.operation}"
        )

    def _calculate_retry_delay(self, strategy: RecoveryStrategy, attempt: int) -> float:
        """Calculate delay before retry"""
        if not strategy.exponential_backoff:
            return strategy.retry_delay

        delay = strategy.retry_delay * (strategy.backoff_multiplier**attempt)
        return min(delay, strategy.max_delay)

    async def _execute_fallback(self, fallback_func: Callable, *args, **kwargs) -> Any:
        """Execute fallback function"""
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback function failed: {e}")
            raise

    def _handle_final_failure(
        self, operation: str, error: Exception, category: ErrorCategory, context: Dict[str, Any]
    ):
        """Handle final failure after all recovery attempts"""
        logger.critical(
            f"Operation '{operation}' failed permanently. "
            f"Category: {category}, Error: {str(error)}"
        )

        # Check if escalation is needed
        if self._should_escalate(category):
            self._escalate_error(operation, error, category, context)

    def _should_escalate(self, category: ErrorCategory) -> bool:
        """Determine if error should be escalated"""
        # Count recent errors in this category
        recent_errors = [
            e
            for e in self.error_history[-50:]  # Last 50 errors
            if e.category == category
            and (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        return len(recent_errors) >= 5  # Escalate if 5+ errors in category in last hour

    def _escalate_error(
        self, operation: str, error: Exception, category: ErrorCategory, context: Dict[str, Any]
    ):
        """Escalate error to higher level handling"""
        logger.critical(f"ESCALATING: Pattern of failures detected for category {category}")
        # Here you would implement escalation logic like:
        # - Send alerts
        # - Trigger automated recovery procedures
        # - Notify operations team
        # - Update monitoring dashboards

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]

        # Categorize errors
        by_category = {}
        by_severity = {}
        by_operation = {}

        for error in recent_errors:
            by_category[error.category.value] = by_category.get(error.category.value, 0) + 1
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
            by_operation[error.operation] = by_operation.get(error.operation, 0) + 1

        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_operation": by_operation,
            "circuit_breakers": {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "last_failure": (
                        breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                    ),
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "error_patterns": dict(
                sorted(self.error_patterns.items(), key=get_error_count, reverse=True)[:10]
            ),
        }

    def get_recovery_recommendations(self) -> List[str]:
        """Get recommendations for improving error recovery"""
        recommendations = []
        error_summary = self.get_error_summary()

        # Analyze patterns and suggest improvements
        if error_summary["total_errors"] > 50:
            recommendations.append("High error rate detected. Consider reviewing system health.")

        if error_summary["by_category"].get("network", 0) > 10:
            recommendations.append(
                "Network errors are frequent. Consider implementing more aggressive retry policies."
            )

        if error_summary["by_severity"].get("critical", 0) > 0:
            recommendations.append("Critical errors detected. Immediate attention required.")

        # Check circuit breaker states
        open_breakers = [
            name
            for name, breaker in self.circuit_breakers.items()
            if breaker.state == CircuitState.OPEN
        ]
        if open_breakers:
            recommendations.append(
                f"Circuit breakers open for: {', '.join(open_breakers)}. Check service health."
            )

        return recommendations


def error_handler(
    category: ErrorCategory = ErrorCategory.UNKNOWN, max_retries: int = 3, fallback: Callable = None
):
    """Decorator for automatic error handling and recovery"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get error manager from first argument (usually self)
            error_manager = None
            if args and hasattr(args[0], "_error_manager"):
                error_manager = args[0]._error_manager

            if error_manager:
                return await error_manager.execute_with_recovery(
                    operation_name=func.__name__, func=func, *args, category=category, **kwargs
                )
            else:
                # Fallback to simple retry logic
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return (
                            await func(*args, **kwargs)
                            if asyncio.iscoroutinefunction(func)
                            else func(*args, **kwargs)
                        )
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                        else:
                            if fallback:
                                return fallback(*args, **kwargs)
                            raise last_exception

        return wrapper

    return decorator


# Global error recovery manager registry
_error_managers: Dict[str, ErrorRecoveryManager] = {}


def get_error_manager(agent_id: str) -> ErrorRecoveryManager:
    """Get or create error recovery manager for agent"""
    if agent_id not in _error_managers:
        _error_managers[agent_id] = ErrorRecoveryManager(agent_id)
    return _error_managers[agent_id]


def create_error_manager(agent_id: str) -> ErrorRecoveryManager:
    """Create new error recovery manager for agent"""
    manager = ErrorRecoveryManager(agent_id)
    _error_managers[agent_id] = manager
    return manager

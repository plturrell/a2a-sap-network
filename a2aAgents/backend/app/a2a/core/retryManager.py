#!/usr/bin/env python3
"""
Production Retry Manager for A2A Agents
Provides robust retry logic with exponential backoff, circuit breakers, and failure tracking.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class RetryResult(Enum):
    SUCCESS = "success"
    FAILED_RETRIES_EXHAUSTED = "failed_retries_exhausted"
    FAILED_CIRCUIT_BREAKER_OPEN = "failed_circuit_breaker_open"
    FAILED_TIMEOUT = "failed_timeout"


@dataclass
class RetryAttempt:
    attempt_number: int
    timestamp: datetime
    duration_seconds: float
    error: Optional[Exception] = None
    success: bool = False


@dataclass
class RetryStats:
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    avg_duration: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    recent_attempts: List[RetryAttempt] = field(default_factory=list)


class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 2  # consecutive successes to close


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    def should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit breaker state."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker CLOSED after successful recovery")

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker returned to OPEN state")


class RetryManager:
    """Production retry manager with circuit breaker and failure tracking."""

    def __init__(self, circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        self.stats: Dict[str, RetryStats] = {}
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_stats(self, operation_name: str) -> RetryStats:
        """Get retry statistics for an operation."""
        if operation_name not in self.stats:
            self.stats[operation_name] = RetryStats()
        return self.stats[operation_name]

    def _calculate_backoff_delay(self, attempt: int, base_delay: float, max_delay: float,
                                jitter: bool = True) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

        if jitter:
            # Add ±20% jitter to prevent thundering herd
            jitter_range = delay * 0.2
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 1.0,
    max_backoff: float = 60.0,
    exceptions: Union[Exception, tuple] = Exception,
    circuit_breaker: bool = True,
    operation_name: Optional[str] = None
):
    """
    Decorator for retry logic with exponential backoff and circuit breaker.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Base delay for exponential backoff
        max_backoff: Maximum backoff delay in seconds
        exceptions: Exception types that trigger retry
        circuit_breaker: Whether to use circuit breaker
        operation_name: Name for tracking statistics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            retry_manager = RetryManager()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            stats = retry_manager.get_stats(op_name)

            if circuit_breaker and not retry_manager.circuit_breaker.should_allow_request():
                logger.warning(f"Circuit breaker OPEN for {op_name}, request blocked")
                stats.failed_attempts += 1
                raise Exception("Circuit breaker is OPEN")

            last_exception = None

            for attempt in range(1, max_attempts + 1):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Record success
                    duration = time.time() - start_time
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        timestamp=datetime.utcnow(),
                        duration_seconds=duration,
                        success=True
                    )

                    stats.successful_attempts += 1
                    stats.total_attempts += 1
                    stats.last_success = datetime.utcnow()
                    stats.recent_attempts.append(attempt_record)

                    # Keep only recent attempts (last 10)
                    stats.recent_attempts = stats.recent_attempts[-10:]

                    # Update average duration
                    total_duration = sum(a.duration_seconds for a in stats.recent_attempts)
                    stats.avg_duration = total_duration / len(stats.recent_attempts)

                    if circuit_breaker:
                        retry_manager.circuit_breaker.record_success()

                    if attempt > 1:
                        logger.info(f"{op_name} succeeded on attempt {attempt}")

                    return result

                except exceptions as e:
                    last_exception = e
                    duration = time.time() - start_time

                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        timestamp=datetime.utcnow(),
                        duration_seconds=duration,
                        error=e,
                        success=False
                    )

                    stats.failed_attempts += 1
                    stats.total_attempts += 1
                    stats.last_failure = datetime.utcnow()
                    stats.recent_attempts.append(attempt_record)
                    stats.recent_attempts = stats.recent_attempts[-10:]

                    if circuit_breaker:
                        retry_manager.circuit_breaker.record_failure()

                    if attempt == max_attempts:
                        logger.error(f"{op_name} failed after {max_attempts} attempts: {e}")
                        break

                    # Calculate backoff delay
                    delay = retry_manager._calculate_backoff_delay(
                        attempt, backoff_factor, max_backoff
                    )

                    logger.warning(f"{op_name} attempt {attempt} failed: {e}. "
                                  f"Retrying in {delay:.2f}s")

                    await asyncio.sleep(delay)

            # All attempts failed
            raise last_exception or Exception("All retry attempts failed")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For synchronous functions, use a simpler retry mechanism
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break

                    delay = min(backoff_factor * (2 ** (attempt - 1)), max_backoff)
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)

            raise last_exception or Exception("All retry attempts failed")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global retry manager instance
retry_manager = RetryManager()


def with_retry(max_attempts: int = 3, backoff_factor: float = 1.0):
    """Simple retry decorator for backward compatibility."""
    return retry_with_backoff(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        exceptions=Exception
    )

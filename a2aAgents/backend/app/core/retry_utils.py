"""
Retry utilities with exponential backoff and circuit breaker pattern
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def call_succeeded(self):
        """Reset failure count on success"""
        self.failure_count = 0
        self.state = "closed"

    def call_failed(self):
        """Record failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def is_open(self) -> bool:
        """Check if circuit should be open"""
        if self.state == "closed":
            return False

        if self.state == "open" and self.last_failure_time:
            if datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
                return False

        return self.state == "open"


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for retrying functions with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry
        circuit_breaker: Optional circuit breaker instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            if circuit_breaker and circuit_breaker.is_open():
                raise CircuitBreakerError("Circuit breaker is open")

            last_exception = None

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.call_succeeded()
                    return result
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        if circuit_breaker:
                            circuit_breaker.call_failed()
                        logger.error(f"Final attempt failed for {func.__name__}: {str(e)}")
                        raise

                    delay = min(backoff_factor ** attempt, max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if circuit_breaker and circuit_breaker.is_open():
                raise CircuitBreakerError("Circuit breaker is open")

            last_exception = None

            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.call_succeeded()
                    return result
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        if circuit_breaker:
                            circuit_breaker.call_failed()
                        logger.error(f"Final attempt failed for {func.__name__}: {str(e)}")
                        raise

                    delay = min(backoff_factor ** attempt, max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RetryManager:
    """Centralized retry management with circuit breakers"""

    def __init__(self):
        self.circuit_breakers = {}

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]

    def retry_with_circuit_breaker(
        self,
        service_name: str,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """Get a retry decorator with circuit breaker for a service"""
        circuit_breaker = self.get_circuit_breaker(service_name)
        return retry_with_backoff(
            max_attempts=max_attempts,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            exceptions=exceptions,
            circuit_breaker=circuit_breaker
        )


# Global retry manager instance
retry_manager = RetryManager()
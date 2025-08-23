"""
Comprehensive error handling and circuit breaker implementation for A2A agents.
Provides production-ready resilience patterns.
"""
import asyncio
import functools
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = datetime.now()
        
        # Metrics
        self._call_history = deque(maxlen=100)
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self._state == CircuitState.OPEN and
            self._last_failure_time and
            datetime.now() - self._last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        if self._should_attempt_reset():
            self._transition_to_half_open()
        
        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        self._total_calls += 1
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            
            # Record metrics
            duration = time.time() - start_time
            self._call_history.append({
                'timestamp': datetime.now(),
                'success': True,
                'duration': duration
            })
            
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            
            # Record metrics
            duration = time.time() - start_time
            self._call_history.append({
                'timestamp': datetime.now(),
                'success': False,
                'duration': duration,
                'error': str(e)
            })
            
            raise
    
    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Async version of call."""
        if self._should_attempt_reset():
            self._transition_to_half_open()
        
        if self._state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        self._total_calls += 1
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            
            # Record metrics
            duration = time.time() - start_time
            self._call_history.append({
                'timestamp': datetime.now(),
                'success': True,
                'duration': duration
            })
            
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            
            # Record metrics
            duration = time.time() - start_time
            self._call_history.append({
                'timestamp': datetime.now(),
                'success': False,
                'duration': duration,
                'error': str(e)
            })
            
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self._total_successes += 1
        self._failure_count = 0
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed call."""
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self._failure_count >= self.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            self._last_state_change = datetime.now()
            logger.warning(f"Circuit {self.name} transitioned to OPEN")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._success_count = 0
            self._failure_count = 0
            self._last_state_change = datetime.now()
            logger.info(f"Circuit {self.name} transitioned to CLOSED")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        if self._state != CircuitState.HALF_OPEN:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            self._last_state_change = datetime.now()
            logger.info(f"Circuit {self.name} transitioned to HALF_OPEN")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        recent_calls = list(self._call_history)
        success_rate = (
            self._total_successes / self._total_calls 
            if self._total_calls > 0 else 0
        )
        
        return {
            'name': self.name,
            'state': self._state.value,
            'total_calls': self._total_calls,
            'total_successes': self._total_successes,
            'total_failures': self._total_failures,
            'success_rate': success_rate,
            'failure_count': self._failure_count,
            'last_failure_time': self._last_failure_time,
            'last_state_change': self._last_state_change,
            'recent_calls': recent_calls
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        self._transition_to_closed()
        logger.info(f"Circuit {self.name} manually reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt number."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
):
    """
    Decorator to apply circuit breaker to a function.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Recovery timeout in seconds
        expected_exception: Exception type to catch
    """
    circuit_breaker = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )
    
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await circuit_breaker.async_call(func, *args, **kwargs)
        
        # Attach circuit breaker for monitoring
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.circuit_breaker = circuit_breaker
        return wrapper
    
    return decorator


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to add retry logic to a function.
    
    Args:
        max_retries: Maximum retry attempts
        initial_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to retry on
    """
    strategy = RetryStrategy(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay
    )
    
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}")
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class ErrorContext:
    """Context manager for comprehensive error handling."""
    
    def __init__(
        self,
        operation_name: str,
        log_errors: bool = True,
        raise_errors: bool = True,
        default_value: Any = None
    ):
        self.operation_name = operation_name
        self.log_errors = log_errors
        self.raise_errors = raise_errors
        self.default_value = default_value
        self.error = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.error = exc_val
            
            if self.log_errors:
                logger.error(
                    f"Error in {self.operation_name} after {duration:.2f}s: "
                    f"{exc_type.__name__}: {exc_val}",
                    exc_info=True
                )
            
            if not self.raise_errors:
                return True  # Suppress exception
        
        return False
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
"""
Production-Ready Circuit Breaker for MCP Services
Implements fault tolerance and cascading failure prevention
"""

import asyncio
import time
import logging
from typing import Callable, Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes in half-open before closing
    timeout: float = 30.0              # Seconds before trying half-open
    expected_exception: tuple = (Exception,)  # Exceptions to catch
    exclude_exceptions: tuple = ()      # Exceptions to not count as failures

@dataclass
class CircuitStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

class CircuitBreaker:
    """Production-ready circuit breaker implementation"""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_test_running = False

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self.stats.total_calls += 1

            # Check if circuit should be opened
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self._log_state_change(CircuitState.OPEN, CircuitState.HALF_OPEN)
                else:
                    self.stats.rejected_calls += 1
                    raise CircuitOpenError(f"Circuit breaker {self.name} is OPEN")

        # Execute the function
        try:
            result = await self._execute_function(func, *args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _execute_function(self, func: Callable, *args, **kwargs):
        """Execute the wrapped function"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            if self.state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self._log_state_change(CircuitState.HALF_OPEN, CircuitState.CLOSED)
                    self.stats.consecutive_successes = 0

    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        # Check if this exception should be counted
        if isinstance(exception, self.config.exclude_exceptions):
            return

        if not isinstance(exception, self.config.expected_exception):
            return

        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self._log_state_change(CircuitState.CLOSED, CircuitState.OPEN)

            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._log_state_change(CircuitState.HALF_OPEN, CircuitState.OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open"""
        return (
            self.stats.last_failure_time and
            time.time() - self.stats.last_failure_time >= self.config.timeout
        )

    def _log_state_change(self, from_state: CircuitState, to_state: CircuitState):
        """Log state changes"""
        change = {
            'timestamp': datetime.utcnow().isoformat(),
            'from_state': from_state.value,
            'to_state': to_state.value,
            'stats': {
                'total_calls': self.stats.total_calls,
                'failures': self.stats.failed_calls,
                'successes': self.stats.successful_calls,
                'rejected': self.stats.rejected_calls
            }
        }
        self.stats.state_changes.append(change)

        logger.warning(
            f"Circuit breaker {self.name} changed state: {from_state.value} -> {to_state.value}",
            extra={'circuit_breaker': self.name, 'state_change': change}
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'stats': {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'rejected_calls': self.stats.rejected_calls,
                'consecutive_failures': self.stats.consecutive_failures,
                'consecutive_successes': self.stats.consecutive_successes,
                'last_failure_time': self.stats.last_failure_time,
                'uptime_percentage': self._calculate_uptime()
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }

    def _calculate_uptime(self) -> float:
        """Calculate service uptime percentage"""
        total = self.stats.successful_calls + self.stats.failed_calls
        if total == 0:
            return 100.0
        return (self.stats.successful_calls / total) * 100

    async def reset(self):
        """Manually reset circuit breaker"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes = 0
            logger.info(f"Circuit breaker {self.name} manually reset")

class CircuitOpenError(Exception):
    """Exception raised when circuit is open"""
    pass

def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 30.0,
    expected_exception: tuple = (Exception,),
    exclude_exceptions: tuple = ()
):
    """Decorator for adding circuit breaker to functions"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception,
        exclude_exceptions=exclude_exceptions
    )

    def decorator(func: Callable):
        cb_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(cb_name, config)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker for monitoring
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        self.breakers[breaker.name] = breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.breakers.get(name)

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }

    async def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            await breaker.reset()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        total_breakers = len(self.breakers)
        open_breakers = sum(
            1 for b in self.breakers.values()
            if b.state == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1 for b in self.breakers.values()
            if b.state == CircuitState.HALF_OPEN
        )

        return {
            'total_breakers': total_breakers,
            'open_breakers': open_breakers,
            'half_open_breakers': half_open_breakers,
            'closed_breakers': total_breakers - open_breakers - half_open_breakers,
            'health_score': (total_breakers - open_breakers) / total_breakers * 100 if total_breakers > 0 else 100,
            'breakers': self.get_all_states()
        }

# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

# Example usage for MCP cross-agent communication
@circuit_breaker(
    name="cross_agent_validation",
    failure_threshold=3,
    timeout=20.0,
    expected_exception=(ConnectionError, TimeoutError)
)
async def call_remote_agent(agent_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example function with circuit breaker protection"""
    # Implementation would make actual HTTP call
    pass

"""
Enhanced Circuit Breaker Implementation for A2A Agent Communications
Provides intelligent error handling, recovery mechanisms, and failure isolation
"""

import time
import logging
import asyncio
import statistics
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""

    failure_threshold: int = 5  # Failures before opening
    recovery_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: int = 60  # Time before half-open attempt
    success_threshold: float = 0.5  # Success rate to maintain closed state
    slow_call_threshold: float = 5.0  # Seconds to consider a call slow
    slow_call_rate_threshold: float = 0.8  # Rate of slow calls to consider failure
    max_concurrent_calls: int = 10  # Max concurrent calls in half-open state

    # Exponential backoff settings
    enable_exponential_backoff: bool = True
    initial_backoff: float = 1.0  # Initial backoff time
    max_backoff: float = 300.0  # Maximum backoff time
    backoff_multiplier: float = 2.0  # Backoff multiplier

    # Health check settings
    enable_health_checks: bool = True
    health_check_interval: int = 30  # Seconds between health checks
    health_check_timeout: float = 5.0  # Health check timeout


@dataclass
class CallResult:
    """Result of a circuit breaker protected call"""

    success: bool
    duration: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""

    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    last_failure_time: Optional[float]
    state_changed_time: float
    success_rate: float
    average_response_time: float
    slow_call_rate: float
    concurrent_calls: int


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with exponential backoff, health checks,
    and intelligent recovery mechanisms for A2A agent communications
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        redis_url: str = "redis://localhost:6379/2",
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.redis_url = redis_url
        self.redis_client = None

        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_time = time.time()
        self.current_backoff = self.config.initial_backoff
        self.concurrent_calls = 0

        # Metrics tracking
        self.call_history: List[CallResult] = []
        self.max_history_size = 100

        # Health check task
        self.health_check_task = None
        self.health_check_callback: Optional[Callable] = None

        # Storage key prefix
        self.storage_prefix = f"circuit_breaker:{self.name}"

    async def initialize(self, health_check_callback: Optional[Callable] = None):
        """Initialize circuit breaker with optional Redis storage and health checks"""
        try:
            # Initialize Redis for distributed circuit breaker state
            if REDIS_AVAILABLE:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()

                # Load existing state from Redis
                await self._load_state_from_redis()

                logger.info(f"✅ Circuit breaker '{self.name}' initialized with Redis")
            else:
                logger.warning(
                    f"Circuit breaker '{self.name}' using local state (Redis not available)"
                )

            # Set up health check callback
            if health_check_callback and self.config.enable_health_checks:
                self.health_check_callback = health_check_callback
                await self._start_health_check_task()

        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker '{self.name}': {e}")

    async def call(
        self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection
        """
        # Check if we should allow the call
        if not await self._should_allow_call():
            if fallback:
                logger.info(f"Circuit breaker '{self.name}' open, executing fallback")
                return (
                    await fallback(*args, **kwargs)
                    if asyncio.iscoroutinefunction(fallback)
                    else fallback(*args, **kwargs)
                )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"State: {self.state.value}, "
                    f"Next retry in: {self._get_time_until_retry():.1f}s"
                )

        # Track concurrent calls
        self.concurrent_calls += 1
        start_time = time.time()

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record successful call
            duration = time.time() - start_time
            await self._record_success(duration)

            return result

        except Exception as e:
            # Record failed call
            duration = time.time() - start_time
            await self._record_failure(str(e), duration)
            raise

        finally:
            self.concurrent_calls -= 1

    async def _should_allow_call(self) -> bool:
        """Determine if a call should be allowed based on current state"""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            time_since_failure = current_time - (self.last_failure_time or 0)

            if self.config.enable_exponential_backoff:
                retry_time = self.current_backoff
            else:
                retry_time = self.config.timeout_seconds

            if time_since_failure >= retry_time:
                await self._transition_to_half_open()
                return True

            return False

        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.concurrent_calls < self.config.max_concurrent_calls

        return False

    async def _record_success(self, duration: float):
        """Record a successful call"""
        result = CallResult(success=True, duration=duration)
        self.call_history.append(result)
        self._trim_history()

        self.success_count += 1

        # Handle state transitions
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.recovery_threshold:
                await self._transition_to_closed()

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

        # Update distributed state
        await self._save_state_to_redis()

        logger.debug(f"Circuit breaker '{self.name}' recorded success (duration: {duration:.2f}s)")

    async def _record_failure(self, error: str, duration: float):
        """Record a failed call"""
        result = CallResult(success=False, duration=duration, error=error)
        self.call_history.append(result)
        self._trim_history()

        self.failure_count += 1
        self.last_failure_time = time.time()

        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            # Check failure threshold
            failure_rate = self._calculate_failure_rate()
            slow_call_rate = self._calculate_slow_call_rate()

            should_open = (
                self.failure_count >= self.config.failure_threshold
                or failure_rate < self.config.success_threshold
                or slow_call_rate > self.config.slow_call_rate_threshold
            )

            if should_open:
                await self._transition_to_open()

        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state goes back to open
            await self._transition_to_open()

        # Update distributed state
        await self._save_state_to_redis()

        logger.warning(
            f"Circuit breaker '{self.name}' recorded failure: {error} (duration: {duration:.2f}s)"
        )

    async def _transition_to_open(self):
        """Transition circuit breaker to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_changed_time = time.time()
        self.success_count = 0  # Reset success count

        # Update exponential backoff
        if self.config.enable_exponential_backoff:
            self.current_backoff = min(
                self.current_backoff * self.config.backoff_multiplier, self.config.max_backoff
            )

        logger.warning(
            f"Circuit breaker '{self.name}' opened "
            f"(failures: {self.failure_count}, backoff: {self.current_backoff:.1f}s)"
        )

        await self._save_state_to_redis()
        await self._emit_state_change_event(old_state, self.state)

    async def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_changed_time = time.time()
        self.success_count = 0  # Reset for counting recovery successes

        logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")

        await self._save_state_to_redis()
        await self._emit_state_change_event(old_state, self.state)

    async def _transition_to_closed(self):
        """Transition circuit breaker to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_changed_time = time.time()
        self.failure_count = 0
        self.success_count = 0

        # Reset exponential backoff
        self.current_backoff = self.config.initial_backoff

        logger.info(f"Circuit breaker '{self.name}' closed (recovered)")

        await self._save_state_to_redis()
        await self._emit_state_change_event(old_state, self.state)

    def _calculate_failure_rate(self) -> float:
        """Calculate recent failure rate"""
        if not self.call_history:
            return 1.0

        # Look at recent history (last 20 calls or 5 minutes)
        recent_time = time.time() - 300  # 5 minutes
        recent_calls = [call for call in self.call_history[-20:] if call.timestamp > recent_time]

        if not recent_calls:
            return 1.0

        successful_calls = sum(1 for call in recent_calls if call.success)
        return successful_calls / len(recent_calls)

    def _calculate_slow_call_rate(self) -> float:
        """Calculate rate of slow calls"""
        if not self.call_history:
            return 0.0

        recent_calls = self.call_history[-20:]  # Last 20 calls
        if not recent_calls:
            return 0.0

        slow_calls = sum(
            1 for call in recent_calls if call.duration > self.config.slow_call_threshold
        )

        return slow_calls / len(recent_calls)

    def _get_time_until_retry(self) -> float:
        """Get time remaining until next retry attempt"""
        if self.state != CircuitState.OPEN or not self.last_failure_time:
            return 0.0

        if self.config.enable_exponential_backoff:
            retry_time = self.current_backoff
        else:
            retry_time = self.config.timeout_seconds

        elapsed = time.time() - self.last_failure_time
        return max(0, retry_time - elapsed)

    def _trim_history(self):
        """Trim call history to prevent memory growth"""
        if len(self.call_history) > self.max_history_size:
            self.call_history = self.call_history[-self.max_history_size :]

    async def _save_state_to_redis(self):
        """Save circuit breaker state to Redis for distribution"""
        if not self.redis_client:
            return

        try:
            state_data = {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "state_changed_time": self.state_changed_time,
                "current_backoff": self.current_backoff,
                "updated_at": time.time(),
            }

            await self.redis_client.setex(
                f"{self.storage_prefix}:state", 300, json.dumps(state_data)  # 5 minute TTL
            )

        except Exception as e:
            logger.error(f"Failed to save circuit breaker state to Redis: {e}")

    async def _load_state_from_redis(self):
        """Load circuit breaker state from Redis"""
        if not self.redis_client:
            return

        try:
            state_json = await self.redis_client.get(f"{self.storage_prefix}:state")
            if not state_json:
                return

            state_data = json.loads(state_json)

            # Only load state if it's recent (within 5 minutes)
            if time.time() - state_data.get("updated_at", 0) < 300:
                self.state = CircuitState(state_data["state"])
                self.failure_count = state_data["failure_count"]
                self.success_count = state_data["success_count"]
                self.last_failure_time = state_data["last_failure_time"]
                self.state_changed_time = state_data["state_changed_time"]
                self.current_backoff = state_data["current_backoff"]

                logger.info(f"Loaded circuit breaker state from Redis: {self.state.value}")

        except Exception as e:
            logger.error(f"Failed to load circuit breaker state from Redis: {e}")

    async def _emit_state_change_event(self, old_state: CircuitState, new_state: CircuitState):
        """Emit state change event for monitoring"""
        try:
            event_data = {
                "circuit_breaker": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "timestamp": datetime.utcnow().isoformat(),
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "metrics": await self.get_metrics(),
            }

            if self.redis_client:
                # Store event for monitoring
                await self.redis_client.lpush("circuit_breaker_events", json.dumps(event_data))
                await self.redis_client.expire("circuit_breaker_events", 86400)  # 24 hours

            logger.info(
                f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
            )

        except Exception as e:
            logger.error(f"Failed to emit state change event: {e}")

    async def _start_health_check_task(self):
        """Start background health check task"""
        if not self.health_check_callback:
            return

        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval)

                    # Only perform health checks when circuit is open
                    if self.state == CircuitState.OPEN:
                        logger.debug(f"Performing health check for circuit breaker '{self.name}'")

                        try:
                            # Execute health check with timeout
                            health_result = await asyncio.wait_for(
                                self.health_check_callback(),
                                timeout=self.config.health_check_timeout,
                            )

                            if health_result:
                                logger.info(
                                    f"Health check passed for '{self.name}', attempting recovery"
                                )
                                await self._transition_to_half_open()

                        except asyncio.TimeoutError:
                            logger.warning(f"Health check timeout for '{self.name}'")
                        except Exception as e:
                            logger.warning(f"Health check failed for '{self.name}': {e}")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check loop error for '{self.name}': {e}")

        self.health_check_task = asyncio.create_task(health_check_loop())

    async def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics"""
        recent_calls = self.call_history[-50:] if self.call_history else []

        if recent_calls:
            success_rate = sum(1 for call in recent_calls if call.success) / len(recent_calls)
            avg_response_time = statistics.mean(call.duration for call in recent_calls)
            slow_call_rate = self._calculate_slow_call_rate()
        else:
            success_rate = 1.0
            avg_response_time = 0.0
            slow_call_rate = 0.0

        return CircuitBreakerMetrics(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            total_calls=len(self.call_history),
            last_failure_time=self.last_failure_time,
            state_changed_time=self.state_changed_time,
            success_rate=success_rate,
            average_response_time=avg_response_time,
            slow_call_rate=slow_call_rate,
            concurrent_calls=self.concurrent_calls,
        )

    async def reset(self):
        """Manually reset circuit breaker to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_time = time.time()
        self.current_backoff = self.config.initial_backoff
        self.call_history.clear()

        await self._save_state_to_redis()
        await self._emit_state_change_event(old_state, self.state)

        logger.info(f"Circuit breaker '{self.name}' manually reset")

    async def close(self):
        """Clean up circuit breaker resources"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            await self.redis_client.close()


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers in an A2A agent system
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis_url = redis_url
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}

    async def get_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        health_check_callback: Optional[Callable] = None,
    ) -> EnhancedCircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            cb = EnhancedCircuitBreaker(name, config, self.redis_url)
            await cb.initialize(health_check_callback)
            self.circuit_breakers[name] = cb

        return self.circuit_breakers[name]

    async def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers"""
        metrics = {}
        for name, cb in self.circuit_breakers.items():
            metrics[name] = await cb.get_metrics()
        return metrics

    async def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            await cb.reset()

    async def close_all(self):
        """Close all circuit breakers"""
        for cb in self.circuit_breakers.values():
            await cb.close()
        self.circuit_breakers.clear()


# Global circuit breaker manager
_circuit_breaker_manager = None


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create the circuit breaker manager instance"""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager


# Decorator for easy circuit breaker usage
def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig = None,
    fallback: Optional[Callable] = None,
    health_check: Optional[Callable] = None,
):
    """
    Decorator to add circuit breaker protection to functions
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = await get_circuit_breaker_manager()
            cb = await manager.get_circuit_breaker(name, config, health_check)
            return await cb.call(func, *args, fallback=fallback, **kwargs)

        return wrapper

    return decorator


# Example usage for A2A agent communication
async def example_agent_communication():
    """Example of using circuit breaker for agent communication"""

    # Configure circuit breaker for agent communication
    agent_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_threshold=2,
        timeout_seconds=30,
        enable_exponential_backoff=True,
        enable_health_checks=True,
        health_check_interval=20,
    )

    # Health check function
    async def agent_health_check():
        # Implement actual health check logic
        # Return True if agent is healthy, False otherwise
        return True

    # Get circuit breaker
    manager = await get_circuit_breaker_manager()
    cb = await manager.get_circuit_breaker("agent_communication", agent_config, agent_health_check)

    # Fallback function
    async def fallback_response():
        return {"error": "Service temporarily unavailable", "fallback": True}

    # Protected agent call
    try:
        result = await cb.call(
            some_agent_function,
            agent_id="agent_123",
            message={"type": "request", "data": {}},
            fallback=fallback_response,
        )
        return result
    except CircuitBreakerOpenError as e:
        logger.error(f"Circuit breaker open: {e}")
        return await fallback_response()


async def some_agent_function(agent_id: str, message: dict):
    """Example agent function that might fail"""
    # Simulate agent communication that might fail
    await asyncio.sleep(0.1)
    return {"status": "success", "data": "response"}

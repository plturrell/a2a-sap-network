"""
Circuit Breaker Implementation for Fault Tolerance
"""

import asyncio
import time
from typing import Optional, Callable, TypeVar, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list = field(default_factory=list)
    
    def record_success(self):
        """Record successful call"""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
    
    def record_failure(self):
        """Record failed call"""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout: Seconds to wait before trying half-open state
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_start: Optional[float] = None
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics"""
        return self._stats
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (self._state == CircuitState.OPEN and
                self._stats.last_failure_time and
                time.time() - self._stats.last_failure_time >= self.timeout)
    
    async def _record_state_change(self, new_state: CircuitState, reason: str):
        """Record state change"""
        old_state = self._state
        self._state = new_state
        
        change_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason,
            "stats": {
                "total_calls": self._stats.total_calls,
                "failure_rate": self._stats.get_failure_rate(),
                "consecutive_failures": self._stats.consecutive_failures
            }
        }
        
        self._stats.state_changes.append(change_record)
        logger.info(f"Circuit breaker state changed: {old_state.value} -> {new_state.value} ({reason})")
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function fails
        """
        async with self._lock:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                await self._record_state_change(CircuitState.HALF_OPEN, "Timeout expired")
                self._half_open_start = time.time()
            
            # Check circuit state
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN (failures: {self._stats.consecutive_failures})"
                )
        
        # Attempt the call
        try:
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            # Record failure
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self._stats.record_success()
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    await self._record_state_change(
                        CircuitState.CLOSED,
                        f"Success threshold reached ({self.success_threshold})"
                    )
                    self._stats.consecutive_failures = 0
            
            elif self._state == CircuitState.CLOSED:
                # Reset consecutive failures on any success
                self._stats.consecutive_failures = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self._stats.record_failure()
            
            if self._state == CircuitState.HALF_OPEN:
                await self._record_state_change(
                    CircuitState.OPEN,
                    "Failure in half-open state"
                )
            
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    await self._record_state_change(
                        CircuitState.OPEN,
                        f"Failure threshold reached ({self.failure_threshold})"
                    )
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self._state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        return self._state == CircuitState.CLOSED
    
    def reset(self):
        """Manually reset circuit to closed state"""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, 
                   name: str,
                   failure_threshold: int = 5,
                   success_threshold: int = 2,
                   timeout: float = 60.0) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout
            )
            logger.info(f"Created circuit breaker: {name}")
        
        return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all breakers"""
        stats = {}
        for name, breaker in self._breakers.items():
            stats[name] = {
                "state": breaker.state.value,
                "total_calls": breaker.stats.total_calls,
                "failure_rate": breaker.stats.get_failure_rate(),
                "consecutive_failures": breaker.stats.consecutive_failures,
                "last_failure": datetime.fromtimestamp(breaker.stats.last_failure_time).isoformat()
                               if breaker.stats.last_failure_time else None
            }
        return stats
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")


# Global circuit breaker manager
_breaker_manager = None

def get_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    global _breaker_manager
    if _breaker_manager is None:
        _breaker_manager = CircuitBreakerManager()
    return _breaker_manager
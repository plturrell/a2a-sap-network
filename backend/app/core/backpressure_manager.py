"""
Enhanced Backpressure Management System for A2A Agents
Provides adaptive rate limiting, load shedding, and graceful degradation
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import statistics
import collections

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..a2a.sdk.types import A2AMessage, TaskStatus
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class BackpressureStrategy(str, Enum):
    """Backpressure handling strategies"""
    QUEUE_LIMIT = "queue_limit"
    RATE_LIMIT = "rate_limit"
    LOAD_SHED = "load_shed"
    CIRCUIT_BREAKER = "circuit_breaker"
    ADAPTIVE_THROTTLE = "adaptive_throttle"
    PRIORITY_QUEUE = "priority_queue"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class LoadLevel(str, Enum):
    """System load levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    OVERLOAD = "overload"


class RequestPriority(str, Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    SYSTEM = "system"


@dataclass
class BackpressureConfig:
    """Backpressure configuration"""
    max_queue_size: int = 1000
    max_requests_per_second: float = 100.0
    load_shed_threshold: float = 0.8
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
    adaptive_window_size: int = 60
    priority_queue_enabled: bool = True
    graceful_degradation_enabled: bool = True
    resource_monitoring_enabled: bool = True


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_depth: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BackpressureDecision:
    """Backpressure decision result"""
    allow_request: bool
    strategy_applied: Optional[BackpressureStrategy]
    load_level: LoadLevel
    reason: str
    retry_after_seconds: Optional[int] = None
    degraded_response: Optional[Dict[str, Any]] = None


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system performance"""
    
    def __init__(self, initial_rate: float = 100.0, window_size: int = 60):
        self.current_rate = initial_rate
        self.initial_rate = initial_rate
        self.window_size = window_size
        self.requests = collections.deque()
        self.performance_history = collections.deque(maxlen=window_size)
        self.last_adjustment = time.time()
        self.adjustment_cooldown = 10  # seconds
    
    async def allow_request(self, system_metrics: SystemMetrics) -> Tuple[bool, float]:
        """Check if request should be allowed and return current rate"""
        now = time.time()
        
        # Remove old requests outside the window
        while self.requests and self.requests[0] <= now - 60:
            self.requests.popleft()
        
        current_rps = len(self.requests)
        
        # Adjust rate based on system performance
        await self._adjust_rate(system_metrics, now)
        
        # Check if we're within the current rate limit
        if current_rps < self.current_rate:
            self.requests.append(now)
            return True, self.current_rate
        else:
            return False, self.current_rate
    
    async def _adjust_rate(self, metrics: SystemMetrics, now: float):
        """Dynamically adjust rate based on system performance"""
        if now - self.last_adjustment < self.adjustment_cooldown:
            return
        
        self.performance_history.append(metrics)
        
        if len(self.performance_history) < 5:
            return  # Need more data points
        
        # Calculate performance indicators
        recent_cpu = statistics.mean([m.cpu_usage for m in list(self.performance_history)[-5:]])
        recent_response_time = statistics.mean([m.response_time_p95 for m in list(self.performance_history)[-5:]])
        recent_error_rate = statistics.mean([m.error_rate for m in list(self.performance_history)[-5:]])
        
        # Determine adjustment factor
        adjustment_factor = 1.0
        
        # CPU-based adjustment
        if recent_cpu > 80:
            adjustment_factor *= 0.7  # Reduce by 30%
        elif recent_cpu > 60:
            adjustment_factor *= 0.9  # Reduce by 10%
        elif recent_cpu < 30:
            adjustment_factor *= 1.1  # Increase by 10%
        
        # Response time-based adjustment
        if recent_response_time > 1.0:  # 1 second
            adjustment_factor *= 0.8
        elif recent_response_time > 0.5:
            adjustment_factor *= 0.95
        elif recent_response_time < 0.1:
            adjustment_factor *= 1.05
        
        # Error rate-based adjustment
        if recent_error_rate > 0.05:  # 5% error rate
            adjustment_factor *= 0.6
        elif recent_error_rate > 0.01:
            adjustment_factor *= 0.9
        elif recent_error_rate < 0.001:
            adjustment_factor *= 1.02
        
        # Apply adjustment with bounds
        self.current_rate = max(
            self.initial_rate * 0.1,  # Never go below 10% of initial
            min(
                self.initial_rate * 2.0,  # Never go above 200% of initial
                self.current_rate * adjustment_factor
            )
        )
        
        self.last_adjustment = now
        
        if adjustment_factor != 1.0:
            logger.info(f"Adjusted rate limit: {self.current_rate:.1f} RPS (factor: {adjustment_factor:.2f})")


class PriorityQueue:
    """Priority-based request queue"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {
            RequestPriority.SYSTEM: asyncio.Queue(maxsize=max_size // 5),
            RequestPriority.CRITICAL: asyncio.Queue(maxsize=max_size // 4),
            RequestPriority.HIGH: asyncio.Queue(maxsize=max_size // 4),
            RequestPriority.NORMAL: asyncio.Queue(maxsize=max_size // 3),
            RequestPriority.LOW: asyncio.Queue(maxsize=max_size // 6)
        }
        self.total_size = 0
    
    async def put(self, item: Any, priority: RequestPriority = RequestPriority.NORMAL) -> bool:
        """Add item to appropriate priority queue"""
        try:
            queue = self.queues[priority]
            if queue.full():
                # Try to make space by dropping low priority items
                await self._make_space_for_priority(priority)
            
            if not queue.full():
                await queue.put(item)
                self.total_size += 1
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to enqueue item with priority {priority}: {e}")
            return False
    
    async def get(self) -> Tuple[Any, RequestPriority]:
        """Get highest priority item"""
        # Check queues in priority order
        for priority in [RequestPriority.SYSTEM, RequestPriority.CRITICAL, 
                        RequestPriority.HIGH, RequestPriority.NORMAL, RequestPriority.LOW]:
            queue = self.queues[priority]
            if not queue.empty():
                item = await queue.get()
                self.total_size -= 1
                return item, priority
        
        # If all queues are empty, wait for any item
        done, pending = await asyncio.wait(
            [asyncio.create_task(queue.get()) for queue in self.queues.values()],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        result = done.pop().result()
        self.total_size -= 1
        
        # Determine which queue the result came from
        for priority, queue in self.queues.items():
            if queue.qsize() == self.queues[priority].qsize():
                return result, priority
        
        return result, RequestPriority.NORMAL
    
    async def _make_space_for_priority(self, priority: RequestPriority):
        """Make space by dropping lower priority items"""
        # Define priority levels
        priority_levels = {
            RequestPriority.LOW: 1,
            RequestPriority.NORMAL: 2,
            RequestPriority.HIGH: 3,
            RequestPriority.CRITICAL: 4,
            RequestPriority.SYSTEM: 5
        }
        
        current_level = priority_levels[priority]
        
        # Try to drop items from lower priority queues
        for p, level in priority_levels.items():
            if level < current_level:
                queue = self.queues[p]
                if not queue.empty():
                    try:
                        dropped_item = queue.get_nowait()
                        self.total_size -= 1
                        logger.warning(f"Dropped {p} priority item to make space for {priority}")
                        return
                    except asyncio.QueueEmpty:
                        continue
    
    def size(self) -> int:
        """Get total queue size"""
        return self.total_size
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.total_size >= self.max_size


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    async def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    async def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class BackpressureManager:
    """Enhanced backpressure manager for A2A agents"""
    
    def __init__(
        self,
        agent_id: str,
        config: BackpressureConfig = None,
        redis_config: RedisConfig = None
    ):
        self.agent_id = agent_id
        self.config = config or BackpressureConfig()
        self.redis_client = RedisClient(redis_config or RedisConfig())
        
        # Components
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=self.config.max_requests_per_second,
            window_size=self.config.adaptive_window_size
        )
        self.priority_queue = PriorityQueue(self.config.max_queue_size)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout
        )
        
        # State
        self.current_load_level = LoadLevel.NORMAL
        self.system_metrics = SystemMetrics()
        self.degradation_handlers: Dict[str, Callable] = {}
        self.load_shed_handlers: Dict[str, Callable] = {}
        
        # Monitoring
        self.metrics_history = collections.deque(maxlen=300)  # 5 minutes at 1-second intervals
        self.running = False
        self.monitoring_task = None
    
    async def initialize(self):
        """Initialize the backpressure manager"""
        await self.redis_client.initialize()
        
        # Start monitoring
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Backpressure manager initialized for agent {self.agent_id}")
    
    async def shutdown(self):
        """Shutdown the backpressure manager"""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.redis_client.close()
        logger.info(f"Backpressure manager shut down for agent {self.agent_id}")
    
    def register_degradation_handler(self, operation: str, handler: Callable):
        """Register a degradation handler for an operation"""
        self.degradation_handlers[operation] = handler
        logger.info(f"Registered degradation handler for operation: {operation}")
    
    def register_load_shed_handler(self, operation: str, handler: Callable):
        """Register a load shedding handler for an operation"""
        self.load_shed_handlers[operation] = handler
        logger.info(f"Registered load shed handler for operation: {operation}")
    
    @trace_async("backpressure_check")
    async def should_allow_request(
        self,
        operation: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        request_context: Dict[str, Any] = None
    ) -> BackpressureDecision:
        """Determine if a request should be allowed"""
        
        add_span_attributes({
            "agent.id": self.agent_id,
            "operation": operation,
            "priority": priority.value,
            "load_level": self.current_load_level.value
        })
        
        # Update system metrics
        await self._update_system_metrics()
        
        # Check circuit breaker first
        if self.circuit_breaker.state == "open":
            return BackpressureDecision(
                allow_request=False,
                strategy_applied=BackpressureStrategy.CIRCUIT_BREAKER,
                load_level=self.current_load_level,
                reason="Circuit breaker is open",
                retry_after_seconds=self.config.circuit_breaker_timeout
            )
        
        # Check rate limits
        rate_allowed, current_rate = await self.rate_limiter.allow_request(self.system_metrics)
        if not rate_allowed:
            return BackpressureDecision(
                allow_request=False,
                strategy_applied=BackpressureStrategy.RATE_LIMIT,
                load_level=self.current_load_level,
                reason=f"Rate limit exceeded (current: {current_rate:.1f} RPS)",
                retry_after_seconds=1
            )
        
        # Check queue capacity
        if self.priority_queue.size() >= self.config.max_queue_size:
            # Try to queue with priority
            if priority in [RequestPriority.CRITICAL, RequestPriority.SYSTEM]:
                # Force enqueue for critical requests
                pass
            else:
                return BackpressureDecision(
                    allow_request=False,
                    strategy_applied=BackpressureStrategy.QUEUE_LIMIT,
                    load_level=self.current_load_level,
                    reason="Queue is full",
                    retry_after_seconds=5
                )
        
        # Check for load shedding
        if self.current_load_level == LoadLevel.CRITICAL:
            if priority in [RequestPriority.LOW, RequestPriority.NORMAL]:
                # Shed low priority requests
                shed_result = await self._apply_load_shedding(operation, request_context)
                if not shed_result["allow"]:
                    return BackpressureDecision(
                        allow_request=False,
                        strategy_applied=BackpressureStrategy.LOAD_SHED,
                        load_level=self.current_load_level,
                        reason="Load shedding active for low priority requests",
                        retry_after_seconds=shed_result.get("retry_after", 10)
                    )
        
        # Check for graceful degradation
        if self.current_load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL]:
            degraded_response = await self._apply_graceful_degradation(operation, request_context)
            if degraded_response:
                return BackpressureDecision(
                    allow_request=True,
                    strategy_applied=BackpressureStrategy.GRACEFUL_DEGRADATION,
                    load_level=self.current_load_level,
                    reason="Request allowed with degraded response",
                    degraded_response=degraded_response
                )
        
        # Allow the request
        return BackpressureDecision(
            allow_request=True,
            strategy_applied=None,
            load_level=self.current_load_level,
            reason="Request allowed normally"
        )
    
    async def enqueue_request(
        self,
        request: Any,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> bool:
        """Enqueue a request with specified priority"""
        return await self.priority_queue.put(request, priority)
    
    async def dequeue_request(self) -> Tuple[Any, RequestPriority]:
        """Dequeue the highest priority request"""
        return await self.priority_queue.get()
    
    async def execute_with_backpressure(
        self,
        func: Callable,
        operation: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        request_context: Dict[str, Any] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with backpressure protection"""
        
        # Check if request should be allowed
        decision = await self.should_allow_request(operation, priority, request_context)
        
        if not decision.allow_request:
            if decision.strategy_applied == BackpressureStrategy.CIRCUIT_BREAKER:
                raise Exception(f"Request rejected: {decision.reason}")
            elif decision.retry_after_seconds:
                raise Exception(f"Request rejected: {decision.reason}. Retry after {decision.retry_after_seconds} seconds")
            else:
                raise Exception(f"Request rejected: {decision.reason}")
        
        # Return degraded response if applicable
        if decision.degraded_response:
            return decision.degraded_response
        
        # Execute with circuit breaker protection
        try:
            result = await self.circuit_breaker.call(func, *args, **kwargs)
            
            # Update success metrics
            await self._record_success(operation)
            
            return result
            
        except Exception as e:
            # Update failure metrics
            await self._record_failure(operation, str(e))
            raise e
    
    async def _update_system_metrics(self):
        """Update current system metrics"""
        try:
            # Get system metrics (simplified - would integrate with actual monitoring)
            import psutil
            
            self.system_metrics = SystemMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                queue_depth=self.priority_queue.size(),
                requests_per_second=len(self.rate_limiter.requests),
                timestamp=datetime.utcnow()
            )
            
            # Update load level based on metrics
            self._update_load_level()
            
            # Store metrics history
            self.metrics_history.append(self.system_metrics)
            
        except ImportError:
            # Fallback metrics if psutil not available
            self.system_metrics = SystemMetrics(
                queue_depth=self.priority_queue.size(),
                requests_per_second=len(self.rate_limiter.requests),
                timestamp=datetime.utcnow()
            )
    
    def _update_load_level(self):
        """Update current load level based on system metrics"""
        metrics = self.system_metrics
        
        # Calculate composite load score
        load_score = 0.0
        
        # CPU contribution (40%)
        if metrics.cpu_usage > 0:
            load_score += (metrics.cpu_usage / 100.0) * 0.4
        
        # Memory contribution (20%)
        if metrics.memory_usage > 0:
            load_score += (metrics.memory_usage / 100.0) * 0.2
        
        # Queue depth contribution (25%)
        queue_ratio = min(1.0, metrics.queue_depth / self.config.max_queue_size)
        load_score += queue_ratio * 0.25
        
        # Response time contribution (15%)
        if metrics.response_time_p95 > 0:
            # Normalize response time (assume 2 seconds is 100% load)
            response_ratio = min(1.0, metrics.response_time_p95 / 2.0)
            load_score += response_ratio * 0.15
        
        # Determine load level
        if load_score >= 0.9:
            self.current_load_level = LoadLevel.OVERLOAD
        elif load_score >= 0.8:
            self.current_load_level = LoadLevel.CRITICAL
        elif load_score >= 0.6:
            self.current_load_level = LoadLevel.HIGH
        elif load_score >= 0.3:
            self.current_load_level = LoadLevel.NORMAL
        else:
            self.current_load_level = LoadLevel.LOW
    
    async def _apply_load_shedding(self, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply load shedding for an operation"""
        handler = self.load_shed_handlers.get(operation)
        
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(context)
                else:
                    result = handler(context)
                return result
            except Exception as e:
                logger.error(f"Load shed handler failed for {operation}: {e}")
        
        # Default load shedding behavior
        return {
            "allow": False,
            "reason": f"Load shedding active for operation {operation}",
            "retry_after": 10
        }
    
    async def _apply_graceful_degradation(self, operation: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply graceful degradation for an operation"""
        if not self.config.graceful_degradation_enabled:
            return None
        
        handler = self.degradation_handlers.get(operation)
        
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(context)
                else:
                    result = handler(context)
                return result
            except Exception as e:
                logger.error(f"Degradation handler failed for {operation}: {e}")
        
        # Default degradation - return cached or simplified response
        return {
            "degraded": True,
            "operation": operation,
            "message": "Service operating in degraded mode",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _record_success(self, operation: str):
        """Record successful operation"""
        try:
            key = f"backpressure:success:{self.agent_id}:{operation}"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # 1 hour expiration
        except Exception as e:
            logger.error(f"Failed to record success for {operation}: {e}")
    
    async def _record_failure(self, operation: str, error: str):
        """Record failed operation"""
        try:
            key = f"backpressure:failure:{self.agent_id}:{operation}"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # 1 hour expiration
            
            # Also record error details
            error_key = f"backpressure:errors:{self.agent_id}:{operation}"
            await self.redis_client.lpush(error_key, json.dumps({
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            }))
            await self.redis_client.ltrim(error_key, 0, 99)  # Keep last 100 errors
            await self.redis_client.expire(error_key, 3600)
            
        except Exception as e:
            logger.error(f"Failed to record failure for {operation}: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                await self._update_system_metrics()
                
                # Log metrics periodically
                if len(self.metrics_history) % 60 == 0:  # Every minute
                    logger.info(
                        f"System metrics - Load: {self.current_load_level.value}, "
                        f"CPU: {self.system_metrics.cpu_usage:.1f}%, "
                        f"Memory: {self.system_metrics.memory_usage:.1f}%, "
                        f"Queue: {self.system_metrics.queue_depth}, "
                        f"RPS: {self.system_metrics.requests_per_second:.1f}"
                    )
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def get_backpressure_status(self) -> Dict[str, Any]:
        """Get current backpressure status"""
        return {
            "agent_id": self.agent_id,
            "load_level": self.current_load_level.value,
            "system_metrics": {
                "cpu_usage": self.system_metrics.cpu_usage,
                "memory_usage": self.system_metrics.memory_usage,
                "queue_depth": self.system_metrics.queue_depth,
                "requests_per_second": self.system_metrics.requests_per_second,
                "error_rate": self.system_metrics.error_rate,
                "response_time_p95": self.system_metrics.response_time_p95
            },
            "rate_limiter": {
                "current_rate": self.rate_limiter.current_rate,
                "initial_rate": self.rate_limiter.initial_rate
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count
            },
            "queue": {
                "size": self.priority_queue.size(),
                "max_size": self.config.max_queue_size,
                "utilization": (self.priority_queue.size() / self.config.max_queue_size) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Global backpressure managers
_backpressure_managers: Dict[str, BackpressureManager] = {}


async def initialize_backpressure_manager(
    agent_id: str,
    config: BackpressureConfig = None,
    redis_config: RedisConfig = None
) -> BackpressureManager:
    """Initialize backpressure manager for an agent"""
    global _backpressure_managers
    
    if agent_id in _backpressure_managers:
        return _backpressure_managers[agent_id]
    
    manager = BackpressureManager(agent_id, config, redis_config)
    await manager.initialize()
    
    _backpressure_managers[agent_id] = manager
    return manager


async def get_backpressure_manager(agent_id: str) -> Optional[BackpressureManager]:
    """Get existing backpressure manager for an agent"""
    return _backpressure_managers.get(agent_id)


async def shutdown_backpressure_manager(agent_id: str):
    """Shutdown backpressure manager for an agent"""
    global _backpressure_managers
    
    if agent_id in _backpressure_managers:
        await _backpressure_managers[agent_id].shutdown()
        del _backpressure_managers[agent_id]
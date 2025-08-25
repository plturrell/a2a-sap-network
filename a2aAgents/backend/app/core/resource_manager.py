"""
Resource Management System for A2A Agents
Provides memory limits, resource cleanup, and leak prevention
"""

import asyncio
import gc
import logging
import psutil
import resource
import threading
import time
import weakref
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import tracemalloc
import sys

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceType(str, Enum):
    """Types of resources managed"""
    MEMORY = "memory"
    CPU = "cpu"
    FILE_DESCRIPTORS = "file_descriptors"
    NETWORK_CONNECTIONS = "network_connections"
    CACHE_ENTRIES = "cache_entries"
    REDIS_CONNECTIONS = "redis_connections"
    DATABASE_CONNECTIONS = "database_connections"
    ASYNC_TASKS = "async_tasks"


class ResourceState(str, Enum):
    """Resource allocation states"""
    ALLOCATED = "allocated"
    IN_USE = "in_use"
    IDLE = "idle"
    CLEANUP_PENDING = "cleanup_pending"
    FREED = "freed"


@dataclass
class ResourceLimit:
    """Resource usage limits"""
    resource_type: ResourceType
    soft_limit: int
    hard_limit: int
    warning_threshold: float = 0.8  # Warn at 80% of soft limit
    auto_cleanup: bool = True
    cleanup_strategy: str = "lru"  # lru, fifo, priority


@dataclass
class ResourceAllocation:
    """Tracks a resource allocation"""
    allocation_id: str
    resource_type: ResourceType
    allocated_at: datetime
    size_bytes: int
    owner: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    state: ResourceState = ResourceState.ALLOCATED


class ResourceTracker:
    """Tracks resource usage and allocations"""

    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_usage: Dict[ResourceType, int] = defaultdict(int)
        self.peak_usage: Dict[ResourceType, int] = defaultdict(int)
        self.allocation_history: deque = deque(maxlen=10000)
        self.lock = threading.RLock()

    def track_allocation(self, allocation: ResourceAllocation):
        """Track a new resource allocation"""
        with self.lock:
            self.allocations[allocation.allocation_id] = allocation
            self.resource_usage[allocation.resource_type] += allocation.size_bytes

            # Update peak usage
            current_usage = self.resource_usage[allocation.resource_type]
            if current_usage > self.peak_usage[allocation.resource_type]:
                self.peak_usage[allocation.resource_type] = current_usage

            # Record in history
            self.allocation_history.append({
                "action": "allocate",
                "timestamp": allocation.allocated_at,
                "allocation_id": allocation.allocation_id,
                "resource_type": allocation.resource_type.value,
                "size_bytes": allocation.size_bytes,
                "owner": allocation.owner
            })

    def track_deallocation(self, allocation_id: str):
        """Track resource deallocation"""
        with self.lock:
            allocation = self.allocations.pop(allocation_id, None)
            if allocation:
                self.resource_usage[allocation.resource_type] -= allocation.size_bytes
                allocation.state = ResourceState.FREED

                # Record in history
                self.allocation_history.append({
                    "action": "deallocate",
                    "timestamp": datetime.utcnow(),
                    "allocation_id": allocation_id,
                    "resource_type": allocation.resource_type.value,
                    "size_bytes": allocation.size_bytes,
                    "owner": allocation.owner
                })

                return allocation
        return None

    def update_access(self, allocation_id: str):
        """Update last access time for an allocation"""
        with self.lock:
            allocation = self.allocations.get(allocation_id)
            if allocation:
                allocation.last_accessed = datetime.utcnow()
                allocation.access_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        with self.lock:
            return {
                "current_usage": dict(self.resource_usage),
                "peak_usage": dict(self.peak_usage),
                "total_allocations": len(self.allocations),
                "allocations_by_type": {
                    rt.value: sum(1 for a in self.allocations.values()
                                 if a.resource_type == rt)
                    for rt in ResourceType
                },
                "memory_by_owner": {
                    owner: sum(a.size_bytes for a in self.allocations.values()
                              if a.owner == owner and a.resource_type == ResourceType.MEMORY)
                    for owner in set(a.owner for a in self.allocations.values())
                }
            }

    def get_stale_allocations(self, max_idle_time: timedelta) -> List[ResourceAllocation]:
        """Get allocations that haven't been accessed recently"""
        cutoff_time = datetime.utcnow() - max_idle_time
        stale_allocations = []

        with self.lock:
            for allocation in self.allocations.values():
                last_access = allocation.last_accessed or allocation.allocated_at
                if last_access < cutoff_time:
                    stale_allocations.append(allocation)

        return stale_allocations


class MemoryPool(Generic[T]):
    """Memory pool for reusable objects"""

    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque()
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0

    def acquire(self) -> T:
        """Acquire an object from the pool"""
        with self.lock:
            if self.pool:
                self.reused_count += 1
                return self.pool.popleft()
            else:
                self.created_count += 1
                return self.factory()

    def release(self, obj: T):
        """Return an object to the pool"""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

    def clear(self):
        """Clear the pool"""
        with self.lock:
            self.pool.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "max_size": self.max_size,
                "created_count": self.created_count,
                "reused_count": self.reused_count,
                "reuse_ratio": self.reused_count / max(1, self.created_count + self.reused_count)
            }


class ResourceManager:
    """Main resource management system"""

    def __init__(self):
        self.limits: Dict[ResourceType, ResourceLimit] = {}
        self.tracker = ResourceTracker()
        self.cleanup_handlers: Dict[ResourceType, List[Callable]] = defaultdict(list)
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.monitoring_enabled = True
        self.cleanup_interval = timedelta(minutes=5)
        self.max_idle_time = timedelta(minutes=30)

        # System monitoring
        self.process = psutil.Process()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Memory tracking
        self.enable_memory_profiling = False
        self.memory_snapshots = deque(maxlen=100)

        # Resource pools
        self._initialize_memory_pools()

        # Default limits
        self._set_default_limits()

    def _set_default_limits(self):
        """Set default resource limits"""
        # Memory limits (in bytes)
        self.set_limit(ResourceType.MEMORY,
                      soft_limit=1024 * 1024 * 1024,  # 1GB
                      hard_limit=2048 * 1024 * 1024)  # 2GB

        # File descriptor limits
        self.set_limit(ResourceType.FILE_DESCRIPTORS,
                      soft_limit=1000,
                      hard_limit=1500)

        # Network connection limits
        self.set_limit(ResourceType.NETWORK_CONNECTIONS,
                      soft_limit=500,
                      hard_limit=1000)

        # Cache entry limits
        self.set_limit(ResourceType.CACHE_ENTRIES,
                      soft_limit=10000,
                      hard_limit=20000)

        # Redis connection limits
        self.set_limit(ResourceType.REDIS_CONNECTIONS,
                      soft_limit=50,
                      hard_limit=100)

        # Database connection limits
        self.set_limit(ResourceType.DATABASE_CONNECTIONS,
                      soft_limit=20,
                      hard_limit=50)

        # Async task limits
        self.set_limit(ResourceType.ASYNC_TASKS,
                      soft_limit=1000,
                      hard_limit=2000)

    def _initialize_memory_pools(self):
        """Initialize memory pools for common objects"""
        # Pool for dictionaries
        self.memory_pools["dict"] = MemoryPool(dict, max_size=1000)

        # Pool for lists
        self.memory_pools["list"] = MemoryPool(list, max_size=1000)

        # Pool for sets
        self.memory_pools["set"] = MemoryPool(set, max_size=500)

    async def initialize(self):
        """Initialize the resource manager"""
        if self.enable_memory_profiling:
            tracemalloc.start()
            logger.info("Memory profiling enabled")

        # Start monitoring tasks
        if self.monitoring_enabled:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Resource manager initialized")

    async def shutdown(self):
        """Shutdown the resource manager"""
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear all pools
        for pool in self.memory_pools.values():
            pool.clear()

        # Force garbage collection
        gc.collect()

        logger.info("Resource manager shut down")

    def set_limit(self, resource_type: ResourceType, soft_limit: int, hard_limit: int, **kwargs):
        """Set resource limits"""
        limit = ResourceLimit(
            resource_type=resource_type,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            **kwargs
        )
        self.limits[resource_type] = limit
        logger.info(f"Set {resource_type.value} limits: soft={soft_limit}, hard={hard_limit}")

    def register_cleanup_handler(self, resource_type: ResourceType, handler: Callable):
        """Register a cleanup handler for a resource type"""
        self.cleanup_handlers[resource_type].append(handler)
        logger.info(f"Registered cleanup handler for {resource_type.value}")

    @trace_async("allocate_resource")
    async def allocate_resource(
        self,
        resource_type: ResourceType,
        size_bytes: int,
        owner: str,
        allocation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Allocate a resource"""

        allocation_id = allocation_id or f"{resource_type.value}_{int(time.time() * 1000000)}"

        add_span_attributes({
            "resource.type": resource_type.value,
            "resource.size": size_bytes,
            "resource.owner": owner
        })

        # Check limits
        await self._check_limits(resource_type, size_bytes)

        # Create allocation record
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            resource_type=resource_type,
            allocated_at=datetime.utcnow(),
            size_bytes=size_bytes,
            owner=owner,
            metadata=metadata or {}
        )

        # Track the allocation
        self.tracker.track_allocation(allocation)

        logger.debug(f"Allocated {resource_type.value}: {allocation_id} ({size_bytes} bytes)")
        return allocation_id

    @trace_async("deallocate_resource")
    async def deallocate_resource(self, allocation_id: str):
        """Deallocate a resource"""

        add_span_attributes({"resource.allocation_id": allocation_id})

        allocation = self.tracker.track_deallocation(allocation_id)
        if allocation:
            # Run cleanup handlers
            handlers = self.cleanup_handlers.get(allocation.resource_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(allocation)
                    else:
                        handler(allocation)
                except Exception as e:
                    logger.error(f"Cleanup handler failed for {allocation_id}: {e}")

            logger.debug(f"Deallocated {allocation.resource_type.value}: {allocation_id}")
        else:
            logger.warning(f"Allocation not found for deallocation: {allocation_id}")

    def access_resource(self, allocation_id: str):
        """Record resource access"""
        self.tracker.update_access(allocation_id)

    async def _check_limits(self, resource_type: ResourceType, requested_size: int):
        """Check if allocation would exceed limits"""
        limit = self.limits.get(resource_type)
        if not limit:
            return  # No limits set

        current_usage = self.tracker.resource_usage[resource_type]
        new_usage = current_usage + requested_size

        # Check hard limit
        if new_usage > limit.hard_limit:
            if limit.auto_cleanup:
                # Try to free resources
                await self._emergency_cleanup(resource_type)
                current_usage = self.tracker.resource_usage[resource_type]
                new_usage = current_usage + requested_size

                if new_usage > limit.hard_limit:
                    raise ResourceExhaustedError(
                        f"{resource_type.value} hard limit exceeded: "
                        f"{new_usage} > {limit.hard_limit}"
                    )
            else:
                raise ResourceExhaustedError(
                    f"{resource_type.value} hard limit exceeded: "
                    f"{new_usage} > {limit.hard_limit}"
                )

        # Check soft limit warning
        if new_usage > limit.soft_limit * limit.warning_threshold:
            logger.warning(
                f"{resource_type.value} usage approaching limit: "
                f"{new_usage}/{limit.soft_limit} ({new_usage/limit.soft_limit*100:.1f}%)"
            )

            if limit.auto_cleanup:
                # Schedule cleanup
                asyncio.create_task(self._cleanup_resources(resource_type))

    async def _emergency_cleanup(self, resource_type: ResourceType):
        """Emergency cleanup when hard limits are hit"""
        logger.warning(f"Emergency cleanup triggered for {resource_type.value}")

        # Get stale allocations
        stale_allocations = self.tracker.get_stale_allocations(
            max_idle_time=timedelta(minutes=5)  # More aggressive for emergency
        )

        # Filter by resource type
        target_allocations = [a for a in stale_allocations if a.resource_type == resource_type]

        # Sort by last access time (oldest first)
        target_allocations.sort(key=lambda a: a.last_accessed or a.allocated_at)

        # Clean up oldest allocations
        cleanup_count = 0
        for allocation in target_allocations[:10]:  # Clean up to 10 allocations
            await self.deallocate_resource(allocation.allocation_id)
            cleanup_count += 1

        logger.info(f"Emergency cleanup freed {cleanup_count} {resource_type.value} allocations")

    async def _cleanup_resources(self, resource_type: Optional[ResourceType] = None):
        """Regular resource cleanup"""
        stale_allocations = self.tracker.get_stale_allocations(self.max_idle_time)

        if resource_type:
            stale_allocations = [a for a in stale_allocations if a.resource_type == resource_type]

        cleanup_count = 0
        for allocation in stale_allocations:
            try:
                await self.deallocate_resource(allocation.allocation_id)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Failed to cleanup allocation {allocation.allocation_id}: {e}")

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} stale resource allocations")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Collect system metrics
                system_stats = {
                    "memory_percent": self.process.memory_percent(),
                    "memory_info": self.process.memory_info()._asdict(),
                    "cpu_percent": self.process.cpu_percent(),
                    "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                    "num_threads": self.process.num_threads(),
                    "connections": len(self.process.connections()) if hasattr(self.process, 'connections') else 0
                }

                # Get resource usage
                resource_stats = self.tracker.get_usage_stats()

                # Combine stats
                monitoring_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": system_stats,
                    "resources": resource_stats,
                    "memory_pools": {
                        name: pool.get_stats()
                        for name, pool in self.memory_pools.items()
                    }
                }

                # Take memory snapshot if profiling enabled
                if self.enable_memory_profiling and tracemalloc.is_tracing():
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')[:10]

                    memory_profile = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "total_size": sum(stat.size for stat in top_stats),
                        "top_allocations": [
                            {
                                "filename": stat.traceback.format()[0],
                                "size_bytes": stat.size,
                                "count": stat.count
                            }
                            for stat in top_stats
                        ]
                    }

                    self.memory_snapshots.append(memory_profile)

                # Log warnings for high usage
                self._check_usage_warnings(system_stats, resource_stats)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())

                # Run regular cleanup
                await self._cleanup_resources()

                # Force garbage collection periodically
                gc.collect()

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def _check_usage_warnings(self, system_stats: Dict, resource_stats: Dict):
        """Check for usage warnings"""
        # Memory warning
        if system_stats["memory_percent"] > 85:
            logger.warning(f"High memory usage: {system_stats['memory_percent']:.1f}%")

        # File descriptor warning
        if system_stats["num_fds"] > 800:
            logger.warning(f"High file descriptor usage: {system_stats['num_fds']}")

        # Check resource limits
        for resource_type, usage in resource_stats["current_usage"].items():
            limit = self.limits.get(ResourceType(resource_type))
            if limit and usage > limit.soft_limit * limit.warning_threshold:
                logger.warning(
                    f"High {resource_type} usage: {usage}/{limit.soft_limit} "
                    f"({usage/limit.soft_limit*100:.1f}%)"
                )

    def get_memory_pool(self, pool_name: str) -> Optional[MemoryPool]:
        """Get a memory pool by name"""
        return self.memory_pools.get(pool_name)

    def create_memory_pool(self, name: str, factory: Callable[[], T], max_size: int = 100) -> MemoryPool[T]:
        """Create a new memory pool"""
        pool = MemoryPool(factory, max_size)
        self.memory_pools[name] = pool
        return pool

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics"""
        system_stats = {
            "memory_percent": self.process.memory_percent(),
            "memory_info": self.process.memory_info()._asdict(),
            "cpu_percent": self.process.cpu_percent(),
            "num_fds": self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            "num_threads": self.process.num_threads()
        }

        resource_stats = self.tracker.get_usage_stats()

        pool_stats = {
            name: pool.get_stats()
            for name, pool in self.memory_pools.items()
        }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_stats,
            "resources": resource_stats,
            "memory_pools": pool_stats,
            "limits": {
                rt.value: {
                    "soft_limit": limit.soft_limit,
                    "hard_limit": limit.hard_limit,
                    "warning_threshold": limit.warning_threshold,
                    "auto_cleanup": limit.auto_cleanup
                }
                for rt, limit in self.limits.items()
            }
        }

    def get_memory_profile(self) -> List[Dict[str, Any]]:
        """Get memory profiling data"""
        return list(self.memory_snapshots)


class ResourceExhaustedError(Exception):
    """Raised when resource limits are exceeded"""
    pass


# Global resource manager
_resource_manager = None


async def initialize_resource_manager() -> ResourceManager:
    """Initialize global resource manager"""
    global _resource_manager

    if _resource_manager is None:
        _resource_manager = ResourceManager()
        await _resource_manager.initialize()

    return _resource_manager


async def get_resource_manager() -> Optional[ResourceManager]:
    """Get the global resource manager"""
    return _resource_manager


async def shutdown_resource_manager():
    """Shutdown global resource manager"""
    global _resource_manager

    if _resource_manager:
        await _resource_manager.shutdown()
        _resource_manager = None


# Context manager for resource allocation
class ManagedResource:
    """Context manager for automatic resource cleanup"""

    def __init__(
        self,
        resource_type: ResourceType,
        size_bytes: int,
        owner: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.size_bytes = size_bytes
        self.owner = owner
        self.metadata = metadata
        self.allocation_id: Optional[str] = None

    async def __aenter__(self):
        manager = await get_resource_manager()
        if not manager:
            raise RuntimeError("Resource manager not initialized")

        self.allocation_id = await manager.allocate_resource(
            self.resource_type,
            self.size_bytes,
            self.owner,
            metadata=self.metadata
        )
        return self.allocation_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.allocation_id:
            manager = await get_resource_manager()
            if manager:
                await manager.deallocate_resource(self.allocation_id)


# Decorators for automatic resource management
def manage_memory(size_bytes: int, owner: str = None):
    """Decorator to automatically manage memory allocation"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            caller_owner = owner or func.__name__
            async with ManagedResource(ResourceType.MEMORY, size_bytes, caller_owner):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def track_resource_usage(resource_type: ResourceType, size_calculator: Callable = None):
    """Decorator to track resource usage of a function"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = await get_resource_manager()
            if not manager:
                return await func(*args, **kwargs)

            # Calculate size if calculator provided
            size = 1  # Default size
            if size_calculator:
                try:
                    size = size_calculator(*args, **kwargs)
                except Exception:
                    size = 1

            allocation_id = await manager.allocate_resource(
                resource_type,
                size,
                func.__name__
            )

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                await manager.deallocate_resource(allocation_id)

        return wrapper
    return decorator

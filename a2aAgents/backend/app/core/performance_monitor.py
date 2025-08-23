"""
Performance monitoring utilities
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge
import functools

logger = logging.getLogger(__name__)

# Prometheus metrics
request_duration = Histogram(
    'a2a_request_duration_seconds',
    'Request duration in seconds',
    ['agent_id', 'operation', 'status']
)

request_count = Counter(
    'a2a_request_total',
    'Total request count',
    ['agent_id', 'operation', 'status']
)

active_operations = Gauge(
    'a2a_active_operations',
    'Currently active operations',
    ['agent_id', 'operation']
)

cache_hits = Counter(
    'a2a_cache_hits_total',
    'Cache hit count',
    ['cache_name']
)

cache_misses = Counter(
    'a2a_cache_misses_total',
    'Cache miss count',
    ['cache_name']
)

db_query_duration = Histogram(
    'a2a_db_query_duration_seconds',
    'Database query duration',
    ['query_type', 'table']
)

memory_usage = Gauge(
    'a2a_memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark operation as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error


class PerformanceMonitor:
    """Central performance monitoring"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.slow_query_threshold = 1.0  # seconds
        self.memory_check_interval = 60  # seconds
        self._memory_monitor_task = None
        
    def start_operation(self, operation_id: str, operation_type: str, metadata: Dict[str, Any] = None) -> PerformanceMetrics:
        """Start monitoring an operation"""
        metric = PerformanceMetrics(
            operation=operation_type,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.metrics[operation_id] = metric
        
        # Update Prometheus metrics
        if 'agent_id' in metric.metadata:
            active_operations.labels(
                agent_id=metric.metadata['agent_id'],
                operation=operation_type
            ).inc()
            
        return metric
    
    def end_operation(self, operation_id: str, success: bool = True, error: Optional[str] = None):
        """End monitoring an operation"""
        if operation_id not in self.metrics:
            return
            
        metric = self.metrics[operation_id]
        metric.complete(success, error)
        
        # Update Prometheus metrics
        agent_id = metric.metadata.get('agent_id', 'unknown')
        status = 'success' if success else 'error'
        
        request_duration.labels(
            agent_id=agent_id,
            operation=metric.operation,
            status=status
        ).observe(metric.duration)
        
        request_count.labels(
            agent_id=agent_id,
            operation=metric.operation,
            status=status
        ).inc()
        
        if 'agent_id' in metric.metadata:
            active_operations.labels(
                agent_id=metric.metadata['agent_id'],
                operation=metric.operation
            ).dec()
        
        # Log slow operations
        if metric.duration > self.slow_query_threshold:
            logger.warning(
                f"Slow operation detected: {metric.operation} took {metric.duration:.2f}s",
                extra={'performance_metric': metric}
            )
        
        # Clean up
        del self.metrics[operation_id]
    
    async def monitor_memory(self):
        """Monitor memory usage periodically"""
        import psutil
        process = psutil.Process()
        
        while True:
            try:
                memory_info = process.memory_info()
                memory_usage.labels(component='rss').set(memory_info.rss)
                memory_usage.labels(component='vms').set(memory_info.vms)
                
                # Log if memory usage is high
                if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
                    logger.warning(f"High memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
            await asyncio.sleep(self.memory_check_interval)
    
    async def start_memory_monitoring(self):
        """Start memory monitoring task"""
        if not self._memory_monitor_task:
            self._memory_monitor_task = asyncio.create_task(self.monitor_memory())
    
    async def stop_memory_monitoring(self):
        """Stop memory monitoring task"""
        if self._memory_monitor_task:
            self._memory_monitor_task.cancel()
            try:
                await self._memory_monitor_task
            except asyncio.CancelledError:
                pass


# Global performance monitor
perf_monitor = PerformanceMonitor()


# Decorators for performance monitoring
def monitor_performance(operation_type: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation_id = f"{operation_type}_{time.time()}"
            metadata = {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Extract agent_id if available
            if args and hasattr(args[0], 'agent_id'):
                metadata['agent_id'] = args[0].agent_id
            
            perf_monitor.start_operation(operation_id, operation_type, metadata)
            
            try:
                result = await func(*args, **kwargs)
                perf_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                perf_monitor.end_operation(operation_id, success=False, error=str(e))
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            operation_id = f"{operation_type}_{time.time()}"
            metadata = {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Extract agent_id if available
            if args and hasattr(args[0], 'agent_id'):
                metadata['agent_id'] = args[0].agent_id
            
            perf_monitor.start_operation(operation_id, operation_type, metadata)
            
            try:
                result = func(*args, **kwargs)
                perf_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                perf_monitor.end_operation(operation_id, success=False, error=str(e))
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def monitor_db_query(query_type: str, table: str):
    """Decorator to monitor database queries"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                db_query_duration.labels(query_type=query_type, table=table).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start
                db_query_duration.labels(query_type=query_type, table=table).observe(duration)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                db_query_duration.labels(query_type=query_type, table=table).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start
                db_query_duration.labels(query_type=query_type, table=table).observe(duration)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
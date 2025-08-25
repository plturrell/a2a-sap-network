"""
Production-ready performance monitoring for A2A platform
Real-time performance tracking and optimization
"""

import time
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
from dataclasses import dataclass
import logging

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any]

class RealTimePerformanceMonitor:
    """Real-time performance monitoring for A2A agents"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = deque(maxlen=max_metrics)
        self.operation_stats = defaultdict(list)
        self.alerts_enabled = True
        self.thresholds = {
            'api_response_ms': 1000,
            'db_query_ms': 500,
            'agent_comm_ms': 2000,
            'memory_usage_mb': 500
        }
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, operation: str, duration_ms: float, success: bool = True, **metadata):
        """Record a performance metric"""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        self.operation_stats[operation].append(duration_ms)
        
        # Keep only recent stats for memory efficiency
        if len(self.operation_stats[operation]) > 1000:
            self.operation_stats[operation] = self.operation_stats[operation][-500:]
        
        # Check for performance alerts
        if self.alerts_enabled and duration_ms > self.thresholds.get(f'{operation}_ms', float('inf')):
            self._trigger_alert(operation, duration_ms)
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if operation:
            durations = self.operation_stats.get(operation, [])
            if not durations:
                return {}
            
            return {
                'operation': operation,
                'count': len(durations),
                'avg_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'p95_ms': sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                'p99_ms': sorted(durations)[int(len(durations) * 0.99)] if durations else 0
            }
        
        # Overall stats
        all_operations = {}
        for op in self.operation_stats:
            all_operations[op] = self.get_stats(op)
        
        return {
            'total_metrics': len(self.metrics),
            'operations': all_operations,
            'timespan_hours': (datetime.now() - self.metrics[0].timestamp).total_seconds() / 3600 if self.metrics else 0
        }
    
    def _trigger_alert(self, operation: str, duration_ms: float):
        """Trigger performance alert"""
        threshold = self.thresholds.get(f'{operation}_ms', 'N/A')
        self.logger.warning(
            f"Performance alert: {operation} took {duration_ms:.1f}ms (threshold: {threshold}ms)",
            extra={
                'operation': operation,
                'duration_ms': duration_ms,
                'threshold_ms': threshold,
                'alert_type': 'performance_threshold'
            }
        )

# Global monitor instance
_global_monitor = RealTimePerformanceMonitor()

def monitor_performance(operation: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _global_monitor.record_metric(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    function=func.__name__
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _global_monitor.record_metric(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    function=func.__name__
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    return _global_monitor.get_stats()

def export_performance_data(file_path: str):
    """Export performance data to JSON file"""
    stats = _global_monitor.get_stats()
    with open(file_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

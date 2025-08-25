"""
Performance Monitoring for SAP Compliance
Implements comprehensive performance tracking and optimization
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
from functools import wraps
from app.core.loggingConfig import get_logger, LogCategory

from opentelemetry import trace, metrics
from opentelemetry.metrics import CallbackOptions, Observation
import psutil

logger = get_logger(__name__, LogCategory.AGENT)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """SAP-compliant performance thresholds"""
    api_response_ms: float = 500.0  # 95th percentile
    db_query_ms: float = 100.0      # Average
    ui_load_ms: float = 3000.0      # Initial load
    agent_comm_ms: float = 1000.0   # 99th percentile


class PerformanceMonitor:
    """
    Enterprise-grade performance monitoring
    Meets SAP performance standards
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.thresholds = PerformanceThresholds()
        self.metrics_buffer: List[PerformanceMetric] = []
        self.buffer_size = 1000

        # Initialize OpenTelemetry
        self.tracer = trace.get_tracer(f"{service_name}.performance")
        self.meter = metrics.get_meter(f"{service_name}.performance")

        # Create metrics
        self._create_metrics()

        # Start background tasks
        self._start_monitoring_tasks()

    def _create_metrics(self):
        """Create OpenTelemetry metrics"""
        # Response time histogram
        self.response_time = self.meter.create_histogram(
            name="http_request_duration_ms",
            description="HTTP request duration in milliseconds",
            unit="ms"
        )

        # Database query time
        self.db_query_time = self.meter.create_histogram(
            name="db_query_duration_ms",
            description="Database query duration in milliseconds",
            unit="ms"
        )

        # Active requests gauge
        self.active_requests = self.meter.create_up_down_counter(
            name="active_requests",
            description="Number of active requests"
        )

        # Error rate
        self.error_counter = self.meter.create_counter(
            name="errors_total",
            description="Total number of errors"
        )

        # System metrics
        self.meter.create_observable_gauge(
            name="cpu_usage_percent",
            callbacks=[self._get_cpu_usage],
            description="CPU usage percentage"
        )

        self.meter.create_observable_gauge(
            name="memory_usage_mb",
            callbacks=[self._get_memory_usage],
            description="Memory usage in MB"
        )

    def _get_cpu_usage(self, options: CallbackOptions) -> List[Observation]:
        """Get current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return [Observation(cpu_percent, {"service": self.service_name})]

    def _get_memory_usage(self, options: CallbackOptions) -> List[Observation]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        return [Observation(memory_mb, {"service": self.service_name})]

    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Schedule periodic metric analysis
        asyncio.create_task(self._analyze_metrics_periodically())

    async def _analyze_metrics_periodically(self):
        """Analyze metrics and check thresholds"""
        while True:
            await asyncio.sleep(60)  # Every minute
            try:
                self._analyze_performance_metrics()
            except Exception as e:
                logger.error(f"Error analyzing metrics: {e}")

    def _analyze_performance_metrics(self):
        """Analyze buffered metrics against thresholds"""
        if not self.metrics_buffer:
            return

        # Group by operation
        operations: Dict[str, List[float]] = {}
        for metric in self.metrics_buffer:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration_ms)

        # Calculate statistics
        for operation, durations in operations.items():
            if durations:
                avg = statistics.mean(durations)
                p95 = sorted(durations)[int(len(durations) * 0.95)]
                p99 = sorted(durations)[int(len(durations) * 0.99)]

                # Check against thresholds
                if operation.startswith("api_"):
                    if p95 > self.thresholds.api_response_ms:
                        logger.warning(
                            f"API performance degradation: {operation} "
                            f"p95={p95:.2f}ms (threshold={self.thresholds.api_response_ms}ms)"
                        )

                elif operation.startswith("db_"):
                    if avg > self.thresholds.db_query_ms:
                        logger.warning(
                            f"Database performance degradation: {operation} "
                            f"avg={avg:.2f}ms (threshold={self.thresholds.db_query_ms}ms)"
                        )

        # Clear old metrics
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        self.metrics_buffer = [
            m for m in self.metrics_buffer
            if m.timestamp > cutoff_time
        ]

    def track_operation(self, operation: str):
        """Decorator to track operation performance"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                span = self.tracer.start_span(operation)
                self.active_requests.add(1, {"operation": operation})

                success = False
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    success = False
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    self.error_counter.add(1, {"operation": operation})
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.end()
                    self.active_requests.add(-1, {"operation": operation})

                    # Record metric
                    self._record_metric(operation, duration_ms, success)

                    # Update histograms
                    if operation.startswith("api_"):
                        self.response_time.record(
                            duration_ms,
                            {"endpoint": operation, "status": "success" if success else "error"}
                        )
                    elif operation.startswith("db_"):
                        self.db_query_time.record(
                            duration_ms,
                            {"query": operation, "status": "success" if success else "error"}
                        )

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                span = self.tracer.start_span(operation)
                self.active_requests.add(1, {"operation": operation})

                success = False
                try:
                    result = func(*args, **kwargs)
                    success = True
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    success = False
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(e))
                    )
                    self.error_counter.add(1, {"operation": operation})
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.end()
                    self.active_requests.add(-1, {"operation": operation})

                    # Record metric
                    self._record_metric(operation, duration_ms, success)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def _record_metric(self, operation: str, duration_ms: float, success: bool):
        """Record performance metric"""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            success=success
        )

        self.metrics_buffer.append(metric)

        # Maintain buffer size
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.buffer_size:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.metrics_buffer:
            return {"status": "no_data"}

        # Group by operation
        operations: Dict[str, List[PerformanceMetric]] = {}
        for metric in self.metrics_buffer:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "operations": {}
        }

        for operation, metrics in operations.items():
            durations = [m.duration_ms for m in metrics]
            success_count = sum(1 for m in metrics if m.success)

            report["operations"][operation] = {
                "count": len(metrics),
                "success_rate": success_count / len(metrics),
                "avg_ms": statistics.mean(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": statistics.median(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
                "p99_ms": sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0]
            }

        # Add system metrics
        report["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_percent": psutil.disk_usage('/').percent
        }

        return report

    async def check_health(self) -> Dict[str, Any]:
        """Check system health against thresholds"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        health["checks"]["cpu"] = {
            "value": cpu_percent,
            "threshold": 80,
            "status": "healthy" if cpu_percent < 80 else "warning"
        }

        # Check memory
        memory_percent = psutil.virtual_memory().percent
        health["checks"]["memory"] = {
            "value": memory_percent,
            "threshold": 85,
            "status": "healthy" if memory_percent < 85 else "warning"
        }

        # Check recent performance
        recent_metrics = [
            m for m in self.metrics_buffer
            if m.timestamp > datetime.utcnow() - timedelta(minutes=1)
        ]

        if recent_metrics:
            api_metrics = [m for m in recent_metrics if m.operation.startswith("api_")]
            if api_metrics:
                api_p95 = sorted([m.duration_ms for m in api_metrics])[int(len(api_metrics) * 0.95)]
                health["checks"]["api_performance"] = {
                    "value": api_p95,
                    "threshold": self.thresholds.api_response_ms,
                    "status": "healthy" if api_p95 < self.thresholds.api_response_ms else "degraded"
                }

        # Overall status
        if any(check["status"] != "healthy" for check in health["checks"].values()):
            health["status"] = "degraded"

        return health


# Singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(service_name: str = "a2a-platform") -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(service_name)
    return _performance_monitor
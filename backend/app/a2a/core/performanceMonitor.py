#!/usr/bin/env python3
"""
Enhanced Performance Monitoring for A2A Agents
Provides comprehensive performance metrics, alerting, and optimization recommendations
"""

import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import logging
from functools import wraps

# Monitoring libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, REGISTRY, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available")

try:
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available")

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""

    timestamp: str
    agent_id: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    response_time_avg: float
    response_time_p95: float
    error_rate: float
    throughput: float
    active_connections: int
    queue_size: int
    cache_hit_rate: float = 0.0
    network_latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertThresholds:
    """Alert threshold configuration"""

    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    response_time_threshold: float = 5000.0  # ms
    error_rate_threshold: float = 0.05  # 5%
    queue_size_threshold: int = 100


class PerformanceAlert:
    """Performance alert"""

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        value: float,
        threshold: float,
        agent_id: str,
    ):
        self.alert_type = alert_type
        self.severity = severity  # low, medium, high, critical
        self.message = message
        self.value = value
        self.threshold = threshold
        self.agent_id = agent_id
        self.timestamp = datetime.utcnow().isoformat()
        self.alert_id = f"{agent_id}_{alert_type}_{int(time.time())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.process = psutil.Process()
        self.start_time = time.time()

        # Metric tracking
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        self.cache_stats = {"hits": 0, "misses": 0}

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "uptime": time.time() - self.start_time,
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}

    def record_request(self, duration: float, success: bool = True):
        """Record a request with its duration and success status"""
        self.request_times.append(duration)
        self.request_count += 1
        if not success:
            self.error_count += 1

    def record_cache_hit(self, hit: bool = True):
        """Record cache hit/miss"""
        if hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.request_times:
            return {
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "error_rate": 0.0,
                "throughput": 0.0,
                "cache_hit_rate": 0.0,
            }

        sorted_times = sorted(self.request_times)
        avg_time = sum(sorted_times) / len(sorted_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]

        error_rate = self.error_count / max(self.request_count, 1)

        # Calculate throughput (requests per second over last minute)
        recent_requests = len([t for t in self.request_times if time.time() - t < 60])
        throughput = recent_requests / 60.0

        # Calculate cache hit rate
        total_cache = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / max(total_cache, 1)

        return {
            "avg_response_time": avg_time * 1000,  # Convert to ms
            "p95_response_time": p95_time * 1000,
            "error_rate": error_rate,
            "throughput": throughput,
            "cache_hit_rate": cache_hit_rate,
        }


class PrometheusMetrics:
    """Prometheus metrics manager"""

    def __init__(self, agent_id: str, registry: Optional = None):
        self.agent_id = agent_id
        self.registry = registry or REGISTRY

        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics will not be exported")
            return

        # Core metrics
        self.request_count = Counter(
            "a2a_agent_requests_total",
            "Total number of requests processed",
            ["agent_id", "method", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "a2a_agent_request_duration_seconds",
            "Request duration in seconds",
            ["agent_id", "method"],
            registry=self.registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self.active_connections = Gauge(
            "a2a_agent_active_connections",
            "Number of active connections",
            ["agent_id"],
            registry=self.registry,
        )

        self.queue_size = Gauge(
            "a2a_agent_queue_size", "Current queue size", ["agent_id"], registry=self.registry
        )

        self.cpu_usage = Gauge(
            "a2a_agent_cpu_usage_percent",
            "CPU usage percentage",
            ["agent_id"],
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "a2a_agent_memory_usage_percent",
            "Memory usage percentage",
            ["agent_id"],
            registry=self.registry,
        )

        self.error_rate = Gauge(
            "a2a_agent_error_rate", "Current error rate", ["agent_id"], registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            "a2a_agent_cache_hit_rate", "Cache hit rate", ["agent_id"], registry=self.registry
        )

        self.agent_info = Info(
            "a2a_agent_info", "Agent information", ["agent_id"], registry=self.registry
        )

        # Set agent info
        self.agent_info.labels(agent_id=self.agent_id).info(
            {"version": "1.0.0", "start_time": datetime.utcnow().isoformat()}
        )

    def update_metrics(self, metrics: PerformanceMetrics):
        """Update all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            agent_id = self.agent_id

            self.cpu_usage.labels(agent_id=agent_id).set(metrics.cpu_usage)
            self.memory_usage.labels(agent_id=agent_id).set(metrics.memory_usage)
            self.active_connections.labels(agent_id=agent_id).set(metrics.active_connections)
            self.queue_size.labels(agent_id=agent_id).set(metrics.queue_size)
            self.error_rate.labels(agent_id=agent_id).set(metrics.error_rate)
            self.cache_hit_rate.labels(agent_id=agent_id).set(metrics.cache_hit_rate)

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")

    def record_request(self, method: str, status: str, duration: float):
        """Record a request in Prometheus"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            self.request_count.labels(agent_id=self.agent_id, method=method, status=status).inc()

            self.request_duration.labels(agent_id=self.agent_id, method=method).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")


class PerformanceMonitor:
    """Main performance monitoring class"""

    def __init__(
        self,
        agent_id: str,
        alert_thresholds: Optional[AlertThresholds] = None,
        metrics_port: int = 8000,
        collection_interval: int = 30,
    ):

        self.agent_id = agent_id
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self.collection_interval = collection_interval

        # Components
        self.metrics_collector = MetricsCollector(agent_id)
        self.prometheus_metrics = PrometheusMetrics(agent_id)

        # State
        self.is_monitoring = False
        self.monitor_task = None
        self.alert_handlers = []
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1 min intervals

        # Start Prometheus server
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(metrics_port)
                logger.info(f"Prometheus metrics server started on port {metrics_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")

        logger.info(f"Performance monitor initialized for {agent_id}")

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Update Prometheus
                self.prometheus_metrics.update_metrics(metrics)

                # Check for alerts
                alerts = self._check_alerts(metrics)
                for alert in alerts:
                    await self._handle_alert(alert)

                # Log periodic summary
                if len(self.metrics_history) % 20 == 0:  # Every ~10 minutes
                    await self._log_performance_summary()

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        system_metrics = self.metrics_collector.get_system_metrics()
        perf_stats = self.metrics_collector.get_performance_stats()

        return PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat(),
            agent_id=self.agent_id,
            cpu_usage=system_metrics.get("cpu_usage", 0.0),
            memory_usage=system_metrics.get("memory_usage", 0.0),
            request_count=self.metrics_collector.request_count,
            response_time_avg=perf_stats.get("avg_response_time", 0.0),
            response_time_p95=perf_stats.get("p95_response_time", 0.0),
            error_rate=perf_stats.get("error_rate", 0.0),
            throughput=perf_stats.get("throughput", 0.0),
            active_connections=0,  # Will be updated by agent
            queue_size=0,  # Will be updated by agent
            cache_hit_rate=perf_stats.get("cache_hit_rate", 0.0),
        )

    def _check_alerts(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Check for alert conditions"""
        alerts = []

        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds.cpu_threshold:
            alerts.append(
                PerformanceAlert(
                    "high_cpu_usage",
                    "high" if metrics.cpu_usage > 90 else "medium",
                    f"CPU usage {metrics.cpu_usage:.1f}% exceeds threshold {self.alert_thresholds.cpu_threshold}%",
                    metrics.cpu_usage,
                    self.alert_thresholds.cpu_threshold,
                    self.agent_id,
                )
            )

        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds.memory_threshold:
            alerts.append(
                PerformanceAlert(
                    "high_memory_usage",
                    "high" if metrics.memory_usage > 95 else "medium",
                    f"Memory usage {metrics.memory_usage:.1f}% exceeds threshold {self.alert_thresholds.memory_threshold}%",
                    metrics.memory_usage,
                    self.alert_thresholds.memory_threshold,
                    self.agent_id,
                )
            )

        # Response time alert
        if metrics.response_time_p95 > self.alert_thresholds.response_time_threshold:
            alerts.append(
                PerformanceAlert(
                    "high_response_time",
                    "medium",
                    f"P95 response time {metrics.response_time_p95:.1f}ms exceeds threshold {self.alert_thresholds.response_time_threshold}ms",
                    metrics.response_time_p95,
                    self.alert_thresholds.response_time_threshold,
                    self.agent_id,
                )
            )

        # Error rate alert
        if metrics.error_rate > self.alert_thresholds.error_rate_threshold:
            alerts.append(
                PerformanceAlert(
                    "high_error_rate",
                    "high" if metrics.error_rate > 0.1 else "medium",
                    f"Error rate {metrics.error_rate:.2%} exceeds threshold {self.alert_thresholds.error_rate_threshold:.2%}",
                    metrics.error_rate,
                    self.alert_thresholds.error_rate_threshold,
                    self.agent_id,
                )
            )

        return alerts

    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alert"""
        logger.warning(f"Performance alert: {alert.message}")

        # Call registered alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    async def _log_performance_summary(self):
        """Log performance summary"""
        if not self.metrics_history:
            return

        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points

        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
        total_requests = sum(m.request_count for m in recent_metrics)

        logger.info(
            f"Performance Summary [{self.agent_id}]: "
            f"CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, "
            f"AvgResponseTime={avg_response_time:.1f}ms, "
            f"TotalRequests={total_requests}"
        )

    def record_request(
        self,
        method: str = "unknown",
        duration: float = 0.0,
        success: bool = True,
        status: str = "200",
    ):
        """Record a request for monitoring"""
        # Update internal collector
        self.metrics_collector.record_request(duration, success)

        # Update Prometheus
        self.prometheus_metrics.record_request(method, status, duration)

    def record_cache_operation(self, hit: bool = True):
        """Record cache hit/miss"""
        self.metrics_collector.record_cache_hit(hit)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self._collect_metrics()

    def get_metrics_history(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get metrics history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history if datetime.fromisoformat(m.timestamp) >= cutoff_time
        ]


# Performance monitoring decorator
def monitor_performance(method_name: str = None):
    """Decorator to monitor function performance"""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            status = "200"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                success = False
                status = "500"
                raise
            finally:
                duration = time.time() - start_time

                # Try to find performance monitor in self
                if args and hasattr(args[0], "_performance_monitor"):
                    monitor = args[0]._performance_monitor
                    monitor.record_request(
                        method=method_name or func.__name__,
                        duration=duration,
                        success=success,
                        status=status,
                    )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            status = "200"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                status = "500"
                raise
            finally:
                duration = time.time() - start_time

                # Try to find performance monitor in self
                if args and hasattr(args[0], "_performance_monitor"):
                    monitor = args[0]._performance_monitor
                    monitor.record_request(
                        method=method_name or func.__name__,
                        duration=duration,
                        success=success,
                        status=status,
                    )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global performance monitor registry
_performance_monitors: Dict[str, PerformanceMonitor] = {}


def get_performance_monitor(agent_id: str) -> Optional[PerformanceMonitor]:
    """Get performance monitor for agent"""
    return _performance_monitors.get(agent_id)


def create_performance_monitor(
    agent_id: str, alert_thresholds: Optional[AlertThresholds] = None, metrics_port: int = None
) -> PerformanceMonitor:
    """Create and register performance monitor"""
    if agent_id in _performance_monitors:
        logger.info(f"Performance monitor already exists for {agent_id}")
        return _performance_monitors[agent_id]

    # Assign unique metrics port if not specified
    if metrics_port is None:
        metrics_port = 8000 + len(_performance_monitors)

    monitor = PerformanceMonitor(
        agent_id=agent_id, alert_thresholds=alert_thresholds, metrics_port=metrics_port
    )

    _performance_monitors[agent_id] = monitor
    logger.info(f"Created performance monitor for {agent_id} on port {metrics_port}")

    return monitor

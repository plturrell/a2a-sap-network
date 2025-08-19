"""
Comprehensive Observability Stack for A2A Agents
Provides distributed tracing, metrics collection, log aggregation, and dashboards
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import threading
import queue
from collections import defaultdict, deque
import statistics

# Observability imports
import os
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.asyncio import AsyncIOInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    description: str
    metric_type: MetricType
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricPoint:
    """Individual metric data point"""
    metric_name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_query: str
    threshold: float
    comparison: str  # gt, lt, eq
    severity: AlertSeverity
    evaluation_window_minutes: int = 5
    notification_channels: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """High-performance metrics collector"""
    
    def __init__(self, export_interval: int = 30):
        self.metrics: Dict[str, MetricDefinition] = {}
        self.metric_data: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.export_interval = export_interval
        self.running = False
        self.export_task = None
        self.lock = threading.Lock()
        
        # Performance optimization: batch metric updates
        self.metric_buffer = defaultdict(deque)
        self.buffer_size = 1000
        
    def register_metric(self, metric: MetricDefinition):
        """Register a new metric"""
        self.metrics[metric.name] = metric
        logger.debug(f"Registered metric: {metric.name}")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: datetime = None
    ):
        """Record a metric value"""
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric: {metric_name}")
            return
        
        point = MetricPoint(
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            timestamp=timestamp or datetime.utcnow()
        )
        
        # Use lock-free approach for performance
        with self.lock:
            self.metric_buffer[metric_name].append(point)
            
            # Prevent memory growth
            if len(self.metric_buffer[metric_name]) > self.buffer_size:
                self.metric_buffer[metric_name].popleft()
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None, delta: float = 1.0):
        """Increment a counter metric"""
        self.record_metric(metric_name, delta, labels)
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        self.record_metric(metric_name, value, labels)
    
    def record_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram measurement"""
        self.record_metric(metric_name, value, labels)
    
    async def start_export(self):
        """Start metrics export loop"""
        if not self.running:
            self.running = True
            self.export_task = asyncio.create_task(self._export_loop())
    
    async def stop_export(self):
        """Stop metrics export loop"""
        self.running = False
        if self.export_task:
            self.export_task.cancel()
            try:
                await self.export_task
            except asyncio.CancelledError:
                pass
    
    async def _export_loop(self):
        """Background loop to export metrics"""
        while self.running:
            try:
                await self._export_metrics()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(self.export_interval)
    
    async def _export_metrics(self):
        """Export metrics to storage/external systems"""
        # Move buffered data to main storage
        with self.lock:
            for metric_name, points in self.metric_buffer.items():
                self.metric_data[metric_name].extend(points)
                points.clear()
                
                # Keep only recent data (last hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                self.metric_data[metric_name] = [
                    point for point in self.metric_data[metric_name]
                    if point.timestamp > cutoff
                ]
        
        # Here you would export to Prometheus, InfluxDB, etc.
        logger.debug(f"Exported metrics for {len(self.metric_data)} metric types")
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 15) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        if metric_name not in self.metric_data:
            return {}
        
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_points = [
            point.value for point in self.metric_data[metric_name]
            if point.timestamp > cutoff
        ]
        
        if not recent_points:
            return {"count": 0}
        
        return {
            "count": len(recent_points),
            "sum": sum(recent_points),
            "min": min(recent_points),
            "max": max(recent_points),
            "avg": statistics.mean(recent_points),
            "p50": statistics.median(recent_points),
            "p95": statistics.quantiles(recent_points, n=20)[18] if len(recent_points) > 20 else max(recent_points),
            "p99": statistics.quantiles(recent_points, n=100)[98] if len(recent_points) > 100 else max(recent_points)
        }


class DistributedTracer:
    """Distributed tracing implementation"""
    
    def __init__(self, service_name: str = "a2a-agent"):
        self.service_name = service_name
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: Dict[str, List[TraceSpan]] = {}
        self.tracer = None
        
        if OTEL_AVAILABLE:
            self._setup_opentelemetry()
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0"
        })
        
        # Setup tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.environ.get("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.environ.get("JAEGER_AGENT_PORT", "6831")),
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Auto-instrument common libraries
        AsyncIOInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Dict[str, str] = None
    ) -> str:
        """Start a new trace span"""
        span_id = str(uuid4())
        trace_id = str(uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            tags=tags or {}
        )
        
        self.active_spans[span_id] = span
        
        # If using OpenTelemetry, also start OTel span
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as otel_span:
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, value)
        
        return span_id
    
    def add_span_tags(self, span_id: str, tags: Dict[str, str]):
        """Add tags to a span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].tags.update(tags)
    
    def add_span_log(self, span_id: str, message: str, fields: Dict[str, Any] = None):
        """Add a log entry to a span"""
        if span_id in self.active_spans:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "message": message,
                "fields": fields or {}
            }
            self.active_spans[span_id].logs.append(log_entry)
    
    def finish_span(self, span_id: str, status: str = "ok"):
        """Finish a trace span"""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans.pop(span_id)
        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        # Store completed span
        if span.trace_id not in self.completed_traces:
            self.completed_traces[span.trace_id] = []
        
        self.completed_traces[span.trace_id].append(span)
        
        # Cleanup old traces
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for trace_id in list(self.completed_traces.keys()):
            trace_spans = self.completed_traces[trace_id]
            if all(s.end_time and s.end_time < cutoff for s in trace_spans):
                del self.completed_traces[trace_id]
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        return self.completed_traces.get(trace_id, [])
    
    def get_trace_summary(self, window_minutes: int = 15) -> Dict[str, Any]:
        """Get tracing summary statistics"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        recent_spans = []
        for trace_spans in self.completed_traces.values():
            for span in trace_spans:
                if span.end_time and span.end_time > cutoff:
                    recent_spans.append(span)
        
        if not recent_spans:
            return {"span_count": 0, "trace_count": 0}
        
        durations = [span.duration_ms for span in recent_spans if span.duration_ms]
        error_count = len([span for span in recent_spans if span.status == "error"])
        
        return {
            "span_count": len(recent_spans),
            "trace_count": len(set(span.trace_id for span in recent_spans)),
            "error_rate": error_count / len(recent_spans) if recent_spans else 0,
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "p95_duration_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else 0
        }


class StructuredLogger:
    """Structured logging with correlation"""
    
    def __init__(self, logger_name: str = "a2a"):
        self.logger_name = logger_name
        self.log_buffer = deque(maxlen=10000)
        self.structured_logger = None
        
        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()
        else:
            self.fallback_logger = logging.getLogger(logger_name)
    
    def _setup_structlog(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.structured_logger = structlog.get_logger(self.logger_name)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        agent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs
    ):
        """Log a structured message"""
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            logger_name=self.logger_name,
            message=message,
            agent_id=agent_id,
            trace_id=trace_id,
            span_id=span_id,
            extra_fields=kwargs
        )
        
        # Add to buffer for analysis
        self.log_buffer.append(log_entry)
        
        # Log using appropriate logger
        if self.structured_logger:
            getattr(self.structured_logger, level.value)(
                message,
                agent_id=agent_id,
                trace_id=trace_id,
                span_id=span_id,
                **kwargs
            )
        else:
            # Fallback to standard logging
            log_level = getattr(logging, level.value.upper())
            self.fallback_logger.log(log_level, f"{message} - {kwargs}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def get_log_analysis(self, window_minutes: int = 15) -> Dict[str, Any]:
        """Analyze recent logs"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_logs = [log for log in self.log_buffer if log.timestamp > cutoff]
        
        if not recent_logs:
            return {"total_logs": 0}
        
        level_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        
        for log in recent_logs:
            level_counts[log.level.value] += 1
            if log.agent_id:
                agent_counts[log.agent_id] += 1
        
        return {
            "total_logs": len(recent_logs),
            "by_level": dict(level_counts),
            "by_agent": dict(agent_counts),
            "error_rate": level_counts["error"] / len(recent_logs) if recent_logs else 0
        }


class AlertManager:
    """Alert management and notification"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = deque(maxlen=1000)
        self.evaluation_task = None
        self.running = False
    
    def register_alert_rule(self, rule: AlertRule):
        """Register an alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    async def start_evaluation(self):
        """Start alert evaluation loop"""
        if not self.running:
            self.running = True
            self.evaluation_task = asyncio.create_task(self._evaluation_loop())
    
    async def stop_evaluation(self):
        """Stop alert evaluation loop"""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
    
    async def _evaluation_loop(self):
        """Background loop to evaluate alert rules"""
        while self.running:
            try:
                for rule in self.alert_rules.values():
                    await self._evaluate_rule(rule)
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            # Simple metric query evaluation
            # In production, this would use a proper query engine
            metric_name = rule.metric_query
            summary = self.metrics_collector.get_metric_summary(
                metric_name, 
                rule.evaluation_window_minutes
            )
            
            if not summary.get("count", 0):
                return
            
            # Get the value to compare (could be avg, max, etc.)
            value = summary.get("avg", 0)
            
            # Evaluate threshold
            is_alerting = False
            if rule.comparison == "gt" and value > rule.threshold:
                is_alerting = True
            elif rule.comparison == "lt" and value < rule.threshold:
                is_alerting = True
            elif rule.comparison == "eq" and value == rule.threshold:
                is_alerting = True
            
            # Handle alert state changes
            if is_alerting and rule.rule_id not in self.active_alerts:
                # New alert
                await self._fire_alert(rule, value)
            elif not is_alerting and rule.rule_id in self.active_alerts:
                # Alert resolved
                await self._resolve_alert(rule, value)
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
    
    async def _fire_alert(self, rule: AlertRule, value: float):
        """Fire an alert"""
        self.active_alerts[rule.rule_id] = datetime.utcnow()
        
        alert_data = {
            "alert_id": str(uuid4()),
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "value": value,
            "threshold": rule.threshold,
            "fired_at": datetime.utcnow().isoformat(),
            "status": "firing"
        }
        
        self.alert_history.append(alert_data)
        
        logger.warning(f"ALERT FIRING: {rule.name} - Value: {value}, Threshold: {rule.threshold}")
        
        # Send notifications (implement actual notification logic)
        await self._send_notifications(rule, alert_data)
    
    async def _resolve_alert(self, rule: AlertRule, value: float):
        """Resolve an alert"""
        fired_at = self.active_alerts.pop(rule.rule_id)
        
        alert_data = {
            "alert_id": str(uuid4()),
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "value": value,
            "threshold": rule.threshold,
            "resolved_at": datetime.utcnow().isoformat(),
            "duration_minutes": (datetime.utcnow() - fired_at).total_seconds() / 60,
            "status": "resolved"
        }
        
        self.alert_history.append(alert_data)
        
        logger.info(f"ALERT RESOLVED: {rule.name} - Value: {value}")
    
    async def _send_notifications(self, rule: AlertRule, alert_data: Dict[str, Any]):
        """Send alert notifications"""
        # Implement actual notification logic (Slack, email, PagerDuty, etc.)
        for channel in rule.notification_channels:
            try:
                if channel.startswith("slack:"):
                    await self._send_slack_notification(channel, alert_data)
                elif channel.startswith("email:"):
                    await self._send_email_notification(channel, alert_data)
                # Add other notification methods
                
            except Exception as e:
                logger.error(f"Failed to send notification to {channel}: {e}")
    
    async def _send_slack_notification(self, channel: str, alert_data: Dict[str, Any]):
        """Send Slack notification"""
        # Implement Slack webhook notification
        logger.info(f"Would send Slack notification to {channel}: {alert_data['rule_name']}")
    
    async def _send_email_notification(self, channel: str, alert_data: Dict[str, Any]):
        """Send email notification"""
        # Implement email notification
        logger.info(f"Would send email notification to {channel}: {alert_data['rule_name']}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "recent_alerts": len([a for a in self.alert_history if 
                                datetime.fromisoformat(a.get("fired_at", "1970-01-01T00:00:00")) > 
                                datetime.utcnow() - timedelta(hours=24)]),
            "alerts_by_severity": {
                severity.value: len([a for a in self.alert_history if a.get("severity") == severity.value])
                for severity in AlertSeverity
            }
        }


class ObservabilityStack:
    """Main observability stack orchestrator"""
    
    def __init__(
        self,
        service_name: str = "a2a-agent",
        redis_config: RedisConfig = None
    ):
        self.service_name = service_name
        self.redis_client = RedisClient(redis_config or RedisConfig())
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer(service_name)
        self.logger = StructuredLogger(service_name)
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Dashboard data
        self.dashboard_data = {}
        self.dashboard_update_task = None
        
        # Initialize default metrics
        self._register_default_metrics()
        self._register_default_alerts()
    
    async def initialize(self):
        """Initialize observability stack"""
        await self.redis_client.initialize()
        
        # Start background tasks
        await self.metrics_collector.start_export()
        await self.alert_manager.start_evaluation()
        
        # Start dashboard updates
        self.dashboard_update_task = asyncio.create_task(self._update_dashboard_loop())
        
        logger.info("Observability stack initialized")
    
    async def shutdown(self):
        """Shutdown observability stack"""
        await self.metrics_collector.stop_export()
        await self.alert_manager.stop_evaluation()
        
        if self.dashboard_update_task:
            self.dashboard_update_task.cancel()
            try:
                await self.dashboard_update_task
            except asyncio.CancelledError:
                pass
        
        await self.redis_client.close()
        logger.info("Observability stack shut down")
    
    def _register_default_metrics(self):
        """Register default A2A metrics"""
        default_metrics = [
            MetricDefinition(
                name="a2a_agent_requests_total",
                description="Total number of agent requests",
                metric_type=MetricType.COUNTER,
                labels=["agent_id", "method", "status"]
            ),
            MetricDefinition(
                name="a2a_agent_request_duration_seconds",
                description="Agent request duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                unit="seconds",
                labels=["agent_id", "method"],
                buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            MetricDefinition(
                name="a2a_agent_active_connections",
                description="Number of active agent connections",
                metric_type=MetricType.GAUGE,
                labels=["agent_id"]
            ),
            MetricDefinition(
                name="a2a_tasks_total",
                description="Total number of tasks processed",
                metric_type=MetricType.COUNTER,
                labels=["agent_id", "task_type", "status"]
            ),
            MetricDefinition(
                name="a2a_messages_total",
                description="Total number of A2A messages",
                metric_type=MetricType.COUNTER,
                labels=["agent_id", "message_type", "status"]
            ),
            MetricDefinition(
                name="a2a_agent_cpu_usage_percent",
                description="Agent CPU usage percentage",
                metric_type=MetricType.GAUGE,
                unit="percent",
                labels=["agent_id"]
            ),
            MetricDefinition(
                name="a2a_agent_memory_usage_bytes",
                description="Agent memory usage in bytes",
                metric_type=MetricType.GAUGE,
                unit="bytes",
                labels=["agent_id"]
            ),
            MetricDefinition(
                name="a2a_queue_depth",
                description="Queue depth for agent processing",
                metric_type=MetricType.GAUGE,
                labels=["agent_id", "queue_type"]
            )
        ]
        
        for metric in default_metrics:
            self.metrics_collector.register_metric(metric)
    
    def _register_default_alerts(self):
        """Register default alert rules"""
        default_alerts = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Error rate exceeds 5%",
                metric_query="a2a_agent_requests_total",  # Simplified
                threshold=0.05,
                comparison="gt",
                severity=AlertSeverity.HIGH,
                evaluation_window_minutes=10,
                notification_channels=["slack:#alerts"]
            ),
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                description="Average response time exceeds 2 seconds",
                metric_query="a2a_agent_request_duration_seconds",
                threshold=2.0,
                comparison="gt",
                severity=AlertSeverity.MEDIUM,
                evaluation_window_minutes=5,
                notification_channels=["slack:#alerts"]
            ),
            AlertRule(
                rule_id="high_queue_depth",
                name="High Queue Depth",
                description="Queue depth exceeds 100 items",
                metric_query="a2a_queue_depth",
                threshold=100,
                comparison="gt",
                severity=AlertSeverity.MEDIUM,
                evaluation_window_minutes=5,
                notification_channels=["email:ops@company.com"]
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage exceeds 80%",
                metric_query="a2a_agent_memory_usage_bytes",
                threshold=0.8,
                comparison="gt",
                severity=AlertSeverity.HIGH,
                evaluation_window_minutes=10,
                notification_channels=["slack:#alerts", "email:ops@company.com"]
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.register_alert_rule(alert)
    
    async def _update_dashboard_loop(self):
        """Background loop to update dashboard data"""
        while True:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(30)
    
    async def _update_dashboard_data(self):
        """Update dashboard data"""
        self.dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_summary": {
                metric_name: self.metrics_collector.get_metric_summary(metric_name)
                for metric_name in self.metrics_collector.metrics.keys()
            },
            "trace_summary": self.tracer.get_trace_summary(),
            "log_analysis": self.logger.get_log_analysis(),
            "alert_status": self.alert_manager.get_alert_status(),
            "system_health": await self._get_system_health()
        }
        
        # Store in Redis for dashboard access
        await self.redis_client.set(
            f"dashboard_data:{self.service_name}",
            json.dumps(self.dashboard_data, default=str)
        )
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        # Collect system health indicators
        health_score = 100.0
        issues = []
        
        # Check error rates
        request_summary = self.metrics_collector.get_metric_summary("a2a_agent_requests_total")
        if request_summary.get("count", 0) > 0:
            error_rate = 0.1  # Simplified calculation
            if error_rate > 0.05:
                health_score -= 20
                issues.append(f"High error rate: {error_rate:.1%}")
        
        # Check response times
        duration_summary = self.metrics_collector.get_metric_summary("a2a_agent_request_duration_seconds")
        if duration_summary.get("avg", 0) > 2.0:
            health_score -= 15
            issues.append(f"High response time: {duration_summary.get('avg', 0):.2f}s")
        
        # Check active alerts
        active_alerts = len(self.alert_manager.active_alerts)
        if active_alerts > 0:
            health_score -= min(active_alerts * 10, 30)
            issues.append(f"{active_alerts} active alerts")
        
        # Determine health status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": max(0, health_score),
            "issues": issues,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data
    
    # Convenience methods for instrumentation
    def record_request(self, agent_id: str, method: str, status: str, duration: float):
        """Record a request metric"""
        self.metrics_collector.increment_counter(
            "a2a_agent_requests_total",
            {"agent_id": agent_id, "method": method, "status": status}
        )
        self.metrics_collector.record_histogram(
            "a2a_agent_request_duration_seconds",
            duration,
            {"agent_id": agent_id, "method": method}
        )
    
    def record_task(self, agent_id: str, task_type: str, status: str):
        """Record a task metric"""
        self.metrics_collector.increment_counter(
            "a2a_tasks_total",
            {"agent_id": agent_id, "task_type": task_type, "status": status}
        )
    
    def record_message(self, agent_id: str, message_type: str, status: str):
        """Record a message metric"""
        self.metrics_collector.increment_counter(
            "a2a_messages_total",
            {"agent_id": agent_id, "message_type": message_type, "status": status}
        )
    
    def set_resource_usage(self, agent_id: str, cpu_percent: float, memory_bytes: int):
        """Set resource usage metrics"""
        self.metrics_collector.set_gauge(
            "a2a_agent_cpu_usage_percent",
            cpu_percent,
            {"agent_id": agent_id}
        )
        self.metrics_collector.set_gauge(
            "a2a_agent_memory_usage_bytes",
            memory_bytes,
            {"agent_id": agent_id}
        )


# Global observability stack
_observability_stack = None


async def initialize_observability(
    service_name: str = "a2a-agent",
    redis_config: RedisConfig = None
) -> ObservabilityStack:
    """Initialize global observability stack"""
    global _observability_stack
    
    if _observability_stack is None:
        _observability_stack = ObservabilityStack(service_name, redis_config)
        await _observability_stack.initialize()
    
    return _observability_stack


async def get_observability_stack() -> Optional[ObservabilityStack]:
    """Get the global observability stack"""
    return _observability_stack


async def shutdown_observability():
    """Shutdown global observability stack"""
    global _observability_stack
    
    if _observability_stack:
        await _observability_stack.shutdown()
        _observability_stack = None


# Instrumentation decorators
def observe_performance(metric_name: str = None):
    """Decorator to observe function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or func.__name__
            
            observability = await get_observability_stack()
            span_id = None
            
            if observability:
                span_id = observability.tracer.start_span(
                    f"function:{function_name}",
                    tags={"function": function_name}
                )
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if observability:
                    observability.metrics_collector.record_histogram(
                        "function_duration_seconds",
                        duration,
                        {"function": function_name}
                    )
                    observability.tracer.finish_span(span_id, "ok")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if observability:
                    observability.metrics_collector.increment_counter(
                        "function_errors_total",
                        {"function": function_name, "error_type": type(e).__name__}
                    )
                    observability.tracer.add_span_log(span_id, f"Error: {str(e)}")
                    observability.tracer.finish_span(span_id, "error")
                
                raise e
        
        return wrapper
    return decorator
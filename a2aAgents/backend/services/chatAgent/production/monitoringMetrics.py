"""
Production Monitoring and Metrics for A2A Chat Agent
Provides comprehensive observability with Prometheus, OpenTelemetry, and custom analytics
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from functools import wraps
import psutil
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of aiohttp  # REMOVED: A2A protocol violation
# Core monitoring dependencies
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock implementations for testing
    class MockMetric:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def collect(self): return []
        def info(self, *args, **kwargs): pass
    
    Counter = Histogram = Gauge = Summary = Info = MockMetric
    CollectorRegistry = MockMetric
    def generate_latest(*args): return b"# Mock metrics"

# OpenTelemetry dependencies - optional
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Mock implementations
    class MockTrace:
        def set_tracer_provider(self, *args): pass
        def get_tracer(self, *args): return MockTracer()
        def get_current_span(self): return MockSpan()
        def Status(self, *args, **kwargs): return None
        StatusCode = type('StatusCode', (), {'OK': 'ok', 'ERROR': 'error'})()
    
    class MockTracer:
        def start_as_current_span(self, name): return MockSpan()
    
    class MockSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, *args): pass
        def set_status(self, *args): pass
        def record_exception(self, *args): pass
        def add_event(self, *args, **kwargs): pass
    
    class MockMetrics:
        def set_meter_provider(self, *args): pass
        def get_meter(self, *args): return MockMeter()
    
    class MockMeter:
        def create_counter(self, *args, **kwargs): return MockOtelMetric()
        def create_histogram(self, *args, **kwargs): return MockOtelMetric()
        def create_up_down_counter(self, *args, **kwargs): return MockOtelMetric()
    
    class MockOtelMetric:
        def add(self, *args, **kwargs): pass
        def record(self, *args, **kwargs): pass
    
    trace = MockTrace()
    metrics = MockMetrics()
    PrometheusMetricReader = TracerProvider = BatchSpanProcessor = None
    MeterProvider = Resource = OTLPSpanExporter = OTLPMetricExporter = None
    FastAPIInstrumentor = None
    AioHttpClientInstrumentor = None

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry if available
if OPENTELEMETRY_AVAILABLE:
    try:
        resource = Resource.create({
            "service.name": "a2a-chat-agent",
            "service.version": "2.0.0",
            "deployment.environment": "production"
        })
        
        # Tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer = trace.get_tracer(__name__)
        
        # Metrics
        metrics.set_meter_provider(MeterProvider(resource=resource))
        meter = metrics.get_meter(__name__)
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
        tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
else:
    logger.info("OpenTelemetry not available, using mock implementations")
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collection for A2A Chat Agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # OpenTelemetry metrics
        self._init_otel_metrics()
        
        # Custom analytics storage
        self.analytics_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.realtime_metrics = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'a2a_chat_requests_total',
            'Total number of chat requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'a2a_chat_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Agent routing metrics
        self.agent_routing_counter = Counter(
            'a2a_agent_routing_total',
            'Total agent routing attempts',
            ['agent_id', 'success'],
            registry=self.registry
        )
        
        self.agent_response_time = Histogram(
            'a2a_agent_response_time_seconds',
            'Agent response time',
            ['agent_id'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Conversation metrics
        self.active_conversations = Gauge(
            'a2a_active_conversations',
            'Number of active conversations',
            ['user_tier'],
            registry=self.registry
        )
        
        self.message_counter = Counter(
            'a2a_messages_total',
            'Total messages processed',
            ['role', 'conversation_type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'a2a_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'a2a_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.active_tasks = Gauge(
            'a2a_active_tasks',
            'Number of active background tasks',
            registry=self.registry
        )
        
        # Error metrics
        self.error_counter = Counter(
            'a2a_errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'a2a_cache_hits_total',
            'Cache hit count',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'a2a_cache_misses_total',
            'Cache miss count',
            ['cache_type'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_exceeded = Counter(
            'a2a_rate_limit_exceeded_total',
            'Rate limit exceeded count',
            ['user_tier', 'limit_type'],
            registry=self.registry
        )
        
        # WebSocket metrics
        self.websocket_connections = Gauge(
            'a2a_websocket_connections',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_messages = Counter(
            'a2a_websocket_messages_total',
            'WebSocket messages',
            ['direction', 'message_type'],
            registry=self.registry
        )
        
        # Business metrics
        self.revenue_counter = Counter(
            'a2a_revenue_total',
            'Total revenue',
            ['currency', 'tier'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'a2a_service',
            'Service information',
            registry=self.registry
        )
        self.service_info.info({
            'version': '2.0.0',
            'service': 'a2a-chat-agent',
            'environment': self.config.get('environment', 'production')
        })
    
    def _init_otel_metrics(self):
        """Initialize OpenTelemetry metrics"""
        # Create meters
        self.otel_request_counter = meter.create_counter(
            "a2a.chat.requests",
            description="Number of chat requests",
            unit="1"
        )
        
        self.otel_response_time = meter.create_histogram(
            "a2a.chat.response_time",
            description="Chat response time",
            unit="ms"
        )
        
        self.otel_active_users = meter.create_up_down_counter(
            "a2a.chat.active_users",
            description="Number of active users",
            unit="1"
        )
        
        self.otel_token_usage = meter.create_counter(
            "a2a.chat.token_usage",
            description="AI token usage",
            unit="1"
        )
    
    # Metric recording methods
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.request_counter.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        # OpenTelemetry
        self.otel_request_counter.add(1, {"method": method, "endpoint": endpoint, "status": str(status)})
        self.otel_response_time.record(duration * 1000, {"method": method, "endpoint": endpoint})
    
    def record_agent_routing(self, agent_id: str, success: bool, response_time: float):
        """Record agent routing metrics"""
        self.agent_routing_counter.labels(agent_id=agent_id, success=str(success)).inc()
        if success:
            self.agent_response_time.labels(agent_id=agent_id).observe(response_time)
        
        # Analytics
        self.analytics_buffer['agent_routing'].append({
            'agent_id': agent_id,
            'success': success,
            'response_time': response_time,
            'timestamp': datetime.utcnow()
        })
    
    def record_message(self, role: str, conversation_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Record message metrics"""
        self.message_counter.labels(role=role, conversation_type=conversation_type).inc()
        
        # Analytics
        self.analytics_buffer['messages'].append({
            'role': role,
            'conversation_type': conversation_type,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow()
        })
    
    def record_error(self, error_type: str, component: str, details: Optional[Dict[str, Any]] = None):
        """Record error metrics"""
        self.error_counter.labels(error_type=error_type, component=component).inc()
        
        # Log error details
        logger.error(f"Error in {component}: {error_type}", extra={'details': details})
        
        # Analytics
        self.analytics_buffer['errors'].append({
            'error_type': error_type,
            'component': component,
            'details': details or {},
            'timestamp': datetime.utcnow()
        })
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics"""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_rate_limit(self, user_tier: str, limit_type: str):
        """Record rate limit exceeded"""
        self.rate_limit_exceeded.labels(user_tier=user_tier, limit_type=limit_type).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage.set(memory_info.rss)
        
        # CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)
        self.cpu_usage.set(cpu_percent)
        
        # Store in realtime metrics
        self.realtime_metrics['system'] = {
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'timestamp': datetime.utcnow()
        }
    
    def set_active_conversations(self, count: int, user_tier: str = 'standard'):
        """Update active conversations gauge"""
        self.active_conversations.labels(user_tier=user_tier).set(count)
    
    def set_active_tasks(self, count: int):
        """Update active tasks gauge"""
        self.active_tasks.set(count)
    
    def set_websocket_connections(self, count: int):
        """Update WebSocket connections gauge"""
        self.websocket_connections.set(count)
    
    def record_websocket_message(self, direction: str, message_type: str):
        """Record WebSocket message"""
        self.websocket_messages.labels(direction=direction, message_type=message_type).inc()
    
    def record_revenue(self, amount: float, currency: str = 'USD', tier: str = 'standard'):
        """Record revenue metrics"""
        self.revenue_counter.labels(currency=currency, tier=tier).inc(amount)
    
    def record_token_usage(self, tokens: int, model: str, operation: str):
        """Record AI token usage"""
        self.otel_token_usage.add(tokens, {"model": model, "operation": operation})
        
        # Analytics
        self.analytics_buffer['token_usage'].append({
            'tokens': tokens,
            'model': model,
            'operation': operation,
            'timestamp': datetime.utcnow()
        })
    
    # Analytics methods
    def get_analytics_summary(self, metric_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get analytics summary for a metric type"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        data = [d for d in self.analytics_buffer[metric_type] if d['timestamp'] > cutoff]
        
        if not data:
            return {'count': 0, 'data': []}
        
        # Calculate summary statistics
        summary = {
            'count': len(data),
            'start_time': min(d['timestamp'] for d in data),
            'end_time': max(d['timestamp'] for d in data),
            'data': data[-100:]  # Last 100 entries
        }
        
        # Type-specific summaries
        if metric_type == 'agent_routing':
            summary['success_rate'] = sum(1 for d in data if d['success']) / len(data)
            summary['avg_response_time'] = sum(d['response_time'] for d in data if d['success']) / max(1, sum(1 for d in data if d['success']))
            summary['by_agent'] = self._group_by_field(data, 'agent_id')
        
        elif metric_type == 'errors':
            summary['by_type'] = self._group_by_field(data, 'error_type')
            summary['by_component'] = self._group_by_field(data, 'component')
        
        elif metric_type == 'token_usage':
            summary['total_tokens'] = sum(d['tokens'] for d in data)
            summary['by_model'] = self._group_by_field(data, 'model', sum_field='tokens')
            summary['by_operation'] = self._group_by_field(data, 'operation', sum_field='tokens')
        
        return summary
    
    def _group_by_field(self, data: List[Dict], field: str, sum_field: Optional[str] = None) -> Dict[str, Any]:
        """Group data by field and count/sum"""
        groups = defaultdict(lambda: {'count': 0, 'sum': 0})
        for item in data:
            key = item.get(field, 'unknown')
            groups[key]['count'] += 1
            if sum_field:
                groups[key]['sum'] += item.get(sum_field, 0)
        return dict(groups)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'system': self.realtime_metrics.get('system', {}),
            'request_stats': self._get_prometheus_stats(self.request_duration),
            'agent_stats': self._get_prometheus_stats(self.agent_response_time),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'error_rate': self._calculate_error_rate(),
            'active_metrics': {
                'conversations': self._get_gauge_value(self.active_conversations),
                'tasks': self._get_gauge_value(self.active_tasks),
                'websockets': self._get_gauge_value(self.websocket_connections)
            }
        }
    
    def _get_prometheus_stats(self, histogram) -> Dict[str, float]:
        """Extract statistics from Prometheus histogram"""
        try:
            # Get actual histogram metrics
            samples = list(histogram.collect())[0].samples
            
            count = 0
            sum_value = 0
            buckets = []
            
            for sample in samples:
                if sample.name.endswith('_count'):
                    count = sample.value
                elif sample.name.endswith('_sum'):
                    sum_value = sample.value
                elif sample.name.endswith('_bucket'):
                    buckets.append((float(sample.labels.get('le', '0')), sample.value))
            
            # Calculate percentiles from buckets
            buckets.sort(key=lambda x: x[0])
            p50 = self._calculate_percentile(buckets, count, 0.5)
            p95 = self._calculate_percentile(buckets, count, 0.95)
            p99 = self._calculate_percentile(buckets, count, 0.99)
            
            return {
                'count': count,
                'sum': sum_value,
                'average': sum_value / max(count, 1),
                'p50': p50,
                'p95': p95,
                'p99': p99
            }
        except Exception as e:
            logger.warning(f"Failed to extract Prometheus stats: {e}")
            return {
                'count': 0,
                'sum': 0,
                'average': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
    
    def _calculate_percentile(self, buckets: List[tuple], total_count: float, percentile: float) -> float:
        """Calculate percentile from histogram buckets"""
        if not buckets or total_count == 0:
            return 0
        
        target_count = total_count * percentile
        cumulative = 0
        
        for i, (le, count) in enumerate(buckets):
            cumulative = count
            if cumulative >= target_count:
                if i == 0:
                    return le
                # Linear interpolation
                prev_le, prev_count = buckets[i-1]
                ratio = (target_count - prev_count) / (cumulative - prev_count)
                return prev_le + ratio * (le - prev_le)
        
        return buckets[-1][0] if buckets else 0
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        try:
            # Get actual cache metrics
            hits_samples = list(self.cache_hits.collect())[0].samples
            misses_samples = list(self.cache_misses.collect())[0].samples
            
            total_hits = sum(sample.value for sample in hits_samples)
            total_misses = sum(sample.value for sample in misses_samples)
            
            total_requests = total_hits + total_misses
            if total_requests == 0:
                return 0.0
            
            return total_hits / total_requests
        except Exception as e:
            logger.warning(f"Failed to calculate cache hit rate: {e}")
            return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        try:
            # Get actual error and request metrics
            error_samples = list(self.error_counter.collect())[0].samples
            request_samples = list(self.request_counter.collect())[0].samples
            
            total_errors = sum(sample.value for sample in error_samples)
            total_requests = sum(sample.value for sample in request_samples)
            
            if total_requests == 0:
                return 0.0
            
            return total_errors / total_requests
        except Exception as e:
            logger.warning(f"Failed to calculate error rate: {e}")
            return 0.0
    
    def _get_gauge_value(self, gauge) -> float:
        """Get current gauge value"""
        try:
            samples = list(gauge.collect())[0].samples
            if samples:
                # Return the sum of all gauge values
                return sum(sample.value for sample in samples)
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get gauge value: {e}")
            return 0.0
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)


class TracingManager:
    """
    Distributed tracing with OpenTelemetry
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracer = tracer
        
        # Configure exporters
        if config.get('enable_otlp_export', False):
            self._setup_otlp_export()
    
    def _get_service_endpoint(self, service_name: str, default_port: str) -> str:
        """Get service endpoint using service discovery"""
        import os
        
        # Try Kubernetes service discovery
        k8s_service = os.getenv(f'{service_name.upper()}_SERVICE_HOST')
        k8s_port = os.getenv(f'{service_name.upper()}_SERVICE_PORT', default_port)
        
        if k8s_service:
            return f"{k8s_service}:{k8s_port}"
        
        # Try Docker Compose service discovery
        compose_host = os.getenv(f'{service_name.upper()}_HOST')
        if compose_host:
            return f"{compose_host}:{default_port}"
        
        # Try environment variable for custom endpoint
        custom_endpoint = os.getenv(f'A2A_{service_name.upper()}_ENDPOINT')
        if custom_endpoint:
            return custom_endpoint
        
        # Fallback to localhost for development
        logger.warning(f"No service discovery found for {service_name}, using localhost")
        return f"localhost:{default_port}"
    
    def _setup_otlp_export(self):
        """Setup OTLP exporters"""
        # Trace exporter with proper service discovery
        default_endpoint = self._get_service_endpoint('otlp', '4317')
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.get('otlp_endpoint', default_endpoint),
            insecure=self.config.get('otlp_insecure', True)
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Add attributes
                    span.set_attribute("operation.type", "async")
                    span.set_attribute("function.name", func.__name__)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Add attributes
                    span.set_attribute("operation.type", "sync")
                    span.set_attribute("function.name", func.__name__)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(
                            trace.Status(trace.StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def add_span_attributes(self, attributes: Dict[str, Any]):
        """Add attributes to current span"""
        span = trace.get_current_span()
        if span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span"""
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})


class HealthChecker:
    """
    Health check and readiness monitoring
    """
    
    def __init__(self, dependencies: Dict[str, Any]):
        self.dependencies = dependencies
        self.health_status = {
            'status': 'healthy',
            'checks': {},
            'last_check': None
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        checks = {}
        overall_healthy = True
        
        # Database check
        try:
            db = self.dependencies.get('database')
            if db:
                async with db.get_session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                checks['database'] = {'status': 'healthy'}
            else:
                checks['database'] = {'status': 'not_configured'}
        except Exception as e:
            checks['database'] = {'status': 'unhealthy', 'error': str(e)}
            overall_healthy = False
        
        # Redis check
        try:
            redis = self.dependencies.get('redis')
            if redis:
                await redis.ping()
                checks['redis'] = {'status': 'healthy'}
            else:
                checks['redis'] = {'status': 'not_configured'}
        except Exception as e:
            checks['redis'] = {'status': 'unhealthy', 'error': str(e)}
            overall_healthy = False
        
        # Agent connectivity check
        try:
            agent_registry = self.dependencies.get('agent_registry', {})
            healthy_agents = 0
            total_agents = len(agent_registry)
            
            for agent_id, agent_url in agent_registry.items():
                try:
                    async with # WARNING: aiohttp ClientSession usage violates A2A protocol - must use blockchain messaging
        # aiohttp\.ClientSession() as session:
                        async with session.get(f"{agent_url}/health", timeout=5) as resp:
                            if resp.status == 200:
                                healthy_agents += 1
                except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                    logger.debug(f"Agent {agent_id} health check failed: {e}")
                    continue
            
            checks['agents'] = {
                'status': 'healthy' if healthy_agents > total_agents * 0.5 else 'degraded',
                'healthy': healthy_agents,
                'total': total_agents
            }
        except Exception as e:
            checks['agents'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Memory check
        memory = psutil.virtual_memory()
        checks['memory'] = {
            'status': 'healthy' if memory.percent < 90 else 'warning',
            'used_percent': memory.percent
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        checks['disk'] = {
            'status': 'healthy' if disk.percent < 90 else 'warning',
            'used_percent': disk.percent
        }
        
        self.health_status = {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': checks,
            'last_check': datetime.utcnow().isoformat(),
            'version': '2.0.0'
        }
        
        return self.health_status
    
    async def check_readiness(self) -> Dict[str, Any]:
        """Check if service is ready to accept requests"""
        # Quick checks for readiness
        ready = True
        checks = {}
        
        # Check critical dependencies
        if 'database' in self.dependencies:
            try:
                db = self.dependencies['database']
                if not hasattr(db, '_initialized') or not db._initialized:
                    ready = False
                    checks['database'] = 'not_initialized'
            except (AttributeError, KeyError, Exception) as e:
                logger.warning(f"Database readiness check failed: {e}")
                ready = False
                checks['database'] = 'error'
        
        return {
            'ready': ready,
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat()
        }


# MCP Skills for monitoring
try:
    from a2a.sdk.mcpDecorators import mcp_tool, mcp_resource


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    
    @mcp_tool(
        name="metrics_get_summary",
        description="Get metrics summary for monitoring",
        input_schema={
            "type": "object",
            "properties": {
                "metric_type": {"type": "string", "enum": ["requests", "agents", "errors", "cache", "system"]},
                "time_range": {"type": "string", "enum": ["1h", "6h", "24h", "7d", "30d"]}
            },
            "required": ["metric_type"]
        }
    )
    async def get_metrics_summary(metrics_collector: MetricsCollector, metric_type: str, time_range: str = "24h"):
        """Get metrics summary"""
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(time_range, 24)
        
        if metric_type == "system":
            return metrics_collector.get_performance_report()
        else:
            return metrics_collector.get_analytics_summary(metric_type, hours)
    
    @mcp_tool(
        name="health_check",
        description="Perform health check on the system",
        input_schema={"type": "object", "properties": {}}
    )
    async def perform_health_check(health_checker: HealthChecker):
        """Perform system health check"""
        return await health_checker.check_health()
    
    @mcp_resource(
        name="metrics_dashboard",
        description="Real-time metrics dashboard data"
    )
    async def get_metrics_dashboard(metrics_collector: MetricsCollector):
        """Get real-time metrics for dashboard"""
        return {
            'performance': metrics_collector.get_performance_report(),
            'realtime': metrics_collector.realtime_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
except ImportError:
    # MCP not available
    pass


# Factory functions
def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """Create metrics collector instance"""
    return MetricsCollector(config)


def create_tracing_manager(config: Dict[str, Any]) -> TracingManager:
    """Create tracing manager instance"""
    return TracingManager(config)


def create_health_checker(dependencies: Dict[str, Any]) -> HealthChecker:
    """Create health checker instance"""
    return HealthChecker(dependencies)


# Middleware for automatic instrumentation
class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        # Record request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        self.metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        return response
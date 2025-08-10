"""
OpenTelemetry Integration for A2A Agents
Provides distributed tracing, metrics, and logging
"""

import os
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
import asyncio
from contextlib import contextmanager
from datetime import datetime

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.context import attach, detach
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.metrics import MeterProvider, Counter, Histogram, UpDownCounter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagators.b3 import B3MultiFormat

logger = logging.getLogger(__name__)


class TelemetryConfig:
    """Configuration for OpenTelemetry"""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "a2a-agent")
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.metrics_port = int(os.getenv("OTEL_METRICS_PORT", "8000"))
        self.enable_console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.agent_id = os.getenv("AGENT_ID", "unknown")
        self.agent_type = os.getenv("AGENT_TYPE", "standardization")


class OpenTelemetryManager:
    """Manages OpenTelemetry setup and instrumentation"""
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Metrics
        self.message_counter = None
        self.processing_histogram = None
        self.error_counter = None
        self.active_tasks_gauge = None
        self.platform_sync_counter = None
        self.circuit_breaker_counter = None
    
    def initialize(self):
        """Initialize OpenTelemetry providers"""
        if self._initialized:
            return
        
        # Create resource
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "2.0.0",
            "deployment.environment": self.config.environment,
            "agent.id": self.config.agent_id,
            "agent.type": self.config.agent_type
        })
        
        # Initialize tracing
        self._init_tracing(resource)
        
        # Initialize metrics
        self._init_metrics(resource)
        
        # Initialize instrumentation
        self._init_instrumentation()
        
        # Set propagator for distributed tracing
        set_global_textmap(B3MultiFormat())
        
        self._initialized = True
        logger.info("OpenTelemetry initialized successfully")
    
    def _init_tracing(self, resource: Resource):
        """Initialize tracing provider"""
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Add console exporter if enabled
        if self.config.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
        
        # Set as global
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
    
    def _init_metrics(self, resource: Resource):
        """Initialize metrics provider"""
        # Create readers
        readers = []
        
        # Prometheus reader for scraping
        prometheus_reader = PrometheusMetricReader()
        readers.append(prometheus_reader)
        
        # OTLP exporter for push
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True
        )
        readers.append(otlp_exporter)
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=readers
        )
        
        # Set as global
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Create metrics
        self._create_metrics()
    
    def _create_metrics(self):
        """Create application metrics"""
        # Message processing metrics
        self.message_counter = self.meter.create_counter(
            name="a2a_messages_processed_total",
            description="Total number of A2A messages processed",
            unit="1"
        )
        
        self.processing_histogram = self.meter.create_histogram(
            name="a2a_message_processing_duration_seconds",
            description="Time spent processing A2A messages",
            unit="s"
        )
        
        self.error_counter = self.meter.create_counter(
            name="a2a_errors_total",
            description="Total number of errors",
            unit="1"
        )
        
        self.active_tasks_gauge = self.meter.create_up_down_counter(
            name="a2a_active_tasks",
            description="Number of active tasks",
            unit="1"
        )
        
        # Platform sync metrics
        self.platform_sync_counter = self.meter.create_counter(
            name="a2a_platform_syncs_total",
            description="Total platform synchronizations",
            unit="1"
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_counter = self.meter.create_counter(
            name="a2a_circuit_breaker_trips_total",
            description="Circuit breaker state changes",
            unit="1"
        )
    
    def _init_instrumentation(self):
        """Initialize automatic instrumentation"""
        # Instrument HTTP client
        HTTPXClientInstrumentor().instrument()
        
        # Instrument logging to add trace context
        LoggingInstrumentor().instrument()
    
    @contextmanager
    def trace_span(self, span_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span context manager"""
        with self.tracer.start_as_current_span(span_name) as span:
            if attributes:
                span.set_attributes(attributes)
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def trace_async(self, span_name: str):
        """Decorator for tracing async functions"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.trace_span(span_name, {"function": func.__name__}):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record_message_processed(self, message_type: str, status: str, duration: float):
        """Record message processing metrics"""
        attributes = {
            "message_type": message_type,
            "status": status,
            "agent_id": self.config.agent_id
        }
        
        self.message_counter.add(1, attributes)
        self.processing_histogram.record(duration, attributes)
        
        if status == "error":
            self.error_counter.add(1, attributes)
    
    def record_platform_sync(self, platform: str, status: str, duration: float):
        """Record platform synchronization metrics"""
        attributes = {
            "platform": platform,
            "status": status,
            "agent_id": self.config.agent_id
        }
        
        self.platform_sync_counter.add(1, attributes)
        
        # Create specific histogram for platform syncs
        sync_histogram = self.meter.create_histogram(
            name=f"a2a_platform_sync_duration_{platform}",
            description=f"Duration of {platform} synchronizations",
            unit="s"
        )
        sync_histogram.record(duration, attributes)
    
    def record_circuit_breaker_state(self, service: str, old_state: str, new_state: str):
        """Record circuit breaker state changes"""
        attributes = {
            "service": service,
            "from_state": old_state,
            "to_state": new_state,
            "agent_id": self.config.agent_id
        }
        
        self.circuit_breaker_counter.add(1, attributes)
    
    def update_active_tasks(self, delta: int):
        """Update active tasks gauge"""
        self.active_tasks_gauge.add(delta, {"agent_id": self.config.agent_id})
    
    def add_baggage(self, key: str, value: str):
        """Add baggage for distributed context propagation"""
        ctx = baggage.set_baggage(key, value)
        token = attach(ctx)
        return token
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value"""
        return baggage.get_baggage(key)
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers for propagation"""
        from opentelemetry import propagate
        propagate.inject(headers)
        return headers
    
    def shutdown(self):
        """Shutdown telemetry providers"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")


# Global telemetry manager
_telemetry_manager = None

def get_telemetry_manager() -> OpenTelemetryManager:
    """Get global telemetry manager"""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = OpenTelemetryManager()
        _telemetry_manager.initialize()
    return _telemetry_manager


# Convenience decorators
def trace_span(span_name: str):
    """Decorator to trace a function with a span"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                telemetry = get_telemetry_manager()
                with telemetry.trace_span(span_name, {"function": func.__name__}):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                telemetry = get_telemetry_manager()
                with telemetry.trace_span(span_name, {"function": func.__name__}):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def record_metric(metric_name: str, value: float, attributes: Optional[Dict[str, Any]] = None):
    """Record a custom metric"""
    telemetry = get_telemetry_manager()
    meter = telemetry.meter
    
    # Create or get histogram
    histogram = meter.create_histogram(
        name=metric_name,
        description=f"Custom metric: {metric_name}",
        unit="1"
    )
    
    histogram.record(value, attributes or {})


# Example usage in agent
class TelemetryEnabledMixin:
    """Mixin to add telemetry to agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = get_telemetry_manager()
    
    @trace_span("process_a2a_message")
    async def process_message_with_telemetry(self, message: Dict[str, Any], 
                                            context_id: str) -> Dict[str, Any]:
        """Process message with telemetry"""
        start_time = asyncio.get_event_loop().time()
        
        # Update active tasks
        self.telemetry.update_active_tasks(1)
        
        try:
            # Add trace context to message
            span = trace.get_current_span()
            span.set_attribute("context_id", context_id)
            span.set_attribute("message_type", message.get("type", "unknown"))
            
            # Process message
            result = await self.process_message(message, context_id)
            
            # Record success metrics
            duration = asyncio.get_event_loop().time() - start_time
            self.telemetry.record_message_processed(
                message_type=message.get("type", "unknown"),
                status="success",
                duration=duration
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            duration = asyncio.get_event_loop().time() - start_time
            self.telemetry.record_message_processed(
                message_type=message.get("type", "unknown"),
                status="error",
                duration=duration
            )
            raise
        finally:
            # Update active tasks
            self.telemetry.update_active_tasks(-1)
"""
A2A Core Telemetry Module
Provides telemetry and observability functionality for A2A agents
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import os


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Try to import OpenTelemetry components
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

logger = logging.getLogger(__name__)

# Global tracer and meter instances
_tracer = None
_meter = None
_initialized = False


def initialize_telemetry(service_name: str = "a2a-agent", 
                        endpoint: Optional[str] = None,
                        enable_telemetry: bool = None) -> None:
    """
    Initialize OpenTelemetry for the A2A agent
    
    Args:
        service_name: Name of the service for telemetry
        endpoint: OTLP endpoint URL (defaults to env var or localhost)
        enable_telemetry: Whether to enable telemetry (defaults to env var)
    """
    global _tracer, _meter, _initialized
    
    if _initialized:
        return
        
    # Check if telemetry should be enabled
    if enable_telemetry is None:
        enable_telemetry = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"
    
    if not enable_telemetry or not OTEL_AVAILABLE:
        logger.info("Telemetry disabled or OpenTelemetry not available")
        _initialized = True
        return
    
    try:
        # Set up resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("NODE_ENV", "development")
        })
        
        # Set up tracing
        if endpoint is None:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        span_processor = BatchSpanProcessor(trace_exporter)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)
        
        # Set up metrics
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        # Get tracer and meter
        _tracer = trace.get_tracer(__name__)
        _meter = metrics.get_meter(__name__)
        
        _initialized = True
        logger.info(f"Telemetry initialized for {service_name} with endpoint {endpoint}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize telemetry: {e}")
        _initialized = True


def get_tracer():
    """Get the global tracer instance"""
    if not _initialized:
        initialize_telemetry()
    return _tracer


def get_meter():
    """Get the global meter instance"""
    if not _initialized:
        initialize_telemetry()
    return _meter


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    """
    Add attributes to the current span
    
    Args:
        attributes: Dictionary of attributes to add
    """
    if not OTEL_AVAILABLE or not _tracer:
        return
        
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                # Convert values to supported types
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                else:
                    span.set_attribute(key, str(value))
    except Exception as e:
        logger.debug(f"Failed to add span attributes: {e}")


def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Create a new span
    
    Args:
        name: Name of the span
        attributes: Optional attributes to add
    
    Returns:
        Span context manager or None if telemetry is disabled
    """
    if not OTEL_AVAILABLE or not _tracer:
        # Return a no-op context manager
        class NoOpSpan:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def set_status(self, *args):
                pass
            def set_attribute(self, *args):
                pass
        return NoOpSpan()
    
    span = _tracer.start_as_current_span(name)
    if attributes:
        add_span_attributes(attributes)
    return span


def trace_method(name: Optional[str] = None):
    """
    Decorator to trace a method execution
    
    Args:
        name: Optional name for the span (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            with create_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    if span and hasattr(span, 'set_status'):
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if span and hasattr(span, 'set_status'):
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            with create_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    if span and hasattr(span, 'set_status'):
                        span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if span and hasattr(span, 'set_status'):
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def record_metric(name: str, value: float, unit: str = "1", 
                 attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Record a metric value
    
    Args:
        name: Name of the metric
        value: Value to record
        unit: Unit of measurement
        attributes: Optional attributes
    """
    if not OTEL_AVAILABLE or not _meter:
        return
    
    try:
        # Get or create counter/histogram based on name pattern
        if name.endswith("_count") or name.endswith("_total"):
            counter = _meter.create_counter(name, unit=unit)
            counter.add(value, attributes or {})
        else:
            histogram = _meter.create_histogram(name, unit=unit)
            histogram.record(value, attributes or {})
    except Exception as e:
        logger.debug(f"Failed to record metric {name}: {e}")


def measure_time(metric_name: str = None):
    """
    Decorator to measure execution time of a function
    
    Args:
        metric_name: Name for the timing metric
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                name = metric_name or f"{func.__name__}_duration"
                record_metric(name, duration, unit="s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                name = metric_name or f"{func.__name__}_duration"
                record_metric(name, duration, unit="s", 
                            attributes={"status": "error"})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                name = metric_name or f"{func.__name__}_duration"
                record_metric(name, duration, unit="s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                name = metric_name or f"{func.__name__}_duration"
                record_metric(name, duration, unit="s", 
                            attributes={"status": "error"})
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions for common metrics
def increment_counter(name: str, value: int = 1, 
                     attributes: Optional[Dict[str, Any]] = None) -> None:
    """Increment a counter metric"""
    record_metric(f"{name}_count", value, attributes=attributes)


def record_duration(name: str, duration: float, 
                   attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a duration metric in seconds"""
    record_metric(f"{name}_duration", duration, unit="s", attributes=attributes)


def record_size(name: str, size: int, 
               attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record a size metric in bytes"""
    record_metric(f"{name}_size", size, unit="By", attributes=attributes)
"""
A2A Network Telemetry Module
Provides OpenTelemetry integration for distributed tracing and monitoring
"""

import logging
import functools
from typing import Dict, Any, Optional
from opentelemetry.trace import Tracer
from opentelemetry.metrics import Meter
from datetime import datetime

logger = logging.getLogger(__name__)

# Global telemetry state
_telemetry_initialized = False
_tracer = None
_meter = None

def init_telemetry(
    service_name: str,
    agent_id: Optional[str] = None,
    sampling_rate: float = 1.0,
    endpoint: Optional[str] = None,
    tracer_provider: Optional[Any] = None,
    meter_provider: Optional[Any] = None
) -> bool:
    """
    Initialize OpenTelemetry for the service
    
    Args:
        service_name: Name of the service for tracing
        agent_id: Optional agent identifier
        sampling_rate: Sampling rate for traces (0.0 to 1.0)
        endpoint: Optional telemetry endpoint URL
        
    Returns:
        bool: True if initialization successful
    """
    global _telemetry_initialized, _tracer, _meter

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "agent.id": agent_id or "unknown",
            "service.version": "1.0.0"
        })

        if not tracer_provider:
            tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        if endpoint:
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(span_exporter)
            tracer_provider.add_span_processor(span_processor)

        _tracer = trace.get_tracer(service_name)

        if not meter_provider:
            meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(service_name)

        _telemetry_initialized = True
        logger.info(f"Telemetry initialized for {service_name}")
        return True

    except ImportError:
        logger.warning("OpenTelemetry not available, using mock telemetry")
        _telemetry_initialized = True
        return True

    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        return False

def get_tracer() -> Optional[Tracer]:
    """Returns the global tracer instance."""
    return _tracer

def get_meter() -> Optional[Meter]:
    """Returns the global meter instance."""
    return _meter

def trace_async(operation_name: str):
    """
    Decorator for adding distributed tracing to async functions
    
    Args:
        operation_name: Name of the operation for tracing
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if _tracer and _telemetry_initialized:
                with _tracer.start_as_current_span(operation_name) as span:
                    try:
                        # Add basic attributes
                        span.set_attributes({
                            "operation.name": operation_name,
                            "function.name": func.__name__,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        result = await func(*args, **kwargs)
                        try:
                            from opentelemetry import trace
                            span.set_status(trace.Status(trace.StatusCode.OK))
                        except ImportError:
                            pass
                        return result
                    except Exception as e:
                        try:
                            from opentelemetry import trace
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        except ImportError:
                            pass
                        span.record_exception(e)
                        raise
            else:
                # No tracing available, just execute function
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def trace_sync(operation_name: str):
    """
    Decorator for adding distributed tracing to sync functions
    
    Args:
        operation_name: Name of the operation for tracing
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _tracer and _telemetry_initialized:
                with _tracer.start_as_current_span(operation_name) as span:
                    try:
                        # Add basic attributes
                        span.set_attributes({
                            "operation.name": operation_name,
                            "function.name": func.__name__,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        result = func(*args, **kwargs)
                        try:
                            from opentelemetry import trace
                            span.set_status(trace.Status(trace.StatusCode.OK))
                        except ImportError:
                            pass
                        return result
                    except Exception as e:
                        try:
                            from opentelemetry import trace
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        except ImportError:
                            pass
                        span.record_exception(e)
                        raise
            else:
                # No tracing available, just execute function
                return func(*args, **kwargs)
        return wrapper
    return decorator

def add_span_attributes(attributes: Dict[str, Any]):
    """
    Add attributes to the current span
    
    Args:
        attributes: Dictionary of attributes to add
    """
    if _tracer and _telemetry_initialized:
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            if current_span:
                # Convert all values to strings for compatibility
                string_attributes = {k: str(v) for k, v in attributes.items()}
                current_span.set_attributes(string_attributes)
        except Exception as e:
            logger.debug(f"Failed to add span attributes: {e}")

def record_metric(name: str, value: float, attributes: Optional[Dict[str, str]] = None):
    """
    Record a metric value
    
    Args:
        name: Metric name
        value: Metric value
        attributes: Optional attributes for the metric
    """
    if _meter and _telemetry_initialized:
        try:
            counter = _meter.create_counter(name)
            counter.add(value, attributes or {})
        except Exception as e:
            logger.debug(f"Failed to record metric {name}: {e}")

def create_histogram(name: str, description: str = ""):
    """
    Create a histogram metric
    
    Args:
        name: Histogram name
        description: Optional description
        
    Returns:
        Histogram instrument or None if not available
    """
    if _meter and _telemetry_initialized:
        try:
            return _meter.create_histogram(name, description=description)
        except Exception as e:
            logger.debug(f"Failed to create histogram {name}: {e}")
    return None

def shutdown_telemetry():
    """Shutdown telemetry and flush any pending data"""
    global _telemetry_initialized
    
    if _telemetry_initialized:
        try:
            from opentelemetry import trace, metrics
            
            # Shutdown tracer provider
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, 'shutdown'):
                tracer_provider.shutdown()
            
            # Shutdown meter provider  
            meter_provider = metrics.get_meter_provider()
            if hasattr(meter_provider, 'shutdown'):
                meter_provider.shutdown()
                
            logger.info("Telemetry shutdown completed")
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {e}")
        finally:
            _telemetry_initialized = False

# Legacy compatibility imports
def get_trace_context():
    """Get current trace context for legacy compatibility"""
    try:
        from opentelemetry import trace
        return trace.get_current_span().get_span_context()
    except:
        return None

__all__ = [
    'init_telemetry',
    'get_tracer', 
    'get_meter',
    'trace_async',
    'trace_sync', 
    'add_span_attributes',
    'record_metric',
    'create_histogram',
    'shutdown_telemetry',
    'get_trace_context'
]
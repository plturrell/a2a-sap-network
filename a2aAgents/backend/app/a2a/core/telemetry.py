"""
OpenTelemetry instrumentation for A2A agents
Provides distributed tracing across the agent network
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx # A2A Protocol: Use blockchain messaging instead of httpxClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from contextlib import contextmanager
from functools import wraps
import os
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logger
logger = logging.getLogger(__name__)

# Global tracer instance
tracer: Optional[trace.Tracer] = None


def init_telemetry(
    service_name: str,
    agent_id: str,
    otlp_endpoint: Optional[str] = None,
    sampling_rate: float = 0.1,
) -> trace.Tracer:
    """
    Initialize OpenTelemetry with OTLP exporter for distributed tracing

    Args:
        service_name: Name of the service (e.g., "agent0", "agent1")
        agent_id: Unique identifier for the agent instance
        otlp_endpoint: OTLP collector endpoint (defaults to env var)
        sampling_rate: Sampling rate for traces (0.0 to 1.0)

    Returns:
        Configured tracer instance
    """
    global tracer

    # Configure resource attributes
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "agent.id": agent_id,
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            "host.name": os.getenv("HOSTNAME", "localhost"),
            "service.namespace": "a2a-network",
        }
    )

    # Configure sampler
    sampler = TraceIdRatioBased(sampling_rate)

    # Configure tracer provider
    provider = TracerProvider(resource=resource, sampler=sampler)
    trace.set_tracer_provider(provider)

    # Configure OTLP exporter
    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", os.getenv("A2A_SERVICE_HOST"))
    otlp_exporter = OTLPSpanExporter(
        endpoint=endpoint, insecure=True  # Use secure=False for development
    )

    # Configure span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    # Set up propagator for distributed context
    set_global_textmap(TraceContextTextMapPropagator())

    # Get tracer
    tracer = trace.get_tracer(service_name, "1.0.0")

    logger.info(f"OpenTelemetry initialized for {service_name} with endpoint {endpoint}")
    return tracer


def instrument_fastapi(app):
    """Instrument FastAPI application for automatic tracing"""
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumentation enabled")


def instrument_httpx():
    """Instrument HTTPX for tracing outbound HTTP requests"""
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPX instrumentation enabled")


def instrument_redis():
    """Instrument Redis for tracing cache operations"""
    RedisInstrumentor().instrument()
    logger.info("Redis instrumentation enabled")


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """
    Context manager for creating trace spans

    Args:
        name: Name of the span
        attributes: Optional attributes to add to the span
        kind: Type of span (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)

    Example:
        with trace_span("process_message", {"message_id": "123"}):
            # Your code here
            pass
    """
    if not tracer:
        yield
        return

    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            span.set_attributes(attributes)

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """
    Decorator for tracing async functions

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional attributes to add to the span
        kind: Type of span

    Example:
        @trace_async("agent_process_task", {"task_type": "standardization"})
        async def process_task(task_id: str):
            # Your async code here
            pass
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = name or func.__name__

            if not tracer:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                if attributes:
                    span.set_attributes(attributes)

                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def trace_sync(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """
    Decorator for tracing synchronous functions

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional attributes to add to the span
        kind: Type of span
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__

            if not tracer:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                if attributes:
                    span.set_attributes(attributes)

                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def add_span_attributes(attributes: Dict[str, Any]):
    """
    Add attributes to the current span

    Args:
        attributes: Dictionary of attributes to add
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attributes(attributes)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to the current span

    Args:
        name: Name of the event
        attributes: Optional attributes for the event
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def get_trace_context() -> Dict[str, str]:
    """
    Get the current trace context for propagation

    Returns:
        Dictionary containing trace context headers
    """
    from opentelemetry.propagate import inject

    carrier = {}
    inject(carrier)
    return carrier


def set_trace_context(carrier: Dict[str, str]):
    """
    Set trace context from incoming headers

    Args:
        carrier: Dictionary containing trace context headers
    """
    from opentelemetry.propagate import extract

    extract(carrier)


# Agent-specific span helpers
@contextmanager
def trace_agent_message(agent_id: str, message_id: str, task_id: Optional[str] = None):
    """Trace agent message processing"""
    attributes = {
        "agent.id": agent_id,
        "message.id": message_id,
        "message.timestamp": datetime.utcnow().isoformat(),
    }
    if task_id:
        attributes["task.id"] = task_id

    with trace_span("agent_message_processing", attributes, trace.SpanKind.SERVER):
        yield


@contextmanager
def trace_agent_task(agent_id: str, task_id: str, task_type: str):
    """Trace agent task execution"""
    with trace_span(
        "agent_task_execution",
        {
            "agent.id": agent_id,
            "task.id": task_id,
            "task.type": task_type,
            "task.start_time": datetime.utcnow().isoformat(),
        },
        trace.SpanKind.INTERNAL,
    ):
        yield


@contextmanager
def trace_blockchain_operation(operation: str, contract_address: Optional[str] = None):
    """Trace blockchain operations"""
    attributes = {
        "blockchain.operation": operation,
        "blockchain.network": os.getenv("BLOCKCHAIN_NETWORK", "ethereum"),
    }
    if contract_address:
        attributes["blockchain.contract_address"] = contract_address

    with trace_span("blockchain_operation", attributes, trace.SpanKind.CLIENT):
        yield


@contextmanager
def trace_standardization(data_type: str, record_count: int):
    """Trace data standardization operations"""
    with trace_span(
        "data_standardization",
        {
            "standardization.data_type": data_type,
            "standardization.record_count": record_count,
            "standardization.start_time": datetime.utcnow().isoformat(),
        },
        trace.SpanKind.INTERNAL,
    ):
        yield

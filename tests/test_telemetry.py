import asyncio
import pytest
from unittest.mock import patch, MagicMock, call

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import a2aNetwork.core.telemetry as telemetry
from a2aNetwork.core.telemetry import init_telemetry, create_histogram, record_metric, trace_sync, trace_async, shutdown_telemetry

@pytest.fixture(autouse=True)
def reset_telemetry_module():
    """Fixture to reset the application's telemetry state before each test."""
    telemetry._telemetry_initialized = False
    telemetry._tracer = None
    telemetry._meter = None

@pytest.fixture
def in_memory_otel_providers():
    """Fixture to provide and manage the lifecycle of in-memory OpenTelemetry providers."""
    # Forcefully reset the global providers to ensure test isolation
    trace._TRACER_PROVIDER = None
    metrics._METER_PROVIDER = None

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    
    # Set the global providers for the test
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)

    yield span_exporter, metric_reader, tracer_provider, meter_provider
    
    # Shutdown providers after the test
    tracer_provider.shutdown()
    meter_provider.shutdown()

def test_init_telemetry_no_opentelemetry():
    """Test initialization when OpenTelemetry is not installed."""
    with patch('sys.modules', {'opentelemetry': None}):
        result = init_telemetry("test-service-no-otel")
        assert result is True
        assert telemetry._telemetry_initialized is True
        assert telemetry.get_tracer() is None
        assert telemetry.get_meter() is None

def test_init_telemetry_with_opentelemetry(in_memory_otel_providers):
    """Test telemetry initialization with OpenTelemetry installed."""
    span_exporter, metric_reader, tracer_provider, meter_provider = in_memory_otel_providers
    result = init_telemetry("test-service-with-otel", tracer_provider=tracer_provider, meter_provider=meter_provider)
    assert result is True
    assert telemetry._telemetry_initialized is True
    assert telemetry.get_tracer() is not None
    assert telemetry.get_meter() is not None

@pytest.mark.asyncio
async def test_trace_async_decorator(in_memory_otel_providers):
    """Test the async tracing decorator using in-memory exporter."""
    span_exporter, _, tracer_provider, _ = in_memory_otel_providers
    init_telemetry("test-async-service", tracer_provider=tracer_provider)

    @trace_async("test_async_operation")
    async def sample_async_func(x, y):
        return x + y

    result = await sample_async_func(2, 3)
    assert result == 5

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_async_operation"
    assert spans[0].status.status_code == trace.StatusCode.OK

def test_trace_sync_decorator_exception(in_memory_otel_providers):
    """Test the sync tracing decorator when an exception occurs."""
    span_exporter, _, tracer_provider, _ = in_memory_otel_providers
    init_telemetry("test-sync-fail-service", tracer_provider=tracer_provider)

    @trace_sync("test_sync_op_fail")
    def sample_sync_func_fail():
        raise ValueError("Sync Error")

    with pytest.raises(ValueError):
        sample_sync_func_fail()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "test_sync_op_fail"
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "ValueError: Sync Error" in span.events[0].name

def test_record_metric(in_memory_otel_providers):
    """Test that metrics are recorded correctly using in-memory reader."""
    _, metric_reader, _, meter_provider = in_memory_otel_providers
    init_telemetry("test-metric-service", meter_provider=meter_provider)

    record_metric("test.counter", 10, {"attr": "value"})

    metrics_data = metric_reader.get_metrics_data()
    assert len(metrics_data.resource_metrics) > 0

    scope_metrics = metrics_data.resource_metrics[0].scope_metrics[0]
    assert len(scope_metrics.metrics) > 0
    metric_data = scope_metrics.metrics[0]

    assert metric_data.name == "test.counter"
    point = list(metric_data.data.data_points)[0]
    assert point.value == 10
    assert point.attributes['attr'] == 'value'

def test_create_histogram(in_memory_otel_providers):
    """Test histogram creation and recording."""
    _, metric_reader, _, meter_provider = in_memory_otel_providers
    init_telemetry("test-histogram-service", meter_provider=meter_provider)

    histogram = create_histogram("test.histogram", "A test histogram")
    assert histogram is not None
    histogram.record(123.45, {"dim": "X"})

    metrics_data = metric_reader.get_metrics_data()
    assert len(metrics_data.resource_metrics) > 0

    scope_metrics = metrics_data.resource_metrics[0].scope_metrics[0]
    assert len(scope_metrics.metrics) > 0
    metric_data = scope_metrics.metrics[0]

    assert metric_data.name == "test.histogram"
    assert metric_data.description == "A test histogram"
    point = list(metric_data.data.data_points)[0]
    assert point.sum == 123.45
    assert point.attributes['dim'] == 'X'

def test_shutdown_telemetry(in_memory_otel_providers):
    """Test the shutdown sequence for initialized providers."""
    _, _, tracer_provider, meter_provider = in_memory_otel_providers
    init_telemetry("test-shutdown", tracer_provider=tracer_provider, meter_provider=meter_provider)
    assert telemetry._telemetry_initialized is True

    # Mock the shutdown methods to verify they are called
    tracer_provider.shutdown = MagicMock()
    meter_provider.shutdown = MagicMock()

    shutdown_telemetry()

    tracer_provider.shutdown.assert_called_once()
    meter_provider.shutdown.assert_called_once()
    assert telemetry._telemetry_initialized is False

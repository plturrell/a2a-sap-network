"""
A2A Core Module
Provides core functionality for A2A agents including telemetry, logging, and utilities
"""

from .telemetry import (
    initialize_telemetry,
    add_span_attributes,
    create_span,
    trace_method,
    record_metric,
    measure_time,
    increment_counter,
    record_duration,
    record_size,
    get_tracer,
    get_meter
)

__all__ = [
    'initialize_telemetry',
    'add_span_attributes',
    'create_span',
    'trace_method',
    'record_metric',
    'measure_time',
    'increment_counter',
    'record_duration',
    'record_size',
    'get_tracer',
    'get_meter'
]
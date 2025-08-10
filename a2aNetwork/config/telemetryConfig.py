"""
Telemetry Configuration for A2A Network
Provides configuration settings for OpenTelemetry integration
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TelemetryConfig:
    """Telemetry configuration class"""
    
    # OpenTelemetry settings
    otel_enabled: bool = True
    otel_service_name: str = "a2a-network"
    otel_service_version: str = "1.0.0"
    otel_environment: str = "development"
    
    # Tracing configuration
    otel_traces_exporter: str = "otlp"
    otel_traces_endpoint: Optional[str] = None
    otel_traces_headers: Optional[Dict[str, str]] = None
    otel_traces_sampler: str = "always_on"
    otel_traces_sampler_arg: float = 1.0
    
    # Metrics configuration
    otel_metrics_exporter: str = "otlp"
    otel_metrics_endpoint: Optional[str] = None
    otel_metrics_headers: Optional[Dict[str, str]] = None
    otel_metrics_interval: int = 60  # seconds
    
    # Logging configuration
    otel_logs_exporter: str = "otlp"
    otel_logs_endpoint: Optional[str] = None
    otel_logs_headers: Optional[Dict[str, str]] = None
    
    # Resource attributes
    resource_attributes: Dict[str, str] = None
    
    # Performance settings
    batch_span_processor_max_queue_size: int = 2048
    batch_span_processor_schedule_delay_millis: int = 5000
    batch_span_processor_export_timeout_millis: int = 30000
    batch_span_processor_max_export_batch_size: int = 512
    
    def __post_init__(self):
        """Initialize resource attributes if not provided"""
        if self.resource_attributes is None:
            self.resource_attributes = {
                "service.name": self.otel_service_name,
                "service.version": self.otel_service_version,
                "deployment.environment": self.otel_environment
            }

def load_telemetry_config() -> TelemetryConfig:
    """
    Load telemetry configuration from environment variables
    
    Returns:
        TelemetryConfig: Loaded configuration
    """
    config = TelemetryConfig()
    
    # Load from environment variables
    config.otel_enabled = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    config.otel_service_name = os.getenv("OTEL_SERVICE_NAME", config.otel_service_name)
    config.otel_service_version = os.getenv("OTEL_SERVICE_VERSION", config.otel_service_version)
    config.otel_environment = os.getenv("OTEL_ENVIRONMENT", config.otel_environment)
    
    # Tracing settings
    config.otel_traces_exporter = os.getenv("OTEL_TRACES_EXPORTER", config.otel_traces_exporter)
    config.otel_traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    config.otel_traces_sampler = os.getenv("OTEL_TRACES_SAMPLER", config.otel_traces_sampler)
    
    # Parse sampler argument
    sampler_arg = os.getenv("OTEL_TRACES_SAMPLER_ARG")
    if sampler_arg:
        try:
            config.otel_traces_sampler_arg = float(sampler_arg)
        except ValueError:
            config.otel_traces_sampler_arg = 1.0
    
    # Metrics settings
    config.otel_metrics_exporter = os.getenv("OTEL_METRICS_EXPORTER", config.otel_metrics_exporter)
    config.otel_metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    
    # Metrics interval
    metrics_interval = os.getenv("OTEL_METRICS_INTERVAL")
    if metrics_interval:
        try:
            config.otel_metrics_interval = int(metrics_interval)
        except ValueError:
            config.otel_metrics_interval = 60
    
    # Logging settings
    config.otel_logs_exporter = os.getenv("OTEL_LOGS_EXPORTER", config.otel_logs_exporter)
    config.otel_logs_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    
    # Headers parsing
    def parse_headers(header_str: Optional[str]) -> Optional[Dict[str, str]]:
        if not header_str:
            return None
        
        headers = {}
        for header in header_str.split(","):
            if "=" in header:
                key, value = header.strip().split("=", 1)
                headers[key] = value
        return headers if headers else None
    
    config.otel_traces_headers = parse_headers(os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS"))
    config.otel_metrics_headers = parse_headers(os.getenv("OTEL_EXPORTER_OTLP_METRICS_HEADERS"))
    config.otel_logs_headers = parse_headers(os.getenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS"))
    
    # Resource attributes from environment
    resource_attrs_env = os.getenv("OTEL_RESOURCE_ATTRIBUTES")
    if resource_attrs_env:
        attrs = parse_headers(resource_attrs_env)
        if attrs:
            config.resource_attributes.update(attrs)
    
    # Performance settings
    def get_int_env(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    config.batch_span_processor_max_queue_size = get_int_env(
        "OTEL_BSP_MAX_QUEUE_SIZE", 
        config.batch_span_processor_max_queue_size
    )
    config.batch_span_processor_schedule_delay_millis = get_int_env(
        "OTEL_BSP_SCHEDULE_DELAY", 
        config.batch_span_processor_schedule_delay_millis
    )
    config.batch_span_processor_export_timeout_millis = get_int_env(
        "OTEL_BSP_EXPORT_TIMEOUT", 
        config.batch_span_processor_export_timeout_millis
    )
    config.batch_span_processor_max_export_batch_size = get_int_env(
        "OTEL_BSP_MAX_EXPORT_BATCH_SIZE", 
        config.batch_span_processor_max_export_batch_size
    )
    
    return config

# Global configuration instance
telemetry_config = load_telemetry_config()

def update_telemetry_config(**kwargs):
    """
    Update telemetry configuration at runtime
    
    Args:
        **kwargs: Configuration parameters to update
    """
    global telemetry_config
    
    for key, value in kwargs.items():
        if hasattr(telemetry_config, key):
            setattr(telemetry_config, key, value)

def get_telemetry_config() -> TelemetryConfig:
    """
    Get current telemetry configuration
    
    Returns:
        TelemetryConfig: Current configuration
    """
    return telemetry_config

def is_telemetry_enabled() -> bool:
    """
    Check if telemetry is enabled
    
    Returns:
        bool: True if telemetry is enabled
    """
    return telemetry_config.otel_enabled

__all__ = [
    'TelemetryConfig',
    'telemetry_config',
    'load_telemetry_config',
    'update_telemetry_config',
    'get_telemetry_config',
    'is_telemetry_enabled'
]
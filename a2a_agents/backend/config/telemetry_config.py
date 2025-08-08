"""
OpenTelemetry configuration for A2A agents
"""

from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any


class TelemetryConfig(BaseSettings):
    """Configuration for OpenTelemetry instrumentation"""
    
    # Basic settings
    otel_enabled: bool = True
    otel_service_name: str = "a2a-agent"
    otel_service_version: str = "1.0.0"
    otel_environment: str = "development"
    
    # OTLP Exporter settings
    otel_exporter_otlp_endpoint: str = "localhost:4317"
    otel_exporter_otlp_headers: Optional[str] = None
    otel_exporter_otlp_timeout: int = 30
    otel_exporter_otlp_insecure: bool = True
    
    # Sampling configuration
    otel_traces_sampler: str = "traceidratio"
    otel_traces_sampler_arg: float = 0.1  # 10% sampling rate
    
    # Resource attributes
    otel_resource_attributes: Optional[str] = None
    
    # Instrumentation settings
    otel_python_logging_auto_instrumentation_enabled: bool = True
    otel_python_log_correlation: bool = True
    otel_python_log_level: str = "INFO"
    
    # Metrics configuration
    otel_metrics_enabled: bool = True
    otel_metrics_exporter: str = "otlp"
    otel_metric_export_interval: int = 60  # seconds
    
    # Propagators
    otel_propagators: str = "tracecontext,baggage"
    
    # Agent-specific settings
    agent_trace_db_operations: bool = True
    agent_trace_http_requests: bool = True
    agent_trace_message_processing: bool = True
    agent_trace_blockchain_operations: bool = True
    
    # Performance settings
    max_span_attributes: int = 128
    max_event_attributes: int = 128
    max_link_attributes: int = 128
    max_attribute_length: int = 1024
    
    class Config:
        env_prefix = "OTEL_"
        case_sensitive = False
        
    def get_resource_attributes(self) -> Dict[str, Any]:
        """Parse and return resource attributes"""
        if not self.otel_resource_attributes:
            return {}
        
        attributes = {}
        for pair in self.otel_resource_attributes.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attributes[key.strip()] = value.strip()
        
        return attributes
    
    def get_otlp_headers(self) -> Dict[str, str]:
        """Parse and return OTLP headers"""
        if not self.otel_exporter_otlp_headers:
            return {}
        
        headers = {}
        for pair in self.otel_exporter_otlp_headers.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers[key.strip()] = value.strip()
        
        return headers


# Global instance
telemetry_config = TelemetryConfig()
"""
Dynamic Configuration System for A2A Platform
Eliminates hardcoded values and provides secure configuration management
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from app.core.loggingConfig import get_logger, LogCategory
from pathlib import Path
from urllib.parse import urlparse
import secrets
import base64

logger = get_logger(__name__, LogCategory.AGENT)


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TEST = "test"


class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_timeout: int = 30
    ssl_mode: str = "prefer"
    connect_timeout: int = 10
    command_timeout: int = 30
    
    def __post_init__(self):
        if not self.url:
            raise ConfigValidationError("DATABASE_URL is required")
        
        # Validate URL format
        try:
            parsed = urlparse(self.url)
            if not parsed.scheme or not parsed.hostname:
                raise ConfigValidationError("Invalid DATABASE_URL format")
        except Exception as e:
            raise ConfigValidationError(f"Invalid DATABASE_URL: {e}")


@dataclass 
class RedisConfig:
    """Redis configuration"""
    url: str
    max_connections: int = 100
    timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    def __post_init__(self):
        if not self.url:
            raise ConfigValidationError("REDIS_URL is required")


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_expiry_days: int = 7
    password_min_length: int = 12
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    def __post_init__(self):
        if not self.jwt_secret:
            raise ConfigValidationError("JWT_SECRET is required")
        
        if len(self.jwt_secret) < 32:
            raise ConfigValidationError("JWT_SECRET must be at least 32 characters")


@dataclass
class CorsConfig:
    """CORS configuration"""
    origins: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization", "X-Requested-With"])
    credentials: bool = True
    max_age: int = 3600
    
    def __post_init__(self):
        # Parse origins from environment variable if string
        if isinstance(self.origins, str):
            self.origins = [origin.strip() for origin in self.origins.split(',') if origin.strip()]


@dataclass
class SAPConfig:
    """SAP integration configuration"""
    graph_url: str
    graph_client_id: str
    graph_client_secret: str
    graph_tenant_id: str
    graph_auth_url: str
    hana_url: str
    btp_subdomain: str
    cloud_sdk_service_url: str
    
    def __post_init__(self):
        required_fields = [
            'graph_client_id', 'graph_client_secret', 'graph_tenant_id',
            'hana_url'
        ]
        
        for field_name in required_fields:
            if not getattr(self, field_name):
                raise ConfigValidationError(f"SAP_{field_name.upper()} is required")


@dataclass
class AgentConfig:
    """Agent configuration"""
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 2
    max_concurrent_requests: int = 100
    health_check_interval: int = 30
    processing_batch_size: int = 10
    
    # Agent-specific timeouts
    agent0_timeout: int = 15
    agent1_timeout: int = 45
    agent2_timeout: int = 120  # AI processing takes longer
    agent3_timeout: int = 60
    agent4_timeout: int = 90   # Complex calculations
    agent5_timeout: int = 30


@dataclass
class ExternalServiceConfig:
    """External service configuration"""
    # AI Services
    grok_api_url: str = "https://api.x.ai/v1/chat/completions"
    grok_api_key: str = ""
    perplexity_api_url: str = "https://api.perplexity.ai"
    perplexity_api_key: str = ""
    
    # Market Data Services
    bloomberg_api_url: str = ""
    bloomberg_api_key: str = ""
    reuters_api_url: str = ""
    reuters_api_key: str = ""
    
    # Default timeouts and limits
    api_timeout_seconds: int = 30
    max_requests_per_minute: int = 60
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    
    # OpenTelemetry
    otel_endpoint: str = ""
    otel_service_name: str = "a2a-agents"
    otel_service_version: str = "1.0.0"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file_max_size: int = 100  # MB
    log_file_backup_count: int = 5
    
    # Metrics
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Health checks
    health_check_timeout: int = 5
    health_check_interval: int = 30


@dataclass
class NotificationConfig:
    """Notification system configuration"""
    page_size: int = 20
    max_notifications: int = 1000
    retention_days: int = 30
    batch_size: int = 100
    
    # Alert thresholds
    error_threshold: int = 10
    warning_threshold: int = 25
    performance_threshold_ms: int = 1000


@dataclass
class UIConfig:
    """UI/Frontend configuration"""
    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100
    
    # Timeouts
    api_timeout_ms: int = 30000
    chart_load_timeout_ms: int = 10000
    file_upload_timeout_ms: int = 300000
    
    # Limits
    max_file_size_mb: int = 50
    max_concurrent_uploads: int = 3
    
    # Feature flags
    enable_dark_mode: bool = True
    enable_analytics: bool = False
    enable_debug_mode: bool = False


class DynamicConfigManager:
    """
    Dynamic Configuration Manager
    Loads configuration from environment variables and config files
    """
    
    def __init__(self, env: Environment = None):
        self.env = env or self._detect_environment()
        self.config_cache = {}
        self._load_config()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_name = os.getenv('ENVIRONMENT', 'development').lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_config(self):
        """Load configuration from all sources"""
        # Load base configuration
        self._load_from_files()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_files(self):
        """Load configuration from YAML files"""
        config_dir = Path(__file__).parent.parent / "config"
        
        # Load base config
        base_config_path = config_dir / "base.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self.config_cache.update(yaml.safe_load(f) or {})
        
        # Load environment-specific config
        env_config_path = config_dir / f"{self.env.value}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_merge(self.config_cache, env_config)
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Database configuration
        self.config_cache.setdefault('database', {}).update({
            'url': os.getenv('DATABASE_URL'),
            'pool_min_size': int(os.getenv('DB_POOL_MIN_SIZE', '5')),
            'pool_max_size': int(os.getenv('DB_POOL_MAX_SIZE', '20')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'ssl_mode': os.getenv('DB_SSL_MODE', 'prefer'),
            'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
            'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
        })
        
        # Redis configuration
        self.config_cache.setdefault('redis', {}).update({
            'url': os.getenv('REDIS_URL'),
            'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '100')),
            'timeout': int(os.getenv('REDIS_TIMEOUT', '5')),
            'retry_on_timeout': os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true',
            'health_check_interval': int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '30'))
        })
        
        # Security configuration
        self.config_cache.setdefault('security', {}).update({
            'jwt_secret': os.getenv('JWT_SECRET'),
            'jwt_algorithm': os.getenv('JWT_ALGORITHM', 'HS256'),
            'jwt_expiry_hours': int(os.getenv('JWT_EXPIRY_HOURS', '24')),
            'jwt_refresh_expiry_days': int(os.getenv('JWT_REFRESH_EXPIRY_DAYS', '7')),
            'password_min_length': int(os.getenv('PASSWORD_MIN_LENGTH', '12')),
            'session_timeout_minutes': int(os.getenv('SESSION_TIMEOUT_MINUTES', '30')),
            'max_login_attempts': int(os.getenv('MAX_LOGIN_ATTEMPTS', '5')),
            'lockout_duration_minutes': int(os.getenv('LOCKOUT_DURATION_MINUTES', '15'))
        })
        
        # CORS configuration
        cors_origins = os.getenv('CORS_ORIGINS', '')
        self.config_cache.setdefault('cors', {}).update({
            'origins': [o.strip() for o in cors_origins.split(',') if o.strip()],
            'methods': os.getenv('CORS_METHODS', 'GET,POST,PUT,DELETE,OPTIONS').split(','),
            'headers': os.getenv('CORS_HEADERS', 'Content-Type,Authorization,X-Requested-With').split(','),
            'credentials': os.getenv('CORS_CREDENTIALS', 'true').lower() == 'true',
            'max_age': int(os.getenv('CORS_MAX_AGE', '3600'))
        })
        
        # SAP configuration
        self.config_cache.setdefault('sap', {}).update({
            'graph_url': os.getenv('SAP_GRAPH_URL'),
            'graph_client_id': os.getenv('SAP_GRAPH_CLIENT_ID'),
            'graph_client_secret': os.getenv('SAP_GRAPH_CLIENT_SECRET'),
            'graph_tenant_id': os.getenv('SAP_GRAPH_TENANT_ID'),
            'graph_auth_url': os.getenv('SAP_GRAPH_AUTH_URL'),
            'hana_url': os.getenv('SAP_HANA_URL'),
            'btp_subdomain': os.getenv('SAP_BTP_SUBDOMAIN'),
            'cloud_sdk_service_url': os.getenv('SAP_CLOUD_SDK_SERVICE_URL')
        })
        
        # Agent configuration
        self.config_cache.setdefault('agents', {}).update({
            'timeout_seconds': int(os.getenv('AGENT_TIMEOUT_SECONDS', '30')),
            'max_retries': int(os.getenv('AGENT_MAX_RETRIES', '3')),
            'retry_delay_seconds': int(os.getenv('AGENT_RETRY_DELAY', '2')),
            'max_concurrent_requests': int(os.getenv('AGENT_MAX_CONCURRENT', '100')),
            'health_check_interval': int(os.getenv('AGENT_HEALTH_CHECK_INTERVAL', '30')),
            'processing_batch_size': int(os.getenv('AGENT_BATCH_SIZE', '10')),
            'agent0_timeout': int(os.getenv('AGENT0_TIMEOUT', '15')),
            'agent1_timeout': int(os.getenv('AGENT1_TIMEOUT', '45')),
            'agent2_timeout': int(os.getenv('AGENT2_TIMEOUT', '120')),
            'agent3_timeout': int(os.getenv('AGENT3_TIMEOUT', '60')),
            'agent4_timeout': int(os.getenv('AGENT4_TIMEOUT', '90')),
            'agent5_timeout': int(os.getenv('AGENT5_TIMEOUT', '30'))
        })
        
        # External services
        self.config_cache.setdefault('external_services', {}).update({
            'grok_api_url': os.getenv('GROK_API_URL', 'https://api.x.ai/v1/chat/completions'),
            'grok_api_key': os.getenv('GROK_API_KEY', ''),
            'perplexity_api_url': os.getenv('PERPLEXITY_API_URL', 'https://api.perplexity.ai'),
            'perplexity_api_key': os.getenv('PERPLEXITY_API_KEY', ''),
            'bloomberg_api_url': os.getenv('BLOOMBERG_API_URL', ''),
            'bloomberg_api_key': os.getenv('BLOOMBERG_API_KEY', ''),
            'reuters_api_url': os.getenv('REUTERS_API_URL', ''),
            'reuters_api_key': os.getenv('REUTERS_API_KEY', ''),
            'api_timeout_seconds': int(os.getenv('EXTERNAL_API_TIMEOUT', '30')),
            'max_requests_per_minute': int(os.getenv('EXTERNAL_API_RATE_LIMIT', '60')),
            'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5')),
            'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', '60'))
        })
        
        # Monitoring
        self.config_cache.setdefault('monitoring', {}).update({
            'enable_metrics': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'enable_tracing': os.getenv('ENABLE_TRACING', 'true').lower() == 'true',
            'enable_logging': os.getenv('ENABLE_LOGGING', 'true').lower() == 'true',
            'otel_endpoint': os.getenv('OTEL_ENDPOINT', ''),
            'otel_service_name': os.getenv('OTEL_SERVICE_NAME', 'a2a-agents'),
            'otel_service_version': os.getenv('OTEL_SERVICE_VERSION', '1.0.0'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_format': os.getenv('LOG_FORMAT', 'json'),
            'log_file_max_size': int(os.getenv('LOG_FILE_MAX_SIZE', '100')),
            'log_file_backup_count': int(os.getenv('LOG_FILE_BACKUP_COUNT', '5')),
            'metrics_port': int(os.getenv('METRICS_PORT', '9090')),
            'metrics_path': os.getenv('METRICS_PATH', '/metrics'),
            'health_check_timeout': int(os.getenv('HEALTH_CHECK_TIMEOUT', '5')),
            'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
        })
        
        # Notifications
        self.config_cache.setdefault('notifications', {}).update({
            'page_size': int(os.getenv('NOTIFICATION_PAGE_SIZE', '20')),
            'max_notifications': int(os.getenv('NOTIFICATION_MAX_COUNT', '1000')),
            'retention_days': int(os.getenv('NOTIFICATION_RETENTION_DAYS', '30')),
            'batch_size': int(os.getenv('NOTIFICATION_BATCH_SIZE', '100')),
            'error_threshold': int(os.getenv('NOTIFICATION_ERROR_THRESHOLD', '10')),
            'warning_threshold': int(os.getenv('NOTIFICATION_WARNING_THRESHOLD', '25')),
            'performance_threshold_ms': int(os.getenv('NOTIFICATION_PERFORMANCE_THRESHOLD', '1000'))
        })
        
        # UI configuration
        self.config_cache.setdefault('ui', {}).update({
            'default_page_size': int(os.getenv('UI_DEFAULT_PAGE_SIZE', '20')),
            'max_page_size': int(os.getenv('UI_MAX_PAGE_SIZE', '100')),
            'api_timeout_ms': int(os.getenv('UI_API_TIMEOUT_MS', '30000')),
            'chart_load_timeout_ms': int(os.getenv('UI_CHART_TIMEOUT_MS', '10000')),
            'file_upload_timeout_ms': int(os.getenv('UI_UPLOAD_TIMEOUT_MS', '300000')),
            'max_file_size_mb': int(os.getenv('UI_MAX_FILE_SIZE_MB', '50')),
            'max_concurrent_uploads': int(os.getenv('UI_MAX_CONCURRENT_UPLOADS', '3')),
            'enable_dark_mode': os.getenv('UI_ENABLE_DARK_MODE', 'true').lower() == 'true',
            'enable_analytics': os.getenv('UI_ENABLE_ANALYTICS', 'false').lower() == 'true',
            'enable_debug_mode': os.getenv('UI_ENABLE_DEBUG', 'false').lower() == 'true'
        })
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self):
        """Validate the loaded configuration"""
        errors = []
        
        # Validate required configurations based on environment
        if self.env == Environment.PRODUCTION:
            # Production requires all secrets
            required_secrets = [
                ('database', 'url'),
                ('security', 'jwt_secret'),
                ('sap', 'graph_client_secret'),
                ('sap', 'hana_url')
            ]
            
            for section, key in required_secrets:
                if not self._get_nested_value(section, key):
                    errors.append(f"Production environment requires {section}.{key}")
        
        if errors:
            raise ConfigValidationError(f"Configuration validation errors: {errors}")
    
    def _get_nested_value(self, *keys):
        """Get nested dictionary value"""
        value = self.config_cache
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self.config_cache.get('database', {})
        return DatabaseConfig(**db_config)
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        redis_config = self.config_cache.get('redis', {})
        return RedisConfig(**redis_config)
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        security_config = self.config_cache.get('security', {})
        return SecurityConfig(**security_config)
    
    def get_cors_config(self) -> CorsConfig:
        """Get CORS configuration"""
        cors_config = self.config_cache.get('cors', {})
        return CorsConfig(**cors_config)
    
    def get_sap_config(self) -> SAPConfig:
        """Get SAP configuration"""
        sap_config = self.config_cache.get('sap', {})
        return SAPConfig(**sap_config)
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        agent_config = self.config_cache.get('agents', {})
        return AgentConfig(**agent_config)
    
    def get_external_service_config(self) -> ExternalServiceConfig:
        """Get external service configuration"""
        external_config = self.config_cache.get('external_services', {})
        return ExternalServiceConfig(**external_config)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        monitoring_config = self.config_cache.get('monitoring', {})
        return MonitoringConfig(**monitoring_config)
    
    def get_notification_config(self) -> NotificationConfig:
        """Get notification configuration"""
        notification_config = self.config_cache.get('notifications', {})
        return NotificationConfig(**notification_config)
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration"""
        ui_config = self.config_cache.get('ui', {})
        return UIConfig(**ui_config)
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.env == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.env == Environment.PRODUCTION
    
    def is_staging(self) -> bool:
        """Check if running in staging mode"""
        return self.env == Environment.STAGING


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> DynamicConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager()
    return _config_manager


def generate_secure_secret(length: int = 32) -> str:
    """Generate cryptographically secure secret"""
    return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode('utf-8')


def validate_production_config():
    """Validate production configuration on startup"""
    config_manager = get_config_manager()
    
    if config_manager.is_production():
        try:
            # Test all critical configurations
            config_manager.get_database_config()
            config_manager.get_security_config()
            config_manager.get_sap_config()
            
            logger.info("Production configuration validation successful")
            
        except ConfigValidationError as e:
            logger.error(f"Production configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in configuration validation: {e}")
            raise ConfigValidationError(f"Configuration validation failed: {e}")


# Configuration constants for use throughout the application
class Constants:
    """Application constants derived from configuration"""
    
    @classmethod
    def get_timeout_config(cls) -> Dict[str, int]:
        """Get timeout configuration for different operations"""
        agent_config = get_config_manager().get_agent_config()
        external_config = get_config_manager().get_external_service_config()
        ui_config = get_config_manager().get_ui_config()
        
        return {
            'AGENT_DEFAULT_TIMEOUT': agent_config.timeout_seconds,
            'AGENT0_TIMEOUT': agent_config.agent0_timeout,
            'AGENT1_TIMEOUT': agent_config.agent1_timeout,
            'AGENT2_TIMEOUT': agent_config.agent2_timeout,
            'AGENT3_TIMEOUT': agent_config.agent3_timeout,
            'AGENT4_TIMEOUT': agent_config.agent4_timeout,
            'AGENT5_TIMEOUT': agent_config.agent5_timeout,
            'EXTERNAL_API_TIMEOUT': external_config.api_timeout_seconds,
            'UI_API_TIMEOUT': ui_config.api_timeout_ms // 1000,  # Convert to seconds
            'DATABASE_CONNECT_TIMEOUT': get_config_manager().get_database_config().connect_timeout,
            'DATABASE_COMMAND_TIMEOUT': get_config_manager().get_database_config().command_timeout
        }
    
    @classmethod
    def get_pagination_config(cls) -> Dict[str, int]:
        """Get pagination configuration"""
        ui_config = get_config_manager().get_ui_config()
        notification_config = get_config_manager().get_notification_config()
        
        return {
            'DEFAULT_PAGE_SIZE': ui_config.default_page_size,
            'MAX_PAGE_SIZE': ui_config.max_page_size,
            'NOTIFICATION_PAGE_SIZE': notification_config.page_size,
            'AGENT_BATCH_SIZE': get_config_manager().get_agent_config().processing_batch_size
        }
    
    @classmethod 
    def get_limits_config(cls) -> Dict[str, int]:
        """Get various limits configuration"""
        agent_config = get_config_manager().get_agent_config()
        external_config = get_config_manager().get_external_service_config()
        ui_config = get_config_manager().get_ui_config()
        notification_config = get_config_manager().get_notification_config()
        
        return {
            'MAX_RETRIES': agent_config.max_retries,
            'MAX_CONCURRENT_REQUESTS': agent_config.max_concurrent_requests,
            'CIRCUIT_BREAKER_THRESHOLD': external_config.circuit_breaker_threshold,
            'MAX_FILE_SIZE_MB': ui_config.max_file_size_mb,
            'MAX_CONCURRENT_UPLOADS': ui_config.max_concurrent_uploads,
            'MAX_NOTIFICATIONS': notification_config.max_notifications,
            'RATE_LIMIT_PER_MINUTE': external_config.max_requests_per_minute
        }


if __name__ == "__main__":
    # Example usage and testing
    try:
        config_manager = DynamicConfigManager(Environment.DEVELOPMENT)
        
        print(f"Environment: {config_manager.env.value}")
        print(f"Database Config: {config_manager.get_database_config()}")
        print(f"Agent Config: {config_manager.get_agent_config()}")
        print(f"Timeout Config: {Constants.get_timeout_config()}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
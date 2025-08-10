"""
Application Constants - Dynamic Configuration Based
Replaces hardcoded magic numbers and strings throughout the application
"""

from typing import Dict, Any
from app.core.dynamic_config import get_config_manager


class AppConstants:
    """Application-wide constants derived from dynamic configuration"""
    
    @classmethod
    def get_timeout_constants(cls) -> Dict[str, int]:
        """Get all timeout-related constants"""
        config_manager = get_config_manager()
        agent_config = config_manager.get_agent_config()
        external_config = config_manager.get_external_service_config()
        ui_config = config_manager.get_ui_config()
        db_config = config_manager.get_database_config()
        
        return {
            # Agent timeouts
            "AGENT_DEFAULT_TIMEOUT": agent_config.timeout_seconds,
            "AGENT0_TIMEOUT": agent_config.agent0_timeout,
            "AGENT1_TIMEOUT": agent_config.agent1_timeout,
            "AGENT2_TIMEOUT": agent_config.agent2_timeout,
            "AGENT3_TIMEOUT": agent_config.agent3_timeout,
            "AGENT4_TIMEOUT": agent_config.agent4_timeout,
            "AGENT5_TIMEOUT": agent_config.agent5_timeout,
            
            # External API timeouts
            "EXTERNAL_API_TIMEOUT": external_config.api_timeout_seconds,
            "GROK_API_TIMEOUT": external_config.api_timeout_seconds,
            "PERPLEXITY_API_TIMEOUT": external_config.api_timeout_seconds,
            
            # UI timeouts (converted to seconds)
            "UI_API_TIMEOUT": ui_config.api_timeout_ms // 1000,
            "UI_CHART_TIMEOUT": ui_config.chart_load_timeout_ms // 1000,
            "UI_UPLOAD_TIMEOUT": ui_config.file_upload_timeout_ms // 1000,
            
            # Database timeouts
            "DB_CONNECT_TIMEOUT": db_config.connect_timeout,
            "DB_COMMAND_TIMEOUT": db_config.command_timeout,
            "DB_POOL_TIMEOUT": db_config.pool_timeout,
            
            # System timeouts
            "HEALTH_CHECK_TIMEOUT": config_manager.get_monitoring_config().health_check_timeout,
        }
    
    @classmethod 
    def get_pagination_constants(cls) -> Dict[str, int]:
        """Get pagination-related constants"""
        config_manager = get_config_manager()
        ui_config = config_manager.get_ui_config()
        notification_config = config_manager.get_notification_config()
        agent_config = config_manager.get_agent_config()
        
        return {
            "DEFAULT_PAGE_SIZE": ui_config.default_page_size,
            "MAX_PAGE_SIZE": ui_config.max_page_size,
            "NOTIFICATION_PAGE_SIZE": notification_config.page_size,
            "AGENT_BATCH_SIZE": agent_config.processing_batch_size,
            "NOTIFICATION_BATCH_SIZE": notification_config.batch_size,
        }
    
    @classmethod
    def get_limits_constants(cls) -> Dict[str, int]:
        """Get various system limits"""
        config_manager = get_config_manager()
        agent_config = config_manager.get_agent_config()
        external_config = config_manager.get_external_service_config()
        ui_config = config_manager.get_ui_config()
        notification_config = config_manager.get_notification_config()
        db_config = config_manager.get_database_config()
        redis_config = config_manager.get_redis_config()
        
        return {
            # Agent limits
            "MAX_RETRIES": agent_config.max_retries,
            "MAX_CONCURRENT_REQUESTS": agent_config.max_concurrent_requests,
            "RETRY_DELAY_SECONDS": agent_config.retry_delay_seconds,
            
            # External service limits
            "CIRCUIT_BREAKER_THRESHOLD": external_config.circuit_breaker_threshold,
            "CIRCUIT_BREAKER_TIMEOUT": external_config.circuit_breaker_timeout,
            "RATE_LIMIT_PER_MINUTE": external_config.max_requests_per_minute,
            
            # UI limits
            "MAX_FILE_SIZE_MB": ui_config.max_file_size_mb,
            "MAX_FILE_SIZE_BYTES": ui_config.max_file_size_mb * 1024 * 1024,
            "MAX_CONCURRENT_UPLOADS": ui_config.max_concurrent_uploads,
            
            # Notification limits
            "MAX_NOTIFICATIONS": notification_config.max_notifications,
            "NOTIFICATION_RETENTION_DAYS": notification_config.retention_days,
            
            # Database limits
            "DB_POOL_MIN_SIZE": db_config.pool_min_size,
            "DB_POOL_MAX_SIZE": db_config.pool_max_size,
            
            # Redis limits
            "REDIS_MAX_CONNECTIONS": redis_config.max_connections,
        }
    
    @classmethod
    def get_security_constants(cls) -> Dict[str, Any]:
        """Get security-related constants"""
        config_manager = get_config_manager()
        security_config = config_manager.get_security_config()
        
        return {
            "JWT_ALGORITHM": security_config.jwt_algorithm,
            "JWT_EXPIRY_HOURS": security_config.jwt_expiry_hours,
            "JWT_REFRESH_EXPIRY_DAYS": security_config.jwt_refresh_expiry_days,
            "PASSWORD_MIN_LENGTH": security_config.password_min_length,
            "SESSION_TIMEOUT_MINUTES": security_config.session_timeout_minutes,
            "MAX_LOGIN_ATTEMPTS": security_config.max_login_attempts,
            "LOCKOUT_DURATION_MINUTES": security_config.lockout_duration_minutes,
        }
    
    @classmethod
    def get_monitoring_constants(cls) -> Dict[str, Any]:
        """Get monitoring and observability constants"""
        config_manager = get_config_manager()
        monitoring_config = config_manager.get_monitoring_config()
        notification_config = config_manager.get_notification_config()
        
        return {
            "LOG_LEVEL": monitoring_config.log_level,
            "LOG_FORMAT": monitoring_config.log_format,
            "LOG_FILE_MAX_SIZE_MB": monitoring_config.log_file_max_size,
            "LOG_FILE_BACKUP_COUNT": monitoring_config.log_file_backup_count,
            "METRICS_PORT": monitoring_config.metrics_port,
            "METRICS_PATH": monitoring_config.metrics_path,
            "HEALTH_CHECK_INTERVAL": monitoring_config.health_check_interval,
            "OTEL_SERVICE_NAME": monitoring_config.otel_service_name,
            "OTEL_SERVICE_VERSION": monitoring_config.otel_service_version,
            
            # Alert thresholds
            "ERROR_THRESHOLD": notification_config.error_threshold,
            "WARNING_THRESHOLD": notification_config.warning_threshold,
            "PERFORMANCE_THRESHOLD_MS": notification_config.performance_threshold_ms,
        }


class HttpStatusCodes:
    """HTTP status code constants"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class AgentTypes:
    """Agent type constants"""
    AGENT0 = "agent0_data_product"
    AGENT1 = "agent1_standardization"  
    AGENT2 = "agent2_ai_preparation"
    AGENT3 = "agent3_vector_processing"
    AGENT4 = "agent4_calc_validation"
    AGENT5 = "agent5_qa_validation"
    
    DATA_MANAGER = "data_manager"
    CATALOG_MANAGER = "catalog_manager"
    AGENT_MANAGER = "agent_manager"
    AGENT_BUILDER = "agent_builder"


class MessageTypes:
    """Message type constants for agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    SHUTDOWN = "shutdown"


class ProcessingStatus:
    """Processing status constants"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DataFormats:
    """Data format constants"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    ORC = "orc"


class SAPEntityTypes:
    """SAP Graph entity type constants"""
    BUSINESS_PARTNER = "BusinessPartner"
    CUSTOMER = "Customer"
    SUPPLIER = "Supplier"
    COST_CENTER = "CostCenter"
    PROFIT_CENTER = "ProfitCenter"
    GL_ACCOUNT = "GLAccount"
    SALES_ORDER = "SalesOrder"
    PURCHASE_ORDER = "PurchaseOrder"


class ErrorCodes:
    """Application-specific error codes"""
    # Configuration errors
    CONFIG_MISSING = "CONFIG_001"
    CONFIG_INVALID = "CONFIG_002"
    CONFIG_VALIDATION_FAILED = "CONFIG_003"
    
    # Agent errors
    AGENT_TIMEOUT = "AGENT_001"
    AGENT_UNAVAILABLE = "AGENT_002"
    AGENT_PROCESSING_FAILED = "AGENT_003"
    AGENT_COMMUNICATION_FAILED = "AGENT_004"
    
    # Data errors
    DATA_VALIDATION_FAILED = "DATA_001"
    DATA_FORMAT_INVALID = "DATA_002"
    DATA_SIZE_EXCEEDED = "DATA_003"
    
    # Authentication/Authorization errors
    AUTH_TOKEN_INVALID = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"
    
    # External service errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXT_001"
    EXTERNAL_SERVICE_TIMEOUT = "EXT_002"
    EXTERNAL_SERVICE_RATE_LIMITED = "EXT_003"
    
    # Database errors
    DATABASE_CONNECTION_FAILED = "DB_001"
    DATABASE_QUERY_FAILED = "DB_002"
    DATABASE_TRANSACTION_FAILED = "DB_003"


# Convenience functions for common constant access
def get_timeout(timeout_type: str) -> int:
    """Get a specific timeout value"""
    timeouts = AppConstants.get_timeout_constants()
    return timeouts.get(timeout_type, 30)  # Default 30 seconds


def get_limit(limit_type: str) -> int:
    """Get a specific limit value"""
    limits = AppConstants.get_limits_constants()
    return limits.get(limit_type, 100)  # Default 100


def get_pagination_size(size_type: str = "DEFAULT_PAGE_SIZE") -> int:
    """Get a pagination size"""
    pagination = AppConstants.get_pagination_constants()
    return pagination.get(size_type, 20)  # Default 20


# Static constants that don't change based on environment
class StaticConstants:
    """Static constants that are the same across all environments"""
    
    # Protocol constants
    HTTP_PROTOCOL = "http"
    HTTPS_PROTOCOL = "https"
    WS_PROTOCOL = "ws"
    WSS_PROTOCOL = "wss"
    
    # Encoding constants
    UTF8_ENCODING = "utf-8"
    BASE64_ENCODING = "base64"
    
    # Content types
    CONTENT_TYPE_JSON = "application/json"
    CONTENT_TYPE_XML = "application/xml"
    CONTENT_TYPE_CSV = "text/csv"
    CONTENT_TYPE_TEXT = "text/plain"
    CONTENT_TYPE_FORM = "application/x-www-form-urlencoded"
    CONTENT_TYPE_MULTIPART = "multipart/form-data"
    
    # Date/time formats
    ISO_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"
    
    # Regex patterns
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    UUID_REGEX = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    
    # Currency codes (ISO 4217)
    CURRENCY_USD = "USD"
    CURRENCY_EUR = "EUR" 
    CURRENCY_GBP = "GBP"
    CURRENCY_JPY = "JPY"
    CURRENCY_CHF = "CHF"
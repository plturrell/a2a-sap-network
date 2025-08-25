"""
A2A Platform Standardized Logging Configuration
Enterprise-grade logging with structured output, correlation IDs, and observability
"""

import logging
import logging.config
import sys
import json
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum
from contextvars import ContextVar
import asyncio
from functools import wraps
import time

# Context variables for correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
agent_id: ContextVar[Optional[str]] = ContextVar('agent_id', default=None)


class LogLevel(str, Enum):
    """Standardized log levels"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class LogCategory(str, Enum):
    """Log categories for classification"""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    INTEGRATION = "integration"
    AGENT = "agent"
    DATABASE = "database"
    API = "api"
    AUDIT = "audit"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""

        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }

        # Add context variables if available
        if correlation_id.get():
            log_entry["correlation_id"] = correlation_id.get()
        if request_id.get():
            log_entry["request_id"] = request_id.get()
        if user_id.get():
            log_entry["user_id"] = user_id.get()
        if agent_id.get():
            log_entry["agent_id"] = agent_id.get()

        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields from record
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                    'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info', 'message'
                }:
                    extra_fields[key] = value

            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, separators=(',', ':'))


class A2ALogger:
    """Enhanced logger with standardized patterns and context management"""

    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.logger = logging.getLogger(name)
        self.category = category
        self.name = name

    def _log(
        self,
        level: LogLevel,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """Internal logging method with standardized structure"""

        # Prepare extra data
        log_extra = {
            "category": (category or self.category).value,
            **(extra or {}),
            **kwargs
        }

        # Log with appropriate level
        getattr(self.logger, level.value.lower())(
            message,
            extra=log_extra,
            exc_info=exc_info
        )

    def critical(
        self,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs
    ):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, category, extra, exc_info, **kwargs)

    def error(
        self,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs
    ):
        """Log error message"""
        self._log(LogLevel.ERROR, message, category, extra, exc_info, **kwargs)

    def warning(
        self,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, category, extra, False, **kwargs)

    def info(
        self,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log info message"""
        self._log(LogLevel.INFO, message, category, extra, False, **kwargs)

    def debug(
        self,
        message: str,
        category: Optional[LogCategory] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, category, extra, False, **kwargs)

    # Convenience methods for common patterns
    def start_operation(self, operation: str, **context):
        """Log operation start"""
        self.info(
            f"Starting operation: {operation}",
            category=LogCategory.BUSINESS,
            operation=operation,
            operation_status="started",
            **context
        )

    def complete_operation(self, operation: str, duration: float = None, **context):
        """Log operation completion"""
        self.info(
            f"Completed operation: {operation}",
            category=LogCategory.BUSINESS,
            operation=operation,
            operation_status="completed",
            duration_seconds=duration,
            **context
        )

    def fail_operation(self, operation: str, error: Exception = None, **context):
        """Log operation failure"""
        self.error(
            f"Failed operation: {operation}",
            category=LogCategory.BUSINESS,
            operation=operation,
            operation_status="failed",
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            exc_info=error is not None,
            **context
        )

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.info(
            f"Performance: {operation}",
            category=LogCategory.PERFORMANCE,
            operation=operation,
            duration_seconds=duration,
            **metrics
        )

    def log_security_event(self, event_type: str, **context):
        """Log security event"""
        self.warning(
            f"Security event: {event_type}",
            category=LogCategory.SECURITY,
            event_type=event_type,
            **context
        )

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        **context
    ):
        """Log API request"""
        level = LogLevel.ERROR if status_code >= 500 else (
            LogLevel.WARNING if status_code >= 400 else LogLevel.INFO
        )

        self._log(
            level,
            f"API {method} {endpoint} -> {status_code}",
            LogCategory.API,
            extra={
                "http_method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_seconds": duration,
                **context
            }
        )

    def log_agent_communication(
        self,
        source_agent: str,
        target_agent: str,
        message_type: str,
        success: bool,
        **context
    ):
        """Log inter-agent communication"""
        level = LogLevel.INFO if success else LogLevel.WARNING

        self._log(
            level,
            f"Agent communication: {source_agent} -> {target_agent} [{message_type}]",
            LogCategory.AGENT,
            extra={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "message_type": message_type,
                "success": success,
                **context
            }
        )


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: str = "logs/a2a.log",
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> Dict[str, Any]:
    """Configure standardized logging for the application"""

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
                "include_extra": True
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
                )
            }
        },
        "handlers": {},
        "root": {
            "level": log_level,
            "handlers": []
        },
        "loggers": {
            "a2a": {
                "level": log_level,
                "handlers": [],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            }
        }
    }

    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "structured" if log_format == "structured" else "detailed",
            "stream": "ext://sys.stdout"
        }
        config["root"]["handlers"].append("console")
        config["loggers"]["a2a"]["handlers"].append("console")
        config["loggers"]["uvicorn"]["handlers"].append("console")
        config["loggers"]["fastapi"]["handlers"].append("console")

    # File handler with rotation
    if enable_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "structured",
            "filename": file_path,
            "maxBytes": max_file_size,
            "backupCount": backup_count
        }
        config["root"]["handlers"].append("file")
        config["loggers"]["a2a"]["handlers"].append("file")

    # Apply configuration
    logging.config.dictConfig(config)

    return config


# Context managers for correlation tracking

class LoggingContext:
    """Context manager for setting logging context variables"""

    def __init__(
        self,
        correlation_id_val: Optional[str] = None,
        request_id_val: Optional[str] = None,
        user_id_val: Optional[str] = None,
        agent_id_val: Optional[str] = None
    ):
        self.correlation_id_val = correlation_id_val or str(uuid.uuid4())
        self.request_id_val = request_id_val
        self.user_id_val = user_id_val
        self.agent_id_val = agent_id_val

        # Store tokens for cleanup
        self.tokens = []

    def __enter__(self):
        """Enter logging context"""
        self.tokens.append(correlation_id.set(self.correlation_id_val))

        if self.request_id_val:
            self.tokens.append(request_id.set(self.request_id_val))
        if self.user_id_val:
            self.tokens.append(user_id.set(self.user_id_val))
        if self.agent_id_val:
            self.tokens.append(agent_id.set(self.agent_id_val))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context"""
        for token in reversed(self.tokens):
            try:
                correlation_id.reset(token)
            except (LookupError, ValueError):
                # Token might have been reset already
                pass


def get_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> A2ALogger:
    """Get standardized logger instance"""
    return A2ALogger(name, category)


# Decorators for automatic operation logging

def log_operation(
    operation_name: Optional[str] = None,
    category: LogCategory = LogCategory.BUSINESS,
    log_args: bool = False,
    log_result: bool = False
):
    """Decorator to automatically log operation start/completion/failure"""

    def decorator(func):
        func_name = operation_name or f"{func.__module__}.{func.__qualname__}"
        logger = get_logger(func.__module__, category)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = datetime.utcnow()

                log_context = {
                    "function": func.__qualname__,
                    "module": func.__module__
                }

                if log_args:
                    log_context["args"] = str(args)[:500]  # Limit length
                    log_context["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

                logger.start_operation(func_name, **log_context)

                try:
                    result = await func(*args, **kwargs)

                    duration = (datetime.utcnow() - start_time).total_seconds()
                    result_context = log_context.copy()

                    if log_result and result is not None:
                        result_context["result"] = str(result)[:200]

                    logger.complete_operation(func_name, duration, **result_context)
                    return result

                except Exception as e:
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.fail_operation(func_name, e, duration_seconds=duration, **log_context)
                    raise

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = datetime.utcnow()

                log_context = {
                    "function": func.__qualname__,
                    "module": func.__module__
                }

                if log_args:
                    log_context["args"] = str(args)[:500]
                    log_context["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

                logger.start_operation(func_name, **log_context)

                try:
                    result = func(*args, **kwargs)

                    duration = (datetime.utcnow() - start_time).total_seconds()
                    result_context = log_context.copy()

                    if log_result and result is not None:
                        result_context["result"] = str(result)[:200]

                    logger.complete_operation(func_name, duration, **result_context)
                    return result

                except Exception as e:
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.fail_operation(func_name, e, duration_seconds=duration, **log_context)
                    raise

            return sync_wrapper

    return decorator


# Performance logging decorator
def log_performance(threshold_seconds: float = 1.0):
    """Decorator to log performance metrics for slow operations"""

    def decorator(func):
        logger = get_logger(func.__module__, LogCategory.PERFORMANCE)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = datetime.utcnow()
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                if duration > threshold_seconds:
                    logger.log_performance(
                        f"{func.__module__}.{func.__qualname__}",
                        duration,
                        threshold=threshold_seconds,
                        slow_operation=True
                    )

                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = datetime.utcnow()
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                if duration > threshold_seconds:
                    logger.log_performance(
                        f"{func.__module__}.{func.__qualname__}",
                        duration,
                        threshold=threshold_seconds,
                        slow_operation=True
                    )

                return result
            return sync_wrapper

    return decorator


# Initialize logging configuration
def init_logging(
    level: str = "INFO",
    format_type: str = "structured",
    console: bool = True,
    file_logging: bool = False
):
    """Initialize application logging"""

    configure_logging(
        log_level=level,
        log_format=format_type,
        enable_console=console,
        enable_file=file_logging
    )

    # Create root logger
    root_logger = get_logger("a2a.startup", LogCategory.SYSTEM)
    root_logger.info(
        "Logging system initialized",
        log_level=level,
        log_format=format_type,
        console_enabled=console,
        file_enabled=file_logging
    )


# Export commonly used items
__all__ = [
    "LogLevel",
    "LogCategory",
    "A2ALogger",
    "LoggingContext",
    "get_logger",
    "log_operation",
    "log_performance",
    "init_logging",
    "configure_logging",
    "correlation_id",
    "request_id",
    "user_id",
    "agent_id"
]

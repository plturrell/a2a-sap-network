"""
SAP CAP Logging Service
Implements proper SAP CAP logging standards with structured logging, correlation IDs, and audit trails
"""
import http

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from contextvars import ContextVar
import asyncio

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class LogLevel(Enum):
    """SAP CAP standard log levels"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"

class LogCategory(Enum):
    """SAP CAP log categories"""
    BUSINESS = "BUSINESS"
    TECHNICAL = "TECHNICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"

class SAPCAPLogger:
    """SAP CAP compliant logger with structured logging"""
    
    def __init__(self, name: str, service_name: str = "a2a-portal"):
        self.name = name
        self.service_name = service_name
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with SAP CAP standards"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = SAPCAPFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_log_entry(self, level: LogLevel, message: str, category: LogCategory = LogCategory.TECHNICAL, 
                         correlation_id: Optional[str] = None, user_id: Optional[str] = None,
                         tenant_id: Optional[str] = None, component: Optional[str] = None,
                         custom_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create structured log entry following SAP CAP standards"""
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "category": category.value,
            "service": self.service_name,
            "component": component or self.name,
            "message": message,
            "correlation_id": correlation_id or correlation_id.get() or str(uuid.uuid4()),
        }
        
        # Add user context if available
        if user_id:
            entry["user_id"] = user_id
        
        # Add tenant context if available
        if tenant_id:
            entry["tenant_id"] = tenant_id
        
        # Add custom fields
        if custom_fields:
            entry["custom_fields"] = custom_fields
        
        return entry
    
    def info(self, message: str, category: LogCategory = LogCategory.TECHNICAL, **kwargs):
        """Log info message"""
        entry = self._create_log_entry(LogLevel.INFO, message, category, **kwargs)
        self.logger.info(json.dumps(entry))
    
    def debug(self, message: str, category: LogCategory = LogCategory.TECHNICAL, **kwargs):
        """Log debug message"""
        entry = self._create_log_entry(LogLevel.DEBUG, message, category, **kwargs)
        self.logger.debug(json.dumps(entry))
    
    def warn(self, message: str, category: LogCategory = LogCategory.TECHNICAL, **kwargs):
        """Log warning message"""
        entry = self._create_log_entry(LogLevel.WARN, message, category, **kwargs)
        self.logger.warning(json.dumps(entry))
    
    def error(self, message: str, category: LogCategory = LogCategory.TECHNICAL, **kwargs):
        """Log error message"""
        entry = self._create_log_entry(LogLevel.ERROR, message, category, **kwargs)
        self.logger.error(json.dumps(entry))
    
    def audit(self, action: str, resource: str, user_id: str, result: str = "SUCCESS", **kwargs):
        """Log audit event following SAP CAP audit standards"""
        entry = self._create_log_entry(
            LogLevel.INFO, 
            f"Audit: {action} on {resource} by {user_id} - {result}",
            LogCategory.AUDIT,
            user_id=user_id,
            custom_fields={
                "action": action,
                "resource": resource,
                "result": result,
                **kwargs
            }
        )
        self.logger.info(json.dumps(entry))
    
    def security(self, event: str, user_id: Optional[str] = None, severity: str = "MEDIUM", **kwargs):
        """Log security event"""
        entry = self._create_log_entry(
            LogLevel.WARN if severity == "MEDIUM" else LogLevel.ERROR,
            f"Security: {event}",
            LogCategory.SECURITY,
            user_id=user_id,
            custom_fields={
                "event": event,
                "severity": severity,
                **kwargs
            }
        )
        self.logger.warning(json.dumps(entry)) if severity == "MEDIUM" else self.logger.error(json.dumps(entry))
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        entry = self._create_log_entry(
            LogLevel.INFO,
            f"Performance: {operation} took {duration_ms}ms",
            LogCategory.PERFORMANCE,
            custom_fields={
                "operation": operation,
                "duration_ms": duration_ms,
                **kwargs
            }
        )
        self.logger.info(json.dumps(entry))

class SAPCAPFormatter(logging.Formatter):
    """Custom formatter for SAP CAP logging standards"""
    
    def format(self, record):
        # If the message is already JSON (structured log), return as-is
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Fallback to standard formatting for non-structured logs
            return super().format(record)

class LoggingMiddleware:
    """FastAPI middleware for request logging with correlation IDs"""
    
    def __init__(self, app, logger: SAPCAPLogger):
        self.app = app
        self.logger = logger
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate correlation ID for request
            corr_id = str(uuid.uuid4())
            correlation_id.set(corr_id)
            
            # Log request start
            request_info = {
                "method": scope["method"],
                "path": scope["path"],
                "query_string": scope.get("query_string", b"").decode(),
                "client": scope.get("client", ["unknown", 0])[0]
            }
            
            self.logger.info(
                f"Request started: {scope['method']} {scope['path']}",
                correlation_id=corr_id,
                custom_fields=request_info
            )
            
            # Measure request duration
            start_time = datetime.utcnow()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Log response
                    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self.logger.performance(
                        f"{scope['method']} {scope['path']}",
                        duration,
                        correlation_id=corr_id,
                        status_code=message["status"]
                    )
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

def get_logger(name: str) -> SAPCAPLogger:
    """Get SAP CAP compliant logger instance"""
    return SAPCAPLogger(name)

def setup_logging(app, service_name: str = "a2a-portal"):
    """Setup SAP CAP logging for FastAPI application"""
    logger = get_logger("middleware")
    
    # Add logging middleware
    app.middleware("http")(LoggingMiddleware(app, logger))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Use custom formatter
    )
    
    return logger

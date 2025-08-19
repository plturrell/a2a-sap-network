"""
OpenTelemetry middleware for trace propagation
"""
import http

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from typing import Dict
import logging

from app.a2a.core.telemetry import add_span_attributes, add_span_event

logger = logging.getLogger(__name__)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for OpenTelemetry trace context propagation"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract trace context from incoming headers
        headers_dict = dict(request.headers)
        extract(headers_dict)
        
        # Get current span and add request attributes
        span = trace.get_current_span()
        if span and span.is_recording():
            add_span_attributes({
                "http.method": request.method,
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": str(request.url.path),
                "http.url": str(request.url),
                "http.user_agent": request.headers.get("user-agent", ""),
                "net.peer.ip": request.client.host if request.client else "unknown",
            })
            
            # Add custom A2A headers if present
            if "x-a2a-agent-id" in headers_dict:
                add_span_attributes({
                    "a2a.source_agent": headers_dict["x-a2a-agent-id"],
                })
            
            if "x-a2a-workflow-id" in headers_dict:
                add_span_attributes({
                    "a2a.workflow_id": headers_dict["x-a2a-workflow-id"],
                })
            
            if "x-a2a-task-id" in headers_dict:
                add_span_attributes({
                    "a2a.task_id": headers_dict["x-a2a-task-id"],
                })
        
        # Process request
        response = await call_next(request)
        
        # Add response attributes
        if span and span.is_recording():
            add_span_attributes({
                "http.status_code": response.status_code,
            })
            
            # Add event for errors
            if response.status_code >= 400:
                add_span_event(
                    "http_error",
                    {
                        "http.status_code": response.status_code,
                        "http.method": request.method,
                        "http.target": str(request.url.path),
                    }
                )
        
        # Inject trace context into response headers for downstream propagation
        carrier = {}
        inject(carrier)
        for key, value in carrier.items():
            response.headers[key] = value
        
        return response


def get_trace_headers() -> Dict[str, str]:
    """Get trace context headers for outgoing requests"""
    carrier = {}
    inject(carrier)
    return carrier


def extract_trace_context(headers: Dict[str, str]):
    """Extract trace context from incoming headers"""
    extract(headers)
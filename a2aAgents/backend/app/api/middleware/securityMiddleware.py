"""
Security Event Reporting Middleware
Automatically detects and reports security events during request processing
"""

import asyncio
import time
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.securityMonitoring import report_security_event, EventType, ThreatLevel
from ...core.errorHandling import SecurityError, AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)


class SecurityEventMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically detect and report security events"""

    def __init__(self, app):
        super().__init__(app)
        self.failed_attempts = {}  # Track failed attempts by IP

        # Security patterns to detect
        self.injection_patterns = [
            r"union\s+select",
            r"or\s+1\s*=\s*1",
            r"drop\s+table",
            r"<script[^>]*>",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"\.\./",
            r"\.\.\\",
        ]

        # Suspicious user agents
        self.suspicious_agents = [
            "sqlmap",
            "nmap",
            "nikto",
            "burp",
            "owasp",
            "masscan"
        ]

        # Rate limiting tracking
        self.request_counts = {}
        self.suspicious_ips = set()

    async def dispatch(self, request: Request, call_next):
        """Process request and detect security events"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "").lower()

        # Pre-request security checks
        await self._check_suspicious_activity(request, client_ip, user_agent)

        try:
            # Process request
            response = await call_next(request)

            # Post-request security analysis
            await self._analyze_response(request, response, client_ip, start_time)

            return response

        except SecurityError as e:
            # Security errors are already handled, just re-raise
            await self._report_security_error(request, client_ip, str(e))
            raise

        except AuthenticationError as e:
            # Report authentication failure
            await self._report_auth_failure(request, client_ip, str(e))
            raise

        except AuthorizationError as e:
            # Report authorization failure
            await self._report_authz_failure(request, client_ip, str(e))
            raise

        except Exception as e:
            # Report unexpected errors that might indicate attacks
            if self._is_suspicious_error(str(e)):
                await self._report_suspicious_error(request, client_ip, str(e))
            raise

    async def _check_suspicious_activity(self, request: Request, client_ip: str, user_agent: str):
        """Check for suspicious activity before processing request"""

        # Check for suspicious user agents
        if any(agent in user_agent for agent in self.suspicious_agents):
            await report_security_event(
                event_type=EventType.SUSPICIOUS_TRAFFIC,
                threat_level=ThreatLevel.HIGH,
                description=f"Suspicious user agent detected: {user_agent}",
                source_ip=client_ip,
                details={
                    "user_agent": user_agent,
                    "url": str(request.url),
                    "method": request.method
                }
            )

        # Check for injection attempts in URL
        url_path = str(request.url.path) + str(request.url.query)
        if self._contains_injection_pattern(url_path):
            event_type = self._get_injection_type(url_path)
            await report_security_event(
                event_type=event_type,
                threat_level=ThreatLevel.HIGH,
                description=f"Injection attempt detected in URL: {request.url.path}",
                source_ip=client_ip,
                details={
                    "url": str(request.url),
                    "method": request.method,
                    "query_params": dict(request.query_params)
                }
            )

        # Check request headers for suspicious patterns
        await self._check_suspicious_headers(request, client_ip)

        # Rate limiting check
        await self._check_rate_limiting(request, client_ip)

    def _contains_injection_pattern(self, text: str) -> bool:
        """Check if text contains injection patterns"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.injection_patterns)

    def _get_injection_type(self, text: str) -> EventType:
        """Determine the type of injection attempt"""
        text_lower = text.lower()

        if any(sql_keyword in text_lower for sql_keyword in ["union", "select", "drop", "insert", "update", "delete"]):
            return EventType.SQL_INJECTION
        elif any(xss_pattern in text_lower for xss_pattern in ["<script", "javascript:", "onerror"]):
            return EventType.XSS_ATTEMPT
        else:
            return EventType.CODE_INJECTION

    async def _check_suspicious_headers(self, request: Request, client_ip: str):
        """Check request headers for suspicious patterns"""
        headers = dict(request.headers)

        # Check for common attack headers
        suspicious_headers = {
            "x-forwarded-for": "Potential IP spoofing",
            "x-real-ip": "Potential IP spoofing",
            "x-originating-ip": "Potential IP spoofing"
        }

        for header, description in suspicious_headers.items():
            if header in headers:
                header_value = headers[header]
                # Check if header contains multiple IPs (potential spoofing)
                if "," in header_value:
                    await report_security_event(
                        event_type=EventType.SUSPICIOUS_TRAFFIC,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"{description}: {header_value}",
                        source_ip=client_ip,
                        details={
                            "suspicious_header": header,
                            "header_value": header_value
                        }
                    )

        # Check for injection in headers
        for header_name, header_value in headers.items():
            if self._contains_injection_pattern(header_value):
                await report_security_event(
                    event_type=EventType.CODE_INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Injection attempt in header {header_name}",
                    source_ip=client_ip,
                    details={
                        "header_name": header_name,
                        "header_value": header_value
                    }
                )

    async def _check_rate_limiting(self, request: Request, client_ip: str):
        """Check for potential DDoS based on request rate"""
        now = time.time()

        # Initialize tracking for new IPs
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # Add current request
        self.request_counts[client_ip].append(now)

        # Remove requests older than 1 minute
        cutoff = now - 60
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip] if req_time > cutoff
        ]

        # Check for DDoS
        recent_requests = len(self.request_counts[client_ip])
        if recent_requests > 100:  # More than 100 requests per minute
            if client_ip not in self.suspicious_ips:
                self.suspicious_ips.add(client_ip)
                await report_security_event(
                    event_type=EventType.DDOS_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Potential DDoS attack: {recent_requests} requests in 1 minute",
                    source_ip=client_ip,
                    details={
                        "request_count": recent_requests,
                        "time_window": "1 minute",
                        "url": str(request.url),
                        "method": request.method
                    }
                )

        # Clean up old entries periodically
        if now % 300 < 1:  # Every 5 minutes approximately
            old_cutoff = now - 3600  # 1 hour
            for ip in list(self.request_counts.keys()):
                self.request_counts[ip] = [
                    req_time for req_time in self.request_counts[ip] if req_time > old_cutoff
                ]
                if not self.request_counts[ip]:
                    del self.request_counts[ip]

    async def _analyze_response(self, request: Request, response: Response, client_ip: str, start_time: float):
        """Analyze response for security indicators"""

        # Check for authentication failures
        if response.status_code == 401:
            await self._track_auth_failure(request, client_ip)

        # Check for authorization failures
        elif response.status_code == 403:
            await report_security_event(
                event_type=EventType.ACCESS_DENIED,
                threat_level=ThreatLevel.MEDIUM,
                description=f"Access denied: {request.method} {request.url.path}",
                source_ip=client_ip,
                details={
                    "url": str(request.url),
                    "method": request.method,
                    "status_code": response.status_code
                }
            )

        # Check for error codes that might indicate attacks
        elif response.status_code >= 500:
            processing_time = time.time() - start_time
            # Very slow responses might indicate attacks
            if processing_time > 10:  # Slower than 10 seconds
                await report_security_event(
                    event_type=EventType.SUSPICIOUS_TRAFFIC,
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Unusually slow response: {processing_time:.2f}s",
                    source_ip=client_ip,
                    details={
                        "processing_time": processing_time,
                        "url": str(request.url),
                        "status_code": response.status_code
                    }
                )

        # Check for data access patterns
        if request.url.path.startswith("/api/v1/data/"):
            await self._check_data_access_pattern(request, client_ip)

    async def _track_auth_failure(self, request: Request, client_ip: str):
        """Track authentication failures for brute force detection"""
        now = time.time()

        # Initialize tracking
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []

        # Add failure
        self.failed_attempts[client_ip].append(now)

        # Remove old failures (older than 5 minutes)
        cutoff = now - 300
        self.failed_attempts[client_ip] = [
            failure_time for failure_time in self.failed_attempts[client_ip] if failure_time > cutoff
        ]

        # Check for brute force
        recent_failures = len(self.failed_attempts[client_ip])

        await report_security_event(
            event_type=EventType.LOGIN_FAILURE,
            threat_level=ThreatLevel.LOW if recent_failures < 3 else ThreatLevel.MEDIUM,
            description=f"Authentication failure from {client_ip}",
            source_ip=client_ip,
            details={
                "url": str(request.url),
                "method": request.method,
                "recent_failures": recent_failures,
                "user_agent": request.headers.get("user-agent")
            }
        )

        # Report brute force if threshold exceeded
        if recent_failures >= 5:
            await report_security_event(
                event_type=EventType.BRUTE_FORCE_ATTEMPT,
                threat_level=ThreatLevel.HIGH,
                description=f"Brute force attack detected: {recent_failures} failures in 5 minutes",
                source_ip=client_ip,
                details={
                    "failure_count": recent_failures,
                    "time_window": "5 minutes",
                    "target_url": str(request.url)
                }
            )

    async def _check_data_access_pattern(self, request: Request, client_ip: str):
        """Check for suspicious data access patterns"""
        # This would integrate with actual data access tracking
        # For now, just report sensitive data access
        await report_security_event(
            event_type=EventType.SENSITIVE_DATA_ACCESS,
            threat_level=ThreatLevel.INFO,
            description=f"Data endpoint accessed: {request.url.path}",
            source_ip=client_ip,
            details={
                "endpoint": request.url.path,
                "method": request.method,
                "query_params": dict(request.query_params)
            }
        )

    async def _report_security_error(self, request: Request, client_ip: str, error_msg: str):
        """Report security errors"""
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,
            threat_level=ThreatLevel.HIGH,
            description=f"Security error: {error_msg}",
            source_ip=client_ip,
            details={
                "url": str(request.url),
                "method": request.method,
                "error": error_msg
            }
        )

    async def _report_auth_failure(self, request: Request, client_ip: str, error_msg: str):
        """Report authentication failures"""
        await self._track_auth_failure(request, client_ip)

    async def _report_authz_failure(self, request: Request, client_ip: str, error_msg: str):
        """Report authorization failures"""
        await report_security_event(
            event_type=EventType.ACCESS_DENIED,
            threat_level=ThreatLevel.MEDIUM,
            description=f"Authorization failure: {error_msg}",
            source_ip=client_ip,
            details={
                "url": str(request.url),
                "method": request.method,
                "error": error_msg
            }
        )

    async def _report_suspicious_error(self, request: Request, client_ip: str, error_msg: str):
        """Report suspicious errors that might indicate attacks"""
        await report_security_event(
            event_type=EventType.SYSTEM_INTRUSION,
            threat_level=ThreatLevel.MEDIUM,
            description=f"Suspicious error: {error_msg}",
            source_ip=client_ip,
            details={
                "url": str(request.url),
                "method": request.method,
                "error": error_msg
            }
        )

    def _is_suspicious_error(self, error_msg: str) -> bool:
        """Check if error message indicates potential attack"""
        suspicious_errors = [
            "command injection",
            "sql injection",
            "path traversal",
            "buffer overflow",
            "memory corruption",
            "privilege escalation"
        ]

        error_lower = error_msg.lower()
        return any(suspicious in error_lower for suspicious in suspicious_errors)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers"""
        # Check for X-Forwarded-For header (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()

        # Check for X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"


# Export middleware class
__all__ = ["SecurityEventMiddleware"]

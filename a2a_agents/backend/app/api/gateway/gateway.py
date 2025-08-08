"""
API Gateway for A2A Network
Provides centralized routing, authentication, rate limiting, and monitoring
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
import asyncio
from typing import Optional, Dict, Any, List
import logging
import time
import json
from datetime import datetime, timedelta
import jwt
from functools import lru_cache

from pydantic import BaseModel, Field
from app.core.config import settings
from app.api.deps import get_current_user
from src.a2a.core.telemetry import trace_async, add_span_attributes, get_trace_context
from app.api.middleware.telemetry import get_trace_headers

logger = logging.getLogger(__name__)


class ServiceConfig(BaseModel):
    """Configuration for a backend service"""
    name: str
    base_url: str
    health_endpoint: str = "/health"
    timeout: float = 30.0
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class GatewayConfig(BaseModel):
    """API Gateway configuration"""
    services: Dict[str, ServiceConfig]
    rate_limits: Dict[str, RateLimitConfig]
    default_rate_limit: RateLimitConfig
    auth_required_paths: List[str]
    auth_optional_paths: List[str]
    public_paths: List[str]


# Service registry
SERVICE_REGISTRY = {
    "main": ServiceConfig(
        name="Main API",
        base_url="http://localhost:8000",
        health_endpoint="/health"
    ),
    "agent0": ServiceConfig(
        name="Agent 0 - Data Product Registration",
        base_url="http://localhost:8001",
        health_endpoint="/a2a/agent0/v1/.well-known/agent.json"
    ),
    "agent1": ServiceConfig(
        name="Agent 1 - Data Standardization",
        base_url="http://localhost:8002",
        health_endpoint="/a2a/v1/.well-known/agent.json"
    ),
    "agent2": ServiceConfig(
        name="Agent 2 - AI Preparation",
        base_url="http://localhost:8003",
        health_endpoint="/a2a/agent2/v1/.well-known/agent.json"
    ),
    "agent3": ServiceConfig(
        name="Agent 3 - Vector Processing",
        base_url="http://localhost:8004",
        health_endpoint="/a2a/agent3/v1/.well-known/agent.json"
    ),
    "data_manager": ServiceConfig(
        name="Data Manager Agent",
        base_url="http://localhost:8005",
        health_endpoint="/a2a/data_manager/v1/.well-known/agent.json"
    ),
    "catalog_manager": ServiceConfig(
        name="Catalog Manager Agent",
        base_url="http://localhost:8006",
        health_endpoint="/a2a/catalog_manager/v1/.well-known/agent.json"
    ),
    "agent_manager": ServiceConfig(
        name="Agent Manager",
        base_url="http://localhost:8010",
        health_endpoint="/a2a/agent_manager/v1/.well-known/agent.json"
    ),
    "registry": ServiceConfig(
        name="A2A Registry",
        base_url="http://localhost:8100",
        health_endpoint="/api/v1/a2a/health"
    )
}


class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self):
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        key: str,
        limit: RateLimitConfig,
        cost: int = 1
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        async with self._lock:
            now = time.time()
            
            if key not in self.buckets:
                self.buckets[key] = {
                    "tokens": limit.burst_size,
                    "last_refill": now,
                    "minute_count": 0,
                    "minute_window": now,
                    "hour_count": 0,
                    "hour_window": now
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = now - bucket["last_refill"]
            refill_rate = limit.requests_per_minute / 60.0
            new_tokens = time_passed * refill_rate
            bucket["tokens"] = min(limit.burst_size, bucket["tokens"] + new_tokens)
            bucket["last_refill"] = now
            
            # Check minute window
            if now - bucket["minute_window"] > 60:
                bucket["minute_count"] = 0
                bucket["minute_window"] = now
            
            # Check hour window
            if now - bucket["hour_window"] > 3600:
                bucket["hour_count"] = 0
                bucket["hour_window"] = now
            
            # Check limits
            if bucket["tokens"] < cost:
                return False, {
                    "limit": "burst",
                    "retry_after": int((cost - bucket["tokens"]) / refill_rate)
                }
            
            if bucket["minute_count"] + cost > limit.requests_per_minute:
                return False, {
                    "limit": "minute",
                    "retry_after": 60 - int(now - bucket["minute_window"])
                }
            
            if bucket["hour_count"] + cost > limit.requests_per_hour:
                return False, {
                    "limit": "hour",
                    "retry_after": 3600 - int(now - bucket["hour_window"])
                }
            
            # Deduct tokens
            bucket["tokens"] -= cost
            bucket["minute_count"] += cost
            bucket["hour_count"] += cost
            
            return True, {
                "tokens_remaining": int(bucket["tokens"]),
                "requests_remaining_minute": limit.requests_per_minute - bucket["minute_count"],
                "requests_remaining_hour": limit.requests_per_hour - bucket["hour_count"]
            }


class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures: Dict[str, List[float]] = {}
        self.open_until: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def is_open(self, service: str) -> bool:
        """Check if circuit is open for a service"""
        async with self._lock:
            if service in self.open_until:
                if time.time() < self.open_until[service]:
                    return True
                else:
                    # Reset circuit
                    del self.open_until[service]
                    self.failures[service] = []
            return False
    
    async def record_success(self, service: str):
        """Record successful request"""
        async with self._lock:
            if service in self.failures:
                self.failures[service] = []
    
    async def record_failure(self, service: str):
        """Record failed request"""
        async with self._lock:
            now = time.time()
            
            if service not in self.failures:
                self.failures[service] = []
            
            # Remove old failures
            self.failures[service] = [
                f for f in self.failures[service]
                if now - f < 60  # Consider failures in last minute
            ]
            
            self.failures[service].append(now)
            
            # Check if we should open circuit
            if len(self.failures[service]) >= self.threshold:
                self.open_until[service] = now + self.timeout
                logger.warning(f"Circuit breaker opened for {service}")


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.rate_limiter = RateLimiter()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize circuit breakers
        for service_name, service_config in config.services.items():
            self.circuit_breakers[service_name] = CircuitBreaker(
                threshold=service_config.circuit_breaker_threshold,
                timeout=service_config.circuit_breaker_timeout
            )
    
    @trace_async("gateway_route_request")
    async def route_request(
        self,
        request: Request,
        service_name: str,
        path: str,
        user_id: Optional[str] = None
    ) -> JSONResponse:
        """Route request to appropriate service"""
        
        # Add trace attributes
        add_span_attributes({
            "gateway.service": service_name,
            "gateway.path": path,
            "gateway.method": request.method,
            "gateway.user_id": user_id or "anonymous"
        })
        
        # Check if service exists
        if service_name not in self.config.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        service_config = self.config.services[service_name]
        
        # Check circuit breaker
        if await self.circuit_breakers[service_name].is_open(service_name):
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} is temporarily unavailable"
            )
        
        # Prepare request
        target_url = f"{service_config.base_url}{path}"
        
        # Forward headers with trace context
        headers = dict(request.headers)
        headers.update(get_trace_headers())
        
        # Add gateway headers
        headers["X-Forwarded-For"] = request.client.host
        headers["X-Forwarded-Proto"] = request.url.scheme
        headers["X-Gateway-Request-Id"] = request.state.request_id
        
        if user_id:
            headers["X-User-Id"] = user_id
        
        # Get request body
        body = await request.body()
        
        # Make request with retries
        for attempt in range(service_config.retry_attempts):
            try:
                response = await self.http_client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                    timeout=service_config.timeout
                )
                
                # Record success
                await self.circuit_breakers[service_name].record_success(service_name)
                
                # Return response
                return JSONResponse(
                    content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
                
            except Exception as e:
                logger.error(f"Request to {service_name} failed (attempt {attempt + 1}): {e}")
                await self.circuit_breakers[service_name].record_failure(service_name)
                
                if attempt == service_config.retry_attempts - 1:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to connect to {service_name}"
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        if service_name not in self.config.services:
            return {"status": "unknown", "error": "Service not found"}
        
        service_config = self.config.services[service_name]
        health_url = f"{service_config.base_url}{service_config.health_endpoint}"
        
        try:
            response = await self.http_client.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def get_all_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health_checks = {}
        
        tasks = [
            self.check_service_health(service_name)
            for service_name in self.config.services
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for service_name, result in zip(self.config.services.keys(), results):
            if isinstance(result, Exception):
                health_checks[service_name] = {
                    "status": "error",
                    "error": str(result)
                }
            else:
                health_checks[service_name] = result
        
        # Overall status
        all_healthy = all(
            check.get("status") == "healthy"
            for check in health_checks.values()
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": health_checks,
            "timestamp": datetime.utcnow().isoformat()
        }


# Create gateway instance
gateway_config = GatewayConfig(
    services=SERVICE_REGISTRY,
    rate_limits={
        "default": RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
        "authenticated": RateLimitConfig(requests_per_minute=120, requests_per_hour=5000),
        "premium": RateLimitConfig(requests_per_minute=300, requests_per_hour=10000)
    },
    default_rate_limit=RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
    auth_required_paths=["/api/v1/a2a/workflows", "/api/v1/a2a/trust"],
    auth_optional_paths=["/a2a/*/v1/rpc", "/a2a/*/v1/messages"],
    public_paths=["/health", "/api/v1/a2a/health", "/.well-known/agent.json"]
)

gateway = APIGateway(gateway_config)
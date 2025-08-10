"""
API Gateway Router
Central entry point for all A2A Network requests
"""

from fastapi import APIRouter, Request, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uuid
import logging

from app.api.deps import get_current_user_optional
from app.models.user import User
from .gateway import gateway, RateLimitConfig
from app.a2a.core.telemetry import trace_async, add_span_attributes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gateway", tags=["API Gateway"])


@router.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@trace_async("gateway_request")
async def gateway_proxy(
    request: Request,
    service_name: str,
    path: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    x_api_key: Optional[str] = Header(None)
):
    """
    Main gateway endpoint - routes requests to appropriate services
    
    Service names:
    - main: Main API
    - agent0: Data Product Registration Agent
    - agent1: Data Standardization Agent  
    - agent2: AI Preparation Agent
    - agent3: Vector Processing Agent
    - data_manager: Data Manager Agent
    - catalog_manager: Catalog Manager Agent
    - agent_manager: Agent Manager
    - registry: A2A Registry
    """
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add trace attributes
    add_span_attributes({
        "gateway.request_id": request_id,
        "gateway.service": service_name,
        "gateway.path": path,
        "gateway.authenticated": current_user is not None,
        "gateway.api_key_present": x_api_key is not None
    })
    
    # Determine rate limit tier
    rate_limit_key = "anonymous"
    rate_limit_config = gateway.config.default_rate_limit
    
    if current_user:
        rate_limit_key = f"user:{current_user.id}"
        # Check user tier
        if hasattr(current_user, "tier"):
            if current_user.tier == "premium":
                rate_limit_config = gateway.config.rate_limits["premium"]
            else:
                rate_limit_config = gateway.config.rate_limits["authenticated"]
        else:
            rate_limit_config = gateway.config.rate_limits["authenticated"]
    elif x_api_key:
        rate_limit_key = f"api_key:{x_api_key[:8]}"
        rate_limit_config = gateway.config.rate_limits["authenticated"]
    
    # Check rate limits
    allowed, rate_limit_info = await gateway.rate_limiter.check_rate_limit(
        rate_limit_key,
        rate_limit_config
    )
    
    if not allowed:
        retry_after = rate_limit_info.get("retry_after", 60)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {rate_limit_info.get('limit')}",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rate_limit_config.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + retry_after)
            }
        )
    
    # Add rate limit headers to response
    response_headers = {
        "X-RateLimit-Limit": str(rate_limit_config.requests_per_minute),
        "X-RateLimit-Remaining": str(rate_limit_info.get("requests_remaining_minute", 0)),
        "X-Gateway-Request-Id": request_id
    }
    
    # Check authentication requirements
    full_path = f"/{path}"
    
    # Check if path requires authentication
    requires_auth = any(
        full_path.startswith(auth_path)
        for auth_path in gateway.config.auth_required_paths
    )
    
    if requires_auth and not current_user and not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for this endpoint"
        )
    
    try:
        # Route request
        response = await gateway.route_request(
            request=request,
            service_name=service_name,
            path=f"/{path}",
            user_id=str(current_user.id) if current_user else None
        )
        
        # Add gateway headers
        for key, value in response_headers.items():
            response.headers[key] = value
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gateway error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal gateway error"
        )


@router.get("/health")
async def gateway_health():
    """Get health status of all services"""
    return await gateway.get_all_service_health()


@router.get("/services")
async def list_services():
    """List all available services"""
    services = {}
    for name, config in gateway.config.services.items():
        services[name] = {
            "name": config.name,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "circuit_breaker": {
                "threshold": config.circuit_breaker_threshold,
                "timeout": config.circuit_breaker_timeout
            }
        }
    
    return {
        "services": services,
        "total": len(services)
    }


@router.get("/rate-limits")
async def get_rate_limits(
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get rate limit information for current user"""
    
    if current_user:
        tier = getattr(current_user, "tier", "authenticated")
        if tier == "premium":
            config = gateway.config.rate_limits["premium"]
        else:
            config = gateway.config.rate_limits["authenticated"]
        
        rate_limit_key = f"user:{current_user.id}"
    else:
        config = gateway.config.default_rate_limit
        rate_limit_key = "anonymous"
    
    # Get current usage
    allowed, info = await gateway.rate_limiter.check_rate_limit(
        rate_limit_key,
        config,
        cost=0  # Don't consume tokens
    )
    
    return {
        "tier": "premium" if current_user and getattr(current_user, "tier", None) == "premium" else "authenticated" if current_user else "anonymous",
        "limits": {
            "requests_per_minute": config.requests_per_minute,
            "requests_per_hour": config.requests_per_hour,
            "burst_size": config.burst_size
        },
        "current_usage": {
            "tokens_remaining": info.get("tokens_remaining", 0),
            "requests_remaining_minute": info.get("requests_remaining_minute", 0),
            "requests_remaining_hour": info.get("requests_remaining_hour", 0)
        }
    }


@router.get("/metrics")
async def gateway_metrics():
    """Get gateway metrics"""
    # Circuit breaker status
    circuit_status = {}
    for service, breaker in gateway.circuit_breakers.items():
        is_open = await breaker.is_open(service)
        circuit_status[service] = {
            "status": "open" if is_open else "closed",
            "failures": len(breaker.failures.get(service, [])),
            "threshold": breaker.threshold
        }
    
    return {
        "circuit_breakers": circuit_status,
        "active_rate_limits": len(gateway.rate_limiter.buckets),
        "services_configured": len(gateway.config.services)
    }


# Import time for rate limit headers
import time
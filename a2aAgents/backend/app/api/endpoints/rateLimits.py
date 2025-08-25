"""
Rate Limiting Management API
Administrative endpoints for monitoring and configuring rate limits
"""

from fastapi import APIRouter, Request, Depends, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ...core.rateLimiting import get_rate_limiter, UserTier, RateLimit
from ...core.errorHandling import ValidationError, AuthorizationError
from ..middleware.auth import get_current_user, require_admin

router = APIRouter(prefix="/rate-limits", tags=["Rate Limiting"])
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_rate_limit_status(
    request: Request,
    user_tier: UserTier = UserTier.ANONYMOUS
) -> Dict[str, Any]:
    """
    Get current rate limit status for the requesting IP/user

    Returns rate limit information including:
    - Current limits
    - Remaining requests
    - Reset times
    - DDoS protection status
    """
    try:
        rate_limiter = await get_rate_limiter()
        status = await rate_limiter.get_rate_limit_status(
            request=request,
            user_tier=user_tier
        )

        # Add DDoS protection status
        if rate_limiter.ddos_detector:
            client_ip = rate_limiter._get_client_ip(request)
            is_blocked = client_ip in rate_limiter.ddos_detector.blocked_ips

            status["ddos_protection"] = {
                "enabled": True,
                "status": "blocked" if is_blocked else "normal",
                "blocked_until": rate_limiter.ddos_detector.blocked_ips.get(client_ip)
            }
        else:
            status["ddos_protection"] = {"enabled": False}

        return {
            "success": True,
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve rate limit status"
        )


@router.get("/config")
async def get_rate_limit_config(
    current_user = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get current rate limiting configuration (admin only)

    Returns:
    - Rate limits by user tier
    - Endpoint-specific limits
    - DDoS protection settings
    """
    try:
        rate_limiter = await get_rate_limiter()

        config = {
            "user_tier_limits": {
                tier.value: {
                    "requests_per_minute": limit.requests_per_minute,
                    "requests_per_hour": limit.requests_per_hour,
                    "requests_per_day": limit.requests_per_day,
                    "burst_capacity": limit.burst_capacity
                }
                for tier, limit in rate_limiter.rate_limits.items()
            },
            "endpoint_limits": rate_limiter.endpoint_limits,
            "ddos_protection": {
                "enabled": rate_limiter.enable_ddos_protection,
                "window_size": rate_limiter.ddos_detector.window_size if rate_limiter.ddos_detector else None,
                "threshold_multiplier": rate_limiter.ddos_detector.threshold_multiplier if rate_limiter.ddos_detector else None
            },
            "backend": "redis" if rate_limiter.redis_client else "memory"
        }

        return {
            "success": True,
            "data": config,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get rate limit config: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve rate limit configuration"
        )


@router.get("/blocked-ips")
async def get_blocked_ips(
    current_user = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get currently blocked IPs from DDoS protection (admin only)
    """
    try:
        rate_limiter = await get_rate_limiter()

        if not rate_limiter.ddos_detector:
            return {
                "success": True,
                "data": {
                    "blocked_ips": {},
                    "ddos_protection_enabled": False
                },
                "timestamp": datetime.utcnow().isoformat()
            }

        import time
        now = time.time()

        blocked_ips = {}
        for ip, unblock_time in rate_limiter.ddos_detector.blocked_ips.items():
            if unblock_time > now:
                remaining_seconds = int(unblock_time - now)
                blocked_ips[ip] = {
                    "unblock_time": datetime.fromtimestamp(unblock_time).isoformat(),
                    "remaining_seconds": remaining_seconds
                }

        return {
            "success": True,
            "data": {
                "blocked_ips": blocked_ips,
                "total_blocked": len(blocked_ips),
                "ddos_protection_enabled": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get blocked IPs: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve blocked IPs"
        )


@router.post("/unblock-ip")
async def unblock_ip(
    ip_address: str,
    current_user = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Manually unblock an IP address (admin only)
    """
    try:
        rate_limiter = await get_rate_limiter()

        if not rate_limiter.ddos_detector:
            raise HTTPException(
                status_code=400,
                detail="DDoS protection is not enabled"
            )

        # Validate IP format
        import ipaddress
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            raise ValidationError(f"Invalid IP address format: {ip_address}")

        # Remove from blocked IPs
        if ip_address in rate_limiter.ddos_detector.blocked_ips:
            del rate_limiter.ddos_detector.blocked_ips[ip_address]
            logger.info(f"IP {ip_address} manually unblocked by admin {current_user.get('sub', 'unknown')}")

            return {
                "success": True,
                "message": f"IP {ip_address} has been unblocked",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": True,
                "message": f"IP {ip_address} was not blocked",
                "timestamp": datetime.utcnow().isoformat()
            }

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Failed to unblock IP {ip_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to unblock IP address"
        )


@router.get("/metrics")
async def get_rate_limit_metrics(
    current_user = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get rate limiting metrics and statistics (admin only)
    """
    try:
        rate_limiter = await get_rate_limiter()

        metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "blocked_ips_count": 0,
            "active_tokens_buckets": len(rate_limiter.token_buckets),
            "memory_usage": {
                "token_buckets": len(rate_limiter.token_buckets),
                "request_counts": sum(len(counts) for counts in rate_limiter.request_counts.values()),
                "window_start_times": sum(len(times) for times in rate_limiter.window_start_times.values())
            }
        }

        if rate_limiter.ddos_detector:
            metrics["blocked_ips_count"] = len([
                ip for ip, unblock_time in rate_limiter.ddos_detector.blocked_ips.items()
                if unblock_time > time.time()
            ])
            metrics["suspicious_patterns"] = len(rate_limiter.ddos_detector.suspicious_patterns)

            # Request history metrics
            import time
            now = time.time()
            recent_requests = 0
            for history in rate_limiter.ddos_detector.request_history.values():
                recent_requests += len([
                    req_time for req_time in history
                    if req_time > now - 300  # Last 5 minutes
                ])
            metrics["recent_requests_5min"] = recent_requests

        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get rate limit metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve rate limit metrics"
        )


@router.post("/clear-history")
async def clear_rate_limit_history(
    current_user = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Clear rate limiting history and reset counters (admin only)

    Warning: This will reset all rate limiting state including:
    - Token buckets
    - Request counters
    - DDoS detection history
    """
    try:
        rate_limiter = await get_rate_limiter()

        # Clear in-memory storage
        rate_limiter.token_buckets.clear()
        rate_limiter.request_counts.clear()
        rate_limiter.window_start_times.clear()

        # Clear DDoS detector history
        if rate_limiter.ddos_detector:
            rate_limiter.ddos_detector.request_history.clear()
            rate_limiter.ddos_detector.suspicious_patterns.clear()
            # Note: We don't clear blocked_ips as that would be a security risk

        # Clear Redis keys if using Redis backend
        if rate_limiter.redis_client:
            try:
                # Get all rate limit keys
                keys = await rate_limiter.redis_client.keys("rate_limit:*")
                if keys:
                    await rate_limiter.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis rate limit keys")
            except Exception as e:
                logger.error(f"Failed to clear Redis rate limit keys: {e}")

        logger.warning(f"Rate limiting history cleared by admin {current_user.get('sub', 'unknown')}")

        return {
            "success": True,
            "message": "Rate limiting history has been cleared",
            "warning": "All rate limiting counters have been reset",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to clear rate limit history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear rate limit history"
        )


# Export router
__all__ = ["router"]

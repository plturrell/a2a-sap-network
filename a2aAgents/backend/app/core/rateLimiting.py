"""
Rate Limiting and DDoS Protection System
Comprehensive rate limiting with token bucket algorithm and DDoS protection
"""

import time
import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Rate limit types"""
    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"


class UserTier(str, Enum):
    """User tiers with different rate limits"""
    ANONYMOUS = "anonymous"
    AUTHENTICATED = "authenticated"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_capacity: int
    tier: UserTier = UserTier.ANONYMOUS


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time before tokens are available"""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class DDoSDetector:
    """DDoS attack detection system"""
    
    def __init__(self, 
                 window_size: int = 300,  # 5 minutes
                 threshold_multiplier: float = 10.0):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        self.suspicious_patterns: Set[str] = set()
    
    def is_ddos_attack(self, client_ip: str, normal_rate: float) -> Tuple[bool, str]:
        """
        Detect if request pattern indicates DDoS attack
        
        Args:
            client_ip: Client IP address
            normal_rate: Normal request rate for comparison
            
        Returns:
            Tuple of (is_attack, reason)
        """
        now = time.time()
        
        # Clean old blocked IPs
        self.blocked_ips = {
            ip: unblock_time 
            for ip, unblock_time in self.blocked_ips.items() 
            if unblock_time > now
        }
        
        # Check if IP is already blocked
        if client_ip in self.blocked_ips:
            remaining = int(self.blocked_ips[client_ip] - now)
            return True, f"IP blocked for DDoS - {remaining}s remaining"
        
        # Record request
        history = self.request_history[client_ip]
        history.append(now)
        
        # Remove old entries outside window
        cutoff = now - self.window_size
        while history and history[0] < cutoff:
            history.popleft()
        
        # Calculate current request rate
        current_rate = len(history) / self.window_size * 60  # requests per minute
        
        # DDoS detection logic
        if current_rate > normal_rate * self.threshold_multiplier:
            # Block IP for escalating duration based on offense count
            offense_count = sum(1 for pattern in self.suspicious_patterns if client_ip in pattern)
            block_duration = min(3600, 300 * (2 ** offense_count))  # Max 1 hour
            
            self.blocked_ips[client_ip] = now + block_duration
            self.suspicious_patterns.add(f"{client_ip}_{int(now)}")
            
            logger.warning(f"🚨 DDoS attack detected from {client_ip}: {current_rate:.1f} req/min (normal: {normal_rate:.1f})")
            return True, f"DDoS protection activated - blocked for {block_duration}s"
        
        # Advanced pattern detection
        if self._detect_suspicious_patterns(client_ip, history):
            self.blocked_ips[client_ip] = now + 600  # 10-minute block
            return True, "Suspicious request pattern detected"
        
        return False, ""
    
    def _detect_suspicious_patterns(self, client_ip: str, history: deque) -> bool:
        """Detect suspicious request patterns"""
        if len(history) < 10:
            return False
        
        recent_requests = list(history)[-10:]
        intervals = [recent_requests[i+1] - recent_requests[i] for i in range(9)]
        
        # Detect very regular intervals (bot-like behavior)
        if len(set(round(interval, 1) for interval in intervals)) == 1:
            logger.warning(f"⚠️ Regular interval pattern detected from {client_ip}")
            return True
        
        # Detect burst patterns
        rapid_requests = sum(1 for interval in intervals if interval < 0.1)
        if rapid_requests > 7:  # More than 7 requests in < 0.1s intervals
            logger.warning(f"⚠️ Burst pattern detected from {client_ip}")
            return True
        
        return False


class RateLimiter:
    """Comprehensive rate limiter with multiple strategies"""
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 enable_ddos_protection: bool = True):
        self.redis_client = redis_client
        self.enable_ddos_protection = enable_ddos_protection
        
        # In-memory storage for when Redis is not available
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.window_start_times: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # DDoS detector
        self.ddos_detector = DDoSDetector() if enable_ddos_protection else None
        
        # Rate limit configurations by user tier
        self.rate_limits = {
            UserTier.ANONYMOUS: RateLimit(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_capacity=10
            ),
            UserTier.AUTHENTICATED: RateLimit(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_capacity=20
            ),
            UserTier.PREMIUM: RateLimit(
                requests_per_minute=300,
                requests_per_hour=15000,
                requests_per_day=200000,
                burst_capacity=50
            ),
            UserTier.ENTERPRISE: RateLimit(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=1000000,
                burst_capacity=100
            )
        }
        
        # Endpoint-specific rate limits
        self.endpoint_limits = {
            "/api/v1/auth/login": (5, 60),  # 5 requests per minute
            "/api/v1/auth/register": (3, 300),  # 3 requests per 5 minutes
            "/api/v1/upload": (10, 60),  # 10 uploads per minute
            "/api/v1/search": (100, 60),  # 100 searches per minute
        }
    
    async def check_rate_limit(self,
                             request: Request,
                             user_id: Optional[str] = None,
                             user_tier: UserTier = UserTier.ANONYMOUS,
                             endpoint: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if request should be rate limited
        
        Args:
            request: FastAPI request object
            user_id: User ID if authenticated
            user_tier: User tier for rate limit configuration
            endpoint: Specific endpoint for endpoint-based limits
            
        Returns:
            Tuple of (is_allowed, error_info)
        """
        client_ip = self._get_client_ip(request)
        
        # DDoS protection check
        if self.ddos_detector:
            rate_limit = self.rate_limits[user_tier]
            is_attack, reason = self.ddos_detector.is_ddos_attack(
                client_ip, 
                rate_limit.requests_per_minute
            )
            
            if is_attack:
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": "Request blocked by DDoS protection",
                    "reason": reason,
                    "retry_after": 300
                }
        
        # Check multiple rate limit layers
        checks = [
            ("ip", client_ip),
            ("global", "global")
        ]
        
        if user_id:
            checks.append(("user", user_id))
        
        if endpoint:
            checks.append(("endpoint", f"{endpoint}:{client_ip}"))
        
        # Perform all checks
        for limit_type, identifier in checks:
            is_allowed, error_info = await self._check_limit(
                limit_type, identifier, user_tier, endpoint
            )
            
            if not is_allowed:
                return False, error_info
        
        return True, None
    
    async def _check_limit(self,
                          limit_type: str,
                          identifier: str,
                          user_tier: UserTier,
                          endpoint: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check specific rate limit"""
        
        # Get rate limit configuration
        if endpoint and endpoint in self.endpoint_limits:
            limit, window = self.endpoint_limits[endpoint]
            rate_config = RateLimit(
                requests_per_minute=limit,
                requests_per_hour=limit * 60 // window,
                requests_per_day=limit * 1440 // window,
                burst_capacity=min(10, limit)
            )
        else:
            rate_config = self.rate_limits[user_tier]
        
        # Use Redis if available, otherwise in-memory
        if self.redis_client:
            return await self._check_redis_limit(limit_type, identifier, rate_config)
        else:
            return await self._check_memory_limit(limit_type, identifier, rate_config)
    
    async def _check_redis_limit(self,
                               limit_type: str,
                               identifier: str,
                               rate_config: RateLimit) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check rate limit using Redis"""
        try:
            now = int(time.time())
            
            # Check minute window
            minute_key = f"rate_limit:{limit_type}:{identifier}:minute:{now // 60}"
            minute_count = await self.redis_client.incr(minute_key)
            
            if minute_count == 1:
                await self.redis_client.expire(minute_key, 60)
            
            if minute_count > rate_config.requests_per_minute:
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded for {limit_type}",
                    "limit": rate_config.requests_per_minute,
                    "window": "minute",
                    "retry_after": 60 - (now % 60)
                }
            
            # Check hour window
            hour_key = f"rate_limit:{limit_type}:{identifier}:hour:{now // 3600}"
            hour_count = await self.redis_client.incr(hour_key)
            
            if hour_count == 1:
                await self.redis_client.expire(hour_key, 3600)
            
            if hour_count > rate_config.requests_per_hour:
                return False, {
                    "error": "rate_limit_exceeded",
                    "message": f"Hourly rate limit exceeded for {limit_type}",
                    "limit": rate_config.requests_per_hour,
                    "window": "hour",
                    "retry_after": 3600 - (now % 3600)
                }
            
            # Token bucket for burst protection
            bucket_key = f"rate_limit:bucket:{limit_type}:{identifier}"
            bucket_data = await self.redis_client.get(bucket_key)
            
            if bucket_data:
                bucket = TokenBucket(**json.loads(bucket_data))
            else:
                bucket = TokenBucket(
                    capacity=rate_config.burst_capacity,
                    tokens=rate_config.burst_capacity,
                    refill_rate=rate_config.requests_per_minute / 60.0
                )
            
            if not bucket.consume():
                wait_time = bucket.get_wait_time()
                return False, {
                    "error": "rate_limit_exceeded", 
                    "message": f"Burst limit exceeded for {limit_type}",
                    "limit": rate_config.burst_capacity,
                    "window": "burst",
                    "retry_after": int(wait_time) + 1
                }
            
            # Update bucket in Redis
            bucket_dict = {
                "capacity": bucket.capacity,
                "tokens": bucket.tokens,
                "refill_rate": bucket.refill_rate,
                "last_refill": bucket.last_refill
            }
            await self.redis_client.setex(
                bucket_key, 
                3600,  # Expire after 1 hour
                json.dumps(bucket_dict)
            )
            
            return True, None
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fall back to in-memory
            return await self._check_memory_limit(limit_type, identifier, rate_config)
    
    async def _check_memory_limit(self,
                                limit_type: str,
                                identifier: str, 
                                rate_config: RateLimit) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check rate limit using in-memory storage"""
        now = time.time()
        key = f"{limit_type}:{identifier}"
        
        # Clean old windows
        self._cleanup_old_windows()
        
        # Check minute window
        minute_window = int(now // 60)
        minute_key = f"{key}:minute:{minute_window}"
        
        self.request_counts[minute_key]["count"] += 1
        self.window_start_times[minute_key]["start"] = now
        
        if self.request_counts[minute_key]["count"] > rate_config.requests_per_minute:
            return False, {
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded for {limit_type}",
                "limit": rate_config.requests_per_minute,
                "window": "minute",
                "retry_after": 60 - int(now % 60)
            }
        
        # Token bucket check
        bucket_key = f"bucket:{key}"
        if bucket_key not in self.token_buckets:
            self.token_buckets[bucket_key] = TokenBucket(
                capacity=rate_config.burst_capacity,
                tokens=rate_config.burst_capacity,
                refill_rate=rate_config.requests_per_minute / 60.0
            )
        
        bucket = self.token_buckets[bucket_key]
        if not bucket.consume():
            wait_time = bucket.get_wait_time()
            return False, {
                "error": "rate_limit_exceeded",
                "message": f"Burst limit exceeded for {limit_type}",
                "limit": rate_config.burst_capacity, 
                "window": "burst",
                "retry_after": int(wait_time) + 1
            }
        
        return True, None
    
    def _cleanup_old_windows(self):
        """Clean up old time windows from memory"""
        now = time.time()
        cutoff = now - 3600  # Keep 1 hour of data
        
        # Clean request counts
        old_keys = [
            key for key in self.request_counts.keys()
            if key in self.window_start_times and 
               self.window_start_times[key].get("start", 0) < cutoff
        ]
        
        for key in old_keys:
            del self.request_counts[key]
            if key in self.window_start_times:
                del self.window_start_times[key]
        
        # Clean old token buckets
        old_buckets = [
            key for key, bucket in self.token_buckets.items()
            if bucket.last_refill < cutoff
        ]
        
        for key in old_buckets:
            del self.token_buckets[key]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
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
    
    async def get_rate_limit_status(self,
                                  request: Request,
                                  user_id: Optional[str] = None,
                                  user_tier: UserTier = UserTier.ANONYMOUS) -> Dict[str, Any]:
        """Get current rate limit status"""
        client_ip = self._get_client_ip(request)
        rate_config = self.rate_limits[user_tier]
        
        status = {
            "limits": {
                "requests_per_minute": rate_config.requests_per_minute,
                "requests_per_hour": rate_config.requests_per_hour,
                "requests_per_day": rate_config.requests_per_day,
                "burst_capacity": rate_config.burst_capacity
            },
            "remaining": {},
            "reset_times": {}
        }
        
        # Get remaining limits (simplified version)
        now = int(time.time())
        
        if self.redis_client:
            try:
                # Minute remaining
                minute_key = f"rate_limit:ip:{client_ip}:minute:{now // 60}"
                minute_used = await self.redis_client.get(minute_key)
                minute_used = int(minute_used) if minute_used else 0
                
                status["remaining"]["minute"] = max(0, rate_config.requests_per_minute - minute_used)
                status["reset_times"]["minute"] = 60 - (now % 60)
                
            except Exception as e:
                logger.error(f"Failed to get rate limit status from Redis: {e}")
                status["remaining"]["minute"] = rate_config.requests_per_minute
                status["reset_times"]["minute"] = 60 - (now % 60)
        else:
            # In-memory status
            minute_window = int(now // 60)
            minute_key = f"ip:{client_ip}:minute:{minute_window}"
            minute_used = self.request_counts[minute_key]["count"]
            
            status["remaining"]["minute"] = max(0, rate_config.requests_per_minute - minute_used)
            status["reset_times"]["minute"] = 60 - int(now % 60)
        
        return status


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

async def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            # Try to connect to Redis
            redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await redis_client.ping()
            _rate_limiter = RateLimiter(redis_client=redis_client)
            logger.info("Rate limiter initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
            _rate_limiter = RateLimiter()
    
    return _rate_limiter


# FastAPI middleware
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for FastAPI"""
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Get user context (simplified - integrate with your auth system)
    user_id = request.headers.get("X-User-ID")
    user_tier_header = request.headers.get("X-User-Tier", "anonymous")
    
    try:
        user_tier = UserTier(user_tier_header)
    except ValueError:
        user_tier = UserTier.ANONYMOUS
    
    # Check rate limits
    rate_limiter = await get_rate_limiter()
    is_allowed, error_info = await rate_limiter.check_rate_limit(
        request=request,
        user_id=user_id,
        user_tier=user_tier,
        endpoint=request.url.path
    )
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content=error_info,
            headers={
                "Retry-After": str(error_info.get("retry_after", 60)),
                "X-RateLimit-Limit": str(error_info.get("limit", 0)),
                "X-RateLimit-Remaining": "0"
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    try:
        status = await rate_limiter.get_rate_limit_status(request, user_id, user_tier)
        response.headers["X-RateLimit-Limit"] = str(status["limits"]["requests_per_minute"])
        response.headers["X-RateLimit-Remaining"] = str(status["remaining"]["minute"])
        response.headers["X-RateLimit-Reset"] = str(status["reset_times"]["minute"])
    except Exception as e:
        logger.error(f"Failed to add rate limit headers: {e}")
    
    return response


# Export main classes and functions
__all__ = [
    'RateLimiter',
    'RateLimit',
    'UserTier',
    'RateLimitType', 
    'DDoSDetector',
    'TokenBucket',
    'get_rate_limiter',
    'rate_limit_middleware'
]
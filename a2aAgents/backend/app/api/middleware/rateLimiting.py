"""
API Rate Limiting Middleware with Agent-Specific Limits
Implements comprehensive rate limiting for A2A agent communications
"""

import time
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import hashlib
import secrets

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import jwt


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """
    Advanced rate limiter with agent-specific limits, burst protection,
    and intelligent throttling for A2A communications
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.fallback_storage = {}  # In-memory fallback
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

        # Rate limit tiers for different agent types
        self.rate_limits = {
            "agent": {
                "requests_per_minute": 100,
                "requests_per_hour": 2000,
                "burst_limit": 10,
                "burst_window": 60  # seconds
            },
            "data_manager": {
                "requests_per_minute": 200,
                "requests_per_hour": 5000,
                "burst_limit": 20,
                "burst_window": 60
            },
            "catalog_manager": {
                "requests_per_minute": 150,
                "requests_per_hour": 3000,
                "burst_limit": 15,
                "burst_window": 60
            },
            "registry": {
                "requests_per_minute": 500,
                "requests_per_hour": 10000,
                "burst_limit": 50,
                "burst_window": 60
            },
            "default": {
                "requests_per_minute": 50,
                "requests_per_hour": 1000,
                "burst_limit": 5,
                "burst_window": 60
            }
        }

    async def initialize(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            if REDIS_AVAILABLE:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("✅ Redis rate limiter initialized")
            else:
                logger.warning("Redis not available, using in-memory rate limiting")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}, using in-memory fallback")
            self.redis_client = None

    async def check_rate_limit(
        self,
        agent_id: str,
        agent_type: str = "default",
        endpoint: str = "general"
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits
        Returns (allowed, limit_info)
        """
        try:
            current_time = time.time()

            # Get rate limit configuration
            limits = self.rate_limits.get(agent_type, self.rate_limits["default"])

            # Create rate limit keys
            minute_key = f"rate_limit:{agent_id}:minute:{int(current_time // 60)}"
            hour_key = f"rate_limit:{agent_id}:hour:{int(current_time // 3600)}"
            burst_key = f"rate_limit:{agent_id}:burst:{int(current_time // limits['burst_window'])}"

            # Check rate limits
            if self.redis_client:
                allowed, limit_info = await self._check_redis_limits(
                    minute_key, hour_key, burst_key, limits
                )
            else:
                allowed, limit_info = await self._check_memory_limits(
                    agent_id, limits, current_time
                )

            # Add metadata
            limit_info.update({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "endpoint": endpoint,
                "timestamp": current_time,
                "limits": limits
            })

            if not allowed:
                logger.warning(f"Rate limit exceeded for agent {agent_id} ({agent_type})")
                await self._log_rate_limit_violation(agent_id, agent_type, endpoint, limit_info)

            return allowed, limit_info

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request but log error
            return True, {"error": str(e), "fallback": True}

    async def _check_redis_limits(
        self,
        minute_key: str,
        hour_key: str,
        burst_key: str,
        limits: Dict
    ) -> Tuple[bool, Dict]:
        """Check rate limits using Redis"""
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()

            # Get current counts
            pipe.get(minute_key)
            pipe.get(hour_key)
            pipe.get(burst_key)

            results = await pipe.execute()

            minute_count = int(results[0] or 0)
            hour_count = int(results[1] or 0)
            burst_count = int(results[2] or 0)

            # Check limits
            minute_exceeded = minute_count >= limits["requests_per_minute"]
            hour_exceeded = hour_count >= limits["requests_per_hour"]
            burst_exceeded = burst_count >= limits["burst_limit"]

            allowed = not (minute_exceeded or hour_exceeded or burst_exceeded)

            if allowed:
                # Increment counters atomically
                pipe = self.redis_client.pipeline()
                pipe.incr(minute_key)
                pipe.expire(minute_key, 60)
                pipe.incr(hour_key)
                pipe.expire(hour_key, 3600)
                pipe.incr(burst_key)
                pipe.expire(burst_key, limits["burst_window"])
                await pipe.execute()

            return allowed, {
                "minute_count": minute_count + (1 if allowed else 0),
                "hour_count": hour_count + (1 if allowed else 0),
                "burst_count": burst_count + (1 if allowed else 0),
                "minute_limit": limits["requests_per_minute"],
                "hour_limit": limits["requests_per_hour"],
                "burst_limit": limits["burst_limit"],
                "minute_exceeded": minute_exceeded,
                "hour_exceeded": hour_exceeded,
                "burst_exceeded": burst_exceeded
            }

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            raise

    async def _check_memory_limits(
        self,
        agent_id: str,
        limits: Dict,
        current_time: float
    ) -> Tuple[bool, Dict]:
        """Check rate limits using in-memory storage (fallback)"""
        try:
            # Cleanup old entries periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                await self._cleanup_memory_storage(current_time)
                self.last_cleanup = current_time

            if agent_id not in self.fallback_storage:
                self.fallback_storage[agent_id] = {
                    "minute_requests": [],
                    "hour_requests": [],
                    "burst_requests": []
                }

            agent_data = self.fallback_storage[agent_id]

            # Clean old requests
            minute_cutoff = current_time - 60
            hour_cutoff = current_time - 3600
            burst_cutoff = current_time - limits["burst_window"]

            agent_data["minute_requests"] = [
                t for t in agent_data["minute_requests"] if t > minute_cutoff
            ]
            agent_data["hour_requests"] = [
                t for t in agent_data["hour_requests"] if t > hour_cutoff
            ]
            agent_data["burst_requests"] = [
                t for t in agent_data["burst_requests"] if t > burst_cutoff
            ]

            # Check limits
            minute_count = len(agent_data["minute_requests"])
            hour_count = len(agent_data["hour_requests"])
            burst_count = len(agent_data["burst_requests"])

            minute_exceeded = minute_count >= limits["requests_per_minute"]
            hour_exceeded = hour_count >= limits["requests_per_hour"]
            burst_exceeded = burst_count >= limits["burst_limit"]

            allowed = not (minute_exceeded or hour_exceeded or burst_exceeded)

            if allowed:
                # Record new request
                agent_data["minute_requests"].append(current_time)
                agent_data["hour_requests"].append(current_time)
                agent_data["burst_requests"].append(current_time)

            return allowed, {
                "minute_count": minute_count + (1 if allowed else 0),
                "hour_count": hour_count + (1 if allowed else 0),
                "burst_count": burst_count + (1 if allowed else 0),
                "minute_limit": limits["requests_per_minute"],
                "hour_limit": limits["requests_per_hour"],
                "burst_limit": limits["burst_limit"],
                "minute_exceeded": minute_exceeded,
                "hour_exceeded": hour_exceeded,
                "burst_exceeded": burst_exceeded,
                "storage": "memory"
            }

        except Exception as e:
            logger.error(f"Memory rate limit check failed: {e}")
            raise

    async def _cleanup_memory_storage(self, current_time: float):
        """Clean up old entries from memory storage"""
        try:
            cutoff_time = current_time - 3600  # Keep 1 hour of data

            for agent_id in list(self.fallback_storage.keys()):
                agent_data = self.fallback_storage[agent_id]

                # Clean all request lists
                agent_data["minute_requests"] = [
                    t for t in agent_data["minute_requests"] if t > cutoff_time
                ]
                agent_data["hour_requests"] = [
                    t for t in agent_data["hour_requests"] if t > cutoff_time
                ]
                agent_data["burst_requests"] = [
                    t for t in agent_data["burst_requests"] if t > cutoff_time
                ]

                # Remove empty entries
                if not any([
                    agent_data["minute_requests"],
                    agent_data["hour_requests"],
                    agent_data["burst_requests"]
                ]):
                    del self.fallback_storage[agent_id]

            logger.debug(f"Cleaned up rate limit storage, {len(self.fallback_storage)} agents remaining")

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    async def _log_rate_limit_violation(
        self,
        agent_id: str,
        agent_type: str,
        endpoint: str,
        limit_info: Dict
    ):
        """Log rate limit violations for monitoring"""
        try:
            violation_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "agent_type": agent_type,
                "endpoint": endpoint,
                "limit_info": limit_info,
                "violation_type": []
            }

            if limit_info.get("minute_exceeded"):
                violation_data["violation_type"].append("minute_limit")
            if limit_info.get("hour_exceeded"):
                violation_data["violation_type"].append("hour_limit")
            if limit_info.get("burst_exceeded"):
                violation_data["violation_type"].append("burst_limit")

            # Store in Redis for monitoring if available
            if self.redis_client:
                violation_key = f"rate_limit_violations:{datetime.utcnow().strftime('%Y-%m-%d')}"
                await self.redis_client.lpush(violation_key, json.dumps(violation_data))
                await self.redis_client.expire(violation_key, 86400 * 7)  # Keep for 7 days

            logger.warning(f"Rate limit violation: {json.dumps(violation_data)}")

        except Exception as e:
            logger.error(f"Failed to log rate limit violation: {e}")

    async def get_rate_limit_status(self, agent_id: str) -> Dict:
        """Get current rate limit status for an agent"""
        try:
            current_time = time.time()

            if self.redis_client:
                # Get current counts from Redis
                minute_key = f"rate_limit:{agent_id}:minute:{int(current_time // 60)}"
                hour_key = f"rate_limit:{agent_id}:hour:{int(current_time // 3600)}"

                pipe = self.redis_client.pipeline()
                pipe.get(minute_key)
                pipe.get(hour_key)
                results = await pipe.execute()

                return {
                    "agent_id": agent_id,
                    "minute_count": int(results[0] or 0),
                    "hour_count": int(results[1] or 0),
                    "timestamp": current_time
                }
            else:
                # Get from memory storage
                if agent_id in self.fallback_storage:
                    agent_data = self.fallback_storage[agent_id]
                    minute_cutoff = current_time - 60
                    hour_cutoff = current_time - 3600

                    minute_count = len([t for t in agent_data["minute_requests"] if t > minute_cutoff])
                    hour_count = len([t for t in agent_data["hour_requests"] if t > hour_cutoff])

                    return {
                        "agent_id": agent_id,
                        "minute_count": minute_count,
                        "hour_count": hour_count,
                        "timestamp": current_time,
                        "storage": "memory"
                    }
                else:
                    return {
                        "agent_id": agent_id,
                        "minute_count": 0,
                        "hour_count": 0,
                        "timestamp": current_time,
                        "storage": "memory"
                    }

        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {"error": str(e)}


class APIKeyManager:
    """
    Secure API key management with automatic rotation and revocation
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self.redis_client = None
        self.fallback_storage = {}
        self.key_prefix = "api_key:"
        self.rotation_schedule_key = "key_rotation_schedule"

        # Key configuration
        self.key_length = 32
        self.default_ttl = 3600  # 1 hour
        self.rotation_warning_threshold = 300  # 5 minutes before expiry

    async def initialize(self):
        """Initialize Redis connection for key storage"""
        try:
            if REDIS_AVAILABLE:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("✅ Redis API key manager initialized")
            else:
                logger.warning("Redis not available, using in-memory key storage")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}, using in-memory fallback")
            self.redis_client = None

    async def generate_api_key(
        self,
        agent_id: str,
        agent_type: str = "agent",
        ttl: int = None
    ) -> Dict[str, any]:
        """Generate a new API key for an agent"""
        try:
            # Generate secure random key
            api_key = secrets.token_urlsafe(self.key_length)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            key_data = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "key_hash": key_hash,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "ttl": ttl,
                "revoked": False,
                "rotation_count": 0
            }

            # Store key data
            if self.redis_client:
                key_storage_key = f"{self.key_prefix}{key_hash}"
                await self.redis_client.setex(
                    key_storage_key,
                    ttl,
                    json.dumps(key_data)
                )

                # Store agent's current key reference
                agent_key = f"agent_key:{agent_id}"
                await self.redis_client.setex(agent_key, ttl, key_hash)

            else:
                # Fallback to memory storage
                self.fallback_storage[key_hash] = key_data
                self.fallback_storage[f"agent:{agent_id}"] = key_hash

            logger.info(f"Generated API key for agent {agent_id} ({agent_type}), expires at {expires_at}")

            return {
                "api_key": api_key,
                "expires_at": expires_at.isoformat(),
                "ttl": ttl,
                "agent_id": agent_id,
                "key_hash": key_hash[:8] + "..."  # Partial hash for identification
            }

        except Exception as e:
            logger.error(f"Failed to generate API key for {agent_id}: {e}")
            raise

    async def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """Validate an API key and return agent information"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            if self.redis_client:
                key_storage_key = f"{self.key_prefix}{key_hash}"
                key_data_json = await self.redis_client.get(key_storage_key)

                if not key_data_json:
                    return False, None

                key_data = json.loads(key_data_json)
            else:
                # Check memory storage
                if key_hash not in self.fallback_storage:
                    return False, None

                key_data = self.fallback_storage[key_hash]

                # Check expiry manually for memory storage
                expires_at = datetime.fromisoformat(key_data["expires_at"])
                if datetime.utcnow() > expires_at:
                    del self.fallback_storage[key_hash]
                    return False, None

            # Check if key is revoked
            if key_data.get("revoked", False):
                return False, None

            return True, key_data

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False, None

    async def rotate_api_key(self, agent_id: str) -> Dict[str, any]:
        """Rotate API key for an agent with grace period"""
        try:
            # Get current key
            old_key_info = await self.get_agent_key_info(agent_id)

            # Generate new key
            new_key_info = await self.generate_api_key(
                agent_id=agent_id,
                agent_type=old_key_info.get("agent_type", "agent") if old_key_info else "agent",
                ttl=self.default_ttl
            )

            # Keep old key valid for grace period (5 minutes)
            if old_key_info:
                grace_period = 300
                await self._extend_key_validity(old_key_info["key_hash"], grace_period)

                # Update rotation count
                if self.redis_client:
                    old_key_storage_key = f"{self.key_prefix}{old_key_info['key_hash']}"
                    old_key_data_json = await self.redis_client.get(old_key_storage_key)
                    if old_key_data_json:
                        old_key_data = json.loads(old_key_data_json)
                        old_key_data["rotation_count"] = old_key_data.get("rotation_count", 0) + 1
                        await self.redis_client.setex(
                            old_key_storage_key,
                            grace_period,
                            json.dumps(old_key_data)
                        )

            logger.info(f"API key rotated for agent {agent_id}")

            return {
                "new_key": new_key_info,
                "old_key_valid_until": (datetime.utcnow() + timedelta(seconds=300)).isoformat(),
                "grace_period_seconds": 300
            }

        except Exception as e:
            logger.error(f"Failed to rotate API key for {agent_id}: {e}")
            raise

    async def get_agent_key_info(self, agent_id: str) -> Optional[Dict]:
        """Get current key information for an agent"""
        try:
            if self.redis_client:
                agent_key = f"agent_key:{agent_id}"
                key_hash = await self.redis_client.get(agent_key)

                if not key_hash:
                    return None

                key_storage_key = f"{self.key_prefix}{key_hash}"
                key_data_json = await self.redis_client.get(key_storage_key)

                if not key_data_json:
                    return None

                return json.loads(key_data_json)
            else:
                # Check memory storage
                agent_key = f"agent:{agent_id}"
                if agent_key not in self.fallback_storage:
                    return None

                key_hash = self.fallback_storage[agent_key]
                if key_hash not in self.fallback_storage:
                    return None

                return self.fallback_storage[key_hash]

        except Exception as e:
            logger.error(f"Failed to get key info for {agent_id}: {e}")
            return None

    async def _extend_key_validity(self, key_hash: str, extension_seconds: int):
        """Extend the validity of a key (for grace periods)"""
        try:
            if self.redis_client:
                key_storage_key = f"{self.key_prefix}{key_hash}"
                await self.redis_client.expire(key_storage_key, extension_seconds)
            else:
                # For memory storage, update expiry time
                if key_hash in self.fallback_storage:
                    key_data = self.fallback_storage[key_hash]
                    new_expiry = datetime.utcnow() + timedelta(seconds=extension_seconds)
                    key_data["expires_at"] = new_expiry.isoformat()

        except Exception as e:
            logger.error(f"Failed to extend key validity: {e}")

    async def revoke_api_key(self, agent_id: str) -> bool:
        """Revoke API key for an agent"""
        try:
            key_info = await self.get_agent_key_info(agent_id)
            if not key_info:
                return False

            key_hash = key_info["key_hash"]

            if self.redis_client:
                key_storage_key = f"{self.key_prefix}{key_hash}"
                key_data_json = await self.redis_client.get(key_storage_key)

                if key_data_json:
                    key_data = json.loads(key_data_json)
                    key_data["revoked"] = True

                    # Update with remaining TTL
                    ttl = await self.redis_client.ttl(key_storage_key)
                    if ttl > 0:
                        await self.redis_client.setex(
                            key_storage_key,
                            ttl,
                            json.dumps(key_data)
                        )
            else:
                # Mark as revoked in memory storage
                if key_hash in self.fallback_storage:
                    self.fallback_storage[key_hash]["revoked"] = True

            logger.info(f"API key revoked for agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke API key for {agent_id}: {e}")
            return False


# Global instances
_rate_limiter = None
_key_manager = None

async def get_rate_limiter() -> APIRateLimiter:
    """Get or create the rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = APIRateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter

async def get_key_manager() -> APIKeyManager:
    """Get or create the API key manager instance"""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
        await _key_manager.initialize()
    return _key_manager


# FastAPI middleware functions
async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting"""
    try:
        # Extract agent information from request
        agent_id = request.headers.get("X-Agent-ID")
        agent_type = request.headers.get("X-Agent-Type", "default")
        api_key = request.headers.get("X-API-Key")

        if not agent_id or not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Missing agent ID or API key"}
            )

        # Validate API key
        key_manager = await get_key_manager()
        valid, key_data = await key_manager.validate_api_key(api_key)

        if not valid:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Invalid or expired API key"}
            )

        # Check rate limits
        rate_limiter = await get_rate_limiter()
        allowed, limit_info = await rate_limiter.check_rate_limit(
            agent_id=agent_id,
            agent_type=agent_type,
            endpoint=str(request.url.path)
        )

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "limit_info": limit_info,
                    "retry_after": 60  # seconds
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limit_info.get("minute_limit", 0)),
                    "X-RateLimit-Remaining": str(max(0, limit_info.get("minute_limit", 0) - limit_info.get("minute_count", 0))),
                    "X-RateLimit-Reset": str(int(time.time() + 60))
                }
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit_info.get("minute_limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit_info.get("minute_limit", 0) - limit_info.get("minute_count", 0)))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response

    except Exception as e:
        logger.error(f"Rate limiting middleware error: {e}")
        # Fail open - allow request but log error
        return await call_next(request)
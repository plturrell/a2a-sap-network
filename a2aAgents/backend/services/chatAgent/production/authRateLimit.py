"""
Production Authentication and Rate Limiting for A2A Chat Agent
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps
import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """
    Production authentication manager with JWT and API key support
    """
    
    def __init__(self, config: Dict[str, Any], database=None):
        self.config = config
        self.database = database
        
        # Generate secure JWT secret if not provided
        self.jwt_secret = self._get_or_generate_jwt_secret(config)
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        self.jwt_expiry_hours = config.get('jwt_expiry_hours', 24)
        self.api_key_header = config.get('api_key_header', 'X-API-Key')
        self.enable_jwt = config.get('enable_jwt', True)
        self.enable_api_key = config.get('enable_api_key', True)
        
        # OAuth2 config
        self.oauth2_providers = config.get('oauth2_providers', {})
        
        # Security scheme
        self.security = HTTPBearer()
    
    def _get_or_generate_jwt_secret(self, config: Dict[str, Any]) -> str:
        """Get JWT secret from environment or generate a secure one"""
        import os
        import secrets
        
        # Try environment variable first
        secret = os.getenv('A2A_JWT_SECRET')
        if secret:
            return secret
        
        # Try config file
        secret = config.get('jwt_secret')
        if secret and secret != 'your-secret-key':  # Reject default insecure value
            return secret
        
        # Generate secure secret and warn
        logger.warning("No secure JWT secret configured! Generating random secret. "
                      "Set A2A_JWT_SECRET environment variable for production.")
        return secrets.token_urlsafe(64)
        
    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_data['user_id'],
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'roles': user_data.get('roles', ['user']),
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            'iat': datetime.utcnow(),
            'iss': 'a2a-chat-agent'
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user"""
        # Generate random API key
        import secrets
        api_key = f"a2a_{user_id}_{secrets.token_urlsafe(32)}"
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        return self.hash_api_key(api_key) == stored_hash
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        api_key: Optional[str] = Header(None, alias="X-API-Key"),
        request: Request = None
    ) -> Dict[str, Any]:
        """Get current authenticated user from JWT or API key"""
        
        # Try JWT authentication first
        if credentials and self.enable_jwt:
            try:
                token = credentials.credentials
                user_data = self.verify_jwt_token(token)
                return user_data
            except HTTPException:
                if not api_key:
                    raise
        
        # Try API key authentication
        if api_key and self.enable_api_key:
            # Validate API key against database
            # This would check against the database
            user_data = await self._validate_api_key(api_key)
            if user_data:
                return user_data
        
        raise HTTPException(status_code=401, detail="Authentication required")
    
    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key against database"""
        try:
            # Check if database is available
            if hasattr(self, 'database') and self.database:
                # Query user by API key hash
                api_key_hash = self.hash_api_key(api_key)
                user_data = await self.database.get_user_by_api_key(api_key_hash)
                if user_data:
                    return {
                        "user_id": user_data['user_id'],
                        "username": user_data['username'],
                        "email": user_data.get('email'),
                        "roles": user_data.get('roles', ['user']),
                        "auth_method": "api_key",
                        "rate_limit_tier": user_data.get('rate_limit_tier', 'standard')
                    }
            else:
                # Enhanced fallback with security validation
                if api_key.startswith("a2a_"):
                    parts = api_key.split("_")
                    if len(parts) >= 3 and len(parts[2]) >= 32:  # Ensure reasonable token length
                        user_id = parts[1]
                        
                        # Additional validation: check if this is development mode
                        if self.config.get('environment', 'production') != 'development':
                            logger.error(f"API key fallback not allowed in production for user {user_id}")
                            return None
                        
                        logger.warning(f"Using fallback API key validation for user {user_id} (development mode only)")
                        return {
                            "user_id": user_id,
                            "username": f"dev_user_{user_id}",
                            "roles": ["user"],
                            "auth_method": "api_key_fallback",
                            "rate_limit_tier": "free"  # Restrict to free tier
                        }
                
                logger.error(f"Invalid API key format or insufficient security")
                return None
            return None
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
    
    def require_roles(self, required_roles: List[str]):
        """Decorator to require specific roles"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user from kwargs
                user = kwargs.get('current_user')
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                user_roles = user.get('roles', [])
                if not any(role in required_roles for role in user_roles):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


class RateLimiter:
    """
    Production rate limiter with Redis backend
    """
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        
        # Rate limit tiers
        self.tiers = {
            'free': {
                'requests_per_minute': 10,
                'requests_per_hour': 100,
                'requests_per_day': 1000,
                'concurrent_conversations': 3,
                'message_length': 1000
            },
            'standard': {
                'requests_per_minute': 30,
                'requests_per_hour': 500,
                'requests_per_day': 5000,
                'concurrent_conversations': 10,
                'message_length': 5000
            },
            'premium': {
                'requests_per_minute': 100,
                'requests_per_hour': 2000,
                'requests_per_day': 20000,
                'concurrent_conversations': 50,
                'message_length': 10000
            },
            'enterprise': {
                'requests_per_minute': 1000,
                'requests_per_hour': 50000,
                'requests_per_day': 500000,
                'concurrent_conversations': -1,  # Unlimited
                'message_length': 50000
            }
        }
        
    async def check_rate_limit(
        self, 
        user_id: str, 
        tier: str = 'standard',
        resource: str = 'api'
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user has exceeded rate limits
        Returns (allowed, limit_info)
        """
        tier_limits = self.tiers.get(tier, self.tiers['standard'])
        
        # Check multiple time windows
        checks = [
            ('minute', 60, tier_limits['requests_per_minute']),
            ('hour', 3600, tier_limits['requests_per_hour']),
            ('day', 86400, tier_limits['requests_per_day'])
        ]
        
        current_time = int(time.time())
        pipeline = self.redis.pipeline()
        
        for window_name, window_seconds, limit in checks:
            if limit <= 0:
                continue
                
            key = f"rate_limit:{user_id}:{resource}:{window_name}"
            window_start = current_time - window_seconds
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, window_start)
            # Count current entries
            pipeline.zcount(key, window_start, current_time)
            # Add current request
            pipeline.zadd(key, {str(current_time): current_time})
            # Set expiry
            pipeline.expire(key, window_seconds + 60)
        
        results = await pipeline.execute()
        
        # Check limits
        limit_info = {
            'tier': tier,
            'limits': tier_limits,
            'current_usage': {}
        }
        
        for i, (window_name, window_seconds, limit) in enumerate(checks):
            if limit <= 0:
                continue
                
            count_index = i * 4 + 1  # Every 4th result starting from index 1
            if count_index < len(results):
                current_count = results[count_index]
                limit_info['current_usage'][f'{window_name}'] = {
                    'used': current_count,
                    'limit': limit,
                    'remaining': max(0, limit - current_count)
                }
                
                if current_count >= limit:
                    return False, limit_info
        
        return True, limit_info
    
    async def get_concurrent_conversations(self, user_id: str) -> int:
        """Get number of active concurrent conversations"""
        key = f"active_conversations:{user_id}"
        conversations = await self.redis.smembers(key)
        return len(conversations)
    
    async def add_conversation(self, user_id: str, conversation_id: str):
        """Add active conversation"""
        key = f"active_conversations:{user_id}"
        await self.redis.sadd(key, conversation_id)
        # Expire after 24 hours
        await self.redis.expire(key, 86400)
    
    async def remove_conversation(self, user_id: str, conversation_id: str):
        """Remove active conversation"""
        key = f"active_conversations:{user_id}"
        await self.redis.srem(key, conversation_id)
    
    async def check_message_length(self, message: str, tier: str = 'standard') -> bool:
        """Check if message length is within tier limits"""
        tier_limits = self.tiers.get(tier, self.tiers['standard'])
        max_length = tier_limits['message_length']
        return len(message) <= max_length
    
    async def get_rate_limit_headers(
        self, 
        user_id: str, 
        tier: str = 'standard'
    ) -> Dict[str, str]:
        """Get rate limit headers for response"""
        tier_limits = self.tiers.get(tier, self.tiers['standard'])
        
        # Get current usage for minute window
        key = f"rate_limit:{user_id}:api:minute"
        current_time = int(time.time())
        window_start = current_time - 60
        
        current_count = await self.redis.zcount(key, window_start, current_time)
        limit = tier_limits['requests_per_minute']
        
        headers = {
            'X-RateLimit-Limit': str(limit),
            'X-RateLimit-Remaining': str(max(0, limit - current_count)),
            'X-RateLimit-Reset': str(current_time + 60),
            'X-RateLimit-Tier': tier
        }
        
        return headers


class CacheManager:
    """
    Production caching layer with Redis
    """
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.default_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.max_cache_size = config.get('max_cache_size', 1000)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(f"cache:{key}")
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ):
        """Set value in cache"""
        try:
            cache_key = f"cache:{key}"
            serialized = json.dumps(value)
            
            # Check cache size
            if len(serialized) > self.max_cache_size * 1024:  # KB to bytes
                logger.warning(f"Cache value too large for key {key}")
                return
            
            await self.redis.set(
                cache_key,
                serialized,
                ex=ttl or self.default_ttl
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        try:
            await self.redis.delete(f"cache:{key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    async def clear_pattern(self, pattern: str):
        """Clear cache keys matching pattern"""
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(
                    cursor,
                    match=f"cache:{pattern}*",
                    count=100
                )
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    # Specific cache methods
    async def get_agent_response(
        self, 
        prompt_hash: str, 
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached agent response"""
        key = f"agent_response:{agent_id}:{prompt_hash}"
        return await self.get(key)
    
    async def set_agent_response(
        self, 
        prompt_hash: str, 
        agent_id: str, 
        response: Dict[str, Any],
        ttl: int = 1800  # 30 minutes
    ):
        """Cache agent response"""
        key = f"agent_response:{agent_id}:{prompt_hash}"
        await self.set(key, response, ttl)
    
    async def get_conversation_context(
        self, 
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached conversation context"""
        key = f"conversation:{conversation_id}"
        return await self.get(key)
    
    async def set_conversation_context(
        self, 
        conversation_id: str, 
        context: Dict[str, Any],
        ttl: int = 3600  # 1 hour
    ):
        """Cache conversation context"""
        key = f"conversation:{conversation_id}"
        await self.set(key, context, ttl)
    
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences"""
        key = f"user_prefs:{user_id}"
        return await self.get(key)
    
    async def set_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any],
        ttl: int = 86400  # 24 hours
    ):
        """Cache user preferences"""
        key = f"user_prefs:{user_id}"
        await self.set(key, preferences, ttl)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.redis.info()
            return {
                'used_memory': info.get('used_memory_human', 'N/A'),
                'total_keys': await self.redis.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1))
                ) * 100
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}


class SecurityMiddleware:
    """
    Security middleware for production
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_cors = config.get('enable_cors', True)
        self.allowed_origins = config.get('allowed_origins', ['*'])
        self.enable_csrf = config.get('enable_csrf', False)
        self.secure_headers = config.get('secure_headers', True)
        
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers"""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()'
        }
        
        if self.enable_cors:
            headers['Access-Control-Allow-Origin'] = ', '.join(self.allowed_origins)
            headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-API-Key'
        
        return headers
    
    async def validate_request(self, request: Request) -> bool:
        """Validate incoming request"""
        # Check for common attack patterns
        suspicious_patterns = [
            '../', '<script', 'javascript:', 'onload=', 'onerror=',
            'SELECT * FROM', 'DROP TABLE', 'INSERT INTO',
            '\\x00', '\\x1a'  # Null bytes
        ]
        
        # Check URL
        url_str = str(request.url)
        for pattern in suspicious_patterns:
            if pattern in url_str:
                logger.warning(f"Suspicious pattern in URL: {pattern}")
                return False
        
        # Check headers
        for header_name, header_value in request.headers.items():
            if any(pattern in str(header_value) for pattern in suspicious_patterns):
                logger.warning(f"Suspicious pattern in header {header_name}")
                return False
        
        return True


# Rate limiting decorator
def rate_limit(tier: str = 'standard'):
    """Decorator for rate limiting endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and user from args/kwargs
            request = None
            user = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
            
            user = kwargs.get('current_user')
            
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Get rate limiter from request state
            rate_limiter = getattr(request.app.state, 'rate_limiter', None)
            if not rate_limiter:
                return await func(*args, **kwargs)
            
            # Check rate limit
            user_id = user.get('user_id')
            user_tier = user.get('tier', tier)
            
            allowed, limit_info = await rate_limiter.check_rate_limit(
                user_id, user_tier
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers=await rate_limiter.get_rate_limit_headers(user_id, user_tier)
                )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            if hasattr(response, 'headers'):
                headers = await rate_limiter.get_rate_limit_headers(user_id, user_tier)
                for key, value in headers.items():
                    response.headers[key] = value
            
            return response
        
        return wrapper
    return decorator
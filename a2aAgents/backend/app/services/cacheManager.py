"""
3-Tier Cache Manager for BDC Core (Data Manager Agent)
Implements L1 (in-memory), L2 (Redis), and L3 (database) caching layers.
"""

import hashlib
import json
import os
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio

import redis.asyncio as redis
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration settings"""
    l1_max_size: int = 10000
    l1_default_ttl: int = 900  # 15 minutes
    l2_default_ttl: int = 3600  # 1 hour
    l3_default_ttl: int = 14400  # 4 hours
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True


class CacheEntry(BaseModel):
    """Cache entry with metadata"""
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    cache_level: int = 1


class CacheManager:
    """
    3-Tier Cache Manager for BDC Core
    
    L1: In-memory LRU cache for hot data
    L2: Redis distributed cache for processed results
    L3: Database cache tables for long-term storage
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l1_access_order: List[str] = []
        self._redis_client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize cache connections"""
        if self.config.enable_l2:
            try:
                self._redis_client = redis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=False
                )
                await self._redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.config.enable_l2 = False
    
    async def close(self):
        """Close cache connections"""
        if self._redis_client:
            await self._redis_client.close()
    
    def _generate_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate cache key with namespace and parameters"""
        key_data = {"key": key, **kwargs}
        key_hash = hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"{namespace}:{key_hash}"
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > entry.expires_at
    
    async def _l1_get(self, cache_key: str) -> Optional[Any]:
        """Get from L1 cache"""
        if not self.config.enable_l1:
            return None
            
        async with self._lock:
            if cache_key in self._l1_cache:
                entry = self._l1_cache[cache_key]
                if self._is_expired(entry):
                    del self._l1_cache[cache_key]
                    if cache_key in self._l1_access_order:
                        self._l1_access_order.remove(cache_key)
                    return None
                
                # Update access
                entry.access_count += 1
                if cache_key in self._l1_access_order:
                    self._l1_access_order.remove(cache_key)
                self._l1_access_order.append(cache_key)
                
                return entry.data
        return None
    
    async def _l1_set(self, cache_key: str, value: Any, ttl: int):
        """Set in L1 cache with LRU eviction"""
        if not self.config.enable_l1:
            return
            
        async with self._lock:
            # Evict if at capacity
            while len(self._l1_cache) >= self.config.l1_max_size:
                if self._l1_access_order:
                    oldest_key = self._l1_access_order.pop(0)
                    self._l1_cache.pop(oldest_key, None)
                else:
                    break
            
            # Add new entry
            entry = CacheEntry(
                data=value,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl),
                cache_level=1
            )
            self._l1_cache[cache_key] = entry
            self._l1_access_order.append(cache_key)
    
    async def _l2_get(self, cache_key: str) -> Optional[Any]:
        """Get from L2 cache (Redis)"""
        if not self.config.enable_l2 or not self._redis_client:
            return None
            
        try:
            data = await self._redis_client.get(cache_key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
        return None
    
    async def _l2_set(self, cache_key: str, value: Any, ttl: int):
        """Set in L2 cache (Redis)"""
        if not self.config.enable_l2 or not self._redis_client:
            return
            
        try:
            serialized = pickle.dumps(value)
            await self._redis_client.setex(cache_key, ttl, serialized)
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
    
    async def _l3_get(self, cache_key: str) -> Optional[Any]:
        """Get from L3 cache (Database)"""
        if not self.config.enable_l3:
            return None
            
        try:
            # Use the new database-backed L3 cache
            from .l3DatabaseCache import get_l3_database_cache
            l3_cache = await get_l3_database_cache()
            return await l3_cache.get(cache_key)
                
        except Exception as e:
            logger.error(f"L3 cache get error: {e}")
            return None
    
    async def _l3_set(self, cache_key: str, value: Any, ttl: int):
        """Set in L3 cache (Database)"""
        if not self.config.enable_l3:
            return
            
        try:
            # Use the new database-backed L3 cache
            from .l3DatabaseCache import get_l3_database_cache
            l3_cache = await get_l3_database_cache()
            await l3_cache.set(cache_key, value, ttl)
                
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
    
    async def get(
        self, 
        namespace: str, 
        key: str, 
        max_level: int = 2,
        **kwargs
    ) -> Optional[Any]:
        """
        Get value from cache with fallback through levels
        
        Args:
            namespace: Cache namespace (e.g., "ord", "query", "std")
            key: Cache key
            max_level: Maximum cache level to check (1=L1 only, 2=L1+L2, 3=L1+L2+L3)
            **kwargs: Additional key parameters for cache key generation
        """
        cache_key = self._generate_key(namespace, key, **kwargs)
        
        # Try L1 first
        if max_level >= 1:
            result = await self._l1_get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit L1: {cache_key}")
                return result
        
        # Try L2 next
        if max_level >= 2:
            result = await self._l2_get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit L2: {cache_key}")
                # Populate L1 for next access
                await self._l1_set(cache_key, result, self.config.l1_default_ttl)
                return result
        
        # L3 database cache implementation
        if max_level >= 3 and self.config.enable_l3:
            result = await self._l3_get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit L3: {cache_key}")
                # Populate L1 and L2 for next access
                await self._l1_set(cache_key, result, self.config.l1_default_ttl)
                await self._l2_set(cache_key, result, self.config.l2_default_ttl)
                return result
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: int = 2,
        **kwargs
    ):
        """
        Set value in cache at specified level(s)
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses defaults if None)
            level: Cache level to write to (1=L1 only, 2=L1+L2, 3=L1+L2+L3)
            **kwargs: Additional key parameters
        """
        cache_key = self._generate_key(namespace, key, **kwargs)
        
        # Determine TTLs
        l1_ttl = ttl or self.config.l1_default_ttl
        l2_ttl = ttl or self.config.l2_default_ttl
        l3_ttl = ttl or self.config.l3_default_ttl
        
        # Write to appropriate levels
        if level >= 1:
            await self._l1_set(cache_key, value, l1_ttl)
        
        if level >= 2:
            await self._l2_set(cache_key, value, l2_ttl)
        
        if level >= 3:
            await self._l3_set(cache_key, value, l3_ttl)
        
        logger.debug(f"Cache set L{level}: {cache_key}")
    
    async def invalidate(self, namespace: str, key: str = None, **kwargs):
        """Invalidate cache entries"""
        if key:
            cache_key = self._generate_key(namespace, key, **kwargs)
            # Remove from L1
            async with self._lock:
                self._l1_cache.pop(cache_key, None)
                if cache_key in self._l1_access_order:
                    self._l1_access_order.remove(cache_key)
            
            # Remove from L2
            if self._redis_client:
                try:
                    await self._redis_client.delete(cache_key)
                except Exception as e:
                    logger.error(f"L2 cache invalidate error: {e}")
        else:
            # Invalidate entire namespace
            if self._redis_client:
                try:
                    pattern = f"{namespace}:*"
                    keys = await self._redis_client.keys(pattern)
                    if keys:
                        await self._redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"L2 namespace invalidate error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "l1_size": len(self._l1_cache),
            "l1_max_size": self.config.l1_max_size,
            "l1_enabled": self.config.enable_l1,
            "l2_enabled": self.config.enable_l2,
            "l3_enabled": self.config.enable_l3,
        }
        
        if self._redis_client:
            try:
                info = await self._redis_client.info("memory")
                stats["l2_memory_used"] = info.get("used_memory", 0)
                stats["l2_memory_peak"] = info.get("used_memory_peak", 0)
            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
        
        # Get L3 database cache stats
        if self.config.enable_l3:
            try:
                from .l3DatabaseCache import get_l3_database_cache
                l3_cache = await get_l3_database_cache()
                l3_stats = await l3_cache.get_stats()
                stats["l3_stats"] = l3_stats
            except Exception as e:
                logger.error(f"Failed to get L3 cache stats: {e}")
        
        return stats


# Decorators for easy caching
def cache_result(
    namespace: str,
    ttl: int = 3600,
    level: int = 2,
    key_func: Optional[callable] = None
):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get cache manager from first argument (should be self with cache_manager)
            if args and hasattr(args[0], 'cache_manager'):
                cache_manager = args[0].cache_manager
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args[1:]) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = await cache_manager.get(namespace, cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await cache_manager.set(namespace, cache_key, result, ttl, level)
                return result
            else:
                # Fallback to normal execution if no cache manager
                return await func(*args, **kwargs)
        return wrapper
    return decorator
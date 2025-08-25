"""
Enhanced cache manager with TTL and LRU eviction
"""

import asyncio
import time
from typing import Any, Dict, Optional, Union
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json

class LRUCache:
    """Thread-safe LRU cache with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key not in self.cache:
                return None

            item = self.cache[key]
            # Check if expired
            if item['expires_at'] < time.time():
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return item['value']

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        async with self.lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + (ttl or self.default_ttl)
            }

    async def delete(self, key: str):
        """Remove item from cache"""
        async with self.lock:
            self.cache.pop(key, None)

    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()

    async def cleanup_expired(self):
        """Remove expired entries"""
        async with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, item in self.cache.items()
                if item['expires_at'] < current_time
            ]
            for key in expired_keys:
                del self.cache[key]


class CacheManager:
    """Centralized cache management with multiple strategies"""

    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
        self.cleanup_task = None

    def get_cache(self, name: str, max_size: int = 1000, default_ttl: int = 300) -> LRUCache:
        """Get or create named cache"""
        if name not in self.caches:
            self.caches[name] = LRUCache(max_size, default_ttl)
        return self.caches[name]

    async def start_cleanup_task(self, interval: int = 60):
        """Start periodic cleanup of expired entries"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                for cache in self.caches.values():
                    await cache.cleanup_expired()

        self.cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    def cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort params for consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        hash_value = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{hash_value}"


# Global cache manager instance
cache_manager = CacheManager()

# Decorator for caching async functions
def cached(cache_name: str, ttl: int = 300):
    """Decorator to cache async function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            cache = cache_manager.get_cache(cache_name)
            cache_key = cache_manager.cache_key(func.__name__, key_data)

            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator
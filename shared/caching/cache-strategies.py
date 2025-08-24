#!/usr/bin/env python3
"""
Advanced Caching Strategies for A2A Platform
Provides intelligent caching patterns with Redis backend integration
"""

import asyncio
import json
import logging
import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Callable, AsyncCallable
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager


class CacheStrategy(Enum):
    """Cache strategy types"""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"
    REFRESH_AHEAD = "refresh_ahead"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    RANDOM = "random"


@dataclass
class CacheConfig:
    """Cache configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "a2a:"
    default_ttl: int = 3600
    max_connections: int = 10
    strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression_enabled: bool = True
    metrics_enabled: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_requests: int = 0
    last_reset: datetime = None
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def reset(self):
        """Reset all metrics"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_requests = 0
        self.last_reset = datetime.now()


class CircuitBreaker:
    """Circuit breaker for cache operations"""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = "open"


class A2ACacheManager:
    """Advanced cache manager with multiple strategies and intelligent features"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        ) if config.circuit_breaker_enabled else None
        self.logger = logging.getLogger(__name__)
        self._write_behind_queue = asyncio.Queue()
        self._write_behind_task = None
        self._refresh_ahead_tasks = {}
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Start write-behind worker if needed
            if self.config.strategy == CacheStrategy.WRITE_BEHIND:
                self._write_behind_task = asyncio.create_task(self._write_behind_worker())
            
            self.logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection and cleanup"""
        if self._write_behind_task:
            self._write_behind_task.cancel()
            
        # Cancel refresh-ahead tasks
        for task in self._refresh_ahead_tasks.values():
            task.cancel()
        
        if self.redis_client:
            await self.redis_client.aclose()
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix"""
        return f"{self.config.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.config.compression_enabled:
            # Use pickle for Python objects with compression
            import zlib
            serialized = pickle.dumps(value)
            return zlib.compress(serialized)
        else:
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.config.compression_enabled:
            import zlib
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        else:
            return pickle.loads(data)
    
    def _hash_key(self, key: str) -> str:
        """Create hash of key for consistent caching"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def _execute_with_circuit_breaker(self, operation: Callable):
        """Execute operation with circuit breaker protection"""
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open")
        
        try:
            result = await operation()
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            return result
        except Exception as e:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise e
    
    # Core cache operations
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        if not self.redis_client:
            return default
        
        self.metrics.total_requests += 1
        
        try:
            async def _get_operation():
                full_key = self._build_key(key)
                data = await self.redis_client.get(full_key)
                
                if data is None:
                    self.metrics.misses += 1
                    return default
                
                self.metrics.hits += 1
                return self._deserialize(data)
            
            if self.circuit_breaker:
                return await self._execute_with_circuit_breaker(_get_operation)
            else:
                return await _get_operation()
                
        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.redis_client:
            return False
        
        self.metrics.total_requests += 1
        
        try:
            async def _set_operation():
                full_key = self._build_key(key)
                serialized_value = self._serialize(value)
                
                ttl_value = ttl or self.config.default_ttl
                if ttl_value > 0:
                    await self.redis_client.setex(full_key, ttl_value, serialized_value)
                else:
                    await self.redis_client.set(full_key, serialized_value)
                
                self.metrics.sets += 1
                return True
            
            if self.circuit_breaker:
                return await self._execute_with_circuit_breaker(_set_operation)
            else:
                return await _set_operation()
                
        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis_client:
            return False
        
        self.metrics.total_requests += 1
        
        try:
            async def _delete_operation():
                full_key = self._build_key(key)
                result = await self.redis_client.delete(full_key)
                self.metrics.deletes += 1
                return result > 0
            
            if self.circuit_breaker:
                return await self._execute_with_circuit_breaker(_delete_operation)
            else:
                return await _delete_operation()
                
        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            full_key = self._build_key(key)
            result = await self.redis_client.exists(full_key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    # Advanced caching patterns
    async def get_or_set(self, key: str, fetcher: AsyncCallable, ttl: Optional[int] = None, **kwargs) -> Any:
        """Get value from cache or fetch and set if not present"""
        value = await self.get(key)
        
        if value is None:
            # Use distributed lock to prevent cache stampede
            lock_key = f"lock:{key}"
            async with self._distributed_lock(lock_key, timeout=10):
                # Double-check after acquiring lock
                value = await self.get(key)
                if value is None:
                    if asyncio.iscoroutinefunction(fetcher):
                        value = await fetcher(**kwargs)
                    else:
                        value = fetcher(**kwargs)
                    await self.set(key, value, ttl)
        
        return value
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.redis_client or not keys:
            return {}
        
        try:
            full_keys = [self._build_key(key) for key in keys]
            results = await self.redis_client.mget(full_keys)
            
            values = {}
            for i, (key, result) in enumerate(zip(keys, results)):
                if result is not None:
                    values[key] = self._deserialize(result)
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1
            
            self.metrics.total_requests += len(keys)
            return values
            
        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(f"Cache mget error: {e}")
            return {}
    
    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs"""
        if not self.redis_client or not mapping:
            return False
        
        try:
            pipe = self.redis_client.pipeline()
            ttl_value = ttl or self.config.default_ttl
            
            for key, value in mapping.items():
                full_key = self._build_key(key)
                serialized_value = self._serialize(value)
                
                if ttl_value > 0:
                    pipe.setex(full_key, ttl_value, serialized_value)
                else:
                    pipe.set(full_key, serialized_value)
            
            await pipe.execute()
            self.metrics.sets += len(mapping)
            self.metrics.total_requests += len(mapping)
            return True
            
        except Exception as e:
            self.metrics.errors += 1
            self.logger.error(f"Cache mset error: {e}")
            return False
    
    # Strategy implementations
    async def cache_aside_get(self, key: str, fetcher: AsyncCallable, ttl: Optional[int] = None, **kwargs) -> Any:
        """Cache-aside pattern implementation"""
        return await self.get_or_set(key, fetcher, ttl, **kwargs)
    
    async def write_through_set(self, key: str, value: Any, persistor: AsyncCallable, ttl: Optional[int] = None, **kwargs) -> bool:
        """Write-through pattern implementation"""
        try:
            # Write to data store first
            if asyncio.iscoroutinefunction(persistor):
                await persistor(key, value, **kwargs)
            else:
                persistor(key, value, **kwargs)
            
            # Then write to cache
            return await self.set(key, value, ttl)
            
        except Exception as e:
            self.logger.error(f"Write-through error for key {key}: {e}")
            return False
    
    async def write_behind_set(self, key: str, value: Any, persistor: AsyncCallable, ttl: Optional[int] = None, **kwargs) -> bool:
        """Write-behind pattern implementation"""
        # Write to cache immediately
        success = await self.set(key, value, ttl)
        
        # Queue for background persistence
        if success:
            await self._write_behind_queue.put({
                'key': key,
                'value': value,
                'persistor': persistor,
                'kwargs': kwargs,
                'timestamp': time.time()
            })
        
        return success
    
    async def refresh_ahead_get(self, key: str, fetcher: AsyncCallable, ttl: Optional[int] = None, refresh_threshold: float = 0.8, **kwargs) -> Any:
        """Refresh-ahead pattern implementation"""
        value = await self.get(key)
        
        if value is not None:
            # Check if we should refresh proactively
            remaining_ttl = await self._get_ttl(key)
            total_ttl = ttl or self.config.default_ttl
            
            if remaining_ttl > 0 and remaining_ttl / total_ttl < refresh_threshold:
                # Trigger background refresh
                task_key = f"refresh:{key}"
                if task_key not in self._refresh_ahead_tasks:
                    self._refresh_ahead_tasks[task_key] = asyncio.create_task(
                        self._background_refresh(key, fetcher, ttl, **kwargs)
                    )
            
            return value
        else:
            # Cache miss, fetch synchronously
            return await self.get_or_set(key, fetcher, ttl, **kwargs)
    
    # Background workers
    async def _write_behind_worker(self):
        """Background worker for write-behind pattern"""
        while True:
            try:
                item = await self._write_behind_queue.get()
                
                try:
                    persistor = item['persistor']
                    if asyncio.iscoroutinefunction(persistor):
                        await persistor(item['key'], item['value'], **item['kwargs'])
                    else:
                        persistor(item['key'], item['value'], **item['kwargs'])
                    
                    self.logger.debug(f"Write-behind completed for key: {item['key']}")
                    
                except Exception as e:
                    self.logger.error(f"Write-behind failed for key {item['key']}: {e}")
                    
                finally:
                    self._write_behind_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Write-behind worker error: {e}")
    
    async def _background_refresh(self, key: str, fetcher: AsyncCallable, ttl: Optional[int], **kwargs):
        """Background refresh for refresh-ahead pattern"""
        try:
            if asyncio.iscoroutinefunction(fetcher):
                new_value = await fetcher(**kwargs)
            else:
                new_value = fetcher(**kwargs)
            
            await self.set(key, new_value, ttl)
            self.logger.debug(f"Background refresh completed for key: {key}")
            
        except Exception as e:
            self.logger.error(f"Background refresh failed for key {key}: {e}")
        finally:
            # Remove from active tasks
            task_key = f"refresh:{key}"
            self._refresh_ahead_tasks.pop(task_key, None)
    
    # Utility methods
    async def _get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key"""
        if not self.redis_client:
            return -1
        
        try:
            full_key = self._build_key(key)
            return await self.redis_client.ttl(full_key)
        except Exception:
            return -1
    
    @asynccontextmanager
    async def _distributed_lock(self, lock_key: str, timeout: int = 10):
        """Distributed lock implementation"""
        identifier = f"{time.time()}:{id(self)}"
        full_lock_key = self._build_key(f"locks:{lock_key}")
        
        try:
            # Acquire lock
            acquired = await self.redis_client.set(
                full_lock_key, identifier, nx=True, ex=timeout
            )
            
            if not acquired:
                raise Exception(f"Failed to acquire lock: {lock_key}")
            
            yield
            
        finally:
            # Release lock using Lua script for atomicity
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                await self.redis_client.eval(script, 1, full_lock_key, identifier)
            except Exception as e:
                self.logger.warning(f"Failed to release lock {lock_key}: {e}")
    
    # Tag-based invalidation
    async def set_with_tags(self, key: str, value: Any, tags: List[str], ttl: Optional[int] = None) -> bool:
        """Set value with tags for group invalidation"""
        success = await self.set(key, value, ttl)
        
        if success and tags:
            pipe = self.redis_client.pipeline()
            for tag in tags:
                tag_key = self._build_key(f"tags:{tag}")
                pipe.sadd(tag_key, key)
                if ttl and ttl > 0:
                    pipe.expire(tag_key, ttl + 300)  # Tag lives longer than data
            
            await pipe.execute()
        
        return success
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cached items with the given tag"""
        if not self.redis_client:
            return 0
        
        try:
            tag_key = self._build_key(f"tags:{tag}")
            keys = await self.redis_client.smembers(tag_key)
            
            if keys:
                full_keys = [self._build_key(key.decode()) for key in keys]
                deleted = await self.redis_client.delete(*full_keys)
                await self.redis_client.delete(tag_key)
                
                self.metrics.deletes += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Tag invalidation error for tag {tag}: {e}")
            return 0
    
    # Monitoring and maintenance
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        return {
            **asdict(self.metrics),
            'circuit_breaker_state': self.circuit_breaker.state if self.circuit_breaker else None,
            'write_behind_queue_size': self._write_behind_queue.qsize() if self._write_behind_queue else 0,
            'active_refresh_tasks': len(self._refresh_ahead_tasks)
        }
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info"""
        if not self.redis_client:
            return {}
        
        try:
            info = await self.redis_client.info()
            return {
                'redis_info': info,
                'cache_metrics': self.get_metrics(),
                'config': asdict(self.config)
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    async def flush_all(self) -> bool:
        """Clear all cached data"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            self.metrics.reset()
            return True
        except Exception as e:
            self.logger.error(f"Failed to flush cache: {e}")
            return False


# Factory function
def create_cache_manager(config: CacheConfig) -> A2ACacheManager:
    """Create and return a cache manager instance"""
    return A2ACacheManager(config)


# Example usage and testing
async def main():
    """Example usage of the cache manager"""
    config = CacheConfig(
        host="localhost",
        port=6379,
        strategy=CacheStrategy.CACHE_ASIDE,
        default_ttl=3600,
        metrics_enabled=True
    )
    
    cache = create_cache_manager(config)
    await cache.connect()
    
    try:
        # Example data fetcher
        async def fetch_user_data(user_id: int):
            # Simulate database call
            await asyncio.sleep(0.1)
            return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
        
        # Cache-aside pattern
        user_data = await cache.cache_aside_get("user:123", fetch_user_data, ttl=1800, user_id=123)
        print(f"User data: {user_data}")
        
        # Set with tags
        await cache.set_with_tags("product:456", {"name": "Product 456", "price": 99.99}, ["products", "electronics"], ttl=7200)
        
        # Get metrics
        metrics = cache.get_metrics()
        print(f"Cache metrics: {metrics}")
        
        # Invalidate by tag
        invalidated = await cache.invalidate_by_tag("products")
        print(f"Invalidated {invalidated} items")
        
    finally:
        await cache.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
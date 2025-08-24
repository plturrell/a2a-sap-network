"""
Async Grok Client with Connection Pooling and Caching
Performance-optimized client for Grok-4 API calls
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class GrokConfig:
    """Configuration for async Grok client"""
    api_key: str
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4-latest"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    pool_connections: int = 10
    pool_maxsize: int = 20
    cache_ttl: int = 300  # 5 minutes


@dataclass
class GrokResponse:
    """Response from Grok API"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]
    cached: bool = False
    response_time: float = 0.0


class AsyncGrokConnectionPool:
    """Connection pool manager for Grok API calls"""
    
    def __init__(self, config: GrokConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.config.pool_connections,
                        max_keepalive_connections=self.config.pool_maxsize
                    )
                    
                    timeout = httpx.Timeout(
                        connect=10.0,
                        read=self.config.timeout,
                        write=10.0,
                        pool=5.0
                    )
                    
                    self._client = # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        # httpx\.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        verify=True,
                        follow_redirects=False,
                        headers={
                            "Authorization": f"Bearer {self.config.api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": "A2A-Reasoning-Agent/1.0"
                        }
                    )
                    
        return self._client
    
    async def close(self):
        """Close the connection pool"""
        if self._client:
            await self._client.aclose()
            self._client = None


class AsyncGrokCache:
    """Async caching layer for Grok API responses"""
    
    def __init__(self, cache_ttl: int = 300, use_redis: bool = False, redis_url: str = None):
        self.cache_ttl = cache_ttl
        self.use_redis = use_redis
        self.redis_url = redis_url
        self._local_cache: Dict[str, Dict] = {}
        self._local_timestamps: Dict[str, float] = {}
        self._redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize the cache system"""
        if self.use_redis and self.redis_url:
            try:
                self._redis_client = redis.from_url(self.redis_url)
                await self._redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed, using local cache: {e}")
                self.use_redis = False
    
    def _generate_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic key from messages and parameters
        key_data = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "response_format": kwargs.get("response_format")
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, cache_key: str) -> Optional[GrokResponse]:
        """Get cached response"""
        try:
            if self.use_redis and self._redis_client:
                # Try Redis first
                cached_data = await self._redis_client.get(f"grok:{cache_key}")
                if cached_data:
                    data = json.loads(cached_data)
                    response = GrokResponse(**data)
                    response.cached = True
                    return response
            else:
                # Use local cache
                if cache_key in self._local_cache:
                    timestamp = self._local_timestamps.get(cache_key, 0)
                    if time.time() - timestamp < self.cache_ttl:
                        data = self._local_cache[cache_key]
                        response = GrokResponse(**data)
                        response.cached = True
                        return response
                    else:
                        # Expired, remove from cache
                        del self._local_cache[cache_key]
                        del self._local_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, cache_key: str, response: GrokResponse):
        """Set cached response"""
        try:
            # Prepare data for caching (exclude non-serializable fields)
            cache_data = {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "raw_response": response.raw_response,
                "response_time": response.response_time
            }
            
            if self.use_redis and self._redis_client:
                # Cache in Redis
                await self._redis_client.setex(
                    f"grok:{cache_key}",
                    self.cache_ttl,
                    json.dumps(cache_data)
                )
            else:
                # Cache locally
                self._local_cache[cache_key] = cache_data
                self._local_timestamps[cache_key] = time.time()
                
                # Simple cleanup - remove oldest if too many
                if len(self._local_cache) > 1000:
                    oldest_key = min(self._local_timestamps.keys(),
                                   key=lambda k: self._local_timestamps[k])
                    del self._local_cache[oldest_key]
                    del self._local_timestamps[oldest_key]
                    
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def close(self):
        """Close cache connections"""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None


class AsyncGrokClient:
    """High-performance async Grok client with connection pooling and caching"""
    
    def __init__(self, config: GrokConfig, use_cache: bool = True, redis_url: str = None):
        self.config = config
        self.connection_pool = AsyncGrokConnectionPool(config)
        self.cache = AsyncGrokCache(config.cache_ttl, redis_url is not None, redis_url) if use_cache else None
        self.request_count = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        self._initialized = False
    
    async def initialize(self):
        """Initialize the async client"""
        if not self._initialized:
            if self.cache:
                await self.cache.initialize()
            self._initialized = True
            logger.info("Async Grok client initialized with connection pooling")
    
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> GrokResponse:
        """Async chat completion with caching and connection pooling"""
        await self.initialize()
        
        start_time = time.time()
        
        # Check cache first
        cache_key = None
        if self.cache:
            cache_key = self.cache._generate_cache_key(
                messages, temperature=temperature, max_tokens=max_tokens,
                response_format=response_format
            )
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                self.cache_hits += 1
                self.request_count += 1
                return cached_response
        
        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if response_format:
            payload["response_format"] = response_format
        
        # Make API call with retries
        response = await self._make_request_with_retries(payload, start_time)
        
        # Cache the response
        if self.cache and cache_key and response:
            await self.cache.set(cache_key, response)
        
        self.request_count += 1
        self.total_response_time += response.response_time if response else 0
        
        return response
    
    async def _make_request_with_retries(self, payload: Dict, start_time: float) -> GrokResponse:
        """Make API request with exponential backoff retries"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                client = await self.connection_pool.get_client()
                
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    return GrokResponse(
                        content=data['choices'][0]['message']['content'],
                        model=data['model'],
                        usage={
                            "prompt_tokens": data.get('usage', {}).get('prompt_tokens', 0),
                            "completion_tokens": data.get('usage', {}).get('completion_tokens', 0),
                            "total_tokens": data.get('usage', {}).get('total_tokens', 0)
                        },
                        finish_reason=data['choices'][0].get('finish_reason', 'stop'),
                        raw_response=data,
                        response_time=response_time
                    )
                elif response.status_code == 429:  # Rate limit
                    if attempt < self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {self.config.max_retries} retries")
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                    break
        
        # If all retries failed, return error response
        response_time = time.time() - start_time
        return GrokResponse(
            content=f"Error: {last_exception}",
            model=self.config.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="error",
            raw_response={"error": str(last_exception)},
            response_time=response_time
        )
    
    async def stream_completion_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async streaming completion with connection pooling"""
        await self.initialize()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            client = await self.connection_pool.get_client()
            
            async with client.stream(
                "POST",
                f"{self.config.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"Streaming failed: HTTP {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        cache_hit_rate = self.cache_hits / max(self.request_count, 1)
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time": avg_response_time,
            "cache_enabled": self.cache is not None,
            "connection_pool_size": self.config.pool_connections
        }
    
    async def close(self):
        """Close all connections and cleanup"""
        await self.connection_pool.close()
        if self.cache:
            await self.cache.close()
        logger.info("Async Grok client closed")


# Enhanced Grok Reasoning with connection pooling
class AsyncGrokReasoning:
    """Async Grok reasoning with performance optimizations"""
    
    def __init__(self, config: Optional[GrokConfig] = None, redis_url: Optional[str] = None):
        if config is None:
            import os
            config = GrokConfig(
                api_key=os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'),
                pool_connections=10,
                pool_maxsize=20,
                cache_ttl=300
            )
        
        self.grok_client = AsyncGrokClient(config, use_cache=True, redis_url=redis_url)
    
    async def decompose_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Decompose question using async Grok-4 with caching"""
        try:
            prompt = f"""
Analyze and decompose this question:

Question: {question}
{f"Context: {context}" if context else ""}

Provide:
1. Main concepts
2. Sub-questions (ordered by importance)
3. Reasoning approach
4. Expected answer structure

Format as JSON.
"""
            
            response = await self.grok_client.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            if response and response.content and not response.content.startswith("Error:"):
                result = json.loads(response.content)
                return {
                    "success": True,
                    "decomposition": result,
                    "model": response.model,
                    "cached": response.cached,
                    "response_time": response.response_time
                }
            
        except Exception as e:
            logger.error(f"Async Grok-4 decomposition error: {e}")
        
        return {"success": False, "reason": "Decomposition failed"}
    
    async def analyze_patterns(self, text: str, existing_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze patterns using async Grok-4 with caching"""
        try:
            prompt = f"""
Analyze patterns in this text:

Text: {text}
{f"Existing patterns: {json.dumps(existing_patterns, indent=2)}" if existing_patterns else ""}

Identify:
1. Semantic patterns
2. Logical relationships
3. Key insights
4. Reasoning frameworks

Return as JSON.
"""
            
            response = await self.grok_client.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            
            if response and response.content and not response.content.startswith("Error:"):
                patterns = json.loads(response.content)
                return {
                    "success": True,
                    "patterns": patterns,
                    "model": response.model,
                    "cached": response.cached,
                    "response_time": response.response_time
                }
                
        except Exception as e:
            logger.error(f"Async Grok-4 pattern analysis error: {e}")
        
        return {"success": False, "patterns": existing_patterns or []}
    
    async def synthesize_answer(self, sub_answers: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """Synthesize answer using async Grok-4 with caching"""
        try:
            prompt = f"""
Synthesize a comprehensive answer:

Original Question: {original_question}

Sub-Answers:
{json.dumps(sub_answers, indent=2)}

Create an answer that:
1. Integrates all information
2. Maintains logical flow
3. Directly addresses the question
4. Includes confidence assessment
"""
            
            response = await self.grok_client.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            if response and response.content and not response.content.startswith("Error:"):
                return {
                    "success": True,
                    "synthesis": response.content,
                    "model": response.model,
                    "cached": response.cached,
                    "response_time": response.response_time
                }
                
        except Exception as e:
            logger.error(f"Async Grok-4 synthesis error: {e}")
        
        return {"success": False, "reason": "Synthesis failed"}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return await self.grok_client.get_performance_stats()
    
    async def close(self):
        """Close the async reasoning client"""
        await self.grok_client.close()


# Example usage and testing
async def test_async_grok_client():
    """Test the async Grok client with connection pooling"""
    try:
        import os
        
        config = GrokConfig(
            api_key=os.getenv('XAI_API_KEY', 'test-key'),
            pool_connections=5,
            cache_ttl=60
        )
        
        grok = AsyncGrokReasoning(config)
        
        # Test multiple concurrent requests
        tasks = []
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
            "Explain deep learning",
            "What is artificial intelligence?"  # Duplicate for cache test
        ]
        
        for question in questions:
            task = grok.decompose_question(question)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # Print results
        for i, result in enumerate(results):
            cached = result.get('cached', False)
            response_time = result.get('response_time', 0)
            print(f"Question {i+1}: Success={result.get('success')}, "
                  f"Cached={cached}, Time={response_time:.2f}s")
        
        # Get performance stats
        stats = await grok.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2f}")
        print(f"  Avg response time: {stats['avg_response_time']:.2f}s")
        
        await grok.close()
        print("✅ Async Grok client test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_async_grok_client())
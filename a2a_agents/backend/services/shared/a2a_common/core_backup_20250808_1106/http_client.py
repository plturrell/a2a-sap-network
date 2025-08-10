"""
Enhanced HTTP Client with Connection Pooling and Retry Logic
"""

import asyncio
import httpx
from typing import Dict, Any, Optional, Union, Callable
import logging
from datetime import datetime
import json
from urllib.parse import urljoin

from .circuit_breaker import get_breaker_manager, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 backoff_factor: float = 2.0,
                 max_backoff: float = 60.0,
                 retry_on_status: list = None):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]


class ConnectionPool:
    """HTTP connection pool manager"""
    
    def __init__(self, 
                 max_connections: int = 100,
                 max_keepalive_connections: int = 20,
                 keepalive_expiry: float = 30.0):
        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry
        )
        self._clients: Dict[str, httpx.AsyncClient] = {}
    
    async def get_client(self, base_url: str) -> httpx.AsyncClient:
        """Get or create client for base URL"""
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                limits=self.limits,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True
            )
        return self._clients[base_url]
    
    async def close_all(self):
        """Close all clients"""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class EnhancedHTTPClient:
    """Enhanced HTTP client with retry, circuit breaker, and pooling"""
    
    def __init__(self, 
                 base_url: str,
                 headers: Optional[Dict[str, str]] = None,
                 retry_config: Optional[RetryConfig] = None):
        self.base_url = base_url.rstrip('/')
        self.default_headers = headers or {}
        self.retry_config = retry_config or RetryConfig()
        self.pool = ConnectionPool()
        self.breaker = get_breaker_manager().get_breaker(
            f"http_{base_url}",
            failure_threshold=5,
            timeout=60.0
        )
    
    async def request(self,
                     method: str,
                     endpoint: str,
                     headers: Optional[Dict[str, str]] = None,
                     json_data: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, str]] = None,
                     data: Optional[Union[Dict, bytes]] = None,
                     timeout: Optional[float] = None) -> httpx.Response:
        """
        Make HTTP request with retry and circuit breaker
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be joined with base_url)
            headers: Request headers (merged with defaults)
            json_data: JSON body data
            params: Query parameters
            data: Form data or raw bytes
            timeout: Request timeout in seconds
            
        Returns:
            HTTP response
            
        Raises:
            httpx.HTTPStatusError: On HTTP errors
            CircuitBreakerOpenError: If circuit is open
        """
        # Merge headers
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)
        
        # Build full URL
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        
        # Prepare request
        async def make_request():
            client = await self.pool.get_client(self.base_url)
            
            logger.debug(f"{method} {url}")
            
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                json=json_data,
                params=params,
                data=data,
                timeout=timeout or 30.0
            )
            
            # Log response
            logger.debug(f"Response: {response.status_code} from {url}")
            
            # Raise for status if needed
            if response.status_code >= 400:
                logger.error(f"HTTP error {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
            
            return response
        
        # Execute with retry and circuit breaker
        return await self._execute_with_retry(make_request)
    
    async def _execute_with_retry(self, 
                                 func: Callable[[], httpx.Response]) -> httpx.Response:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Use circuit breaker
                return await self.breaker.call(func)
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                
                # Check if we should retry this status code
                if e.response.status_code not in self.retry_config.retry_on_status:
                    raise
                
                # Calculate backoff
                if attempt < self.retry_config.max_attempts - 1:
                    backoff = min(
                        self.retry_config.backoff_factor ** attempt,
                        self.retry_config.max_backoff
                    )
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.retry_config.max_attempts}), "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                    
            except CircuitBreakerOpenError:
                # Don't retry if circuit is open
                raise
                
            except Exception as e:
                last_exception = e
                
                # Retry on connection errors
                if attempt < self.retry_config.max_attempts - 1:
                    backoff = min(
                        self.retry_config.backoff_factor ** attempt,
                        self.retry_config.max_backoff
                    )
                    logger.warning(
                        f"Request error (attempt {attempt + 1}/{self.retry_config.max_attempts}), "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
        
        # All retries exhausted
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request"""
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request"""
        return await self.request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request"""
        return await self.request("PUT", endpoint, **kwargs)
    
    async def patch(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PATCH request"""
        return await self.request("PATCH", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request"""
        return await self.request("DELETE", endpoint, **kwargs)
    
    async def close(self):
        """Close HTTP client"""
        await self.pool.close_all()


class PlatformHTTPClient(EnhancedHTTPClient):
    """Platform-specific HTTP client with auth integration"""
    
    def __init__(self,
                 platform_id: str,
                 base_url: str,
                 auth_manager=None,
                 retry_config: Optional[RetryConfig] = None):
        super().__init__(base_url, retry_config=retry_config)
        self.platform_id = platform_id
        self.auth_manager = auth_manager
    
    async def request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make request with automatic auth headers"""
        # Get auth headers if auth manager provided
        if self.auth_manager:
            auth_headers = await self.auth_manager.get_auth_headers(self.platform_id)
            
            # Merge with provided headers
            headers = kwargs.get("headers", {})
            headers.update(auth_headers)
            kwargs["headers"] = headers
        
        return await super().request(method, endpoint, **kwargs)


# Global connection pool
_global_pool = None

def get_global_connection_pool() -> ConnectionPool:
    """Get global connection pool"""
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectionPool()
    return _global_pool
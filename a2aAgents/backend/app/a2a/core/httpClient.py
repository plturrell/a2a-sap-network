"""
Enhanced HTTP Client with Connection Pooling and Retry Logic
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
# Performance: Consider using asyncio.gather for concurrent operations
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from typing import Dict, Any, Optional, Union, Callable
import logging
from urllib.parse import urljoin

from .circuitBreaker import get_circuit_breaker_manager, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        retry_on_status: list = None,
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retry_on_status = retry_on_status or [429, 500, 502, 503, 504]


class ConnectionPool:
    """HTTP connection pool manager"""

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 30.0,
    ):
        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )
        self._clients: Dict[str, httpx.AsyncClient] = {}

    async def get_client(self, base_url: str) -> httpx.AsyncClient:
        """Get or create client for base URL"""
        if base_url not in self._clients:
            # Security: Validate base_url is HTTPS for production
            if not base_url.startswith("https://") and not base_url.startswith("http://localhost"):
                logger.warning(f"⚠️ Using insecure HTTP for non-localhost: {base_url}")

            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            # Use placeholder client or implement blockchain-based HTTP alternative
            self._clients[base_url] = None  # Placeholder - implement A2A compliant client
        return self._clients[base_url]

    async def close_all(self):
        """Close all clients"""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class EnhancedHTTPClient:
    """Enhanced HTTP client with retry, circuit breaker, and pooling"""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_headers = headers or {}
        self.retry_config = retry_config or RetryConfig()
        self.pool = ConnectionPool()
        # Initialize circuit breaker asynchronously - will be set up in initialization
        self.breaker = None
        asyncio.create_task(self._initialize_circuit_breaker(base_url))

    async def _initialize_circuit_breaker(self, base_url: str):
        """Initialize circuit breaker asynchronously"""
        try:
            from .circuitBreaker import CircuitBreakerConfig
            manager = await get_circuit_breaker_manager()
            config = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60)
            self.breaker = await manager.get_circuit_breaker(f"http_{base_url}", config)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"Failed to initialize circuit breaker: {e}")

    async def request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
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
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))

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
                timeout=timeout or 30.0,
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

    async def _execute_with_retry(self, func: Callable[[], httpx.Response]) -> httpx.Response:
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
                        self.retry_config.backoff_factor**attempt, self.retry_config.max_backoff
                    )
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.retry_config.max_attempts}), "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)

            except CircuitBreakerOpenError as e:
                # Don't retry if circuit is open
                raise CircuitBreakerOpenError("Circuit breaker is open") from e

            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e

                # Retry on connection errors
                if attempt < self.retry_config.max_attempts - 1:
                    backoff = min(
                        self.retry_config.backoff_factor**attempt, self.retry_config.max_backoff
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

    def __init__(
        self,
        platform_id: str,
        base_url: str,
        auth_manager=None,
        retry_config: Optional[RetryConfig] = None,
    ):
        super().__init__(base_url, retry_config=retry_config)
        self.platform_id = platform_id
        self.auth_manager = auth_manager

    async def request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        data: Optional[Union[Dict, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make request with automatic auth headers"""
        # Get auth headers if auth manager provided
        if self.auth_manager:
            auth_headers = await self.auth_manager.get_auth_headers(self.platform_id)

            # Merge with provided headers
            if headers is None:
                headers = {}
            headers.update(auth_headers)

        return await super().request(method, endpoint, headers, json_data, params, data, timeout)


# Global connection pool
_global_pool = None


def get_global_connection_pool() -> ConnectionPool:
    """Get global connection pool"""
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectionPool()
    return _global_pool

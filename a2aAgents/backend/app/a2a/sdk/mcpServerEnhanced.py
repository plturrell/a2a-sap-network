"""
Enhanced MCP Server with batch requests, streaming, and improved error handling
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, AsyncIterator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from .mcpServer import A2AMCPServer
from .mcpTypes import MCPRequest, MCPResponse, MCPError, MCPErrorCodes
from ...core.retry_utils import retry_with_backoff
from ...core.dead_letter_queue import dlq

logger = logging.getLogger(__name__)


@dataclass
class StreamingResponse:
    """Represents a streaming response"""
    id: str
    chunks: AsyncIterator[Dict[str, Any]]
    metadata: Dict[str, Any]


class EnhancedMCPServer(A2AMCPServer):
    """
    Enhanced MCP server with additional features:
    - Batch request support
    - Streaming responses
    - Rate limiting
    - Authentication
    - Better error handling
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_version: str,
        max_batch_size: int = 100,
        rate_limit_per_minute: int = 1000,
        enable_auth: bool = True
    ):
        super().__init__(agent_id, agent_name, agent_version)
        self.max_batch_size = max_batch_size
        self.rate_limit_per_minute = rate_limit_per_minute
        self.enable_auth = enable_auth
        self.request_counts: Dict[str, List[datetime]] = {}
        self.auth_tokens: Dict[str, Dict[str, Any]] = {}
        self.streaming_handlers: Dict[str, Callable] = {}

    async def handle_request(self, request: Union[MCPRequest, List[MCPRequest]], auth_token: Optional[str] = None) -> Union[MCPResponse, List[MCPResponse]]:
        """Handle single or batch MCP requests"""

        # Authentication check
        if self.enable_auth and not self._validate_auth(auth_token):
            error = MCPError(
                code=MCPErrorCodes.INVALID_REQUEST,
                message="Authentication required"
            )
            if isinstance(request, list):
                return [MCPResponse(id=req.id, error=error) for req in request]
            else:
                return MCPResponse(id=request.id, error=error)

        # Handle batch requests
        if isinstance(request, list):
            return await self._handle_batch_request(request, auth_token)
        else:
            return await self._handle_single_request(request, auth_token)

    async def _handle_batch_request(self, requests: List[MCPRequest], auth_token: str) -> List[MCPResponse]:
        """Handle batch of requests"""

        if len(requests) > self.max_batch_size:
            return [
                MCPResponse(
                    id=req.id,
                    error=MCPError(
                        code=MCPErrorCodes.INVALID_REQUEST,
                        message=f"Batch size exceeds maximum of {self.max_batch_size}"
                    )
                ) for req in requests
            ]

        # Process requests concurrently
        tasks = []
        for req in requests:
            tasks.append(self._handle_single_request(req, auth_token))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                final_responses.append(
                    MCPResponse(
                        id=requests[i].id,
                        error=MCPError(
                            code=MCPErrorCodes.INTERNAL_ERROR,
                            message=str(response)
                        )
                    )
                )
            else:
                final_responses.append(response)

        return final_responses

    async def _handle_single_request(self, request: MCPRequest, auth_token: str) -> MCPResponse:
        """Handle single request with rate limiting"""

        # Rate limiting check
        client_id = auth_token or "anonymous"
        if not self._check_rate_limit(client_id):
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCodes.INVALID_REQUEST,
                    message="Rate limit exceeded"
                )
            )

        # Check for streaming request
        if request.method.endswith("/stream"):
            return await self._handle_streaming_request(request)

        # Delegate to parent implementation with retry
        @retry_with_backoff(max_attempts=3, exceptions=(Exception,))
        async def execute_request():
            return await super().handle_request(request)

        try:
            return await execute_request()
        except Exception as e:
            # Log to dead letter queue
            await dlq.add_message(
                message_id=request.id,
                content=request.model_dump() if hasattr(request, 'model_dump') else {
                    "id": request.id,
                    "method": request.method,
                    "params": request.params
                },
                error=str(e),
                source=f"mcp_server_{self.agent_id}",
                metadata={"auth_token": auth_token}
            )

            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCodes.INTERNAL_ERROR,
                    message="Request processing failed",
                    data=str(e)
                )
            )

    async def _handle_streaming_request(self, request: MCPRequest) -> MCPResponse:
        """Handle streaming request"""

        method = request.method.replace("/stream", "")

        if method not in self.streaming_handlers:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCodes.METHOD_NOT_FOUND,
                    message=f"Streaming method not found: {method}"
                )
            )

        try:
            # Create streaming response
            stream_id = f"stream_{request.id}"
            handler = self.streaming_handlers[method]
            chunks = handler(request.params)

            # Store stream for client retrieval
            streaming_response = StreamingResponse(
                id=stream_id,
                chunks=chunks,
                metadata={
                    "method": method,
                    "started_at": datetime.utcnow().isoformat()
                }
            )

            # Return initial response with stream ID
            return MCPResponse(
                id=request.id,
                result={
                    "stream_id": stream_id,
                    "type": "streaming",
                    "metadata": streaming_response.metadata
                }
            )

        except Exception as e:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=MCPErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to create stream: {str(e)}"
                )
            )

    def register_streaming_handler(self, method: str, handler: Callable[..., AsyncIterator[Dict[str, Any]]]):
        """Register a streaming handler"""
        self.streaming_handlers[method] = handler
        logger.info(f"Registered streaming handler: {method}")

    def _validate_auth(self, token: Optional[str]) -> bool:
        """Validate authentication token"""
        if not self.enable_auth:
            return True

        if not token:
            return False

        # Check if token exists and is valid
        if token in self.auth_tokens:
            token_data = self.auth_tokens[token]
            if token_data.get("expires_at", datetime.max) > datetime.utcnow():
                return True

        return False

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.utcnow()

        # Initialize or clean old requests
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []

        # Remove requests older than 1 minute
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if (now - req_time).total_seconds() < 60
        ]

        # Check rate limit
        if len(self.request_counts[client_id]) >= self.rate_limit_per_minute:
            return False

        # Record new request
        self.request_counts[client_id].append(now)
        return True

    def add_auth_token(self, token: str, metadata: Dict[str, Any], expires_in_seconds: int = 3600):
        """Add an authentication token"""
        self.auth_tokens[token] = {
            **metadata,
            "expires_at": datetime.utcnow() + timedelta(seconds=expires_in_seconds)
        }

    async def cleanup_expired_tokens(self):
        """Remove expired authentication tokens"""
        now = datetime.utcnow()
        expired = [
            token for token, data in self.auth_tokens.items()
            if data.get("expires_at", datetime.max) <= now
        ]

        for token in expired:
            del self.auth_tokens[token]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens")

async def main():
    """Main MCP server runner"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create enhanced MCP server
        server = EnhancedMCPServer()
        logger.info("Enhanced MCP Server initialized")

        # Start HTTP server on port 8100
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI(title="Enhanced MCP Server")

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "Enhanced MCP Server"}

        logger.info("Starting Enhanced MCP Server on port 8100")
        uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start Enhanced MCP Server: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


"""
Real MCP Server Implementation for A2A Agents
Implements the Model Context Protocol JSON-RPC 2.0 specification
"""

import asyncio
import json
import logging
import os
import traceback
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import inspect
from dataclasses import dataclass, asdict
from enum import Enum

# MCP Protocol imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MCPErrorCode(Enum):
    """MCP Error Codes according to JSON-RPC 2.0 specification"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # MCP-specific errors
    TOOL_NOT_FOUND = -32000
    RESOURCE_NOT_FOUND = -32001
    PROMPT_NOT_FOUND = -32002


@dataclass
class MCPError:
    """MCP Error Response"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class MCPRequest:
    """MCP JSON-RPC Request"""
    jsonrpc: str
    method: str
    id: Union[str, int, None]
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC Response"""
    jsonrpc: str
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[MCPError] = None


@dataclass
class MCPToolInfo:
    """MCP Tool Information"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class MCPResourceInfo:
    """MCP Resource Information"""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"


@dataclass
class MCPPromptInfo:
    """MCP Prompt Information"""
    name: str
    description: str
    arguments: List[Dict[str, Any]]


class A2AMCPServer:
    """
    Real MCP Server Implementation for A2A Agents
    Serves the Model Context Protocol over WebSocket and HTTP
    """

    def __init__(
        self,
        agent_instance,
        host: Optional[str] = None,
        port: Optional[int] = None,
        websocket_port: Optional[int] = None,
        enable_http: bool = True,
        enable_websocket: bool = True
    ):
        self.agent = agent_instance
        # Load from environment with defaults
        self.host = host or os.getenv('MCP_SERVER_HOST', 'localhost')
        self.port = port or int(os.getenv('MCP_BASE_PORT', '8080'))
        self.websocket_port = websocket_port or int(os.getenv('MCP_WEBSOCKET_PORT', str(self.port + 1000)))
        self.enable_http = enable_http and FASTAPI_AVAILABLE
        self.enable_websocket = enable_websocket and WEBSOCKETS_AVAILABLE

        # MCP registry
        self.tools: Dict[str, MCPToolInfo] = {}
        self.resources: Dict[str, MCPResourceInfo] = {}
        self.prompts: Dict[str, MCPPromptInfo] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.resource_handlers: Dict[str, Callable] = {}
        self.prompt_handlers: Dict[str, Callable] = {}

        # Server state
        self.is_running = False
        self.websocket_server = None
        self.http_app = None
        self.connected_clients: Dict[str, Any] = {}

        # Performance metrics
        self.metrics = {
            "requests_handled": 0,
            "tools_called": 0,
            "resources_accessed": 0,
            "prompts_executed": 0,
            "errors": 0,
            "start_time": None,
            "last_activity": None
        }

        # Discover and register MCP components from agent
        self._discover_mcp_components()

        # Initialize servers
        if self.enable_http:
            self._setup_http_server()

        logger.info(f"A2A MCP Server initialized for {getattr(agent_instance, 'agent_id', 'unknown')} "
                   f"- Tools: {len(self.tools)}, Resources: {len(self.resources)}, Prompts: {len(self.prompts)}")

    def _discover_mcp_components(self):
        """Discover MCP tools, resources, and prompts from agent instance"""
        logger.info("Discovering MCP components from agent...")

        for attr_name in dir(self.agent):
            try:
                attr = getattr(self.agent, attr_name)
                if not callable(attr):
                    continue

                # Check for MCP tool
                if hasattr(attr, '_mcp_tool'):
                    tool_info = attr._mcp_tool
                    self.tools[tool_info['name']] = MCPToolInfo(
                        name=tool_info['name'],
                        description=tool_info['description'],
                        input_schema=tool_info['input_schema'],
                        output_schema=tool_info.get('output_schema')
                    )
                    self.tool_handlers[tool_info['name']] = attr
                    logger.debug(f"Registered MCP tool: {tool_info['name']}")

                # Check for MCP resource
                if hasattr(attr, '_mcp_resource'):
                    resource_info = attr._mcp_resource
                    self.resources[resource_info['uri']] = MCPResourceInfo(
                        uri=resource_info['uri'],
                        name=resource_info['name'],
                        description=resource_info['description'],
                        mime_type=resource_info['mime_type']
                    )
                    self.resource_handlers[resource_info['uri']] = attr
                    logger.debug(f"Registered MCP resource: {resource_info['uri']}")

                # Check for MCP prompt
                if hasattr(attr, '_mcp_prompt'):
                    prompt_info = attr._mcp_prompt
                    self.prompts[prompt_info['name']] = MCPPromptInfo(
                        name=prompt_info['name'],
                        description=prompt_info['description'],
                        arguments=prompt_info['arguments']
                    )
                    self.prompt_handlers[prompt_info['name']] = attr
                    logger.debug(f"Registered MCP prompt: {prompt_info['name']}")

            except Exception as e:
                logger.warning(f"Error discovering MCP component {attr_name}: {e}")

        logger.info(f"MCP Discovery complete - Tools: {len(self.tools)}, "
                   f"Resources: {len(self.resources)}, Prompts: {len(self.prompts)}")

    def _setup_http_server(self):
        """Setup FastAPI HTTP server for MCP"""
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available - HTTP MCP server disabled")
            return

        self.http_app = FastAPI(
            title=f"A2A MCP Server - {getattr(self.agent, 'agent_id', 'unknown')}",
            description="Model Context Protocol server for A2A agent",
            version="1.0.0"
        )

        @self.http_app.post("/mcp")
        async def handle_mcp_request(request: Dict[str, Any]):
            """Handle MCP JSON-RPC requests over HTTP"""
            return await self._handle_mcp_request(request)

        @self.http_app.get("/mcp/tools")
        async def list_tools():
            """List available MCP tools"""
            return {
                "tools": [asdict(tool) for tool in self.tools.values()]
            }

        @self.http_app.get("/mcp/resources")
        async def list_resources():
            """List available MCP resources"""
            return {
                "resources": [asdict(resource) for resource in self.resources.values()]
            }

        @self.http_app.get("/mcp/prompts")
        async def list_prompts():
            """List available MCP prompts"""
            return {
                "prompts": [asdict(prompt) for prompt in self.prompts.values()]
            }

        @self.http_app.get("/mcp/status")
        async def get_status():
            """Get MCP server status and metrics"""
            uptime = None
            if self.metrics["start_time"]:
                uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()

            return {
                "status": "running" if self.is_running else "stopped",
                "agent_id": getattr(self.agent, 'agent_id', 'unknown'),
                "tools_count": len(self.tools),
                "resources_count": len(self.resources),
                "prompts_count": len(self.prompts),
                "connected_clients": len(self.connected_clients),
                "metrics": {
                    **self.metrics,
                    "uptime_seconds": uptime,
                    "start_time": self.metrics["start_time"].isoformat() if self.metrics["start_time"] else None,
                    "last_activity": self.metrics["last_activity"].isoformat() if self.metrics["last_activity"] else None
                }
            }

        @self.http_app.websocket("/mcp/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for MCP"""
            await self._handle_websocket_connection(websocket)

    async def _handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP JSON-RPC request"""
        self.metrics["requests_handled"] += 1
        self.metrics["last_activity"] = datetime.utcnow()

        try:
            # Parse request
            request = MCPRequest(
                jsonrpc=request_data.get("jsonrpc", "2.0"),
                method=request_data["method"],
                id=request_data.get("id"),
                params=request_data.get("params", {})
            )

            # Validate JSON-RPC
            if request.jsonrpc != "2.0":
                return self._create_error_response(
                    request.id, MCPErrorCode.INVALID_REQUEST, "Invalid JSON-RPC version"
                )

            # Route request
            result = await self._route_mcp_request(request)

            return {
                "jsonrpc": "2.0",
                "id": request.id,
                "result": result
            }

        except KeyError as e:
            self.metrics["errors"] += 1
            return self._create_error_response(
                request_data.get("id"), MCPErrorCode.INVALID_REQUEST, f"Missing required field: {e}"
            )
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error handling MCP request: {e}\n{traceback.format_exc()}")
            return self._create_error_response(
                request_data.get("id"), MCPErrorCode.INTERNAL_ERROR, str(e)
            )

    async def _route_mcp_request(self, request: MCPRequest) -> Any:
        """Route MCP request to appropriate handler"""
        method = request.method
        params = request.params or {}

        # MCP Core Methods
        if method == "initialize":
            return await self._handle_initialize(params)
        elif method == "tools/list":
            return await self._handle_list_tools()
        elif method == "tools/call":
            return await self._handle_call_tool(params)
        elif method == "resources/list":
            return await self._handle_list_resources()
        elif method == "resources/read":
            return await self._handle_read_resource(params)
        elif method == "prompts/list":
            return await self._handle_list_prompts()
        elif method == "prompts/get":
            return await self._handle_get_prompt(params)
        else:
            raise Exception(f"Unknown method: {method}")

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True},
                "prompts": {"listChanged": True}
            },
            "serverInfo": {
                "name": f"A2A-MCP-{getattr(self.agent, 'agent_id', 'unknown')}",
                "version": "1.0.0"
            }
        }

    async def _handle_list_tools(self) -> Dict[str, Any]:
        """Handle tools/list request"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema
                }
                for tool in self.tools.values()
            ]
        }

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tool_handlers:
            raise Exception(f"Tool not found: {tool_name}")

        self.metrics["tools_called"] += 1

        try:
            handler = self.tool_handlers[tool_name]

            # Call the tool handler
            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}\n{traceback.format_exc()}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool: {str(e)}"
                    }
                ],
                "isError": True
            }

    async def _handle_list_resources(self) -> Dict[str, Any]:
        """Handle resources/list request"""
        return {
            "resources": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mime_type
                }
                for resource in self.resources.values()
            ]
        }

    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        uri = params.get("uri")

        if uri not in self.resource_handlers:
            raise Exception(f"Resource not found: {uri}")

        self.metrics["resources_accessed"] += 1

        try:
            handler = self.resource_handlers[uri]

            # Call the resource handler
            if inspect.iscoroutinefunction(handler):
                result = await handler()
            else:
                result = handler()

            # Get resource info for MIME type
            resource_info = self.resources[uri]

            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource_info.mime_type,
                        "text": json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}\n{traceback.format_exc()}")
            raise

    async def _handle_list_prompts(self) -> Dict[str, Any]:
        """Handle prompts/list request"""
        return {
            "prompts": [
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": prompt.arguments
                }
                for prompt in self.prompts.values()
            ]
        }

    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request"""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self.prompt_handlers:
            raise Exception(f"Prompt not found: {name}")

        self.metrics["prompts_executed"] += 1

        try:
            handler = self.prompt_handlers[name]

            # Call the prompt handler
            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)

            return {
                "description": self.prompts[name].description,
                "messages": [
                    {
                        "role": "assistant",
                        "content": {
                            "type": "text",
                            "text": str(result)
                        }
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error executing prompt {name}: {e}\n{traceback.format_exc()}")
            raise

    async def _handle_websocket_connection(self, websocket):
        """Handle WebSocket connection for MCP"""
        client_id = str(uuid.uuid4())
        self.connected_clients[client_id] = {
            "websocket": websocket,
            "connected_at": datetime.utcnow()
        }

        try:
            await websocket.accept()
            logger.info(f"MCP WebSocket client connected: {client_id}")

            async for message in websocket.iter_text():
                try:
                    request_data = json.loads(message)
                    response = await self._handle_mcp_request(request_data)
                    await websocket.send_text(json.dumps(response))
                except json.JSONDecodeError:
                    error_response = self._create_error_response(
                        None, MCPErrorCode.PARSE_ERROR, "Invalid JSON"
                    )
                    await websocket.send_text(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    error_response = self._create_error_response(
                        None, MCPErrorCode.INTERNAL_ERROR, str(e)
                    )
                    await websocket.send_text(json.dumps(error_response))

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            if client_id in self.connected_clients:
                del self.connected_clients[client_id]
            logger.info(f"MCP WebSocket client disconnected: {client_id}")

    def _create_error_response(self, request_id: Any, error_code: MCPErrorCode, message: str) -> Dict[str, Any]:
        """Create MCP error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": error_code.value,
                "message": message
            }
        }

    async def start(self):
        """Start the MCP server"""
        if self.is_running:
            logger.warning("MCP server is already running")
            return

        self.is_running = True
        self.metrics["start_time"] = datetime.utcnow()

        tasks = []

        # Start HTTP server
        if self.enable_http and self.http_app:
            logger.info(f"Starting MCP HTTP server on {self.host}:{self.port}")
            tasks.append(self._start_http_server())

        # Start WebSocket server
        if self.enable_websocket and WEBSOCKETS_AVAILABLE:
            logger.info(f"Starting MCP WebSocket server on {self.host}:{self.websocket_port}")
            tasks.append(self._start_websocket_server())

        if not tasks:
            logger.error("No transport available - install fastapi and/or websockets")
            self.is_running = False
            return

        # Run all servers concurrently
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error running MCP servers: {e}")
            self.is_running = False
            raise

    async def _start_http_server(self):
        """Start HTTP server using uvicorn"""
        if not FASTAPI_AVAILABLE:
            return

        config = uvicorn.Config(
            self.http_app,
            host=self.host,
            port=self.port,
            log_level=os.getenv('MCP_LOG_LEVEL', 'info').lower()
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _start_websocket_server(self):
        """Start WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            return

        async def handle_websocket(websocket, path):
            # Convert websockets WebSocket to our format
            class WebSocketAdapter:
                def __init__(self, ws):
                    self.ws = ws

                async def accept(self):
                    pass  # Already accepted by websockets

                async def send_text(self, text):
                    await self.ws.send(text)

                async def iter_text(self):
                    async for message in self.ws:
                        yield message

            await self._handle_websocket_connection(WebSocketAdapter(websocket))

        self.websocket_server = await websockets.serve(
            handle_websocket,
            self.host,
            self.websocket_port
        )

        # Keep server running
        await self.websocket_server.wait_closed()

    async def stop(self):
        """Stop the MCP server"""
        if not self.is_running:
            return

        logger.info("Stopping MCP server...")
        self.is_running = False

        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

        # Disconnect all clients
        for client_id, client_info in list(self.connected_clients.items()):
            try:
                if hasattr(client_info["websocket"], "close"):
                    await client_info["websocket"].close()
            except Exception as e:
                logger.warning(f"Error closing client connection {client_id}: {e}")

        self.connected_clients.clear()
        logger.info("MCP server stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        uptime = None
        if self.metrics["start_time"]:
            uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()

        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "connected_clients": len(self.connected_clients),
            "is_running": self.is_running
        }


# Helper function to create MCP server for agent
def create_mcp_server(agent_instance, **kwargs) -> A2AMCPServer:
    """Create MCP server for agent instance"""
    return A2AMCPServer(agent_instance, **kwargs)


# Agent mixin for MCP integration
class MCPServerMixin:
    """Mixin to add MCP server capabilities to A2A agents"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_server: Optional[A2AMCPServer] = None
        self._mcp_config = kwargs.get('mcp_config', {})

    def initialize_mcp_server(self, **server_kwargs):
        """Initialize MCP server for this agent"""
        config = {**self._mcp_config, **server_kwargs}
        self.mcp_server = create_mcp_server(self, **config)
        logger.info(f"MCP server initialized for agent {getattr(self, 'agent_id', 'unknown')}")
        return self.mcp_server

    async def start_mcp_server(self):
        """Start the MCP server"""
        if not self.mcp_server:
            self.initialize_mcp_server()

        if self.mcp_server:
            await self.mcp_server.start()

    async def stop_mcp_server(self):
        """Stop the MCP server"""
        if self.mcp_server:
            await self.mcp_server.stop()

    def get_mcp_metrics(self) -> Optional[Dict[str, Any]]:
        """Get MCP server metrics"""
        if self.mcp_server:
            return self.mcp_server.get_metrics()
        return None
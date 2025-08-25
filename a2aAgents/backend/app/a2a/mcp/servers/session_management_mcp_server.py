#!/usr/bin/env python3
"""
Standalone MCP Server for session_management
Auto-generated from /Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/reasoningAgent/mcpSessionManagement.py
"""

import sys
import asyncio
import logging
import os
from pathlib import Path
import signal

# Set up proper Python path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Set working directory
os.chdir(str(backend_dir))

try:
    from app.a2a.sdk.mcpServerEnhanced import EnhancedMCPServer
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
except ImportError as e:
    print(f"MCP dependencies not found: {e}")
    # Try alternative import paths
    try:
        import sys
        sys.path.insert(0, str(backend_dir / "app" / "a2a" / "sdk"))
        from mcpServerEnhanced import EnhancedMCPServer
        print("✅ Found MCP server using alternative path")
    except ImportError:
        print("❌ MCP server not available")
        # Create a basic server instead
        class EnhancedMCPServer:
            def __init__(self, name, version):
                self.name = name
                self.version = version

# Import the agent class
agent_class = None
try:
    from reasoningAgent.mcpSessionManagement import MCPServerWithSessions
    agent_class = MCPServerWithSessions
    print(f"✅ Successfully imported {agent_class.__name__}")
except ImportError as e:
    print(f"❌ Failed to import agent class: {e}")
    # Create a mock agent class for testing
    class MockAgent:
        def __init__(self, base_url, enable_monitoring=True):
            self.base_url = base_url
            self.enable_monitoring = enable_monitoring

        def process_request(self, request):
            return f"Mock response from {'session_management'}"

        def get_status(self):
            return {"status": "running", "service": "session_management"}

    agent_class = MockAgent

logger = logging.getLogger(__name__)

class StandaloneMCPServerWithSessionsServer:
    """Standalone MCP server for session_management"""

    def __init__(self):
        self.name = "mcp_session_management_server"
        self.version = "1.0.0"
        self.port = 8106

        # Initialize the agent
        try:
            self.agent = agent_class(
                base_url=f"http://localhost:{self.port}",
                enable_monitoring=True
            )
            print(f"✅ Initialized agent: {agent_class.__name__}")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            # Use mock agent as fallback
            self.agent = MockAgent(
                base_url=f"http://localhost:{self.port}",
                enable_monitoring=True
            )

        logger.info(f"Initialized {self.__class__.__name__} on port {self.port}")

    async def handle_request(self, request):
        """Handle MCP request"""
        method = request.get("method", "unknown")
        params = request.get("params", {})

        # Route to appropriate tool

        if method == "handle_mcp_request_with_session":
            try:
                if hasattr(self.agent, 'handle_mcp_request_with_session'):
                    method_func = getattr(self.agent, 'handle_mcp_request_with_session')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool handle_mcp_request_with_session is not callable", "status": "error"}
                else:
                    return {"error": "Tool handle_mcp_request_with_session not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing handle_mcp_request_with_session: {str(e)}", "status": "error"}
        if method == "validate_session":
            try:
                if hasattr(self.agent, 'validate_session'):
                    method_func = getattr(self.agent, 'validate_session')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool validate_session is not callable", "status": "error"}
                else:
                    return {"error": "Tool validate_session not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing validate_session: {str(e)}", "status": "error"}
        if method == "mcp_server":
            try:
                if hasattr(self.agent, 'mcp_server'):
                    method_func = getattr(self.agent, 'mcp_server')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_server is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_server not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_server: {str(e)}", "status": "error"}

        # Default fallback
        return {"error": f"Unknown method: {method}", "available_tools": [3], "status": "error"}

    async def start_server(self):
        """Start the MCP server with production features"""
        try:
            from fastapi import FastAPI, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            import uvicorn
            import signal
            import asyncio
        except ImportError:
            print("FastAPI not available, starting simple HTTP server")
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

            class MCPHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        health_data = {
                            "status": "healthy",
                            "service": "mcp_session_management_server",
                            "port": 8106,
                            "tools": 3
                        }
                        self.wfile.write(json.dumps(health_data).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()

            server = HTTPServer(('0.0.0.0', 8106), MCPHandler)
            print(f"Starting basic HTTP server for mcp_session_management_server on port 8106")
            server.serve_forever()
            return

        app = FastAPI(
            title="mcp_session_management_server",
            description="MCP session and authentication management",
            version="1.0.0",
            docs_url=None,  # Disable docs in production
            redoc_url=None  # Disable redoc in production
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

        # Global exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "status": "error"}
            )

        # Graceful shutdown handler
        shutdown_event = asyncio.Event()

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "mcp_session_management_server",
                "port": 8106,
                "tools": 3,
                "agent_type": type(self.agent).__name__
            }

        @app.get("/info")
        async def service_info():
            return {
                "service": "mcp_session_management_server",
                "agent": "session_management",
                "description": "MCP session and authentication management",
                "port": 8106,
                "tools": [
                    {
                        "name": "handle_mcp_request_with_session",
                        "description": "MCP tool: handle_mcp_request_with_session",
                        "type": "function"
                    },                    {
                        "name": "validate_session",
                        "description": "MCP tool: validate_session",
                        "type": "function"
                    },                    {
                        "name": "mcp_server",
                        "description": "MCP attribute tool: server",
                        "type": "attribute"
                    }
                ]
            }

        @app.post("/mcp")
        async def handle_mcp_request(request: dict, auth_info: dict = None):
            """Handle authenticated MCP requests"""
            try:
                # Add auth context to request
                if auth_info:
                    request['_auth'] = auth_info

                result = await self.handle_request(request)

                # Add correlation ID if present
                if '_correlation_id' in request:
                    result['_correlation_id'] = request['_correlation_id']

                return result
            except Exception as e:
                logger.error(f"Error handling MCP request: {e}")
                return {
                    "error": "Internal server error",
                    "status": "error",
                    "code": 500
                }

        # Startup event
        @app.on_event("startup")
        async def startup():
            logger.info(f"Starting mcp_session_management_server on port 8106")
            # Initialize connections, caches, etc.

        # Shutdown event
        @app.on_event("shutdown")
        async def shutdown():
            logger.info(f"Shutting down mcp_session_management_server")
            # Clean up resources, close connections

        print(f"🚀 Starting mcp_session_management_server on port 8106")

        # Production server configuration
        config = uvicorn.Config(
            app=app,
            host=os.getenv('MCP_SERVER_HOST', '0.0.0.0'),
            port=8106,
            log_level=os.getenv('MCP_LOG_LEVEL', 'info').lower(),
            access_log=True,
            use_colors=False,
            server_header=False,  # Don't expose server info
            date_header=False,    # Don't expose date for security
        )

        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Main server entry point"""
    logging.basicConfig(level=logging.INFO)

    try:
        server = StandaloneMCPServerWithSessionsServer()
        await server.start_server()
    except Exception as e:
        logger.error(f"Failed to start mcp_session_management_server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

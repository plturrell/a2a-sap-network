#!/usr/bin/env python3
"""
Standalone MCP Server for vector_ranking
Auto-generated from /Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent3VectorProcessing/active/mcpHybridRankingSkills.py
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
    from agent3VectorProcessing.active.mcpHybridRankingSkills import MCPAgent
    agent_class = MCPAgent
    print(f"✅ Successfully imported {agent_class.__name__}")
except ImportError as e:
    print(f"❌ Failed to import agent class: {e}")
    # Create a mock agent class for testing
    class MockAgent:
        def __init__(self, base_url, enable_monitoring=True):
            self.base_url = base_url
            self.enable_monitoring = enable_monitoring
        
        def process_request(self, request):
            return f"Mock response from {'vector_ranking'}"
        
        def get_status(self):
            return {"status": "running", "service": "vector_ranking"}
    
    agent_class = MockAgent

logger = logging.getLogger(__name__)

class StandaloneMCPAgentServer:
    """Standalone MCP server for vector_ranking"""
    
    def __init__(self):
        self.name = "mcp_vector_ranking_server"
        self.version = "1.0.0"
        self.port = 8103
        
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
        
        if method == "hybrid_ranking_search_mcp":
            try:
                if hasattr(self.agent, 'hybrid_ranking_search_mcp'):
                    method_func = getattr(self.agent, 'hybrid_ranking_search_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool hybrid_ranking_search_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool hybrid_ranking_search_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing hybrid_ranking_search_mcp: {str(e)}", "status": "error"}        
        if method == "calculate_bm25_scores_mcp":
            try:
                if hasattr(self.agent, 'calculate_bm25_scores_mcp'):
                    method_func = getattr(self.agent, 'calculate_bm25_scores_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool calculate_bm25_scores_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool calculate_bm25_scores_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing calculate_bm25_scores_mcp: {str(e)}", "status": "error"}        
        if method == "calculate_pagerank_scores_mcp":
            try:
                if hasattr(self.agent, 'calculate_pagerank_scores_mcp'):
                    method_func = getattr(self.agent, 'calculate_pagerank_scores_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool calculate_pagerank_scores_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool calculate_pagerank_scores_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing calculate_pagerank_scores_mcp: {str(e)}", "status": "error"}        
        if method == "calculate_vector_similarity_mcp":
            try:
                if hasattr(self.agent, 'calculate_vector_similarity_mcp'):
                    method_func = getattr(self.agent, 'calculate_vector_similarity_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool calculate_vector_similarity_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool calculate_vector_similarity_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing calculate_vector_similarity_mcp: {str(e)}", "status": "error"}        
        if method == "mcp_hybrid_ranking":
            try:
                if hasattr(self.agent, 'mcp_hybrid_ranking'):
                    method_func = getattr(self.agent, 'mcp_hybrid_ranking')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_hybrid_ranking is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_hybrid_ranking not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_hybrid_ranking: {str(e)}", "status": "error"}
        
        # Default fallback
        return {"error": f"Unknown method: {method}", "available_tools": [5], "status": "error"}
    
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
            
            class MCPHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        health_data = {
                            "status": "healthy",
                            "service": "mcp_vector_ranking_server",
                            "port": 8103,
                            "tools": 5
                        }
                        self.wfile.write(json.dumps(health_data).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            server = HTTPServer(('0.0.0.0', 8103), MCPHandler)
            print(f"Starting basic HTTP server for mcp_vector_ranking_server on port 8103")
            server.serve_forever()
            return
        
        app = FastAPI(
            title="mcp_vector_ranking_server",
            description="Hybrid ranking and scoring algorithms",
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
                "service": "mcp_vector_ranking_server",
                "port": 8103,
                "tools": 5,
                "agent_type": type(self.agent).__name__
            }
        
        @app.get("/info")
        async def service_info():
            return {
                "service": "mcp_vector_ranking_server",
                "agent": "vector_ranking",
                "description": "Hybrid ranking and scoring algorithms",
                "port": 8103,
                "tools": [
                    {
                        "name": "hybrid_ranking_search_mcp",
                        "description": "MCP tool: hybrid_ranking_search_mcp",
                        "type": "function"
                    },                    {
                        "name": "calculate_bm25_scores_mcp",
                        "description": "MCP tool: calculate_bm25_scores_mcp",
                        "type": "function"
                    },                    {
                        "name": "calculate_pagerank_scores_mcp",
                        "description": "MCP tool: calculate_pagerank_scores_mcp",
                        "type": "function"
                    },                    {
                        "name": "calculate_vector_similarity_mcp",
                        "description": "MCP tool: calculate_vector_similarity_mcp",
                        "type": "function"
                    },                    {
                        "name": "mcp_hybrid_ranking",
                        "description": "MCP attribute tool: hybrid_ranking",
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
            logger.info(f"Starting mcp_vector_ranking_server on port 8103")
            # Initialize connections, caches, etc.
            
        # Shutdown event
        @app.on_event("shutdown")
        async def shutdown():
            logger.info(f"Shutting down mcp_vector_ranking_server")
            # Clean up resources, close connections
            
        print(f"🚀 Starting mcp_vector_ranking_server on port 8103")
        
        # Production server configuration
        config = uvicorn.Config(
            app=app,
            host=os.getenv('MCP_SERVER_HOST', '0.0.0.0'),
            port=8103,
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
        server = StandaloneMCPAgentServer()
        await server.start_server()
    except Exception as e:
        logger.error(f"Failed to start mcp_vector_ranking_server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

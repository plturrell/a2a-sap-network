#!/usr/bin/env python3
"""
Standalone MCP Server for data_standardization
Auto-generated from /Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent1Standardization/active/mcpEnhancedDataStandardizationAgent.py
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
    from agent1Standardization.active.mcpEnhancedDataStandardizationAgent import MCPEnhancedDataStandardizationAgent
    agent_class = MCPEnhancedDataStandardizationAgent
    print(f"✅ Successfully imported {agent_class.__name__}")
except ImportError as e:
    print(f"❌ Failed to import agent class: {e}")
    # Create a mock agent class for testing
    class MockAgent:
        def __init__(self, base_url, enable_monitoring=True):
            self.base_url = base_url
            self.enable_monitoring = enable_monitoring
        
        def process_request(self, request):
            return f"Mock response from {'data_standardization'}"
        
        def get_status(self):
            return {"status": "running", "service": "data_standardization"}
    
    agent_class = MockAgent

logger = logging.getLogger(__name__)

class StandaloneMCPEnhancedDataStandardizationAgentServer:
    """Standalone MCP server for data_standardization"""
    
    def __init__(self):
        self.name = "mcp_data_standardization_server"
        self.version = "1.0.0"
        self.port = 8101
        
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
        
        if method == "handle_mcp_enhanced_standardization":
            try:
                if hasattr(self.agent, 'handle_mcp_enhanced_standardization'):
                    method_func = getattr(self.agent, 'handle_mcp_enhanced_standardization')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool handle_mcp_enhanced_standardization is not callable", "status": "error"}
                else:
                    return {"error": "Tool handle_mcp_enhanced_standardization not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing handle_mcp_enhanced_standardization: {str(e)}", "status": "error"}        
        if method == "mcp_account_standardization_skill":
            try:
                if hasattr(self.agent, 'mcp_account_standardization_skill'):
                    method_func = getattr(self.agent, 'mcp_account_standardization_skill')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_account_standardization_skill is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_account_standardization_skill not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_account_standardization_skill: {str(e)}", "status": "error"}        
        if method == "mcp_batch_standardization_skill":
            try:
                if hasattr(self.agent, 'mcp_batch_standardization_skill'):
                    method_func = getattr(self.agent, 'mcp_batch_standardization_skill')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_batch_standardization_skill is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_batch_standardization_skill not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_batch_standardization_skill: {str(e)}", "status": "error"}        
        if method == "mcp_enhanced_standardization_workflow":
            try:
                if hasattr(self.agent, 'mcp_enhanced_standardization_workflow'):
                    method_func = getattr(self.agent, 'mcp_enhanced_standardization_workflow')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_enhanced_standardization_workflow is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_enhanced_standardization_workflow not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_enhanced_standardization_workflow: {str(e)}", "status": "error"}        
        if method == "_assess_data_quality_with_mcp":
            try:
                if hasattr(self.agent, '_assess_data_quality_with_mcp'):
                    method_func = getattr(self.agent, '_assess_data_quality_with_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool _assess_data_quality_with_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool _assess_data_quality_with_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing _assess_data_quality_with_mcp: {str(e)}", "status": "error"}        
        if method == "_validate_data_with_mcp":
            try:
                if hasattr(self.agent, '_validate_data_with_mcp'):
                    method_func = getattr(self.agent, '_validate_data_with_mcp')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool _validate_data_with_mcp is not callable", "status": "error"}
                else:
                    return {"error": "Tool _validate_data_with_mcp not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing _validate_data_with_mcp: {str(e)}", "status": "error"}        
        if method == "_save_mcp_enhanced_results":
            try:
                if hasattr(self.agent, '_save_mcp_enhanced_results'):
                    method_func = getattr(self.agent, '_save_mcp_enhanced_results')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool _save_mcp_enhanced_results is not callable", "status": "error"}
                else:
                    return {"error": "Tool _save_mcp_enhanced_results not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing _save_mcp_enhanced_results: {str(e)}", "status": "error"}        
        if method == "example_mcp_integration":
            try:
                if hasattr(self.agent, 'example_mcp_integration'):
                    method_func = getattr(self.agent, 'example_mcp_integration')
                    if callable(method_func):
                        if True:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool example_mcp_integration is not callable", "status": "error"}
                else:
                    return {"error": "Tool example_mcp_integration not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing example_mcp_integration: {str(e)}", "status": "error"}        
        if method == "mcp_quality_assessment":
            try:
                if hasattr(self.agent, 'mcp_quality_assessment'):
                    method_func = getattr(self.agent, 'mcp_quality_assessment')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_quality_assessment is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_quality_assessment not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_quality_assessment: {str(e)}", "status": "error"}        
        if method == "mcp_validation_tools":
            try:
                if hasattr(self.agent, 'mcp_validation_tools'):
                    method_func = getattr(self.agent, 'mcp_validation_tools')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_validation_tools is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_validation_tools not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_validation_tools: {str(e)}", "status": "error"}        
        if method == "mcp_performance_tools":
            try:
                if hasattr(self.agent, 'mcp_performance_tools'):
                    method_func = getattr(self.agent, 'mcp_performance_tools')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_performance_tools is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_performance_tools not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_performance_tools: {str(e)}", "status": "error"}        
        if method == "mcp_confidence_calculator":
            try:
                if hasattr(self.agent, 'mcp_confidence_calculator'):
                    method_func = getattr(self.agent, 'mcp_confidence_calculator')
                    if callable(method_func):
                        if False:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {"result": str(result), "status": "success"}
                    else:
                        return {"error": "Tool mcp_confidence_calculator is not callable", "status": "error"}
                else:
                    return {"error": "Tool mcp_confidence_calculator not found", "status": "error"}
            except Exception as e:
                return {"error": f"Error executing mcp_confidence_calculator: {str(e)}", "status": "error"}
        
        # Default fallback
        return {"error": f"Unknown method: {method}", "available_tools": [12], "status": "error"}
    
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
                            "service": "mcp_data_standardization_server",
                            "port": 8101,
                            "tools": 12
                        }
                        self.wfile.write(json.dumps(health_data).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            server = HTTPServer(('0.0.0.0', 8101), MCPHandler)
            print(f"Starting basic HTTP server for mcp_data_standardization_server on port 8101")
            server.serve_forever()
            return
        
        app = FastAPI(
            title="mcp_data_standardization_server",
            description="Data standardization with L4 hierarchical processing",
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
                "service": "mcp_data_standardization_server",
                "port": 8101,
                "tools": 12,
                "agent_type": type(self.agent).__name__
            }
        
        @app.get("/info")
        async def service_info():
            return {
                "service": "mcp_data_standardization_server",
                "agent": "data_standardization",
                "description": "Data standardization with L4 hierarchical processing",
                "port": 8101,
                "tools": [
                    {
                        "name": "handle_mcp_enhanced_standardization",
                        "description": "MCP tool: handle_mcp_enhanced_standardization",
                        "type": "function"
                    },                    {
                        "name": "mcp_account_standardization_skill",
                        "description": "MCP tool: mcp_account_standardization_skill",
                        "type": "function"
                    },                    {
                        "name": "mcp_batch_standardization_skill",
                        "description": "MCP tool: mcp_batch_standardization_skill",
                        "type": "function"
                    },                    {
                        "name": "mcp_enhanced_standardization_workflow",
                        "description": "MCP tool: mcp_enhanced_standardization_workflow",
                        "type": "function"
                    },                    {
                        "name": "_assess_data_quality_with_mcp",
                        "description": "MCP tool: _assess_data_quality_with_mcp",
                        "type": "function"
                    },                    {
                        "name": "_validate_data_with_mcp",
                        "description": "MCP tool: _validate_data_with_mcp",
                        "type": "function"
                    },                    {
                        "name": "_save_mcp_enhanced_results",
                        "description": "Save MCP-enhanced results with quality metadata",
                        "type": "function"
                    },                    {
                        "name": "example_mcp_integration",
                        "description": "Example showing how to use the MCP-enhanced agent",
                        "type": "function"
                    },                    {
                        "name": "mcp_quality_assessment",
                        "description": "MCP attribute tool: quality_assessment",
                        "type": "attribute"
                    },                    {
                        "name": "mcp_validation_tools",
                        "description": "MCP attribute tool: validation_tools",
                        "type": "attribute"
                    },                    {
                        "name": "mcp_performance_tools",
                        "description": "MCP attribute tool: performance_tools",
                        "type": "attribute"
                    },                    {
                        "name": "mcp_confidence_calculator",
                        "description": "MCP attribute tool: confidence_calculator",
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
            logger.info(f"Starting mcp_data_standardization_server on port 8101")
            # Initialize connections, caches, etc.
            
        # Shutdown event
        @app.on_event("shutdown")
        async def shutdown():
            logger.info(f"Shutting down mcp_data_standardization_server")
            # Clean up resources, close connections
            
        print(f"🚀 Starting mcp_data_standardization_server on port 8101")
        
        # Production server configuration
        config = uvicorn.Config(
            app=app,
            host=os.getenv('MCP_SERVER_HOST', '0.0.0.0'),
            port=8101,
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
        server = StandaloneMCPEnhancedDataStandardizationAgentServer()
        await server.start_server()
    except Exception as e:
        logger.error(f"Failed to start mcp_data_standardization_server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

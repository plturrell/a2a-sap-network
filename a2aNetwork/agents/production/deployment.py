"""
Production deployment configuration for MCP-enabled A2A Chat Agent
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from .mcpChatAgent import MCPChatAgent
from .chatStorage import ChatStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('a2a_chat_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionChatAgentDeployment:
    """
    Production deployment manager for the MCP Chat Agent
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "chat_agent_config.json"
        self.chat_agent: Optional[MCPChatAgent] = None
        self.app: Optional[FastAPI] = None
        self.config = self._load_config()
        self._shutdown_event = asyncio.Event()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables"""
        
        # Default configuration
        default_config = {
            "agent": {
                "agent_id": os.getenv("A2A_AGENT_ID", "prod-mcp-chat-agent"),
                "name": os.getenv("A2A_AGENT_NAME", "A2A Production MCP Chat Agent"),
                "description": "Production MCP-enabled chat agent for A2A network",
                "version": "2.1.0",
                "base_url": os.getenv("A2A_BASE_URL", "http://localhost:8000")
            },
            "storage": {
                "database_path": os.getenv("A2A_DB_PATH", "production_chat.db"),
                "max_connections": int(os.getenv("A2A_DB_MAX_CONN", "20")),
                "enable_encryption": os.getenv("A2A_DB_ENCRYPT", "false").lower() == "true",
                "encryption_key": os.getenv("A2A_DB_ENCRYPT_KEY")
            },
            "blockchain": {
                "rpc_url": os.getenv("A2A_BLOCKCHAIN_RPC", "http://localhost:8545"),
                "private_key": os.getenv("A2A_BLOCKCHAIN_PRIVATE_KEY"),
                "contract_addresses": {
                    "message_router": os.getenv("A2A_MESSAGE_ROUTER_ADDRESS"),
                    "agent_registry": os.getenv("A2A_AGENT_REGISTRY_ADDRESS")
                },
                "prefer_blockchain": os.getenv("A2A_PREFER_BLOCKCHAIN", "false").lower() == "true"
            },
            "auth": {
                "enabled": os.getenv("A2A_AUTH_ENABLED", "true").lower() == "true",
                "redis_url": os.getenv("A2A_REDIS_URL", "redis://localhost:6379"),
                "jwt_secret": os.getenv("A2A_JWT_SECRET", "your-secret-key"),
                "rate_limit_per_minute": int(os.getenv("A2A_RATE_LIMIT", "30"))
            },
            "performance": {
                "max_concurrent_conversations": int(os.getenv("A2A_MAX_CONVERSATIONS", "100")),
                "message_batch_size": int(os.getenv("A2A_MESSAGE_BATCH_SIZE", "50")),
                "response_timeout": int(os.getenv("A2A_RESPONSE_TIMEOUT", "30")),
                "retry_attempts": int(os.getenv("A2A_RETRY_ATTEMPTS", "3"))
            },
            "mcp": {
                "host": os.getenv("A2A_MCP_HOST", "0.0.0.0"),
                "port": int(os.getenv("A2A_MCP_PORT", "8080")),
                "enable_http": os.getenv("A2A_MCP_HTTP", "true").lower() == "true",
                "enable_websocket": os.getenv("A2A_MCP_WS", "true").lower() == "true",
                "max_concurrent_mcp_requests": int(os.getenv("A2A_MCP_MAX_REQUESTS", "50"))
            },
            "server": {
                "host": os.getenv("A2A_SERVER_HOST", "0.0.0.0"),
                "port": int(os.getenv("A2A_SERVER_PORT", "8000")),
                "workers": int(os.getenv("A2A_SERVER_WORKERS", "1")),
                "log_level": os.getenv("A2A_LOG_LEVEL", "info"),
                "access_log": os.getenv("A2A_ACCESS_LOG", "true").lower() == "true"
            }
        }
        
        # Try to load from config file if it exists
        if os.path.exists(self.config_path):
            try:
                import json
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge file config with defaults
                default_config.update(file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_path}: {e}")
        
        return default_config
    
    async def initialize_agent(self) -> MCPChatAgent:
        """Initialize the chat agent with production configuration"""
        
        self.chat_agent = MCPChatAgent(
            agent_id=self.config["agent"]["agent_id"],
            name=self.config["agent"]["name"],
            description=self.config["agent"]["description"],
            version=self.config["agent"]["version"],
            base_url=self.config["agent"]["base_url"],
            storage_config=self.config["storage"],
            blockchain_config=self.config["blockchain"],
            auth_config=self.config["auth"],
            performance_config=self.config["performance"],
            mcp_config=self.config["mcp"]
        )
        
        await self.chat_agent.initialize()
        logger.info("Chat agent initialized successfully")
        
        return self.chat_agent
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan context manager"""
        # Startup
        logger.info("Starting A2A MCP Chat Agent...")
        await self.initialize_agent()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, starting graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        yield
        
        # Shutdown
        logger.info("Shutting down A2A MCP Chat Agent...")
        await self.shutdown()
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        
        app = FastAPI(
            title="A2A MCP Chat Agent",
            description="Production MCP-enabled chat agent for A2A network communication",
            version="2.1.0",
            lifespan=self.lifespan
        )
        
        # Add CORS middleware for development
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add standard A2A agent endpoints
        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            if self.chat_agent:
                return self.chat_agent.get_agent_card()
            return {"error": "Agent not initialized"}
        
        @app.get("/health")
        async def health_check():
            if not self.chat_agent:
                return {"status": "unhealthy", "reason": "Agent not initialized"}
            
            try:
                # Get system health from MCP tool
                health_result = await self.chat_agent.get_system_health({})
                return health_result
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        @app.post("/chat")
        async def chat_endpoint(request: dict):
            """Simple chat endpoint for easy integration"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            try:
                user_id = request.get("user_id")
                message = request.get("message")
                target_agent = request.get("target_agent", "data-processor")
                
                if not user_id or not message:
                    return {"error": "user_id and message are required", "success": False}
                
                # Use MCP tool
                result = await self.chat_agent.send_message_to_agent_mcp(
                    message=message,
                    target_agent=target_agent,
                    user_id=user_id,
                    context_id=request.get("conversation_id")
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                return {"error": str(e), "success": False}
        
        @app.post("/start-conversation")
        async def start_conversation_endpoint(request: dict):
            """Start a new conversation"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            try:
                result = await self.chat_agent.start_conversation_mcp(
                    user_id=request["user_id"],
                    participants=request["participants"],
                    initial_message=request["initial_message"],
                    conversation_title=request.get("conversation_title"),
                    conversation_type=request.get("conversation_type", "group")
                )
                return result
                
            except Exception as e:
                logger.error(f"Start conversation error: {e}")
                return {"error": str(e), "success": False}
        
        @app.post("/broadcast")
        async def broadcast_endpoint(request: dict):
            """Broadcast message to multiple agents"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            try:
                result = await self.chat_agent.broadcast_to_agents_mcp(
                    message=request["message"],
                    user_id=request["user_id"],
                    capability_filter=request.get("capability_filter"),
                    max_agents=request.get("max_agents", 5),
                    mcp_only=request.get("mcp_only", False)
                )
                return result
                
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                return {"error": str(e), "success": False}
        
        @app.get("/conversations/{user_id}")
        async def get_user_conversations(user_id: str, limit: int = 20):
            """Get user's conversation history"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            try:
                result = await self.chat_agent.get_conversation_history_mcp(
                    user_id=user_id,
                    limit=limit
                )
                return result
                
            except Exception as e:
                logger.error(f"Get conversations error: {e}")
                return {"error": str(e), "success": False}
        
        @app.get("/agents")
        async def get_agent_directory():
            """Get information about available agents"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            return {
                "agents": self.chat_agent.agent_directory,
                "total_agents": len(self.chat_agent.agent_directory),
                "mcp_enabled_agents": len([
                    a for a in self.chat_agent.agent_directory.values() 
                    if a.get("mcp_enabled")
                ])
            }
        
        @app.get("/metrics")
        async def get_metrics():
            """Get system metrics"""
            if not self.chat_agent:
                return {"error": "Agent not initialized", "success": False}
            
            try:
                result = await self.chat_agent.get_system_health({})
                return result
                
            except Exception as e:
                logger.error(f"Get metrics error: {e}")
                return {"error": str(e), "success": False}
        
        # Store app reference
        self.app = app
        
        return app
    
    async def shutdown(self):
        """Graceful shutdown"""
        self._shutdown_event.set()
        
        if self.chat_agent:
            await self.chat_agent.shutdown()
    
    def run(self):
        """Run the production server"""
        app = self.create_app()
        
        server_config = self.config["server"]
        
        uvicorn.run(
            app,
            host=server_config["host"],
            port=server_config["port"],
            log_level=server_config["log_level"],
            access_log=server_config["access_log"],
            workers=1  # Always use 1 worker for async apps with shared state
        )
    
    async def run_async(self):
        """Run the server asynchronously (for testing or embedding)"""
        app = self.create_app()
        server_config = self.config["server"]
        
        config = uvicorn.Config(
            app,
            host=server_config["host"],
            port=server_config["port"],
            log_level=server_config["log_level"],
            access_log=server_config["access_log"]
        )
        
        server = uvicorn.Server(config)
        
        # Run until shutdown event
        await server.serve()


def main():
    """Main entry point for production deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A MCP Chat Agent Production Server")
    parser.add_argument("--config", help="Configuration file path", default="chat_agent_config.json")
    parser.add_argument("--host", help="Server host", default=None)
    parser.add_argument("--port", help="Server port", type=int, default=None)
    parser.add_argument("--log-level", help="Log level", default=None)
    
    args = parser.parse_args()
    
    # Create deployment
    deployment = ProductionChatAgentDeployment(config_path=args.config)
    
    # Override config with command line args
    if args.host:
        deployment.config["server"]["host"] = args.host
    if args.port:
        deployment.config["server"]["port"] = args.port
    if args.log_level:
        deployment.config["server"]["log_level"] = args.log_level
    
    try:
        deployment.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
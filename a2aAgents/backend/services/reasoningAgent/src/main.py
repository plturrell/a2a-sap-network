"""
Main entry point for Reasoning Agent
A2A Microservice for logical reasoning and decision making
"""

import warnings

# Suppress warnings about unrecognized blockchain networks from eth_utils
warnings.filterwarnings("ignore", message="Network 345 with name 'Yooldo Verse Mainnet'")
warnings.filterwarnings("ignore", message="Network 12611 with name 'Astar zkEVM'")

import asyncio
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

# Try to import the agent, fallback to simple service if not available
from agent import ReasoningAgent
AGENT_AVAILABLE = True

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

if True:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle"""
        global agent_instance
        
        # Startup
        logger.info("Starting Reasoning Agent service...")
        
        # Initialize agent - REQUIRE environment variables for A2A compliance
        base_url = os.getenv("A2A_AGENT_URL")
        if not base_url:
            raise ValueError("A2A_AGENT_URL environment variable is required for A2A protocol compliance")
        
        agent_manager_url = os.getenv("A2A_MANAGER_URL") or os.getenv("A2A_BASE_URL")
        if not agent_manager_url:
            raise ValueError("A2A_MANAGER_URL or A2A_BASE_URL environment variable is required")
        
        downstream_agent_url = os.getenv("A2A_DOWNSTREAM_URL")
        if not downstream_agent_url:
            raise ValueError("A2A_DOWNSTREAM_URL environment variable is required")
        
        try:
            agent_instance = ReasoningAgent(
                base_url=base_url,
                agent_manager_url=agent_manager_url,
                downstream_agent_url=downstream_agent_url
            )
            
            await agent_instance.initialize()
            await agent_instance.register_with_network()
            logger.info("Reasoning Agent service started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            agent_instance = None
        
        yield
        
        # Shutdown
        logger.info("Shutting down Reasoning Agent service...")
        if agent_instance:
            try:
                await agent_instance.deregister_from_network()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        logger.info("Reasoning Agent service shut down")

    # Create FastAPI app with A2A microservice base
    app = FastAPI(
        title="Reasoning Agent",
        description="A2A Microservice for logical reasoning and decision making",
        version="3.0.0",
        lifespan=lifespan
    )
else:
    # Simple FastAPI app when agent is not available
    app = FastAPI(
        title="Reasoning Agent",
        description="A2A Microservice for logical reasoning (Simple Mode)",
        version="3.0.0"
    )

# Use standardized health endpoints
try:
    import sys
    sys.path.append('../../app')
    from app.a2a.sdk.health import StandardHealthCheck
    
    # Create standardized health endpoints
    if AGENT_AVAILABLE and agent_instance:
        StandardHealthCheck.create_health_endpoint(app, agent_instance)
    else:
        # Fallback health endpoint for simple mode
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "service": "Reasoning Agent",
                "version": "3.0.0",
                "mode": "simple",
                "note": "Running without A2A agent (missing dependencies)"
            }
except ImportError:
    # Fallback if StandardHealthCheck not available
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        if AGENT_AVAILABLE and agent_instance:
            return {
                "status": "healthy",
                "service": "Reasoning Agent",
                "version": "3.0.0",
                "agent_registered": agent_instance.is_registered,
                "mode": "full"
            }
        else:
            return {
                "status": "healthy",
                "service": "Reasoning Agent",
                "version": "3.0.0",
                "mode": "simple",
                "note": "Running without A2A agent (missing dependencies)"
            }

if True:
    @app.get("/stats")
    async def get_stats():
        """Get reasoning statistics"""
        if not agent_instance:
            return {"error": "Agent not initialized"}
        
        return {
            "service": "Reasoning Agent",
            "stats": agent_instance.reasoning_stats,
            "config": agent_instance.reasoning_config,
            "knowledge_base_size": {
                "rules": len(agent_instance.knowledge_base.get("rules", [])),
                "facts": len(agent_instance.knowledge_base.get("facts", [])),
                "patterns": len(agent_instance.knowledge_base.get("patterns", {}))
            },
            "status": "active" if agent_instance.is_registered else "inactive"
        }

    @app.post("/reason")
    async def perform_reasoning(request: dict):
        """Endpoint for direct reasoning requests"""
        if not agent_instance:
            return {"error": "Agent not initialized"}
        
        try:
            # Create mock A2A message
            from a2aCommon import A2AMessage, MessageRole
            
            message = A2AMessage(
                role=MessageRole.USER,
                content=request
            )
            
            context_id = request.get('context_id', f"direct_{asyncio.get_event_loop().time()}")
            
            # Process reasoning
            result = await agent_instance.handle_reasoning_request(message, context_id)
            return result
            
        except Exception as e:
            logger.error(f"Error processing reasoning request: {e}")
            return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Reasoning Agent Service",
        "version": "3.0.0",
        "capabilities": ["deductive", "pattern_analysis", "decision_support"],
        "mode": "full" if AGENT_AVAILABLE else "simple"
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get configuration from environment
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_AGENT_PORT", "8008"))
    
    logger.info(f"Starting Reasoning Agent on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
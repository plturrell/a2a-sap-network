"""
Main entry point for Agent 6 - Quality Control Manager
A2A Microservice for orchestrating quality control processes
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

import sys
import os
# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

from agent import QualityControlAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global agent_instance
    
    # Startup
    logger.info("Starting Quality Control Manager service...")
    
    # Initialize agent
    base_url = os.getenv("A2A_AGENT_URL")
    agent_manager_url = os.getenv("A2A_MANAGER_URL", os.getenv("A2A_BASE_URL"))
    downstream_agent_url = os.getenv("A2A_DOWNSTREAM_URL")
    
    agent_instance = QualityControlAgent(
        base_url=base_url,
        agent_manager_url=agent_manager_url,
        downstream_agent_url=downstream_agent_url
    )
    
    try:
        await agent_instance.initialize()
        await agent_instance.register_with_network()
        logger.info("Quality Control Manager service started successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Quality Control Manager service...")
        if agent_instance:
            await agent_instance.deregister_from_network()
        logger.info("Quality Control Manager service shut down")


# Create FastAPI app with A2A microservice base
app = FastAPI(
    title="Agent 6 - Quality Control Manager",
    description="A2A Microservice for orchestrating quality control processes and managing quality gates",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Quality Control Manager",
        "version": "3.0.0",
        "agent_registered": agent_instance.is_registered if agent_instance else False
    }


@app.get("/stats")
async def get_stats():
    """Get quality control statistics"""
    if not agent_instance:
        return {"error": "Agent not initialized"}
    
    return {
        "service": "Quality Control Manager",
        "stats": agent_instance.qc_stats,
        "quality_thresholds": agent_instance.quality_thresholds,
        "active_sessions": len(agent_instance.active_sessions),
        "status": "active" if agent_instance.is_registered else "inactive"
    }


@app.get("/sessions")
async def get_active_sessions():
    """Get active quality control sessions"""
    if not agent_instance:
        return {"error": "Agent not initialized"}
    
    return {
        "active_sessions": list(agent_instance.active_sessions.keys()),
        "session_count": len(agent_instance.active_sessions),
        "sessions": agent_instance.active_sessions
    }


@app.post("/orchestrate")
async def orchestrate_quality_control(request: dict):
    """Endpoint for direct quality control orchestration requests"""
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
        
        # Process orchestration
        result = await agent_instance.handle_qc_orchestration(message, context_id)
        return result
        
    except Exception as e:
        logger.error(f"Error processing QC orchestration request: {e}")
        return {"error": str(e)}


@app.post("/gate/{gate_type}")
async def execute_quality_gate(gate_type: str, request: dict):
    """Execute a specific quality gate"""
    if not agent_instance:
        return {"error": "Agent not initialized"}
    
    try:
        data = request.get("data", {})
        criteria = request.get("criteria", {})
        
        result = await agent_instance.execute_quality_gate(gate_type, data, criteria)
        return result
        
    except Exception as e:
        logger.error(f"Error executing quality gate {gate_type}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get configuration from environment
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_AGENT_PORT", "8007"))
    
    logger.info(f"Starting Quality Control Manager on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

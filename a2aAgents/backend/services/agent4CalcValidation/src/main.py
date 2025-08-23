#!/usr/bin/env python3
"""
Agent 4 - Calculation Validation Microservice
A2A compliant agent for validating financial calculations
"""

import warnings

# Suppress warnings about unrecognized blockchain networks from eth_utils
warnings.filterwarnings("ignore", message="Network 345 with name 'Yooldo Verse Mainnet'")
warnings.filterwarnings("ignore", message="Network 12611 with name 'Astar zkEVM'")

import asyncio
import os
import sys
import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

import sys
import os
# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

from agent import CalculationValidationAgent
from a2aCommon import A2AMessage, MessageRole

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global agent_instance
    
    # Startup
    logger.info("Starting Calculation Validation Agent service...")
    
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
        agent_instance = CalculationValidationAgent(
            base_url=base_url,
            agent_manager_url=agent_manager_url,
            downstream_agent_url=downstream_agent_url
        )
        
        await agent_instance.initialize()
        await agent_instance.register_with_network()
        logger.info("Calculation Validation Agent service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        agent_instance = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Calculation Validation Agent service...")
    if agent_instance:
        try:
            await agent_instance.deregister_from_network()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    logger.info("Calculation Validation Agent service shut down")

# Create FastAPI app with A2A microservice base
app = FastAPI(
    title="Agent 4 - Calculation Validation",
    description="A2A Microservice for validating financial calculations and computational accuracy",
    version="3.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    if agent_instance:
        return {
            "status": "healthy",
            "service": "Calculation Validation Agent",
            "version": "3.0.0",
            "agent_registered": agent_instance.is_registered
        }
    else:
        return {
            "status": "unhealthy",
            "service": "Calculation Validation Agent",
            "version": "3.0.0",
            "error": "Agent not initialized"
        }

@app.get("/stats")
async def get_stats():
    """Get validation statistics"""
    if not agent_instance:
        return {"error": "Agent not initialized"}
    
    return {
        "service": "Calculation Validation Agent",
        "stats": agent_instance.validation_stats,
        "status": "active" if agent_instance.is_registered else "inactive"
    }

@app.post("/validate")
async def validate_calculations(request: dict):
    """Endpoint for direct validation requests"""
    if not agent_instance:
        return {"error": "Agent not initialized"}
    
    try:
        # Create mock A2A message
        message = A2AMessage(
            role=MessageRole.USER,
            content=request
        )
        
        context_id = request.get('context_id', f"direct_{asyncio.get_event_loop().time()}")
        
        # Process validation
        result = await agent_instance.handle_validation_request(message, context_id)
        return result
        
    except Exception as e:
        logger.error(f"Error processing validation request: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agent 4 - Calculation Validation Service",
        "version": "3.0.0"
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get configuration from environment
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_AGENT_PORT", "8005"))
    
    logger.info(f"Starting Calculation Validation Agent on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
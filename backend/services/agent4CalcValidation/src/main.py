"""
Main entry point for Agent 4 - Calculation Validation Agent
A2A Microservice for validating financial calculations
"""

import asyncio
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

sys.path.append('../shared')
sys.path.append('../../shared')

# Try to import the agent, fallback to simple service if not available
try:
    from agent import CalculationValidationAgent
    from a2aCommon.microservice.baseService import A2AMicroservice
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Agent import failed: {e}. Running in simple mode.")
    AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

if AGENT_AVAILABLE:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle"""
        global agent_instance
        
        # Startup
        logger.info("Starting Calculation Validation Agent service...")
        
        # Initialize agent
        base_url = os.getenv("A2A_AGENT_URL", "http://localhost:8004")
        agent_manager_url = os.getenv("A2A_MANAGER_URL", "http://localhost:8000")
        downstream_agent_url = os.getenv("A2A_DOWNSTREAM_URL", "http://localhost:8005")
        
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
    app = A2AMicroservice(
        title="Agent 4 - Calculation Validation",
        description="A2A Microservice for validating financial calculations and computational accuracy",
        version="3.0.0",
        lifespan=lifespan
    ).app
else:
    # Simple FastAPI app when agent is not available
    app = FastAPI(
        title="Agent 4 - Calculation Validation",
        description="A2A Microservice for validating financial calculations (Simple Mode)",
        version="3.0.0"
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    if AGENT_AVAILABLE and agent_instance:
        return {
            "status": "healthy",
            "service": "Calculation Validation Agent",
            "version": "3.0.0",
            "agent_registered": agent_instance.is_registered,
            "mode": "full"
        }
    else:
        return {
            "status": "healthy",
            "service": "Calculation Validation Agent",
            "version": "3.0.0",
            "mode": "simple",
            "note": "Running without A2A agent (missing dependencies)"
        }

if AGENT_AVAILABLE:
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
            from a2aCommon import A2AMessage, MessageRole
            
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
        "version": "3.0.0",
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
    port = int(os.getenv("A2A_AGENT_PORT", "8005"))
    
    logger.info(f"Starting Calculation Validation Agent on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
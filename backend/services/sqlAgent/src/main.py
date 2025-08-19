"""
Main entry point for SQL Agent
A2A Microservice for SQL operations and database management
"""

import asyncio
import logging
import os
import sys
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from contextlib import asynccontextmanager

sys.path.append('../shared')
sys.path.append('../../shared')

# Try to import the agent, fallback to simple service if not available
try:
    from agent import SQLAgent
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
        logger.info("Starting SQL Agent service...")
        
        # Initialize agent
        base_url = os.getenv("A2A_AGENT_URL", "http://localhost:8009")
        agent_manager_url = os.getenv("A2A_MANAGER_URL", "http://localhost:8000")
        downstream_agent_url = os.getenv("A2A_DOWNSTREAM_URL", "http://localhost:8010")
        
        try:
            agent_instance = SQLAgent(
                base_url=base_url,
                agent_manager_url=agent_manager_url,
                downstream_agent_url=downstream_agent_url
            )
            
            await agent_instance.initialize()
            await agent_instance.register_with_network()
            logger.info("SQL Agent service started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            agent_instance = None
        
        yield
        
        # Shutdown
        logger.info("Shutting down SQL Agent service...")
        if agent_instance:
            try:
                await agent_instance.deregister_from_network()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        logger.info("SQL Agent service shut down")

    # Create FastAPI app with A2A microservice base
    app = A2AMicroservice(
        title="SQL Agent",
        description="A2A Microservice for SQL operations and database management",
        version="3.0.0",
        lifespan=lifespan
    ).app
else:
    # Simple FastAPI app when agent is not available
    app = FastAPI(
        title="SQL Agent",
        description="A2A Microservice for SQL operations (Simple Mode)",
        version="3.0.0"
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    if AGENT_AVAILABLE and agent_instance:
        return {
            "status": "healthy",
            "service": "SQL Agent",
            "version": "3.0.0",
            "agent_registered": agent_instance.is_registered,
            "mode": "full"
        }
    else:
        return {
            "status": "healthy",
            "service": "SQL Agent",
            "version": "3.0.0",
            "mode": "simple",
            "note": "Running without A2A agent (missing dependencies)"
        }

if AGENT_AVAILABLE:
    @app.get("/stats")
    async def get_stats():
        """Get SQL operation statistics"""
        if not agent_instance:
            return {"error": "Agent not initialized"}
        
        return {
            "service": "SQL Agent",
            "stats": agent_instance.sql_stats,
            "config": agent_instance.sql_config,
            "schema_registry": {
                "tables": len(agent_instance.schema_registry.get("tables", {})),
                "relationships": len(agent_instance.schema_registry.get("relationships", [])),
                "indexes": len(agent_instance.schema_registry.get("indexes", {}))
            },
            "status": "active" if agent_instance.is_registered else "inactive"
        }

    @app.post("/sql")
    async def perform_sql_operations(request: dict):
        """Endpoint for direct SQL operation requests"""
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
            
            # Process SQL operation
            result = await agent_instance.handle_sql_request(message, context_id)
            return result
            
        except Exception as e:
            logger.error(f"Error processing SQL request: {e}")
            return {"error": str(e)}

    @app.get("/schema")
    async def get_schema_registry():
        """Get current schema registry"""
        if not agent_instance:
            return {"error": "Agent not initialized"}
        
        return {
            "schema_registry": agent_instance.schema_registry,
            "last_updated": datetime.utcnow().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SQL Agent Service",
        "version": "3.0.0",
        "capabilities": ["generate", "validate", "optimize", "analyze_schema"],
        "supported_databases": ["postgresql", "mysql", "sqlite", "sqlserver"],
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
    port = int(os.getenv("A2A_AGENT_PORT", "8009"))
    
    logger.info(f"Starting SQL Agent on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
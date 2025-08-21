#!/usr/bin/env python3
"""
Launch script for Glean Agent
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Import the agent
from app.a2a.agents.gleanAgent import GleanAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for Glean Agent"""
    logger.info("Starting Glean Agent...")
    
    # Create agent instance
    agent = GleanAgent()
    
    # Initialize agent
    await agent.initialize()
    
    # Create FastAPI app
    app = agent.create_fastapi_app()
    
    # Run with uvicorn
    import uvicorn
    
    port = int(os.getenv("A2A_AGENT_PORT", os.getenv("GLEAN_AGENT_PORT", "8016")))
    host = os.getenv("A2A_AGENT_HOST", os.getenv("GLEAN_AGENT_HOST", "0.0.0.0"))
    
    logger.info(f"Glean Agent starting on {host}:{port}")
    logger.info(f"Agent ID: {agent.agent_id}")
    logger.info(f"Agent capabilities: code analysis, linting, testing, security scanning")
    
    # Run the server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down Glean Agent...")
        await agent.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Glean Agent shutdown complete")
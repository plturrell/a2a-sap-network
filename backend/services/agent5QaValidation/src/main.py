#!/usr/bin/env python3
"""
Agent 5 - QA Validation Microservice
A2A compliant agent for quality assurance validation
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

from agent import QAValidationAgent
from a2aCommon import A2AMessage, MessageRole

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global agent_instance
    
    # Startup
    logger.info("Starting QA Validation Agent service...")
    
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
        agent_instance = QAValidationAgent(
            base_url=base_url,
            agent_manager_url=agent_manager_url,
            downstream_agent_url=downstream_agent_url
        )
        
        await agent_instance.initialize()
        await agent_instance.register_with_network()
        logger.info("QA Validation Agent service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        agent_instance = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down QA Validation Agent service...")
    if agent_instance:
        try:
            await agent_instance.deregister_from_network()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    logger.info("QA Validation Agent service shut down")

async def main():
    """Main entry point for standalone execution"""
    # Create FastAPI app with A2A microservice base
    app = FastAPI(
        title="Agent 5 - QA Validation",
        description="A2A Microservice for quality assurance and data validation",
        version="3.0.0",
        lifespan=lifespan
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get configuration from environment
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_AGENT_PORT", "8006"))
    
    logger.info(f"Starting QA Validation Agent on {host}:{port}")
    
    # Run the service
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())

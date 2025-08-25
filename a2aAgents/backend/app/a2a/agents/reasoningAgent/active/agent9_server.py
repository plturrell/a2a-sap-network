import os
import sys
import logging
import asyncio
from datetime import datetime

from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Agent 9 (Reasoning Agent) REST API Server
Provides HTTP endpoints for reasoning and decision-making services
"""

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the router
from agent9Router import router as agent9_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent 9 - Reasoning Agent API",
    description="REST API for advanced logical reasoning and decision-making",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the agent router
app.include_router(agent9_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "Reasoning Agent",
        "agent_id": 9,
        "version": "1.0.0",
        "status": "operational",
        "api_docs": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # Set environment variables
    os.environ["A2A_AGENT_ID"] = "reasoning-agent-9"
    os.environ["A2A_BLOCKCHAIN_URL"] = os.getenv("A2A_BLOCKCHAIN_URL", "http://localhost:8545")

    # Run the server
    logger.info("Starting Agent 9 (Reasoning Agent) API Server...")
    logger.info("Server will be available at http://localhost:8086")

    uvicorn.run(
        "agent9_server:app",
        host="0.0.0.0",
        port=8086,  # Agent 9 uses port 8086 as per the adapter
        reload=True,
        log_level="info"
    )

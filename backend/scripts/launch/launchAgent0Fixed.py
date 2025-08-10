#!/usr/bin/env python3
"""
Launch Agent 0 (Data Product Registration Agent) with correct imports
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the correct import path
from app.a2a.agents.agent0_data_product.active.agent0_router import router as agent0_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for Agent 0
app = FastAPI(
    title="Agent 0 - Data Product Registration Agent",
    description="Independent A2A microservice for data product registration",
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
app.include_router(agent0_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Agent 0 - Data Product Registration",
        "version": "1.0.0",
        "capabilities": ["data_product_registration", "dublin_core_metadata"]
    }

# Trust endpoint
@app.get("/trust/public-key")
async def get_public_key():
    return {
        "public_key": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...",
        "agent_id": "agent0_data_product",
        "trust_level": 100
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    logger.info(f"Starting Agent 0 on port {port}")
    
    uvicorn.run(
        "launch_agent0_fixed:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
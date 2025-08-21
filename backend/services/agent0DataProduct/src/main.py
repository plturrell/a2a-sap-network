#!/usr/bin/env python3
"""
Agent 0 - Data Product Registration Microservice
Main entry point for the containerized agent
"""

import warnings

# Suppress warnings about unrecognized blockchain networks from eth_utils
warnings.filterwarnings("ignore", message="Network 345 with name 'Yooldo Verse Mainnet'")
warnings.filterwarnings("ignore", message="Network 12611 with name 'Astar zkEVM'")

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
# Add the services directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent0DataProduct.src.agent import DataProductRegistrationAgentSDK
from agent0DataProduct.src.router import create_router


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Get configuration from environment variables
    port = int(os.getenv("A2A_AGENT_PORT", "8001"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL")
    if not base_url:
        raise ValueError("A2A_AGENT_BASE_URL environment variable is required for A2A protocol compliance")
    ord_registry_url = os.getenv("ORD_REGISTRY_URL", "https://ord-registry:8000/api/v1/ord")
    
    # Create agent instance
    agent = DataProductRegistrationAgentSDK(
        base_url=base_url,
        ord_registry_url=ord_registry_url
    )
    
    # Initialize agent
    await agent.initialize()
    
    # Create FastAPI app
    app = FastAPI(
        title=agent.name,
        description=agent.description,
        version=agent.version
    )
    
    # Add CORS middleware with secure configuration
    # SECURITY: Never use "*" for origins when credentials are allowed
    allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
        os.getenv("A2A_FRONTEND_URL"),    # Default development frontend
        os.getenv("A2A_GATEWAY_URL"),    # Gateway
        os.getenv("A2A_SERVICE_URL"),    # localhost alternative
        "https://gateway:8080",      # Internal gateway service
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["Accept", "Content-Type", "Authorization", "X-API-Key", "X-Requested-With"],
    )
    
    # Add agent router
    router = create_router(agent)
    app.include_router(router)
    
    # Add health check
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "agent": agent.agent_id,
            "version": agent.version
        }
    
    print(f"🚀 Starting {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🛠️  Available Skills: {len(agent.skills)}")
    
    # Start server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
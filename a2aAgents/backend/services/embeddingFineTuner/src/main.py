#!/usr/bin/env python3
"""
Embedding Fine-Tuner - A2A Microservice
Specialized service for fine-tuning embedding models
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import EmbeddingFineTunerAgent
from router import create_a2a_router


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Get configuration from environment variables
    port = int(os.getenv("A2A_AGENT_PORT", "8015"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL", f"http://localhost:{port}")
    
    # A2A network configuration
    agent_manager_url = os.getenv("A2A_AGENT_MANAGER_URL")
    
    # Create agent instance
    agent = EmbeddingFineTunerAgent(
        base_url=base_url,
        agent_manager_url=agent_manager_url
    )
    
    # Initialize agent and register with A2A network
    await agent.initialize()
    
    try:
        await agent.register_with_network()
    except Exception as e:
        print(f"Registration failed: {e}")
    
    # Create FastAPI app
    app = FastAPI(
        title=f"A2A {agent.name}",
        description=f"{agent.description} - A2A Protocol v0.2.9",
        version=agent.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware for A2A communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add A2A router
    a2a_router = create_a2a_router(agent)
    app.include_router(a2a_router)
    
    # Add health check endpoint
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "agent_id": agent.agent_id,
            "agent_type": "embedding_fine_tuner",
            "version": agent.version,
            "a2a_protocol": "0.2.9",
            "models_loaded": len(getattr(agent, 'available_models', {})),
            "training_jobs": len(getattr(agent, 'training_jobs', {}))
        }
    
    # A2A readiness check
    @app.get("/ready")
    async def ready():
        return {
            "ready": agent.is_ready,
            "registered": getattr(agent, 'is_registered', False),
            "capabilities": len(agent.handlers),
            "skills": len(agent.skills)
        }
    
    print(f"🚀 Starting A2A {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🔗 A2A Network: {agent_manager_url}")
    
    # Start server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        # Cleanup on shutdown
        if hasattr(agent, 'deregister_from_network'):
            await agent.deregister_from_network()
        if hasattr(agent, 'shutdown'):
            await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
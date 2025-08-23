#!/usr/bin/env python3
"""
Data Manager - A2A Microservice
Central data persistence service for the A2A network
"""

import warnings

# Suppress warnings about unrecognized blockchain networks from eth_utils
warnings.filterwarnings("ignore", message="Network 345 with name 'Yooldo Verse Mainnet'")
warnings.filterwarnings("ignore", message="Network 12611 with name 'Astar zkEVM'")

import sys
import os
# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import DataManagerAgent
from router import create_a2a_router


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Get configuration from environment variables
    port = int(os.getenv("A2A_AGENT_PORT", "8011"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL", f"http://localhost:{port}")
    
    # A2A network configuration
    agent_manager_url = os.getenv("A2A_AGENT_MANAGER_URL", "https://agent-manager:8007")
    
    # Storage configuration
    storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
    
    # Create agent instance
    agent = DataManagerAgent(
        base_url=base_url,
        agent_manager_url=agent_manager_url,
        storage_backend=storage_backend
    )
    
    # Initialize agent and register with A2A network
    await agent.initialize()
    # Skip registration if Agent Manager is not available
    try:
        await agent.register_with_network()
    except Exception as e:
        print(f"Warning: Could not register with Agent Manager: {e}")
        print("Continuing without registration...")
    
    # Create FastAPI app
    app = FastAPI(
        title=f"A2A {agent.name}",
        description=f"{agent.description} - A2A Protocol v0.2.9",
        version=agent.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware for A2A communication with secure configuration
    # SECURITY: Never use "*" for origins when credentials are allowed
    allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
        os.getenv("A2A_FRONTEND_URL"),
        os.getenv("A2A_GATEWAY_URL"),  # Gateway
        "https://gateway:8080",       # Internal gateway service
        "https://agent-manager:8010", # Agent Manager internal
    ]
    
    # Filter out None values and empty strings
    valid_origins = [origin.strip() for origin in allowed_origins if origin and isinstance(origin, str) and origin.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=valid_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["Accept", "Content-Type", "Authorization", "X-API-Key", "X-Agent-ID", "X-A2A-Protocol"],
    )
    
    # Add A2A router
    a2a_router = create_a2a_router(agent)
    app.include_router(a2a_router)
    
    # Add health check endpoint
    @app.get("/health")
    async def health():
        db_healthy = True
        cache_healthy = True
        
        try:
            # Check database
            if agent.db_connection:
                await agent.db_connection.execute("SELECT 1")
        except:
            db_healthy = False
        
        try:
            # Check cache
            if agent.redis_client:
                await agent.redis_client.ping()
        except:
            cache_healthy = False
        
        return {
            "status": "healthy" if db_healthy else "degraded",
            "agent_id": agent.agent_id,
            "agent_type": "data_manager",
            "version": agent.version,
            "a2a_protocol": "0.2.9",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unavailable",
                "storage_backend": agent.storage_backend.value
            }
        }
    
    # A2A readiness check
    @app.get("/ready")
    async def ready():
        return {
            "ready": agent.is_ready,
            "registered": agent.is_registered,
            "storage_backend": agent.storage_backend.value,
            "capabilities": len(agent.handlers),
            "skills": len(agent.skills)
        }
    
    print(f"🚀 Starting A2A {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🔗 A2A Network: Connected to {agent_manager_url}")
    print(f"💾 Storage: {storage_backend}")
    print(f"📊 Cache: {'Enabled' if agent.redis_client else 'Disabled'}")
    
    # Start server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        # Cleanup on shutdown
        await agent.deregister_from_network()
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
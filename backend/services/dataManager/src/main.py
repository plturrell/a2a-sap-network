#!/usr/bin/env python3
"""
Data Manager - A2A Microservice
Central data persistence service for the A2A network
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import DataManagerAgent
from .router import create_a2a_router

async def main():
    # Get configuration from environment variables
    port = int(os.getenv("AGENT_PORT", "8008"))
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("AGENT_BASE_URL", f"http://localhost:{port}")
    
    # A2A network configuration
    agent_manager_url = os.getenv("AGENT_MANAGER_URL", "http://agent-manager:8007")
    
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
    await agent.register_with_network()
    
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
        "http://localhost:3000",     # Development frontend
        "http://localhost:8080",     # Gateway
        "http://gateway:8080",       # Internal gateway service
        "http://agent-manager:8007", # Agent Manager internal
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
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
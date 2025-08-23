#!/usr/bin/env python3
"""
Agent Manager - A2A Network Orchestrator Microservice
Central management for A2A agent discovery, health monitoring, and workflow orchestration
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
import redis.asyncio as redis

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AgentManager
from router import create_a2a_router

async def main():
    # Get configuration from environment variables
    port = int(os.getenv("A2A_AGENT_PORT", "8010"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL", f"http://localhost:{port}")
    
    # Redis configuration for state management
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Create Redis client
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    
    # Create agent instance
    agent = AgentManager(
        base_url=base_url,
        redis_client=redis_client
    )
    
    # Initialize agent
    await agent.initialize()
    
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
            "agent_type": "orchestrator",
            "version": agent.version,
            "a2a_protocol": "0.2.9",
            "registered_agents": len(agent.registered_agents),
            "active_workflows": len(agent.active_workflows)
        }
    
    # A2A readiness check
    @app.get("/ready")
    async def ready():
        return {
            "ready": agent.is_ready,
            "redis_connected": await agent.check_redis_connection(),
            "monitoring_active": agent.monitoring_active
        }
    
    print(f"🚀 Starting A2A {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🔗 Redis: {redis_url}")
    print(f"📊 Monitoring: Enabled")
    
    # Start server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        # Cleanup on shutdown
        await agent.shutdown()
        await redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
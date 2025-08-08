#!/usr/bin/env python3
"""
Agent 1 - Data Standardization Microservice
A2A compliant agent for standardizing financial data
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import DataStandardizationAgent
from .router import create_a2a_router

async def main():
    # Get configuration from environment variables
    port = int(os.getenv("AGENT_PORT", "8002"))
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("AGENT_BASE_URL", f"http://localhost:{port}")
    
    # A2A network configuration
    agent_manager_url = os.getenv("AGENT_MANAGER_URL", "http://agent-manager:8007")
    downstream_agent_url = os.getenv("DOWNSTREAM_AGENT_URL", "http://agent2-ai-preparation:8003")
    
    # Create agent instance
    agent = DataStandardizationAgent(
        base_url=base_url,
        agent_manager_url=agent_manager_url,
        downstream_agent_url=downstream_agent_url
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
    
    # Add CORS middleware for A2A communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to known A2A agents
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
            "agent_type": "data_standardization",
            "version": agent.version,
            "a2a_protocol": "0.2.9"
        }
    
    # A2A readiness check
    @app.get("/ready")
    async def ready():
        return {
            "ready": agent.is_ready,
            "registered": agent.is_registered,
            "capabilities": len(agent.handlers),
            "skills": len(agent.skills)
        }
    
    print(f"🚀 Starting A2A {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🔗 A2A Network: Connected to {agent_manager_url}")
    print(f"➡️  Downstream: {downstream_agent_url}")
    print(f"🛠️  Standardizers: {', '.join(agent.standardizers.keys())}")
    
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
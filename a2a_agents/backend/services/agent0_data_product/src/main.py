#!/usr/bin/env python3
"""
Agent 0 - Data Product Registration Microservice
Main entry point for the containerized agent
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import DataProductRegistrationAgentSDK
from .router import create_router

async def main():
    # Get configuration from environment variables
    port = int(os.getenv("AGENT_PORT", "8001"))
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("AGENT_BASE_URL", f"http://localhost:{port}")
    ord_registry_url = os.getenv("ORD_REGISTRY_URL", "http://ord-registry:8000/api/v1/ord")
    
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
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
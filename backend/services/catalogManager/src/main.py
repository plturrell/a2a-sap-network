#!/usr/bin/env python3
"""
Catalog Manager - A2A Microservice
ORD Registry and Data Product Catalog for the A2A network
"""

import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import CatalogManagerAgent
from .router import create_a2a_router

async def main():
    # Get configuration from environment variables
    port = int(os.getenv("A2A_AGENT_PORT", "8012"))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    base_url = os.getenv("A2A_AGENT_BASE_URL", f"http://localhost:{port}")
    
    # A2A network configuration
    agent_manager_url = os.getenv("A2A_AGENT_MANAGER_URL", "http://agent-manager:8010")
    data_manager_url = os.getenv("A2A_DATA_MANAGER_URL", "http://data-manager:8011")
    
    # Create agent instance
    agent = CatalogManagerAgent(
        base_url=base_url,
        agent_manager_url=agent_manager_url,
        data_manager_url=data_manager_url
    )
    
    # Initialize agent and register with A2A network
    await agent.initialize()
    await agent.register_with_network()
    
    # Add update product usage method
    async def _update_product_usage(product_id: str):
        """Update product usage count in database"""
        query = """
            UPDATE data_products 
            SET usage_count = usage_count + 1, last_accessed = ?
            WHERE product_id = ?
        """
        await agent.db_connection.execute(query, (
            datetime.utcnow().isoformat(),
            product_id
        ))
        await agent.db_connection.commit()
    
    agent._update_product_usage = _update_product_usage
    
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
            "agent_type": "catalog_manager",
            "version": agent.version,
            "a2a_protocol": "0.2.9",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unavailable",
                "ord_resources": len(agent.ord_cache),
                "data_products": len(agent.product_cache)
            }
        }
    
    # A2A readiness check
    @app.get("/ready")
    async def ready():
        return {
            "ready": agent.is_ready,
            "registered": agent.is_registered,
            "capabilities": len(agent.handlers),
            "skills": len(agent.skills),
            "catalog_loaded": len(agent.ord_cache) > 0 or len(agent.product_cache) > 0
        }
    
    print(f"🚀 Starting A2A {agent.name} v{agent.version}")
    print(f"📡 Listening on {host}:{port}")
    print(f"🎯 Agent ID: {agent.agent_id}")
    print(f"🔗 A2A Network: Connected to {agent_manager_url}")
    print(f"💾 Data Manager: {data_manager_url}")
    print(f"📚 ORD Resources: {len(agent.ord_cache)}")
    print(f"📦 Data Products: {len(agent.product_cache)}")
    
    # Start server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    # Import datetime for the update method
    from datetime import datetime
    
    try:
        await server.serve()
    finally:
        # Cleanup on shutdown
        await agent.deregister_from_network()
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
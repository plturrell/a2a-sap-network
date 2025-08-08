#!/usr/bin/env python3
"""
Launch Catalog Manager Agent - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.catalog_manager.active.catalog_manager_agent_sdk import CatalogManagerAgentSDK

async def main():
    # Create agent
    agent = CatalogManagerAgentSDK(
        base_url="http://localhost:8006",
        ord_registry_url="http://localhost:8000/api/v1/ord"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8006")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8006, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Shutting down {agent.name}...")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
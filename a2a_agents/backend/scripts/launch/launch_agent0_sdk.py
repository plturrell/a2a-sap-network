#!/usr/bin/env python3
"""
Launch Agent 0 (Data Product Registration) - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.data_product_agent_sdk import DataProductRegistrationAgentSDK

async def main():
    # Create agent
    agent = DataProductRegistrationAgentSDK(
        base_url="http://localhost:8001",
        ord_registry_url="http://localhost:8000/api/v1/ord"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8001")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

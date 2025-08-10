#!/usr/bin/env python3
"""
Launch Agent 1 (Data Standardization) - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.data_standardization_agent_sdk import DataStandardizationAgentSDK

async def main():
    # Create agent
    agent = DataStandardizationAgentSDK(
        base_url="http://localhost:8002"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8002")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8002, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

import os
#!/usr/bin/env python3
"""
Launch Agent 2 (AI Preparation) - SDK Version
"""

import asyncio
import uvicorn
from app.a2a.agents.agent2_ai_preparation.active.ai_preparation_agent_sdk import AIPreparationAgentSDK


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
async def main():
    # Create agent
    agent = AIPreparationAgentSDK(
        base_url="os.getenv("AGENT_MANAGER_URL")"
    )
    
    # Initialize
    await agent.initialize()
    
    try:
        # Create FastAPI app
        app = agent.create_fastapi_app()
        
        print(f"🚀 Starting {agent.name} v{agent.version}")
        print(f"📡 Listening on http://localhost:8003")
        print(f"🎯 Agent ID: {agent.agent_id}")
        print(f"🛠️  Available Skills: {len(agent.skills)}")
        print(f"📋 Available Handlers: {len(agent.handlers)}")
        
        # Start server
        config = uvicorn.Config(app, host="0.0.0.0", port=8003, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Shutting down {agent.name}...")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
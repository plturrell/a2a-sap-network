#!/usr/bin/env python3
"""
Launch script for the Advanced Reasoning Agent
Starts the agent with FastAPI server for A2A protocol support
"""

import asyncio
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(app_path))

from app.a2a.agents.reasoningAgent.reasoningAgent import ReasoningAgent
from app.a2a.sdk.server import create_a2a_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the reasoning agent"""
    try:
        # Configuration
        config = {
            "base_url": os.getenv("REASONING_AGENT_URL", "http://localhost:8008"),
            "agent_network_url": os.getenv("AGENT_NETWORK_URL", "http://localhost:8000"),
            "data_manager_url": os.getenv("DATA_MANAGER_URL", "http://localhost:8001"),
            "catalog_manager_url": os.getenv("CATALOG_MANAGER_URL", "http://localhost:8002"),
            "agent_manager_url": os.getenv("AGENT_MANAGER_URL", "http://localhost:8003"),
            "max_sub_agents": int(os.getenv("MAX_SUB_AGENTS", "10")),
            "reasoning_timeout": int(os.getenv("REASONING_TIMEOUT", "300"))
        }
        
        logger.info("🚀 Starting Advanced Reasoning Agent...")
        logger.info(f"Configuration: {config}")
        
        # Create agent instance
        agent = ReasoningAgent(**config)
        
        # Initialize agent
        logger.info("🔄 Initializing agent...")
        init_result = await agent.initialize()
        logger.info(f"✅ Agent initialized: {init_result}")
        
        # Create FastAPI server with A2A endpoints
        app = create_a2a_server(agent)
        
        # Add custom endpoints
        @app.get("/reasoning/metrics")
        async def get_metrics():
            """Get reasoning system metrics"""
            return await agent.get_reasoning_metrics()
        
        @app.get("/reasoning/architectures")
        async def get_architectures():
            """Get supported reasoning architectures"""
            return {
                "architectures": [arch.value for arch in agent.ReasoningArchitecture],
                "default": "hierarchical"
            }
        
        # Log agent information
        logger.info("=" * 60)
        logger.info("🤖 REASONING AGENT READY")
        logger.info(f"   Agent ID: {agent.agent_id}")
        logger.info(f"   Name: {agent.name}")
        logger.info(f"   Version: {agent.version}")
        logger.info(f"   Base URL: {agent.base_url}")
        logger.info(f"   Architectures: {len(agent.get_agent_card()['capabilities']['reasoningArchitectures'])}")
        logger.info("=" * 60)
        
        # Print agent card URL
        agent_card = agent.get_agent_card()
        logger.info(f"📋 Agent Card available at: {agent.base_url}/a2a/agent-card")
        logger.info(f"📚 API Documentation at: {agent.base_url}/docs")
        logger.info(f"💓 Health check at: {agent.base_url}/health")
        
        # Start the server
        host = "0.0.0.0"
        port = int(agent.base_url.split(":")[-1]) if ":" in agent.base_url else 8008
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        
        # Run the server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Failed to start agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        await agent.shutdown()
        logger.info("✅ Agent shutdown completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Reasoning Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
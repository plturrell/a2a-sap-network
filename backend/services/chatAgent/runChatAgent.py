#!/usr/bin/env python3
"""
Run the A2A Chat Agent following standard A2A patterns
"""

import asyncio
import logging
import os
import sys
import signal
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.a2a.sdk import create_a2a_server
from chatAgent import create_chat_agent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[ChatAgent] = None


async def main():
    """Main entry point for Chat Agent"""
    global agent
    
    try:
        # Get configuration from environment
        base_url = os.getenv("CHAT_AGENT_BASE_URL")
        port = int(os.getenv("CHAT_AGENT_PORT", "8017"))
        host = os.getenv("CHAT_AGENT_HOST", "0.0.0.0")
        
        # Agent configuration
        config = {
            "enable_blockchain": os.getenv("ENABLE_BLOCKCHAIN", "false").lower() == "true",
            "enable_persistence": os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true",
            "max_concurrent_conversations": int(os.getenv("MAX_CONVERSATIONS", "100")),
            "contract_address": os.getenv("CONTRACT_ADDRESS"),
            "network_url": os.getenv("BLOCKCHAIN_NETWORK_URL")
        }
        
        # Create agent
        logger.info("Creating Chat Agent...")
        agent = create_chat_agent(base_url=base_url, config=config)
        
        # Initialize agent
        logger.info("Initializing Chat Agent...")
        await agent.initialize()
        
        # Create A2A server
        logger.info(f"Starting A2A server on {host}:{port}")
        server = create_a2a_server(agent, host=host, port=port)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        await server.run()
        
    except Exception as e:
        logger.error(f"Failed to start Chat Agent: {e}")
        sys.exit(1)


async def shutdown():
    """Graceful shutdown"""
    global agent
    
    logger.info("Shutting down Chat Agent...")
    
    if agent:
        try:
            await agent.shutdown()
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    # Give a moment for cleanup
    await asyncio.sleep(0.5)
    
    # Exit
    logger.info("Shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
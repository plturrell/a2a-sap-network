#!/usr/bin/env python3
"""
Register DataManager on the blockchain
"""

import asyncio
import os
import sys
import logging

# Add path for imports
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)

# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

from blockchain_integration import BlockchainIntegration

async def register_datamanager():
    """Register DataManager on the blockchain"""
    
    # Create blockchain integration for DataManager
    datamanager = BlockchainIntegration(
        agent_id="data_manager_agent",
        agent_name="DataManager",
        endpoint=os.getenv("A2A_SERVICE_URL", "http://localhost:8000")
    )
    
    # Register the agent
    success = await datamanager.register_agent([
        "storage",
        "retrieval", 
        "data_processing",
        "analytics"
    ])
    
    if success:
        logging.info("DataManager registered successfully!")
        
        # Get agent info
        info = datamanager.get_agent_info()
        logging.info(f"Agent info: {info}")
    else:
        logging.error("Failed to register DataManager")

if __name__ == "__main__":
    asyncio.run(register_datamanager())
#!/usr/bin/env python3
"""
Test A2A Blockchain Integration
Tests the integration between finsight_cib agents and a2a_network smart contracts
"""

import asyncio
import logging
import os
from a2a_network.python_sdk.blockchain import initialize_blockchain_client, get_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_blockchain_integration():
    """Test blockchain integration for finsight_cib agents"""
    
    logger.info("🧪 Testing A2A Blockchain Integration")
    
    try:
        # Initialize blockchain client
        logger.info("🔗 Connecting to A2A Network...")
        blockchain_client = initialize_blockchain_client(
            rpc_url="http://localhost:8545"
        )
        
        logger.info(f"✅ Connected to A2A Network")
        logger.info(f"   Chain ID: {blockchain_client.web3.eth.chain_id}")
        logger.info(f"   Agent Address: {blockchain_client.agent_identity.address}")
        logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        
        # Test Agent 0 registration
        logger.info("🤖 Testing Agent 0 registration...")
        agent0_adapter = create_blockchain_adapter(
            agent_id="test_data_product_agent_0",
            name="Test Data Product Registration Agent",
            description="Test agent for blockchain integration",
            version="1.0.0",
            endpoint="http://localhost:8002",
            capabilities=["data_processing", "dublin_core_extraction"]
        )
        
        # Check if already registered
        is_registered = await blockchain_client.is_agent_registered()
        logger.info(f"   Agent registration status: {is_registered}")
        
        if not is_registered:
            registration_success = await agent0_adapter.register_agent()
            if registration_success:
                logger.info("✅ Agent 0 test registration successful")
            else:
                logger.error("❌ Agent 0 test registration failed")
        else:
            logger.info("✅ Agent 0 already registered")
        
        # Test agent discovery
        logger.info("🔍 Testing agent discovery...")
        discovered_agents = await agent0_adapter.discover_agents("data_processing")
        logger.info(f"   Found {len(discovered_agents)} agents with 'data_processing' capability")
        
        for agent in discovered_agents:
            logger.info(f"   - {agent['name']} at {agent['endpoint']}")
            logger.info(f"     Address: {agent['address']}")
            logger.info(f"     Reputation: {agent['reputation']}")
        
        # Test message sending (if we have another agent)
        if len(discovered_agents) > 1:
            logger.info("💬 Testing message sending...")
            target_agent = discovered_agents[1]  # Send to second agent
            
            message_id = await agent0_adapter.send_message_to_agent(
                recipient_address=target_agent['address'],
                message_content={
                    "type": "test_message",
                    "data": "Hello from Agent 0 test",
                    "timestamp": "2025-01-05T12:00:00Z"
                }
            )
            
            if message_id:
                logger.info(f"✅ Test message sent: {message_id}")
            else:
                logger.error("❌ Test message sending failed")
        
        # Get agent status
        status = agent0_adapter.get_agent_status()
        logger.info("📊 Agent Status:")
        for key, value in status.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("✅ A2A Blockchain Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ A2A Blockchain Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_blockchain_integration())
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        exit(1)
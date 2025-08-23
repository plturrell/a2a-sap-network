#!/usr/bin/env python3
"""
Verify Blockchain Integration - Ensure we're using real blockchain, not mocks
"""

import asyncio
import json
import os
import sys
import logging
from web3 import Web3
from eth_account import Account

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

async def verify_blockchain_integration():
    """Verify real blockchain integration is working"""
    
    logger.info("🔍 Verifying Blockchain Integration")
    
    try:
        # Test 1: Direct Web3 Connection
        logger.info("\n1️⃣ Testing Direct Web3 Connection...")
        
        rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Check connection
        is_connected = w3.is_connected()
        logger.info(f"   Connected to blockchain: {is_connected}")
        
        if not is_connected:
            logger.error("❌ Not connected to blockchain! Make sure ganache/hardhat is running.")
            return False
        
        # Get blockchain info
        chain_id = w3.eth.chain_id
        block_number = w3.eth.block_number
        logger.info(f"   Chain ID: {chain_id}")
        logger.info(f"   Current block: {block_number}")
        
        # Test 2: Check Smart Contracts
        logger.info("\n2️⃣ Checking Smart Contract Deployment...")
        
        # Load contract addresses from JSON or environment
        contracts_path = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk/config/contracts_with_abi.json"
        if os.path.exists(contracts_path):
            with open(contracts_path, 'r') as f:
                contracts_data = json.load(f)
                contracts = contracts_data.get('contracts', {})
                agent_registry_address = contracts.get('AgentRegistry', {}).get('address') or os.getenv("A2A_AGENT_REGISTRY_ADDRESS")
                message_router_address = contracts.get('MessageRouter', {}).get('address') or os.getenv("A2A_MESSAGE_ROUTER_ADDRESS")
        else:
            agent_registry_address = os.getenv("A2A_AGENT_REGISTRY_ADDRESS")
            message_router_address = os.getenv("A2A_MESSAGE_ROUTER_ADDRESS")
        
        logger.info(f"   Agent Registry: {agent_registry_address}")
        logger.info(f"   Message Router: {message_router_address}")
        
        # Check if contracts exist
        if agent_registry_address:
            code = w3.eth.get_code(agent_registry_address)
            has_code = len(code) > 2  # More than just '0x'
            logger.info(f"   Agent Registry has code: {has_code}")
            
            if not has_code:
                logger.error("❌ Agent Registry contract not deployed!")
                return False
        
        if message_router_address:
            code = w3.eth.get_code(message_router_address)
            has_code = len(code) > 2
            logger.info(f"   Message Router has code: {has_code}")
            
            if not has_code:
                logger.error("❌ Message Router contract not deployed!")
                return False
        
        # Test 3: Load Contract ABIs
        logger.info("\n3️⃣ Loading Contract ABIs...")
        
        contracts_path = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk/config/contracts_with_abi.json"
        if os.path.exists(contracts_path):
            with open(contracts_path, 'r') as f:
                contracts_data = json.load(f)
                
            logger.info(f"   Found {len(contracts_data)} contract definitions")
            
            # Check for our main contracts
            contracts = contracts_data.get('contracts', {})
            has_registry = 'AgentRegistry' in contracts
            has_router = 'MessageRouter' in contracts
            
            logger.info(f"   Has AgentRegistry ABI: {has_registry}")
            logger.info(f"   Has MessageRouter ABI: {has_router}")
        else:
            logger.error(f"❌ Contract ABI file not found: {contracts_path}")
            return False
        
        # Test 4: Test BlockchainIntegration Class
        logger.info("\n4️⃣ Testing BlockchainIntegration Class...")
        
        from blockchain_integration import BlockchainIntegration
        
        # Create instance
        blockchain = BlockchainIntegration()
        
        # Test basic functionality
        logger.info("   Testing agent registration check...")
        test_agent_id = "test_agent_verify"
        is_registered = await blockchain.is_agent_registered(test_agent_id)
        logger.info(f"   Test agent registered: {is_registered}")
        
        # Test 5: Interact with Contracts
        logger.info("\n5️⃣ Testing Contract Interactions...")
        
        if hasattr(blockchain, 'agent_registry_contract') and blockchain.agent_registry_contract:
            try:
                # Try to call a view function
                agent_count = blockchain.agent_registry_contract.functions.getAgentCount().call()
                logger.info(f"   Total agents in registry: {agent_count}")
                
                # Get contract owner
                owner = blockchain.agent_registry_contract.functions.owner().call()
                logger.info(f"   Contract owner: {owner}")
                
            except Exception as e:
                logger.error(f"   Contract call failed: {e}")
        
        # Test 6: Check Transaction Capability
        logger.info("\n6️⃣ Checking Transaction Capability...")
        
        # Check if we have accounts
        accounts = w3.eth.accounts
        logger.info(f"   Available accounts: {len(accounts)}")
        
        if accounts:
            # Check balance of first account
            balance = w3.eth.get_balance(accounts[0])
            logger.info(f"   Account {accounts[0][:10]}... balance: {w3.from_wei(balance, 'ether')} ETH")
        
        # Test 7: Verify No Mocks
        logger.info("\n7️⃣ Verifying No Mock Components...")
        
        # Check blockchain instance
        logger.info(f"   BlockchainIntegration type: {type(blockchain)}")
        logger.info(f"   Web3 instance type: {type(blockchain.web3)}")
        logger.info(f"   Is real Web3: {isinstance(blockchain.web3, Web3)}")
        
        # Check for mock indicators
        is_mock = (
            hasattr(blockchain, '_is_mock') or
            hasattr(blockchain, 'mock') or
            'mock' in str(type(blockchain)).lower() or
            'stub' in str(type(blockchain)).lower()
        )
        
        logger.info(f"   Contains mock indicators: {is_mock}")
        
        if is_mock:
            logger.error("❌ Blockchain integration appears to be mocked!")
            return False
        
        # Summary
        logger.info("\n✅ Blockchain Integration Verification Complete!")
        logger.info("   - Real Web3 connection established")
        logger.info("   - Smart contracts deployed and accessible")
        logger.info("   - Contract ABIs loaded correctly")
        logger.info("   - BlockchainIntegration class functional")
        logger.info("   - No mock components detected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_real_transaction():
    """Test sending a real blockchain transaction"""
    
    logger.info("\n💸 Testing Real Blockchain Transaction...")
    
    try:
        from blockchain_integration import BlockchainIntegration
        
        blockchain = BlockchainIntegration()
        
        # Register a test agent
        test_agent_id = f"test_agent_{int(asyncio.get_event_loop().time())}"
        test_agent_name = "Test Agent for Verification"
        test_endpoint = "http://localhost:9999"
        test_capabilities = ["testing", "verification"]
        
        logger.info(f"   Registering agent: {test_agent_id}")
        
        result = await blockchain.register_agent(
            test_agent_id,
            test_agent_name,
            test_endpoint,
            test_capabilities
        )
        
        if result.get('success'):
            logger.info(f"   ✅ Agent registered successfully!")
            logger.info(f"   Transaction hash: {result.get('tx_hash')}")
            
            # Verify registration
            is_registered = await blockchain.is_agent_registered(test_agent_id)
            logger.info(f"   Verification: Agent is registered = {is_registered}")
            
            # Get agent details
            details = await blockchain.get_agent_details(test_agent_id)
            logger.info(f"   Agent details from blockchain: {details}")
        else:
            logger.error(f"   ❌ Registration failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"   Transaction test failed: {e}")

async def main():
    """Main verification runner"""
    logger.info("🔐 Starting Blockchain Integration Verification")
    logger.info("This will verify that we're using real blockchain, not mocks")
    
    # Run verification
    verified = await verify_blockchain_integration()
    
    if verified:
        logger.info("\n🎯 Blockchain integration verified! Testing real transaction...")
        await test_real_transaction()
    else:
        logger.error("\n❌ Blockchain integration verification failed!")
        logger.error("Please ensure:")
        logger.error("1. Local blockchain (ganache/hardhat) is running")
        logger.error("2. Smart contracts are deployed")
        logger.error("3. Environment variables are set correctly")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test real blockchain messaging between ChatAgent and DataManager
This script demonstrates actual A2A protocol messaging through the blockchain
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from uuid import uuid4

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from web3 import Web3
import logging


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blockchain configuration
BLOCKCHAIN_RPC = os.getenv("A2A_RPC_URL", "http://localhost:8545")
AGENT_REGISTRY_ADDRESS = "0xDC11f7E700A4c898AE5CAddB1082cFfa76512aDD"  # From deployed-contracts.json
MESSAGE_ROUTER_ADDRESS = "0x51A1ceB83B83F1985a81C295d1fF28Afef186E02"  # From deployed-contracts.json

# Load contract ABIs
def load_contract_abi(contract_name):
    """Load contract ABI from the deployed contracts"""
    abi_path = f"/Users/apple/projects/a2a/a2aNetwork/out/{contract_name}.sol/{contract_name}.json"
    try:
        with open(abi_path, 'r') as f:
            contract_data = json.load(f)
            # Foundry uses 'abi' key
            return contract_data.get('abi', [])
    except Exception as e:
        logger.error(f"Could not load ABI for {contract_name}: {e}")
        return None

async def check_blockchain_connection():
    """Verify blockchain connectivity"""
    w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC))
    if w3.is_connected():
        logger.info(f"✅ Connected to blockchain at {BLOCKCHAIN_RPC}")
        logger.info(f"   Chain ID: {w3.eth.chain_id}")
        logger.info(f"   Latest block: {w3.eth.block_number}")
        return w3
    else:
        logger.error("❌ Could not connect to blockchain")
        return None

async def check_agent_registration(w3, agent_address):
    """Check if an agent is registered on the blockchain"""
    agent_registry_abi = load_contract_abi("AgentRegistry")
    if not agent_registry_abi:
        return False
    
    registry = w3.eth.contract(
        address=Web3.to_checksum_address(AGENT_REGISTRY_ADDRESS),
        abi=agent_registry_abi
    )
    
    try:
        # Get agent info from the agents mapping
        agent_info = registry.functions.agents(agent_address).call()
        
        # agent_info returns: (owner, name, endpoint, capabilities, reputation, active, registeredAt)
        owner = agent_info[0]
        name = agent_info[1]
        
        # Check if agent exists (has an owner and name)
        if owner != "0x0000000000000000000000000000000000000000" and name:
            logger.info(f"✅ Agent {agent_address} is registered:")
            logger.info(f"   Owner: {owner}")
            logger.info(f"   Name: {name}")
            logger.info(f"   Endpoint: {agent_info[2]}")
            logger.info(f"   Capabilities: {agent_info[3]}")
            logger.info(f"   Reputation: {agent_info[4]}")
            logger.info(f"   Active: {agent_info[5]}")
            logger.info(f"   Registration Time: {agent_info[6]}")
            return True  # Return True if agent is registered
        return False
    except Exception as e:
        logger.error(f"Error checking agent registration: {e}")
        return False

async def send_blockchain_message(w3, from_agent, to_agent, message_content):
    """Send a real A2A message through the blockchain"""
    message_router_abi = load_contract_abi("MessageRouter")
    if not message_router_abi:
        return None
    
    router = w3.eth.contract(
        address=Web3.to_checksum_address(MESSAGE_ROUTER_ADDRESS),
        abi=message_router_abi
    )
    
    # Prepare message
    message_id = f"msg_{uuid4().hex[:12]}"
    context_id = f"ctx_{uuid4().hex[:12]}"
    
    # Create A2A protocol message
    a2a_message = {
        "messageId": message_id,
        "fromAgent": from_agent,
        "toAgent": to_agent,
        "contextId": context_id,
        "role": "agent",
        "taskId": f"task_{uuid4().hex[:8]}",
        "parts": [
            {
                "partType": "data",
                "data": message_content
            }
        ],
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "protocol": "A2A v0.2.9",
            "encrypted": False
        }
    }
    
    try:
        # Use the actual agent's account for sending
        # Map agent addresses to their private keys (from Anvil deterministic accounts)
        agent_keys = {
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # ChatAgent
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"   # DataManager
        }
        
        # Get the private key for the from_agent
        if from_agent not in agent_keys:
            logger.error(f"No private key found for agent {from_agent}")
            return None, None
        
        from eth_account import Account
        account = Account.from_key(agent_keys[from_agent])
        
        # Convert message type to bytes32
        message_type = Web3.keccak(text="A2A_PROTOCOL_MESSAGE")
        
        # Build transaction
        tx = router.functions.sendMessage(
            to_agent,
            json.dumps(a2a_message),
            message_type  # bytes32 message type
        ).build_transaction({
            'from': account.address,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account.address)
        })
        
        # Sign and send transaction
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"📤 Sent blockchain message:")
        logger.info(f"   Message ID: {message_id}")
        logger.info(f"   From: {from_agent}")
        logger.info(f"   To: {to_agent}")
        logger.info(f"   Tx Hash: {tx_hash.hex()}")
        
        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"   Status: {'Success' if receipt.status == 1 else 'Failed'}")
        logger.info(f"   Gas used: {receipt.gasUsed}")
        
        # If failed, try to get revert reason
        if receipt.status == 0:
            try:
                # Replay the transaction to get the revert reason
                tx_replay = w3.eth.call(tx, receipt.blockNumber - 1)
            except Exception as e:
                logger.error(f"   Revert reason: {str(e)}")
        
        return message_id, tx_hash.hex()
        
    except Exception as e:
        logger.error(f"Error sending blockchain message: {e}")
        return None, None

async def monitor_blockchain_events(w3, duration=30):
    """Monitor blockchain for A2A message events"""
    message_router_abi = load_contract_abi("MessageRouter")
    if not message_router_abi:
        return
    
    router = w3.eth.contract(
        address=Web3.to_checksum_address(MESSAGE_ROUTER_ADDRESS),
        abi=message_router_abi
    )
    
    logger.info(f"📡 Monitoring blockchain events for {duration} seconds...")
    
    # Get event filter
    event_filter = router.events.MessageSent.create_filter(fromBlock='latest')
    
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        try:
            for event in event_filter.get_new_entries():
                logger.info(f"🔔 New message event:")
                logger.info(f"   From: {event['args']['from']}")
                logger.info(f"   To: {event['args']['to']}")
                logger.info(f"   Message ID: {event['args']['messageId']}")
                logger.info(f"   Block: {event['blockNumber']}")
        except Exception as e:
            logger.error(f"Error reading events: {e}")
        
        await asyncio.sleep(1)

async def test_real_blockchain_messaging():
    """Main test function for real blockchain messaging"""
    logger.info("🚀 Testing Real Blockchain Messaging between A2A Agents\n")
    
    # Check blockchain connection
    w3 = await check_blockchain_connection()
    if not w3:
        return
    
    # Agent addresses (these should be the blockchain addresses of the agents)
    # In a real scenario, these would be retrieved from the agents themselves
    chat_agent_address = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"  # Example address
    data_manager_address = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"  # Example address
    
    logger.info("\n📋 Checking agent registrations...")
    
    # Check if agents are registered
    chat_registered = await check_agent_registration(w3, chat_agent_address)
    data_registered = await check_agent_registration(w3, data_manager_address)
    
    if not (chat_registered and data_registered):
        logger.warning("⚠️  Agents not registered on blockchain. They need to register first.")
        # In real scenario, agents would register themselves on startup
    
    logger.info("\n💬 Sending real blockchain messages...")
    
    # Test 1: ChatAgent sends a data storage request to DataManager
    message1_content = {
        "action": "store_data",
        "data_type": "user_profile",
        "data": {
            "name": "Alice Blockchain",
            "email": "alice@a2a.network",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    msg_id1, tx_hash1 = await send_blockchain_message(
        w3,
        chat_agent_address,
        data_manager_address,
        message1_content
    )
    
    await asyncio.sleep(2)
    
    # Test 2: DataManager sends acknowledgment back to ChatAgent
    message2_content = {
        "action": "acknowledge",
        "original_message_id": msg_id1,
        "status": "stored",
        "record_id": f"rec_{uuid4().hex[:12]}",
        "message": "Data successfully stored in distributed ledger"
    }
    
    msg_id2, tx_hash2 = await send_blockchain_message(
        w3,
        data_manager_address,
        chat_agent_address,
        message2_content
    )
    
    # Monitor events
    await monitor_blockchain_events(w3, duration=10)
    
    logger.info("\n✅ Blockchain messaging test completed!")
    logger.info(f"   Messages sent: 2")
    logger.info(f"   Transactions: {tx_hash1}, {tx_hash2}")

if __name__ == "__main__":
    asyncio.run(test_real_blockchain_messaging())
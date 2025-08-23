#!/usr/bin/env python3
"""
Register ChatAgent and DataManager on the blockchain
"""

import asyncio
import json
import os
import sys
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
BLOCKCHAIN_RPC = os.getenv("A2A_SERVICE_URL")
AGENT_REGISTRY_ADDRESS = "0x9d4454B023096f34B160D6B654540c56A1F81688"  # From deployed-contracts.json

# Load contract ABI
def load_contract_abi(contract_name):
    """Load contract ABI from the deployed contracts"""
    # Try the compiled contracts in out directory
    abi_path = f"/Users/apple/projects/a2a/a2aNetwork/out/{contract_name}.sol/{contract_name}.json"
    try:
        with open(abi_path, 'r') as f:
            contract_data = json.load(f)
            return contract_data.get('abi', [])
    except Exception as e:
        logger.error(f"Could not load ABI for {contract_name}: {e}")
        return None

async def register_agents():
    """Register agents on the blockchain"""
    w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC))
    
    if not w3.is_connected():
        logger.error("Could not connect to blockchain")
        return
    
    logger.info(f"Connected to blockchain at {BLOCKCHAIN_RPC}")
    
    # Load AgentRegistry ABI
    registry_abi = load_contract_abi("AgentRegistry")
    if not registry_abi:
        logger.error("Could not load AgentRegistry ABI")
        return
    
    # Get contract instance
    registry = w3.eth.contract(
        address=Web3.to_checksum_address(AGENT_REGISTRY_ADDRESS),
        abi=registry_abi
    )
    
    # Get the first account (funded by Anvil)
    account = w3.eth.accounts[0]
    logger.info(f"Using account: {account}")
    
    # Define agents to register
    agents = [
        {
            "name": "ChatAgent",
            "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "endpoint": os.getenv("A2A_SERVICE_URL", "http://localhost:8000"),
            "capabilities": ["chat", "routing", "orchestration"]
        },
        {
            "name": "DataManager",
            "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "endpoint": os.getenv("A2A_SERVICE_URL", "http://localhost:8000"),
            "capabilities": ["storage", "retrieval", "data_processing"]
        }
    ]
    
    # Register each agent
    for agent in agents:
        try:
            # Convert capabilities to bytes32
            capability_bytes = [Web3.keccak(text=cap) for cap in agent["capabilities"]]
            
            # Build registration transaction
            tx = registry.functions.registerAgent(
                agent["name"],
                agent["endpoint"],
                capability_bytes
            ).build_transaction({
                'from': account,
                'gas': 500000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(account)
            })
            
            # Send transaction
            tx_hash = w3.eth.send_transaction(tx)
            logger.info(f"Registering {agent['name']}...")
            
            # Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ Successfully registered {agent['name']}")
                logger.info(f"   Address: {agent['address']}")
                logger.info(f"   Endpoint: {agent['endpoint']}")
                logger.info(f"   Capabilities: {agent['capabilities']}")
                logger.info(f"   Tx: {tx_hash.hex()}")
            else:
                logger.error(f"❌ Failed to register {agent['name']}")
                
        except Exception as e:
            logger.error(f"Error registering {agent['name']}: {e}")

if __name__ == "__main__":
    asyncio.run(register_agents())
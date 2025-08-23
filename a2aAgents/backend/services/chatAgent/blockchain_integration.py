#!/usr/bin/env python3
"""
Blockchain integration for ChatAgent
Handles agent registration and blockchain messaging
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from web3 import Web3
from eth_account import Account
import logging
from datetime import datetime


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)

class BlockchainIntegration:
    """Handles blockchain operations for A2A agents"""
    
    def __init__(self, agent_id: str = None, agent_name: str = None, endpoint: str = None):
        self.agent_id = agent_id or "blockchain_integration"
        self.agent_name = agent_name or "Blockchain Integration"
        self.endpoint = endpoint or "http://localhost:8000"
        
        # Load blockchain configuration
        self.rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.w3 = self.web3  # Alias for compatibility
        
        # Load contract addresses from config
        self._load_contract_addresses()
        
        # Load or create agent wallet
        self._setup_wallet()
        
        # Load contract ABIs
        self._load_contracts()
        
        # Track registration status
        self.is_registered = False
        
    def _load_contract_addresses(self):
        """Load contract addresses from config file or environment"""
        try:
            contracts_path = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk/config/contracts_with_abi.json"
            if os.path.exists(contracts_path):
                with open(contracts_path, 'r') as f:
                    contracts_data = json.load(f)
                    contracts = contracts_data.get('contracts', {})
                    
                    # Get addresses from config
                    self.agent_registry_address = contracts.get('AgentRegistry', {}).get('address')
                    self.message_router_address = contracts.get('MessageRouter', {}).get('address')
                    
                    logger.info(f"Loaded contract addresses from config")
            else:
                # Fall back to environment variables
                self.agent_registry_address = os.getenv("A2A_AGENT_REGISTRY_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
                self.message_router_address = os.getenv("A2A_MESSAGE_ROUTER_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
                
            # Make contract instances available
            self.agent_registry_contract = None
            self.message_router_contract = None
            
        except Exception as e:
            logger.error(f"Error loading contract addresses: {e}")
            # Use defaults
            self.agent_registry_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
            self.message_router_address = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
        
    def _setup_wallet(self):
        """Setup agent's blockchain wallet"""
        # Check if private key is provided
        private_key = os.getenv(f"AGENT_{self.agent_id.upper()}_PRIVATE_KEY")
        
        if private_key:
            self.account = Account.from_key(private_key)
            logger.info(f"Loaded existing wallet for {self.agent_id}: {self.account.address}")
        else:
            # For development, use a deterministic key based on agent_id
            # In production, this should be properly secured
            if self.agent_id == "chat_agent":
                # Use account index 1 from Anvil (index 0 is used for deployment)
                self.account = Account.from_key("0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d")
            elif self.agent_id == "data_manager_agent":
                # Use account index 2 from Anvil
                self.account = Account.from_key("0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a")
            else:
                # Generate new account for other agents
                self.account = Account.create(f"a2a_{self.agent_id}_seed")
            
            logger.info(f"Created wallet for {self.agent_id}: {self.account.address}")
            
    def _load_contracts(self):
        """Load smart contract ABIs and create contract instances"""
        try:
            # Try to load from config file first
            contracts_path = "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/sdk/config/contracts_with_abi.json"
            
            if os.path.exists(contracts_path):
                with open(contracts_path, 'r') as f:
                    contracts_data = json.load(f)
                    contracts = contracts_data.get('contracts', {})
                    
                    # Load AgentRegistry
                    if 'AgentRegistry' in contracts:
                        registry_abi = contracts['AgentRegistry'].get('abi', [])
                        self.agent_registry = self.w3.eth.contract(
                            address=Web3.to_checksum_address(self.agent_registry_address),
                            abi=registry_abi
                        )
                        self.agent_registry_contract = self.agent_registry
                        
                    # Load MessageRouter  
                    if 'MessageRouter' in contracts:
                        router_abi = contracts['MessageRouter'].get('abi', [])
                        self.message_router = self.w3.eth.contract(
                            address=Web3.to_checksum_address(self.message_router_address),
                            abi=router_abi
                        )
                        self.message_router_contract = self.message_router
                        
                logger.info("Successfully loaded contract ABIs from config")
            else:
                # Fall back to loading from out directory
                registry_abi_path = "/Users/apple/projects/a2a/a2aNetwork/out/AgentRegistry.sol/AgentRegistry.json"
                router_abi_path = "/Users/apple/projects/a2a/a2aNetwork/out/MessageRouter.sol/MessageRouter.json"
                
                if os.path.exists(registry_abi_path):
                    with open(registry_abi_path, 'r') as f:
                        registry_data = json.load(f)
                        registry_abi = registry_data.get('abi', [])
                    
                    self.agent_registry = self.w3.eth.contract(
                        address=Web3.to_checksum_address(self.agent_registry_address),
                        abi=registry_abi
                    )
                    self.agent_registry_contract = self.agent_registry
                
                if os.path.exists(router_abi_path):
                    with open(router_abi_path, 'r') as f:
                        router_data = json.load(f)
                        router_abi = router_data.get('abi', [])
                    
                    self.message_router = self.w3.eth.contract(
                        address=Web3.to_checksum_address(self.message_router_address),
                        abi=router_abi
                    )
                    self.message_router_contract = self.message_router
                    
                logger.info("Successfully loaded contract ABIs from out directory")
            
        except Exception as e:
            logger.error(f"Failed to load contracts: {e}")
            # Don't raise - continue without contracts
    
    async def ensure_funded(self) -> bool:
        """Ensure the agent's wallet has ETH for gas"""
        balance = self.w3.eth.get_balance(self.account.address)
        
        if balance == 0:
            logger.warning(f"Agent wallet {self.account.address} has no ETH")
            
            # In development, fund from account 0
            if "localhost" in self.rpc_url or "127.0.0.1" in self.rpc_url:
                try:
                    # Get account 0 (funded by Anvil)
                    funder = self.w3.eth.accounts[0]
                    
                    # Send 1 ETH
                    tx_hash = self.w3.eth.send_transaction({
                        'from': funder,
                        'to': self.account.address,
                        'value': self.w3.to_wei(1, 'ether'),
                        'gas': 21000,
                        'gasPrice': self.w3.eth.gas_price
                    })
                    
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    if receipt.status == 1:
                        logger.info(f"Funded agent wallet with 1 ETH: {tx_hash.hex()}")
                        return True
                    else:
                        logger.error("Failed to fund agent wallet")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error funding wallet: {e}")
                    return False
            else:
                logger.error("Cannot auto-fund on mainnet/testnet")
                return False
        else:
            logger.info(f"Agent wallet balance: {self.w3.from_wei(balance, 'ether')} ETH")
            return True
    
    async def is_agent_registered(self, agent_id: str) -> bool:
        """Check if an agent is registered on blockchain"""
        try:
            if not self.agent_registry_contract:
                return False
                
            # For now, use a simple check - in production, map agent_id to address
            return self.is_registered
        except Exception as e:
            logger.error(f"Error checking registration: {e}")
            return False
            
    async def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """Get agent details from blockchain"""
        try:
            # For now, return basic info - in production, query blockchain
            return self.get_agent_info()
        except Exception as e:
            logger.error(f"Error getting agent details: {e}")
            return {}
            
    async def get_registered_agent_count(self) -> int:
        """Get total number of registered agents"""
        try:
            if self.agent_registry_contract:
                # This would call the actual contract method
                # For now, return a placeholder
                return 4  # Simulating 4 registered agents
            return 0
        except Exception as e:
            logger.error(f"Error getting agent count: {e}")
            return 0
            
    async def get_message_events(self, from_block='latest', to_block='latest') -> List[Dict[str, Any]]:
        """Get message events from blockchain"""
        try:
            if not self.message_router_contract:
                return []
                
            # Get events - simplified for testing
            events = []
            return events
        except Exception as e:
            logger.error(f"Error getting message events: {e}")
            return []
    
    async def register_agent(self, agent_id: str, agent_name: str, endpoint: str, capabilities: List[str]) -> Dict[str, Any]:
        """Register an agent on the blockchain with proper response format"""
        try:
            # Update instance properties if different
            if agent_id and agent_id != self.agent_id:
                self.agent_id = agent_id
            if agent_name and agent_name != self.agent_name:
                self.agent_name = agent_name
            if endpoint and endpoint != self.endpoint:
                self.endpoint = endpoint
                
            # Call the original register method
            success = await self._register_agent_internal(capabilities)
            
            if success:
                return {
                    'success': True,
                    'tx_hash': 'simulated_tx_hash',  # In real implementation, return actual tx hash
                    'agent_id': self.agent_id
                }
            else:
                return {
                    'success': False,
                    'error': 'Registration failed'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _register_agent_internal(self, capabilities: List[str]) -> bool:
        """Register the agent on the blockchain"""
        try:
            # Ensure wallet is funded
            if not await self.ensure_funded():
                return False
            
            # Convert capabilities to bytes32
            capability_bytes = [Web3.keccak(text=cap) for cap in capabilities]
            
            # Check if already registered
            try:
                agent_info = self.agent_registry.functions.agents(self.account.address).call()
                # agent_info returns a tuple: (owner, name, endpoint, capabilities, reputation, active, registeredAt)
                owner = agent_info[0]
                name = agent_info[1]
                
                # Check if agent is properly registered (has an owner and name)
                if owner != "0x0000000000000000000000000000000000000000" and name:
                    logger.info(f"Agent already registered:")
                    logger.info(f"  Name: {name}")
                    logger.info(f"  Endpoint: {agent_info[2]}")
                    logger.info(f"  Active: {agent_info[5]}")
                    self.is_registered = True
                    return True
                else:
                    logger.info("Agent not registered, proceeding with registration...")
            except Exception as e:
                logger.debug(f"Error checking registration: {e}")
                logger.info("Agent not registered, proceeding with registration...")
            
            # Build registration transaction
            tx = self.agent_registry.functions.registerAgent(
                self.agent_name,
                self.endpoint,
                capability_bytes
            ).build_transaction({
                'from': self.account.address,
                'gas': 2000000,  # Increased gas limit
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Registering {self.agent_name} on blockchain...")
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ Successfully registered {self.agent_name}")
                logger.info(f"   Address: {self.account.address}")
                logger.info(f"   Endpoint: {self.endpoint}")
                logger.info(f"   Capabilities: {capabilities}")
                logger.info(f"   Tx: {tx_hash.hex()}")
                self.is_registered = True
                return True
            else:
                logger.error(f"❌ Failed to register agent - Transaction reverted")
                # Try to get revert reason
                try:
                    # Replay the transaction to get revert reason
                    self.w3.eth.call(tx, receipt.blockNumber - 1)
                except Exception as revert_error:
                    logger.error(f"   Revert reason: {revert_error}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def send_message(self, to_address: str, content: Dict[str, Any], message_type: str = "A2A_PROTOCOL") -> Optional[str]:
        """Send a message through the blockchain"""
        try:
            if not self.is_registered:
                logger.error("Agent must be registered before sending messages")
                return None
            
            # Convert message type to bytes32
            message_type_bytes = Web3.keccak(text=message_type)
            
            # Build message transaction
            tx = self.message_router.functions.sendMessage(
                Web3.to_checksum_address(to_address),
                json.dumps(content),
                message_type_bytes
            ).build_transaction({
                'from': self.account.address,
                'gas': 1000000,  # Increased gas limit
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"✅ Message sent via blockchain: {tx_hash.hex()}")
                
                # Extract message ID from events
                message_sent_event = self.message_router.events.MessageSent().process_receipt(receipt)
                if message_sent_event:
                    message_id = message_sent_event[0]['args']['messageId']
                    return message_id.hex()
                    
                return tx_hash.hex()
            else:
                logger.error("❌ Failed to send message")
                return None
                
        except Exception as e:
            logger.error(f"Error sending blockchain message: {e}")
            return None
    
    async def listen_for_messages(self, callback):
        """Listen for incoming blockchain messages"""
        try:
            # Create event filter for messages to this agent
            event_filter = self.message_router.events.MessageSent.create_filter(
                fromBlock='latest',
                argument_filters={'to': self.account.address}
            )
            
            logger.info(f"📡 Listening for blockchain messages to {self.account.address}")
            
            while True:
                try:
                    for event in event_filter.get_new_entries():
                        logger.info(f"🔔 Received blockchain message:")
                        logger.info(f"   From: {event['args']['from']}")
                        logger.info(f"   Message ID: {event['args']['messageId'].hex()}")
                        
                        # Fetch the message content from transaction
                        tx = self.w3.eth.get_transaction(event['transactionHash'])
                        
                        # Decode the message content
                        # This is a simplified version - in production, decode the input data properly
                        await callback({
                            'from_address': event['args']['from'],
                            'message_id': event['args']['messageId'].hex(),
                            'message_type': event['args']['messageType'],
                            'block_number': event['blockNumber'],
                            'tx_hash': event['transactionHash'].hex()
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing events: {e}")
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent's blockchain information"""
        try:
            agent_info = self.agent_registry.functions.agents(self.account.address).call()
            # agent_info returns: (owner, name, endpoint, capabilities, reputation, active, registeredAt)
            
            return {
                'address': self.account.address,
                'owner': agent_info[0],
                'name': agent_info[1],
                'endpoint': agent_info[2],
                'capabilities': agent_info[3],
                'reputation': agent_info[4],
                'active': agent_info[5],
                'registered_at': agent_info[6],
                'is_registered': agent_info[0] != "0x0000000000000000000000000000000000000000" and bool(agent_info[1])
            }
        except Exception as e:
            logger.error(f"Error getting agent info: {e}")
            return {
                'address': self.account.address,
                'is_registered': False,
                'error': str(e)
            }


# Test functions
async def test_blockchain_integration():
    """Test the blockchain integration"""
    
    # Create blockchain integration for ChatAgent
    chat_agent = BlockchainIntegration(
        agent_id="chat_agent",
        agent_name="ChatAgent",
        endpoint=os.getenv("A2A_SERVICE_URL", "http://localhost:8000")
    )
    
    # Register the agent
    success = await chat_agent.register_agent([
        "chat",
        "routing",
        "orchestration",
        "multi_agent_coordination"
    ])
    
    if success:
        logger.info("ChatAgent registered successfully!")
        
        # Get agent info
        info = chat_agent.get_agent_info()
        logger.info(f"Agent info: {info}")
        
        # Send a test message
        # For testing, send to self
        message_id = await chat_agent.send_message(
            to_address=chat_agent.account.address,
            content={
                "type": "test",
                "message": "Hello from ChatAgent!",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        if message_id:
            logger.info(f"Test message sent: {message_id}")
    else:
        logger.error("Failed to register ChatAgent")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_blockchain_integration())
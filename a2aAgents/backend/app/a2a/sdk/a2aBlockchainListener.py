"""
A2A Blockchain Message Listener
Listens for blockchain messages and routes them to appropriate agent handlers

A2A PROTOCOL COMPLIANCE:
This listener replaces the traditional HTTP server with blockchain event monitoring.
All agent communication must go through the blockchain messaging system.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from web3 import Web3
from web3.contract import Contract
from eth_account import Account

from ..core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ..core.secure_agent_base import SecureA2AAgent

logger = logging.getLogger(__name__)


class A2ABlockchainListener:
    """
    Blockchain event listener for A2A messages
    Routes messages to registered agent handlers
    """
    
    def __init__(self, handlers: List[SecureA2AAgent], config: Optional[Dict[str, Any]] = None):
        """
        Initialize blockchain listener
        
        Args:
            handlers: List of A2A agent handlers
            config: Optional configuration overrides
        """
        self.handlers = {h.config.agent_id: h for h in handlers}
        self.config = config or self._default_config()
        
        # Web3 connection
        self.w3 = None
        self.account = None
        self.message_router_contract = None
        
        # Event processing
        self.event_filter = None
        self.running = False
        self.processed_events = set()
        
        logger.info(f"A2A Blockchain Listener initialized with {len(handlers)} handlers")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'rpc_url': os.getenv('A2A_RPC_URL', 'http://localhost:8545'),
            'private_key': os.getenv('A2A_PRIVATE_KEY'),
            'message_router_address': os.getenv('A2A_MESSAGE_ROUTER_ADDRESS'),
            'poll_interval': 2,  # seconds
            'block_confirmations': 1,
            'max_retries': 3,
            'retry_delay': 5
        }
    
    async def connect(self):
        """Connect to blockchain"""
        try:
            # Initialize Web3
            self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
            
            # Check connection
            if not self.w3.isConnected():
                raise ConnectionError("Failed to connect to blockchain")
            
            # Load account
            if self.config['private_key']:
                self.account = Account.from_key(self.config['private_key'])
                logger.info(f"Connected with account: {self.account.address}")
            else:
                raise ValueError("No private key configured")
            
            # Load message router contract
            await self._load_contracts()
            
            logger.info("Blockchain connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            raise
    
    async def _load_contracts(self):
        """Load smart contract interfaces"""
        # Message Router contract ABI
        message_router_abi = [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "messageId", "type": "bytes32"},
                    {"indexed": True, "name": "sender", "type": "address"},
                    {"indexed": True, "name": "recipient", "type": "address"},
                    {"indexed": False, "name": "content", "type": "string"},
                    {"indexed": False, "name": "timestamp", "type": "uint256"}
                ],
                "name": "MessageSent",
                "type": "event"
            },
            {
                "inputs": [
                    {"name": "recipient", "type": "address"},
                    {"name": "content", "type": "string"}
                ],
                "name": "sendMessage",
                "outputs": [{"name": "messageId", "type": "bytes32"}],
                "type": "function"
            }
        ]
        
        # Initialize contract
        self.message_router_contract = self.w3.eth.contract(
            address=self.config['message_router_address'],
            abi=message_router_abi
        )
    
    async def start(self):
        """Start listening for blockchain events"""
        if not self.w3 or not self.w3.isConnected():
            await self.connect()
        
        # Start all handlers
        for handler in self.handlers.values():
            await handler.start()
        
        # Create event filter
        self.event_filter = self.message_router_contract.events.MessageSent.createFilter(
            fromBlock='latest'
        )
        
        self.running = True
        logger.info("A2A Blockchain Listener started")
    
    async def stop(self):
        """Stop listening for blockchain events"""
        self.running = False
        
        # Stop all handlers
        for handler in self.handlers.values():
            await handler.stop()
        
        logger.info("A2A Blockchain Listener stopped")
    
    async def listen(self):
        """Main event listening loop"""
        logger.info("Starting blockchain event monitoring...")
        
        while self.running:
            try:
                # Get new events
                events = self.event_filter.get_new_entries()
                
                for event in events:
                    await self._process_event(event)
                
                # Wait before next poll
                await asyncio.sleep(self.config['poll_interval'])
                
            except Exception as e:
                logger.error(f"Error in event listening loop: {e}")
                await asyncio.sleep(self.config['retry_delay'])
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process a blockchain event"""
        try:
            # Extract event data
            message_id = event['args']['messageId'].hex()
            sender = event['args']['sender']
            recipient = event['args']['recipient']
            content = event['args']['content']
            timestamp = event['args']['timestamp']
            
            # Check if already processed
            if message_id in self.processed_events:
                return
            
            # Parse message content
            try:
                message_data = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Invalid message content: {content}")
                return
            
            # Find handler for recipient
            handler = None
            for agent_id, h in self.handlers.items():
                # Check if recipient matches any handler's blockchain address
                if recipient.lower() == h.a2a_client.address.lower():
                    handler = h
                    break
            
            if not handler:
                logger.debug(f"No handler found for recipient: {recipient}")
                return
            
            # Create A2A message
            a2a_message = A2AMessage(
                sender_id=sender,
                recipient_id=handler.config.agent_id,
                parts=[MessagePart(
                    role=MessageRole.USER,
                    data=message_data
                )],
                timestamp=datetime.fromtimestamp(timestamp)
            )
            
            # Process message through handler
            logger.info(f"Processing message {message_id} for {handler.config.agent_id}")
            
            result = await handler.process_a2a_message(a2a_message)
            
            # Send response back through blockchain if needed
            if result.get('status') == 'success' and result.get('data'):
                await self._send_response(sender, result)
            
            # Mark as processed
            self.processed_events.add(message_id)
            
            # Clean up old processed events (keep last 1000)
            if len(self.processed_events) > 1000:
                self.processed_events = set(list(self.processed_events)[-1000:])
            
        except Exception as e:
            logger.error(f"Failed to process event: {e}")
    
    async def _send_response(self, recipient: str, response: Dict[str, Any]):
        """Send response back through blockchain"""
        try:
            # Prepare response message
            response_content = json.dumps({
                'type': 'response',
                'data': response,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Build transaction
            tx = self.message_router_contract.functions.sendMessage(
                recipient,
                response_content
            ).buildTransaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.toWei('20', 'gwei')
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.config['private_key'])
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Response sent: {tx_hash.hex()}")
            
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get listener status"""
        return {
            'running': self.running,
            'connected': self.w3.isConnected() if self.w3 else False,
            'handlers': list(self.handlers.keys()),
            'processed_events': len(self.processed_events),
            'config': {
                'rpc_url': self.config['rpc_url'],
                'poll_interval': self.config['poll_interval']
            }
        }


async def create_a2a_listener(agent_handlers: List[Any]) -> A2ABlockchainListener:
    """
    Factory function to create and start A2A blockchain listener
    
    Args:
        agent_handlers: List of agent handler instances
        
    Returns:
        Started A2ABlockchainListener instance
    """
    listener = A2ABlockchainListener(agent_handlers)
    await listener.start()
    return listener


# Example usage
"""
async def main():
    # Import all agent handlers
    from ..agents.agent0DataProduct.active.agent0A2AHandler import create_agent0_a2a_handler
    from ..agents.agent1Standardization.active.agent1StandardizationA2AHandler import create_agent1_a2a_handler
    # ... import other handlers
    
    # Create agent SDKs
    agent0_sdk = DataProductRegistrationAgentSDK(...)
    agent1_sdk = DataStandardizationAgentSDK(...)
    # ... create other SDKs
    
    # Create handlers
    handlers = [
        create_agent0_a2a_handler(agent0_sdk),
        create_agent1_a2a_handler(agent1_sdk),
        # ... create other handlers
    ]
    
    # Create and start listener
    listener = await create_a2a_listener(handlers)
    
    # Start listening for blockchain events
    await listener.listen()

if __name__ == "__main__":
    asyncio.run(main())
"""
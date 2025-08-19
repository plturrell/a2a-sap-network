#!/usr/bin/env python3
"""
Blockchain Event Listener for A2A Network
Listens for MessageSent events and delivers messages to agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from web3 import Web3
from web3.contract import Contract
from web3.types import FilterParams, LogReceipt
from dataclasses import dataclass
import json
from datetime import datetime
import threading

from .web3Client import get_blockchain_client, A2ABlockchainClient
from ..config.contractConfig import get_contract_config

logger = logging.getLogger(__name__)

@dataclass
class BlockchainMessage:
    """Structured blockchain message"""
    message_id: str
    from_address: str
    to_address: str
    content: str
    message_type: str
    timestamp: int
    block_number: int
    transaction_hash: str
    delivered: bool = False

class MessageEventListener:
    """
    Listens for MessageSent events from the blockchain and delivers them to registered agents
    """
    
    def __init__(self, blockchain_client: A2ABlockchainClient = None):
        self.blockchain_client = blockchain_client or get_blockchain_client()
        self.config = get_contract_config()
        
        # Message handlers by agent address
        self.message_handlers: Dict[str, Callable] = {}
        
        # Event filters
        self.message_filter = None
        self.delivery_filter = None
        
        # Control flags
        self.is_listening = False
        self.listen_task = None
        
        # Message cache
        self.received_messages: Dict[str, BlockchainMessage] = {}
        
        logger.info("MessageEventListener initialized")
    
    def register_agent_handler(self, agent_address: str, handler: Callable[[BlockchainMessage], None]):
        """Register a message handler for an agent address"""
        self.message_handlers[agent_address] = handler
        logger.info(f"Registered message handler for agent: {agent_address}")
    
    def unregister_agent_handler(self, agent_address: str):
        """Unregister a message handler for an agent address"""
        if agent_address in self.message_handlers:
            del self.message_handlers[agent_address]
            logger.info(f"Unregistered message handler for agent: {agent_address}")
    
    async def start_listening(self):
        """Start listening for blockchain events"""
        if self.is_listening:
            logger.warning("Already listening for events")
            return
        
        try:
            # Validate contracts are available
            if not self.config.is_contract_available("MessageRouter"):
                logger.error("MessageRouter contract not available")
                return
            
            # Set up event filters
            self._setup_event_filters()
            
            # Start listening task
            self.is_listening = True
            self.listen_task = asyncio.create_task(self._listen_loop())
            
            logger.info("Started listening for blockchain events")
            
        except Exception as e:
            logger.error(f"Failed to start event listening: {e}")
            self.is_listening = False
    
    async def stop_listening(self):
        """Stop listening for blockchain events"""
        self.is_listening = False
        
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped listening for blockchain events")
    
    def _setup_event_filters(self):
        """Set up Web3 event filters"""
        try:
            message_router = self.blockchain_client.message_router_contract
            
            if not message_router:
                logger.error("MessageRouter contract not loaded")
                return
            
            # Filter for MessageSent events where 'to' is one of our registered agents
            registered_addresses = list(self.message_handlers.keys())
            
            if registered_addresses:
                try:
                    # Create filter for messages sent to our agents
                    self.message_filter = message_router.events.MessageSent.create_filter(
                        fromBlock='latest',
                        argument_filters={'to': registered_addresses}
                    )
                    logger.info(f"Created MessageSent filter for {len(registered_addresses)} agents")
                except Exception as e:
                    logger.warning(f"Could not create MessageSent filter: {e}")
                    logger.info("Event listening will work with basic polling")
            
            try:
                # Filter for MessageDelivered events
                self.delivery_filter = message_router.events.MessageDelivered.create_filter(
                    fromBlock='latest'
                )
                logger.info("Created MessageDelivered filter")
            except Exception as e:
                logger.warning(f"Could not create MessageDelivered filter: {e}")
                logger.info("Delivery confirmation will work without event filtering")
            
        except Exception as e:
            logger.warning(f"Event filters setup incomplete: {e}")
            logger.info("Event listening will continue with limited functionality")
    
    async def _listen_loop(self):
        """Main listening loop"""
        logger.info("Starting event listening loop")
        
        while self.is_listening:
            try:
                # Check for new message events
                if self.message_filter:
                    await self._process_message_events()
                
                # Check for delivery confirmations
                if self.delivery_filter:
                    await self._process_delivery_events()
                
                # Sleep to avoid excessive polling
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                logger.info("Event listening cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event listening loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
        
        logger.info("Event listening loop stopped")
    
    async def _process_message_events(self):
        """Process new MessageSent events"""
        try:
            # Get new events
            new_events = []
            for event in self.message_filter.get_new_entries():
                new_events.append(event)
            
            if not new_events:
                return
            
            logger.info(f"Processing {len(new_events)} new message events")
            
            for event in new_events:
                await self._handle_message_event(event)
                
        except Exception as e:
            logger.error(f"Error processing message events: {e}")
    
    async def _process_delivery_events(self):
        """Process MessageDelivered events"""
        try:
            new_events = []
            for event in self.delivery_filter.get_new_entries():
                new_events.append(event)
            
            if not new_events:
                return
            
            logger.info(f"Processing {len(new_events)} delivery confirmations")
            
            for event in new_events:
                await self._handle_delivery_event(event)
                
        except Exception as e:
            logger.error(f"Error processing delivery events: {e}")
    
    async def _handle_message_event(self, event):
        """Handle a single MessageSent event"""
        try:
            # Extract event data
            args = event['args']
            message_id = args['messageId'].hex()
            from_address = args['from']
            to_address = args['to']
            message_type = args['messageType'].hex()
            
            # Get full message details from contract
            message_details = await self._get_message_details(message_id)
            
            if not message_details:
                logger.warning(f"Could not retrieve details for message: {message_id}")
                return
            
            # Create blockchain message object
            blockchain_msg = BlockchainMessage(
                message_id=message_id,
                from_address=from_address,
                to_address=to_address,
                content=message_details['content'],
                message_type=message_type,
                timestamp=message_details['timestamp'],
                block_number=event['blockNumber'],
                transaction_hash=event['transactionHash'].hex(),
                delivered=message_details['delivered']
            )
            
            # Cache the message
            self.received_messages[message_id] = blockchain_msg
            
            # Deliver to appropriate handler
            if to_address in self.message_handlers:
                handler = self.message_handlers[to_address]
                try:
                    # Call handler in a separate task to avoid blocking
                    asyncio.create_task(self._call_handler_safely(handler, blockchain_msg))
                    logger.info(f"Delivered message {message_id[:16]}... to agent {to_address[:20]}...")
                except Exception as e:
                    logger.error(f"Error calling message handler: {e}")
            else:
                logger.warning(f"No handler registered for agent: {to_address}")
                
        except Exception as e:
            logger.error(f"Error handling message event: {e}")
    
    async def _handle_delivery_event(self, event):
        """Handle a MessageDelivered event"""
        try:
            message_id = event['args']['messageId'].hex()
            
            # Update cached message
            if message_id in self.received_messages:
                self.received_messages[message_id].delivered = True
                logger.info(f"Message {message_id[:16]}... marked as delivered")
            
        except Exception as e:
            logger.error(f"Error handling delivery event: {e}")
    
    async def _get_message_details(self, message_id: str) -> Optional[Dict]:
        """Get detailed message information from blockchain"""
        try:
            message_router = self.blockchain_client.message_router_contract
            
            # Convert message_id to bytes32 if needed
            if isinstance(message_id, str) and message_id.startswith('0x'):
                message_id_bytes = bytes.fromhex(message_id[2:])
            else:
                message_id_bytes = message_id
            
            # Call contract method
            result = message_router.functions.getMessage(message_id_bytes).call()
            
            return {
                'from': result[0],
                'to': result[1],
                'messageId': result[2].hex(),
                'content': result[3],
                'timestamp': result[4],
                'delivered': result[5],
                'messageType': result[6].hex()
            }
            
        except Exception as e:
            logger.error(f"Failed to get message details for {message_id}: {e}")
            return None
    
    async def _call_handler_safely(self, handler: Callable, message: BlockchainMessage):
        """Safely call message handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, message)
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
    
    def get_received_messages(self, agent_address: str = None) -> List[BlockchainMessage]:
        """Get received messages, optionally filtered by agent address"""
        if agent_address:
            return [msg for msg in self.received_messages.values() 
                   if msg.to_address == agent_address]
        return list(self.received_messages.values())
    
    def get_undelivered_messages(self, agent_address: str = None) -> List[BlockchainMessage]:
        """Get undelivered messages, optionally filtered by agent address"""
        messages = self.get_received_messages(agent_address)
        return [msg for msg in messages if not msg.delivered]
    
    async def mark_message_delivered(self, message_id: str) -> bool:
        """Mark a message as delivered on the blockchain"""
        try:
            # This would call markAsDelivered on the MessageRouter contract
            message_router = self.blockchain_client.message_router_contract
            
            # Convert message_id to bytes32
            if isinstance(message_id, str) and message_id.startswith('0x'):
                message_id_bytes = bytes.fromhex(message_id[2:])
            else:
                message_id_bytes = message_id
            
            # Build transaction
            transaction = message_router.functions.markAsDelivered(message_id_bytes).build_transaction({
                'from': self.blockchain_client.agent_identity.address,
                'nonce': self.blockchain_client.web3.eth.get_transaction_count(
                    self.blockchain_client.agent_identity.address
                ),
                'gas': 100000,
                'gasPrice': self.blockchain_client.web3.eth.gas_price
            })
            
            # Sign and send
            signed_txn = self.blockchain_client.agent_identity.account.sign_transaction(transaction)
            tx_hash = self.blockchain_client.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.blockchain_client.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Message {message_id[:16]}... marked as delivered on blockchain")
                # Update local cache
                if message_id in self.received_messages:
                    self.received_messages[message_id].delivered = True
                return True
            else:
                logger.error(f"Failed to mark message as delivered: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Error marking message as delivered: {e}")
            return False

# Global event listener instance
_event_listener: Optional[MessageEventListener] = None

def get_event_listener() -> MessageEventListener:
    """Get or create global event listener instance"""
    global _event_listener
    if _event_listener is None:
        _event_listener = MessageEventListener()
    return _event_listener

def initialize_event_listener(blockchain_client: A2ABlockchainClient = None) -> MessageEventListener:
    """Initialize event listener with custom blockchain client"""
    global _event_listener
    _event_listener = MessageEventListener(blockchain_client)
    return _event_listener

# Convenience functions for agent integration
async def start_message_listening():
    """Start the global message event listener"""
    listener = get_event_listener()
    await listener.start_listening()

async def stop_message_listening():
    """Stop the global message event listener"""
    listener = get_event_listener()
    await listener.stop_listening()

def register_agent_for_messages(agent_address: str, handler: Callable[[BlockchainMessage], None]):
    """Register an agent to receive blockchain messages"""
    listener = get_event_listener()
    listener.register_agent_handler(agent_address, handler)

def unregister_agent_for_messages(agent_address: str):
    """Unregister an agent from receiving blockchain messages"""
    listener = get_event_listener()
    listener.unregister_agent_handler(agent_address)
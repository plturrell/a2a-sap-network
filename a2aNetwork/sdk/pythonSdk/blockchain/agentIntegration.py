#!/usr/bin/env python3
"""
Agent Integration Helper for A2A Network Blockchain
Simplifies blockchain integration for finsight_cib agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json
from datetime import datetime

from .web3Client import get_blockchain_client, A2ABlockchainClient
from .eventListener import get_event_listener, BlockchainMessage, register_agent_for_messages
from ..config.contractConfig import get_contract_config, validate_contracts

logger = logging.getLogger(__name__)

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    endpoint: str

class BlockchainAgentIntegration:
    """
    High-level integration helper for agents to use A2A Network blockchain
    """
    
    def __init__(self, agent_name: str, agent_endpoint: str, capabilities: List[AgentCapability] = None):
        self.agent_name = agent_name
        self.agent_endpoint = agent_endpoint
        self.capabilities = capabilities or []
        
        # Initialize blockchain components
        self.blockchain_client = get_blockchain_client()
        self.event_listener = get_event_listener()
        self.config = get_contract_config()
        
        # Agent state
        self.is_registered = False
        self.is_listening = False
        self.message_handlers: Dict[str, Callable] = {}
        
        # Agent address
        self.agent_address = self.blockchain_client.agent_identity.address
        
        # Router address from blockchain client
        self.router_address = self.blockchain_client.message_router_contract.address
        
        logger.info(f"Initialized blockchain integration for {agent_name} at {self.agent_address}")
    
    async def initialize(self) -> bool:
        """Initialize the agent on blockchain"""
        try:
            # Validate configuration
            if not validate_contracts():
                logger.error("Contract configuration validation failed")
                return False
            
            # Check if already registered
            self.is_registered = await self.blockchain_client.is_agent_registered()
            
            if not self.is_registered:
                logger.info(f"Registering {self.agent_name} on blockchain...")
                capability_names = [cap.name for cap in self.capabilities]
                
                success = await self.blockchain_client.register_agent(
                    name=self.agent_name,
                    endpoint=self.agent_endpoint,
                    capabilities=capability_names
                )
                
                if success:
                    self.is_registered = True
                    logger.info(f"Successfully registered {self.agent_name} on blockchain")
                else:
                    logger.error(f"Failed to register {self.agent_name} on blockchain")
                    return False
            else:
                logger.info(f"{self.agent_name} already registered on blockchain")
            
            # Register for message events
            register_agent_for_messages(self.agent_address, self._handle_incoming_message)
            
            # Start event listening if not already started
            if not self.event_listener.is_listening:
                await self.event_listener.start_listening()
                self.is_listening = True
                logger.info("Started blockchain event listening")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain integration: {e}")
            return False
    
    async def send_message(
        self, 
        to_agent_address: str, 
        content: str, 
        message_type: str = "a2a_message"
    ) -> Optional[str]:
        """Send a message to another agent via blockchain"""
        try:
            if not self.is_registered:
                logger.error("Agent not registered on blockchain")
                return None
            
            message_id = await self.blockchain_client.send_message(
                to_address=to_agent_address,
                content=content,
                message_type=message_type
            )
            
            if message_id:
                logger.info(f"Sent blockchain message {message_id[:16]}... to {to_agent_address[:20]}...")
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send blockchain message: {e}")
            return None
    
    def register_message_handler(self, message_type: str, handler: Callable[[Dict], Any]):
        """Register a handler for specific message types"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered message handler for type: {message_type}")
    
    async def _handle_incoming_message(self, blockchain_msg: BlockchainMessage):
        """Handle incoming blockchain message"""
        try:
            logger.info(f"Received blockchain message {blockchain_msg.message_id[:16]}... from {blockchain_msg.from_address[:20]}...")
            
            # Parse message content
            try:
                content = json.loads(blockchain_msg.content) if blockchain_msg.content.startswith('{') else blockchain_msg.content
            except json.JSONDecodeError:
                content = blockchain_msg.content
            
            # Create message dict for handlers
            message_dict = {
                'id': blockchain_msg.message_id,
                'from': blockchain_msg.from_address,
                'to': blockchain_msg.to_address,
                'content': content,
                'type': blockchain_msg.message_type,
                'timestamp': blockchain_msg.timestamp,
                'block_number': blockchain_msg.block_number,
                'transaction_hash': blockchain_msg.transaction_hash
            }
            
            # Find appropriate handler
            handler = self.message_handlers.get(blockchain_msg.message_type)
            if not handler:
                # Try default handler
                handler = self.message_handlers.get('default')
            
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message_dict)
                    else:
                        handler(message_dict)
                    
                    # Mark as delivered
                    await self.event_listener.mark_message_delivered(blockchain_msg.message_id)
                    logger.info(f"Processed message {blockchain_msg.message_id[:16]}...")
                    
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            else:
                logger.warning(f"No handler for message type: {blockchain_msg.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling incoming blockchain message: {e}")
    
    async def find_agents_by_capability(self, capability: str) -> List[Dict]:
        """Find other agents with specific capability"""
        try:
            agent_addresses = await self.blockchain_client.find_agents_by_capability(capability)
            
            agents = []
            for address in agent_addresses:
                if address != self.agent_address:  # Exclude self
                    agent_info = await self.blockchain_client.get_agent_info(address)
                    if agent_info and agent_info.get('active'):
                        agents.append({
                            'address': address,
                            'name': agent_info.get('name'),
                            'endpoint': agent_info.get('endpoint'),
                            'reputation': agent_info.get('reputation'),
                            'capabilities': agent_info.get('capabilities', [])
                        })
            
            logger.info(f"Found {len(agents)} agents with capability '{capability}'")
            return agents
            
        except Exception as e:
            logger.error(f"Failed to find agents by capability: {e}")
            return []
    
    async def get_my_messages(self, undelivered_only: bool = False) -> List[Dict]:
        """Get messages received by this agent"""
        try:
            if undelivered_only:
                messages = self.event_listener.get_undelivered_messages(self.agent_address)
            else:
                messages = self.event_listener.get_received_messages(self.agent_address)
            
            return [
                {
                    'id': msg.message_id,
                    'from': msg.from_address,
                    'content': msg.content,
                    'type': msg.message_type,
                    'timestamp': msg.timestamp,
                    'delivered': msg.delivered,
                    'block_number': msg.block_number
                }
                for msg in messages
            ]
            
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'agent_name': self.agent_name,
            'agent_address': self.agent_address,
            'blockchain_connected': self.blockchain_client.web3.is_connected(),
            'registered': self.is_registered,
            'listening': self.is_listening,
            'balance': self.blockchain_client.get_balance(),
            'network': self.config.network,
            'contract_validation': validate_contracts(),
            'capabilities': [cap.name for cap in self.capabilities],
            'message_handlers': list(self.message_handlers.keys())
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Stop listening if we started it
            if self.is_listening:
                await self.event_listener.stop_listening()
            
            logger.info(f"Cleaned up blockchain integration for {self.agent_name}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function for easy agent integration
def create_blockchain_agent_integration(
    agent_name: str,
    agent_endpoint: str,
    capabilities: List[str] = None
) -> BlockchainAgentIntegration:
    """Create a blockchain integration for an agent"""
    
    capability_objects = []
    if capabilities:
        for cap in capabilities:
            capability_objects.append(AgentCapability(
                name=cap,
                description=f"Agent capability: {cap}",
                endpoint=agent_endpoint
            ))
    
    return BlockchainAgentIntegration(
        agent_name=agent_name,
        agent_endpoint=agent_endpoint,
        capabilities=capability_objects
    )

# Example usage for agents
async def initialize_agent_blockchain(agent_name: str, endpoint: str, capabilities: List[str] = None) -> BlockchainAgentIntegration:
    """Initialize blockchain integration for an agent"""
    integration = create_blockchain_agent_integration(agent_name, endpoint, capabilities)
    
    success = await integration.initialize()
    if not success:
        raise RuntimeError(f"Failed to initialize blockchain integration for {agent_name}")
    
    return integration
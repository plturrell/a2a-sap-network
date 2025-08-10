#!/usr/bin/env python3
"""
A2A Agent Blockchain Adapter
Replaces centralized registry with blockchain-based agent registration and messaging
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .web3_client import get_blockchain_client, A2ABlockchainClient

logger = logging.getLogger(__name__)

@dataclass
class A2AAgentCard:
    """Agent card compatible with blockchain registration"""
    agent_id: str
    name: str
    description: str
    version: str
    endpoint: str
    capabilities: List[str]
    skills: List[Dict[str, Any]]
    provider: Dict[str, str]

class BlockchainAgentAdapter:
    """
    Adapter that enables finsight_cib agents to use a2a_network blockchain
    Replaces centralized A2A registry with smart contract interactions
    """
    
    def __init__(self, agent_card: A2AAgentCard):
        self.agent_card = agent_card
        self.blockchain_client = get_blockchain_client()
        self.is_registered = False
        self._message_handlers = {}
        
    async def register_agent(self) -> bool:
        """Register agent on blockchain"""
        try:
            logger.info(f"Registering agent {self.agent_card.agent_id} on blockchain...")
            
            # Check if already registered
            if await self.blockchain_client.is_agent_registered():
                logger.info("Agent already registered on blockchain")
                self.is_registered = True
                return True
            
            # Check balance for gas fees
            balance = self.blockchain_client.get_balance()
            if balance < 0.01:  # Need at least 0.01 ETH for transactions
                logger.warning(f"Low balance: {balance} ETH. May not have enough gas for transactions.")
            
            # Register on blockchain
            success = await self.blockchain_client.register_agent(
                name=self.agent_card.name,
                endpoint=self.agent_card.endpoint,
                capabilities=self.agent_card.capabilities
            )
            
            if success:
                self.is_registered = True
                logger.info(f"✅ Agent {self.agent_card.agent_id} registered on blockchain")
                
                # Log agent details
                agent_info = await self.blockchain_client.get_agent_info(
                    self.blockchain_client.agent_identity.address
                )
                if agent_info:
                    logger.info(f"Agent reputation: {agent_info['reputation']}")
                    logger.info(f"Registration timestamp: {agent_info['registered_at']}")
                
                return True
            else:
                logger.error(f"Failed to register agent {self.agent_card.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Agent registration error: {e}")
            return False
    
    async def deregister_agent(self) -> bool:
        """Deactivate agent on blockchain"""
        try:
            logger.info(f"Deactivating agent {self.agent_card.agent_id} on blockchain...")
            # Note: AgentRegistry.sol has deactivateAgent() function
            # Implementation would call that function
            self.is_registered = False
            return True
            
        except Exception as e:
            logger.error(f"Agent deregistration error: {e}")
            return False
    
    async def discover_agents(self, capability: str) -> List[Dict[str, Any]]:
        """Discover agents with specific capability from blockchain"""
        try:
            agent_addresses = await self.blockchain_client.find_agents_by_capability(capability)
            
            discovered_agents = []
            for address in agent_addresses:
                agent_info = await self.blockchain_client.get_agent_info(address)
                if agent_info and agent_info['active']:
                    discovered_agents.append({
                        "address": address,
                        "name": agent_info['name'],
                        "endpoint": agent_info['endpoint'],
                        "reputation": agent_info['reputation'],
                        "capabilities": agent_info['capabilities']
                    })
            
            logger.info(f"Discovered {len(discovered_agents)} agents with capability '{capability}'")
            return discovered_agents
            
        except Exception as e:
            logger.error(f"Agent discovery error: {e}")
            return []
    
    async def send_message_to_agent(
        self,
        recipient_address: str,
        message_content: Dict[str, Any],
        message_type: str = "a2a_processing_request"
    ) -> Optional[str]:
        """Send message to another agent via blockchain"""
        try:
            # Serialize message content
            import json
            content_json = json.dumps(message_content)
            
            message_id = await self.blockchain_client.send_message(
                to_address=recipient_address,
                content=content_json,
                message_type=message_type
            )
            
            if message_id:
                logger.info(f"Message sent to {recipient_address}: {message_id}")
            
            return message_id
            
        except Exception as e:
            logger.error(f"Message sending error: {e}")
            return None
    
    async def get_agent_messages(self) -> List[Dict[str, Any]]:
        """Get messages for this agent from blockchain"""
        try:
            # This would interact with MessageRouter.sol to get messages
            # For now, return empty list as placeholder
            messages = []
            return messages
            
        except Exception as e:
            logger.error(f"Message retrieval error: {e}")
            return []
    
    def register_message_handler(self, message_type: str, handler_func):
        """Register handler for specific message types"""
        self._message_handlers[message_type] = handler_func
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def process_incoming_messages(self):
        """Process incoming messages from blockchain"""
        try:
            messages = await self.get_agent_messages()
            
            for message in messages:
                message_type = message.get('type', 'unknown')
                if message_type in self._message_handlers:
                    try:
                        await self._message_handlers[message_type](message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
                else:
                    logger.warning(f"No handler for message type: {message_type}")
                    
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def get_agent_address(self) -> str:
        """Get blockchain address of this agent"""
        return self.blockchain_client.agent_identity.address
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_card.agent_id,
            "blockchain_address": self.get_agent_address(),
            "is_registered": self.is_registered,
            "balance": self.blockchain_client.get_balance(),
            "endpoint": self.agent_card.endpoint,
            "capabilities": self.agent_card.capabilities
        }


def create_blockchain_adapter(
    agent_id: str,
    name: str,
    description: str,
    version: str,
    endpoint: str,
    capabilities: List[str],
    skills: List[Dict[str, Any]] = None,
    provider: Dict[str, str] = None
) -> BlockchainAgentAdapter:
    """Factory function to create blockchain adapter for agent"""
    
    agent_card = A2AAgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        version=version,
        endpoint=endpoint,
        capabilities=capabilities,
        skills=skills or [],
        provider=provider or {"organization": "FinSight CIB", "url": "https://finsight-cib.com"}
    )
    
    return BlockchainAgentAdapter(agent_card)
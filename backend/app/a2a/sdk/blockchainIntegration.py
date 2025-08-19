"""
Blockchain Integration Module for A2A Agents
Provides blockchain capabilities to all agents in the network
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio


try:
    from blockchain.web3Client import A2ABlockchainClient, AgentIdentity
    from blockchain.agentIntegration import BlockchainAgentIntegration, AgentCapability
    from blockchain.eventListener import MessageEventListener
    from config.contractConfig import ContractConfigManager
except ImportError as e:
    logging.warning(f"Blockchain modules not available: {e}")
    # Define placeholder classes for when blockchain is not available
    class A2ABlockchainClient:
        pass
    class AgentIdentity:
        pass
    class BlockchainAgentIntegration:
        pass
    class AgentCapability:
        pass
    class MessageEventListener:
        pass
    class ContractConfigManager:
        pass


class BlockchainIntegrationMixin:
    """
    Mixin class that provides blockchain integration capabilities to any agent.
    Add this to your agent class to enable blockchain features.
    """
    
    def __init__(self):
        self.blockchain_client = None
        self.blockchain_integration = None
        self.agent_identity = None
        self.message_listener = None
        self.blockchain_enabled = os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true"
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _initialize_blockchain(self, agent_name: str, capabilities: List[str], endpoint: str = None):
        """Initialize blockchain integration for the agent"""
        if not self.blockchain_enabled:
            self.logger.info("Blockchain integration disabled")
            return
            
        try:
            # Initialize blockchain client
            rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
            self.blockchain_client = A2ABlockchainClient(rpc_url)
            
            # Load contract configuration
            config_manager = ContractConfigManager()
            contract_config = config_manager.load_config()
            
            # Create agent identity
            private_key = os.getenv(f"{agent_name.upper()}_PRIVATE_KEY") or os.getenv("A2A_PRIVATE_KEY")
            if not private_key:
                self.logger.error("No private key configured for blockchain")
                return
                
            self.agent_identity = AgentIdentity(
                private_key=private_key,
                name=agent_name,
                endpoint=endpoint or f"http://localhost:8000"
            )
            
            # Initialize blockchain integration
            self.blockchain_integration = BlockchainAgentIntegration(
                blockchain_client=self.blockchain_client,
                agent_identity=self.agent_identity,
                registry_address=contract_config['contracts']['AgentRegistry']['address'],
                router_address=contract_config['contracts']['MessageRouter']['address']
            )
            
            # Convert capabilities to blockchain format
            blockchain_capabilities = [
                AgentCapability(name=cap, version="1.0.0")
                for cap in capabilities
            ]
            
            # Register agent on blockchain if not already registered
            if not self.blockchain_integration.is_registered():
                self.logger.info(f"Registering {agent_name} on blockchain...")
                tx_hash = self.blockchain_integration.register_agent(blockchain_capabilities)
                self.logger.info(f"Registration transaction: {tx_hash}")
            else:
                self.logger.info(f"{agent_name} already registered on blockchain")
                
            # Start message listener
            self._start_message_listener()
            
            self.logger.info(f"Blockchain integration initialized for {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain: {e}")
            self.blockchain_enabled = False
            
    def _start_message_listener(self):
        """Start listening for blockchain messages"""
        if not self.blockchain_integration:
            return
            
        try:
            self.message_listener = MessageEventListener(
                blockchain_client=self.blockchain_client,
                router_address=self.blockchain_integration.router_address,
                agent_address=self.agent_identity.address
            )
            
            # Set up message handlers
            self.message_listener.on_message_received = self._handle_blockchain_message
            
            # Start listener in background
            asyncio.create_task(self._run_message_listener())
            
        except Exception as e:
            self.logger.error(f"Failed to start message listener: {e}")
            
    async def _run_message_listener(self):
        """Run the message listener"""
        try:
            await self.message_listener.start()
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
            
    def _handle_blockchain_message(self, message: Dict[str, Any]):
        """Handle incoming blockchain messages"""
        self.logger.info(f"Received blockchain message: {message}")
        
        # Override this method in your agent to handle specific message types
        message_type = message.get('messageType', '')
        content = message.get('content', {})
        
        # Default implementation - log and acknowledge
        self.logger.info(f"Message type: {message_type}, Content: {content}")
        
        # Mark message as delivered
        if self.blockchain_integration and message.get('messageId'):
            try:
                self.blockchain_integration.mark_message_delivered(message['messageId'])
            except Exception as e:
                self.logger.error(f"Failed to mark message as delivered: {e}")
                
    def send_blockchain_message(self, to_address: str, content: Dict[str, Any], 
                               message_type: str = "GENERAL") -> Optional[str]:
        """Send a message via blockchain"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            self.logger.warning("Blockchain not enabled or initialized")
            return None
            
        try:
            # Convert content to JSON string
            content_str = json.dumps(content)
            
            # Send message
            tx_hash, message_id = self.blockchain_integration.send_message(
                to_address=to_address,
                content=content_str,
                message_type=message_type
            )
            
            self.logger.info(f"Sent blockchain message: {message_id} (tx: {tx_hash})")
            return message_id
            
        except Exception as e:
            self.logger.error(f"Failed to send blockchain message: {e}")
            return None
            
    def update_reputation(self, amount: int, reason: str = ""):
        """Update agent's reputation on blockchain"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            return
            
        try:
            # This would require admin privileges or a reputation oracle
            self.logger.info(f"Reputation update requested: {amount} ({reason})")
            # Implementation depends on contract design
            
        except Exception as e:
            self.logger.error(f"Failed to update reputation: {e}")
            
    def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find agents with specific capability via blockchain"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            return []
            
        try:
            agents = self.blockchain_integration.find_agents_by_capability(capability)
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to find agents by capability: {e}")
            return []
            
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain integration statistics"""
        if not self.blockchain_enabled:
            return {"enabled": False}
            
        stats = {
            "enabled": True,
            "address": self.agent_identity.address if self.agent_identity else None,
            "registered": False,
            "message_count": 0,
            "reputation": 0
        }
        
        if self.blockchain_integration:
            try:
                agent_info = self.blockchain_integration.get_agent_info()
                stats.update({
                    "registered": True,
                    "reputation": agent_info.get('reputation', 0),
                    "active": agent_info.get('active', False)
                })
            except:
                pass
                
        return stats
        
    def verify_trust(self, agent_address: str, min_reputation: int = 50) -> bool:
        """Verify if an agent meets trust requirements"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            return True  # Default to trust if blockchain not enabled
            
        try:
            agent_info = self.blockchain_integration.get_agent_info(agent_address)
            reputation = agent_info.get('reputation', 0)
            active = agent_info.get('active', False)
            
            return active and reputation >= min_reputation
            
        except Exception as e:
            self.logger.error(f"Failed to verify trust: {e}")
            return False


class BlockchainEnabledAgent:
    """
    Example of how to use the BlockchainIntegrationMixin in an agent
    """
    
    def __init__(self, agent_name: str, capabilities: List[str]):
        # Initialize blockchain mixin
        BlockchainIntegrationMixin.__init__(self)
        
        # Initialize blockchain for this agent
        self._initialize_blockchain(
            agent_name=agent_name,
            capabilities=capabilities,
            endpoint=f"http://localhost:8000"
        )
        
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Example request handler with blockchain integration"""
        
        # Check if request came from blockchain
        if request.get('source') == 'blockchain':
            # Verify trust of sender
            sender = request.get('from_address')
            if sender and not self.verify_trust(sender):
                return {"error": "Sender does not meet trust requirements"}
                
        # Process request
        result = self._process_request(request)
        
        # If blockchain enabled, send result via blockchain
        if self.blockchain_enabled and request.get('reply_to'):
            self.send_blockchain_message(
                to_address=request['reply_to'],
                content=result,
                message_type="RESPONSE"
            )
            
        return result
        
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Override this in your agent"""
        return {"status": "processed"}


# Decorator for adding blockchain capabilities to existing agents
def blockchain_enabled(agent_name: str, capabilities: List[str]):
    """
    Decorator to add blockchain capabilities to an agent class
    
    Usage:
        @blockchain_enabled("MyAgent", ["capability1", "capability2"])
        class MyAgent:
            def __init__(self):
                ...
    """
    def decorator(cls):
        # Save original __init__
        original_init = cls.__init__
        
        # Create new __init__ that adds blockchain
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Add blockchain mixin methods
            BlockchainIntegrationMixin.__init__(self)
            
            # Initialize blockchain
            self._initialize_blockchain(
                agent_name=agent_name,
                capabilities=capabilities,
                endpoint=getattr(self, 'endpoint', f"http://localhost:8000")
            )
            
        # Add mixin methods to class
        for attr_name in dir(BlockchainIntegrationMixin):
            if not attr_name.startswith('_') or attr_name in ['_initialize_blockchain', 
                                                               '_start_message_listener',
                                                               '_run_message_listener', 
                                                               '_handle_blockchain_message']:
                attr_value = getattr(BlockchainIntegrationMixin, attr_name)
                if callable(attr_value) and not isinstance(attr_value, type):
                    setattr(cls, attr_name, attr_value)
                    
        # Replace __init__
        cls.__init__ = new_init
        
        return cls
        
    return decorator
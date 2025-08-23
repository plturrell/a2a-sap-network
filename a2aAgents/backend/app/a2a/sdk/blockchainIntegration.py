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


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

# A2A Protocol Compliance: Direct imports only - no fallback implementations allowed
from app.a2a.sdk.blockchain.web3Client import A2ABlockchainClient, AgentIdentity
from app.a2a.sdk.blockchain.agentIntegration import BlockchainAgentIntegration, AgentCapability
from app.a2a.sdk.blockchain.eventListener import MessageEventListener
from app.a2a.config.contractConfig import ContractConfigManager
from .blockchain_error_handler import BlockchainErrorHandler, BlockchainFallbackHandler


class BlockchainIntegrationMixin:
    """
    Mixin class that provides blockchain integration capabilities to any agent.
    Add this to your agent class to enable blockchain features.
    Enhanced with comprehensive error handling and fallback mechanisms.
    """
    
    def __init__(self):
        self.blockchain_client = None
        self.blockchain_integration = None
        self.agent_identity = None
        self.message_listener = None
        self.blockchain_enabled = os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize error handlers
        self.blockchain_error_handler = BlockchainErrorHandler(
            agent_name=getattr(self, 'agent_id', 'unknown')
        )
        self.blockchain_fallback_handler = BlockchainFallbackHandler(
            agent_name=getattr(self, 'agent_id', 'unknown')
        )
        
    async def _initialize_blockchain(self, agent_name: str, capabilities: List[str], endpoint: str = None):
        """Initialize blockchain integration for the agent with comprehensive error handling"""
        if not self.blockchain_enabled:
            self.logger.info("Blockchain integration disabled")
            return False
            
        return await self.blockchain_error_handler.execute_with_retry(
            self._perform_blockchain_initialization,
            agent_name, capabilities, endpoint
        )
    
    async def _perform_blockchain_initialization(self, agent_name: str, capabilities: List[str], endpoint: str = None):
        """Perform the actual blockchain initialization"""
        try:
            # Initialize blockchain client
            rpc_url = os.getenv("A2A_RPC_URL", os.getenv("A2A_BLOCKCHAIN_RPC", "http://localhost:8545"))
            self.blockchain_client = A2ABlockchainClient(rpc_url)
            
            # Load contract configuration
            config_manager = ContractConfigManager()
            contract_config = config_manager.load_config()
            
            # Create agent identity
            private_key = os.getenv(f"{agent_name.upper()}_PRIVATE_KEY") or os.getenv("A2A_PRIVATE_KEY")
            if not private_key:
                raise ValueError("No private key configured for blockchain - set A2A_PRIVATE_KEY")
                
            # Ensure endpoint is properly set
            agent_endpoint = endpoint or os.getenv("A2A_AGENT_URL") or os.getenv("A2A_SERVICE_URL")
            if not agent_endpoint:
                raise ValueError(f"No endpoint configured for {agent_name}. Set A2A_AGENT_URL or A2A_SERVICE_URL")
                
            # Create account from private key
            from eth_account import Account
            account = Account.from_key(private_key)
            
            self.agent_identity = AgentIdentity(
                address=account.address,
                private_key=private_key,
                account=account,
                name=agent_name,
                endpoint=agent_endpoint
            )
            
            # Initialize blockchain integration
            self.blockchain_integration = BlockchainAgentIntegration(
                agent_name=agent_name,
                agent_endpoint=agent_endpoint,
                capabilities=capabilities
            )
            
            # Register agent on blockchain if not already registered
            if not self.blockchain_integration.is_registered:
                self.logger.info(f"Registering {agent_name} on blockchain...")
                self.logger.info(f"  Name: {agent_name}")
                self.logger.info(f"  Endpoint: {agent_endpoint}")
                self.logger.info(f"  Capabilities: {capabilities}")
                
                # The blockchain client will handle bytes32 conversion internally
                success = await self.blockchain_client.register_agent(
                    name=agent_name,
                    endpoint=agent_endpoint,
                    capabilities=capabilities
                )
                
                if success:
                    self.logger.info(f"✅ Successfully registered {agent_name} on blockchain")
                else:
                    raise RuntimeError(f"Blockchain registration failed for {agent_name}")
            else:
                self.logger.info(f"{agent_name} already registered on blockchain")
                
            # Start message listener
            self._start_message_listener()
            
            self.logger.info(f"✅ Blockchain integration initialized for {agent_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize blockchain: {e}")
            # Don't disable blockchain entirely - let error handler decide
            raise
            
    def _start_message_listener(self):
        """Start listening for blockchain messages"""
        if not self.blockchain_integration:
            return
            
        try:
            self.message_listener = MessageEventListener(
                blockchain_client=self.blockchain_client
            )
            
            # Set up message handlers
            self.message_listener.register_agent_handler(
                self.agent_identity.address, 
                self._handle_blockchain_message
            )
            
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
                
    async def send_blockchain_message(self, to_address: str, content: Dict[str, Any], 
                               message_type: str = "GENERAL") -> Optional[str]:
        """Send a message via blockchain with comprehensive error handling"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            self.logger.warning("Blockchain not enabled or initialized")
            # Try fallback mechanism
            return await self.blockchain_fallback_handler.handle_offline_message(
                to_address, content, message_type
            )
            
        return await self.blockchain_error_handler.execute_with_retry(
            self._perform_blockchain_message_send,
            to_address, content, message_type
        )
    
    async def _perform_blockchain_message_send(self, to_address: str, content: Dict[str, Any], 
                                             message_type: str = "GENERAL") -> Optional[str]:
        """Perform the actual blockchain message sending"""
        # Convert content to JSON string
        content_str = json.dumps(content)
        
        # Send message
        tx_hash, message_id = self.blockchain_integration.send_message(
            to_address=to_address,
            content=content_str,
            message_type=message_type
        )
        
        self.logger.info(f"✅ Sent blockchain message: {message_id} (tx: {tx_hash})")
        return message_id
            
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
            
    async def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find agents with specific capability via blockchain with error handling"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            # Try fallback lookup
            return await self.blockchain_fallback_handler.get_cached_agents_by_capability(capability)
            
        return await self.blockchain_error_handler.execute_with_retry(
            self._perform_agent_lookup,
            capability
        )
    
    async def _perform_agent_lookup(self, capability: str) -> List[Dict[str, Any]]:
        """Perform the actual agent lookup by capability"""
        agents = self.blockchain_integration.find_agents_by_capability(capability)
        self.logger.info(f"✅ Found {len(agents)} agents with capability: {capability}")
        return agents
            
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
    
    # Enhanced error handling and monitoring methods
    
    async def check_blockchain_health(self) -> Dict[str, Any]:
        """Comprehensive blockchain health check"""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "blockchain_enabled": self.blockchain_enabled,
            "client_connected": False,
            "agent_registered": False,
            "listener_active": False,
            "last_block": None,
            "network_latency": None,
            "error_count": getattr(self.blockchain_error_handler, 'total_errors', 0),
            "retry_count": getattr(self.blockchain_error_handler, 'total_retries', 0),
            "fallback_used": getattr(self.blockchain_fallback_handler, 'fallback_used_count', 0)
        }
        
        if not self.blockchain_enabled:
            health_status["status"] = "disabled"
            return health_status
        
        try:
            # Check client connection
            if self.blockchain_client:
                start_time = asyncio.get_event_loop().time()
                latest_block = await self.blockchain_client.get_latest_block()
                network_latency = asyncio.get_event_loop().time() - start_time
                
                health_status.update({
                    "client_connected": True,
                    "last_block": latest_block,
                    "network_latency": round(network_latency * 1000, 2)  # ms
                })
            
            # Check registration status
            if self.blockchain_integration:
                health_status["agent_registered"] = self.blockchain_integration.is_registered()
            
            # Check message listener
            if self.message_listener:
                health_status["listener_active"] = True
            
            # Determine overall status
            if (health_status["client_connected"] and 
                health_status["agent_registered"] and 
                health_status["listener_active"]):
                health_status["status"] = "healthy"
            else:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "error": str(e)
            })
            self.logger.error(f"Blockchain health check failed: {e}")
        
        return health_status
    
    async def recover_blockchain_connection(self) -> bool:
        """Attempt to recover blockchain connection after failures"""
        self.logger.info("🔄 Attempting blockchain connection recovery...")
        
        try:
            # Reset error counters
            if hasattr(self.blockchain_error_handler, 'reset_counters'):
                self.blockchain_error_handler.reset_counters()
            
            # Re-initialize blockchain components
            agent_name = getattr(self, 'agent_id', 'unknown')
            capabilities = getattr(self, 'blockchain_capabilities', [])
            endpoint = getattr(self, 'base_url', None)
            
            result = await self._initialize_blockchain(agent_name, capabilities, endpoint)
            
            if result:
                self.logger.info("✅ Blockchain connection recovery successful")
                self.blockchain_enabled = True
                return True
            else:
                self.logger.error("❌ Blockchain connection recovery failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Blockchain recovery failed: {e}")
            return False
    
    async def sync_fallback_messages(self) -> int:
        """Sync any cached fallback messages to blockchain when connection is restored"""
        if not self.blockchain_enabled or not self.blockchain_integration:
            return 0
        
        try:
            synced_count = await self.blockchain_fallback_handler.sync_pending_messages(
                self.blockchain_integration
            )
            if synced_count > 0:
                self.logger.info(f"✅ Synced {synced_count} fallback messages to blockchain")
            return synced_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to sync fallback messages: {e}")
            return 0
    
    def get_blockchain_error_metrics(self) -> Dict[str, Any]:
        """Get comprehensive blockchain error metrics"""
        return {
            "error_handler_metrics": self.blockchain_error_handler.get_metrics(),
            "fallback_handler_metrics": self.blockchain_fallback_handler.get_metrics(),
            "blockchain_enabled": self.blockchain_enabled,
            "health_status": "unknown"  # Will be filled by health check
        }
    
    async def shutdown_blockchain_integration(self):
        """Gracefully shutdown blockchain integration"""
        self.logger.info("🔄 Shutting down blockchain integration...")
        
        try:
            # Stop message listener
            if self.message_listener:
                await self.message_listener.stop()
            
            # Sync any pending fallback messages
            await self.sync_fallback_messages()
            
            # Close blockchain client connections
            if self.blockchain_client:
                await self.blockchain_client.close()
            
            # Cleanup error handlers
            if hasattr(self.blockchain_error_handler, 'cleanup'):
                await self.blockchain_error_handler.cleanup()
            
            if hasattr(self.blockchain_fallback_handler, 'cleanup'):
                await self.blockchain_fallback_handler.cleanup()
            
            self.logger.info("✅ Blockchain integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during blockchain shutdown: {e}")


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
            endpoint=os.getenv("A2A_SERVICE_URL")
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
                endpoint=getattr(self, 'endpoint', os.getenv("A2A_SERVICE_URL"))
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
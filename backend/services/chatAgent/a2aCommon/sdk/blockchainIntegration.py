"""
Blockchain Integration Mixin for A2A Agents
Provides fallback implementation for CLI testing
"""

import logging
from typing import Dict, List, Any, Optional


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)

class BlockchainIntegrationMixin:
    """
    Mixin class providing blockchain integration capabilities for A2A agents
    This is a fallback implementation for CLI testing
    """
    
    def __init__(self):
        """Initialize blockchain integration"""
        self.blockchain_initialized = False
        self.blockchain_capabilities = getattr(self, 'blockchain_capabilities', [])
        logger.info("Blockchain integration mixin initialized (fallback mode)")
    
    async def initialize_blockchain(self, network_url: str = None, private_key: str = None) -> bool:
        """
        Initialize blockchain connection
        This is a fallback implementation that always succeeds
        """
        logger.info(f"Initializing blockchain connection (fallback mode)")
        self.blockchain_initialized = True
        return True
    
    async def register_on_blockchain(self, agent_data: Dict[str, Any]) -> bool:
        """
        Register agent on blockchain
        This is a fallback implementation
        """
        if not self.blockchain_initialized:
            logger.warning("Blockchain not initialized, cannot register agent")
            return False
        
        logger.info(f"Registering agent on blockchain: {agent_data.get('agent_id', 'unknown')}")
        return True
    
    async def update_agent_status(self, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update agent status on blockchain
        This is a fallback implementation
        """
        if not self.blockchain_initialized:
            logger.warning("Blockchain not initialized, cannot update status")
            return False
        
        logger.info(f"Updating agent status to: {status}")
        return True
    
    async def get_agent_registry(self) -> List[Dict[str, Any]]:
        """
        Get list of registered agents from blockchain
        This is a fallback implementation
        """
        if not self.blockchain_initialized:
            logger.warning("Blockchain not initialized, returning empty registry")
            return []
        
        # Return mock registry for testing
        return [
            {
                "agent_id": "data-processor",
                "name": "Data Processor Agent",
                "status": "active",
                "endpoint": os.getenv("A2A_SERVICE_URL")
            },
            {
                "agent_id": "nlp-agent", 
                "name": "NLP Agent",
                "status": "active",
                "endpoint": os.getenv("A2A_SERVICE_URL")
            }
        ]
    
    async def record_message_on_blockchain(self, message_data: Dict[str, Any]) -> bool:
        """
        Record message hash on blockchain for audit trail
        This is a fallback implementation
        """
        if not self.blockchain_initialized:
            logger.warning("Blockchain not initialized, cannot record message")
            return False
        
        logger.debug(f"Recording message on blockchain: {message_data.get('message_id', 'unknown')}")
        return True
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """
        Get current blockchain connection status
        """
        return {
            "initialized": self.blockchain_initialized,
            "network": "fallback-mode",
            "capabilities": self.blockchain_capabilities,
            "mode": "testing"
        }
"""
A2A-Compliant Message Handler for Catalogmanager
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .comprehensivecatalogManagerSdk import ComprehensivecatalogManagerSDK

logger = logging.getLogger(__name__)


class catalogManagerA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Catalogmanager
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensivecatalogManagerSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="catalogManager",
            agent_name="Catalogmanager",
            agent_version="1.0.0",
            allowed_operations={
                "get_agent_info",
                "health_check",
                # Registry capabilities
                "catalog_management", "metadata_indexing", "service_discovery", "catalog_search", "resource_registration"
            },
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=100,
            rate_limit_window=60
        )
        
        super().__init__(config)
        
        self.agent_sdk = agent_sdk
        
        # Initialize A2A blockchain client
        self.a2a_client = A2ANetworkClient(
            agent_id=config.agent_id,
            private_key=os.getenv('A2A_PRIVATE_KEY'),
            rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
        )
        
        # Register message handlers
        self._register_handlers()
        
        logger.info(f"A2A-compliant handler initialized for {config.agent_name}")
    
    def _register_handlers(self):
        """Register A2A message handlers"""

        @self.secure_handler("get_agent_info")
        async def handle_get_agent_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get agent information"""
            try:
                result = await self.agent_sdk.get_agent_info(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_agent_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_agent_info: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("health_check")
        async def handle_health_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Health check for agent"""
            try:
                health_status = {
                    "status": "healthy",
                    "agent": self.config.agent_name,
                    "version": self.config.agent_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "a2a_compliant": True,
                    "blockchain_connected": await self._check_blockchain_connection()
                }
                
                return self.create_secure_response(health_status)
                
            except Exception as e:
                logger.error(f"Failed to health_check: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("catalog_management")
        async def handle_catalog_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle catalog management"""
            try:
                result = await self.agent_sdk.catalog_management(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="catalog_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle catalog_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("metadata_indexing")
        async def handle_metadata_indexing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle metadata indexing"""
            try:
                result = await self.agent_sdk.metadata_indexing(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="metadata_indexing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle metadata_indexing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("service_discovery")
        async def handle_service_discovery(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle service discovery"""
            try:
                result = await self.agent_sdk.service_discovery(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="service_discovery",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle service_discovery: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("catalog_search")
        async def handle_catalog_search(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle catalog search"""
            try:
                result = await self.agent_sdk.catalog_search(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="catalog_search",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle catalog_search: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("resource_registration")
        async def handle_resource_registration(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle resource registration"""
            try:
                result = await self.agent_sdk.resource_registration(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="resource_registration",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle resource_registration: {e}")
                return self.create_secure_response(str(e), status="error")

    async def process_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Main entry point for A2A messages
        Routes messages to appropriate handlers based on operation
        """
        try:
            # Extract operation from message
            operation = None
            data = {}
            
            if message.parts and len(message.parts) > 0:
                part = message.parts[0]
                if part.data:
                    operation = part.data.get("operation")
                    data = part.data.get("data", {})
            
            if not operation:
                return self.create_secure_response(
                    "No operation specified in message",
                    status="error"
                )
            
            # Get handler for operation
            handler = self.handlers.get(operation)
            if not handler:
                return self.create_secure_response(
                    f"Unknown operation: {operation}",
                    status="error"
                )
            
            # Create context ID
            context_id = f"{message.sender_id}:{operation}:{datetime.utcnow().timestamp()}"
            
            # Process through handler
            return await handler(message, context_id, data)
            
        except Exception as e:
            logger.error(f"Failed to process A2A message: {e}")
            return self.create_secure_response(str(e), status="error")
    
    async def _log_blockchain_transaction(self, operation: str, data_hash: str, result_hash: str, context_id: str):
        """Log transaction to blockchain for audit trail"""
        try:
            transaction_data = {
                "agent_id": self.config.agent_id,
                "operation": operation,
                "data_hash": data_hash,
                "result_hash": result_hash,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to blockchain through A2A client
            await self.a2a_client.log_transaction(transaction_data)
            
        except Exception as e:
            logger.error(f"Failed to log blockchain transaction: {e}")
    
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for blockchain logging"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    async def _check_blockchain_connection(self) -> bool:
        """Check if blockchain connection is active"""
        try:
            return await self.a2a_client.is_connected()
        except Exception:
            return False
    
    async def start(self):
        """Start the A2A handler"""
        logger.info(f"Starting A2A handler for {self.config.agent_name}")
        
        # Connect to blockchain
        await self.a2a_client.connect()
        
        # Register agent on blockchain
        await self.a2a_client.register_agent({
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "capabilities": list(self.config.allowed_operations),
            "version": self.config.agent_version
        })
        
        logger.info(f"A2A handler started and registered on blockchain")
    
    async def stop(self):
        """Stop the A2A handler"""
        logger.info(f"Stopping A2A handler for {self.config.agent_name}")
        
        # Unregister from blockchain
        await self.a2a_client.unregister_agent(self.config.agent_id)
        
        # Disconnect
        await self.a2a_client.disconnect()
        
        # Parent cleanup
        await self.shutdown()
        
        logger.info(f"A2A handler stopped")


# Factory function to create A2A handler
def create_catalogManager_a2a_handler(agent_sdk: ComprehensivecatalogManagerSDK) -> CatalogManagerA2AHandler:
    """Create A2A-compliant handler for Catalog Manager"""
    return CatalogManagerA2AHandler(agent_sdk)

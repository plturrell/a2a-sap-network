"""
A2A-Compliant Message Handler for Agent 0 - Data Product Registration
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
from .dataProductAgentSdk import DataProductRegistrationAgentSDK

logger = logging.getLogger(__name__)


class Agent0DataproductA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 0 - Data Product Registration
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: DataProductRegistrationAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="agent0DataProduct",
            agent_name="Agent 0 - Data Product Registration",
            agent_version="1.0.0",
            allowed_operations={
                "get_agent_card",
                "json_rpc",
                "process_message",
                "get_task_status",
                "get_queue_status",
                "get_message_status",
                "cancel_message",
                "health_check",
                # Registry capabilities
                "data_product_creation",
                "data_ingestion",
                "data_transformation",
                "quality_control",
                "metadata_management"
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

        @self.secure_handler("get_agent_card")
        async def handle_get_agent_card(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get agent card information"""
            try:
                agent_card = await self.agent_sdk.get_agent_card()
                result = agent_card
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_agent_card",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_agent_card: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("json_rpc")
        async def handle_json_rpc(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle json_rpc operation"""
            try:
                # Implement json_rpc logic using agent SDK
                result = await self.agent_sdk.json_rpc_handler(data)
                if not result:
                    result = {"status": "success", "operation": "json_rpc", "message": "No specific handler found"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="json_rpc",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to json_rpc: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("process_message")
        async def handle_process_message(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Process incoming message"""
            try:
                # Process message through agent SDK
                result = await self.agent_sdk.process_message(message, context_id)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="process_message",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to process_message: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_task_status")
        async def handle_get_task_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get status of a specific task"""
            try:
                task_id = data.get("task_id")
                if not task_id:
                    raise ValueError("task_id is required")
                
                status = await self.agent_sdk.get_task_status(task_id)
                result = status
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_task_status",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_task_status: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_queue_status")
        async def handle_get_queue_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_queue_status operation"""
            try:
                # Implement get_queue_status logic using agent SDK
                result = await self.agent_sdk.get_queue_status(data)
                if not result:
                    result = {"status": "success", "operation": "get_queue_status", "queue_size": 0, "pending_messages": 0}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_queue_status",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_queue_status: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_message_status")
        async def handle_get_message_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_message_status operation"""
            try:
                # Implement get_message_status logic using agent SDK
                result = await self.agent_sdk.get_message_status(data)
                if not result:
                    result = {"status": "success", "operation": "get_message_status", "message_status": "not_found"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_message_status",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_message_status: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("cancel_message")
        async def handle_cancel_message(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle cancel_message operation"""
            try:
                # Implement cancel_message logic using agent SDK
                result = await self.agent_sdk.cancel_message(data)
                if not result:
                    result = {"status": "success", "operation": "cancel_message", "cancelled": False, "reason": "message_not_found"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cancel_message",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to cancel_message: {e}")
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
                result = health_status
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="health_check",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to health_check: {e}")
                return self.create_secure_response(str(e), status="error")

        # Registry capability handlers
        @self.secure_handler("data_product_creation")
        async def handle_data_product_creation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data product creation"""
            try:
                result = await self.agent_sdk.data_product_creation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_product_creation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle data_product_creation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_ingestion")
        async def handle_data_ingestion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data ingestion"""
            try:
                result = await self.agent_sdk.data_ingestion(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_ingestion",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle data_ingestion: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_transformation")
        async def handle_data_transformation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data transformation"""
            try:
                result = await self.agent_sdk.data_transformation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_transformation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle data_transformation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("quality_control")
        async def handle_quality_control(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle quality control"""
            try:
                result = await self.agent_sdk.quality_control(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="quality_control",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle quality_control: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("metadata_management")
        async def handle_metadata_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle metadata management"""
            try:
                result = await self.agent_sdk.metadata_management(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="metadata_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to handle metadata_management: {e}")
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
def create_agent0DataProduct_a2a_handler(agent_sdk: DataProductRegistrationAgentSDK) -> Agent0DataproductA2AHandler:
    """Create A2A-compliant handler for Agent 0 - Data Product Registration"""
    return Agent0DataproductA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_agent0DataProduct_a2a_handler(agent0DataProduct_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
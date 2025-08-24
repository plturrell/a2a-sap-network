"""
A2A-Compliant Message Handler for Agent 0
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
from .comprehensiveDataProductAgentSdk import ComprehensiveDataProductAgentSDK

logger = logging.getLogger(__name__)


class Agent0A2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 0 (Data Product Registration)
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveDataProductAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="agent0_data_product",
            agent_name="Data Product Registration Agent",
            agent_version="2.0.0",
            allowed_operations={
                "data_product_creation",
                "data_ingestion", 
                "data_transformation",
                "quality_control",
                "metadata_management",
                "register_data_product",
                "validate_data_product",
                "extract_metadata",
                "assess_quality",
                "create_lineage",
                "dublin_core_compliance",
                "data_integrity_check",
                "cross_agent_validation",
                "get_agent_card",
                "get_task_status",
                "get_queue_status",
                "cancel_message",
                "health_check"
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

        @self.secure_handler("data_product_creation")
        async def handle_data_product_creation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Create new data products with comprehensive metadata"""
            try:
                result = await self.agent_sdk.data_product_creation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_product_creation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to create data product: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_ingestion")
        async def handle_data_ingestion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data ingestion with quality validation"""
            try:
                result = await self.agent_sdk.data_ingestion(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_ingestion",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to ingest data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_transformation")
        async def handle_data_transformation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Transform data with lineage tracking"""
            try:
                result = await self.agent_sdk.data_transformation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_transformation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to transform data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("quality_control")
        async def handle_quality_control(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform quality control assessment"""
            try:
                result = await self.agent_sdk.quality_control(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="quality_control",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed quality control: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("metadata_management")
        async def handle_metadata_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Manage Dublin Core metadata"""
            try:
                result = await self.agent_sdk.metadata_management(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="metadata_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed metadata management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("extract_metadata")
        async def handle_extract_metadata(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Extract metadata using ML techniques"""
            try:
                result = await self.agent_sdk.extract_metadata(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="extract_metadata",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to extract metadata: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("assess_quality")
        async def handle_assess_quality(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Assess data quality using AI"""
            try:
                result = await self.agent_sdk.assess_quality(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="assess_quality",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to assess quality: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_lineage")
        async def handle_create_lineage(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Create data lineage graph"""
            try:
                result = await self.agent_sdk.create_lineage(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_lineage",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to create lineage: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("dublin_core_compliance")
        async def handle_dublin_core_compliance(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Verify Dublin Core compliance"""
            try:
                result = await self.agent_sdk.dublin_core_compliance(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="dublin_core_compliance",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed Dublin Core compliance check: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_integrity_check")
        async def handle_data_integrity_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Check data integrity with blockchain verification"""
            try:
                result = await self.agent_sdk.data_integrity_check(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_integrity_check",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed data integrity check: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("cross_agent_validation")
        async def handle_cross_agent_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate data across multiple agents"""
            try:
                result = await self.agent_sdk.cross_agent_validation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cross_agent_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed cross-agent validation: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("get_agent_card")
        async def handle_get_agent_card(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get agent card information"""
            try:
                agent_card = await self.agent_sdk.get_agent_card()
                return self.create_secure_response(agent_card)
            except Exception as e:
                logger.error(f"Failed to get agent card: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("register_data_product")
        async def handle_register_data_product(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Register a new data product"""
            try:
                # Validate required fields
                required_fields = ["product_name", "description", "schema"]
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Process through agent SDK
                result = await self.agent_sdk.register_data_product(
                    product_name=data["product_name"],
                    description=data["description"],
                    schema=data["schema"],
                    metadata=data.get("metadata", {})
                )
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="register_data_product",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to register data product: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("get_task_status")
        async def handle_get_task_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get status of a specific task"""
            try:
                task_id = data.get("task_id")
                if not task_id:
                    raise ValueError("task_id is required")
                
                status = await self.agent_sdk.get_task_status(task_id)
                return self.create_secure_response(status)
                
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("get_queue_status")
        async def handle_get_queue_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get message queue status"""
            try:
                if self.agent_sdk and self.agent_sdk.message_queue:
                    queue_status = self.agent_sdk.message_queue.get_queue_status()
                    return self.create_secure_response(queue_status)
                else:
                    raise ValueError("Message queue not available")
                    
            except Exception as e:
                logger.error(f"Failed to get queue status: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("cancel_message")
        async def handle_cancel_message(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Cancel a queued or processing message"""
            try:
                message_id = data.get("message_id")
                if not message_id:
                    raise ValueError("message_id is required")
                
                if self.agent_sdk and self.agent_sdk.message_queue:
                    cancelled = await self.agent_sdk.message_queue.cancel_message(message_id)
                    if cancelled:
                        return self.create_secure_response({"message": "Message cancelled successfully"})
                    else:
                        raise ValueError("Message not found or cannot be cancelled")
                else:
                    raise ValueError("Message queue not available")
                    
            except Exception as e:
                logger.error(f"Failed to cancel message: {e}")
                return self.create_secure_response(str(e), status="error")
        
        @self.secure_handler("health_check")
        async def handle_health_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Health check for agent"""
            try:
                queue_info = {}
                if self.agent_sdk and self.agent_sdk.message_queue:
                    queue_status = self.agent_sdk.message_queue.get_queue_status()
                    queue_info = {
                        "queue_depth": queue_status["queue_status"]["queue_depth"],
                        "processing_count": queue_status["queue_status"]["processing_count"],
                        "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
                        "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
                    }
                
                health_status = {
                    "status": "healthy",
                    "agent": self.config.agent_name,
                    "version": self.config.agent_version,
                    "protocol_version": "0.2.9",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_queue": queue_info,
                    "a2a_compliant": True,
                    "blockchain_connected": await self._check_blockchain_connection()
                }
                
                return self.create_secure_response(health_status)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return self.create_secure_response(
                    {"status": "unhealthy", "error": str(e)}, 
                    status="error"
                )
    
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
def create_agent0_a2a_handler(agent_sdk: DataProductRegistrationAgentSDK) -> Agent0A2AHandler:
    """Create A2A-compliant handler for Agent 0"""
    return Agent0A2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_agent0_a2a_handler(agent0_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
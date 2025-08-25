"""
A2A-Compliant Message Handler for Catalog Manager - ORD Repository Management
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .comprehensiveCatalogManagerSdk import ComprehensiveCatalogManagerSDK

logger = logging.getLogger(__name__)


class CatalogManagerA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Catalog Manager - ORD Repository Management
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveCatalogManagerSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="catalog_manager",
            agent_name="Catalog Manager - ORD Repository Management",
            agent_version="1.0.0",
            allowed_operations={
                "get_agent_card",
                # Registry capabilities
                "catalog_management",
                "metadata_indexing",
                "service_discovery",
                "catalog_search",
                "resource_registration",
                # Enhanced operations
                "json_rpc",
                "process_message",
                "get_task_status",
                "get_queue_status",
                "get_message_status",
                "cancel_message",
                "health_check",
                "register_ord_document",
                "enhance_ord_document",
                "search_ord_repository",
                "assess_ord_quality"
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
                # Process JSON-RPC request
                result = await self.agent_sdk.json_rpc_handler(data)

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
                # Get queue processing status
                result = await self.agent_sdk.get_queue_status(data)

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
                # Get message processing status
                result = await self.agent_sdk.get_message_status(data)

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
                # TODO: Implement cancel_message logic
                # Example: result = await self.agent_sdk.cancel_message(data)
                result = {"status": "success", "operation": "cancel_message"}

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

        @self.secure_handler("register_ord_document")
        async def handle_register_ord_document(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle register_ord_document operation"""
            try:
                # Register ORD document in catalog
                result = await self.agent_sdk.register_ord_document(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="register_ord_document",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to register_ord_document: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("enhance_ord_document")
        async def handle_enhance_ord_document(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle enhance_ord_document operation"""
            try:
                # Enhance ORD document with AI
                result = await self.agent_sdk.enhance_ord_document(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="enhance_ord_document",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to enhance_ord_document: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("search_ord_repository")
        async def handle_search_ord_repository(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle search_ord_repository operation"""
            try:
                # Search ORD repository
                result = await self.agent_sdk.search_ord_repository(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="search_ord_repository",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to search_ord_repository: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("assess_ord_quality")
        async def handle_assess_ord_quality(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle assess_ord_quality operation"""
            try:
                # Assess ORD document quality
                result = await self.agent_sdk.assess_ord_quality(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="assess_ord_quality",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to assess_ord_quality: {e}")
                return self.create_secure_response(str(e), status="error")

        # Registry capability handlers
        @self.secure_handler("catalog_management")
        async def handle_catalog_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle catalog management operations"""
            try:
                result = await self.agent_sdk.manage_catalog(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="catalog_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to catalog_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("metadata_indexing")
        async def handle_metadata_indexing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle metadata indexing operations"""
            try:
                result = await self.agent_sdk.index_metadata(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="metadata_indexing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to metadata_indexing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("service_discovery")
        async def handle_service_discovery(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle service discovery operations"""
            try:
                result = await self.agent_sdk.discover_services(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="service_discovery",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to service_discovery: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("catalog_search")
        async def handle_catalog_search(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle catalog search operations"""
            try:
                result = await self.agent_sdk.search_catalog(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="catalog_search",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to catalog_search: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("resource_registration")
        async def handle_resource_registration(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle resource registration operations"""
            try:
                result = await self.agent_sdk.register_resource(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="resource_registration",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to resource_registration: {e}")
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
def create_catalog_manager_a2a_handler(agent_sdk: ComprehensiveCatalogManagerSDK) -> CatalogManagerA2AHandler:
    """Create A2A-compliant handler for Catalog Manager - ORD Repository Management"""
    return CatalogManagerA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_catalog_manager_a2a_handler(catalog_manager_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""

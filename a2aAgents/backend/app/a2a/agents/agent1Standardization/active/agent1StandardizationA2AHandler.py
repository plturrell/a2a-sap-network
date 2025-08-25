"""
A2A-Compliant Message Handler for Agent 1 - Financial Standardization
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from ....core.a2aTypes import A2AMessage
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .enhancedDataStandardizationAgentMcp import EnhancedDataStandardizationAgentMcp

logger = logging.getLogger(__name__)


class Agent1StandardizationA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 1 - Financial Standardization
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: EnhancedDataStandardizationAgentMcp):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="data_standardization_agent",
            agent_name="Data Standardization Agent",
            agent_version="2.0.0",
            allowed_operations={
                "data_standardization",
                "schema_validation",
                "format_conversion",
                "data_normalization",
                "quality_improvement",
                "standardize_data",
                "validate_schema",
                "convert_format",
                "normalize_data",
                "improve_quality",
                "cross_domain_standardization",
                "pattern_learning_standardization",
                "get_agent_card",
                "json_rpc",
                "process_message",
                "get_task_status",
                "get_queue_status",
                "get_message_status",
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

        @self.secure_handler("data_standardization")
        async def handle_data_standardization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive data standardization"""
            try:
                result = await self.agent_sdk.data_standardization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_standardization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to data_standardization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("schema_validation")
        async def handle_schema_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle schema validation operations"""
            try:
                result = await self.agent_sdk.schema_validation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="schema_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to schema_validation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("format_conversion")
        async def handle_format_conversion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle format conversion operations"""
            try:
                result = await self.agent_sdk.format_conversion(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="format_conversion",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to format_conversion: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_normalization")
        async def handle_data_normalization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data normalization operations"""
            try:
                result = await self.agent_sdk.data_normalization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_normalization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to data_normalization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("quality_improvement")
        async def handle_quality_improvement(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle quality improvement operations"""
            try:
                result = await self.agent_sdk.quality_improvement(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="quality_improvement",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to quality_improvement: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("standardize_data")
        async def handle_standardize_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data standardization with ML enhancement"""
            try:
                result = await self.agent_sdk.standardize_data(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="standardize_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to standardize_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_schema")
        async def handle_validate_schema(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle schema validation with AI analysis"""
            try:
                result = await self.agent_sdk.validate_schema(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_schema",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_schema: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("convert_format")
        async def handle_convert_format(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle format conversion with intelligence"""
            try:
                result = await self.agent_sdk.convert_format(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="convert_format",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to convert_format: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("normalize_data")
        async def handle_normalize_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data normalization with AI optimization"""
            try:
                result = await self.agent_sdk.normalize_data(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="normalize_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to normalize_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("improve_quality")
        async def handle_improve_quality(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle quality improvement with ML recommendations"""
            try:
                result = await self.agent_sdk.improve_quality(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="improve_quality",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to improve_quality: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("cross_domain_standardization")
        async def handle_cross_domain_standardization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle cross-domain standardization operations"""
            try:
                result = await self.agent_sdk.cross_domain_standardization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cross_domain_standardization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to cross_domain_standardization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("pattern_learning_standardization")
        async def handle_pattern_learning_standardization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle pattern learning standardization operations"""
            try:
                result = await self.agent_sdk.pattern_learning_standardization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="pattern_learning_standardization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to pattern_learning_standardization: {e}")
                return self.create_secure_response(str(e), status="error")

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
                # TODO: Implement json_rpc logic
                # Example: result = await self.agent_sdk.json_rpc_handler(data)
                result = {"status": "success", "operation": "json_rpc"}

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
                # TODO: Implement get_queue_status logic
                # Example: result = await self.agent_sdk.get_queue_status(data)
                result = {"status": "success", "operation": "get_queue_status"}

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
                # TODO: Implement get_message_status logic
                # Example: result = await self.agent_sdk.get_message_status(data)
                result = {"status": "success", "operation": "get_message_status"}

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
def create_agent1Standardization_a2a_handler(agent_sdk: DataStandardizationAgentSDK) -> Agent1StandardizationA2AHandler:
    """Create A2A-compliant handler for Agent 1 - Financial Standardization"""
    return Agent1StandardizationA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent1Standardization_a2a_handler(agent1Standardization_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""

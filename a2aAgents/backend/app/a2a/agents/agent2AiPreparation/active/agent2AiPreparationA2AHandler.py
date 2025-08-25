"""
A2A-Compliant Message Handler for Agent 2 - AI Preparation
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

from ....core.a2aTypes import A2AMessage
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .aiPreparationAgentSdk import AIPreparationAgentSDK

logger = logging.getLogger(__name__)


class Agent2AipreparationA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 2 - AI Preparation
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: AIPreparationAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="ai_preparation_agent",
            agent_name="AI Preparation Agent",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "ai_data_preparation",
                "feature_engineering",
                "data_preprocessing",
                "ml_optimization",
                "embedding_preparation",
                # Enhanced operations
                "prepare_ai_data",
                "engineer_features",
                "preprocess_data",
                "optimize_ml",
                "prepare_embeddings",
                "advanced_feature_extraction",
                "neural_preprocessing",
                # Base operations
                "get_agent_card",
                "json_rpc",
                "process_message",
                "get_task_status",
                "get_queue_status",
                "get_message_status",
                "cancel_message",
                "health_check",
                "get_skills",
                "get_capabilities"
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

        @self.secure_handler("get_skills")
        async def handle_get_skills(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_skills operation"""
            try:
                # TODO: Implement get_skills logic
                # Example: result = await self.agent_sdk.get_skills(data)
                result = {"status": "success", "operation": "get_skills"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_skills",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_skills: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_capabilities")
        async def handle_get_capabilities(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_capabilities operation"""
            try:
                # TODO: Implement get_capabilities logic
                # Example: result = await self.agent_sdk.get_capabilities(data)
                result = {"status": "success", "operation": "get_capabilities"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_capabilities",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_capabilities: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("ai_data_preparation")
        async def handle_ai_data_preparation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive AI data preparation"""
            try:
                result = await self.agent_sdk.ai_data_preparation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="ai_data_preparation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to ai_data_preparation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("feature_engineering")
        async def handle_feature_engineering(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle advanced feature engineering operations"""
            try:
                result = await self.agent_sdk.feature_engineering(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="feature_engineering",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to feature_engineering: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_preprocessing")
        async def handle_data_preprocessing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive data preprocessing"""
            try:
                result = await self.agent_sdk.data_preprocessing(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_preprocessing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to data_preprocessing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("ml_optimization")
        async def handle_ml_optimization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle ML model optimization operations"""
            try:
                result = await self.agent_sdk.ml_optimization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="ml_optimization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to ml_optimization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("embedding_preparation")
        async def handle_embedding_preparation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding preparation operations"""
            try:
                result = await self.agent_sdk.embedding_preparation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="embedding_preparation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to embedding_preparation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("prepare_ai_data")
        async def handle_prepare_ai_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle AI-specific data preparation"""
            try:
                result = await self.agent_sdk.ai_data_preparation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="prepare_ai_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to prepare_ai_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("engineer_features")
        async def handle_engineer_features(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle intelligent feature engineering"""
            try:
                result = await self.agent_sdk.feature_engineering(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="engineer_features",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to engineer_features: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("preprocess_data")
        async def handle_preprocess_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle advanced data preprocessing"""
            try:
                result = await self.agent_sdk.data_preprocessing(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="preprocess_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to preprocess_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_ml")
        async def handle_optimize_ml(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle ML optimization with advanced techniques"""
            try:
                result = await self.agent_sdk.ml_optimization(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="optimize_ml",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to optimize_ml: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("prepare_embeddings")
        async def handle_prepare_embeddings(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding preparation with optimization"""
            try:
                result = await self.agent_sdk.embedding_preparation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="prepare_embeddings",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to prepare_embeddings: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("advanced_feature_extraction")
        async def handle_advanced_feature_extraction(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle advanced feature extraction operations"""
            try:
                # Use feature engineering with advanced configuration
                advanced_config = {**data, "feature_config": {"advanced_mode": True, "extraction_type": "deep"}}
                result = await self.agent_sdk.feature_engineering(advanced_config)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="advanced_feature_extraction",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to advanced_feature_extraction: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("neural_preprocessing")
        async def handle_neural_preprocessing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle neural network-based preprocessing"""
            try:
                # Use data preprocessing with neural configuration
                neural_config = {**data, "preprocessing_config": {"neural_mode": True, "deep_learning": True}}
                result = await self.agent_sdk.data_preprocessing(neural_config)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="neural_preprocessing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to neural_preprocessing: {e}")
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
def create_agent2AiPreparation_a2a_handler(agent_sdk: AIPreparationAgentSDK) -> Agent2AipreparationA2AHandler:
    """Create A2A-compliant handler for Agent 2 - AI Preparation"""
    return Agent2AipreparationA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent2AiPreparation_a2a_handler(agent2AiPreparation_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""

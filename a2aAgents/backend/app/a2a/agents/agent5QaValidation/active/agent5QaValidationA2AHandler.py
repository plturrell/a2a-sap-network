"""
A2A-Compliant Message Handler for Agent 5 - QA Validation
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

from ....core.a2aTypes import A2AMessage, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .enhancedQaValidationAgentMcp import EnhancedQAValidationAgentMCP

logger = logging.getLogger(__name__)


class Agent5QavalidationA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 5 - QA Validation
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: EnhancedQAValidationAgentMCP):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="qa_validation_agent",
            agent_name="QA Validation Agent",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "qa_validation",
                "quality_assurance",
                "test_execution",
                "validation_reporting",
                "compliance_checking",
                # Enhanced operations
                "generate_qa_tests",
                "validate_answers",
                "execute_test_suite",
                "generate_validation_report",
                "check_compliance",
                "semantic_validation",
                "batch_processing",
                "websocket_management",
                "template_management",
                # Base operations
                "health_check",
                "get_supported_formats"
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

        @self.secure_handler("get_supported_formats")
        async def handle_get_supported_formats(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_supported_formats operation"""
            try:
                result = {
                    "supported_formats": [
                        "json", "xml", "yaml", "csv", "text"
                    ],
                    "validation_methods": [
                        "exact_match", "semantic_similarity", "fuzzy_matching",
                        "knowledge_graph", "contextual_analysis", "multi_modal"
                    ],
                    "template_types": [
                        "factual", "inferential", "comparative", "analytical", "evaluative", "synthetic"
                    ]
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_supported_formats",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_supported_formats: {e}")
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
        @self.secure_handler("qa_validation")
        async def handle_qa_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive QA validation"""
            try:
                result = await self.agent_sdk.generate_sophisticated_qa_tests_mcp(
                    content_data=data.get("content_data", {}),
                    template_complexity=data.get("complexity", "intermediate"),
                    test_count=data.get("test_count", 20),
                    validation_methods=data.get("validation_methods")
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="qa_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to qa_validation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("quality_assurance")
        async def handle_quality_assurance(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle quality assurance operations"""
            try:
                result = await self.agent_sdk.validate_answers_semantically_mcp(
                    qa_pairs=data.get("qa_pairs", []),
                    validation_methods=data.get("validation_methods"),
                    confidence_threshold=data.get("confidence_threshold", 0.7)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="quality_assurance",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to quality_assurance: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("test_execution")
        async def handle_test_execution(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle test execution operations"""
            try:
                result = await self.agent_sdk.optimize_qa_batch_processing_mcp(
                    test_data=data.get("test_data", []),
                    optimization_strategy=data.get("strategy", "adaptive"),
                    max_batch_size=data.get("batch_size", 100)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="test_execution",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to test_execution: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validation_reporting")
        async def handle_validation_reporting(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validation reporting operations"""
            try:
                result = await self.agent_sdk.get_batch_processing_metrics()

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validation_reporting",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validation_reporting: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("compliance_checking")
        async def handle_compliance_checking(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle compliance checking operations"""
            try:
                result = await self.agent_sdk.get_semantic_validation_status()

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="compliance_checking",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to compliance_checking: {e}")
                return self.create_secure_response(str(e), status="error")

        # Enhanced operation handlers
        @self.secure_handler("generate_qa_tests")
        async def handle_generate_qa_tests(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle QA test generation with sophisticated templates"""
            try:
                result = await self.agent_sdk.generate_sophisticated_qa_tests_mcp(
                    content_data=data.get("content_data", {}),
                    template_complexity=data.get("complexity", "intermediate"),
                    test_count=data.get("test_count", 20),
                    batch_optimization=data.get("batch_optimization", True)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_qa_tests",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to generate_qa_tests: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_answers")
        async def handle_validate_answers(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle answer validation with semantic analysis"""
            try:
                result = await self.agent_sdk.validate_answers_semantically_mcp(
                    qa_pairs=data.get("qa_pairs", []),
                    validation_methods=data.get("validation_methods"),
                    confidence_threshold=data.get("confidence_threshold", 0.7),
                    enable_consensus=data.get("enable_consensus", True)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_answers",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_answers: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execute_test_suite")
        async def handle_execute_test_suite(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle test suite execution with optimization"""
            try:
                result = await self.agent_sdk.optimize_qa_batch_processing_mcp(
                    test_data=data.get("test_data", []),
                    optimization_strategy=data.get("strategy", "adaptive"),
                    max_batch_size=data.get("batch_size", 100),
                    enable_caching=data.get("enable_caching", True)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execute_test_suite",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to execute_test_suite: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_validation_report")
        async def handle_generate_validation_report(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validation report generation"""
            try:
                result = await self.agent_sdk.get_batch_processing_metrics()

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_validation_report",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to generate_validation_report: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("check_compliance")
        async def handle_check_compliance(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle compliance checking with advanced validation"""
            try:
                result = await self.agent_sdk.get_semantic_validation_status()

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="check_compliance",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to check_compliance: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("semantic_validation")
        async def handle_semantic_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle advanced semantic validation"""
            try:
                result = await self.agent_sdk.validate_answers_semantically_mcp(
                    qa_pairs=data.get("qa_pairs", []),
                    validation_methods=data.get("validation_methods", ["semantic_similarity", "contextual_analysis"]),
                    confidence_threshold=data.get("confidence_threshold", 0.8)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="semantic_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to semantic_validation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("batch_processing")
        async def handle_batch_processing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle optimized batch processing"""
            try:
                result = await self.agent_sdk.optimize_qa_batch_processing_mcp(
                    test_data=data.get("test_data", []),
                    optimization_strategy=data.get("strategy", "adaptive"),
                    max_batch_size=data.get("batch_size", 100)
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="batch_processing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to batch_processing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("websocket_management")
        async def handle_websocket_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle WebSocket connection management"""
            try:
                result = await self.agent_sdk.manage_websocket_connections_mcp(
                    action=data.get("action", "status"),
                    task_id=data.get("task_id"),
                    connection_config=data.get("connection_config", {})
                )

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="websocket_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to websocket_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("template_management")
        async def handle_template_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle template management operations"""
            try:
                result = await self.agent_sdk.get_template_capabilities()

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="template_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to template_management: {e}")
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
def create_agent5QaValidation_a2a_handler(agent_sdk: EnhancedQAValidationAgentMCP) -> Agent5QavalidationA2AHandler:
    """Create A2A-compliant handler for Agent 5 - QA Validation Agent"""
    return Agent5QavalidationA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent5QaValidation_a2a_handler(agent5QaValidation_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""

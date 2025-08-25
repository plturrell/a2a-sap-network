"""
A2A-Compliant Message Handler for Agent 4: Computation Quality Testing
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
from .comprehensiveCalcValidationSdk import ComprehensiveCalcValidationSDK

logger = logging.getLogger(__name__)


class Agent4CalcvalidationA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 4: Computation Quality Testing
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveCalcValidationSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="calculation_validation_agent",
            agent_name="Calculation Validation Agent",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "calculation_validation",
                "numerical_verification",
                "statistical_analysis",
                "accuracy_checking",
                "error_detection",
                # Enhanced operations
                "validate_calculation",
                "verify_numerical",
                "analyze_statistical",
                "check_accuracy",
                "detect_errors",
                "symbolic_validation",
                "monte_carlo_verification",
                "mathematical_proof",
                "precision_analysis",
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

        @self.secure_handler("get_supported_formats")
        async def handle_get_supported_formats(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_supported_formats operation"""
            try:
                # TODO: Implement get_supported_formats logic
                # Example: result = await self.agent_sdk.get_supported_formats(data)
                result = {"status": "success", "operation": "get_supported_formats"}

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

        # Registry capability handlers
        @self.secure_handler("calculation_validation")
        async def handle_calculation_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive calculation validation"""
            try:
                result = await self.agent_sdk.calculation_validation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="calculation_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to calculation_validation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("numerical_verification")
        async def handle_numerical_verification(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle numerical verification operations"""
            try:
                result = await self.agent_sdk.numerical_verification(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="numerical_verification",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to numerical_verification: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("statistical_analysis")
        async def handle_statistical_analysis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle statistical analysis operations"""
            try:
                result = await self.agent_sdk.statistical_analysis(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="statistical_analysis",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to statistical_analysis: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("accuracy_checking")
        async def handle_accuracy_checking(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle accuracy checking operations"""
            try:
                result = await self.agent_sdk.accuracy_checking(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="accuracy_checking",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to accuracy_checking: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("error_detection")
        async def handle_error_detection(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle error detection operations"""
            try:
                result = await self.agent_sdk.error_detection(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="error_detection",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to error_detection: {e}")
                return self.create_secure_response(str(e), status="error")

        # Enhanced operation handlers
        @self.secure_handler("validate_calculation")
        async def handle_validate_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle calculation validation with comprehensive methods"""
            try:
                result = await self.agent_sdk.validate_expression(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("verify_numerical")
        async def handle_verify_numerical(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle numerical verification with precision analysis"""
            try:
                result = await self.agent_sdk.numerical_verification(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="verify_numerical",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to verify_numerical: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("analyze_statistical")
        async def handle_analyze_statistical(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle statistical analysis with distribution testing"""
            try:
                result = await self.agent_sdk.statistical_analysis(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="analyze_statistical",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to analyze_statistical: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("check_accuracy")
        async def handle_check_accuracy(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle accuracy checking with error bound analysis"""
            try:
                result = await self.agent_sdk.accuracy_checking(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="check_accuracy",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to check_accuracy: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("detect_errors")
        async def handle_detect_errors(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle error detection with pattern analysis"""
            try:
                result = await self.agent_sdk.error_detection(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="detect_errors",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to detect_errors: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("symbolic_validation")
        async def handle_symbolic_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle symbolic mathematical validation"""
            try:
                result = await self.agent_sdk.symbolic_validation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="symbolic_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to symbolic_validation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("monte_carlo_verification")
        async def handle_monte_carlo_verification(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle Monte Carlo statistical verification"""
            try:
                result = await self.agent_sdk.monte_carlo_validation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="monte_carlo_verification",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to monte_carlo_verification: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("mathematical_proof")
        async def handle_mathematical_proof(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle mathematical proof generation and verification"""
            try:
                result = await self.agent_sdk.generate_mathematical_proof(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="mathematical_proof",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to mathematical_proof: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("precision_analysis")
        async def handle_precision_analysis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle precision and error analysis"""
            try:
                result = await self.agent_sdk.precision_analysis(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="precision_analysis",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to precision_analysis: {e}")
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
def create_agent4CalcValidation_a2a_handler(agent_sdk: ComprehensiveCalcValidationSDK) -> Agent4CalcvalidationA2AHandler:
    """Create A2A-compliant handler for Agent 4: Calculation Validation Agent"""
    return Agent4CalcvalidationA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent4CalcValidation_a2a_handler(agent4CalcValidation_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
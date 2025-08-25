"""
A2A-Compliant Message Handler for calculation
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
from .comprehensiveCalculationAgentSdk import ComprehensiveCalculationAgentSDK

logger = logging.getLogger(__name__)


class CalculationAgentA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for calculation
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveCalculationAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="calculation_agent",
            agent_name="Calculation Agent",
            agent_version="1.0.0",
            allowed_operations={
                "calculation_agent_info",
                # Registry capabilities
                "mathematical_calculations",
                "statistical_analysis",
                "formula_execution",
                "numerical_processing",
                "computation_services",
                # Enhanced operations
                "perform_calculation",
                "natural_language_calculation",
                "intelligent_dispatch",
                "evaluate_expression",
                "differentiate_expression",
                "integrate_expression",
                "solve_equations",
                "price_bond",
                "price_option",
                "analyze_graph",
                "find_shortest_path",
                "distributed_calculation",
                "ai_assisted_calculation",
                "multi_step_calculation",
                "create_workflow",
                "execute_workflow",
                "get_workflow_status",
                "get_calculation_stats",
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

        @self.secure_handler("calculation_agent_info")
        async def handle_calculation_agent_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle calculation_agent_info operation"""
            try:
                # Get agent information and capabilities
                result = await self.agent_sdk.get_agent_info(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="calculation_agent_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to calculation_agent_info: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("perform_calculation")
        async def handle_perform_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle perform_calculation operation"""
            try:
                # Perform calculation using comprehensive SDK
                result = await self.agent_sdk.perform_calculation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="perform_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to perform_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("natural_language_calculation")
        async def handle_natural_language_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle natural_language_calculation operation"""
            try:
                # Process natural language calculation request
                result = await self.agent_sdk.natural_language_calculation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="natural_language_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to natural_language_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("intelligent_dispatch")
        async def handle_intelligent_dispatch(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle intelligent_dispatch operation"""
            try:
                # Dispatch calculation request intelligently
                result = await self.agent_sdk.intelligent_dispatch(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="intelligent_dispatch",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to intelligent_dispatch: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("evaluate_expression")
        async def handle_evaluate_expression(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle evaluate_expression operation"""
            try:
                # Evaluate mathematical expression
                result = await self.agent_sdk.evaluate_expression(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="evaluate_expression",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to evaluate_expression: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("differentiate_expression")
        async def handle_differentiate_expression(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle differentiate_expression operation"""
            try:
                # Differentiate mathematical expression
                result = await self.agent_sdk.differentiate_expression(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="differentiate_expression",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to differentiate_expression: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("integrate_expression")
        async def handle_integrate_expression(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle integrate_expression operation"""
            try:
                # Integrate mathematical expression
                result = await self.agent_sdk.integrate_expression(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="integrate_expression",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to integrate_expression: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("solve_equations")
        async def handle_solve_equations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle solve_equations operation"""
            try:
                # Solve mathematical equations
                result = await self.agent_sdk.solve_equations(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="solve_equations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to solve_equations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("price_bond")
        async def handle_price_bond(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle price_bond operation"""
            try:
                # Price financial bond
                result = await self.agent_sdk.price_bond(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="price_bond",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to price_bond: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("price_option")
        async def handle_price_option(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle price_option operation"""
            try:
                # Price financial option
                result = await self.agent_sdk.price_option(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="price_option",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to price_option: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("analyze_graph")
        async def handle_analyze_graph(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle analyze_graph operation"""
            try:
                # Analyze graph structure
                result = await self.agent_sdk.analyze_graph(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="analyze_graph",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to analyze_graph: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("find_shortest_path")
        async def handle_find_shortest_path(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle find_shortest_path operation"""
            try:
                # Find shortest path in graph
                result = await self.agent_sdk.find_shortest_path(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="find_shortest_path",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to find_shortest_path: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("distributed_calculation")
        async def handle_distributed_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle distributed_calculation operation"""
            try:
                # Execute distributed calculation
                result = await self.agent_sdk.distributed_calculation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="distributed_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to distributed_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("ai_assisted_calculation")
        async def handle_ai_assisted_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle ai_assisted_calculation operation"""
            try:
                # Perform AI-assisted calculation
                result = await self.agent_sdk.ai_assisted_calculation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="ai_assisted_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to ai_assisted_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("intelligent_dispatch")
        async def handle_intelligent_dispatch(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle intelligent_dispatch operation"""
            try:
                # Dispatch calculation request intelligently
                result = await self.agent_sdk.intelligent_dispatch(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="intelligent_dispatch",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to intelligent_dispatch: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("multi_step_calculation")
        async def handle_multi_step_calculation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle multi_step_calculation operation"""
            try:
                # Execute multi-step calculation
                result = await self.agent_sdk.multi_step_calculation(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="multi_step_calculation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to multi_step_calculation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_workflow")
        async def handle_create_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_workflow operation"""
            try:
                # Create calculation workflow
                result = await self.agent_sdk.create_workflow(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to create_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execute_workflow")
        async def handle_execute_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle execute_workflow operation"""
            try:
                # Execute calculation workflow
                result = await self.agent_sdk.execute_workflow(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execute_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to execute_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_workflow_status")
        async def handle_get_workflow_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_workflow_status operation"""
            try:
                # Get workflow execution status
                result = await self.agent_sdk.get_workflow_status(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_workflow_status",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_workflow_status: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_calculation_stats")
        async def handle_get_calculation_stats(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_calculation_stats operation"""
            try:
                # Get calculation statistics
                result = await self.agent_sdk.get_calculation_stats(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_calculation_stats",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_calculation_stats: {e}")
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
        @self.secure_handler("mathematical_calculations")
        async def handle_mathematical_calculations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle mathematical calculations operations"""
            try:
                result = await self.agent_sdk.perform_mathematical_calculations(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="mathematical_calculations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to mathematical_calculations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("statistical_analysis")
        async def handle_statistical_analysis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle statistical analysis operations"""
            try:
                result = await self.agent_sdk.perform_statistical_analysis(data)

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

        @self.secure_handler("formula_execution")
        async def handle_formula_execution(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle formula execution operations"""
            try:
                result = await self.agent_sdk.execute_formula(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="formula_execution",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to formula_execution: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("numerical_processing")
        async def handle_numerical_processing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle numerical processing operations"""
            try:
                result = await self.agent_sdk.process_numerical_data(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="numerical_processing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to numerical_processing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("computation_services")
        async def handle_computation_services(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle computation services operations"""
            try:
                result = await self.agent_sdk.provide_computation_services(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="computation_services",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to computation_services: {e}")
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
def create_calculation_agent_a2a_handler(agent_sdk: ComprehensiveCalculationAgentSDK) -> CalculationAgentA2AHandler:
    """Create A2A-compliant handler for calculation"""
    return CalculationAgentA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_calculation_agent_a2a_handler(calculation_agent_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
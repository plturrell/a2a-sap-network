"""
A2A-Compliant Message Handler for Agent Manager - A2A Ecosystem Orchestration
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
from .comprehensiveAgentManagerSdk import ComprehensiveAgentManagerSDK

logger = logging.getLogger(__name__)


class AgentManagerA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent Manager - A2A Ecosystem Orchestration
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveAgentManagerSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="agent_lifecycle_manager",
            agent_name="Agent Lifecycle Manager",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "agent_lifecycle_management",
                "agent_registration",
                "health_monitoring",
                "performance_tracking",
                "agent_coordination",
                # Enhanced operations
                "register_agent",
                "deregister_agent",
                "list_agents",
                "get_agent_info",
                "check_agent_health",
                "monitor_agent_health",
                "track_performance",
                "coordinate_agents",
                "manage_lifecycle",
                "create_trust_contract",
                "list_trust_contracts",
                "get_trust_contract",
                "revoke_trust_contract",
                "create_workflow",
                "list_workflows",
                "get_workflow_status",
                "cancel_workflow",
                "system_health_check",
                "system_metrics",
                "all_agents_health",
                # Base operations
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

        @self.secure_handler("register_agent")
        async def handle_register_agent(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle register_agent operation"""
            try:
                # TODO: Implement register_agent logic
                # Example: result = await self.agent_sdk.register_agent(data)
                result = {"status": "success", "operation": "register_agent"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="register_agent",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to register_agent: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("deregister_agent")
        async def handle_deregister_agent(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle deregister_agent operation"""
            try:
                # TODO: Implement deregister_agent logic
                # Example: result = await self.agent_sdk.deregister_agent(data)
                result = {"status": "success", "operation": "deregister_agent"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="deregister_agent",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to deregister_agent: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("list_agents")
        async def handle_list_agents(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle list_agents operation"""
            try:
                # TODO: Implement list_agents logic
                # Example: result = await self.agent_sdk.list_agents(data)
                result = {"status": "success", "operation": "list_agents"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="list_agents",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to list_agents: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_agent_info")
        async def handle_get_agent_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_agent_info operation"""
            try:
                # TODO: Implement get_agent_info logic
                # Example: result = await self.agent_sdk.get_agent_info(data)
                result = {"status": "success", "operation": "get_agent_info"}

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

        @self.secure_handler("check_agent_health")
        async def handle_check_agent_health(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle check_agent_health operation"""
            try:
                # TODO: Implement check_agent_health logic
                # Example: result = await self.agent_sdk.check_agent_health(data)
                result = {"status": "success", "operation": "check_agent_health"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="check_agent_health",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to check_agent_health: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_trust_contract")
        async def handle_create_trust_contract(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_trust_contract operation"""
            try:
                # TODO: Implement create_trust_contract logic
                # Example: result = await self.agent_sdk.create_trust_contract(data)
                result = {"status": "success", "operation": "create_trust_contract"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_trust_contract",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to create_trust_contract: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("list_trust_contracts")
        async def handle_list_trust_contracts(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle list_trust_contracts operation"""
            try:
                # TODO: Implement list_trust_contracts logic
                # Example: result = await self.agent_sdk.list_trust_contracts(data)
                result = {"status": "success", "operation": "list_trust_contracts"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="list_trust_contracts",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to list_trust_contracts: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_trust_contract")
        async def handle_get_trust_contract(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_trust_contract operation"""
            try:
                # TODO: Implement get_trust_contract logic
                # Example: result = await self.agent_sdk.get_trust_contract(data)
                result = {"status": "success", "operation": "get_trust_contract"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_trust_contract",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_trust_contract: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("revoke_trust_contract")
        async def handle_revoke_trust_contract(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle revoke_trust_contract operation"""
            try:
                # TODO: Implement revoke_trust_contract logic
                # Example: result = await self.agent_sdk.revoke_trust_contract(data)
                result = {"status": "success", "operation": "revoke_trust_contract"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="revoke_trust_contract",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to revoke_trust_contract: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_workflow")
        async def handle_create_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_workflow operation"""
            try:
                # TODO: Implement create_workflow logic
                # Example: result = await self.agent_sdk.create_workflow(data)
                result = {"status": "success", "operation": "create_workflow"}

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

        @self.secure_handler("list_workflows")
        async def handle_list_workflows(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle list_workflows operation"""
            try:
                # TODO: Implement list_workflows logic
                # Example: result = await self.agent_sdk.list_workflows(data)
                result = {"status": "success", "operation": "list_workflows"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="list_workflows",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to list_workflows: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_workflow_status")
        async def handle_get_workflow_status(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_workflow_status operation"""
            try:
                # TODO: Implement get_workflow_status logic
                # Example: result = await self.agent_sdk.get_workflow_status(data)
                result = {"status": "success", "operation": "get_workflow_status"}

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

        @self.secure_handler("cancel_workflow")
        async def handle_cancel_workflow(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle cancel_workflow operation"""
            try:
                # TODO: Implement cancel_workflow logic
                # Example: result = await self.agent_sdk.cancel_workflow(data)
                result = {"status": "success", "operation": "cancel_workflow"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cancel_workflow",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to cancel_workflow: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("system_health_check")
        async def handle_system_health_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle system_health_check operation"""
            try:
                # TODO: Implement system_health_check logic
                # Example: result = await self.agent_sdk.system_health_check(data)
                result = {"status": "success", "operation": "system_health_check"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="system_health_check",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to system_health_check: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("system_metrics")
        async def handle_system_metrics(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle system_metrics operation"""
            try:
                # TODO: Implement system_metrics logic
                # Example: result = await self.agent_sdk.system_metrics(data)
                result = {"status": "success", "operation": "system_metrics"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="system_metrics",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to system_metrics: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("all_agents_health")
        async def handle_all_agents_health(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle all_agents_health operation"""
            try:
                # TODO: Implement all_agents_health logic
                # Example: result = await self.agent_sdk.all_agents_health(data)
                result = {"status": "success", "operation": "all_agents_health"}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="all_agents_health",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to all_agents_health: {e}")
                return self.create_secure_response(str(e), status="error")
        # Registry capability handlers
        @self.secure_handler("agent_lifecycle_management")
        async def handle_agent_lifecycle_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive agent lifecycle management"""
            try:
                result = await self.agent_sdk.manage_lifecycle(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="agent_lifecycle_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to agent_lifecycle_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("agent_registration")
        async def handle_agent_registration_capability(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle agent registration capability"""
            try:
                result = await self.agent_sdk.register_agent_enhanced(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="agent_registration",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to agent_registration: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("health_monitoring")
        async def handle_health_monitoring_capability(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle comprehensive health monitoring capability"""
            try:
                result = await self.agent_sdk.monitor_health(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="health_monitoring",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to health_monitoring: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("performance_tracking")
        async def handle_performance_tracking_capability(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle performance tracking capability"""
            try:
                result = await self.agent_sdk.track_performance(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="performance_tracking",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to performance_tracking: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("agent_coordination")
        async def handle_agent_coordination_capability(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle agent coordination capability"""
            try:
                result = await self.agent_sdk.coordinate_agents(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="agent_coordination",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to agent_coordination: {e}")
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
def create_agent_manager_a2a_handler(agent_sdk: ComprehensiveAgentManagerSDK) -> AgentManagerA2AHandler:
    """Create A2A-compliant handler for Agent Manager - A2A Ecosystem Orchestration"""
    return AgentManagerA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent_manager_a2a_handler(agent_manager_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""

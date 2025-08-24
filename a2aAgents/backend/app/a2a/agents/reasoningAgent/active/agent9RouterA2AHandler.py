"""
A2A-Compliant Message Handler for Agent 9 - Reasoning
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
from .comprehensiveReasoningAgentSdk import ComprehensiveReasoningAgentSdk

logger = logging.getLogger(__name__)


class Agent9RouterA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 9 - Reasoning
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveReasoningAgentSdk):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="agent9Router",
            agent_name="Agent 9 - Reasoning",
            agent_version="1.0.0",
            allowed_operations={
                "get_agent_card",
                "json_rpc",
                "create_reasoning_task",
                "list_reasoning_tasks",
                "start_reasoning",
                "validate_conclusion",
                "explain_reasoning",
                "add_knowledge",
                "validate_knowledge_base",
                "generate_inferences",
                "make_decision",
                "solve_problem",
                "get_dashboard_data",
                "get_reasoning_options",
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

        @self.secure_handler("create_reasoning_task")
        async def handle_create_reasoning_task(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_reasoning_task operation"""
            try:
                # TODO: Implement create_reasoning_task logic
                # Example: result = await self.agent_sdk.create_reasoning_task(data)
                result = {"status": "success", "operation": "create_reasoning_task"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_reasoning_task",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to create_reasoning_task: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("list_reasoning_tasks")
        async def handle_list_reasoning_tasks(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle list_reasoning_tasks operation"""
            try:
                # TODO: Implement list_reasoning_tasks logic
                # Example: result = await self.agent_sdk.list_reasoning_tasks(data)
                result = {"status": "success", "operation": "list_reasoning_tasks"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="list_reasoning_tasks",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to list_reasoning_tasks: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("start_reasoning")
        async def handle_start_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle start_reasoning operation"""
            try:
                # TODO: Implement start_reasoning logic
                # Example: result = await self.agent_sdk.start_reasoning(data)
                result = {"status": "success", "operation": "start_reasoning"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="start_reasoning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to start_reasoning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_conclusion")
        async def handle_validate_conclusion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validate_conclusion operation"""
            try:
                # TODO: Implement validate_conclusion logic
                # Example: result = await self.agent_sdk.validate_conclusion(data)
                result = {"status": "success", "operation": "validate_conclusion"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_conclusion",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to validate_conclusion: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("explain_reasoning")
        async def handle_explain_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle explain_reasoning operation"""
            try:
                # TODO: Implement explain_reasoning logic
                # Example: result = await self.agent_sdk.explain_reasoning(data)
                result = {"status": "success", "operation": "explain_reasoning"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="explain_reasoning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to explain_reasoning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("add_knowledge")
        async def handle_add_knowledge(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle add_knowledge operation"""
            try:
                # TODO: Implement add_knowledge logic
                # Example: result = await self.agent_sdk.add_knowledge(data)
                result = {"status": "success", "operation": "add_knowledge"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="add_knowledge",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to add_knowledge: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_knowledge_base")
        async def handle_validate_knowledge_base(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validate_knowledge_base operation"""
            try:
                # TODO: Implement validate_knowledge_base logic
                # Example: result = await self.agent_sdk.validate_knowledge_base(data)
                result = {"status": "success", "operation": "validate_knowledge_base"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_knowledge_base",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to validate_knowledge_base: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_inferences")
        async def handle_generate_inferences(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generate_inferences operation"""
            try:
                # TODO: Implement generate_inferences logic
                # Example: result = await self.agent_sdk.generate_inferences(data)
                result = {"status": "success", "operation": "generate_inferences"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_inferences",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to generate_inferences: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("make_decision")
        async def handle_make_decision(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle make_decision operation"""
            try:
                # TODO: Implement make_decision logic
                # Example: result = await self.agent_sdk.make_decision(data)
                result = {"status": "success", "operation": "make_decision"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="make_decision",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to make_decision: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("solve_problem")
        async def handle_solve_problem(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle solve_problem operation"""
            try:
                # TODO: Implement solve_problem logic
                # Example: result = await self.agent_sdk.solve_problem(data)
                result = {"status": "success", "operation": "solve_problem"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="solve_problem",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to solve_problem: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_dashboard_data")
        async def handle_get_dashboard_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_dashboard_data operation"""
            try:
                # TODO: Implement get_dashboard_data logic
                # Example: result = await self.agent_sdk.get_dashboard_data(data)
                result = {"status": "success", "operation": "get_dashboard_data"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_dashboard_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_dashboard_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_reasoning_options")
        async def handle_get_reasoning_options(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_reasoning_options operation"""
            try:
                # TODO: Implement get_reasoning_options logic
                # Example: result = await self.agent_sdk.get_reasoning_options(data)
                result = {"status": "success", "operation": "get_reasoning_options"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_reasoning_options",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_reasoning_options: {e}")
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
def create_agent9Router_a2a_handler(agent_sdk: ComprehensiveReasoningAgentSdk) -> Agent9RouterA2AHandler:
    """Create A2A-compliant handler for Agent 9 - Reasoning"""
    return Agent9RouterA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_agent9Router_a2a_handler(agent9Router_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
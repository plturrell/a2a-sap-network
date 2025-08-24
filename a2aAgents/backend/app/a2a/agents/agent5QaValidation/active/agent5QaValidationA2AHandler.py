"""
A2A-Compliant Message Handler for Agent 5 - QA Validation
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
from .None import None

logger = logging.getLogger(__name__)


class Agent5QavalidationA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 5 - QA Validation
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: None):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="agent5QaValidation",
            agent_name="Agent 5 - QA Validation",
            agent_version="1.0.0",
            allowed_operations={
                "initialize_agent_endpoint",
                "health_check",
                "get_agent_card",
                "discover_ord_products",
                "generate_dynamic_tests",
                "execute_a2a_task",
                "get_task_status",
                "get_task_report",
                "get_partial_task_report",
                "get_question_templates",
                "get_metrics",
                "reset_validation_state",
                "shutdown_agent"
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

        @self.secure_handler("initialize_agent_endpoint")
        async def handle_initialize_agent_endpoint(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle initialize_agent_endpoint operation"""
            try:
                # TODO: Implement initialize_agent_endpoint logic
                # Example: result = await self.agent_sdk.initialize_agent_endpoint(data)
                result = {"status": "success", "operation": "initialize_agent_endpoint"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="initialize_agent_endpoint",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to initialize_agent_endpoint: {e}")
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

        @self.secure_handler("discover_ord_products")
        async def handle_discover_ord_products(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle discover_ord_products operation"""
            try:
                # TODO: Implement discover_ord_products logic
                # Example: result = await self.agent_sdk.discover_ord_products(data)
                result = {"status": "success", "operation": "discover_ord_products"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="discover_ord_products",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to discover_ord_products: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_dynamic_tests")
        async def handle_generate_dynamic_tests(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generate_dynamic_tests operation"""
            try:
                # TODO: Implement generate_dynamic_tests logic
                # Example: result = await self.agent_sdk.generate_dynamic_tests(data)
                result = {"status": "success", "operation": "generate_dynamic_tests"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_dynamic_tests",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to generate_dynamic_tests: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execute_a2a_task")
        async def handle_execute_a2a_task(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle execute_a2a_task operation"""
            try:
                # TODO: Implement execute_a2a_task logic
                # Example: result = await self.agent_sdk.execute_a2a_task(data)
                result = {"status": "success", "operation": "execute_a2a_task"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execute_a2a_task",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to execute_a2a_task: {e}")
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

        @self.secure_handler("get_task_report")
        async def handle_get_task_report(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_task_report operation"""
            try:
                # TODO: Implement get_task_report logic
                # Example: result = await self.agent_sdk.get_task_report(data)
                result = {"status": "success", "operation": "get_task_report"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_task_report",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_task_report: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_partial_task_report")
        async def handle_get_partial_task_report(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_partial_task_report operation"""
            try:
                # TODO: Implement get_partial_task_report logic
                # Example: result = await self.agent_sdk.get_partial_task_report(data)
                result = {"status": "success", "operation": "get_partial_task_report"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_partial_task_report",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_partial_task_report: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_question_templates")
        async def handle_get_question_templates(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_question_templates operation"""
            try:
                # TODO: Implement get_question_templates logic
                # Example: result = await self.agent_sdk.get_question_templates(data)
                result = {"status": "success", "operation": "get_question_templates"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_question_templates",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_question_templates: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_metrics")
        async def handle_get_metrics(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_metrics operation"""
            try:
                # TODO: Implement get_metrics logic
                # Example: result = await self.agent_sdk.get_metrics(data)
                result = {"status": "success", "operation": "get_metrics"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_metrics",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_metrics: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("reset_validation_state")
        async def handle_reset_validation_state(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle reset_validation_state operation"""
            try:
                # TODO: Implement reset_validation_state logic
                # Example: result = await self.agent_sdk.reset_validation_state(data)
                result = {"status": "success", "operation": "reset_validation_state"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="reset_validation_state",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to reset_validation_state: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("shutdown_agent")
        async def handle_shutdown_agent(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle shutdown_agent operation"""
            try:
                # TODO: Implement shutdown_agent logic
                # Example: result = await self.agent_sdk.shutdown_agent(data)
                result = {"status": "success", "operation": "shutdown_agent"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="shutdown_agent",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to shutdown_agent: {e}")
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
def create_agent5QaValidation_a2a_handler(agent_sdk: None) -> Agent5QavalidationA2AHandler:
    """Create A2A-compliant handler for Agent 5 - QA Validation"""
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
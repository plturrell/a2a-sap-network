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
            agent_id="logical_reasoning_agent",
            agent_name="Logical Reasoning Agent",
            agent_version="2.0.0",
            allowed_operations={
                "get_agent_card",\n                # Registry capabilities\n                "logical_reasoning",\n                "inference_generation",\n                "decision_making",\n                "knowledge_synthesis",\n                "problem_solving",\n                # Enhanced operations\n                "perform_logical_reasoning",\n                "generate_inferences_enhanced",\n                "make_decisions_enhanced",\n                "synthesize_knowledge_enhanced",\n                "solve_problems_enhanced",
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
                # Process JSON-RPC request through agent SDK
                result = await self.agent_sdk.handle_json_rpc(data)
                
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
                # Start reasoning process using logical reasoning skill
                reasoning_query = data.get("query", "")
                reasoning_type = data.get("reasoning_type", "deductive") 
                premises = data.get("premises", [])
                domain = data.get("domain", "general")
                
                result = await self.agent_sdk.logical_reasoning({
                    "query": reasoning_query,
                    "reasoning_type": reasoning_type,
                    "domain": domain,
                    "premises": premises,
                    "context": data.get("context", {})
                })
                
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
                # Validate conclusion using confidence assessment
                conclusion = data.get("conclusion", "")
                evidence = data.get("evidence", [])
                reasoning_type = data.get("reasoning_type", "deductive")
                chain_id = data.get("chain_id")
                
                result = await self.agent_sdk.confidence_assessment({
                    "chain_id": chain_id,
                    "conclusion": conclusion,
                    "evidence": evidence,
                    "reasoning_type": reasoning_type
                })
                
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
                # Generate inferences using pattern analysis
                inference_data = data.get("data", [])
                pattern_type = data.get("pattern_type", "logical")
                analysis_depth = data.get("analysis_depth", "comprehensive")
                
                result = await self.agent_sdk.pattern_analysis({
                    "data": inference_data,
                    "pattern_type": pattern_type,
                    "analysis_depth": analysis_depth
                })
                
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
                # Solve problem using collaborative reasoning if needed
                problem_query = data.get("query", data.get("problem", ""))
                participants = data.get("participant_agents", [])
                strategy = data.get("strategy", "consensus")
                domain = data.get("domain", "general")
                
                if participants:
                    # Use collaborative reasoning for complex problems
                    result = await self.agent_sdk.collaborative_reasoning({
                        "participant_agents": participants,
                        "query": problem_query,
                        "strategy": strategy,
                        "domain": domain
                    })
                else:
                    # Use logical reasoning for individual problem solving
                    result = await self.agent_sdk.logical_reasoning({
                        "query": problem_query,
                        "reasoning_type": data.get("reasoning_type", "abductive"),
                        "domain": domain,
                        "premises": data.get("premises", [])
                    })
                
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
    
        # Registry capability handlers\n        @self.secure_handler(\"logical_reasoning\")\n        async def handle_logical_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:\n            \"\"\"Handle logical reasoning operations\"\"\"\n            try:\n                result = await self.agent_sdk.perform_logical_reasoning(data)\n                \n                # Log blockchain transaction\n                await self._log_blockchain_transaction(\n                    operation=\"logical_reasoning\",\n                    data_hash=self._hash_data(data),\n                    result_hash=self._hash_data(result),\n                    context_id=context_id\n                )\n                \n                return self.create_secure_response(result)\n                \n            except Exception as e:\n                logger.error(f\"Failed to logical_reasoning: {e}\")\n                return self.create_secure_response(str(e), status=\"error\")\n\n        @self.secure_handler(\"inference_generation\")\n        async def handle_inference_generation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:\n            \"\"\"Handle inference generation\"\"\"\n            try:\n                result = await self.agent_sdk.generate_inferences_enhanced(data)\n                \n                # Log blockchain transaction\n                await self._log_blockchain_transaction(\n                    operation=\"inference_generation\",\n                    data_hash=self._hash_data(data),\n                    result_hash=self._hash_data(result),\n                    context_id=context_id\n                )\n                \n                return self.create_secure_response(result)\n                \n            except Exception as e:\n                logger.error(f\"Failed to inference_generation: {e}\")\n                return self.create_secure_response(str(e), status=\"error\")\n\n        @self.secure_handler(\"decision_making\")\n        async def handle_decision_making(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:\n            \"\"\"Handle decision making operations\"\"\"\n            try:\n                result = await self.agent_sdk.make_decisions_enhanced(data)\n                \n                # Log blockchain transaction\n                await self._log_blockchain_transaction(\n                    operation=\"decision_making\",\n                    data_hash=self._hash_data(data),\n                    result_hash=self._hash_data(result),\n                    context_id=context_id\n                )\n                \n                return self.create_secure_response(result)\n                \n            except Exception as e:\n                logger.error(f\"Failed to decision_making: {e}\")\n                return self.create_secure_response(str(e), status=\"error\")\n\n        @self.secure_handler(\"knowledge_synthesis\")\n        async def handle_knowledge_synthesis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:\n            \"\"\"Handle knowledge synthesis operations\"\"\"\n            try:\n                result = await self.agent_sdk.synthesize_knowledge_enhanced(data)\n                \n                # Log blockchain transaction\n                await self._log_blockchain_transaction(\n                    operation=\"knowledge_synthesis\",\n                    data_hash=self._hash_data(data),\n                    result_hash=self._hash_data(result),\n                    context_id=context_id\n                )\n                \n                return self.create_secure_response(result)\n                \n            except Exception as e:\n                logger.error(f\"Failed to knowledge_synthesis: {e}\")\n                return self.create_secure_response(str(e), status=\"error\")\n\n        @self.secure_handler(\"problem_solving\")\n        async def handle_problem_solving(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:\n            \"\"\"Handle problem solving operations\"\"\"\n            try:\n                result = await self.agent_sdk.solve_problems_enhanced(data)\n                \n                # Log blockchain transaction\n                await self._log_blockchain_transaction(\n                    operation=\"problem_solving\",\n                    data_hash=self._hash_data(data),\n                    result_hash=self._hash_data(result),\n                    context_id=context_id\n                )\n                \n                return self.create_secure_response(result)\n                \n            except Exception as e:\n                logger.error(f\"Failed to problem_solving: {e}\")\n                return self.create_secure_response(str(e), status=\"error\")\n    \n    async def process_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
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
"""
A2A-Compliant Message Handler for Agent 3 - SAP HANA Vector Engine Ingestion
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
from .comprehensiveVectorProcessingSdk import ComprehensiveVectorProcessingSDK

logger = logging.getLogger(__name__)


class Agent3VectorprocessingA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 3 - SAP HANA Vector Engine Ingestion
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveVectorProcessingSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="vector_processing_agent",
            agent_name="Vector Processing Agent",
            agent_version="2.0.0",
            allowed_operations={
                # Registry capabilities
                "vector_generation",
                "embedding_creation",
                "similarity_search",
                "vector_optimization",
                "semantic_analysis",
                # Enhanced operations
                "generate_vectors",
                "create_embeddings",
                "search_similar",
                "optimize_vectors",
                "analyze_semantics",
                "hybrid_search",
                "cluster_vectors",
                "reduce_dimensions",
                "detect_anomalies",
                "assess_quality",
                # Base operations
                "get_agent_card",
                "json_rpc",
                "process_message",
                "get_task_status",
                "vector_search",
                "sparql_query",
                "get_vector_stores",
                "get_knowledge_graph_info",
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

        @self.secure_handler("vector_search")
        async def handle_vector_search(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector_search operation"""
            try:
                # TODO: Implement vector_search logic
                # Example: result = await self.agent_sdk.vector_search(data)
                result = {"status": "success", "operation": "vector_search"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="vector_search",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to vector_search: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("sparql_query")
        async def handle_sparql_query(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle sparql_query operation"""
            try:
                # TODO: Implement sparql_query logic
                # Example: result = await self.agent_sdk.sparql_query(data)
                result = {"status": "success", "operation": "sparql_query"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="sparql_query",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to sparql_query: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_vector_stores")
        async def handle_get_vector_stores(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_vector_stores operation"""
            try:
                # TODO: Implement get_vector_stores logic
                # Example: result = await self.agent_sdk.get_vector_stores(data)
                result = {"status": "success", "operation": "get_vector_stores"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_vector_stores",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_vector_stores: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_knowledge_graph_info")
        async def handle_get_knowledge_graph_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_knowledge_graph_info operation"""
            try:
                # TODO: Implement get_knowledge_graph_info logic
                # Example: result = await self.agent_sdk.get_knowledge_graph_info(data)
                result = {"status": "success", "operation": "get_knowledge_graph_info"}
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_knowledge_graph_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get_knowledge_graph_info: {e}")
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

        # Registry capability handlers
        @self.secure_handler("vector_generation")
        async def handle_vector_generation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector generation operations"""
            try:
                result = await self.agent_sdk.generate_embeddings(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="vector_generation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to vector_generation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("embedding_creation")
        async def handle_embedding_creation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding creation operations"""
            try:
                result = await self.agent_sdk.generate_embeddings(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="embedding_creation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to embedding_creation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("similarity_search")
        async def handle_similarity_search(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle similarity search operations"""
            try:
                result = await self.agent_sdk.search_vectors(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="similarity_search",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to similarity_search: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("vector_optimization")
        async def handle_vector_optimization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector optimization operations"""
            try:
                result = await self.agent_sdk.vector_dimensionality_reduction(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="vector_optimization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to vector_optimization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("semantic_analysis")
        async def handle_semantic_analysis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle semantic analysis operations"""
            try:
                result = await self.agent_sdk.vector_quality_assessment(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="semantic_analysis",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to semantic_analysis: {e}")
                return self.create_secure_response(str(e), status="error")

        # Enhanced operation handlers
        @self.secure_handler("generate_vectors")
        async def handle_generate_vectors(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector generation with advanced options"""
            try:
                result = await self.agent_sdk.generate_embeddings(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_vectors",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to generate_vectors: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_embeddings")
        async def handle_create_embeddings(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding creation with ML optimization"""
            try:
                result = await self.agent_sdk.generate_embeddings(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_embeddings",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to create_embeddings: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("search_similar")
        async def handle_search_similar(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle similarity search with ML re-ranking"""
            try:
                result = await self.agent_sdk.search_vectors(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="search_similar",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to search_similar: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_vectors")
        async def handle_optimize_vectors(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector optimization and compression"""
            try:
                result = await self.agent_sdk.vector_dimensionality_reduction(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="optimize_vectors",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to optimize_vectors: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("analyze_semantics")
        async def handle_analyze_semantics(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle semantic analysis and quality assessment"""
            try:
                result = await self.agent_sdk.vector_quality_assessment(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="analyze_semantics",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze_semantics: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("hybrid_search")
        async def handle_hybrid_search(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle hybrid vector search operations"""
            try:
                result = await self.agent_sdk.hybrid_vector_search(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="hybrid_search",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to hybrid_search: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("cluster_vectors")
        async def handle_cluster_vectors(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector clustering operations"""
            try:
                result = await self.agent_sdk.vector_clustering(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cluster_vectors",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to cluster_vectors: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("reduce_dimensions")
        async def handle_reduce_dimensions(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle dimensionality reduction operations"""
            try:
                result = await self.agent_sdk.vector_dimensionality_reduction(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="reduce_dimensions",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to reduce_dimensions: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("detect_anomalies")
        async def handle_detect_anomalies(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector anomaly detection operations"""
            try:
                result = await self.agent_sdk.vector_anomaly_detection(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="detect_anomalies",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to detect_anomalies: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("assess_quality")
        async def handle_assess_quality(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector quality assessment operations"""
            try:
                result = await self.agent_sdk.vector_quality_assessment(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="assess_quality",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to assess_quality: {e}")
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
def create_agent3VectorProcessing_a2a_handler(agent_sdk: ComprehensiveVectorProcessingSDK) -> Agent3VectorprocessingA2AHandler:
    """Create A2A-compliant handler for Agent 3 - Vector Processing Agent"""
    return Agent3VectorprocessingA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_agent3VectorProcessing_a2a_handler(agent3VectorProcessing_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
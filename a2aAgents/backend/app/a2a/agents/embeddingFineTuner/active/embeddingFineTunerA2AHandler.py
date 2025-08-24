"""
A2A-Compliant Message Handler for Embedding Fine-Tuner Agent
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
from .comprehensiveEmbeddingFineTunerSdk import ComprehensiveEmbeddingFineTunerSDK

logger = logging.getLogger(__name__)


class EmbeddingFineTunerA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Embedding Fine-Tuner Agent
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveEmbeddingFineTunerSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="embedding_fine_tuner",
            agent_name="Embedding Fine-Tuner",
            agent_version="1.0.0",
            allowed_operations={
                "embedding_fine_tuner_info",
                "embedding_optimization",
                "model_fine_tuning", 
                "vector_improvement",
                "performance_tuning",
                "embedding_evaluation",
                "train_embedding_model",
                "optimize_embeddings",
                "evaluate_model_performance",
                "batch_embedding_processing",
                "hyperparameter_optimization",
                "cross_validation",
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

        @self.secure_handler("embedding_fine_tuner_info")
        async def handle_embedding_fine_tuner_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding_fine_tuner_info operation"""
            try:
                # Get embedding fine-tuner agent information and capabilities
                result = await self.agent_sdk.get_agent_info(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="embedding_fine_tuner_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to embedding_fine_tuner_info: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("embedding_optimization")
        async def handle_embedding_optimization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding_optimization operation"""
            try:
                # Optimize embeddings using AI techniques
                result = await self.agent_sdk.embedding_optimization(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="embedding_optimization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to embedding_optimization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("model_fine_tuning")
        async def handle_model_fine_tuning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle model_fine_tuning operation"""
            try:
                # Fine-tune embedding models
                result = await self.agent_sdk.model_fine_tuning(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="model_fine_tuning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to model_fine_tuning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("vector_improvement")
        async def handle_vector_improvement(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector_improvement operation"""
            try:
                # Improve vector quality and representation
                result = await self.agent_sdk.vector_improvement(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="vector_improvement",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to vector_improvement: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("performance_tuning")
        async def handle_performance_tuning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle performance_tuning operation"""
            try:
                # Tune model performance parameters
                result = await self.agent_sdk.performance_tuning(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="performance_tuning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to performance_tuning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("embedding_evaluation")
        async def handle_embedding_evaluation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle embedding_evaluation operation"""
            try:
                # Evaluate embedding model quality
                result = await self.agent_sdk.embedding_evaluation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="embedding_evaluation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to embedding_evaluation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("train_embedding_model")
        async def handle_train_embedding_model(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle train_embedding_model operation"""
            try:
                # Train new embedding model
                result = await self.agent_sdk.train_embedding_model(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="train_embedding_model",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to train_embedding_model: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_embeddings")
        async def handle_optimize_embeddings(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle optimize_embeddings operation"""
            try:
                # Optimize existing embeddings
                result = await self.agent_sdk.optimize_embeddings(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="optimize_embeddings",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to optimize_embeddings: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("evaluate_model_performance")
        async def handle_evaluate_model_performance(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle evaluate_model_performance operation"""
            try:
                # Evaluate model performance metrics
                result = await self.agent_sdk.evaluate_model_performance(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="evaluate_model_performance",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate_model_performance: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("batch_embedding_processing")
        async def handle_batch_embedding_processing(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle batch_embedding_processing operation"""
            try:
                # Process embeddings in batches
                result = await self.agent_sdk.batch_embedding_processing(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="batch_embedding_processing",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to batch_embedding_processing: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("hyperparameter_optimization")
        async def handle_hyperparameter_optimization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle hyperparameter_optimization operation"""
            try:
                # Optimize model hyperparameters
                result = await self.agent_sdk.hyperparameter_optimization(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="hyperparameter_optimization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to hyperparameter_optimization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("cross_validation")
        async def handle_cross_validation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle cross_validation operation"""
            try:
                # Perform cross-validation on models
                result = await self.agent_sdk.cross_validation(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="cross_validation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to cross_validation: {e}")
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
def create_embedding_fine_tuner_a2a_handler(agent_sdk: ComprehensiveEmbeddingFineTunerSDK) -> EmbeddingFineTunerA2AHandler:
    """Create A2A-compliant handler for Embedding Fine-Tuner Agent"""
    return EmbeddingFineTunerA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_embedding_fine_tuner_a2a_handler(embedding_fine_tuner_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
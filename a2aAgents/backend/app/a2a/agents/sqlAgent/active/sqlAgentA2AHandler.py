"""
A2A-Compliant Message Handler for SQL Agent
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
from .comprehensiveSqlAgentSdk import ComprehensiveSqlAgentSDK

logger = logging.getLogger(__name__)


class SqlAgentA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for SQL Agent
    All communication through blockchain messaging only
    """
    
    def __init__(self, agent_sdk: ComprehensiveSqlAgentSDK):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="sql_agent",
            agent_name="SQL Agent",
            agent_version="1.0.0",
            allowed_operations={
                "sql_agent_info",
                # Registry capabilities
                "sql_query_execution",
                "database_operations",
                "query_optimization",
                "data_extraction",
                "schema_management",
                # Enhanced operations
                "execute_sql_query",
                "natural_language_to_sql",
                "optimize_query",
                "validate_query",
                "analyze_schema",
                "generate_report",
                "batch_operations",
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

        @self.secure_handler("sql_agent_info")
        async def handle_sql_agent_info(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle sql_agent_info operation"""
            try:
                # Get SQL agent information and capabilities
                result = await self.agent_sdk.get_agent_info(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="sql_agent_info",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to sql_agent_info: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("execute_sql_query")
        async def handle_execute_sql_query(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle execute_sql_query operation"""
            try:
                # Execute SQL query with security validation
                result = await self.agent_sdk.execute_sql_query(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="execute_sql_query",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to execute_sql_query: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("natural_language_to_sql")
        async def handle_natural_language_to_sql(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle natural_language_to_sql operation"""
            try:
                # Convert natural language to SQL using AI
                result = await self.agent_sdk.natural_language_to_sql(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="natural_language_to_sql",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to natural_language_to_sql: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_query")
        async def handle_optimize_query(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle optimize_query operation"""
            try:
                # Optimize SQL query performance
                result = await self.agent_sdk.optimize_query(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="optimize_query",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to optimize_query: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_query")
        async def handle_validate_query(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validate_query operation"""
            try:
                # Validate SQL query for security and syntax
                result = await self.agent_sdk.validate_query(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_query",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to validate_query: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("analyze_schema")
        async def handle_analyze_schema(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle analyze_schema operation"""
            try:
                # Analyze database schema structure
                result = await self.agent_sdk.analyze_schema(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="analyze_schema",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze_schema: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_report")
        async def handle_generate_report(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generate_report operation"""
            try:
                # Generate database report
                result = await self.agent_sdk.generate_report(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_report",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to generate_report: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("batch_operations")
        async def handle_batch_operations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle batch_operations operation"""
            try:
                # Execute batch SQL operations
                result = await self.agent_sdk.batch_operations(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="batch_operations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to batch_operations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("database_operations")
        async def handle_database_operations(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle database_operations operation"""
            try:
                # Perform general database operations
                result = await self.agent_sdk.database_operations(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="database_operations",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to database_operations: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("schema_management")
        async def handle_schema_management(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle schema_management operation"""
            try:
                # Manage database schema
                result = await self.agent_sdk.schema_management(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="schema_management",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to schema_management: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("query_optimization")
        async def handle_query_optimization(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle query_optimization operation"""
            try:
                # Optimize database queries
                result = await self.agent_sdk.query_optimization(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="query_optimization",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to query_optimization: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("data_extraction")
        async def handle_data_extraction(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle data_extraction operation"""
            try:
                # Extract data from database
                result = await self.agent_sdk.data_extraction(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="data_extraction",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to data_extraction: {e}")
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
        @self.secure_handler("sql_query_execution")
        async def handle_sql_query_execution(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle SQL query execution with enhanced features"""
            try:
                result = await self.agent_sdk.execute_sql_queries(data)
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="sql_query_execution",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to sql_query_execution: {e}")
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
def create_sql_agent_a2a_handler(agent_sdk: ComprehensiveSqlAgentSDK) -> SqlAgentA2AHandler:
    """Create A2A-compliant handler for SQL Agent"""
    return SqlAgentA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW: 
   handler = create_sql_agent_a2a_handler(sql_agent_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()
   
3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
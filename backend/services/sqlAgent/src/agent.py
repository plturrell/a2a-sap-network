"""
SQL Agent - A2A Microservice
Specialized agent for SQL query generation, optimization, and execution
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import re
import json

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


class SQLAgent(A2AAgentBase):
    """
    SQL Agent
    A2A compliant agent for SQL operations and database interactions
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="sql_agent",
            name="SQL Agent",
            description="A2A v0.2.9 compliant agent for SQL query generation and database operations",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # SQL configuration
        self.sql_config = {
            "max_query_length": 10000,
            "supported_databases": ["postgresql", "mysql", "sqlite", "sqlserver"],
            "query_timeout": 30,
            "enable_optimization": True
        }

        # Schema knowledge base
        self.schema_registry = {
            "tables": {},
            "relationships": [],
            "indexes": {}
        }

        self.sql_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "query_types": {},
            "optimization_suggestions": 0
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing SQL Agent...")

        # Initialize output directory
        self.output_dir = os.getenv("SQL_OUTPUT_DIR", "/tmp/sql_queries")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load schema registry if available
        await self._load_schema_registry()

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("SQL Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "sql_operations": ["query_generation", "optimization", "validation", "execution"],
                "query_types": ["select", "insert", "update", "delete", "create", "alter"],
                "database_support": self.sql_config["supported_databases"],
                "features": ["schema_analysis", "performance_tuning", "security_validation"]
            }

            logger.info("Registered with A2A network at %s", self.agent_manager_url)
            self.is_registered = True

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to register with A2A network: %s", e)
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
        logger.info("Successfully deregistered from A2A network")
    
    @a2a_handler("sql_operation", "Perform SQL operations and query management")
    async def handle_sql_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for SQL requests"""
        try:
            # Extract SQL request from A2A message
            sql_request = self._extract_sql_request(message)
            
            if not sql_request:
                return create_error_response(400, "No SQL request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("sql_operation", {
                "context_id": context_id,
                "request": sql_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_sql_operations(task_id, sql_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "operations": list(sql_request.keys()),
                "message": "SQL operations started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling SQL request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("generate_query", "Generate SQL queries from natural language")
    async def generate_query(self, description: str, schema_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate SQL query from natural language description"""
        try:
            # Simple query generation based on keywords
            description_lower = description.lower()
            
            # Initialize query components
            select_parts = []
            from_parts = []
            where_parts = []
            joins = []
            
            # Detect operation type
            if any(keyword in description_lower for keyword in ["select", "find", "get", "show", "list"]):
                operation = "SELECT"
            elif any(keyword in description_lower for keyword in ["insert", "add", "create"]):
                operation = "INSERT"
            elif any(keyword in description_lower for keyword in ["update", "modify", "change"]):
                operation = "UPDATE"
            elif any(keyword in description_lower for keyword in ["delete", "remove"]):
                operation = "DELETE"
            else:
                operation = "SELECT"  # Default
            
            # Extract table names (basic pattern matching)
            table_patterns = ["from table", "in table", "table named"]
            tables_found = []
            
            if schema_context and "tables" in schema_context:
                # Use provided schema context
                for table_name in schema_context["tables"]:
                    if table_name.lower() in description_lower:
                        tables_found.append(table_name)
            else:
                # Try to extract table names from description
                words = description_lower.split()
                for i, word in enumerate(words):
                    if word in ["table", "from"] and i + 1 < len(words):
                        potential_table = words[i + 1].strip(',.')
                        tables_found.append(potential_table)
            
            # Generate basic query structure
            if operation == "SELECT":
                if "count" in description_lower:
                    select_parts.append("COUNT(*)")
                elif "all" in description_lower or "everything" in description_lower:
                    select_parts.append("*")
                else:
                    # Extract column hints
                    column_keywords = ["name", "id", "date", "amount", "price", "quantity"]
                    for keyword in column_keywords:
                        if keyword in description_lower:
                            select_parts.append(keyword)
                    
                    if not select_parts:
                        select_parts.append("*")
                
                from_parts = tables_found if tables_found else ["your_table"]
                
                # Extract WHERE conditions
                if "where" in description_lower:
                    where_index = description_lower.find("where")
                    where_clause = description[where_index + 5:].strip()
                    where_parts.append(f"/* {where_clause} */")
                
                # Build SELECT query
                query = f"SELECT {', '.join(select_parts)}"
                query += f" FROM {', '.join(from_parts)}"
                if where_parts:
                    query += f" WHERE {' AND '.join(where_parts)}"
            
            else:
                # For non-SELECT operations, provide template
                query = f"/* {operation} query template based on: {description} */"
                if tables_found:
                    query += f" -- Table(s): {', '.join(tables_found)}"
            
            # Add semicolon
            if not query.endswith(';'):
                query += ';'
            
            return {
                "query": query,
                "operation": operation,
                "description": description,
                "tables_identified": tables_found,
                "confidence": 0.7 if tables_found else 0.4,
                "generated_at": datetime.utcnow().isoformat(),
                "note": "This is a generated query template. Please review and modify as needed."
            }
            
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return {"error": str(e)}

    @a2a_skill("validate_query", "Validate SQL query syntax and security")
    async def validate_query(self, query: str, database_type: str = "postgresql") -> Dict[str, Any]:
        """Validate SQL query for syntax and security issues"""
        try:
            validation_results = {
                "is_valid": True,
                "syntax_issues": [],
                "security_issues": [],
                "warnings": [],
                "suggestions": []
            }
            
            query_upper = query.upper().strip()
            
            # Basic syntax validation
            if not query.strip():
                validation_results["is_valid"] = False
                validation_results["syntax_issues"].append("Empty query")
                return validation_results
            
            # Check for SQL injection patterns
            injection_patterns = [
                r"';.*--",  # SQL injection with comment
                r"UNION\s+SELECT",  # UNION-based injection
                r"DROP\s+TABLE",  # Dangerous DROP statements
                r"DELETE\s+FROM.*WHERE.*1\s*=\s*1",  # Dangerous DELETE
                r"UPDATE.*SET.*WHERE.*1\s*=\s*1",  # Dangerous UPDATE
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, query_upper):
                    validation_results["security_issues"].append(f"Potential SQL injection pattern: {pattern}")
                    validation_results["is_valid"] = False
            
            # Check for missing semicolon (warning)
            if not query.strip().endswith(';'):
                validation_results["warnings"].append("Query should end with semicolon")
            
            # Check for SELECT without WHERE clause
            if "SELECT" in query_upper and "FROM" in query_upper and "WHERE" not in query_upper:
                if "LIMIT" not in query_upper and "TOP" not in query_upper:
                    validation_results["warnings"].append("SELECT without WHERE clause may return large result set")
            
            # Performance suggestions
            if "SELECT *" in query_upper:
                validation_results["suggestions"].append("Consider specifying column names instead of SELECT *")
            
            # Database-specific validations
            if database_type.lower() == "mysql":
                if "LIMIT" in query_upper and "OFFSET" not in query_upper:
                    validation_results["suggestions"].append("Consider using OFFSET with LIMIT for pagination")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return {"error": str(e)}

    @a2a_skill("optimize_query", "Provide query optimization suggestions")
    async def optimize_query(self, query: str, schema_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze query and provide optimization suggestions"""
        try:
            optimization_results = {
                "original_query": query,
                "suggestions": [],
                "estimated_improvement": "unknown",
                "optimized_query": None
            }
            
            query_upper = query.upper().strip()
            
            # Index recommendations
            if "WHERE" in query_upper:
                where_match = re.search(r'WHERE\s+(\w+)', query_upper)
                if where_match:
                    column = where_match.group(1)
                    optimization_results["suggestions"].append(
                        f"Consider creating an index on column '{column}' to improve WHERE clause performance"
                    )
            
            # JOIN optimization
            if "JOIN" in query_upper:
                optimization_results["suggestions"].append(
                    "Ensure JOIN conditions use indexed columns for better performance"
                )
            
            # ORDER BY optimization
            if "ORDER BY" in query_upper:
                optimization_results["suggestions"].append(
                    "Consider creating a composite index for ORDER BY columns"
                )
            
            # DISTINCT optimization
            if "DISTINCT" in query_upper:
                optimization_results["suggestions"].append(
                    "DISTINCT can be expensive; consider if it's necessary or if GROUP BY would be more appropriate"
                )
            
            # Subquery optimization
            if "SELECT" in query_upper and query_upper.count("SELECT") > 1:
                optimization_results["suggestions"].append(
                    "Consider rewriting subqueries as JOINs where possible for better performance"
                )
            
            # Estimate improvement potential
            suggestion_count = len(optimization_results["suggestions"])
            if suggestion_count == 0:
                optimization_results["estimated_improvement"] = "minimal"
            elif suggestion_count <= 2:
                optimization_results["estimated_improvement"] = "low"
            elif suggestion_count <= 4:
                optimization_results["estimated_improvement"] = "medium"
            else:
                optimization_results["estimated_improvement"] = "high"
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return {"error": str(e)}

    @a2a_skill("analyze_schema", "Analyze database schema and relationships")
    async def analyze_schema(self, schema_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database schema for optimization opportunities"""
        try:
            analysis_results = {
                "tables": {},
                "relationships": [],
                "index_recommendations": [],
                "normalization_suggestions": []
            }
            
            tables = schema_definition.get("tables", {})
            
            for table_name, table_info in tables.items():
                columns = table_info.get("columns", {})
                
                # Analyze table structure
                table_analysis = {
                    "column_count": len(columns),
                    "primary_keys": [],
                    "foreign_keys": [],
                    "nullable_columns": [],
                    "data_types": {}
                }
                
                for col_name, col_info in columns.items():
                    data_type = col_info.get("type", "unknown")
                    is_nullable = col_info.get("nullable", True)
                    is_primary = col_info.get("primary_key", False)
                    is_foreign = col_info.get("foreign_key", False)
                    
                    if is_primary:
                        table_analysis["primary_keys"].append(col_name)
                    if is_foreign:
                        table_analysis["foreign_keys"].append(col_name)
                    if is_nullable:
                        table_analysis["nullable_columns"].append(col_name)
                    
                    table_analysis["data_types"][col_name] = data_type
                
                analysis_results["tables"][table_name] = table_analysis
                
                # Generate index recommendations
                if len(table_analysis["foreign_keys"]) > 0:
                    for fk in table_analysis["foreign_keys"]:
                        analysis_results["index_recommendations"].append(
                            f"Consider adding index on foreign key '{fk}' in table '{table_name}'"
                        )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing schema: {e}")
            return {"error": str(e)}
    
    async def _process_sql_operations(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process SQL operations request asynchronously"""
        try:
            operation_results = {}
            sql_operations = request.get('operations', {})

            # Process each SQL operation
            for op_name, op_data in sql_operations.items():
                op_type = op_data.get('type', 'validate')
                logger.info("Processing %s SQL operation: %s", op_type, op_name)

                try:
                    if op_type == "generate":
                        result = await self.generate_query(
                            op_data.get('description', ''),
                            op_data.get('schema_context', {})
                        )
                    elif op_type == "validate":
                        result = await self.validate_query(
                            op_data.get('query', ''),
                            op_data.get('database_type', 'postgresql')
                        )
                    elif op_type == "optimize":
                        result = await self.optimize_query(
                            op_data.get('query', ''),
                            op_data.get('schema_info', {})
                        )
                    elif op_type == "analyze_schema":
                        result = await self.analyze_schema(
                            op_data.get('schema_definition', {})
                        )
                    else:
                        result = {"error": f"Unsupported SQL operation type: {op_type}"}
                    
                    operation_results[op_name] = result
                    
                    # Update stats
                    if "error" not in result:
                        self.sql_stats["successful_queries"] += 1
                        if op_type == "optimize":
                            self.sql_stats["optimization_suggestions"] += len(result.get("suggestions", []))
                    else:
                        self.sql_stats["failed_queries"] += 1
                    
                    # Track operation type usage
                    self.sql_stats["query_types"][op_type] = \
                        self.sql_stats["query_types"].get(op_type, 0) + 1
                        
                except Exception as e:
                    operation_results[op_name] = {"error": str(e)}
                    self.sql_stats["failed_queries"] += 1

            # Update overall stats
            self.sql_stats["total_queries"] += 1

            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(operation_results, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "sql_operations": list(operation_results.keys()),
                "successful_operations": sum(1 for r in operation_results.values() if "error" not in r),
                "failed_operations": sum(1 for r in operation_results.values() if "error" in r)
            })

        except Exception as e:
            logger.error("Error processing SQL operations: %s", e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send SQL results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "sql_results": data,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sql_stats": self.sql_stats
            }

            logger.info("Sent SQL results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_sql_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract SQL request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('sql_operations', content.get('operations', None))
        return None
    
    async def _load_schema_registry(self):
        """Load schema registry from configuration"""
        # This would typically load from a database or configuration file
        self.schema_registry = {
            "tables": {
                "users": {"columns": ["id", "name", "email", "created_at"]},
                "orders": {"columns": ["id", "user_id", "total", "order_date"]},
                "products": {"columns": ["id", "name", "price", "category"]}
            },
            "relationships": [
                {"from": "orders.user_id", "to": "users.id"},
                {"from": "order_items.order_id", "to": "orders.id"}
            ],
            "indexes": {}
        }
        logger.info("Loaded schema registry with %d tables", len(self.schema_registry["tables"]))
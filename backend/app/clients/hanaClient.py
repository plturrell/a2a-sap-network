"""
SAP HANA Cloud Production Client
Production-ready client for SAP HANA Cloud database integration
"""

import os
import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager, asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv

from app.core.databaseSecurityManager import DatabaseSecurityManager, SecurityLevel

try:
    from hdbcli import dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    dbapi = None

load_dotenv()
logger = logging.getLogger(__name__)


def validate_sql_identifier(identifier: str) -> str:
    """
    Validate SQL identifier to prevent injection attacks.
    SAP HANA identifiers must follow specific rules.
    """
    if not identifier:
        raise ValueError("SQL identifier cannot be empty")
    
    # HANA identifier rules: can contain letters, digits, underscores, and special characters
    # Must start with letter or underscore
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}. Must start with letter/underscore and contain only alphanumeric/underscore characters")
    
    # Convert to uppercase for HANA standard
    return identifier.upper()


def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Retry decorator for HANA operations with exponential backoff.
    SAP enterprise standard for connection resilience.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"HANA operation failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s: {e}")
                        asyncio.sleep(wait_time) if asyncio.iscoroutinefunction(func) else __import__('time').sleep(wait_time)
                    else:
                        logger.error(f"HANA operation failed after {max_attempts} attempts: {e}")
                        
            raise last_exception
        return wrapper
    return decorator


class HanaPerformanceMonitor:
    """Enterprise HANA performance monitoring and alerting"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.performance_logger = logging.getLogger(f"{__name__}.performance")
        self.slow_query_threshold = 5.0  # seconds
        self.connection_threshold = 2.0  # seconds
        
    def log_query_performance(self, query: str, execution_time: float, 
                            row_count: int = 0, user_id: Optional[str] = None):
        """Log query performance metrics"""
        if not self.enabled:
            return
            
        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_type": self._classify_query(query),
            "execution_time": execution_time,
            "row_count": row_count,
            "user_id": user_id or "system",
            "database_type": "hana",
            "slow_query": execution_time > self.slow_query_threshold
        }
        
        if execution_time > self.slow_query_threshold:
            # Truncate query for logging (security)
            query_preview = query[:100] + "..." if len(query) > 100 else query
            performance_record["query_preview"] = query_preview
            self.performance_logger.warning(f"SLOW_QUERY: {json.dumps(performance_record)}")
        else:
            self.performance_logger.info(f"QUERY_PERF: {json.dumps(performance_record)}")
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for monitoring"""
        query_upper = query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        elif query_upper.startswith('CALL'):
            return 'PROCEDURE'
        elif any(keyword in query_upper for keyword in ['CREATE', 'ALTER', 'DROP']):
            return 'DDL'
        else:
            return 'OTHER'
    
    def log_connection_performance(self, connection_time: float):
        """Log connection performance"""
        if not self.enabled:
            return
            
        if connection_time > self.connection_threshold:
            self.performance_logger.warning(f"SLOW_CONNECTION: {connection_time:.2f}s")
        
    def suggest_optimization(self, query: str, execution_time: float) -> List[str]:
        """Suggest query optimizations based on performance analysis"""
        suggestions = []
        
        if execution_time > self.slow_query_threshold:
            query_upper = query.upper()
            
            if 'SELECT' in query_upper and 'WHERE' not in query_upper:
                suggestions.append("Consider adding WHERE clause to limit result set")
            
            if 'ORDER BY' in query_upper and 'LIMIT' not in query_upper:
                suggestions.append("Consider adding LIMIT clause with ORDER BY")
            
            if 'JOIN' in query_upper and 'INDEX' not in query_upper:
                suggestions.append("Ensure proper indexes exist on JOIN columns")
            
            if query_upper.count('SELECT') > 1:
                suggestions.append("Consider optimizing subqueries or using WITH clauses")
        
        return suggestions


@dataclass
class HanaConfig:
    """Configuration for HANA client"""
    address: str
    port: int
    user: str
    password: str
    encrypt: bool = True
    ssl_validate_certificate: bool = True
    auto_commit: bool = True
    timeout: int = 30
    pool_size: int = 10


@dataclass
class QueryResult:
    """Structured result from HANA query"""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: Optional[float] = None
    raw_result: Optional[Any] = None


class HanaConnectionPool:
    """Connection pool for HANA database connections"""
    
    def __init__(self, config: HanaConfig):
        if not HANA_AVAILABLE:
            raise ImportError("SAP HANA client not available. Install with: pip install hdbcli")
        
        self.config = config
        self._pool = []
        self._lock = threading.Lock()
        self._initialized = False
    
    def _create_connection(self):
        """Create a new HANA database connection"""
        return dbapi.connect(
            address=self.config.address,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            encrypt=self.config.encrypt,
            sslValidateCertificate=self.config.ssl_validate_certificate,
            autocommit=self.config.auto_commit
        )
    
    def get_connection(self):
        """Get a connection from the pool with health check"""
        with self._lock:
            if self._pool:
                connection = self._pool.pop()
                # Basic health check - test if connection is still alive
                try:
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1 FROM SYS.DUMMY")
                    cursor.fetchone()
                    cursor.close()
                    return connection
                except:
                    # Connection is stale, create a new one
                    try:
                        connection.close()
                    except:
                        pass
                    return self._create_connection()
            else:
                return self._create_connection()
    
    def return_connection(self, connection):
        """Return a connection to the pool"""
        with self._lock:
            if len(self._pool) < self.config.pool_size:
                self._pool.append(connection)
            else:
                connection.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except:
                    pass
            self._pool.clear()


class HanaClient:
    """Production-ready SAP HANA Cloud client for A2A agents"""
    
    def __init__(self, config: Optional[HanaConfig] = None, enable_security: bool = True):
        """Initialize HANA client with configuration"""
        if not HANA_AVAILABLE:
            raise ImportError("SAP HANA client not available. Install with: pip install hdbcli")
        
        if config is None:
            config = HanaConfig(
                address=os.getenv('HANA_HOSTNAME'),
                port=int(os.getenv('HANA_PORT', 443)),
                user=os.getenv('HANA_USERNAME'),
                password=os.getenv('HANA_PASSWORD')
            )
        
        if not all([config.address, config.port, config.user, config.password]):
            raise ValueError("HANA connection parameters are required")
        
        self.config = config
        
        # Initialize enterprise performance monitoring
        self.performance_monitor = HanaPerformanceMonitor(enabled=True)
        self.pool = HanaConnectionPool(config)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize enterprise security management
        self.security_manager = DatabaseSecurityManager("hana") if enable_security else None
        
        logger.info(f"HANA client initialized for {config.address}:{config.port}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            connection = self.pool.get_connection()
            yield connection
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Union[List, Tuple, Dict]] = None,
        fetch_results: bool = True,
        user_id: Optional[str] = None,
        table_name: Optional[str] = None,
        data_security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> QueryResult:
        """Execute a SQL query synchronously with enterprise monitoring and security"""
        try:
            # Enterprise security check
            if self.security_manager and user_id and table_name:
                operation = self._extract_operation_from_query(query)
                if not self.security_manager.check_permission(user_id, operation, table_name, data_security_level):
                    raise PermissionError(f"User {user_id} does not have {operation} permission on {table_name}")
            
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                import time
                start_time = time.time()
                
                if parameters:
                    cursor.execute(query, parameters)
                else:
                    cursor.execute(query)
                
                execution_time = time.time() - start_time
                
                if fetch_results and cursor.description:
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Fetch all results
                    rows = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    data = [dict(zip(columns, row)) for row in rows]
                    
                    result = QueryResult(
                        data=data,
                        columns=columns,
                        row_count=len(data),
                        execution_time=execution_time
                    )
                else:
                    result = QueryResult(
                        data=[],
                        columns=[],
                        row_count=cursor.rowcount if hasattr(cursor, 'rowcount') else 0,
                        execution_time=execution_time
                    )
                
                cursor.close()
                
                # Enterprise performance monitoring
                row_count = len(result.data) if result.data else result.row_count
                self.performance_monitor.log_query_performance(
                    query=query,
                    execution_time=execution_time,
                    row_count=row_count,
                    user_id=user_id
                )
                
                # Generate optimization suggestions for slow queries
                if execution_time > self.performance_monitor.slow_query_threshold:
                    suggestions = self.performance_monitor.suggest_optimization(query, execution_time)
                    if suggestions:
                        logger.warning(f"Query optimization suggestions: {', '.join(suggestions)}")
                
                return result
        
        except Exception as e:
            logger.error(f"HANA query execution error: {e}")
            raise
    
    async def execute_query_async(
        self,
        query: str,
        parameters: Optional[Union[List, Tuple, Dict]] = None,
        fetch_results: bool = True
    ) -> QueryResult:
        """Execute a SQL query asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.execute_query,
            query,
            parameters,
            fetch_results
        )
    
    def execute_batch(
        self,
        queries: List[Tuple[str, Optional[Union[List, Tuple, Dict]]]]
    ) -> List[QueryResult]:
        """Execute multiple queries in a batch"""
        results = []
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                for query, parameters in queries:
                    import time
                    start_time = time.time()
                    
                    if parameters:
                        cursor.execute(query, parameters)
                    else:
                        cursor.execute(query)
                    
                    execution_time = time.time() - start_time
                    
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()
                        data = [dict(zip(columns, row)) for row in rows]
                        
                        result = QueryResult(
                            data=data,
                            columns=columns,
                            row_count=len(data),
                            execution_time=execution_time
                        )
                    else:
                        result = QueryResult(
                            data=[],
                            columns=[],
                            row_count=cursor.rowcount if hasattr(cursor, 'rowcount') else 0,
                            execution_time=execution_time
                        )
                    
                    results.append(result)
                
                cursor.close()
                return results
        
        except Exception as e:
            logger.error(f"HANA batch execution error: {e}")
            raise
    
    def get_table_info(self, table_name: str, schema: str = None) -> QueryResult:
        """Get information about a table"""
        if schema:
            query = """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, IS_NULLABLE, DEFAULT_VALUE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            ORDER BY POSITION
            """
            parameters = (schema.upper(), table_name.upper())
        else:
            query = """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, IS_NULLABLE, DEFAULT_VALUE
            FROM SYS.TABLE_COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY POSITION
            """
            parameters = (table_name.upper(),)
        
        return self.execute_query(query, parameters)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get HANA system information"""
        queries = [
            ("SELECT VERSION FROM SYS.M_DATABASE", None),
            ("SELECT CURRENT_TIMESTAMP FROM SYS.DUMMY", None),
            ("SELECT COUNT(*) as TOTAL_TABLES FROM SYS.TABLES", None),
            ("SELECT COUNT(*) as TOTAL_SCHEMAS FROM SYS.SCHEMAS", None)
        ]
        
        results = self.execute_batch(queries)
        
        return {
            "version": results[0].data[0]["VERSION"] if results[0].data else "Unknown",
            "current_time": str(results[1].data[0]["CURRENT_TIMESTAMP"]) if results[1].data else "Unknown",
            "total_tables": results[2].data[0]["TOTAL_TABLES"] if results[2].data else 0,
            "total_schemas": results[3].data[0]["TOTAL_SCHEMAS"] if results[3].data else 0
        }
    
    def insert_financial_data(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        schema: str = None
    ) -> QueryResult:
        """Insert financial data into a table"""
        if not data:
            raise ValueError("No data provided for insertion")
        
        # Get column names from first record
        columns = list(data[0].keys())
        column_names = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        
        # Validate table and schema names for security
        validated_table = validate_sql_identifier(table_name)
        if schema:
            validated_schema = validate_sql_identifier(schema)
            full_table_name = f"{validated_schema}.{validated_table}"
        else:
            full_table_name = validated_table
            
        query = f"INSERT INTO {full_table_name} ({column_names}) VALUES ({placeholders})"
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                import time
                start_time = time.time()
                
                for record in data:
                    values = [record[col] for col in columns]
                    cursor.execute(query, values)
                
                execution_time = time.time() - start_time
                cursor.close()
                
                return QueryResult(
                    data=[],
                    columns=[],
                    row_count=len(data),
                    execution_time=execution_time
                )
        
        except Exception as e:
            logger.error(f"HANA data insertion error: {e}")
            raise
    
    async def process_a2a_data_request(
        self,
        request_type: str,
        query_params: Dict[str, Any]
    ) -> QueryResult:
        """Process A2A agent data requests"""
        if request_type == "financial_summary":
            query = """
            SELECT 
                ACCOUNT_TYPE,
                COUNT(*) as RECORD_COUNT,
                SUM(AMOUNT) as TOTAL_AMOUNT
            FROM FINANCIAL_DATA 
            WHERE DATE >= ?
            GROUP BY ACCOUNT_TYPE
            """
            parameters = (query_params.get('start_date'),)
        
        elif request_type == "entity_lookup":
            query = """
            SELECT * FROM ENTITY_MASTER 
            WHERE ENTITY_NAME LIKE ?
            LIMIT 100
            """
            parameters = (f"%{query_params.get('entity_name', '')}%",)
        
        else:
            raise ValueError(f"Unsupported A2A request type: {request_type}")
        
        return await self.execute_query_async(query, parameters)
    
    def _extract_operation_from_query(self, query: str) -> str:
        """Extract operation type from SQL query for security validation"""
        query_upper = query.upper().strip()
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        elif any(keyword in query_upper for keyword in ['CREATE', 'ALTER', 'DROP']):
            return 'CREATE'
        else:
            return 'SELECT'  # Default to most restrictive
    
    def create_user(self, username: str, roles: List[str], security_clearance: str, 
                   department: Optional[str] = None) -> Dict[str, Any]:
        """Create database user with enterprise security roles"""
        if not self.security_manager:
            raise RuntimeError("Security manager not enabled")
        
        from app.core.databaseSecurityManager import DatabaseRole, SecurityLevel
        
        # Convert string roles and security level to enums
        role_enums = [DatabaseRole(role) for role in roles]
        clearance_enum = SecurityLevel(security_clearance)
        
        user = self.security_manager.create_user(username, role_enums, clearance_enum, department)
        
        return {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "security_clearance": user.security_clearance.value,
            "created_at": user.created_at.isoformat()
        }
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report"""
        if not self.security_manager:
            return {"error": "Security manager not enabled"}
        
        return self.security_manager.get_security_report()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the HANA client"""
        try:
            result = self.execute_query("SELECT 1 as STATUS FROM SYS.DUMMY")
            system_info = self.get_system_info()
            
            health_report = {
                "status": "healthy",
                "connection": "active",
                "response_data": result.data[0] if result.data else None,
                "system_info": system_info,
                "execution_time": result.execution_time
            }
            
            # Add security status if enabled
            if self.security_manager:
                security_report = self.security_manager.get_security_report()
                health_report["security"] = {
                    "enabled": True,
                    "total_users": security_report["total_users"],
                    "active_users": security_report["active_users"]
                }
            else:
                health_report["security"] = {"enabled": False}
            
            return health_report
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close all connections and cleanup resources"""
        self.pool.close_all()
        self.executor.shutdown(wait=True)
        logger.info("HANA client connections closed")


# Factory function for easy instantiation
def create_hana_client(config: Optional[HanaConfig] = None) -> HanaClient:
    """Factory function to create a HANA client"""
    return HanaClient(config)


# Singleton instance for global use
_hana_client_instance: Optional[HanaClient] = None

def get_hana_client() -> HanaClient:
    """Get singleton HANA client instance"""
    global _hana_client_instance
    
    if _hana_client_instance is None:
        _hana_client_instance = create_hana_client()
    
    return _hana_client_instance

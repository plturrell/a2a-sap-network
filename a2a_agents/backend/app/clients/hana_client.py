"""
SAP HANA Cloud Production Client
Production-ready client for SAP HANA Cloud database integration
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager, asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

try:
    from hdbcli import dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    dbapi = None

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class HanaConfig:
    """Configuration for HANA client"""
    address: str
    port: int
    user: str
    password: str
    encrypt: bool = True
    ssl_validate_certificate: bool = False
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
    
    def __init__(self, config: Optional[HanaConfig] = None):
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
        self.pool = HanaConnectionPool(config)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
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
        fetch_results: bool = True
    ) -> QueryResult:
        """Execute a SQL query synchronously"""
        try:
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
        
        full_table_name = f"{schema}.{table_name}" if schema else table_name
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
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the HANA client"""
        try:
            result = self.execute_query("SELECT 1 as STATUS FROM SYS.DUMMY")
            system_info = self.get_system_info()
            
            return {
                "status": "healthy",
                "connection": "active",
                "response_data": result.data[0] if result.data else None,
                "system_info": system_info,
                "execution_time": result.execution_time
            }
        
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

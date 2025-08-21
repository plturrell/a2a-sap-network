"""
SQLite Production Client
Production-ready client for SQLite integration as fallback database
"""

import os
import asyncio
import logging
import sqlite3
import json
import queue
import threading
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import aiosqlite

# Enterprise security imports
try:
    import sqlcipher3
    SQLCIPHER_AVAILABLE = True
except ImportError:
    sqlcipher3 = None
    SQLCIPHER_AVAILABLE = False

from app.core.databaseSecurityManager import DatabaseSecurityManager, SecurityLevel
from app.core.connection_pool import pool_manager
from app.core.cache_manager import cache_manager, cached
from app.core.pagination import PaginationParams, PaginatedResponse, paginate_query

logger = logging.getLogger(__name__)


@dataclass
class SQLiteConfig:
    """Enterprise SQLite client configuration"""
    db_path: str = "./data/a2a_fallback.db"
    check_same_thread: bool = False
    timeout: float = 30.0
    isolation_level: Optional[str] = None
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"  # Write-Ahead Logging for better concurrency
    
    # Enterprise security features
    enable_encryption: bool = True
    encryption_key: Optional[str] = None  # Will be generated if None
    key_derivation_iterations: int = 64000  # PBKDF2 iterations
    
    # Performance optimization
    enable_connection_pooling: bool = True
    pool_size: int = 5
    cache_size: int = 10000  # SQLite page cache size
    mmap_size: int = 268435456  # 256MB memory-mapped I/O
    
    # Enterprise monitoring
    enable_audit_logging: bool = True
    log_all_queries: bool = False  # Only in development
    performance_monitoring: bool = True


@dataclass
class SQLiteResponse:
    """Structured response from SQLite operations"""
    data: Union[List[Dict[str, Any]], Dict[str, Any], None]
    count: Optional[int]
    status_code: int
    error: Optional[Dict[str, Any]]
    raw_response: Optional[Any] = None
    execution_time: Optional[float] = None


class SQLiteSecurityAuditor:
    """Enterprise security audit logging for SQLite operations"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.audit_logger = logging.getLogger(f"{__name__}.security_audit")
    
    def log_operation(self, operation: str, table: str, user_id: Optional[str] = None, 
                     sensitive_data: bool = False, execution_time: Optional[float] = None):
        """Log database operations for security audit"""
        if not self.enabled:
            return
            
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "table": table,
            "user_id": user_id or "system",
            "sensitive_data": sensitive_data,
            "execution_time_ms": execution_time * 1000 if execution_time else None,
            "database_type": "sqlite"
        }
        
        self.audit_logger.info(f"DB_AUDIT: {json.dumps(audit_record)}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        if not self.enabled:
            return
            
        security_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "database_type": "sqlite"
        }
        
        self.audit_logger.warning(f"DB_SECURITY: {json.dumps(security_record)}")


class SQLiteConnectionPool:
    """Thread-safe connection pool for SQLite with encryption support"""
    
    def __init__(self, config: SQLiteConfig):
        self.config = config
        self.pool = queue.Queue(maxsize=config.pool_size)
        self.lock = threading.Lock()
        self._async_pool = None
        self._pool_name = f"sqlite_{os.path.basename(config.db_path)}"
        self._initialize_pool()
        # Initialize async pool in background
        asyncio.create_task(self._initialize_async_pool())
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.config.pool_size):
            conn = self._create_connection()
            self.pool.put(conn)
    
    async def _initialize_async_pool(self):
        """Initialize async connection pool using global pool manager"""
        try:
            self._async_pool = await pool_manager.create_pool(
                name=self._pool_name,
                db_type="sqlite",
                connection_factory=self._create_async_connection,
                min_size=max(1, self.config.pool_size // 2),
                max_size=self.config.pool_size * 2,
                max_age_seconds=3600,
                max_idle_seconds=300
            )
            logger.info(f"Initialized async connection pool '{self._pool_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")
    
    async def _create_async_connection(self):
        """Create async SQLite connection"""
        db_path = self.config.db_path
        conn = await aiosqlite.connect(db_path)
        
        # Configure connection
        await conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
        if self.config.enable_foreign_keys:
            await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
        
        return conn
    
    def _create_connection(self):
        """Create a new database connection with enterprise features"""
        try:
            # Use SQLCipher for encryption if available and enabled
            if self.config.enable_encryption and SQLCIPHER_AVAILABLE:
                conn = sqlcipher3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread
                )
                
                # Set encryption key
                encryption_key = self._get_encryption_key()
                conn.execute(f"PRAGMA key = '{encryption_key}'")
                conn.execute(f"PRAGMA kdf_iter = {self.config.key_derivation_iterations}")
                
            else:
                # Fall back to standard SQLite
                if self.config.enable_encryption:
                    logger.warning("⚠️ SQLite encryption requested but SQLCipher not available")
                
                conn = sqlite3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread,
                    isolation_level=self.config.isolation_level
                )
            
            # Configure enterprise performance settings
            self._configure_connection(conn)
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create SQLite connection: {e}")
            raise
    
    def _get_encryption_key(self) -> str:
        """Get or generate encryption key securely"""
        if self.config.encryption_key:
            return self.config.encryption_key
        
        # Generate key from environment or create secure random key
        key_source = os.getenv('SQLITE_ENCRYPTION_KEY')
        if key_source:
            # Derive key from password using PBKDF2
            salt = hashlib.sha256(self.config.db_path.encode()).digest()[:16]
            key = hashlib.pbkdf2_hmac('sha256', key_source.encode(), salt, self.config.key_derivation_iterations)
            return key.hex()
        else:
            logger.warning("⚠️ No encryption key configured, using generated key (not persistent)")
            return secrets.token_hex(32)
    
    def _configure_connection(self, conn):
        """Configure connection with enterprise optimization settings"""
        cursor = conn.cursor()
        
        # Enable foreign keys for data integrity
        if self.config.enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON")
        
        # Set journal mode for concurrency
        cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
        
        # Performance optimizations
        cursor.execute(f"PRAGMA cache_size = {self.config.cache_size}")
        cursor.execute(f"PRAGMA mmap_size = {self.config.mmap_size}")
        cursor.execute("PRAGMA temp_store = memory")
        cursor.execute("PRAGMA synchronous = NORMAL")  # Balance safety/performance
        
        cursor.close()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return"""
        conn = None
        try:
            with self.lock:
                conn = self.pool.get(timeout=self.config.timeout)
            yield conn
        finally:
            if conn:
                with self.lock:
                    if not self.pool.full():
                        self.pool.put(conn)
                    else:
                        conn.close()
    
    async def get_async_connection(self):
        """Get async connection from enhanced pool"""
        if not self._async_pool:
            # Fallback to creating connection directly
            return await self._create_async_connection()
        
        return self._async_pool.acquire()


class SQLiteClient:
    """Production-ready SQLite client for A2A agents as fallback to HANA"""
    
    def __init__(self, config: Optional[SQLiteConfig] = None, enable_security: bool = True):
        """Initialize enterprise SQLite client with security and performance features"""
        if config is None:
            config = SQLiteConfig(
                db_path=os.getenv('SQLITE_DB_PATH', './data/a2a_fallback.db')
            )
        
        self.config = config
        
        # Initialize enterprise components
        self.security_auditor = SQLiteSecurityAuditor(enabled=config.enable_audit_logging)
        
        # Initialize enterprise security management
        self.security_manager = DatabaseSecurityManager("sqlite") if enable_security else None
        
        # Initialize connection pool if enabled
        if config.enable_connection_pooling:
            self.connection_pool = SQLiteConnectionPool(config)
            logger.info("✅ SQLite connection pooling enabled")
        else:
            self.connection_pool = None
        
        # Ensure data directory exists
        db_dir = Path(config.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database with security validation
        self._init_database()
        
        # Log initialization with security status
        encryption_status = "🔒 ENCRYPTED" if (config.enable_encryption and SQLCIPHER_AVAILABLE) else "⚠️ UNENCRYPTED"
        logger.info(f"✅ Enterprise SQLite client initialized at {config.db_path} - {encryption_status}")
        
        # Log security audit event
        self.security_auditor.log_security_event("database_initialized", {
            "path": config.db_path,
            "encryption_enabled": config.enable_encryption and SQLCIPHER_AVAILABLE,
            "pooling_enabled": config.enable_connection_pooling,
            "audit_enabled": config.enable_audit_logging
        })
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign keys
            if self.config.enable_foreign_keys:
                cursor.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode for better concurrency
            cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            
            # Create tables for A2A agents
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_data (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_data_agent_id ON agent_data(agent_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_data_type ON agent_data(data_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_data_created_at ON agent_data(created_at)
            """)
            
            # Create table for agent interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create table for financial data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_data (
                    id TEXT PRIMARY KEY,
                    data_source TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    record_data TEXT NOT NULL,
                    validation_status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create ORD registry tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ord_registrations (
                    registration_id TEXT PRIMARY KEY,
                    ord_document TEXT,
                    registered_by TEXT,
                    registered_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    version TEXT,
                    status TEXT,
                    validation_result TEXT,
                    governance_info TEXT,
                    analytics_info TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ord_resource_index (
                    ord_id TEXT PRIMARY KEY,
                    registration_id TEXT,
                    resource_type TEXT,
                    title TEXT,
                    description TEXT,
                    short_description TEXT,
                    version TEXT,
                    tags TEXT,
                    labels TEXT,
                    domain TEXT,
                    category TEXT,
                    indexed_at TIMESTAMP,
                    searchable_content TEXT,
                    access_strategies TEXT,
                    dublin_core TEXT,
                    dc_creator TEXT,
                    dc_subject TEXT,
                    dc_publisher TEXT,
                    dc_format TEXT,
                    FOREIGN KEY (registration_id) REFERENCES ord_registrations(registration_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ord_replication_log (
                    id TEXT PRIMARY KEY,
                    table_name TEXT,
                    operation TEXT,
                    record_id TEXT,
                    timestamp TIMESTAMP,
                    status TEXT,
                    error_message TEXT
                )
            """)
            
            conn.commit()
            logger.info("SQLite database tables initialized")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with enterprise features (pooling, encryption, monitoring)"""
        import time
        start_time = time.time()
        
        if self.connection_pool:
            # Use enterprise connection pool
            with self.connection_pool.get_connection() as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                yield conn
        else:
            # Fall back to direct connection
            if self.config.enable_encryption and SQLCIPHER_AVAILABLE:
                conn = sqlcipher3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread
                )
                # Set encryption key
                encryption_key = self._get_fallback_encryption_key()
                conn.execute(f"PRAGMA key = '{encryption_key}'")
            else:
                conn = sqlite3.connect(
                    self.config.db_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread,
                    isolation_level=self.config.isolation_level
                )
            
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                # Log performance if monitoring is enabled
                if self.config.performance_monitoring:
                    execution_time = time.time() - start_time
                    if execution_time > 1.0:  # Log slow connections
                        logger.warning(f"Slow SQLite connection: {execution_time:.2f}s")
                conn.close()
    
    def _get_fallback_encryption_key(self) -> str:
        """Get encryption key for non-pooled connections"""
        if self.connection_pool:
            return self.connection_pool._get_encryption_key()
        
        key_source = os.getenv('SQLITE_ENCRYPTION_KEY')
        if key_source:
            salt = hashlib.sha256(self.config.db_path.encode()).digest()[:16]
            key = hashlib.pbkdf2_hmac('sha256', key_source.encode(), salt, self.config.key_derivation_iterations)
            return key.hex()
        return secrets.token_hex(32)
    
    def _check_sensitive_data(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Check if data contains sensitive information for audit logging"""
        # Define sensitive table patterns and field patterns
        sensitive_tables = ['user', 'auth', 'credential', 'token', 'secret', 'key', 'password']
        sensitive_fields = ['password', 'token', 'secret', 'key', 'ssn', 'credit_card', 'api_key']
        
        # Check if table name suggests sensitive data
        if any(pattern in table.lower() for pattern in sensitive_tables):
            return True
        
        # Check if data contains sensitive field names
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data
        
        for record in data_list:
            if isinstance(record, dict):
                for field_name in record.keys():
                    if any(pattern in field_name.lower() for pattern in sensitive_fields):
                        return True
        
        return False
    
    async def _get_async_connection(self):
        """Get async database connection"""
        return await aiosqlite.connect(
            self.config.db_path,
            timeout=self.config.timeout,
            isolation_level=self.config.isolation_level
        )
    
    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        user_id: Optional[str] = None,
        data_security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> SQLiteResponse:
        """Select data from a table"""
        # Enterprise security check
        if self.security_manager and user_id:
            if not self.security_manager.check_permission(user_id, "SELECT", table, data_security_level):
                raise PermissionError(f"User {user_id} does not have SELECT permission on {table}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query - validate inputs to prevent SQL injection
                # Validate table name (should only contain alphanumeric and underscores)
                if not table.replace('_', '').isalnum():
                    raise ValueError(f"Invalid table name: {table}")
                
                # Validate column names if not wildcard
                if columns != "*":
                    # Split and validate each column name
                    column_list = [col.strip() for col in columns.split(',')]
                    for col in column_list:
                        if not col.replace('_', '').isalnum() and col != "*":
                            raise ValueError(f"Invalid column name: {col}")
                
                query = f"SELECT {columns} FROM {table}"
                params = []
                
                # Add filters
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if isinstance(value, dict):
                            # Handle complex filters
                            for op, val in value.items():
                                if op == "gte":
                                    conditions.append(f"{key} >= ?")
                                    params.append(val)
                                elif op == "lte":
                                    conditions.append(f"{key} <= ?")
                                    params.append(val)
                                elif op == "gt":
                                    conditions.append(f"{key} > ?")
                                    params.append(val)
                                elif op == "lt":
                                    conditions.append(f"{key} < ?")
                                    params.append(val)
                                elif op == "like":
                                    conditions.append(f"{key} LIKE ?")
                                    params.append(val)
                        else:
                            conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                # Add ordering - validate column name to prevent SQL injection
                if order_by:
                    ascending = not order_by.startswith('-')
                    column = order_by.lstrip('-')
                    # Validate column name
                    if not column.replace('_', '').isalnum():
                        raise ValueError(f"Invalid order column name: {column}")
                    query += f" ORDER BY {column} {'ASC' if ascending else 'DESC'}"
                
                # Add pagination - validate numeric values
                if limit:
                    if not isinstance(limit, int) or limit < 0:
                        raise ValueError(f"Invalid limit value: {limit}")
                    query += f" LIMIT {limit}"
                if offset:
                    if not isinstance(offset, int) or offset < 0:
                        raise ValueError(f"Invalid offset value: {offset}")
                    query += f" OFFSET {offset}"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries
                data = [dict(row) for row in rows]
                
                # Get count
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                if filters and conditions:
                    count_query += " WHERE " + " AND ".join(conditions)
                cursor.execute(count_query, params[:len(conditions)] if filters else [])
                count = cursor.fetchone()['count']
                
                return SQLiteResponse(
                    data=data,
                    count=count,
                    status_code=200,
                    error=None
                )
        
        except Exception as e:
            logger.error(f"SQLite select error: {e}")
            return SQLiteResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)}
            )
    
    def insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        upsert: bool = False,
        user_id: Optional[str] = None,
        data_security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> SQLiteResponse:
        """Insert data into a table with enterprise security audit logging"""
        import time
        start_time = time.time()
        
        # Enterprise security check
        if self.security_manager and user_id:
            if not self.security_manager.check_permission(user_id, "INSERT", table, data_security_level):
                raise PermissionError(f"User {user_id} does not have INSERT permission on {table}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Validate table name to prevent SQL injection
                if not table.replace('_', '').isalnum():
                    raise ValueError(f"Invalid table name: {table}")
                
                # Ensure data is a list
                if isinstance(data, dict):
                    data = [data]
                
                # Add UUID if not present
                for record in data:
                    if 'id' not in record:
                        import uuid
                        record['id'] = str(uuid.uuid4())
                
                # Get column names from first record and validate them
                columns = list(data[0].keys())
                for col in columns:
                    if not col.replace('_', '').isalnum():
                        raise ValueError(f"Invalid column name: {col}")
                
                placeholders = ','.join(['?' for _ in columns])
                
                if upsert:
                    # SQLite UPSERT syntax
                    query = f"""
                        INSERT INTO {table} ({','.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT(id) DO UPDATE SET
                        {','.join([f"{col}=excluded.{col}" for col in columns if col != 'id'])}
                    """
                else:
                    query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
                
                # Insert all records
                for record in data:
                    # Convert complex types to JSON
                    values = []
                    for col in columns:
                        val = record.get(col)
                        if isinstance(val, (dict, list)):
                            val = json.dumps(val)
                        elif isinstance(val, datetime):
                            val = val.isoformat()
                        values.append(val)
                    
                    cursor.execute(query, values)
                
                conn.commit()
                
                # Enterprise audit logging
                execution_time = time.time() - start_time
                sensitive_data = self._check_sensitive_data(table, data)
                self.security_auditor.log_operation(
                    operation="INSERT",
                    table=table,
                    user_id=user_id,
                    sensitive_data=sensitive_data,
                    execution_time=execution_time
                )
                
                return SQLiteResponse(
                    data=data,
                    count=len(data),
                    status_code=201,
                    error=None,
                    execution_time=execution_time
                )
        
        except Exception as e:
            logger.error(f"SQLite insert error: {e}")
            return SQLiteResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)}
            )
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        filters: Dict[str, Any],
        user_id: Optional[str] = None,
        data_security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> SQLiteResponse:
        """Update data in a table"""
        # Enterprise security check
        if self.security_manager and user_id:
            if not self.security_manager.check_permission(user_id, "UPDATE", table, data_security_level):
                raise PermissionError(f"User {user_id} does not have UPDATE permission on {table}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Validate table name to prevent SQL injection
                if not table.replace('_', '').isalnum():
                    raise ValueError(f"Invalid table name: {table}")
                
                # Build update query - validate column names
                set_clause = []
                set_params = []
                for key, value in data.items():
                    if not key.replace('_', '').isalnum():
                        raise ValueError(f"Invalid column name: {key}")
                    set_clause.append(f"{key} = ?")
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    elif isinstance(value, datetime):
                        value = value.isoformat()
                    set_params.append(value)
                
                # Build where clause - validate filter column names
                where_clause = []
                where_params = []
                for key, value in filters.items():
                    if not key.replace('_', '').isalnum():
                        raise ValueError(f"Invalid filter column name: {key}")
                    where_clause.append(f"{key} = ?")
                    where_params.append(value)
                
                query = f"""
                    UPDATE {table}
                    SET {','.join(set_clause)}, updated_at = CURRENT_TIMESTAMP
                    WHERE {' AND '.join(where_clause)}
                """
                
                cursor.execute(query, set_params + where_params)
                conn.commit()
                
                return SQLiteResponse(
                    data={"affected_rows": cursor.rowcount},
                    count=cursor.rowcount,
                    status_code=200,
                    error=None
                )
        
        except Exception as e:
            logger.error(f"SQLite update error: {e}")
            return SQLiteResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)}
            )
    
    def delete(
        self,
        table: str,
        filters: Dict[str, Any],
        user_id: Optional[str] = None,
        data_security_level: SecurityLevel = SecurityLevel.PUBLIC
    ) -> SQLiteResponse:
        """Delete data from a table"""
        # Enterprise security check
        if self.security_manager and user_id:
            if not self.security_manager.check_permission(user_id, "DELETE", table, data_security_level):
                raise PermissionError(f"User {user_id} does not have DELETE permission on {table}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Validate table name to prevent SQL injection
                if not table.replace('_', '').isalnum():
                    raise ValueError(f"Invalid table name: {table}")
                
                # Build where clause - validate filter column names
                where_clause = []
                params = []
                for key, value in filters.items():
                    if not key.replace('_', '').isalnum():
                        raise ValueError(f"Invalid filter column name: {key}")
                    where_clause.append(f"{key} = ?")
                    params.append(value)
                
                query = f"DELETE FROM {table} WHERE {' AND '.join(where_clause)}"
                
                cursor.execute(query, params)
                conn.commit()
                
                return SQLiteResponse(
                    data={"affected_rows": cursor.rowcount},
                    count=cursor.rowcount,
                    status_code=200,
                    error=None
                )
        
        except Exception as e:
            logger.error(f"SQLite delete error: {e}")
            return SQLiteResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)}
            )
    
    # A2A Specific Operations
    def store_agent_data(
        self,
        agent_id: str,
        data_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        table_name: str = "agent_data"
    ) -> SQLiteResponse:
        """Store A2A agent data with metadata"""
        record = {
            "agent_id": agent_id,
            "data_type": data_type,
            "data": data,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return self.insert(table_name, record)
    
    def get_agent_data(
        self,
        agent_id: str,
        data_type: Optional[str] = None,
        limit: int = 100
    ) -> SQLiteResponse:
        """Retrieve A2A agent data"""
        filters = {"agent_id": agent_id}
        if data_type:
            filters["data_type"] = data_type
        
        result = self.select(
            table="agent_data",
            filters=filters,
            limit=limit,
            order_by="-created_at"
        )
        
        # Parse JSON data
        if result.data:
            for row in result.data:
                if row.get('data') and isinstance(row['data'], str):
                    try:
                        row['data'] = json.loads(row['data'])
                    except:
                        pass
                if row.get('metadata') and isinstance(row['metadata'], str):
                    try:
                        row['metadata'] = json.loads(row['metadata'])
                    except:
                        pass
        
        return result
    
    def log_agent_interaction(
        self,
        agent_id: str,
        interaction_type: str,
        details: Dict[str, Any],
        success: bool = True
    ) -> SQLiteResponse:
        """Log A2A agent interactions"""
        import uuid
        record = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "interaction_type": interaction_type,
            "details": details,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self.insert("agent_interactions", record)
    
    def store_financial_data(
        self,
        data_source: str,
        data_type: str,
        records: List[Dict[str, Any]],
        validation_status: str = "pending"
    ) -> SQLiteResponse:
        """Store financial data with validation tracking"""
        import uuid
        processed_records = []
        
        for record in records:
            processed_record = {
                "id": str(uuid.uuid4()),
                "data_source": data_source,
                "data_type": data_type,
                "record_data": record,
                "validation_status": validation_status,
                "created_at": datetime.utcnow().isoformat()
            }
            processed_records.append(processed_record)
        
        return self.insert("financial_data", processed_records)
    
    def get_financial_data(
        self,
        data_source: Optional[str] = None,
        data_type: Optional[str] = None,
        validation_status: Optional[str] = None,
        limit: int = 1000
    ) -> SQLiteResponse:
        """Retrieve financial data with optional filters"""
        filters = {}
        if data_source:
            filters["data_source"] = data_source
        if data_type:
            filters["data_type"] = data_type
        if validation_status:
            filters["validation_status"] = validation_status
        
        result = self.select(
            table="financial_data",
            filters=filters,
            limit=limit,
            order_by="-created_at"
        )
        
        # Parse JSON data
        if result.data:
            for row in result.data:
                if row.get('record_data') and isinstance(row['record_data'], str):
                    try:
                        row['record_data'] = json.loads(row['record_data'])
                    except:
                        pass
        
        return result
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> SQLiteResponse:
        """Execute a raw SQL query"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or [])
                
                # Check if it's a SELECT query
                if query.strip().upper().startswith('SELECT'):
                    rows = cursor.fetchall()
                    data = [dict(row) for row in rows]
                    return SQLiteResponse(
                        data=data,
                        count=len(data),
                        status_code=200,
                        error=None
                    )
                else:
                    conn.commit()
                    return SQLiteResponse(
                        data={"affected_rows": cursor.rowcount},
                        count=cursor.rowcount,
                        status_code=200,
                        error=None
                    )
        
        except Exception as e:
            logger.error(f"SQLite query error: {e}")
            return SQLiteResponse(
                data=None,
                count=None,
                status_code=500,
                error={"message": str(e)}
            )
    
    def create_user(self, username: str, roles: List[str], security_clearance: str, 
                   department: Optional[str] = None) -> Dict[str, Any]:
        """Create database user with enterprise security roles"""
        if not self.security_manager:
            raise RuntimeError("Security manager not enabled")
        
        from app.core.databaseSecurityManager import DatabaseRole, SecurityLevel


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        
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
    
    # Async methods using connection pool
    @cached(cache_name="sqlite_select", ttl=60)
    async def async_select_cached(
        self,
        table: str,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
        pagination: Optional[PaginationParams] = None
    ) -> PaginatedResponse:
        """Cached and paginated async select"""
        if not pagination:
            pagination = PaginationParams()
            
        # Get total count first
        count_result = await self.async_select(
            table=table,
            columns=["COUNT(*) as total"],
            filters=filters
        )
        total = count_result.data[0]['total'] if count_result.success else 0
        
        # Get paginated data
        result = await self.async_select(
            table=table,
            columns=columns,
            filters=filters,
            limit=pagination.limit,
            offset=pagination.offset
        )
        
        if result.success:
            return PaginatedResponse.create(
                items=result.data,
                total=total,
                params=pagination
            )
        else:
            return PaginatedResponse.create([], 0, pagination)
    
    async def async_select(
        self,
        table: str,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = None
    ) -> SQLiteResponse:
        """Async select operation using connection pool"""
        try:
            # Validate inputs
            if not self._is_valid_table_name(table):
                raise ValueError(f"Invalid table name: {table}")
            
            # Get connection from pool
            async with self.connection_pool.get_async_connection() as conn:
                # Build query
                query_parts = ["SELECT"]
                
                if columns:
                    validated_columns = [col for col in columns if self._is_valid_column_name(col)]
                    query_parts.append(", ".join(validated_columns))
                else:
                    query_parts.append("*")
                
                query_parts.append(f"FROM {table}")
                
                # Add WHERE clause
                params = []
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if self._is_valid_column_name(key):
                            conditions.append(f"{key} = ?")
                            params.append(value)
                    
                    if conditions:
                        query_parts.append("WHERE " + " AND ".join(conditions))
                
                # Add ORDER BY
                if order_by:
                    if order_by.startswith("-"):
                        column = order_by[1:]
                        direction = "DESC"
                    else:
                        column = order_by
                        direction = "ASC"
                    
                    if self._is_valid_column_name(column):
                        query_parts.append(f"ORDER BY {column} {direction}")
                
                # Add LIMIT and OFFSET
                query_parts.append(f"LIMIT {limit} OFFSET {offset}")
                
                query = " ".join(query_parts)
                
                # Execute query
                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    # Convert to dict format
                    results = []
                    for row in rows:
                        results.append(dict(zip(columns, row)))
                
                return SQLiteResponse(
                    success=True,
                    data=results,
                    count=len(results)
                )
                
        except Exception as e:
            logger.error(f"Async select failed: {e}")
            return SQLiteResponse(
                success=False,
                error=str(e)
            )
    
    async def async_insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> SQLiteResponse:
        """Async insert operation using connection pool"""
        try:
            # Validate table name
            if not self._is_valid_table_name(table):
                raise ValueError(f"Invalid table name: {table}")
            
            # Ensure data is a list
            records = data if isinstance(data, list) else [data]
            
            if not records:
                return SQLiteResponse(success=True, count=0)
            
            # Get connection from pool
            async with self.connection_pool.get_async_connection() as conn:
                # Prepare insert statement
                first_record = records[0]
                columns = [col for col in first_record.keys() if self._is_valid_column_name(col)]
                
                placeholders = ", ".join(["?" for _ in columns])
                column_list = ", ".join(columns)
                
                query = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"
                
                # Execute inserts
                inserted_count = 0
                for record in records:
                    values = [record.get(col) for col in columns]
                    await conn.execute(query, values)
                    inserted_count += 1
                
                await conn.commit()
                
                return SQLiteResponse(
                    success=True,
                    count=inserted_count
                )
                
        except Exception as e:
            logger.error(f"Async insert failed: {e}")
            return SQLiteResponse(
                success=False,
                error=str(e)
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the SQLite client"""
        try:
            # Test database connection
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                
                # Get database info
                cursor.execute("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()['count']
                
                # Get file size
                db_size = os.path.getsize(self.config.db_path) if os.path.exists(self.config.db_path) else 0
                
                health_report = {
                    "status": "healthy",
                    "database": "connected",
                    "table_count": table_count,
                    "db_size_bytes": db_size,
                    "db_path": self.config.db_path,
                    "journal_mode": self.config.journal_mode,
                    "encryption_enabled": self.config.enable_encryption and SQLCIPHER_AVAILABLE,
                    "pooling_enabled": self.config.enable_connection_pooling
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
    
    def validate_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    [table_name]
                )
                return cursor.fetchone() is not None
        except:
            return False
    
    async def close(self):
        """Close client connections (if any persistent connections exist)"""
        logger.info("SQLite client connections closed")


# Factory function for easy instantiation
def create_sqlite_client(config: Optional[SQLiteConfig] = None) -> SQLiteClient:
    """Factory function to create a SQLite client"""
    return SQLiteClient(config)


# Singleton instance for global use
_sqlite_client_instance: Optional[SQLiteClient] = None

def get_sqlite_client() -> SQLiteClient:
    """Get singleton SQLite client instance"""
    global _sqlite_client_instance
    
    if _sqlite_client_instance is None:
        _sqlite_client_instance = create_sqlite_client()
    
    return _sqlite_client_instance
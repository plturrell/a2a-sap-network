"""
SAP HANA Cloud Production Client
Production-ready client for SAP HANA Cloud database integration
"""

import os
import asyncio
import logging
import re
import json
import time
import hashlib
import uuid
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager, asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv

from app.core.databaseSecurityManager import DatabaseSecurityManager, SecurityLevel

# Import BTP service configuration
try:
    from config.btpServiceConfig import btp_config
    BTP_CONFIG_AVAILABLE = True
except ImportError:
    BTP_CONFIG_AVAILABLE = False
    btp_config = None

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
    max_overflow: int = 10
    pool_recycle: int = 3600
    pool_timeout: int = 30
    pool_pre_ping: bool = True
    database: Optional[str] = None
    schema: Optional[str] = None
    isolation_level: str = 'READ_COMMITTED'
    connection_timeout: int = 30
    compress: bool = True


def create_hana_config_from_btp() -> HanaConfig:
    """
    Create HANA configuration from BTP service bindings
    Falls back to environment variables for local development
    """
    if BTP_CONFIG_AVAILABLE and btp_config:
        try:
            # Get HANA configuration from BTP service bindings
            hana_config_dict = btp_config.get_hana_config()

            # Create HanaConfig from BTP service binding
            config = HanaConfig(
                address=hana_config_dict['address'],
                port=hana_config_dict['port'],
                user=hana_config_dict['user'],
                password=hana_config_dict['password'],
                encrypt=hana_config_dict.get('encrypt', True),
                ssl_validate_certificate=hana_config_dict.get('ssl_validate_certificate', True),
                auto_commit=hana_config_dict.get('auto_commit', False),
                timeout=hana_config_dict.get('connection_timeout', 30),
                pool_size=hana_config_dict.get('pool_size', 15),
                max_overflow=hana_config_dict.get('max_overflow', 25),
                pool_recycle=hana_config_dict.get('pool_recycle', 3600),
                pool_timeout=hana_config_dict.get('pool_timeout', 30),
                pool_pre_ping=hana_config_dict.get('pool_pre_ping', True),
                database=hana_config_dict.get('database'),
                schema=hana_config_dict.get('schema'),
                isolation_level=hana_config_dict.get('isolation_level', 'READ_COMMITTED'),
                connection_timeout=hana_config_dict.get('connection_timeout', 30),
                compress=hana_config_dict.get('compress', True)
            )

            logger.info("✅ HANA configuration loaded from BTP service bindings")
            return config

        except Exception as e:
            logger.warning(f"⚠️ Failed to load HANA config from BTP service bindings: {e}")
            logger.info("🔄 Falling back to environment configuration")

    # Fallback to environment variables for local development
    config = HanaConfig(
        address=os.getenv("HANA_HOST", "localhost"),
        port=int(os.getenv("HANA_PORT", "30015")),
        user=os.getenv("HANA_USER", "SYSTEM"),
        password=os.getenv("HANA_PASSWORD", ""),
        encrypt=os.getenv("HANA_ENCRYPT", "true").lower() == "true",
        ssl_validate_certificate=os.getenv("HANA_SSL_VALIDATE", "false").lower() == "true",
        auto_commit=os.getenv("HANA_AUTO_COMMIT", "false").lower() == "true",
        timeout=int(os.getenv("HANA_TIMEOUT", "30")),
        pool_size=int(os.getenv("HANA_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("HANA_MAX_OVERFLOW", "10")),
        pool_recycle=int(os.getenv("HANA_POOL_RECYCLE", "3600")),
        pool_timeout=int(os.getenv("HANA_POOL_TIMEOUT", "30")),
        pool_pre_ping=os.getenv("HANA_POOL_PRE_PING", "true").lower() == "true",
        database=os.getenv("HANA_DATABASE"),
        schema=os.getenv("HANA_SCHEMA", "A2A_AGENTS"),
        isolation_level=os.getenv("HANA_ISOLATION_LEVEL", "READ_COMMITTED"),
        connection_timeout=int(os.getenv("HANA_CONNECTION_TIMEOUT", "30")),
        compress=os.getenv("HANA_COMPRESS", "true").lower() == "true"
    )

    logger.info("✅ HANA configuration loaded from environment variables")
    return config


@dataclass
class QueryResult:
    """Structured result from HANA query"""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: Optional[float] = None
    raw_result: Optional[Any] = None


class HanaConnectionPool:
    """Enterprise-grade HANA connection pool with advanced features"""

    def __init__(self, config: HanaConfig):
        if not HANA_AVAILABLE:
            raise ImportError("SAP HANA client not available. Install with: pip install hdbcli")

        self.config = config
        self._pool = []
        self._active_connections = set()
        self._connection_timestamps = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._connection_id_counter = 0

        # Enhanced pool configuration
        self.config.pool_size = max(5, getattr(config, 'pool_size', 10))
        self.config.max_overflow = getattr(config, 'max_overflow', 10)
        self.config.pool_recycle = getattr(config, 'pool_recycle', 3600)  # 1 hour
        self.config.pool_timeout = getattr(config, 'pool_timeout', 30)
        self.config.pool_pre_ping = getattr(config, 'pool_pre_ping', True)

        # Health monitoring
        self.health_metrics = {
            "active_connections": 0,
            "idle_connections": 0,
            "failed_connections": 0,
            "avg_connection_time": 0.0,
            "connection_errors": 0,
            "pool_exhausted_count": 0,
            "stale_connections_recycled": 0
        }

        # Circuit breaker pattern
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        self.circuit_breaker_last_failure = 0

        # Query optimization tracking
        self.query_cache = {}
        self.execution_stats = {}

        # Initialize pool
        self._initialize_pool()

        # Start maintenance thread
        self._start_maintenance_thread()

        logger.info(f"Enterprise HANA connection pool initialized: {self.config.pool_size} base connections, {self.config.max_overflow} overflow")

    def _initialize_pool(self):
        """Initialize connection pool with base connections"""
        for i in range(self.config.pool_size):
            try:
                connection = self._create_connection()
                self._pool.append(connection)
                logger.debug(f"Added connection {i+1} to pool")
            except Exception as e:
                logger.error(f"Failed to create initial connection {i+1}: {e}")
                self.health_metrics["failed_connections"] += 1

        self._initialized = True
        logger.info(f"HANA pool initialized with {len(self._pool)} connections")

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        def maintenance_worker():
            while True:
                try:
                    self._perform_maintenance()
                    time.sleep(30)  # Run every 30 seconds
                except Exception as e:
                    logger.error(f"Pool maintenance error: {e}")

        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
        logger.info("Started connection pool maintenance thread")

    def _perform_maintenance(self):
        """Perform regular pool maintenance"""
        current_time = time.time()

        with self._lock:
            # Check for stale connections
            stale_connections = []
            for conn_id, timestamp in self._connection_timestamps.items():
                if current_time - timestamp > self.config.pool_recycle:
                    stale_connections.append(conn_id)

            # Update health metrics
            self.health_metrics["active_connections"] = len(self._active_connections)
            self.health_metrics["idle_connections"] = len(self._pool)

            # Reset circuit breaker if timeout elapsed
            if (current_time - self.circuit_breaker_last_failure) > self.circuit_breaker_timeout:
                if self.circuit_breaker_failures > 0:
                    logger.info("Circuit breaker reset after timeout")
                self.circuit_breaker_failures = 0

            # Log health metrics periodically
            if int(current_time) % 300 == 0:  # Every 5 minutes
                logger.info(f"Pool health: {self.health_metrics}")

    def _check_circuit_breaker(self):
        """Check if circuit breaker should prevent new connections"""
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            if (time.time() - self.circuit_breaker_last_failure) < self.circuit_breaker_timeout:
                raise Exception("Circuit breaker OPEN - too many connection failures")

    def _record_connection_failure(self):
        """Record a connection failure for circuit breaker"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        self.health_metrics["connection_errors"] += 1

        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            logger.warning(f"Circuit breaker OPENED after {self.circuit_breaker_failures} failures")

    def _create_connection(self):
        """Create a new HANA database connection with enterprise settings"""
        import time

        # Check circuit breaker
        self._check_circuit_breaker()

        start_time = time.time()

        try:
            # Create connection with enterprise parameters
            connection = dbapi.connect(
                address=self.config.address,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                encrypt=self.config.encrypt,
                sslValidateCertificate=self.config.ssl_validate_certificate,
                autocommit=self.config.auto_commit,
                # Enterprise connection parameters
                timeout=getattr(self.config, 'connection_timeout', 30),
                reconnect=True,
                compress=getattr(self.config, 'compress', True)
            )

            # Assign unique connection ID for tracking
            with self._lock:
                self._connection_id_counter += 1
                connection._a2a_connection_id = self._connection_id_counter
                self._connection_timestamps[connection._a2a_connection_id] = time.time()

            # Set connection-level optimizations
            cursor = connection.cursor()

            # Set isolation level for better performance
            isolation_level = getattr(self.config, 'isolation_level', 'READ_COMMITTED')
            cursor.execute(f"SET ISOLATION LEVEL {isolation_level}")

            # Enable query optimization hints
            cursor.execute("SET STATEMENT_MEMORY_LIMIT = '2GB'")
            cursor.execute("SET STATEMENT_TIMEOUT = 300000")  # 5 minutes

            # Set schema if specified
            if hasattr(self.config, 'schema') and self.config.schema:
                cursor.execute(f"SET SCHEMA {self.config.schema}")

            cursor.close()

            connection_time = time.time() - start_time
            self.health_metrics["avg_connection_time"] = (
                (self.health_metrics["avg_connection_time"] + connection_time) / 2
            )

            logger.debug(f"Created HANA connection {connection._a2a_connection_id} in {connection_time:.2f}s")
            return connection

        except Exception as e:
            self._record_connection_failure()
            logger.error(f"Failed to create HANA connection: {e}")
            raise

    def get_connection(self):
        """Get a connection from the pool with enterprise health checking"""
        import time
        start_time = time.time()

        try:
            with self._lock:
                # Check if we can create overflow connections
                total_active = len(self._active_connections)
                max_total = self.config.pool_size + self.config.max_overflow

                if self._pool:
                    # Get connection from pool
                    connection = self._pool.pop()

                    # Enhanced health check with pre-ping
                    if self.config.pool_pre_ping:
                        if not self._test_connection_health(connection):
                            # Connection is stale, close and create new one
                            self._close_connection_safely(connection)
                            self.health_metrics["stale_connections_recycled"] += 1
                            connection = self._create_connection()

                    # Track active connection
                    if hasattr(connection, '_a2a_connection_id'):
                        self._active_connections.add(connection._a2a_connection_id)

                    self.health_metrics["active_connections"] = len(self._active_connections)
                    return connection

                elif total_active < max_total:
                    # Create new connection (within overflow limit)
                    connection = self._create_connection()

                    if hasattr(connection, '_a2a_connection_id'):
                        self._active_connections.add(connection._a2a_connection_id)

                    self.health_metrics["active_connections"] = len(self._active_connections)
                    return connection

                else:
                    # Pool exhausted
                    self.health_metrics["pool_exhausted_count"] += 1
                    raise Exception(f"Connection pool exhausted: {total_active} active connections")

        except Exception as e:
            wait_time = time.time() - start_time
            if wait_time > self.config.pool_timeout:
                raise Exception(f"Connection timeout after {wait_time:.2f}s")
            raise

    def return_connection(self, connection):
        """Return a connection to the pool with proper cleanup"""
        with self._lock:
            try:
                # Remove from active connections tracking
                if hasattr(connection, '_a2a_connection_id'):
                    self._active_connections.discard(connection._a2a_connection_id)
                    self.health_metrics["active_connections"] = len(self._active_connections)

                # Check if connection should be recycled
                if self._should_recycle_connection(connection):
                    self._close_connection_safely(connection)
                    self.health_metrics["stale_connections_recycled"] += 1
                    return

                # Return to pool if there's space
                if len(self._pool) < self.config.pool_size:
                    self._pool.append(connection)
                    self.health_metrics["idle_connections"] = len(self._pool)
                else:
                    # Pool is full, close the connection
                    self._close_connection_safely(connection)

            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
                self._close_connection_safely(connection)

    def _test_connection_health(self, connection) -> bool:
        """Test if connection is still healthy"""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM SYS.DUMMY")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception as e:
            logger.debug(f"Connection health check failed: {e}")
            return False

    def _should_recycle_connection(self, connection) -> bool:
        """Check if connection should be recycled based on age"""
        import time
        if not hasattr(connection, '_a2a_connection_id'):
            return True

        conn_id = connection._a2a_connection_id
        if conn_id not in self._connection_timestamps:
            return True

        connection_age = time.time() - self._connection_timestamps[conn_id]
        return connection_age > self.config.pool_recycle

    def _close_connection_safely(self, connection):
        """Safely close a connection with error handling"""
        try:
            if hasattr(connection, '_a2a_connection_id'):
                conn_id = connection._a2a_connection_id
                self._active_connections.discard(conn_id)
                self._connection_timestamps.pop(conn_id, None)

            connection.close()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self._lock:
            return {
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "active_connections": len(self._active_connections),
                "idle_connections": len(self._pool),
                "total_connections": len(self._active_connections) + len(self._pool),
                "health_metrics": self.health_metrics.copy(),
                "circuit_breaker": {
                    "failures": self.circuit_breaker_failures,
                    "threshold": self.circuit_breaker_threshold,
                    "state": "OPEN" if self.circuit_breaker_failures >= self.circuit_breaker_threshold else "CLOSED"
                }
            }

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

        # Initialize query optimization engine
        self.query_optimizer = QueryOptimizer()

        # Initialize transaction manager
        self.transaction_manager = EnterpriseTransactionManager()

        # Initialize backup manager if enabled
        self.backup_manager = None
        if getattr(config, 'backup_enabled', True):
            from .enterpriseBackupManager import create_enterprise_backup_manager, BackupConfig
            backup_config = BackupConfig()
            self.backup_manager = create_enterprise_backup_manager(self, backup_config)

        logger.info(f"Enterprise HANA client initialized for {config.address}:{config.port}")


class QueryOptimizer:
    """Advanced query optimization for HANA"""

    def __init__(self):
        self.query_cache = {}
        self.execution_stats = {}
        self.index_recommendations = {}

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for optimization opportunities"""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()

        analysis = {
            "query_hash": query_hash,
            "complexity": self._estimate_complexity(query),
            "suggestions": self._get_optimization_suggestions(query),
            "estimated_cost": self._estimate_query_cost(query)
        }

        return analysis

    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity"""
        query_lower = query.lower()
        join_count = query_lower.count('join')
        subquery_count = query_lower.count('select') - 1

        if join_count > 3 or subquery_count > 2:
            return "HIGH"
        elif join_count > 1 or subquery_count > 0:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_optimization_suggestions(self, query: str) -> List[str]:
        """Get HANA-specific optimization suggestions"""
        suggestions = []
        query_lower = query.lower()

        if "select *" in query_lower:
            suggestions.append("Avoid SELECT * - specify only needed columns")

        if "order by" in query_lower and "limit" not in query_lower:
            suggestions.append("Consider adding LIMIT clause for large result sets")

        if query_lower.count("join") > 2:
            suggestions.append("Consider using CTEs or temporary tables for complex joins")

        if "where" in query_lower and "index" not in query_lower:
            suggestions.append("Ensure indexes exist on WHERE clause columns")

        return suggestions

    def _estimate_query_cost(self, query: str) -> int:
        """Simple query cost estimation"""
        cost = 1
        query_lower = query.lower()

        # Add cost for operations
        cost += query_lower.count('join') * 10
        cost += query_lower.count('group by') * 5
        cost += query_lower.count('order by') * 3
        cost += query_lower.count('distinct') * 2

        return cost


class EnterpriseTransactionManager:
    """Advanced transaction management with retry logic"""

    def __init__(self):
        self.active_transactions = {}
        self.transaction_lock = threading.Lock()
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 0.1,
            "max_delay": 2.0,
            "exponential_base": 2
        }

    @contextmanager
    def transaction(self, connection, isolation_level: Optional[str] = None):
        """Advanced transaction context manager with retry logic"""
        import time
        import uuid

        transaction_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Set isolation level if specified
            if isolation_level:
                cursor = connection.cursor()
                cursor.execute(f"SET ISOLATION LEVEL {isolation_level}")
                cursor.close()

            # Begin transaction
            connection.setautocommit(False)

            # Track transaction
            with self.transaction_lock:
                self.active_transactions[transaction_id] = {
                    "start_time": start_time,
                    "isolation_level": isolation_level,
                    "connection_id": getattr(connection, '_a2a_connection_id', 'unknown')
                }

            logger.debug(f"Started transaction {transaction_id}")
            yield transaction_id

            # Commit transaction
            connection.commit()
            logger.debug(f"Committed transaction {transaction_id}")

        except Exception as e:
            # Rollback on error
            try:
                connection.rollback()
                logger.warning(f"Rolled back transaction {transaction_id}: {e}")
            except Exception as rollback_error:
                logger.error(f"Rollback failed for transaction {transaction_id}: {rollback_error}")
            raise

        finally:
            # Cleanup
            try:
                connection.setautocommit(True)
                with self.transaction_lock:
                    self.active_transactions.pop(transaction_id, None)

                duration = time.time() - start_time
                if duration > 60:  # Long transaction warning
                    logger.warning(f"Long transaction {transaction_id}: {duration:.2f}s")

            except Exception as e:
                logger.error(f"Transaction cleanup failed: {e}")

    def retry_on_deadlock(self, func, *args, **kwargs):
        """Retry function on deadlock or timeout"""
        import time
        import random

        for attempt in range(self.retry_config["max_retries"]):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "deadlock" in error_msg or "timeout" in error_msg:
                    if attempt < self.retry_config["max_retries"] - 1:
                        delay = min(
                            self.retry_config["base_delay"] * (self.retry_config["exponential_base"] ** attempt),
                            self.retry_config["max_delay"]
                        )
                        # Add jitter
                        delay *= (0.5 + random.random() * 0.5)

                        logger.warning(f"Retrying after deadlock/timeout (attempt {attempt + 1}): {e}")
                        time.sleep(delay)
                        continue
                raise

        raise Exception(f"Transaction failed after {self.retry_config['max_retries']} retries")

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
        """Execute a SQL query with enterprise optimization, monitoring, and security"""
        try:
            # Query optimization analysis
            query_analysis = self.query_optimizer.analyze_query(query)

            # Log optimization suggestions for complex queries
            if query_analysis["complexity"] in ["MEDIUM", "HIGH"] and query_analysis["suggestions"]:
                logger.info(f"Query optimization suggestions: {query_analysis['suggestions']}")

            # Enterprise security check
            if self.security_manager and user_id and table_name:
                operation = self._extract_operation_from_query(query)
                if not self.security_manager.check_permission(user_id, operation, table_name, data_security_level):
                    raise PermissionError(f"User {user_id} does not have {operation} permission on {table_name}")

            # Use transaction manager for retry logic on deadlocks
            def execute_with_retry():
                with self.get_connection() as connection:
                    cursor = connection.cursor()

                    import time
                    start_time = time.time()

                    # Apply query hints for HANA optimization
                    optimized_query = self._apply_hana_hints(query, query_analysis)

                    if parameters:
                        cursor.execute(optimized_query, parameters)
                    else:
                        cursor.execute(optimized_query)

                    execution_time = time.time() - start_time

                    # Record performance metrics
                    self.query_optimizer.execution_stats[query_analysis["query_hash"]] = execution_time

                    return cursor, execution_time

            # Execute with deadlock retry
            cursor, execution_time = self.transaction_manager.retry_on_deadlock(execute_with_retry)

            with self.get_connection() as connection:
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
                        execution_time=execution_time,
                        raw_result=rows
                    )
                else:
                    # Non-SELECT query or no results requested
                    row_count = getattr(cursor, 'rowcount', 0)
                    result = QueryResult(
                        data=[],
                        columns=[],
                        row_count=row_count,
                        execution_time=execution_time
                    )

                cursor.close()

                # Enterprise performance monitoring
                self.performance_monitor.log_query_performance(
                    query, execution_time, result.row_count, user_id
                )

                # Log slow queries with optimization suggestions
                if execution_time > self.performance_monitor.slow_query_threshold:
                    suggestions = self.performance_monitor.suggest_optimization(query, execution_time)
                    if suggestions:
                        logger.warning(f"Slow query optimization suggestions: {suggestions}")

                return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            if self.security_manager and user_id:
                self.security_manager.log_security_event(
                    user_id, "QUERY_FAILED", {"error": str(e), "query_preview": query[:100]}
                )
            raise

    def _apply_hana_hints(self, query: str, analysis: Dict[str, Any]) -> str:
        """Apply HANA-specific optimization hints"""
        hints = []

        # Add hints based on query complexity
        if analysis["complexity"] == "HIGH":
            hints.append("USE_PARALLEL")
            hints.append("DISABLE_OPTIMIZER_TRACE")

        if analysis["estimated_cost"] > 50:
            hints.append("USE_HASH_JOIN")

        # Apply hints if any
        if hints and "select" in query.lower():
            hint_string = f"WITH HINT({', '.join(hints)}) "
            query = query.replace("SELECT", f"SELECT {hint_string}", 1)

        return query

    def execute_transaction(self, operations: List[Tuple[str, Optional[Union[List, Tuple, Dict]]]],
                           isolation_level: Optional[str] = None) -> List[QueryResult]:
        """Execute multiple operations in a single transaction"""
        results = []

        with self.get_connection() as connection:
            with self.transaction_manager.transaction(connection, isolation_level):
                for query, parameters in operations:
                    result = self.execute_query(query, parameters, connection=connection)
                    results.append(result)

        return results

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database and pool metrics"""
        pool_stats = self.pool.get_pool_statistics()

        metrics = {
            "pool_statistics": pool_stats,
            "query_optimizer": {
                "cached_queries": len(self.query_optimizer.query_cache),
                "execution_stats_count": len(self.query_optimizer.execution_stats),
                "index_recommendations_count": len(self.query_optimizer.index_recommendations)
            },
            "transaction_manager": {
                "active_transactions": len(self.transaction_manager.active_transactions),
                "retry_config": self.transaction_manager.retry_config
            }
        }

        # Add backup manager metrics if available
        if self.backup_manager:
            metrics["backup_manager"] = self.backup_manager.get_backup_status()

        return metrics

    def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization tasks"""
        optimization_results = {
            "statistics_updated": False,
            "index_analysis": {},
            "recommendations": []
        }

        try:
            # Update table statistics for better query planning
            with self.get_connection() as connection:
                cursor = connection.cursor()

                # Update statistics on all tables in schema
                cursor.execute("UPDATE STATISTICS")
                optimization_results["statistics_updated"] = True

                # Analyze unused indexes
                unused_index_query = """
                    SELECT SCHEMA_NAME, TABLE_NAME, INDEX_NAME, LAST_ACCESS_TIME
                    FROM SYS.M_TABLE_STATISTICS
                    WHERE LAST_ACCESS_TIME < ADD_DAYS(CURRENT_DATE, -30)
                """

                cursor.execute(unused_index_query)
                unused_indexes = cursor.fetchall()

                if unused_indexes:
                    optimization_results["recommendations"].append(
                        f"Consider dropping {len(unused_indexes)} unused indexes"
                    )

                # Check for missing indexes on frequently queried columns
                for query_hash, execution_time in self.query_optimizer.execution_stats.items():
                    if execution_time > 5.0:  # Slow queries
                        optimization_results["recommendations"].append(
                            "Review indexes for slow queries - check query execution plans"
                        )
                        break

                cursor.close()

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            optimization_results["error"] = str(e)

        return optimization_results

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

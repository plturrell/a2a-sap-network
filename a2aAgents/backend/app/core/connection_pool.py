"""
Connection pooling for database connections with health checks and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import sqlite3
import aiomysql
import aioredis
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
connection_acquired = Counter('db_connections_acquired_total', 'Total connections acquired', ['pool_name', 'db_type'])
connection_released = Counter('db_connections_released_total', 'Total connections released', ['pool_name', 'db_type'])
connection_errors = Counter('db_connection_errors_total', 'Total connection errors', ['pool_name', 'db_type', 'error_type'])
active_connections = Gauge('db_active_connections', 'Active connections', ['pool_name', 'db_type'])
pool_size = Gauge('db_pool_size', 'Current pool size', ['pool_name', 'db_type'])
connection_wait_time = Histogram('db_connection_wait_seconds', 'Time waiting for connection', ['pool_name', 'db_type'])


@dataclass
class PooledConnection:
    """Wrapper for pooled connections"""
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    pool_name: str = ""
    db_type: str = ""

    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if connection is too old"""
        return (datetime.utcnow() - self.created_at).total_seconds() > max_age_seconds

    def is_idle(self, max_idle_seconds: int = 300) -> bool:
        """Check if connection has been idle too long"""
        return (datetime.utcnow() - self.last_used).total_seconds() > max_idle_seconds


class ConnectionPool:
    """Generic connection pool with health checks"""

    def __init__(
        self,
        name: str,
        db_type: str,
        min_size: int = 5,
        max_size: int = 20,
        max_age_seconds: int = 3600,
        max_idle_seconds: int = 300,
        health_check_interval: int = 60
    ):
        self.name = name
        self.db_type = db_type
        self.min_size = min_size
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.max_idle_seconds = max_idle_seconds
        self.health_check_interval = health_check_interval

        self._pool: List[PooledConnection] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._closing = False
        self._connection_factory: Optional[Callable] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Update metrics
        pool_size.labels(pool_name=name, db_type=db_type).set(0)
        active_connections.labels(pool_name=name, db_type=db_type).set(0)

    async def initialize(self, connection_factory: Callable):
        """Initialize the pool with minimum connections"""
        self._connection_factory = connection_factory

        # Create minimum connections
        for _ in range(self.min_size):
            conn = await self._create_connection()
            if conn:
                await self._available.put(conn)

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Initialized connection pool '{self.name}' with {len(self._pool)} connections")

    async def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new connection"""
        try:
            conn = await self._connection_factory()
            pooled_conn = PooledConnection(
                connection=conn,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                pool_name=self.name,
                db_type=self.db_type
            )

            async with self._lock:
                self._pool.append(pooled_conn)
                pool_size.labels(pool_name=self.name, db_type=self.db_type).set(len(self._pool))

            return pooled_conn

        except Exception as e:
            logger.error(f"Failed to create connection for pool '{self.name}': {e}")
            connection_errors.labels(
                pool_name=self.name,
                db_type=self.db_type,
                error_type='creation_failed'
            ).inc()
            return None

    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0):
        """Acquire a connection from the pool"""
        start_time = time.time()
        conn = None

        try:
            # Try to get available connection
            try:
                conn = await asyncio.wait_for(self._available.get(), timeout=timeout)
            except asyncio.TimeoutError:
                # Try to create new connection if under max size
                async with self._lock:
                    if len(self._pool) < self.max_size:
                        conn = await self._create_connection()

                if not conn:
                    raise TimeoutError(f"Could not acquire connection from pool '{self.name}' within {timeout}s")

            # Check if connection is still valid
            if conn.is_stale(self.max_age_seconds):
                await self._close_connection(conn)
                conn = await self._create_connection()
                if not conn:
                    raise RuntimeError(f"Failed to create replacement connection for pool '{self.name}'")

            # Update metrics
            wait_time = time.time() - start_time
            connection_wait_time.labels(pool_name=self.name, db_type=self.db_type).observe(wait_time)
            connection_acquired.labels(pool_name=self.name, db_type=self.db_type).inc()
            active_connections.labels(pool_name=self.name, db_type=self.db_type).inc()

            # Update connection usage
            conn.last_used = datetime.utcnow()
            conn.use_count += 1

            yield conn.connection

        except Exception as e:
            connection_errors.labels(
                pool_name=self.name,
                db_type=self.db_type,
                error_type='acquire_failed'
            ).inc()
            raise

        finally:
            if conn:
                # Return connection to pool
                await self._available.put(conn)
                connection_released.labels(pool_name=self.name, db_type=self.db_type).inc()
                active_connections.labels(pool_name=self.name, db_type=self.db_type).dec()

    async def _close_connection(self, conn: PooledConnection):
        """Close a connection and remove from pool"""
        try:
            if hasattr(conn.connection, 'close'):
                if asyncio.iscoroutinefunction(conn.connection.close):
                    await conn.connection.close()
                else:
                    conn.connection.close()

            async with self._lock:
                if conn in self._pool:
                    self._pool.remove(conn)
                    pool_size.labels(pool_name=self.name, db_type=self.db_type).set(len(self._pool))

        except Exception as e:
            logger.error(f"Error closing connection in pool '{self.name}': {e}")

    async def _health_check_loop(self):
        """Periodic health check of connections"""
        while not self._closing:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._health_check()
            except Exception as e:
                logger.error(f"Health check failed for pool '{self.name}': {e}")

    async def _health_check(self):
        """Check health of all connections"""
        logger.debug(f"Running health check for pool '{self.name}'")

        # Get all available connections
        connections_to_check = []
        while not self._available.empty():
            try:
                conn = self._available.get_nowait()
                connections_to_check.append(conn)
            except asyncio.QueueEmpty:
                break

        # Check each connection
        healthy_connections = []
        for conn in connections_to_check:
            if conn.is_stale(self.max_age_seconds) or conn.is_idle(self.max_idle_seconds):
                logger.info(f"Removing stale/idle connection from pool '{self.name}'")
                await self._close_connection(conn)
            else:
                # Perform health check based on database type
                if await self._check_connection_health(conn):
                    healthy_connections.append(conn)
                else:
                    logger.warning(f"Removing unhealthy connection from pool '{self.name}'")
                    await self._close_connection(conn)

        # Return healthy connections to pool
        for conn in healthy_connections:
            await self._available.put(conn)

        # Ensure minimum pool size
        async with self._lock:
            current_size = len(self._pool)
            if current_size < self.min_size:
                for _ in range(self.min_size - current_size):
                    new_conn = await self._create_connection()
                    if new_conn:
                        await self._available.put(new_conn)

    async def _check_connection_health(self, conn: PooledConnection) -> bool:
        """Check if a specific connection is healthy"""
        try:
            if self.db_type == "sqlite":
                conn.connection.execute("SELECT 1")
            elif self.db_type == "mysql":
                await conn.connection.ping()
            elif self.db_type == "redis":
                await conn.connection.ping()
            elif self.db_type == "hana":
                # HANA-specific health check
                cursor = conn.connection.cursor()
                cursor.execute("SELECT 1 FROM DUMMY")
                cursor.close()
            return True
        except Exception as e:
            logger.debug(f"Connection health check failed: {e}")
            return False

    async def close(self):
        """Close all connections and shutdown pool"""
        self._closing = True

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for conn in self._pool:
                await self._close_connection(conn)

        logger.info(f"Closed connection pool '{self.name}'")


class ConnectionPoolManager:
    """Manages multiple connection pools"""

    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}

    async def create_pool(
        self,
        name: str,
        db_type: str,
        connection_factory: Callable,
        **pool_kwargs
    ) -> ConnectionPool:
        """Create and initialize a new connection pool"""

        if name in self.pools:
            raise ValueError(f"Pool '{name}' already exists")

        pool = ConnectionPool(name, db_type, **pool_kwargs)
        await pool.initialize(connection_factory)

        self.pools[name] = pool
        return pool

    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name"""
        return self.pools.get(name)

    async def close_all(self):
        """Close all connection pools"""
        for pool in self.pools.values():
            await pool.close()
        self.pools.clear()


# Global connection pool manager
pool_manager = ConnectionPoolManager()

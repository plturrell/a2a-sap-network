"""
SQLite Database Client for a2aNetwork
Handles connections and operations with SQLite database
"""

import logging
import os
import aiosqlite
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class SqliteClient:
    """Client for interacting with SQLite database"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "a2a_network.db")
        self.connection = None
        logger.info(f"Initialized SQLite client with database: {self.db_path}")
    
    async def connect(self):
        """Establish connection to SQLite database"""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self.connection.execute("PRAGMA foreign_keys = ON")
            logger.info("Connected to SQLite database")
        except Exception as e:
            logger.error(f"SQLite connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close connection to SQLite database"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Disconnected from SQLite database")
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions"""
        await self.connect()
        try:
            yield self
            await self.connection.commit()
        except Exception as e:
            await self.connection.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            await self.disconnect()
    
    async def execute(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        try:
            cursor = await self.connection.execute(query, params or ())
            columns = [description[0] for description in cursor.description or []]
            rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """Execute a query multiple times with different parameters"""
        try:
            await self.connection.executemany(query, params_list)
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise
    
    async def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert data into a table"""
        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            
            cursor = await self.connection.execute(query, tuple(data.values()))
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise
    
    async def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update data in a table"""
        try:
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            
            params = list(data.values()) + list(where.values())
            cursor = await self.connection.execute(query, params)
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise
    
    async def delete(self, table: str, where: Dict[str, Any]) -> int:
        """Delete data from a table"""
        try:
            where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
            query = f"DELETE FROM {table} WHERE {where_clause}"
            
            cursor = await self.connection.execute(query, tuple(where.values()))
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise
    
    async def create_tables(self, schema: str):
        """Create tables from a schema"""
        try:
            await self.connection.executescript(schema)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise


# Singleton instance
_sqlite_client = None


def get_sqlite_client() -> SqliteClient:
    """Get or create the global SQLite client instance"""
    global _sqlite_client
    if _sqlite_client is None:
        _sqlite_client = SqliteClient()
    return _sqlite_client
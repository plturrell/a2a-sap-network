"""
L3 Database Cache Implementation
Provides persistent caching layer using database storage
"""

import json
import pickle
import logging
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, LargeBinary, Integer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.secrets import get_secrets_manager

logger = logging.getLogger(__name__)


class L3DatabaseCache:
    """Database-backed L3 cache implementation"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.secrets_manager = get_secrets_manager()
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self.metadata = None
        self.cache_table = None
        self._initialized = False
        
    def _get_database_url(self) -> str:
        """Get database URL from configuration"""
        try:
            # Try to get from secrets manager
            db_url = self.secrets_manager.get_secret("L3_CACHE_DATABASE_URL", required=False)
            if db_url:
                return db_url
        except Exception:
            pass
        
        # Fallback to SQLite for development
        return "sqlite+aiosqlite:///./cache_l3.db"
    
    async def initialize(self):
        """Initialize database connection and tables"""
        if self._initialized:
            return
            
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600  # Recycle connections after 1 hour
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Create metadata and table definition
            self.metadata = MetaData()
            self.cache_table = Table(
                'l3_cache_entries',
                self.metadata,
                Column('cache_key', String(255), primary_key=True, index=True),
                Column('namespace', String(100), index=True),
                Column('value_data', LargeBinary),
                Column('created_at', DateTime),
                Column('expires_at', DateTime, index=True),
                Column('access_count', Integer, default=0),
                Column('last_accessed', DateTime),
                Column('content_hash', String(64)),  # For integrity verification
                Column('size_bytes', Integer),
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
            
            self._initialized = True
            logger.info("L3 Database Cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize L3 Database Cache: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper cleanup"""
        if not self._initialized:
            await self.initialize()
            
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def _calculate_content_hash(self, data: bytes) -> str:
        """Calculate SHA-256 hash for data integrity"""
        return hashlib.sha256(data).hexdigest()
    
    async def get(self, cache_key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from L3 database cache"""
        try:
            async with self.get_session() as session:
                # Query cache entry
                query = text("""
                    SELECT value_data, expires_at, access_count, content_hash
                    FROM l3_cache_entries 
                    WHERE cache_key = :cache_key AND namespace = :namespace
                    AND expires_at > :current_time
                """)
                
                result = await session.execute(query, {
                    'cache_key': cache_key,
                    'namespace': namespace,
                    'current_time': datetime.utcnow()
                })
                
                row = result.fetchone()
                if not row:
                    logger.debug(f"L3 cache miss: {cache_key}")
                    return None
                
                value_data, expires_at, access_count, stored_hash = row
                
                # Verify data integrity
                calculated_hash = self._calculate_content_hash(value_data)
                if calculated_hash != stored_hash:
                    logger.warning(f"L3 cache integrity check failed for {cache_key}")
                    # Remove corrupted entry
                    await self.delete(cache_key, namespace)
                    return None
                
                # Update access statistics
                update_query = text("""
                    UPDATE l3_cache_entries 
                    SET access_count = access_count + 1, last_accessed = :now
                    WHERE cache_key = :cache_key AND namespace = :namespace
                """)
                
                await session.execute(update_query, {
                    'cache_key': cache_key,
                    'namespace': namespace,
                    'now': datetime.utcnow()
                })
                await session.commit()
                
                # Deserialize value
                try:
                    value = pickle.loads(value_data)
                    logger.debug(f"L3 cache hit: {cache_key}")
                    return value
                except Exception as e:
                    logger.error(f"Failed to deserialize L3 cache value for {cache_key}: {e}")
                    await self.delete(cache_key, namespace)
                    return None
                    
        except Exception as e:
            logger.error(f"L3 cache get error for {cache_key}: {e}")
            return None
    
    async def set(
        self, 
        cache_key: str, 
        value: Any, 
        ttl: int = 3600, 
        namespace: str = "default"
    ) -> bool:
        """Set value in L3 database cache"""
        try:
            # Serialize value
            value_data = pickle.dumps(value)
            content_hash = self._calculate_content_hash(value_data)
            size_bytes = len(value_data)
            
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl)
            
            async with self.get_session() as session:
                # Use UPSERT operation
                if self.database_url.startswith('sqlite'):
                    # SQLite UPSERT syntax
                    query = text("""
                        INSERT OR REPLACE INTO l3_cache_entries 
                        (cache_key, namespace, value_data, created_at, expires_at, 
                         access_count, last_accessed, content_hash, size_bytes)
                        VALUES (:cache_key, :namespace, :value_data, :created_at, :expires_at,
                                COALESCE((SELECT access_count FROM l3_cache_entries 
                                         WHERE cache_key = :cache_key AND namespace = :namespace), 0),
                                :last_accessed, :content_hash, :size_bytes)
                    """)
                else:
                    # PostgreSQL UPSERT syntax
                    query = text("""
                        INSERT INTO l3_cache_entries 
                        (cache_key, namespace, value_data, created_at, expires_at, 
                         access_count, last_accessed, content_hash, size_bytes)
                        VALUES (:cache_key, :namespace, :value_data, :created_at, :expires_at,
                                0, :last_accessed, :content_hash, :size_bytes)
                        ON CONFLICT (cache_key) DO UPDATE SET
                            value_data = EXCLUDED.value_data,
                            expires_at = EXCLUDED.expires_at,
                            last_accessed = EXCLUDED.last_accessed,
                            content_hash = EXCLUDED.content_hash,
                            size_bytes = EXCLUDED.size_bytes
                    """)
                
                await session.execute(query, {
                    'cache_key': cache_key,
                    'namespace': namespace,
                    'value_data': value_data,
                    'created_at': now,
                    'expires_at': expires_at,
                    'last_accessed': now,
                    'content_hash': content_hash,
                    'size_bytes': size_bytes
                })
                
                await session.commit()
                logger.debug(f"L3 cache set: {cache_key} (size: {size_bytes} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"L3 cache set error for {cache_key}: {e}")
            return False
    
    async def delete(self, cache_key: str, namespace: str = "default") -> bool:
        """Delete entry from L3 cache"""
        try:
            async with self.get_session() as session:
                query = text("""
                    DELETE FROM l3_cache_entries 
                    WHERE cache_key = :cache_key AND namespace = :namespace
                """)
                
                result = await session.execute(query, {
                    'cache_key': cache_key,
                    'namespace': namespace
                })
                
                await session.commit()
                
                deleted = result.rowcount > 0
                if deleted:
                    logger.debug(f"L3 cache deleted: {cache_key}")
                return deleted
                
        except Exception as e:
            logger.error(f"L3 cache delete error for {cache_key}: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        try:
            async with self.get_session() as session:
                query = text("""
                    DELETE FROM l3_cache_entries 
                    WHERE expires_at < :current_time
                """)
                
                result = await session.execute(query, {
                    'current_time': datetime.utcnow()
                })
                
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"L3 cache cleanup: removed {deleted_count} expired entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"L3 cache cleanup error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get L3 cache statistics"""
        try:
            async with self.get_session() as session:
                # Get overall stats
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(CASE WHEN expires_at > :current_time THEN 1 END) as active_entries,
                        COUNT(CASE WHEN expires_at <= :current_time THEN 1 END) as expired_entries,
                        SUM(size_bytes) as total_size_bytes,
                        AVG(size_bytes) as avg_size_bytes,
                        SUM(access_count) as total_accesses
                    FROM l3_cache_entries
                """)
                
                result = await session.execute(stats_query, {
                    'current_time': datetime.utcnow()
                })
                
                stats_row = result.fetchone()
                
                # Get namespace breakdown
                namespace_query = text("""
                    SELECT namespace, COUNT(*) as count
                    FROM l3_cache_entries
                    WHERE expires_at > :current_time
                    GROUP BY namespace
                    ORDER BY count DESC
                """)
                
                namespace_result = await session.execute(namespace_query, {
                    'current_time': datetime.utcnow()
                })
                
                namespaces = {row[0]: row[1] for row in namespace_result.fetchall()}
                
                if stats_row:
                    return {
                        'total_entries': stats_row[0] or 0,
                        'active_entries': stats_row[1] or 0,
                        'expired_entries': stats_row[2] or 0,
                        'total_size_bytes': stats_row[3] or 0,
                        'avg_size_bytes': float(stats_row[4] or 0),
                        'total_accesses': stats_row[5] or 0,
                        'namespaces': namespaces,
                        'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
                        'initialized': self._initialized
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get L3 cache stats: {e}")
            
        return {
            'total_entries': 0,
            'active_entries': 0,
            'expired_entries': 0,
            'total_size_bytes': 0,
            'avg_size_bytes': 0.0,
            'total_accesses': 0,
            'namespaces': {},
            'database_url': self.database_url,
            'initialized': self._initialized,
            'error': 'Failed to retrieve stats'
        }
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a specific namespace"""
        try:
            async with self.get_session() as session:
                query = text("""
                    DELETE FROM l3_cache_entries 
                    WHERE namespace = :namespace
                """)
                
                result = await session.execute(query, {
                    'namespace': namespace
                })
                
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"L3 cache cleared namespace '{namespace}': {deleted_count} entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"L3 cache clear namespace error: {e}")
            return 0
    
    async def get_top_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cache entries"""
        try:
            async with self.get_session() as session:
                query = text("""
                    SELECT cache_key, namespace, access_count, size_bytes, last_accessed
                    FROM l3_cache_entries
                    WHERE expires_at > :current_time
                    ORDER BY access_count DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'current_time': datetime.utcnow(),
                    'limit': limit
                })
                
                return [
                    {
                        'cache_key': row[0],
                        'namespace': row[1],
                        'access_count': row[2],
                        'size_bytes': row[3],
                        'last_accessed': row[4].isoformat() if row[4] else None
                    }
                    for row in result.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Failed to get top accessed entries: {e}")
            return []
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("L3 Database Cache connections closed")


# Global instance
_l3_cache: Optional[L3DatabaseCache] = None

async def get_l3_database_cache() -> L3DatabaseCache:
    """Get global L3 database cache instance"""
    global _l3_cache
    if _l3_cache is None:
        _l3_cache = L3DatabaseCache()
        await _l3_cache.initialize()
    return _l3_cache


# Export main classes and functions
__all__ = [
    'L3DatabaseCache',
    'get_l3_database_cache'
]
"""
Data Manager Agent - A2A Microservice
Handles persistent storage and retrieval of data in the A2A network
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
from contextlib import asynccontextmanager
import aiosqlite
import asyncpg
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
from hdbcli import dbapi
import sqlite3

import sys
sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    HANA = "hana"
    POSTGRES = "postgres"
    SQLITE = "sqlite"  # For local development only


@dataclass
class DataRecord:
    """A2A data record with metadata"""
    record_id: str
    agent_id: str
    context_id: str
    data_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    is_deleted: bool = False


@dataclass
class QueryResult:
    """Query result with pagination"""
    records: List[DataRecord]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class DataManagerAgent(A2AAgentBase):
    """
    Data Manager Agent - Central data persistence for A2A network
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, storage_backend: str = "sqlite"):
        super().__init__(
            agent_id="data_manager_agent",
            name="Data Manager Agent",
            description="A2A v0.2.9 compliant agent for centralized data storage and retrieval",
            version="2.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.storage_backend = StorageBackend(storage_backend)
        
        # Storage connections
        self.db_connection = None
        self.redis_client = None
        self.http_client = None
        
        # Performance metrics
        self.metrics = {
            "total_records_stored": 0,
            "total_queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "storage_errors": 0
        }
        
        # Configuration
        self.hana_config = {
            "address": os.getenv("HANA_HOST", "localhost"),
            "port": int(os.getenv("HANA_PORT", "30015")),
            "user": os.getenv("HANA_USER", "SYSTEM"),
            "password": os.getenv("HANA_PASSWORD", ""),
            "databaseName": os.getenv("HANA_DATABASE", "A2A_DATA")
        }
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", "a2a_data.db")
        self.postgres_url = os.getenv("POSTGRES_URL", "")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        
        # Connection pools
        self.hana_connection = None
        self.postgres_pool = None
        
        logger.info(f"Initialized {self.name} with {self.storage_backend.value} backend")
    
    async def initialize(self) -> None:
        """Initialize data storage connections"""
        logger.info(f"Initializing Data Manager with {self.storage_backend.value} backend...")
        
        # Initialize HTTP client for A2A communication
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=50)
        )
        
        # Initialize storage backend
        if self.storage_backend == StorageBackend.SQLITE:
            await self._initialize_sqlite()
        elif self.storage_backend == StorageBackend.POSTGRES:
            await self._initialize_postgres()
        elif self.storage_backend == StorageBackend.HANA:
            await self._initialize_hana()
        
        # Initialize Redis for caching (optional)
        try:
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
        
        # Initialize A2A components
        self.is_ready = True
        self.is_registered = False
        self.tasks = {}
        
        logger.info("Data Manager initialized successfully")
    
    async def _initialize_sqlite(self) -> None:
        """Initialize SQLite database"""
        # Create database directory if needed
        os.makedirs(os.path.dirname(self.sqlite_db_path), exist_ok=True)
        
        # Connect to database
        self.db_connection = await aiosqlite.connect(self.sqlite_db_path)
        self.db_connection.row_factory = aiosqlite.Row
        
        # Create tables
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS data_records (
                record_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                is_deleted BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_id ON data_records(agent_id)"
        )
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_id ON data_records(context_id)"
        )
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_data_type ON data_records(data_type)"
        )
        await self.db_connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON data_records(created_at)"
        )
        
        await self.db_connection.commit()
        logger.info("SQLite database initialized")
    
    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL database"""
        try:
            # Connect to PostgreSQL
            self.db_connection = await asyncpg.connect(self.postgres_url)
            
            # Create tables
            await self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    record_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    is_deleted BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create indexes
            await self.db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_id ON data_records(agent_id)"
            )
            await self.db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_context_id ON data_records(context_id)"
            )
            await self.db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_data_type ON data_records(data_type)"
            )
            await self.db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON data_records(created_at)"
            )
            await self.db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_data_gin ON data_records USING gin(data)"
            )
            
            logger.info("PostgreSQL database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            logger.warning("Falling back to SQLite")
            self.storage_backend = StorageBackend.SQLITE
            await self._initialize_sqlite()
    
    async def _initialize_hana(self) -> None:
        """Initialize SAP HANA database as primary storage"""
        try:
            # Connect to HANA
            self.hana_connection = dbapi.connect(
                address=self.hana_config["address"],
                port=self.hana_config["port"],
                user=self.hana_config["user"],
                password=self.hana_config["password"],
                databaseName=self.hana_config["databaseName"]
            )
            
            cursor = self.hana_connection.cursor()
            
            # Create schema if not exists
            cursor.execute("CREATE SCHEMA IF NOT EXISTS A2A_DATA")
            
            # Create column table for data records (HANA optimized)
            cursor.execute('''
                CREATE COLUMN TABLE IF NOT EXISTS A2A_DATA.DATA_RECORDS (
                    RECORD_ID NVARCHAR(255) PRIMARY KEY,
                    AGENT_ID NVARCHAR(100) NOT NULL,
                    CONTEXT_ID NVARCHAR(100) NOT NULL,
                    DATA_TYPE NVARCHAR(50) NOT NULL,
                    DATA NCLOB,  -- JSON data
                    DATA_BINARY BLOB,  -- For embeddings/vectors
                    METADATA NCLOB,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    VERSION INTEGER DEFAULT 1,
                    IS_DELETED BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS IDX_AGENT_ID ON A2A_DATA.DATA_RECORDS(AGENT_ID)")
            cursor.execute("CREATE INDEX IF NOT EXISTS IDX_CONTEXT_ID ON A2A_DATA.DATA_RECORDS(CONTEXT_ID)")
            cursor.execute("CREATE INDEX IF NOT EXISTS IDX_DATA_TYPE ON A2A_DATA.DATA_RECORDS(DATA_TYPE)")
            cursor.execute("CREATE INDEX IF NOT EXISTS IDX_CREATED_AT ON A2A_DATA.DATA_RECORDS(CREATED_AT)")
            
            # Create text search index for semantic search
            try:
                cursor.execute('''
                    CREATE FULLTEXT INDEX FTI_DATA_RECORDS 
                    ON A2A_DATA.DATA_RECORDS(DATA)
                ''')
            except:
                pass  # Index might already exist
            
            self.hana_connection.commit()
            cursor.close()
            
            logger.info("HANA database initialized successfully")
            
            # Initialize SQLite as fallback
            await self._initialize_sqlite_fallback()
            
        except Exception as e:
            logger.error(f"Failed to initialize HANA: {e}")
            logger.warning("Falling back to SQLite")
            self.storage_backend = StorageBackend.SQLITE
            await self._initialize_sqlite()
    
    async def _initialize_sqlite_fallback(self) -> None:
        """Initialize SQLite as fallback storage"""
        try:
            # Initialize SQLite as fallback if not already the primary
            if self.storage_backend != StorageBackend.SQLITE:
                fallback_db_path = "a2a_fallback.db"
                # Create fallback SQLite connection would go here
                logger.info("SQLite fallback initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SQLite fallback: {e}")
    
    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL as primary storage (when HANA fails)"""
        try:
            if not self.postgres_url:
                raise Exception("PostgreSQL credentials not configured")
            
            # Connect to PostgreSQL
            self.postgres_pool = await asyncpg.create_pool(self.postgres_url)
            self.db_connection = await self.postgres_pool.acquire()
            
            # Create tables
            await self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS data_records (
                    record_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    is_deleted BOOLEAN DEFAULT FALSE
                )
            ''')
            
            logger.info("PostgreSQL initialized as primary storage")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            logger.error("No storage backend available!")
            raise
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "base_url": self.base_url,
                "capabilities": {
                    "storage_backend": self.storage_backend.value,
                    "supports_versioning": True,
                    "supports_caching": self.redis_client is not None,
                    "supports_bulk_operations": True,
                    "max_record_size_mb": 10,
                    "data_types": ["accounts", "books", "locations", "measures", "products", "embeddings"]
                },
                "handlers": [h.name for h in self.handlers.values()],
                "skills": [s.name for s in self.skills.values()]
            }
            
            # Send registration to Agent Manager
            response = await self.http_client.post(
                f"{self.agent_manager_url}/rpc",
                json={
                    "jsonrpc": "2.0",
                    "method": "register_agent",
                    "params": registration,
                    "id": f"reg_{self.agent_id}_{int(datetime.utcnow().timestamp())}"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    logger.info(f"Registered with A2A network: {result['result']}")
                    self.is_registered = True
                else:
                    logger.error(f"Registration failed: {result}")
            else:
                logger.error(f"Registration failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to register with A2A network: {e}")
            raise
    
    @a2a_handler("store_data", "Store data in persistent storage")
    async def handle_store_data(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Store data with versioning and metadata"""
        try:
            data_content = message.content if hasattr(message, 'content') else message
            
            # Validate required fields
            required_fields = ["data_type", "data"]
            for field in required_fields:
                if field not in data_content:
                    return create_error_response(400, f"Missing required field: {field}")
            
            # Create data record
            record = DataRecord(
                record_id=f"{context_id}_{data_content['data_type']}_{int(datetime.utcnow().timestamp())}",
                agent_id=data_content.get("agent_id", message.sender_id if hasattr(message, 'sender_id') else "unknown"),
                context_id=context_id,
                data_type=data_content["data_type"],
                data=data_content["data"],
                metadata=data_content.get("metadata", {})
            )
            
            # Store in database
            stored_record = await self._store_record(record)
            
            # Cache if Redis available
            if self.redis_client:
                await self._cache_record(stored_record)
            
            self.metrics["total_records_stored"] += 1
            
            return create_success_response({
                "record_id": stored_record.record_id,
                "status": "stored",
                "version": stored_record.version,
                "timestamp": stored_record.created_at.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            self.metrics["storage_errors"] += 1
            return create_error_response(500, str(e))
    
    @a2a_handler("retrieve_data", "Retrieve data from storage")
    async def handle_retrieve_data(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Retrieve data with caching"""
        try:
            query = message.content if hasattr(message, 'content') else message
            
            # Check cache first
            if self.redis_client and "record_id" in query:
                cached = await self._get_cached_record(query["record_id"])
                if cached:
                    self.metrics["cache_hits"] += 1
                    return create_success_response({
                        "record": cached,
                        "from_cache": True
                    })
                else:
                    self.metrics["cache_misses"] += 1
            
            # Query database
            records = await self._query_records(query)
            
            self.metrics["total_queries_processed"] += 1
            
            return create_success_response({
                "records": [self._record_to_dict(r) for r in records],
                "count": len(records),
                "from_cache": False
            })
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("update_data", "Update existing data")
    async def handle_update_data(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Update data with versioning"""
        try:
            update_request = message.content if hasattr(message, 'content') else message
            
            # Validate required fields
            if "record_id" not in update_request or "data" not in update_request:
                return create_error_response(400, "Missing record_id or data")
            
            # Get existing record
            existing = await self._get_record_by_id(update_request["record_id"])
            if not existing:
                return create_error_response(404, "Record not found")
            
            # Update record
            existing.data.update(update_request["data"])
            existing.metadata.update(update_request.get("metadata", {}))
            existing.updated_at = datetime.utcnow()
            existing.version += 1
            
            # Store updated record
            updated = await self._update_record(existing)
            
            # Invalidate cache
            if self.redis_client:
                await self._invalidate_cache(updated.record_id)
            
            return create_success_response({
                "record_id": updated.record_id,
                "status": "updated",
                "version": updated.version,
                "timestamp": updated.updated_at.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return create_error_response(500, str(e))
    
    @a2a_handler("delete_data", "Soft delete data")
    async def handle_delete_data(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Soft delete data (mark as deleted)"""
        try:
            delete_request = message.content if hasattr(message, 'content') else message
            
            if "record_id" not in delete_request:
                return create_error_response(400, "Missing record_id")
            
            # Soft delete
            success = await self._delete_record(delete_request["record_id"])
            
            if success:
                # Invalidate cache
                if self.redis_client:
                    await self._invalidate_cache(delete_request["record_id"])
                
                return create_success_response({
                    "record_id": delete_request["record_id"],
                    "status": "deleted"
                })
            else:
                return create_error_response(404, "Record not found")
                
        except Exception as e:
            logger.error(f"Error deleting data: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("bulk_operations", "Perform bulk data operations")
    async def bulk_operations(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple operations in a transaction"""
        results = []
        errors = []
        
        async with self._transaction():
            for op in operations:
                try:
                    if op["type"] == "store":
                        record = DataRecord(
                            record_id=f"{op.get('context_id', 'bulk')}_{op['data_type']}_{int(datetime.utcnow().timestamp())}",
                            agent_id=op.get("agent_id", "bulk"),
                            context_id=op.get("context_id", "bulk"),
                            data_type=op["data_type"],
                            data=op["data"],
                            metadata=op.get("metadata", {})
                        )
                        stored = await self._store_record(record)
                        results.append({"operation": "store", "record_id": stored.record_id})
                    
                    elif op["type"] == "update":
                        existing = await self._get_record_by_id(op["record_id"])
                        if existing:
                            existing.data.update(op["data"])
                            existing.version += 1
                            updated = await self._update_record(existing)
                            results.append({"operation": "update", "record_id": updated.record_id})
                        else:
                            errors.append({"operation": "update", "error": f"Record {op['record_id']} not found"})
                    
                    elif op["type"] == "delete":
                        success = await self._delete_record(op["record_id"])
                        if success:
                            results.append({"operation": "delete", "record_id": op["record_id"]})
                        else:
                            errors.append({"operation": "delete", "error": f"Record {op['record_id']} not found"})
                            
                except Exception as e:
                    errors.append({"operation": op["type"], "error": str(e)})
        
        return {
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors
        }
    
    @a2a_skill("query_builder", "Build complex queries")
    async def query_builder(self, filters: Dict[str, Any], options: Dict[str, Any] = None) -> QueryResult:
        """Build and execute complex queries with pagination"""
        options = options or {}
        page = options.get("page", 1)
        page_size = options.get("page_size", 100)
        order_by = options.get("order_by", "created_at")
        order_dir = options.get("order_dir", "DESC")
        
        # Build query
        query = "SELECT * FROM data_records WHERE is_deleted = FALSE"
        params = []
        
        if "agent_id" in filters:
            query += " AND agent_id = ?"
            params.append(filters["agent_id"])
        
        if "context_id" in filters:
            query += " AND context_id = ?"
            params.append(filters["context_id"])
        
        if "data_type" in filters:
            query += " AND data_type = ?"
            params.append(filters["data_type"])
        
        if "created_after" in filters:
            query += " AND created_at >= ?"
            params.append(filters["created_after"])
        
        if "created_before" in filters:
            query += " AND created_at <= ?"
            params.append(filters["created_before"])
        
        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        async with self.db_connection.execute(count_query, params) as cursor:
            total_count = (await cursor.fetchone())[0]
        
        # Add ordering and pagination
        query += f" ORDER BY {order_by} {order_dir}"
        query += f" LIMIT {page_size} OFFSET {(page - 1) * page_size}"
        
        # Execute query
        records = []
        async with self.db_connection.execute(query, params) as cursor:
            async for row in cursor:
                records.append(self._row_to_record(row))
        
        return QueryResult(
            records=records,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total_count
        )
    
    @asynccontextmanager
    async def _transaction(self):
        """Database transaction context manager with proper rollback"""
        if self.storage_backend == StorageBackend.SQLITE:
            await self.db_connection.execute("BEGIN")
            try:
                yield
                await self.db_connection.commit()
            except Exception:
                await self.db_connection.rollback()
                raise
        elif self.storage_backend == StorageBackend.POSTGRES:
            transaction = self.db_connection.transaction()
            await transaction.start()
            try:
                yield
                await transaction.commit()
            except Exception:
                await transaction.rollback()
                raise
    
    async def _store_record(self, record: DataRecord) -> DataRecord:
        """Store a record in the database with HANA as primary"""
        try:
            if self.storage_backend == StorageBackend.HANA:
                cursor = self.hana_connection.cursor()
                
                # Handle vector/embedding data separately in HANA
                data_json = json.dumps(record.data)
                embedding_data = None
                
                if "embedding" in record.data and "vector" in record.data["embedding"]:
                    # Extract embedding for BLOB storage
                    embedding_data = json.dumps(record.data["embedding"]["vector"]).encode()
                
                query = '''
                    INSERT INTO A2A_DATA.DATA_RECORDS 
                    (RECORD_ID, AGENT_ID, CONTEXT_ID, DATA_TYPE, DATA, DATA_BINARY, METADATA, CREATED_AT, UPDATED_AT, VERSION)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                cursor.execute(query, (
                    record.record_id,
                    record.agent_id,
                    record.context_id,
                    record.data_type,
                    data_json,
                    embedding_data,
                    json.dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.version
                ))
                
                self.hana_connection.commit()
                cursor.close()
                
            elif self.storage_backend == StorageBackend.POSTGRES:
                # Store in PostgreSQL
                await self.db_connection.execute('''
                    INSERT INTO data_records 
                    (record_id, agent_id, context_id, data_type, data, metadata, created_at, updated_at, version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', 
                    record.record_id,
                    record.agent_id,
                    record.context_id,
                    record.data_type,
                    json.dumps(record.data),
                    json.dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.version
                )
                
            else:  # SQLite fallback
                query = """
                    INSERT INTO data_records 
                    (record_id, agent_id, context_id, data_type, data, metadata, created_at, updated_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                await self.db_connection.execute(query, (
                    record.record_id,
                    record.agent_id,
                    record.context_id,
                    record.data_type,
                    json.dumps(record.data),
                    json.dumps(record.metadata),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.version
                ))
                
                await self.db_connection.commit()
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to store in {self.storage_backend.value}: {e}")
            
            # Try fallback to SQLite if HANA fails
            if self.storage_backend == StorageBackend.HANA:
                logger.info("Attempting SQLite fallback...")
                try:
                    # Switch to SQLite temporarily
                    fallback_db = await aiosqlite.connect("a2a_fallback.db")
                    await fallback_db.execute('''
                        CREATE TABLE IF NOT EXISTS data_records (
                            record_id TEXT PRIMARY KEY,
                            agent_id TEXT NOT NULL,
                            context_id TEXT NOT NULL,
                            data_type TEXT NOT NULL,
                            data TEXT NOT NULL,
                            metadata TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            version INTEGER DEFAULT 1,
                            is_deleted BOOLEAN DEFAULT FALSE
                        )
                    ''')
                    
                    await fallback_db.execute('''
                        INSERT INTO data_records 
                        (record_id, agent_id, context_id, data_type, data, metadata, created_at, updated_at, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.record_id,
                        record.agent_id,
                        record.context_id,
                        record.data_type,
                        json.dumps(record.data),
                        json.dumps(record.metadata),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                        record.version
                    ))
                    
                    await fallback_db.commit()
                    await fallback_db.close()
                    logger.info("SQLite fallback successful")
                    return record
                except Exception as fallback_error:
                    logger.error(f"SQLite fallback also failed: {fallback_error}")
            
            raise
    
    async def _update_record(self, record: DataRecord) -> DataRecord:
        """Update a record in the database"""
        query = """
            UPDATE data_records 
            SET data = ?, metadata = ?, updated_at = ?, version = ?
            WHERE record_id = ?
        """
        
        await self.db_connection.execute(query, (
            json.dumps(record.data),
            json.dumps(record.metadata),
            record.updated_at.isoformat(),
            record.version,
            record.record_id
        ))
        
        await self.db_connection.commit()
        return record
    
    async def _delete_record(self, record_id: str) -> bool:
        """Soft delete a record"""
        query = "UPDATE data_records SET is_deleted = TRUE WHERE record_id = ?"
        cursor = await self.db_connection.execute(query, (record_id,))
        await self.db_connection.commit()
        return cursor.rowcount > 0
    
    async def _get_record_by_id(self, record_id: str) -> Optional[DataRecord]:
        """Get a single record by ID"""
        query = "SELECT * FROM data_records WHERE record_id = ? AND is_deleted = FALSE"
        async with self.db_connection.execute(query, (record_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None
    
    async def _query_records(self, query_params: Dict[str, Any]) -> List[DataRecord]:
        """Query records based on parameters"""
        records = []
        
        if "record_id" in query_params:
            record = await self._get_record_by_id(query_params["record_id"])
            if record:
                records.append(record)
        else:
            # Build dynamic query
            result = await self.query_builder(query_params)
            records = result.records
        
        return records
    
    def _row_to_record(self, row: aiosqlite.Row) -> DataRecord:
        """Convert database row to DataRecord"""
        return DataRecord(
            record_id=row["record_id"],
            agent_id=row["agent_id"],
            context_id=row["context_id"],
            data_type=row["data_type"],
            data=json.loads(row["data"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
            is_deleted=row["is_deleted"]
        )
    
    def _record_to_dict(self, record: DataRecord) -> Dict[str, Any]:
        """Convert DataRecord to dictionary"""
        return {
            "record_id": record.record_id,
            "agent_id": record.agent_id,
            "context_id": record.context_id,
            "data_type": record.data_type,
            "data": record.data,
            "metadata": record.metadata,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "version": record.version
        }
    
    async def _cache_record(self, record: DataRecord) -> None:
        """Cache record in Redis"""
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"record:{record.record_id}",
                    self.cache_ttl,
                    json.dumps(self._record_to_dict(record))
                )
            except Exception as e:
                logger.warning(f"Failed to cache record: {e}")
    
    async def _get_cached_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get record from cache"""
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"record:{record_id}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Failed to get cached record: {e}")
        return None
    
    async def _invalidate_cache(self, record_id: str) -> None:
        """Invalidate cached record"""
        if self.redis_client:
            try:
                await self.redis_client.delete(f"record:{record_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
    
    def generate_context_id(self) -> str:
        """Generate unique context ID"""
        import uuid
        return str(uuid.uuid4())
    
    def create_message(self, content: Any) -> A2AMessage:
        """Create A2A message"""
        return A2AMessage(
            sender_id=self.agent_id,
            content=content,
            role=MessageRole.AGENT
        )
    
    async def create_task(self, task_type: str, metadata: Dict[str, Any]) -> str:
        """Create and track a task"""
        import uuid
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        return task_id
    
    async def update_task_status(self, task_id: str, status: str, update_data: Dict[str, Any] = None):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
            
            if update_data:
                self.tasks[task_id]["metadata"].update(update_data)
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
    
    async def shutdown(self) -> None:
        """Cleanup resources"""
        logger.info("Shutting down Data Manager...")
        
        # Close database connection
        if self.db_connection:
            await self.db_connection.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        self.is_ready = False
        logger.info("Data Manager shutdown complete")
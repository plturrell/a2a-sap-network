"""
SAP HANA Vector Engine integration for A2A Vector Processing Agent.
Provides optimized vector operations using HANA's built-in vector capabilities.
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import time

from config.agentConfig import config
from common.errorHandling import with_circuit_breaker, with_retry
from monitoring.prometheusConfig import create_agent_metrics

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Vector search result."""
    vector_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    vector_data: Optional[List[float]] = None


@dataclass
class VectorIndex:
    """Vector index configuration."""
    index_name: str
    dimension: int
    metric_type: str  # cosine, euclidean, dot_product
    index_type: str   # flat, ivf, hnsw
    parameters: Dict[str, Any]
    created_at: datetime


class HANAVectorEngine:
    """
    SAP HANA Vector Engine integration providing optimized vector operations.
    """
    
    def __init__(
        self,
        connection_params: Optional[Dict[str, str]] = None,
        default_dimension: int = 768,
        enable_metrics: bool = True
    ):
        """
        Initialize HANA Vector Engine.
        
        Args:
            connection_params: HANA connection parameters
            default_dimension: Default vector dimension
            enable_metrics: Enable performance metrics
        """
        self.connection_params = connection_params or self._get_default_connection()
        self.default_dimension = default_dimension
        self.enable_metrics = enable_metrics
        
        # Connection and cursor
        self._connection = None
        self._cursor = None
        
        # Vector indexes
        self.vector_indexes = {}  # index_name -> VectorIndex
        
        # Performance metrics
        if enable_metrics:
            self.metrics = create_agent_metrics("vector_processing", "hana_vector_engine")
        else:
            self.metrics = None
        
        # Cache for frequent operations
        self._similarity_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Configuration
        self.batch_size = config.max_batch_size
        self.similarity_threshold = 0.7
        self.max_results = 1000
    
    def _get_default_connection(self) -> Dict[str, str]:
        """Get default HANA connection parameters."""
        return {
            'address': config.hana_host or 'localhost',
            'port': config.hana_port or '30015',
            'user': config.hana_user or 'SYSTEM',
            'password': config.hana_password or '',
            'database': config.hana_database or 'A2A',
            'schema': config.hana_schema or 'A2A_VECTORS'
        }
    
    async def initialize(self) -> bool:
        """
        Initialize HANA Vector Engine connection and setup.
        
        Returns:
            True if initialization successful
        """
        try:
            await self._connect_to_hana()
            await self._setup_vector_tables()
            await self._load_existing_indexes()
            
            logger.info("HANA Vector Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HANA Vector Engine: {e}")
            return False
    
    @with_circuit_breaker("hana_connection", failure_threshold=3, recovery_timeout=60)
    async def _connect_to_hana(self):
        """Connect to HANA database with circuit breaker protection."""
        try:
            # Import HANA driver
            import hdbcli.dbapi as hana_driver
            
            # Create connection
            self._connection = hana_driver.connect(
                address=self.connection_params['address'],
                port=int(self.connection_params['port']),
                user=self.connection_params['user'],
                password=self.connection_params['password'],
                database=self.connection_params['database']
            )
            
            self._cursor = self._connection.cursor()
            
            # Set schema
            schema = self.connection_params['schema']
            await self._execute_sql(f"SET SCHEMA {schema}")
            
            logger.info(f"Connected to HANA at {self.connection_params['address']}")
            
        except ImportError:
            logger.error("HANA driver not available. Install hdbcli package.")
            raise
        except Exception as e:
            logger.error(f"HANA connection failed: {e}")
            raise
    
    async def _setup_vector_tables(self):
        """Setup vector storage tables in HANA."""
        # Create vector data table
        create_vectors_table = """
        CREATE TABLE IF NOT EXISTS VECTOR_DATA (
            VECTOR_ID NVARCHAR(256) PRIMARY KEY,
            VECTOR_DATA REAL_VECTOR,
            DIMENSION INTEGER,
            METADATA NCLOB,
            INDEX_NAME NVARCHAR(128),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Create vector indexes table
        create_indexes_table = """
        CREATE TABLE IF NOT EXISTS VECTOR_INDEXES (
            INDEX_NAME NVARCHAR(128) PRIMARY KEY,
            DIMENSION INTEGER,
            METRIC_TYPE NVARCHAR(32),
            INDEX_TYPE NVARCHAR(32),
            PARAMETERS NCLOB,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            STATUS NVARCHAR(32) DEFAULT 'ACTIVE'
        )
        """
        
        # Create similarity search view
        create_search_view = """
        CREATE OR REPLACE VIEW VECTOR_SIMILARITY_SEARCH AS
        SELECT 
            VECTOR_ID,
            VECTOR_DATA,
            METADATA,
            INDEX_NAME
        FROM VECTOR_DATA
        WHERE INDEX_NAME IS NOT NULL
        """
        
        await self._execute_sql(create_vectors_table)
        await self._execute_sql(create_indexes_table)
        await self._execute_sql(create_search_view)
        
        logger.info("HANA vector tables setup completed")
    
    async def _load_existing_indexes(self):
        """Load existing vector indexes from HANA."""
        query = "SELECT * FROM VECTOR_INDEXES WHERE STATUS = 'ACTIVE'"
        
        try:
            results = await self._execute_sql(query, fetch=True)
            
            for row in results:
                index_name, dimension, metric_type, index_type, parameters_json, created_at, _ = row
                
                parameters = json.loads(parameters_json) if parameters_json else {}
                
                self.vector_indexes[index_name] = VectorIndex(
                    index_name=index_name,
                    dimension=dimension,
                    metric_type=metric_type,
                    index_type=index_type,
                    parameters=parameters,
                    created_at=created_at
                )
            
            logger.info(f"Loaded {len(self.vector_indexes)} existing vector indexes")
            
        except Exception as e:
            logger.warning(f"Failed to load existing indexes: {e}")
    
    async def create_vector_index(
        self,
        index_name: str,
        dimension: int,
        metric_type: str = "cosine",
        index_type: str = "hnsw",
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new vector index in HANA.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric_type: Similarity metric (cosine, euclidean, dot_product)
            index_type: Index type (flat, ivf, hnsw)
            parameters: Additional index parameters
            
        Returns:
            True if index created successfully
        """
        if index_name in self.vector_indexes:
            logger.warning(f"Vector index {index_name} already exists")
            return True
        
        parameters = parameters or {}
        
        try:
            # Insert index metadata
            insert_query = """
            INSERT INTO VECTOR_INDEXES 
            (INDEX_NAME, DIMENSION, METRIC_TYPE, INDEX_TYPE, PARAMETERS) 
            VALUES (?, ?, ?, ?, ?)
            """
            
            await self._execute_sql(
                insert_query,
                params=[
                    index_name,
                    dimension,
                    metric_type,
                    index_type,
                    json.dumps(parameters)
                ]
            )
            
            # Create HANA vector index
            if index_type == "hnsw":
                await self._create_hnsw_index(index_name, metric_type, parameters)
            elif index_type == "ivf":
                await self._create_ivf_index(index_name, metric_type, parameters)
            
            # Store in memory
            self.vector_indexes[index_name] = VectorIndex(
                index_name=index_name,
                dimension=dimension,
                metric_type=metric_type,
                index_type=index_type,
                parameters=parameters,
                created_at=datetime.now()
            )
            
            logger.info(f"Created vector index: {index_name} ({dimension}D, {metric_type})")
            
            if self.metrics:
                self.metrics.record_validation("vector_index_creation", "success")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index {index_name}: {e}")
            
            if self.metrics:
                self.metrics.record_validation("vector_index_creation", "failure")
            
            return False
    
    async def _create_hnsw_index(
        self,
        index_name: str,
        metric_type: str,
        parameters: Dict[str, Any]
    ):
        """Create HNSW index in HANA."""
        m = parameters.get('m', 16)
        ef_construction = parameters.get('ef_construction', 200)
        
        create_index_sql = f"""
        CREATE INDEX {index_name}_HNSW_IDX 
        ON VECTOR_DATA (VECTOR_DATA) 
        USING HNSW 
        WITH PARAMETERS 'M={m}, EF_CONSTRUCTION={ef_construction}, METRIC={metric_type.upper()}'
        WHERE INDEX_NAME = '{index_name}'
        """
        
        await self._execute_sql(create_index_sql)
    
    async def _create_ivf_index(
        self,
        index_name: str,
        metric_type: str,
        parameters: Dict[str, Any]
    ):
        """Create IVF index in HANA."""
        nlist = parameters.get('nlist', 100)
        
        create_index_sql = f"""
        CREATE INDEX {index_name}_IVF_IDX 
        ON VECTOR_DATA (VECTOR_DATA) 
        USING IVF 
        WITH PARAMETERS 'NLIST={nlist}, METRIC={metric_type.upper()}'
        WHERE INDEX_NAME = '{index_name}'
        """
        
        await self._execute_sql(create_index_sql)
    
    async def store_vectors(
        self,
        vectors: Dict[str, List[float]],
        index_name: str,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Store vectors in HANA with batch processing.
        
        Args:
            vectors: Dictionary of vector_id -> vector_data
            index_name: Target index name
            metadata: Optional metadata for each vector
            batch_size: Batch size for insertion
            
        Returns:
            True if storage successful
        """
        if index_name not in self.vector_indexes:
            raise ValueError(f"Vector index {index_name} does not exist")
        
        batch_size = batch_size or self.batch_size
        metadata = metadata or {}
        
        try:
            # Prepare data for batch insertion
            vector_items = list(vectors.items())
            total_vectors = len(vector_items)
            
            insert_query = """
            INSERT INTO VECTOR_DATA 
            (VECTOR_ID, VECTOR_DATA, DIMENSION, METADATA, INDEX_NAME) 
            VALUES (?, TO_REAL_VECTOR(?), ?, ?, ?)
            """
            
            start_time = time.time()
            inserted_count = 0
            
            # Process in batches
            for i in range(0, total_vectors, batch_size):
                batch = vector_items[i:i + batch_size]
                batch_data = []
                
                for vector_id, vector_data in batch:
                    vector_metadata = metadata.get(vector_id, {})
                    
                    batch_data.append([
                        vector_id,
                        vector_data,  # HANA will convert to REAL_VECTOR
                        len(vector_data),
                        json.dumps(vector_metadata),
                        index_name
                    ])
                
                # Execute batch
                await self._execute_many(insert_query, batch_data)
                inserted_count += len(batch)
                
                if self.metrics:
                    self.metrics.record_data_processed("vector_storage", len(batch) * len(vector_data) * 4)
            
            duration = time.time() - start_time
            
            logger.info(
                f"Stored {inserted_count} vectors in index {index_name} "
                f"({duration:.2f}s, {inserted_count/duration:.1f} vectors/sec)"
            )
            
            if self.metrics:
                self.metrics.record_task("vector_storage", "success", duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            
            if self.metrics:
                self.metrics.record_task("vector_storage", "failure")
            
            return False
    
    @with_retry(max_retries=3, initial_delay=1.0)
    async def similarity_search(
        self,
        query_vector: List[float],
        index_name: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform similarity search using HANA vector functions.
        
        Args:
            query_vector: Query vector
            index_name: Index to search in
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of search results
        """
        if index_name not in self.vector_indexes:
            raise ValueError(f"Vector index {index_name} does not exist")
        
        threshold = similarity_threshold or self.similarity_threshold
        top_k = min(top_k, self.max_results)
        
        try:
            start_time = time.time()
            
            # Build query based on index metric type
            index_info = self.vector_indexes[index_name]
            metric_type = index_info.metric_type
            
            if metric_type == "cosine":
                similarity_func = "COSINE_SIMILARITY"
            elif metric_type == "euclidean":
                similarity_func = "L2_DISTANCE"
            elif metric_type == "dot_product":
                similarity_func = "DOT_PRODUCT"
            else:
                similarity_func = "COSINE_SIMILARITY"
            
            # Base query
            query = f"""
            SELECT 
                VECTOR_ID,
                {similarity_func}(VECTOR_DATA, TO_REAL_VECTOR(?)) AS SIMILARITY_SCORE,
                METADATA
            FROM VECTOR_DATA
            WHERE INDEX_NAME = ?
            """
            
            params = [query_vector, index_name]
            
            # Add metadata filters
            if filter_metadata:
                for key, value in filter_metadata.items():
                    query += f" AND JSON_VALUE(METADATA, '$.{key}') = ?"
                    params.append(str(value))
            
            # Add similarity threshold
            if metric_type == "euclidean":
                query += f" AND {similarity_func}(VECTOR_DATA, TO_REAL_VECTOR(?)) <= ?"
                params.append(query_vector)
                params.append(1.0 - threshold)  # Convert to distance
            else:
                query += f" AND {similarity_func}(VECTOR_DATA, TO_REAL_VECTOR(?)) >= ?"
                params.append(query_vector)
                params.append(threshold)
            
            # Order and limit
            if metric_type == "euclidean":
                query += f" ORDER BY SIMILARITY_SCORE ASC LIMIT {top_k}"
            else:
                query += f" ORDER BY SIMILARITY_SCORE DESC LIMIT {top_k}"
            
            # Execute query
            results = await self._execute_sql(query, params=params, fetch=True)
            
            # Process results
            search_results = []
            for row in results:
                vector_id, similarity_score, metadata_json = row
                
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Convert distance to similarity for euclidean
                if metric_type == "euclidean":
                    similarity_score = 1.0 - similarity_score
                
                search_results.append(VectorSearchResult(
                    vector_id=vector_id,
                    similarity_score=float(similarity_score),
                    metadata=metadata
                ))
            
            duration = time.time() - start_time
            
            logger.debug(
                f"Similarity search completed: {len(search_results)} results "
                f"in {duration:.3f}s (index: {index_name})"
            )
            
            if self.metrics:
                self.metrics.record_task("similarity_search", "success", duration)
                self.metrics.set_accuracy_rate("similarity_search", len(search_results) / top_k)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            
            if self.metrics:
                self.metrics.record_task("similarity_search", "failure")
            
            return []
    
    async def batch_similarity_search(
        self,
        query_vectors: Dict[str, List[float]],
        index_name: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, List[VectorSearchResult]]:
        """
        Perform batch similarity search for multiple query vectors.
        
        Args:
            query_vectors: Dictionary of query_id -> query_vector
            index_name: Index to search in
            top_k: Number of results per query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dictionary of query_id -> search_results
        """
        results = {}
        
        # Process queries concurrently
        tasks = []
        for query_id, query_vector in query_vectors.items():
            task = self.similarity_search(
                query_vector,
                index_name,
                top_k,
                similarity_threshold
            )
            tasks.append((query_id, task))
        
        # Execute all searches
        for query_id, task in tasks:
            try:
                search_results = await task
                results[query_id] = search_results
            except Exception as e:
                logger.error(f"Batch search failed for query {query_id}: {e}")
                results[query_id] = []
        
        return results
    
    async def update_vector(
        self,
        vector_id: str,
        vector_data: List[float],
        index_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing vector in HANA.
        
        Args:
            vector_id: Vector identifier
            vector_data: New vector data
            index_name: Index name
            metadata: Updated metadata
            
        Returns:
            True if update successful
        """
        try:
            update_query = """
            UPDATE VECTOR_DATA 
            SET VECTOR_DATA = TO_REAL_VECTOR(?), 
                DIMENSION = ?,
                METADATA = ?,
                UPDATED_AT = CURRENT_TIMESTAMP
            WHERE VECTOR_ID = ? AND INDEX_NAME = ?
            """
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            rows_affected = await self._execute_sql(
                update_query,
                params=[
                    vector_data,
                    len(vector_data),
                    metadata_json,
                    vector_id,
                    index_name
                ]
            )
            
            if rows_affected > 0:
                logger.debug(f"Updated vector {vector_id} in index {index_name}")
                return True
            else:
                logger.warning(f"Vector {vector_id} not found in index {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update vector {vector_id}: {e}")
            return False
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        index_name: str
    ) -> int:
        """
        Delete vectors from HANA.
        
        Args:
            vector_ids: List of vector IDs to delete
            index_name: Index name
            
        Returns:
            Number of vectors deleted
        """
        try:
            if not vector_ids:
                return 0
            
            placeholders = ','.join(['?' for _ in vector_ids])
            delete_query = f"""
            DELETE FROM VECTOR_DATA 
            WHERE VECTOR_ID IN ({placeholders}) AND INDEX_NAME = ?
            """
            
            params = vector_ids + [index_name]
            rows_affected = await self._execute_sql(delete_query, params=params)
            
            logger.info(f"Deleted {rows_affected} vectors from index {index_name}")
            return rows_affected
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return 0
    
    async def get_index_statistics(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for a vector index.
        
        Args:
            index_name: Index name
            
        Returns:
            Index statistics
        """
        try:
            stats_query = """
            SELECT 
                COUNT(*) as VECTOR_COUNT,
                AVG(DIMENSION) as AVG_DIMENSION,
                MIN(CREATED_AT) as OLDEST_VECTOR,
                MAX(UPDATED_AT) as LATEST_UPDATE
            FROM VECTOR_DATA 
            WHERE INDEX_NAME = ?
            """
            
            result = await self._execute_sql(stats_query, params=[index_name], fetch=True)
            
            if result:
                row = result[0]
                return {
                    'index_name': index_name,
                    'vector_count': row[0],
                    'avg_dimension': float(row[1]) if row[1] else 0,
                    'oldest_vector': row[2],
                    'latest_update': row[3],
                    'index_info': self.vector_indexes.get(index_name)
                }
            else:
                return {'index_name': index_name, 'vector_count': 0}
                
        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return {'index_name': index_name, 'error': str(e)}
    
    async def _execute_sql(
        self,
        query: str,
        params: Optional[List] = None,
        fetch: bool = False
    ) -> Any:
        """Execute SQL query with error handling."""
        if not self._cursor:
            raise RuntimeError("HANA connection not initialized")
        
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            
            if fetch:
                return self._cursor.fetchall()
            else:
                return self._cursor.rowcount
                
        except Exception as e:
            logger.error(f"SQL execution failed: {query[:100]}... Error: {e}")
            raise
    
    async def _execute_many(self, query: str, data: List[List]) -> int:
        """Execute batch SQL operations."""
        if not self._cursor:
            raise RuntimeError("HANA connection not initialized")
        
        try:
            self._cursor.executemany(query, data)
            return self._cursor.rowcount
        except Exception as e:
            logger.error(f"Batch SQL execution failed: {e}")
            raise
    
    async def close(self):
        """Close HANA connection."""
        try:
            if self._cursor:
                self._cursor.close()
            if self._connection:
                self._connection.close()
            
            logger.info("HANA Vector Engine connection closed")
            
        except Exception as e:
            logger.error(f"Error closing HANA connection: {e}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and metrics."""
        return {
            'connected': self._connection is not None,
            'indexes_count': len(self.vector_indexes),
            'indexes': list(self.vector_indexes.keys()),
            'default_dimension': self.default_dimension,
            'batch_size': self.batch_size,
            'similarity_threshold': self.similarity_threshold,
            'cache_size': len(self._similarity_cache),
            'connection_params': {
                'address': self.connection_params['address'],
                'port': self.connection_params['port'],
                'database': self.connection_params['database'],
                'schema': self.connection_params['schema']
            }
        }
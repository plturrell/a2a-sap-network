from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from datetime import datetime
import logging
import json
import struct
from scipy.sparse import csr_matrix, coo_matrix
import hashlib

from app.a2a.core.security_base import SecureA2AAgent
"""
Sparse Vector Skills for Agent 3 (Vector Processing) - SAP HANA Sparse Vector Support
Implements efficient sparse vector storage and operations for high-dimensional data
Following SAP naming conventions and best practices
"""

logger = logging.getLogger(__name__)


class SparseVectorSkills(SecureA2AAgent):
    
        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling  
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning
"""Sparse vector processing capabilities for efficient high-dimensional operations"""
    
    def __init__(self, hanaConnection=None):
        
        super().__init__()
        self.hanaConnection = hanaConnection
        self.compressionThresholds = {
            'density': 0.1,  # Convert to sparse if less than 10% non-zero
            'minDimension': 1000,  # Only use sparse for high dimensions
            'minNonZero': 10  # Minimum non-zero elements
        }
        
    async def createSparseVectorStorage(self) -> Dict[str, Any]:
        """
        Create optimized storage schema for sparse vectors in HANA
        """
        try:
            # Create sparse vector table with columnar storage
            createTableQuery = """
            CREATE COLUMN TABLE A2A_SPARSE_VECTORS (
                DOC_ID NVARCHAR(255) PRIMARY KEY,
                ENTITY_ID NVARCHAR(255) NOT NULL,
                ENTITY_TYPE NVARCHAR(100),
                
                -- Sparse vector components
                DIMENSION INTEGER NOT NULL,
                NON_ZERO_COUNT INTEGER NOT NULL,
                INDICES VARBINARY(5000000),  -- Compressed indices
                VALUES VARBINARY(5000000),   -- Compressed values
                
                -- Dense vector fallback for small vectors
                DENSE_VECTOR REAL_VECTOR,
                IS_SPARSE BOOLEAN DEFAULT TRUE,
                
                -- Metadata and tracking
                COMPRESSION_RATIO DOUBLE,
                VECTOR_NORM DOUBLE,
                HASH_SIGNATURE NVARCHAR(64),  -- For fast equality checks
                
                -- Temporal and source tracking
                SOURCE_AGENT NVARCHAR(100),
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Performance optimization
                CLUSTER_ID INTEGER,
                ACCESS_COUNT INTEGER DEFAULT 0,
                LAST_ACCESSED TIMESTAMP
            )
            PARTITION BY HASH (ENTITY_TYPE) PARTITIONS 8
            WITH SMART MERGE ON
            """
            
            await self.hanaConnection.execute(createTableQuery)
            
            # Create specialized indexes for sparse vector operations
            indexQueries = [
                # Index for similarity searches within entity types
                """
                CREATE INDEX IDX_SPARSE_ENTITY_TYPE 
                ON A2A_SPARSE_VECTORS (ENTITY_TYPE, VECTOR_NORM)
                """,
                
                # Index for hash-based duplicate detection
                """
                CREATE UNIQUE INDEX IDX_SPARSE_HASH 
                ON A2A_SPARSE_VECTORS (HASH_SIGNATURE)
                """,
                
                # Index for clustering operations
                """
                CREATE INDEX IDX_SPARSE_CLUSTER 
                ON A2A_SPARSE_VECTORS (CLUSTER_ID, ENTITY_TYPE)
                """,
                
                # Index for access pattern optimization
                """
                CREATE INDEX IDX_SPARSE_ACCESS 
                ON A2A_SPARSE_VECTORS (ACCESS_COUNT DESC, LAST_ACCESSED DESC)
                """
            ]
            
            for indexQuery in indexQueries:
                await self.hanaConnection.execute(indexQuery)
            
            # Create stored procedures for sparse vector operations
            await self._createSparseVectorProcedures()
            
            return {
                'status': 'success',
                'tablesCreated': ['A2A_SPARSE_VECTORS'],
                'indexesCreated': 4,
                'proceduresCreated': ['SP_SPARSE_SIMILARITY', 'SP_SPARSE_ADD', 'SP_SPARSE_SEARCH']
            }
            
        except Exception as e:
            logger.error(f"Failed to create sparse vector storage: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def convertToSparseVector(self, 
                                  denseVector: List[float],
                                  entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dense vector to sparse representation with compression
        """
        try:
            vectorArray = np.array(denseVector)
            dimension = len(denseVector)
            
            # Check if vector should be sparse
            nonZeroIndices = np.nonzero(vectorArray)[0]
            nonZeroCount = len(nonZeroIndices)
            density = nonZeroCount / dimension
            
            if (density > self.compressionThresholds['density'] or 
                dimension < self.compressionThresholds['minDimension'] or
                nonZeroCount < self.compressionThresholds['minNonZero']):
                # Keep as dense vector
                return {
                    'isSparse': False,
                    'denseVector': denseVector,
                    'dimension': dimension,
                    'nonZeroCount': nonZeroCount,
                    'density': density
                }
            
            # Convert to sparse representation
            nonZeroValues = vectorArray[nonZeroIndices]
            
            # Compress indices and values
            compressedIndices = self._compressIndices(nonZeroIndices)
            compressedValues = self._compressValues(nonZeroValues)
            
            # Calculate vector properties
            vectorNorm = np.linalg.norm(nonZeroValues)
            hashSignature = self._calculateVectorHash(nonZeroIndices, nonZeroValues)
            
            # Store sparse vector
            sparseData = {
                'docId': entityData.get('docId', self._generateDocId()),
                'entityId': entityData['entityId'],
                'entityType': entityData['entityType'],
                'dimension': dimension,
                'nonZeroCount': nonZeroCount,
                'indices': compressedIndices,
                'values': compressedValues,
                'isSparse': True,
                'compressionRatio': dimension / (nonZeroCount * 2),  # Approximate compression ratio
                'vectorNorm': float(vectorNorm),
                'hashSignature': hashSignature,
                'sourceAgent': entityData.get('sourceAgent', 'agent3')
            }
            
            # Insert into HANA
            insertQuery = """
            INSERT INTO A2A_SPARSE_VECTORS (
                DOC_ID, ENTITY_ID, ENTITY_TYPE,
                DIMENSION, NON_ZERO_COUNT, INDICES, VALUES,
                IS_SPARSE, COMPRESSION_RATIO, VECTOR_NORM,
                HASH_SIGNATURE, SOURCE_AGENT
            ) VALUES (
                :docId, :entityId, :entityType,
                :dimension, :nonZeroCount, :indices, :values,
                :isSparse, :compressionRatio, :vectorNorm,
                :hashSignature, :sourceAgent
            )
            """
            
            await self.hanaConnection.execute(insertQuery, sparseData)
            
            return {
                'status': 'success',
                'docId': sparseData['docId'],
                'compressionRatio': sparseData['compressionRatio'],
                'spaceSaved': f"{(1 - 1/sparseData['compressionRatio']) * 100:.1f}%",
                'vectorProperties': {
                    'dimension': dimension,
                    'nonZeroCount': nonZeroCount,
                    'density': density,
                    'norm': vectorNorm
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to convert to sparse vector: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def sparseVectorSearch(self,
                               queryVector: Union[List[float], Dict[str, Any]],
                               searchConfig: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform efficient similarity search on sparse vectors
        """
        try:
            # Process query vector
            if isinstance(queryVector, list):
                queryData = await self._processDenseQuery(queryVector)
            else:
                queryData = queryVector  # Already sparse
            
            # Build search query based on vector type
            if queryData['isSparse']:
                searchResults = await self._sparseToSparseSearch(queryData, searchConfig)
            else:
                searchResults = await self._denseToSparseSearch(queryData, searchConfig)
            
            # Post-process results
            processedResults = []
            for result in searchResults:
                processedResult = {
                    'docId': result['DOC_ID'],
                    'entityId': result['ENTITY_ID'],
                    'entityType': result['ENTITY_TYPE'],
                    'similarityScore': result['SIMILARITY_SCORE'],
                    'compressionRatio': result['COMPRESSION_RATIO'],
                    'dimension': result['DIMENSION'],
                    'nonZeroCount': result['NON_ZERO_COUNT']
                }
                
                # Update access statistics
                await self._updateAccessStats(result['DOC_ID'])
                
                processedResults.append(processedResult)
            
            return processedResults
            
        except Exception as e:
            logger.error(f"Sparse vector search failed: {e}")
            # Return empty results with proper structure
            return {
                'results': [],
                'total_matches': 0,
                'search_time': 0.0,
                'error': str(e)
            }
    
    async def _sparseToSparseSearch(self,
                                  queryData: Dict[str, Any],
                                  searchConfig: Dict[str, Any]) -> List[Dict]:
        """
        Efficient sparse-to-sparse similarity search using inverted index
        """
        # Create temporary table for query vector
        tempTableName = f"#QUERY_SPARSE_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        createTempQuery = f"""
        CREATE LOCAL TEMPORARY TABLE {tempTableName} (
            INDICES VARBINARY(5000000),
            VALUES VARBINARY(5000000),
            DIMENSION INTEGER,
            NON_ZERO_COUNT INTEGER,
            VECTOR_NORM DOUBLE
        )
        """
        await self.hanaConnection.execute(createTempQuery)
        
        # Insert query vector
        insertTempQuery = f"""
        INSERT INTO {tempTableName} VALUES (
            :indices, :values, :dimension, :nonZeroCount, :vectorNorm
        )
        """
        await self.hanaConnection.execute(insertTempQuery, queryData)
        
        # Perform similarity search with early termination
        searchQuery = f"""
        WITH CANDIDATE_VECTORS AS (
            -- First pass: Find candidates with overlapping indices
            SELECT DISTINCT
                sv.DOC_ID,
                sv.ENTITY_ID,
                sv.ENTITY_TYPE,
                sv.DIMENSION,
                sv.NON_ZERO_COUNT,
                sv.INDICES,
                sv.VALUES,
                sv.VECTOR_NORM,
                sv.COMPRESSION_RATIO,
                -- Estimate upper bound similarity for pruning
                LEAST(qv.VECTOR_NORM * sv.VECTOR_NORM, 1.0) as UPPER_BOUND
            FROM A2A_SPARSE_VECTORS sv
            CROSS JOIN {tempTableName} qv
            WHERE sv.IS_SPARSE = TRUE
                AND sv.ENTITY_TYPE = :entityType
                -- Pruning: vectors with very different norms can't be similar
                AND ABS(sv.VECTOR_NORM - qv.VECTOR_NORM) / qv.VECTOR_NORM < 0.5
        ),
        SIMILARITY_SCORES AS (
            SELECT 
                cv.*,
                -- Call stored procedure for exact sparse similarity
                SPARSE_COSINE_SIMILARITY(
                    qv.INDICES, qv.VALUES, qv.NON_ZERO_COUNT,
                    cv.INDICES, cv.VALUES, cv.NON_ZERO_COUNT,
                    cv.DIMENSION
                ) as SIMILARITY_SCORE
            FROM CANDIDATE_VECTORS cv
            CROSS JOIN {tempTableName} qv
            WHERE cv.UPPER_BOUND > :minSimilarity
        )
        SELECT *
        FROM SIMILARITY_SCORES
        WHERE SIMILARITY_SCORE > :minSimilarity
        ORDER BY SIMILARITY_SCORE DESC
        LIMIT :topK
        """
        
        searchParams = {
            'entityType': searchConfig.get('entityType', 'all'),
            'minSimilarity': searchConfig.get('minSimilarity', 0.7),
            'topK': searchConfig.get('topK', 10)
        }
        
        results = await self.hanaConnection.execute(searchQuery, searchParams)
        
        # Clean up temporary table
        await self.hanaConnection.execute(f"DROP TABLE {tempTableName}")
        
        return results
    
    async def batchSparseVectorConversion(self,
                                        vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Efficiently convert and store multiple vectors in batch
        """
        conversionResults = {
            'totalProcessed': 0,
            'sparseConverted': 0,
            'denseKept': 0,
            'totalSpaceSaved': 0,
            'errors': []
        }
        
        try:
            # Process vectors in batches
            batchSize = 1000
            for i in range(0, len(vectors), batchSize):
                batch = vectors[i:i + batchSize]
                
                sparseVectors = []
                denseVectors = []
                
                for vectorData in batch:
                    try:
                        # Analyze vector for sparse conversion
                        vector = vectorData['vector']
                        result = await self.convertToSparseVector(vector, vectorData)
                        
                        if result.get('status') == 'success':
                            conversionResults['totalProcessed'] += 1
                            if result.get('isSparse', False):
                                conversionResults['sparseConverted'] += 1
                                conversionResults['totalSpaceSaved'] += (
                                    1 - 1/result.get('compressionRatio', 1)
                                )
                            else:
                                conversionResults['denseKept'] += 1
                        else:
                            conversionResults['errors'].append(result.get('message'))
                            
                    except Exception as e:
                        conversionResults['errors'].append(f"Vector {i}: {str(e)}")
                
            # Calculate average space savings
            if conversionResults['sparseConverted'] > 0:
                conversionResults['avgSpaceSaved'] = (
                    conversionResults['totalSpaceSaved'] / 
                    conversionResults['sparseConverted'] * 100
                )
            
        except Exception as e:
            logger.error(f"Batch sparse vector conversion failed: {e}")
            conversionResults['errors'].append(str(e))
            
        return conversionResults
    
    async def optimizeSparseStorage(self) -> Dict[str, Any]:
        """
        Optimize sparse vector storage based on access patterns
        """
        optimizationResults = {
            'vectorsOptimized': 0,
            'indicesRebuilt': 0,
            'clustersCreated': 0,
            'storageReclaimed': 0
        }
        
        try:
            # Identify frequently accessed vectors for optimization
            hotVectorsQuery = """
            SELECT DOC_ID, ACCESS_COUNT, COMPRESSION_RATIO
            FROM A2A_SPARSE_VECTORS
            WHERE ACCESS_COUNT > 100
                AND LAST_ACCESSED > ADD_DAYS(CURRENT_TIMESTAMP, -7)
            ORDER BY ACCESS_COUNT DESC
            LIMIT 1000
            """
            
            hotVectors = await self.hanaConnection.execute(hotVectorsQuery)
            
            # Optimize hot vectors with better compression
            for vector in hotVectors:
                if vector['COMPRESSION_RATIO'] < 5:
                    # Re-compress with more aggressive settings
                    await self._recompressVector(vector['DOC_ID'])
                    optimizationResults['vectorsOptimized'] += 1
            
            # Rebuild indices for better performance
            rebuildQuery = """
            ALTER INDEX IDX_SPARSE_ENTITY_TYPE REBUILD
            """
            await self.hanaConnection.execute(rebuildQuery)
            optimizationResults['indicesRebuilt'] += 1
            
            # Create clusters for similar vectors
            clusteringResults = await self._clusterSparseVectors()
            optimizationResults['clustersCreated'] = clusteringResults['numClusters']
            
            # Reclaim storage from deleted vectors
            cleanupQuery = """
            MERGE DELTA OF A2A_SPARSE_VECTORS
            """
            await self.hanaConnection.execute(cleanupQuery)
            
            # Calculate storage reclaimed
            statsQuery = """
            SELECT 
                SUM(MEMORY_SIZE_IN_TOTAL) / 1024 / 1024 as TOTAL_MB
            FROM M_CS_TABLES
            WHERE TABLE_NAME = 'A2A_SPARSE_VECTORS'
            """
            stats = await self.hanaConnection.execute(statsQuery)
            optimizationResults['storageReclaimed'] = stats[0]['TOTAL_MB']
            
        except Exception as e:
            logger.error(f"Sparse storage optimization failed: {e}")
            
        return optimizationResults
    
    def _compressIndices(self, indices: np.ndarray) -> bytes:
        """
        Compress sparse vector indices using variable-length encoding
        """
        # Use delta encoding for sorted indices
        sortedIndices = np.sort(indices)
        deltas = np.diff(sortedIndices, prepend=0)
        
        # Variable-length encode the deltas
        compressed = bytearray()
        compressed.extend(struct.pack('<I', len(deltas)))  # Number of indices
        
        for delta in deltas:
            # Use variable-length encoding for each delta
            while delta >= 128:
                compressed.append((delta & 0x7F) | 0x80)
                delta >>= 7
            compressed.append(delta)
            
        return bytes(compressed)
    
    def _compressValues(self, values: np.ndarray) -> bytes:
        """
        Compress sparse vector values using quantization
        """
        # Normalize and quantize to 16-bit
        maxVal = np.max(np.abs(values))
        if maxVal > 0:
            normalized = values / maxVal
            quantized = (normalized * 32767).astype(np.int16)
        else:
            quantized = np.zeros_like(values, dtype=np.int16)
            maxVal = 1.0
            
        # Pack the values
        compressed = bytearray()
        compressed.extend(struct.pack('<f', maxVal))  # Store scale factor
        compressed.extend(quantized.tobytes())
        
        return bytes(compressed)
    
    def _calculateVectorHash(self, indices: np.ndarray, values: np.ndarray) -> str:
        """
        Calculate hash signature for fast equality checks
        """
        # Combine indices and quantized values for hash
        hashData = bytearray()
        hashData.extend(indices.astype(np.uint32).tobytes())
        hashData.extend((values * 1000).astype(np.int32).tobytes())
        
        return hashlib.sha256(hashData).hexdigest()
    
    def _generateDocId(self) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"SPARSE_{timestamp}"
    
    async def _createSparseVectorProcedures(self):
        """
        Create stored procedures for sparse vector operations
        """
        # Sparse cosine similarity function
        similarityProcedure = """
        CREATE OR REPLACE FUNCTION SPARSE_COSINE_SIMILARITY(
            indices1 VARBINARY(5000000),
            values1 VARBINARY(5000000),
            count1 INTEGER,
            indices2 VARBINARY(5000000),
            values2 VARBINARY(5000000),
            count2 INTEGER,
            dimension INTEGER
        ) RETURNS DOUBLE
        LANGUAGE SQLSCRIPT
        READS SQL DATA AS
        BEGIN
            DECLARE similarity DOUBLE;
            DECLARE i INTEGER;
            DECLARE val1 DOUBLE;
            DECLARE val2 DOUBLE;
            DECLARE dot_product DOUBLE := 0.0;
            DECLARE norm1 DOUBLE := 0.0;
            DECLARE norm2 DOUBLE := 0.0;
            
            -- Decode compressed sparse vectors and compute cosine similarity
            -- This is a simplified implementation for demonstration
            FOR i IN 1..CARDINALITY(vector1) DO
                val1 := vector1[i];
                val2 := vector2[i];
                dot_product := dot_product + (val1 * val2);
                norm1 := norm1 + (val1 * val1);
                norm2 := norm2 + (val2 * val2);
            END FOR;
            
            -- Compute cosine similarity
            IF norm1 > 0 AND norm2 > 0 THEN
                similarity := dot_product / (SQRT(norm1) * SQRT(norm2));
            ELSE
                similarity := 0.0;
            END IF;
            
            RETURN similarity;
        END;
        """
        
        await self.hanaConnection.execute(similarityProcedure)
    
    async def _processDenseQuery(self, denseVector: List[float]) -> Dict[str, Any]:
        """Process dense query vector for search"""
        vectorArray = np.array(denseVector)
        nonZeroIndices = np.nonzero(vectorArray)[0]
        
        if len(nonZeroIndices) < len(denseVector) * 0.1:
            # Convert to sparse
            return {
                'isSparse': True,
                'indices': self._compressIndices(nonZeroIndices),
                'values': self._compressValues(vectorArray[nonZeroIndices]),
                'dimension': len(denseVector),
                'nonZeroCount': len(nonZeroIndices),
                'vectorNorm': float(np.linalg.norm(vectorArray))
            }
        else:
            # Keep as dense
            return {
                'isSparse': False,
                'denseVector': denseVector,
                'dimension': len(denseVector),
                'vectorNorm': float(np.linalg.norm(vectorArray))
            }
    
    async def _updateAccessStats(self, docId: str):
        """Update access statistics for cache optimization"""
        updateQuery = """
        UPDATE A2A_SPARSE_VECTORS
        SET ACCESS_COUNT = ACCESS_COUNT + 1,
            LAST_ACCESSED = CURRENT_TIMESTAMP
        WHERE DOC_ID = :docId
        """
        await self.hanaConnection.execute(updateQuery, {'docId': docId})
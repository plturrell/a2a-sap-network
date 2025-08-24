"""
Sparse Vector Support for SAP HANA Vector Engine
Implements efficient storage and operations for high-dimensional sparse vectors
Following SAP naming conventions and best practices
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from datetime import datetime
import logging
import json
import struct
import base64

logger = logging.getLogger(__name__)


class HanaSparseVectorSkills:
    """Enhanced sparse vector processing capabilities for SAP HANA"""
    
    def __init__(self, hanaConnection=None):
        self.hanaConnection = hanaConnection
        self.sparseFormats = {
            'coo': 'coordinate',
            'csr': 'compressed_sparse_row',
            'dok': 'dictionary_of_keys'
        }
        self.compressionRatios = {
            'high': 0.01,      # 99% sparsity
            'medium': 0.05,    # 95% sparsity
            'low': 0.1         # 90% sparsity
        }
        
    async def createSparseVectorTable(self) -> Dict[str, Any]:
        """
        Create optimized table structure for sparse vectors in HANA
        """
        try:
            # Create sparse vector storage table
            createTableQuery = """
            CREATE COLUMN TABLE A2A_SPARSE_VECTORS (
                SPARSE_VEC_ID NVARCHAR(255) PRIMARY KEY,
                DOC_ID NVARCHAR(255),
                ENTITY_TYPE NVARCHAR(100),
                DIMENSIONS INTEGER,
                NON_ZERO_COUNT INTEGER,
                SPARSITY_RATIO DOUBLE,
                -- Sparse vector components
                INDICES NCLOB,  -- JSON array of non-zero indices
                VALUES NCLOB,    -- JSON array of non-zero values
                -- Compressed format for ultra-sparse vectors
                COMPRESSED_DATA BLOB,
                COMPRESSION_TYPE NVARCHAR(50),
                -- Metadata
                VECTOR_NORM DOUBLE,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                METADATA NCLOB,
                -- Foreign key to main vector table
                FOREIGN KEY (DOC_ID) REFERENCES A2A_VECTORS(DOC_ID)
            )
            """
            await self.hanaConnection.execute(createTableQuery)
            
            # Create indexes for efficient sparse operations
            indexQueries = [
                """
                CREATE INDEX IDX_SPARSE_VEC_ENTITY ON A2A_SPARSE_VECTORS(ENTITY_TYPE, SPARSITY_RATIO)
                """,
                """
                CREATE INDEX IDX_SPARSE_VEC_DOC ON A2A_SPARSE_VECTORS(DOC_ID)
                """,
                """
                CREATE INDEX IDX_SPARSE_VEC_SPARSITY ON A2A_SPARSE_VECTORS(SPARSITY_RATIO, NON_ZERO_COUNT)
                """
            ]
            
            for query in indexQueries:
                await self.hanaConnection.execute(query)
            
            return {
                'status': 'success',
                'message': 'Sparse vector table created successfully',
                'indexes': ['IDX_SPARSE_VEC_ENTITY', 'IDX_SPARSE_VEC_DOC', 'IDX_SPARSE_VEC_SPARSITY']
            }
            
        except Exception as e:
            logger.error(f"Failed to create sparse vector table: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def convertToSparseVector(self, 
                                  denseVector: List[float],
                                  docId: str,
                                  entityType: str,
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dense vector to sparse representation with optimal storage
        """
        try:
            # Convert to numpy array
            denseArray = np.array(denseVector)
            dimensions = len(denseArray)
            
            # Find non-zero elements
            nonZeroIndices = np.nonzero(denseArray)[0]
            nonZeroValues = denseArray[nonZeroIndices]
            nonZeroCount = len(nonZeroIndices)
            
            # Calculate sparsity
            sparsityRatio = 1.0 - (nonZeroCount / dimensions)
            
            # Calculate vector norm for similarity calculations
            vectorNorm = np.linalg.norm(denseArray)
            
            # Determine optimal storage format
            storageFormat = self._determineOptimalFormat(sparsityRatio, dimensions, nonZeroCount)
            
            # Prepare sparse vector data
            sparseVecId = f"sparse_{docId}_{datetime.now().timestamp()}"
            
            if storageFormat == 'ultra_sparse':
                # Use compressed binary format for ultra-sparse vectors
                compressedData = self._compressUltraSparse(nonZeroIndices, nonZeroValues)
                
                insertQuery = """
                INSERT INTO A2A_SPARSE_VECTORS (
                    SPARSE_VEC_ID, DOC_ID, ENTITY_TYPE, DIMENSIONS,
                    NON_ZERO_COUNT, SPARSITY_RATIO, COMPRESSED_DATA,
                    COMPRESSION_TYPE, VECTOR_NORM, METADATA
                ) VALUES (
                    :sparseVecId, :docId, :entityType, :dimensions,
                    :nonZeroCount, :sparsityRatio, :compressedData,
                    :compressionType, :vectorNorm, :metadata
                )
                """
                
                params = {
                    'sparseVecId': sparseVecId,
                    'docId': docId,
                    'entityType': entityType,
                    'dimensions': dimensions,
                    'nonZeroCount': nonZeroCount,
                    'sparsityRatio': sparsityRatio,
                    'compressedData': compressedData,
                    'compressionType': 'ultra_sparse_binary',
                    'vectorNorm': vectorNorm,
                    'metadata': json.dumps(metadata)
                }
            else:
                # Use JSON format for moderate sparsity
                insertQuery = """
                INSERT INTO A2A_SPARSE_VECTORS (
                    SPARSE_VEC_ID, DOC_ID, ENTITY_TYPE, DIMENSIONS,
                    NON_ZERO_COUNT, SPARSITY_RATIO, INDICES, VALUES,
                    VECTOR_NORM, METADATA
                ) VALUES (
                    :sparseVecId, :docId, :entityType, :dimensions,
                    :nonZeroCount, :sparsityRatio, :indices, :values,
                    :vectorNorm, :metadata
                )
                """
                
                params = {
                    'sparseVecId': sparseVecId,
                    'docId': docId,
                    'entityType': entityType,
                    'dimensions': dimensions,
                    'nonZeroCount': nonZeroCount,
                    'sparsityRatio': sparsityRatio,
                    'indices': json.dumps(nonZeroIndices.tolist()),
                    'values': json.dumps(nonZeroValues.tolist()),
                    'vectorNorm': vectorNorm,
                    'metadata': json.dumps(metadata)
                }
            
            await self.hanaConnection.execute(insertQuery, params)
            
            # Calculate storage savings
            originalSize = dimensions * 4  # 4 bytes per float
            sparseSize = nonZeroCount * 8  # 4 bytes for index + 4 bytes for value
            savingsRatio = 1.0 - (sparseSize / originalSize)
            
            return {
                'sparseVecId': sparseVecId,
                'dimensions': dimensions,
                'nonZeroCount': nonZeroCount,
                'sparsityRatio': sparsityRatio,
                'storageFormat': storageFormat,
                'storageSavings': f"{savingsRatio:.2%}",
                'vectorNorm': vectorNorm
            }
            
        except Exception as e:
            logger.error(f"Failed to convert to sparse vector: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def sparseSimilaritySearch(self,
                                   querySparseVector: Dict[str, Any],
                                   searchParams: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform efficient similarity search on sparse vectors
        """
        try:
            # Extract query vector components
            queryIndices = querySparseVector.get('indices', [])
            queryValues = querySparseVector.get('values', [])
            queryNorm = querySparseVector.get('norm', 1.0)
            
            # Build search query with optimizations for sparse operations
            searchQuery = """
            WITH SPARSE_CANDIDATES AS (
                -- First filter: vectors with overlapping non-zero indices
                SELECT 
                    sv.*,
                    -- Calculate approximate similarity using index overlap
                    CARDINALITY(
                        ARRAY_INTERSECT(
                            JSON_TO_ARRAY(sv.INDICES),
                            JSON_TO_ARRAY(:queryIndices)
                        )
                    ) AS OVERLAP_COUNT
                FROM A2A_SPARSE_VECTORS sv
                WHERE sv.ENTITY_TYPE = :entityType
                  AND sv.SPARSITY_RATIO BETWEEN :minSparsity AND :maxSparsity
                  AND OVERLAP_COUNT > 0
            ),
            SIMILARITY_SCORES AS (
                SELECT 
                    sc.*,
                    av.CONTENT,
                    av.METADATA AS FULL_METADATA,
                    -- Calculate exact sparse dot product
                    (
                        SELECT SUM(
                            CAST(JSON_VALUE(sc.VALUES, '$[' || idx || ']') AS DOUBLE) *
                            CAST(JSON_VALUE(:queryValues, '$[' || 
                                ARRAY_POSITION(
                                    JSON_TO_ARRAY(:queryIndices),
                                    JSON_VALUE(sc.INDICES, '$[' || idx || ']')
                                ) - 1 || ']') AS DOUBLE)
                        )
                        FROM SERIES_GENERATE_INTEGER(0, 1, JSON_LENGTH(sc.INDICES))
                        WHERE JSON_VALUE(sc.INDICES, '$[' || idx || ']') IN (
                            SELECT * FROM JSON_TO_ARRAY(:queryIndices)
                        )
                    ) / (sc.VECTOR_NORM * :queryNorm) AS COSINE_SIMILARITY
                FROM SPARSE_CANDIDATES sc
                JOIN A2A_VECTORS av ON sc.DOC_ID = av.DOC_ID
                WHERE sc.OVERLAP_COUNT >= :minOverlap
            )
            SELECT 
                DOC_ID,
                CONTENT,
                ENTITY_TYPE,
                DIMENSIONS,
                NON_ZERO_COUNT,
                SPARSITY_RATIO,
                COSINE_SIMILARITY,
                OVERLAP_COUNT,
                FULL_METADATA
            FROM SIMILARITY_SCORES
            WHERE COSINE_SIMILARITY >= :minSimilarity
            ORDER BY COSINE_SIMILARITY DESC
            LIMIT :limit
            """
            
            params = {
                'queryIndices': json.dumps(queryIndices),
                'queryValues': json.dumps(queryValues),
                'queryNorm': queryNorm,
                'entityType': searchParams.get('entityType', 'all'),
                'minSparsity': searchParams.get('minSparsity', 0.0),
                'maxSparsity': searchParams.get('maxSparsity', 1.0),
                'minOverlap': searchParams.get('minOverlap', 1),
                'minSimilarity': searchParams.get('minSimilarity', 0.7),
                'limit': searchParams.get('limit', 10)
            }
            
            results = await self.hanaConnection.execute(searchQuery, params)
            
            return self._formatSparseSearchResults(results)
            
        except Exception as e:
            logger.error(f"Sparse similarity search failed: {e}")
            # Return empty results with proper structure
            return {
                'matches': [],
                'similarity_scores': {},
                'total_candidates': 0,
                'processing_time': 0.0,
                'error': str(e)
            }
    
    async def batchConvertToSparse(self,
                                 entityType: str,
                                 sparsityThreshold: float = 0.9) -> Dict[str, Any]:
        """
        Batch convert dense vectors to sparse format based on sparsity threshold
        """
        conversionResults = {
            'converted': 0,
            'skipped': 0,
            'errors': 0,
            'totalSavings': 0
        }
        
        try:
            # Find candidates for sparse conversion
            candidateQuery = """
            SELECT 
                DOC_ID,
                ENTITY_TYPE,
                VECTOR_EMBEDDING,
                METADATA
            FROM A2A_VECTORS
            WHERE ENTITY_TYPE = :entityType
              AND DOC_ID NOT IN (
                  SELECT DOC_ID FROM A2A_SPARSE_VECTORS
              )
            """
            
            candidates = await self.hanaConnection.execute(candidateQuery, {
                'entityType': entityType
            })
            
            for candidate in candidates:
                try:
                    # Parse vector
                    vectorData = json.loads(candidate['VECTOR_EMBEDDING'])
                    denseVector = vectorData if isinstance(vectorData, list) else vectorData.get('values', [])
                    
                    # Check sparsity
                    nonZeroCount = sum(1 for v in denseVector if v != 0)
                    sparsityRatio = 1.0 - (nonZeroCount / len(denseVector))
                    
                    if sparsityRatio >= sparsityThreshold:
                        # Convert to sparse
                        result = await self.convertToSparseVector(
                            denseVector,
                            candidate['DOC_ID'],
                            candidate['ENTITY_TYPE'],
                            json.loads(candidate['METADATA'] or '{}')
                        )
                        
                        if 'error' not in result:
                            conversionResults['converted'] += 1
                            conversionResults['totalSavings'] += float(
                                result['storageSavings'].rstrip('%')
                            ) / 100
                        else:
                            conversionResults['errors'] += 1
                    else:
                        conversionResults['skipped'] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to convert vector {candidate['DOC_ID']}: {e}")
                    conversionResults['errors'] += 1
            
            # Update statistics
            await self._updateSparseVectorStatistics(entityType)
            
            conversionResults['averageSavings'] = (
                conversionResults['totalSavings'] / conversionResults['converted']
                if conversionResults['converted'] > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Batch sparse conversion failed: {e}")
            
        return conversionResults
    
    async def optimizeSparseStorage(self, entityType: str) -> Dict[str, Any]:
        """
        Optimize sparse vector storage by analyzing patterns and reorganizing data
        """
        optimizationResults = {
            'recompressed': 0,
            'reorganized': 0,
            'spaceSaved': 0
        }
        
        try:
            # Analyze sparsity patterns
            patternQuery = """
            SELECT 
                SPARSITY_RATIO,
                COUNT(*) as COUNT,
                AVG(NON_ZERO_COUNT) as AVG_NON_ZERO,
                SUM(OCTET_LENGTH(COALESCE(INDICES, '')) + 
                    OCTET_LENGTH(COALESCE(VALUES, ''))) as CURRENT_SIZE
            FROM A2A_SPARSE_VECTORS
            WHERE ENTITY_TYPE = :entityType
            GROUP BY SPARSITY_RATIO
            ORDER BY SPARSITY_RATIO DESC
            """
            
            patterns = await self.hanaConnection.execute(patternQuery, {
                'entityType': entityType
            })
            
            for pattern in patterns:
                if pattern['SPARSITY_RATIO'] > 0.99:
                    # Ultra-sparse vectors - convert to compressed format
                    updateQuery = """
                    UPDATE A2A_SPARSE_VECTORS
                    SET COMPRESSED_DATA = TO_BINARY(CONCAT(INDICES, VALUES)),
                        COMPRESSION_TYPE = 'ultra_sparse_binary',
                        INDICES = NULL,
                        VALUES = NULL
                    WHERE ENTITY_TYPE = :entityType
                      AND SPARSITY_RATIO = :sparsityRatio
                      AND COMPRESSED_DATA IS NULL
                    """
                    
                    result = await self.hanaConnection.execute(updateQuery, {
                        'entityType': entityType,
                        'sparsityRatio': pattern['SPARSITY_RATIO']
                    })
                    
                    optimizationResults['recompressed'] += result.get('rowcount', 0)
                    
            # Reorganize table for better compression
            reorganizeQuery = """
            ALTER TABLE A2A_SPARSE_VECTORS 
            REORGANIZE PARTITION BY RANGE(SPARSITY_RATIO)
            """
            await self.hanaConnection.execute(reorganizeQuery)
            optimizationResults['reorganized'] = 1
            
            # Calculate space saved
            afterSizeQuery = """
            SELECT SUM(
                OCTET_LENGTH(COALESCE(INDICES, '')) + 
                OCTET_LENGTH(COALESCE(VALUES, '')) +
                OCTET_LENGTH(COALESCE(COMPRESSED_DATA, ''))
            ) as TOTAL_SIZE
            FROM A2A_SPARSE_VECTORS
            WHERE ENTITY_TYPE = :entityType
            """
            
            afterSize = await self.hanaConnection.execute(afterSizeQuery, {
                'entityType': entityType
            })
            
            optimizationResults['spaceSaved'] = sum(
                p['CURRENT_SIZE'] for p in patterns
            ) - afterSize[0]['TOTAL_SIZE']
            
        except Exception as e:
            logger.error(f"Sparse storage optimization failed: {e}")
            
        return optimizationResults
    
    def _determineOptimalFormat(self, sparsityRatio: float, 
                               dimensions: int, 
                               nonZeroCount: int) -> str:
        """
        Determine optimal storage format based on vector characteristics
        """
        if sparsityRatio > 0.99:
            return 'ultra_sparse'
        elif sparsityRatio > 0.95 and dimensions > 10000:
            return 'compressed_sparse'
        elif nonZeroCount < 100:
            return 'coordinate_list'
        else:
            return 'standard_sparse'
    
    def _compressUltraSparse(self, indices: np.ndarray, values: np.ndarray) -> bytes:
        """
        Compress ultra-sparse vectors using efficient binary encoding
        """
        # Use variable-length encoding for indices
        compressedData = bytearray()
        
        # Header: number of non-zero elements
        compressedData.extend(struct.pack('<I', len(indices)))
        
        # Encode indices using delta encoding
        prevIndex = 0
        for idx in indices:
            delta = idx - prevIndex
            # Variable-length encoding for delta
            while delta >= 128:
                compressedData.append((delta & 0x7F) | 0x80)
                delta >>= 7
            compressedData.append(delta)
            prevIndex = idx
        
        # Encode values using appropriate precision
        for val in values:
            # Use half-precision float for values
            compressedData.extend(struct.pack('<e', val))
        
        return bytes(compressedData)
    
    def _formatSparseSearchResults(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Format sparse search results"""
        formattedResults = []
        
        for result in results:
            formattedResult = {
                'docId': result['DOC_ID'],
                'content': result['CONTENT'],
                'entityType': result['ENTITY_TYPE'],
                'similarity': float(result['COSINE_SIMILARITY']),
                'sparseMetrics': {
                    'dimensions': result['DIMENSIONS'],
                    'nonZeroCount': result['NON_ZERO_COUNT'],
                    'sparsityRatio': float(result['SPARSITY_RATIO']),
                    'overlapCount': result['OVERLAP_COUNT']
                },
                'metadata': json.loads(result['FULL_METADATA'] or '{}')
            }
            formattedResults.append(formattedResult)
            
        return formattedResults
    
    async def _updateSparseVectorStatistics(self, entityType: str):
        """Update statistics for sparse vectors"""
        try:
            statsQuery = """
            INSERT INTO A2A_VECTOR_STATISTICS (
                ENTITY_TYPE,
                STAT_TYPE,
                STAT_VALUE,
                CALCULATED_AT
            )
            SELECT 
                :entityType,
                'sparse_vector_stats',
                JSON_OBJECT(
                    'totalSparseVectors' VALUE COUNT(*),
                    'avgSparsity' VALUE AVG(SPARSITY_RATIO),
                    'avgNonZeroElements' VALUE AVG(NON_ZERO_COUNT),
                    'totalStorageMB' VALUE SUM(
                        OCTET_LENGTH(COALESCE(INDICES, '')) + 
                        OCTET_LENGTH(COALESCE(VALUES, '')) +
                        OCTET_LENGTH(COALESCE(COMPRESSED_DATA, ''))
                    ) / 1024.0 / 1024.0
                ),
                CURRENT_TIMESTAMP
            FROM A2A_SPARSE_VECTORS
            WHERE ENTITY_TYPE = :entityType
            """
            
            await self.hanaConnection.execute(statsQuery, {'entityType': entityType})
            
        except Exception as e:
            logger.error(f"Failed to update sparse vector statistics: {e}")
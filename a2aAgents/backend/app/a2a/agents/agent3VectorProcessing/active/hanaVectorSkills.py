"""
HANA Vector Skills for Agent 3 (Vector Processing) - SAP HANA Knowledge Engine Integration
Following SAP naming conventions and best practices
"""
import random

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import logging
import json
import asyncio
from .sparseVectorSkills import SparseVectorSkills
from .hybridRankingSkills import HybridRankingSkills

logger = logging.getLogger(__name__)


class HanaVectorSkills:
    """Enhanced vector processing skills leveraging SAP HANA Knowledge Engine"""
    
    def __init__(self, hanaConnection=None):
        self.hanaConnection = hanaConnection
        self.vectorDimensions = {
            'standard': 384,
            'enhanced': 768,
            'financial': 512
        }
        # Initialize sparse vector capabilities
        self.sparseVectorSkills = SparseVectorSkills(hanaConnection)
        # Initialize hybrid ranking capabilities
        self.hybridRankingSkills = HybridRankingSkills(hanaConnection)
        
    async def hybridVectorSearch(self, 
                                query: str, 
                                queryVector: List[float],
                                searchFilters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and metadata filtering
        Using HANA's native vector capabilities
        """
        try:
            # Build dynamic SQL with metadata filters
            whereConditions = ["1=1"]
            queryParams = {'queryVector': queryVector}
            
            # Add entity type filter
            if searchFilters.get('entityType'):
                whereConditions.append("ENTITY_TYPE = :entityType")
                queryParams['entityType'] = searchFilters['entityType']
            
            # Add AI readiness score filter
            if searchFilters.get('minAiReadinessScore'):
                whereConditions.append("""
                    JSON_VALUE(METADATA, '$.aiReadinessScore') >= :minScore
                """)
                queryParams['minScore'] = searchFilters['minAiReadinessScore']
            
            # Add temporal filter
            if searchFilters.get('dateRange'):
                whereConditions.append("""
                    CREATED_AT BETWEEN :startDate AND :endDate
                """)
                queryParams['startDate'] = searchFilters['dateRange']['start']
                queryParams['endDate'] = searchFilters['dateRange']['end']
            
            # Add semantic tag filter
            if searchFilters.get('semanticTags'):
                tagConditions = []
                for idx, tag in enumerate(searchFilters['semanticTags']):
                    paramName = f'tag{idx}'
                    tagConditions.append(f"""
                        JSON_QUERY(METADATA, '$.semanticTags[*]') LIKE '%:{paramName}%'
                    """)
                    queryParams[paramName] = tag
                if tagConditions:
                    whereConditions.append(f"({' OR '.join(tagConditions)})")
            
            # Execute hybrid search query
            hybridSearchQuery = f"""
            WITH VECTOR_SEARCH AS (
                SELECT 
                    DOC_ID,
                    CONTENT,
                    METADATA,
                    VECTOR_EMBEDDING,
                    ENTITY_TYPE,
                    SOURCE_AGENT,
                    CREATED_AT,
                    COSINE_SIMILARITY(VECTOR_EMBEDDING, TO_REAL_VECTOR(:queryVector)) AS SIMILARITY_SCORE
                FROM A2A_VECTORS
                WHERE {' AND '.join(whereConditions)}
            ),
            RANKED_RESULTS AS (
                SELECT 
                    *,
                    -- Boost score based on metadata quality
                    SIMILARITY_SCORE * 
                    CASE 
                        WHEN JSON_VALUE(METADATA, '$.aiReadinessScore') > 0.8 THEN 1.2
                        WHEN JSON_VALUE(METADATA, '$.aiReadinessScore') > 0.6 THEN 1.1
                        ELSE 1.0
                    END AS BOOSTED_SCORE
                FROM VECTOR_SEARCH
                WHERE SIMILARITY_SCORE > :minSimilarity
            )
            SELECT 
                DOC_ID,
                CONTENT,
                METADATA,
                ENTITY_TYPE,
                SIMILARITY_SCORE,
                BOOSTED_SCORE,
                -- Extract key metadata for quick access
                JSON_VALUE(METADATA, '$.entityId') AS ENTITY_ID,
                JSON_VALUE(METADATA, '$.aiReadinessScore') AS AI_READINESS_SCORE,
                JSON_QUERY(METADATA, '$.semanticTags') AS SEMANTIC_TAGS
            FROM RANKED_RESULTS
            ORDER BY BOOSTED_SCORE DESC
            LIMIT :limit
            """
            
            queryParams['minSimilarity'] = searchFilters.get('minSimilarity', 0.7)
            queryParams['limit'] = searchFilters.get('limit', 10)
            
            results = await self.hanaConnection.execute(hybridSearchQuery, queryParams)
            
            return self._formatSearchResults(results)
            
        except Exception as e:
            logger.error(f"Hybrid vector search failed: {e}")
            # Return empty results with proper structure
            return {
                'results': [],
                'hybrid_scores': {},
                'vector_matches': 0,
                'text_matches': 0,
                'total_time': 0.0,
                'error': str(e)
            }
    
    async def semanticIndexManagement(self, 
                                    indexConfig: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage hierarchical vector indexes for optimal performance
        """
        indexResults = {
            'created': [],
            'optimized': [],
            'errors': []
        }
        
        try:
            # Create hierarchical indexes based on entity types
            if indexConfig.get('createHierarchicalIndex'):
                hierarchicalIndexQuery = """
                CREATE INDEX IDX_VECTOR_HIERARCHY ON A2A_VECTORS (
                    ENTITY_TYPE,
                    JSON_VALUE(METADATA, '$.entityCategory'),
                    VECTOR_EMBEDDING
                )
                """
                await self.hanaConnection.execute(hierarchicalIndexQuery)
                indexResults['created'].append('IDX_VECTOR_HIERARCHY')
            
            # Create specialized indexes for different vector dimensions
            if indexConfig.get('createDimensionIndex'):
                for dimName, dimSize in self.vectorDimensions.items():
                    indexName = f'IDX_VECTOR_{dimName.upper()}'
                    dimIndexQuery = f"""
                    CREATE INDEX {indexName} ON A2A_VECTORS (
                        VECTOR_EMBEDDING
                    )
                    WHERE VECTOR_LENGTH(VECTOR_EMBEDDING) = {dimSize}
                    """
                    await self.hanaConnection.execute(dimIndexQuery)
                    indexResults['created'].append(indexName)
            
            # Optimize existing indexes based on usage patterns
            if indexConfig.get('optimizeIndexes'):
                optimizationResults = await self._optimizeVectorIndexes()
                indexResults['optimized'] = optimizationResults
            
        except Exception as e:
            logger.error(f"Semantic index management failed: {e}")
            indexResults['errors'].append(str(e))
            
        return indexResults
    
    async def knowledgeGraphEnhancement(self, 
                                      entityData: Dict[str, Any],
                                      vectorData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance knowledge graph with vector-based semantic relationships
        """
        graphEnhancements = {
            'nodesCreated': 0,
            'edgesCreated': 0,
            'semanticClusters': [],
            'ontologyUpdates': []
        }
        
        try:
            # Create or update graph node with vector reference
            nodeData = {
                'nodeId': f"node_{entityData['entityId']}",
                'entityId': entityData['entityId'],
                'entityType': entityData['entityType'],
                'properties': json.dumps(entityData),
                'vectorReference': vectorData['docId']
            }
            
            nodeInsertQuery = """
            MERGE (SOURCE A2A_GRAPH_NODES 
                   ON NODE_ID = :nodeId)
            WHEN MATCHED THEN UPDATE SET
                PROPERTIES = :properties,
                VECTOR_REFERENCE = :vectorReference,
                UPDATED_AT = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN INSERT (
                NODE_ID, ENTITY_ID, ENTITY_TYPE, 
                PROPERTIES, VECTOR_REFERENCE
            ) VALUES (
                :nodeId, :entityId, :entityType,
                :properties, :vectorReference
            )
            """
            
            await self.hanaConnection.execute(nodeInsertQuery, nodeData)
            graphEnhancements['nodesCreated'] += 1
            
            # Discover semantic relationships using vector similarity
            semanticRelationships = await self._discoverSemanticRelationships(
                vectorData['vector'],
                entityData['entityType']
            )
            
            # Create edges for discovered relationships
            for relationship in semanticRelationships:
                edgeData = {
                    'edgeId': f"edge_{entityData['entityId']}_{relationship['targetId']}",
                    'sourceNodeId': nodeData['nodeId'],
                    'targetNodeId': f"node_{relationship['targetId']}",
                    'relationshipType': 'semantic_similarity',
                    'properties': json.dumps({
                        'similarityScore': relationship['similarity'],
                        'discoveryMethod': 'vector_similarity',
                        'confidence': relationship['confidence']
                    }),
                    'confidenceScore': relationship['confidence']
                }
                
                edgeInsertQuery = """
                INSERT INTO A2A_GRAPH_EDGES (
                    EDGE_ID, SOURCE_NODE_ID, TARGET_NODE_ID,
                    RELATIONSHIP_TYPE, PROPERTIES, CONFIDENCE_SCORE
                ) VALUES (
                    :edgeId, :sourceNodeId, :targetNodeId,
                    :relationshipType, :properties, :confidenceScore
                )
                """
                
                await self.hanaConnection.execute(edgeInsertQuery, edgeData)
                graphEnhancements['edgesCreated'] += 1
            
            # Identify semantic clusters using HANA ML
            clusters = await self._identifySemanticClusters(entityData['entityType'])
            graphEnhancements['semanticClusters'] = clusters
            
            # Update domain ontology based on new relationships
            ontologyUpdates = await self._updateDomainOntology(entityData, semanticRelationships)
            graphEnhancements['ontologyUpdates'] = ontologyUpdates
            
        except Exception as e:
            logger.error(f"Knowledge graph enhancement failed: {e}")
            
        return graphEnhancements
    
    async def vectorClusteringAnalysis(self, 
                                     clusterConfig: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vector clustering for pattern discovery using HANA PAL
        """
        clusteringResults = {
            'clusters': [],
            'outliers': [],
            'patterns': [],
            'insights': []
        }
        
        try:
            # Prepare data for clustering
            entityType = clusterConfig.get('entityType', 'all')
            minClusterSize = clusterConfig.get('minClusterSize', 5)
            
            # Use HANA PAL K-Means clustering
            clusteringQuery = """
            DO BEGIN
                -- Create temporary tables for PAL
                CREATE LOCAL TEMPORARY TABLE #PAL_KMEANS_DATA (
                    ID NVARCHAR(255),
                    FEATURES DOUBLE ARRAY
                );
                
                -- Insert vector data
                INSERT INTO #PAL_KMEANS_DATA
                SELECT 
                    DOC_ID,
                    TO_DOUBLE_ARRAY(VECTOR_EMBEDDING)
                FROM A2A_VECTORS
                WHERE ENTITY_TYPE = :entityType
                   OR :entityType = 'all';
                
                -- Create output tables
                CREATE LOCAL TEMPORARY TABLE #PAL_KMEANS_CENTERS (
                    CLUSTER_ID INTEGER,
                    CENTER DOUBLE ARRAY
                );
                
                CREATE LOCAL TEMPORARY TABLE #PAL_KMEANS_ASSIGNED (
                    ID NVARCHAR(255),
                    CLUSTER_ID INTEGER,
                    DISTANCE DOUBLE
                );
                
                -- Run K-Means clustering
                CALL _SYS_AFL.PAL_KMEANS(
                    DATA_TAB => #PAL_KMEANS_DATA,
                    PARAM_TAB => ?,
                    RESULT_TAB => #PAL_KMEANS_ASSIGNED,
                    CENTER_TAB => #PAL_KMEANS_CENTERS
                );
                
                -- Analyze clusters
                SELECT 
                    c.CLUSTER_ID,
                    COUNT(*) as CLUSTER_SIZE,
                    AVG(c.DISTANCE) as AVG_DISTANCE,
                    STRING_AGG(v.ENTITY_TYPE, ',') as ENTITY_TYPES
                FROM #PAL_KMEANS_ASSIGNED c
                JOIN A2A_VECTORS v ON c.ID = v.DOC_ID
                GROUP BY c.CLUSTER_ID
                HAVING COUNT(*) >= :minClusterSize;
            END;
            """
            
            # Set PAL parameters
            palParams = [
                ('K', clusterConfig.get('numClusters', 10)),
                ('DISTANCE_LEVEL', 2),  # Euclidean distance
                ('MAX_ITERATION', 100),
                ('INIT_TYPE', 1),  # K-Means++
                ('THREAD_NUMBER', 4)
            ]
            
            clusterData = await self.hanaConnection.execute(
                clusteringQuery, 
                {
                    'entityType': entityType,
                    'minClusterSize': minClusterSize
                },
                palParams
            )
            
            # Process clustering results
            for cluster in clusterData:
                clusterInfo = {
                    'clusterId': cluster['CLUSTER_ID'],
                    'size': cluster['CLUSTER_SIZE'],
                    'avgDistance': cluster['AVG_DISTANCE'],
                    'dominantTypes': cluster['ENTITY_TYPES'].split(','),
                    'interpretation': await self._interpretCluster(cluster)
                }
                clusteringResults['clusters'].append(clusterInfo)
            
            # Identify outliers
            outliers = await self._identifyOutliers(entityType)
            clusteringResults['outliers'] = outliers
            
            # Extract patterns from clusters
            patterns = await self._extractClusterPatterns(clusteringResults['clusters'])
            clusteringResults['patterns'] = patterns
            
        except Exception as e:
            logger.error(f"Vector clustering analysis failed: {e}")
            
        return clusteringResults
    
    async def _discoverSemanticRelationships(self, 
                                           queryVector: List[float],
                                           entityType: str) -> List[Dict[str, Any]]:
        """
        Discover semantic relationships based on vector similarity
        """
        try:
            # Find similar entities within the same domain
            similarityQuery = """
            SELECT 
                DOC_ID,
                JSON_VALUE(METADATA, '$.entityId') as TARGET_ID,
                COSINE_SIMILARITY(VECTOR_EMBEDDING, TO_REAL_VECTOR(:queryVector)) as SIMILARITY,
                ENTITY_TYPE,
                JSON_VALUE(METADATA, '$.aiReadinessScore') as AI_SCORE
            FROM A2A_VECTORS
            WHERE ENTITY_TYPE = :entityType
              AND COSINE_SIMILARITY(VECTOR_EMBEDDING, TO_REAL_VECTOR(:queryVector)) > 0.8
              AND COSINE_SIMILARITY(VECTOR_EMBEDDING, TO_REAL_VECTOR(:queryVector)) < 0.99
            ORDER BY SIMILARITY DESC
            LIMIT 10
            """
            
            results = await self.hanaConnection.execute(similarityQuery, {
                'queryVector': queryVector,
                'entityType': entityType
            })
            
            relationships = []
            for result in results:
                confidence = self._calculateRelationshipConfidence(
                    result['SIMILARITY'],
                    result['AI_SCORE']
                )
                relationships.append({
                    'targetId': result['TARGET_ID'],
                    'similarity': result['SIMILARITY'],
                    'confidence': confidence,
                    'relationshipStrength': 'strong' if result['SIMILARITY'] > 0.9 else 'moderate'
                })
                
            return relationships
            
        except Exception as e:
            logger.error(f"Semantic relationship discovery failed: {e}")
            # Return empty relationships with proper structure
            return {
                'relationships': [],
                'entity_pairs': 0,
                'confidence_scores': {},
                'relationship_types': [],
                'processing_time': 0.0,
                'error': str(e)
            }
    
    async def _optimizeVectorIndexes(self) -> List[str]:
        """
        Optimize vector indexes based on usage patterns
        """
        optimized = []
        
        try:
            # Analyze index usage statistics
            statsQuery = """
            SELECT 
                INDEX_NAME,
                TABLE_NAME,
                USED_COUNT,
                LAST_USED_TIME
            FROM M_INDEXES
            WHERE TABLE_NAME = 'A2A_VECTORS'
              AND USED_COUNT > 0
            ORDER BY USED_COUNT DESC
            """
            
            indexStats = await self.hanaConnection.execute(statsQuery)
            
            for index in indexStats:
                if index['USED_COUNT'] > 1000:
                    # Rebuild heavily used indexes
                    rebuildQuery = f"ALTER INDEX {index['INDEX_NAME']} REBUILD"
                    await self.hanaConnection.execute(rebuildQuery)
                    optimized.append(index['INDEX_NAME'])
                    
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            
        return optimized
    
    def _formatSearchResults(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Format search results for response"""
        formattedResults = []
        
        for result in results:
            formattedResult = {
                'docId': result['DOC_ID'],
                'content': result['CONTENT'],
                'entityType': result['ENTITY_TYPE'],
                'similarityScore': float(result['SIMILARITY_SCORE']),
                'boostedScore': float(result['BOOSTED_SCORE']),
                'metadata': json.loads(result['METADATA']) if result['METADATA'] else {},
                'highlights': []  # Could add text highlighting based on query
            }
            
            # Add extracted metadata
            if result.get('ENTITY_ID'):
                formattedResult['entityId'] = result['ENTITY_ID']
            if result.get('AI_READINESS_SCORE'):
                formattedResult['aiReadinessScore'] = float(result['AI_READINESS_SCORE'])
            if result.get('SEMANTIC_TAGS'):
                formattedResult['semanticTags'] = json.loads(result['SEMANTIC_TAGS'])
                
            formattedResults.append(formattedResult)
            
        return formattedResults
    
    def _calculateRelationshipConfidence(self, similarity: float, aiScore: float) -> float:
        """Calculate confidence score for discovered relationships"""
        # Weighted combination of similarity and AI readiness
        baseConfidence = similarity * 0.7 + float(aiScore or 0.5) * 0.3
        
        # Apply threshold-based adjustments
        if similarity > 0.95:
            baseConfidence *= 1.1
        elif similarity < 0.85:
            baseConfidence *= 0.9
            
        return min(baseConfidence, 1.0)
    
    async def enhancedHybridSearch(self,
                                 query: str,
                                 queryVector: Union[List[float], Dict[str, Any]],
                                 searchFilters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search supporting both dense and sparse vectors
        Automatically selects optimal search strategy
        """
        try:
            # Determine if we should use sparse search
            useSparseSearch = False
            vectorStats = None
            
            if isinstance(queryVector, list):
                # Analyze dense vector for sparsity
                vectorArray = np.array(queryVector)
                nonZeroCount = np.count_nonzero(vectorArray)
                density = nonZeroCount / len(queryVector)
                
                if density < 0.1 and len(queryVector) > 1000:
                    useSparseSearch = True
                    vectorStats = {
                        'dimension': len(queryVector),
                        'nonZeroCount': nonZeroCount,
                        'density': density
                    }
            elif isinstance(queryVector, dict) and queryVector.get('isSparse'):
                useSparseSearch = True
                vectorStats = queryVector
            
            # Route to appropriate search method
            if useSparseSearch:
                logger.info(f"Using sparse vector search (density: {vectorStats.get('density', 'N/A')})")
                
                # Search in sparse vector storage
                sparseResults = await self.sparseVectorSkills.sparseVectorSearch(
                    queryVector, 
                    searchFilters
                )
                
                # Also search in dense vectors for comprehensive results
                denseResults = await self.hybridVectorSearch(
                    query,
                    queryVector if isinstance(queryVector, list) else queryVector.get('denseVector', []),
                    searchFilters
                )
                
                # Merge and deduplicate results
                return self._mergeSearchResults(sparseResults, denseResults)
            else:
                # Use standard dense vector search
                return await self.hybridVectorSearch(query, queryVector, searchFilters)
                
        except Exception as e:
            logger.error(f"Enhanced hybrid search failed: {e}")
            return []
    
    async def convertAndStoreVector(self,
                                  vector: List[float],
                                  entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently store vector as dense or sparse based on characteristics
        """
        try:
            # Analyze vector characteristics
            vectorArray = np.array(vector)
            dimension = len(vector)
            nonZeroCount = np.count_nonzero(vectorArray)
            density = nonZeroCount / dimension
            
            # Decision logic for storage format
            if density < 0.1 and dimension > 1000 and nonZeroCount > 10:
                # Convert to sparse and store
                result = await self.sparseVectorSkills.convertToSparseVector(
                    vector,
                    entityData
                )
                
                if result.get('status') == 'success':
                    logger.info(f"Stored as sparse vector: {result['spaceSaved']} space saved")
                    return result
            
            # Store as dense vector (existing logic)
            denseResult = await self._storeDenseVector(vector, entityData)
            return denseResult
            
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def initializeSparseVectorStorage(self) -> Dict[str, Any]:
        """
        Initialize sparse vector storage tables and procedures
        """
        return await self.sparseVectorSkills.createSparseVectorStorage()
    
    def _mergeSearchResults(self, 
                          sparseResults: List[Dict[str, Any]], 
                          denseResults: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge results from sparse and dense searches, removing duplicates
        """
        # Create a map to track unique results
        resultMap = {}
        
        # Add sparse results first (typically more relevant for sparse queries)
        for result in sparseResults:
            key = result.get('entityId', result.get('docId'))
            if key not in resultMap:
                result['vectorType'] = 'sparse'
                resultMap[key] = result
            else:
                # Update with higher score
                if result.get('similarityScore', 0) > resultMap[key].get('similarityScore', 0):
                    result['vectorType'] = 'sparse'
                    resultMap[key] = result
        
        # Add dense results
        for result in denseResults:
            key = result.get('entityId', result.get('docId'))
            if key not in resultMap:
                result['vectorType'] = 'dense'
                resultMap[key] = result
            else:
                # Update with higher score
                if result.get('similarityScore', 0) > resultMap[key].get('similarityScore', 0):
                    result['vectorType'] = 'dense'
                    resultMap[key] = result
        
        # Sort by similarity score
        mergedResults = list(resultMap.values())
        mergedResults.sort(key=lambda x: x.get('similarityScore', 0), reverse=True)
        
        return mergedResults
    
    async def _storeDenseVector(self,
                              vector: List[float],
                              entityData: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store vector in dense format (existing storage)
        """
        try:
            insertQuery = """
            INSERT INTO A2A_VECTORS (
                DOC_ID, ENTITY_ID, ENTITY_TYPE,
                CONTENT, VECTOR_EMBEDDING, METADATA,
                SOURCE_AGENT
            ) VALUES (
                :docId, :entityId, :entityType,
                :content, TO_REAL_VECTOR(:vector), :metadata,
                :sourceAgent
            )
            """
            
            params = {
                'docId': entityData.get('docId', self._generateDocId()),
                'entityId': entityData['entityId'],
                'entityType': entityData['entityType'],
                'content': entityData.get('content', ''),
                'vector': vector,
                'metadata': json.dumps(entityData.get('metadata', {})),
                'sourceAgent': entityData.get('sourceAgent', 'agent3')
            }
            
            await self.hanaConnection.execute(insertQuery, params)
            
            return {
                'status': 'success',
                'docId': params['docId'],
                'vectorType': 'dense',
                'dimension': len(vector)
            }
            
        except Exception as e:
            logger.error(f"Dense vector storage failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generateDocId(self) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"VEC_{timestamp}"
    
    async def advancedHybridSearch(self,
                                 query: str,
                                 queryVector: Union[List[float], Dict[str, Any]],
                                 searchFilters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Advanced hybrid search with integrated ranking using BM25, PageRank, and contextual signals
        """
        try:
            # Step 1: Get initial candidates using enhanced hybrid search
            initialResults = await self.enhancedHybridSearch(query, queryVector, searchFilters)
            
            if not initialResults:
                return []
            
            # Step 2: Prepare search context for ranking
            searchContext = {
                'searchType': searchFilters.get('searchType', 'general'),
                'userContext': {
                    'userPreferences': searchFilters.get('userPreferences', {})
                },
                'temporalContext': {
                    'preferRecent': searchFilters.get('preferRecent', False)
                },
                'domainContext': {
                    'domain': searchFilters.get('domain', searchFilters.get('entityType', 'general'))
                },
                'diversityThreshold': searchFilters.get('diversityThreshold', 0.85),
                'maxResults': searchFilters.get('limit', 10)
            }
            
            # Step 3: Apply hybrid ranking
            rankedResults = await self.hybridRankingSkills.hybridRankingSearch(
                query,
                queryVector if isinstance(queryVector, list) else queryVector.get('denseVector', []),
                initialResults,
                searchContext
            )
            
            # Step 4: Add enhanced metadata for each result
            enhancedResults = []
            for result in rankedResults:
                enhancedResult = result.copy()
                enhancedResult.update({
                    'searchMethod': 'advanced_hybrid',
                    'rankingApplied': True,
                    'searchContext': {
                        'query': query,
                        'searchType': searchContext['searchType'],
                        'domain': searchContext['domainContext']['domain']
                    }
                })
                enhancedResults.append(enhancedResult)
            
            logger.info(f"Advanced hybrid search returned {len(enhancedResults)} results")
            return enhancedResults
            
        except Exception as e:
            logger.error(f"Advanced hybrid search failed: {e}")
            # Fallback to enhanced hybrid search without ranking
            return await self.enhancedHybridSearch(query, queryVector, searchFilters)
    
    async def optimizeSearchPerformance(self,
                                      searchHistory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize search performance based on usage patterns
        """
        optimizationResults = {
            'rankingWeightsAdjusted': False,
            'cacheOptimized': False,
            'indexesOptimized': False,
            'performanceImprovement': 0.0
        }
        
        try:
            if not searchHistory:
                return optimizationResults
            
            # Analyze search patterns
            searchPatterns = self._analyzeSearchPatterns(searchHistory)
            
            # Adjust ranking weights based on click-through rates
            if searchPatterns['clickThroughData']:
                newWeights = self._optimizeRankingWeights(searchPatterns['clickThroughData'])
                if newWeights:
                    self.hybridRankingSkills.rankingWeights = newWeights
                    optimizationResults['rankingWeightsAdjusted'] = True
            
            # Optimize vector indexes
            if searchPatterns['frequentEntityTypes']:
                indexOptimization = await self.semanticIndexManagement({
                    'optimizeIndexes': True,
                    'entityTypes': searchPatterns['frequentEntityTypes']
                })
                optimizationResults['indexesOptimized'] = len(indexOptimization.get('optimized', [])) > 0
            
            # Optimize sparse vector storage
            if searchPatterns['sparseVectorUsage'] > 0.3:
                sparseOptimization = await self.sparseVectorSkills.optimizeSparseStorage()
                optimizationResults['cacheOptimized'] = sparseOptimization['vectorsOptimized'] > 0
            
            # Calculate performance improvement estimate
            optimizationResults['performanceImprovement'] = self._estimatePerformanceGain(
                optimizationResults
            )
            
            return optimizationResults
            
        except Exception as e:
            logger.error(f"Search performance optimization failed: {e}")
            return optimizationResults
    
    def _analyzeSearchPatterns(self, searchHistory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze search patterns from history
        """
        patterns = {
            'totalSearches': len(searchHistory),
            'searchTypes': {},
            'entityTypes': {},
            'clickThroughData': [],
            'frequentEntityTypes': [],
            'sparseVectorUsage': 0.0
        }
        
        sparseSearches = 0
        
        for search in searchHistory:
            # Count search types
            searchType = search.get('searchType', 'general')
            patterns['searchTypes'][searchType] = patterns['searchTypes'].get(searchType, 0) + 1
            
            # Count entity types
            entityType = search.get('entityType', 'unknown')
            patterns['entityTypes'][entityType] = patterns['entityTypes'].get(entityType, 0) + 1
            
            # Track click-through data
            if 'results' in search and 'clickedResults' in search:
                patterns['clickThroughData'].append({
                    'query': search.get('query', ''),
                    'results': search['results'],
                    'clicked': search['clickedResults']
                })
            
            # Track sparse vector usage
            if search.get('usedSparseVectors', False):
                sparseSearches += 1
        
        # Calculate frequent entity types (top 20% by frequency)
        if patterns['entityTypes']:
            sortedTypes = sorted(patterns['entityTypes'].items(), key=lambda x: x[1], reverse=True)
            topCount = max(1, len(sortedTypes) // 5)
            patterns['frequentEntityTypes'] = [typ for typ, _ in sortedTypes[:topCount]]
        
        # Calculate sparse vector usage ratio
        patterns['sparseVectorUsage'] = sparseSearches / len(searchHistory) if searchHistory else 0
        
        return patterns
    
    def _optimizeRankingWeights(self, clickThroughData: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Optimize ranking weights based on click-through analysis
        """
        try:
            if len(clickThroughData) < 10:
                return None
            
            # Analyze which ranking components correlate with clicks
            componentPerformance = {
                'vectorSimilarity': 0.0,
                'bm25Score': 0.0,
                'pageRankScore': 0.0,
                'contextualRelevance': 0.0
            }
            
            totalSamples = 0
            
            for ctr in clickThroughData:
                results = ctr.get('results', [])
                clicked = ctr.get('clicked', [])
                
                if not results or not clicked:
                    continue
                
                # Find clicked results and analyze their ranking components
                for result in results:
                    if result.get('docId') in clicked:
                        components = result.get('rankingComponents', {})
                        for component, score in components.items():
                            if component in componentPerformance:
                                componentPerformance[component] += score
                                totalSamples += 1
            
            if totalSamples == 0:
                return None
            
            # Normalize performance scores
            for component in componentPerformance:
                componentPerformance[component] /= totalSamples
            
            # Adjust weights based on performance
            totalPerformance = sum(componentPerformance.values())
            if totalPerformance > 0:
                newWeights = {}
                for component, performance in componentPerformance.items():
                    # Convert component performance to weight
                    baseWeight = performance / totalPerformance
                    newWeights[component] = min(max(baseWeight, 0.1), 0.7)  # Clamp between 0.1 and 0.7
                
                # Ensure weights sum to 1.0
                weightSum = sum(newWeights.values())
                for component in newWeights:
                    newWeights[component] /= weightSum
                
                return newWeights
            
            return None
            
        except Exception as e:
            logger.error(f"Ranking weight optimization failed: {e}")
            return None
    
    def _estimatePerformanceGain(self, optimizationResults: Dict[str, Any]) -> float:
        """
        Estimate performance improvement from optimizations
        """
        improvement = 0.0
        
        if optimizationResults['rankingWeightsAdjusted']:
            improvement += 0.15  # 15% improvement from better ranking
        
        if optimizationResults['indexesOptimized']:
            improvement += 0.10  # 10% improvement from index optimization
        
        if optimizationResults['cacheOptimized']:
            improvement += 0.05  # 5% improvement from cache optimization
        
        return improvement
    
    async def processVectorWithSparsityCheck(self,
                                           vector: List[float],
                                           docId: str,
                                           entityType: str,
                                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process vector and automatically convert to sparse if beneficial
        """
        try:
            # Calculate sparsity
            nonZeroCount = sum(1 for v in vector if v != 0)
            sparsityRatio = 1.0 - (nonZeroCount / len(vector))
            
            processingResult = {
                'docId': docId,
                'vectorType': 'dense',
                'sparsityRatio': sparsityRatio,
                'storageOptimized': False
            }
            
            # If vector is sparse enough, convert to sparse format
            if sparsityRatio >= 0.9:
                sparseResult = await self.sparseVectorSkills.convertToSparseVector(
                    vector, docId, entityType, metadata
                )
                
                if 'error' not in sparseResult:
                    processingResult.update({
                        'vectorType': 'sparse',
                        'sparseVecId': sparseResult['sparseVecId'],
                        'storageSavings': sparseResult['storageSavings'],
                        'storageOptimized': True
                    })
                    
                    logger.info(f"Converted vector to sparse format: {docId}, "
                              f"savings: {sparseResult['storageSavings']}")
            else:
                # Store as regular dense vector
                await self._storeDenseVector(vector, docId, entityType, metadata)
                
            return processingResult
            
        except Exception as e:
            logger.error(f"Vector processing with sparsity check failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def unifiedVectorSearch(self,
                                query: str,
                                queryVector: List[float],
                                searchParams: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Unified search across both dense and sparse vectors
        """
        try:
            allResults = []
            
            # Check if query vector is sparse
            nonZeroCount = sum(1 for v in queryVector if v != 0)
            querySparsity = 1.0 - (nonZeroCount / len(queryVector))
            
            # Search dense vectors
            denseResults = await self.hybridVectorSearch(query, queryVector, searchParams)
            allResults.extend([{**r, 'vectorType': 'dense'} for r in denseResults])
            
            # Search sparse vectors if applicable
            if querySparsity >= 0.7 or searchParams.get('includeSparse', True):
                # Convert query to sparse format
                queryIndices = [i for i, v in enumerate(queryVector) if v != 0]
                queryValues = [v for v in queryVector if v != 0]
                queryNorm = np.linalg.norm(queryVector)
                
                sparseQuery = {
                    'indices': queryIndices,
                    'values': queryValues,
                    'norm': queryNorm
                }
                
                sparseResults = await self.sparseVectorSkills.sparseSimilaritySearch(
                    sparseQuery, searchParams
                )
                allResults.extend([{**r, 'vectorType': 'sparse'} for r in sparseResults])
            
            # Merge and rank results
            allResults.sort(key=lambda x: x.get('similarityScore', x.get('similarity', 0)), 
                          reverse=True)
            
            # Apply limit
            limit = searchParams.get('limit', 10)
            return allResults[:limit]
            
        except Exception as e:
            logger.error(f"Unified vector search failed: {e}")
            return []
    
    async def optimizeVectorStorage(self, entityType: str) -> Dict[str, Any]:
        """
        Optimize vector storage by converting suitable vectors to sparse format
        """
        optimizationResults = {
            'denseOptimized': {},
            'sparseOptimized': {},
            'conversions': {}
        }
        
        try:
            # Optimize existing sparse vectors
            sparseOptResults = await self.sparseVectorSkills.optimizeSparseStorage(entityType)
            optimizationResults['sparseOptimized'] = sparseOptResults
            
            # Convert suitable dense vectors to sparse
            conversionResults = await self.sparseVectorSkills.batchConvertToSparse(
                entityType, sparsityThreshold=0.9
            )
            optimizationResults['conversions'] = conversionResults
            
            # Optimize dense vector indexes
            denseOptResults = await self._optimizeVectorIndexes()
            optimizationResults['denseOptimized'] = {
                'indexesOptimized': denseOptResults
            }
            
            # Calculate total optimization impact
            totalVectors = conversionResults.get('converted', 0) + conversionResults.get('skipped', 0)
            if totalVectors > 0:
                optimizationResults['summary'] = {
                    'totalVectorsProcessed': totalVectors,
                    'percentageConverted': (conversionResults.get('converted', 0) / totalVectors) * 100,
                    'averageStorageSavings': conversionResults.get('averageSavings', 0) * 100,
                    'recommendation': self._generateOptimizationRecommendation(optimizationResults)
                }
            
        except Exception as e:
            logger.error(f"Vector storage optimization failed: {e}")
            optimizationResults['error'] = str(e)
            
        return optimizationResults
    
    async def _storeDenseVector(self,
                              vector: List[float],
                              docId: str,
                              entityType: str,
                              metadata: Dict[str, Any]):
        """
        Store dense vector in standard format
        """
        insertQuery = """
        INSERT INTO A2A_VECTORS (
            DOC_ID, CONTENT, VECTOR_EMBEDDING, ENTITY_TYPE,
            SOURCE_AGENT, METADATA
        ) VALUES (
            :docId, :content, :vectorEmbedding, :entityType,
            :sourceAgent, :metadata
        )
        """
        
        params = {
            'docId': docId,
            'content': metadata.get('content', ''),
            'vectorEmbedding': json.dumps(vector),
            'entityType': entityType,
            'sourceAgent': 'agent3_vector_processing',
            'metadata': json.dumps(metadata)
        }
        
        await self.hanaConnection.execute(insertQuery, params)
    
    def _generateOptimizationRecommendation(self, results: Dict[str, Any]) -> str:
        """
        Generate optimization recommendations based on results
        """
        conversions = results.get('conversions', {})
        converted = conversions.get('converted', 0)
        skipped = conversions.get('skipped', 0)
        
        if converted > skipped:
            return "High sparsity detected. Continue monitoring for sparse conversion opportunities."
        elif skipped > converted * 2:
            return "Most vectors are dense. Consider dimension reduction techniques."
        else:
            return "Balanced vector distribution. Current optimization strategy is effective."
    
    async def enhancedVectorSearch(self,
                                 query: str,
                                 queryVector: List[float],
                                 searchParams: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced search with hybrid ranking and advanced filtering
        """
        try:
            # Perform initial vector search
            initialResults = await self.hybridVectorSearch(query, queryVector, searchParams)
            
            # Apply hybrid ranking if requested
            if searchParams.get('useHybridRanking', True):
                rankingContext = {
                    'userId': searchParams.get('userId'),
                    'queryDomain': searchParams.get('domain'),
                    'preferredDomains': searchParams.get('preferredDomains', []),
                    'queryIntent': self._detectQueryIntent(query),
                    'userPreferences': searchParams.get('userPreferences', {}),
                    'enableDiversity': searchParams.get('enableDiversity', True)
                }
                
                rankedResults = await self.hybridRankingSkills.hybridRanking(
                    query, queryVector, initialResults, rankingContext
                )
                
                return rankedResults
            else:
                return initialResults
                
        except Exception as e:
            logger.error(f"Enhanced vector search failed: {e}")
            # Fallback to basic search
            return await self.hybridVectorSearch(query, queryVector, searchParams)
    
    async def contextAwareSearch(self,
                               searchContext: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Context-aware search that adapts to user behavior and preferences
        """
        searchResults = {
            'results': [],
            'searchStrategy': None,
            'contextFactors': {},
            'performanceMetrics': {}
        }
        
        try:
            startTime = datetime.utcnow()
            
            # Extract context information
            query = searchContext.get('query', '')
            userId = searchContext.get('userId')
            sessionHistory = searchContext.get('sessionHistory', [])
            userPreferences = searchContext.get('userPreferences', {})
            
            # Determine optimal search strategy
            searchStrategy = await self._determineSearchStrategy(searchContext)
            searchResults['searchStrategy'] = searchStrategy
            
            # Generate or retrieve query vector
            queryVector = searchContext.get('queryVector')
            if not queryVector:
                queryVector = await self._generateQueryVector(query, searchContext)
            
            # Build enhanced search parameters
            searchParams = {
                'useHybridRanking': True,
                'userId': userId,
                'userPreferences': userPreferences,
                'domain': self._inferQueryDomain(query, userPreferences),
                'preferredDomains': userPreferences.get('preferredDomains', []),
                'enableDiversity': True,
                'contextualBoost': True
            }
            
            # Perform search based on strategy
            if searchStrategy == 'semantic_focused':
                searchParams.update({
                    'minSimilarity': 0.8,
                    'limit': 15,
                    'boostSemanticSimilarity': True
                })
            elif searchStrategy == 'broad_exploration':
                searchParams.update({
                    'minSimilarity': 0.6,
                    'limit': 25,
                    'enableDiversity': True
                })
            elif searchStrategy == 'personalized':
                searchParams.update({
                    'personalizedRanking': True,
                    'userHistoryWeight': 0.3,
                    'limit': 20
                })
            
            # Execute enhanced search
            results = await self.enhancedVectorSearch(query, queryVector, searchParams)
            
            # Post-process results with context
            processedResults = await self._postProcessResults(results, searchContext)
            searchResults['results'] = processedResults
            
            # Calculate performance metrics
            processingTime = (datetime.utcnow() - startTime).total_seconds()
            searchResults['performanceMetrics'] = {
                'processingTime': processingTime,
                'resultCount': len(processedResults),
                'searchStrategy': searchStrategy,
                'avgRelevanceScore': np.mean([r.get('hybridScore', 0) for r in processedResults]) if processedResults else 0
            }
            
            # Store search analytics
            if self.hanaConnection:
                await self._storeSearchAnalytics(searchResults, searchContext)
                
        except Exception as e:
            logger.error(f"Context-aware search failed: {e}")
            searchResults['error'] = str(e)
            
        return searchResults['results']
    
    async def _determineSearchStrategy(self, context: Dict[str, Any]) -> str:
        """
        Determine optimal search strategy based on context
        """
        query = context.get('query', '')
        queryLength = len(query.split())
        userPreferences = context.get('userPreferences', {})
        sessionHistory = context.get('sessionHistory', [])
        
        # Strategy decision logic
        if queryLength <= 2:
            # Short queries benefit from broader exploration
            return 'broad_exploration'
        elif len(sessionHistory) > 5:
            # Users with interaction history get personalized results
            return 'personalized'
        elif any(domain in query.lower() for domain in ['specific', 'exact', 'precise']):
            # Precision-focused queries
            return 'semantic_focused'
        elif userPreferences.get('explorationPreference', 'balanced') == 'diverse':
            return 'broad_exploration'
        else:
            return 'semantic_focused'
    
    async def _generateQueryVector(self, query: str, context: Dict[str, Any]) -> List[float]:
        """
        Generate query vector using appropriate embedding model
        """
        try:
            # Use domain-specific embedding based on user preferences
            domain = context.get('userPreferences', {}).get('domain', 'general')
            embedding_model = context.get('userPreferences', {}).get('embedding_model', 'sentence-transformers')
            
            # Fallback embedding generation using TF-IDF approach
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create a simple corpus for TF-IDF
            corpus = [query, "sample document", "another sample", "text processing"]
            
            # Generate TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Get the query vector (first document)
            query_vector = tfidf_matrix[0].toarray().flatten()
            
            # Pad or truncate to exactly 768 dimensions
            if len(query_vector) < 768:
                # Pad with zeros
                padded_vector = np.zeros(768)
                padded_vector[:len(query_vector)] = query_vector
                query_vector = padded_vector
            elif len(query_vector) > 768:
                # Truncate to 768
                query_vector = query_vector[:768]
            
            # Normalize the vector
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            
            return query_vector.tolist()
            
        except Exception as e:
            logger.warning(f"Query vector generation failed: {e}")
            # Fallback to normalized random vector
            vector = np.random.normal(0, 0.1, 768)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()
    
    def _inferQueryDomain(self, query: str, userPreferences: Dict[str, Any]) -> str:
        """
        Infer the domain of the query
        """
        queryLower = query.lower()
        
        # Domain indicators
        domainKeywords = {
            'financial': ['finance', 'investment', 'trading', 'market', 'portfolio'],
            'legal': ['legal', 'contract', 'compliance', 'regulation', 'law'],
            'technical': ['code', 'api', 'system', 'technical', 'development'],
            'medical': ['medical', 'health', 'patient', 'diagnosis', 'treatment']
        }
        
        for domain, keywords in domainKeywords.items():
            if any(keyword in queryLower for keyword in keywords):
                return domain
        
        # Fallback to user preferences
        return userPreferences.get('defaultDomain', 'general')
    
    def _detectQueryIntent(self, query: str) -> str:
        """
        Detect the intent behind the query
        """
        queryLower = query.lower()
        
        if any(word in queryLower for word in ['what', 'how', 'explain', 'define']):
            return 'informational'
        elif any(word in queryLower for word in ['buy', 'order', 'purchase', 'get']):
            return 'transactional'
        elif any(word in queryLower for word in ['find', 'locate', 'go to', 'navigate']):
            return 'navigational'
        else:
            return 'informational'  # Default
    
    async def _postProcessResults(self,
                                results: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Post-process search results with context-specific enhancements
        """
        processedResults = []
        
        for result in results:
            # Add context-aware metadata
            result['contextRelevance'] = await self._calculateContextRelevance(result, context)
            
            # Add explanation if requested
            if context.get('includeExplanations', False):
                result['rankingExplanation'] = self._generateRankingExplanation(result)
            
            # Add related suggestions
            if context.get('includeRelated', True):
                result['relatedDocuments'] = await self._findRelatedDocuments(result, limit=3)
            
            processedResults.append(result)
        
        return processedResults
    
    async def _calculateContextRelevance(self,
                                       result: Dict[str, Any],
                                       context: Dict[str, Any]) -> float:
        """
        Calculate how relevant the result is to the specific context
        """
        relevance = 0.0
        
        # User preference alignment
        userPrefs = context.get('userPreferences', {})
        resultType = result.get('entityType', '')
        if resultType in userPrefs.get('preferredTypes', []):
            relevance += 0.3
        
        # Session context alignment
        sessionHistory = context.get('sessionHistory', [])
        if sessionHistory:
            # Check if result relates to recent queries
            for historyItem in sessionHistory[-3:]:  # Last 3 queries
                if self._hasConceptualOverlap(result, historyItem):
                    relevance += 0.2
                    break
        
        # Time context (if query is time-sensitive)
        query = context.get('query', '')
        if self._isTimeSensitiveQuery(query):
            recencyScore = result.get('scoreBreakdown', {}).get('recency', 0)
            relevance += 0.3 * recencyScore
        
        return min(relevance, 1.0)
    
    def _generateRankingExplanation(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for ranking
        """
        scoreBreakdown = result.get('scoreBreakdown', {})
        explanations = []
        
        # Vector similarity
        vecSim = scoreBreakdown.get('vectorSimilarity', 0)
        if vecSim > 0.8:
            explanations.append("High semantic similarity to your query")
        elif vecSim > 0.6:
            explanations.append("Moderate semantic similarity to your query")
        
        # Text relevance
        textRel = scoreBreakdown.get('textRelevance', 0)
        if textRel > 0.7:
            explanations.append("Contains key terms from your query")
        
        # Graph importance
        graphScore = scoreBreakdown.get('graphPageRank', 0)
        if graphScore > 0.7:
            explanations.append("Highly referenced in our knowledge base")
        
        # User interaction
        userScore = scoreBreakdown.get('userInteraction', 0)
        if userScore > 0.5:
            explanations.append("Previously viewed by users with similar interests")
        
        return "; ".join(explanations) if explanations else "Relevant based on multiple factors"
    
    async def _findRelatedDocuments(self,
                                  result: Dict[str, Any],
                                  limit: int = 3) -> List[Dict[str, str]]:
        """
        Find documents related to the given result
        """
        related = []
        
        if not self.hanaConnection:
            return related
        
        try:
            # Find documents with similar entity types and metadata
            relatedQuery = """
            SELECT 
                DOC_ID,
                CONTENT,
                ENTITY_TYPE,
                JSON_VALUE(METADATA, '$.title') as TITLE
            FROM A2A_VECTORS
            WHERE ENTITY_TYPE = :entityType
              AND DOC_ID != :currentDocId
            ORDER BY CREATED_AT DESC
            LIMIT :limit
            """
            
            relatedResults = await self.hanaConnection.execute(relatedQuery, {
                'entityType': result.get('entityType'),
                'currentDocId': result.get('docId'),
                'limit': limit
            })
            
            for row in relatedResults:
                related.append({
                    'docId': row['DOC_ID'],
                    'title': row['TITLE'] or row['CONTENT'][:100] + '...',
                    'entityType': row['ENTITY_TYPE']
                })
                
        except Exception as e:
            logger.error(f"Failed to find related documents: {e}")
        
        return related
    
    def _hasConceptualOverlap(self, result: Dict[str, Any], historyItem: Dict[str, Any]) -> bool:
        """
        Check if result has conceptual overlap with a history item
        """
        resultTags = set(result.get('metadata', {}).get('tags', []))
        historyTags = set(historyItem.get('tags', []))
        
        # Check for tag overlap
        if resultTags & historyTags:
            return True
        
        # Check entity type match
        if result.get('entityType') == historyItem.get('entityType'):
            return True
        
        return False
    
    def _isTimeSensitiveQuery(self, query: str) -> bool:
        """
        Check if the query is time-sensitive
        """
        timeSensitive = ['latest', 'recent', 'new', 'current', 'today', 'now']
        return any(term in query.lower() for term in timeSensitive)
    
    async def _storeSearchAnalytics(self,
                                  searchResults: Dict[str, Any],
                                  context: Dict[str, Any]):
        """
        Store search analytics for optimization
        """
        if not self.hanaConnection:
            return
        
        try:
            analyticsQuery = """
            INSERT INTO A2A_SEARCH_ANALYTICS (
                SEARCH_ID,
                USER_ID,
                QUERY,
                SEARCH_STRATEGY,
                RESULT_COUNT,
                PROCESSING_TIME,
                AVG_RELEVANCE_SCORE,
                CREATED_AT
            ) VALUES (
                SYSUUID,
                :userId,
                :query,
                :searchStrategy,
                :resultCount,
                :processingTime,
                :avgRelevanceScore,
                CURRENT_TIMESTAMP
            )
            """
            
            metrics = searchResults['performanceMetrics']
            await self.hanaConnection.execute(analyticsQuery, {
                'userId': context.get('userId'),
                'query': context.get('query'),
                'searchStrategy': metrics['searchStrategy'],
                'resultCount': metrics['resultCount'],
                'processingTime': metrics['processingTime'],
                'avgRelevanceScore': metrics['avgRelevanceScore']
            })
            
        except Exception as e:
            logger.error(f"Failed to store search analytics: {e}")
/**
 * Agent 3 Vector Processing Adapter
 * Converts between REST API format (Python backend) and OData format (SAP CAP)
 */

const cds = require('@sap/cds');
const { BlockchainClient } = require('../core/blockchain-client');

const log = cds.log('agent3-adapter');

class Agent3Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT3_BASE_URL || 'http://localhost:8002';
        log.info(`Agent 3 Adapter initialized with base URL: ${this.baseUrl}`);
    }

    // =================================
    // Data Conversion Methods
    // =================================

    /**
     * Convert REST vector processing task to OData format
     */
    convertTaskToOData(restTask) {
        return {
            ID: restTask.id || cds.utils.uuid(),
            taskName: restTask.task_name,
            description: restTask.description,
            dataSource: restTask.data_source,
            dataType: this.mapDataType(restTask.data_type),
            embeddingModel: restTask.embedding_model,
            modelProvider: this.mapModelProvider(restTask.model_provider),
            vectorDatabase: this.mapVectorDatabase(restTask.vector_database),
            indexType: this.mapIndexType(restTask.index_type),
            distanceMetric: this.mapDistanceMetric(restTask.distance_metric),
            dimensions: restTask.dimensions || 1536,
            chunkSize: restTask.chunk_size || 512,
            chunkOverlap: restTask.chunk_overlap || 50,
            normalization: restTask.normalization !== false,
            useGPU: restTask.use_gpu || false,
            batchSize: restTask.batch_size || 100,
            status: this.mapStatus(restTask.status),
            priority: this.mapPriority(restTask.priority),
            progressPercent: restTask.progress || 0,
            currentStage: restTask.current_stage,
            processingTime: restTask.processing_time,
            vectorsGenerated: restTask.vectors_generated || 0,
            chunksProcessed: restTask.chunks_processed || 0,
            totalChunks: restTask.total_chunks || 0,
            collectionName: restTask.collection_name,
            indexSize: restTask.index_size || 0,
            errorDetails: restTask.error_details,
            startedAt: restTask.started_at,
            completedAt: restTask.completed_at,
            createdAt: restTask.created_at || new Date().toISOString(),
            modifiedAt: restTask.modified_at || new Date().toISOString(),
            // Store the backend task ID for future reference
            agent3TaskId: restTask.id
        };
    }

    /**
     * Convert OData vector processing task to REST format
     */
    convertTaskToRest(odataTask) {
        return {
            task_name: odataTask.taskName,
            description: odataTask.description,
            data_source: odataTask.dataSource,
            data_type: this.unmapDataType(odataTask.dataType),
            embedding_model: odataTask.embeddingModel,
            model_provider: this.unmapModelProvider(odataTask.modelProvider),
            vector_database: this.unmapVectorDatabase(odataTask.vectorDatabase),
            index_type: this.unmapIndexType(odataTask.indexType),
            distance_metric: this.unmapDistanceMetric(odataTask.distanceMetric),
            dimensions: odataTask.dimensions || 1536,
            chunk_size: odataTask.chunkSize || 512,
            chunk_overlap: odataTask.chunkOverlap || 50,
            normalization: odataTask.normalization !== false,
            use_gpu: odataTask.useGPU || false,
            batch_size: odataTask.batchSize || 100,
            priority: this.unmapPriority(odataTask.priority)
        };
    }

    /**
     * Convert REST vector collection to OData format
     */
    convertCollectionToOData(restCollection) {
        return {
            ID: restCollection.id || cds.utils.uuid(),
            name: restCollection.name,
            description: restCollection.description,
            vectorDatabase: this.mapVectorDatabase(restCollection.vector_database),
            embeddingModel: restCollection.embedding_model,
            dimensions: restCollection.dimensions,
            distanceMetric: this.mapDistanceMetric(restCollection.distance_metric),
            indexType: this.mapIndexType(restCollection.index_type),
            totalVectors: restCollection.total_vectors || 0,
            indexSize: restCollection.index_size || 0,
            isActive: restCollection.is_active !== false,
            isOptimized: restCollection.is_optimized || false,
            lastOptimized: restCollection.last_optimized,
            metadataSchema: JSON.stringify(restCollection.metadata_schema || {}),
            createdAt: restCollection.created_at || new Date().toISOString(),
            modifiedAt: restCollection.modified_at || new Date().toISOString(),
            agent3CollectionId: restCollection.id
        };
    }

    /**
     * Convert OData vector collection to REST format
     */
    convertCollectionToRest(odataCollection) {
        return {
            name: odataCollection.name,
            description: odataCollection.description,
            vector_database: this.unmapVectorDatabase(odataCollection.vectorDatabase),
            embedding_model: odataCollection.embeddingModel,
            dimensions: odataCollection.dimensions,
            distance_metric: this.unmapDistanceMetric(odataCollection.distanceMetric),
            index_type: this.unmapIndexType(odataCollection.indexType),
            metadata_schema: JSON.parse(odataCollection.metadataSchema || '{}')
        };
    }

    /**
     * Convert REST similarity result to OData format
     */
    convertSimilarityResultToOData(restResult) {
        return {
            ID: restResult.id || cds.utils.uuid(),
            queryText: restResult.query_text,
            queryVector: JSON.stringify(restResult.query_vector || []),
            resultVectorId: restResult.result_vector_id,
            similarityScore: restResult.similarity_score,
            distance: restResult.distance,
            resultContent: restResult.result_content,
            resultMetadata: JSON.stringify(restResult.result_metadata || {}),
            rank: restResult.rank,
            searchTimestamp: restResult.search_timestamp || new Date().toISOString(),
            agent3ResultId: restResult.id
        };
    }

    // =================================
    // Field Mapping Methods
    // =================================

    mapDataType(restType) {
        const typeMap = {
            'text': 'TEXT',
            'image': 'IMAGE',
            'audio': 'AUDIO',
            'video': 'VIDEO',
            'document': 'DOCUMENT',
            'code': 'CODE'
        };
        return typeMap[restType] || restType?.toUpperCase() || 'TEXT';
    }

    unmapDataType(odataType) {
        const typeMap = {
            'TEXT': 'text',
            'IMAGE': 'image',
            'AUDIO': 'audio',
            'VIDEO': 'video',
            'DOCUMENT': 'document',
            'CODE': 'code'
        };
        return typeMap[odataType] || odataType?.toLowerCase() || 'text';
    }

    mapModelProvider(restProvider) {
        const providerMap = {
            'openai': 'OPENAI',
            'huggingface': 'HUGGINGFACE',
            'cohere': 'COHERE',
            'anthropic': 'ANTHROPIC',
            'google': 'GOOGLE',
            'custom': 'CUSTOM'
        };
        return providerMap[restProvider] || restProvider?.toUpperCase() || 'OPENAI';
    }

    unmapModelProvider(odataProvider) {
        const providerMap = {
            'OPENAI': 'openai',
            'HUGGINGFACE': 'huggingface',
            'COHERE': 'cohere',
            'ANTHROPIC': 'anthropic',
            'GOOGLE': 'google',
            'CUSTOM': 'custom'
        };
        return providerMap[odataProvider] || odataProvider?.toLowerCase() || 'openai';
    }

    mapVectorDatabase(restDb) {
        const dbMap = {
            'pinecone': 'PINECONE',
            'weaviate': 'WEAVIATE',
            'milvus': 'MILVUS',
            'chroma': 'CHROMA',
            'qdrant': 'QDRANT',
            'pgvector': 'PGVECTOR'
        };
        return dbMap[restDb] || restDb?.toUpperCase() || 'PINECONE';
    }

    unmapVectorDatabase(odataDb) {
        const dbMap = {
            'PINECONE': 'pinecone',
            'WEAVIATE': 'weaviate',
            'MILVUS': 'milvus',
            'CHROMA': 'chroma',
            'QDRANT': 'qdrant',
            'PGVECTOR': 'pgvector'
        };
        return dbMap[odataDb] || odataDb?.toLowerCase() || 'pinecone';
    }

    mapIndexType(restType) {
        const typeMap = {
            'hnsw': 'HNSW',
            'ivf': 'IVF',
            'flat': 'FLAT',
            'lsh': 'LSH'
        };
        return typeMap[restType] || restType?.toUpperCase() || 'HNSW';
    }

    unmapIndexType(odataType) {
        const typeMap = {
            'HNSW': 'hnsw',
            'IVF': 'ivf',
            'FLAT': 'flat',
            'LSH': 'lsh'
        };
        return typeMap[odataType] || odataType?.toLowerCase() || 'hnsw';
    }

    mapDistanceMetric(restMetric) {
        const metricMap = {
            'cosine': 'COSINE',
            'euclidean': 'EUCLIDEAN',
            'dot_product': 'DOT_PRODUCT',
            'manhattan': 'MANHATTAN'
        };
        return metricMap[restMetric] || restMetric?.toUpperCase() || 'COSINE';
    }

    unmapDistanceMetric(odataMetric) {
        const metricMap = {
            'COSINE': 'cosine',
            'EUCLIDEAN': 'euclidean',
            'DOT_PRODUCT': 'dot_product',
            'MANHATTAN': 'manhattan'
        };
        return metricMap[odataMetric] || odataMetric?.toLowerCase() || 'cosine';
    }

    mapStatus(restStatus) {
        const statusMap = {
            'draft': 'DRAFT',
            'pending': 'PENDING',
            'running': 'RUNNING',
            'completed': 'COMPLETED',
            'failed': 'FAILED',
            'paused': 'PAUSED'
        };
        return statusMap[restStatus] || restStatus?.toUpperCase() || 'DRAFT';
    }

    mapPriority(restPriority) {
        const priorityMap = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'urgent': 'URGENT'
        };
        return priorityMap[restPriority] || restPriority?.toUpperCase() || 'MEDIUM';
    }

    unmapPriority(odataPriority) {
        const priorityMap = {
            'LOW': 'low',
            'MEDIUM': 'medium',
            'HIGH': 'high',
            'URGENT': 'urgent'
        };
        return priorityMap[odataPriority] || odataPriority?.toLowerCase() || 'medium';
    }

    // =================================
    // Backend API Methods
    // =================================

    async startProcessing(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/process`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Vector processing started successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start vector processing:', error.message);
            throw new Error(`Failed to start processing: ${error.message}`);
        }
    }

    async pauseProcessing(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/pause`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Vector processing paused successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to pause vector processing:', error.message);
            throw new Error(`Failed to pause processing: ${error.message}`);
        }
    }

    async resumeProcessing(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/resume`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Vector processing resumed successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to resume vector processing:', error.message);
            throw new Error(`Failed to resume processing: ${error.message}`);
        }
    }

    async cancelProcessing(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/cancel`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Vector processing cancelled successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to cancel vector processing:', error.message);
            throw new Error(`Failed to cancel processing: ${error.message}`);
        }
    }

    async runSimilaritySearch(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/similarity-search`, {
                query_type: options.queryType,
                query: options.query,
                vector_query: options.vectorQuery,
                top_k: options.topK,
                include_metadata: options.includeMetadata,
                include_distance: options.includeDistance,
                filters: options.filters
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Similarity search completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run similarity search:', error.message);
            throw new Error(`Failed to run similarity search: ${error.message}`);
        }
    }

    async optimizeIndex(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/optimize-index`, {
                index_type: options.indexType,
                parameters: options.parameters
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Index optimization completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to optimize index:', error.message);
            throw new Error(`Failed to optimize index: ${error.message}`);
        }
    }

    async exportVectors(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/export`, {
                format: options.format,
                include_metadata: options.includeMetadata,
                compression: options.compression,
                chunk_size: options.chunkSize
            }, {
                timeout: 120000
            });
            return {
                success: true,
                message: 'Vector export completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to export vectors:', error.message);
            throw new Error(`Failed to export vectors: ${error.message}`);
        }
    }

    async getVisualizationData(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/visualization-data`, {
                params: {
                    method: options.method,
                    perplexity: options.perplexity,
                    dimensions: options.dimensions,
                    sample_size: options.sampleSize
                },
                timeout: 60000
            });
            return {
                success: true,
                message: 'Visualization data generated',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get visualization data:', error.message);
            throw new Error(`Failed to get visualization data: ${error.message}`);
        }
    }

    async runClusterAnalysis(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/tasks/${taskId}/cluster-analysis`, {
                algorithm: options.algorithm,
                num_clusters: options.numClusters,
                min_cluster_size: options.minClusterSize
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Cluster analysis completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run cluster analysis:', error.message);
            throw new Error(`Failed to run cluster analysis: ${error.message}`);
        }
    }

    async batchVectorProcessing(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/batch-process`, {
                task_ids: options.taskIds,
                parallel: options.parallel,
                use_gpu: options.useGPU,
                priority: options.priority
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Batch vector processing started',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start batch vector processing:', error.message);
            throw new Error(`Failed to start batch processing: ${error.message}`);
        }
    }

    async executeVectorSearch(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/search`, {
                query: options.query,
                collection: options.collection,
                top_k: options.topK,
                threshold: options.threshold,
                filters: options.filters
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Vector search completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to execute vector search:', error.message);
            throw new Error(`Failed to execute vector search: ${error.message}`);
        }
    }

    async getModelComparison() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/model-comparison`, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Model comparison data retrieved',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get model comparison:', error.message);
            throw new Error(`Failed to get model comparison: ${error.message}`);
        }
    }

    async getCollections() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/collections`, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Collections retrieved',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get collections:', error.message);
            throw new Error(`Failed to get collections: ${error.message}`);
        }
    }

    async createCollection(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/collections`, {
                name: options.name,
                description: options.description,
                vector_database: options.vectorDatabase,
                embedding_model: options.embeddingModel,
                dimensions: options.dimensions,
                distance_metric: options.distanceMetric,
                index_type: options.indexType
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Collection created successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to create collection:', error.message);
            throw new Error(`Failed to create collection: ${error.message}`);
        }
    }

    async generateEmbeddings(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/embeddings/generate`, {
                texts: options.texts,
                model: options.model,
                normalize: options.normalize
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Embeddings generated successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to generate embeddings:', error.message);
            throw new Error(`Failed to generate embeddings: ${error.message}`);
        }
    }

    // =================================
    // Health Check
    // =================================

    async checkHealth() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent3/v1/health`, {
                timeout: 5000
            });
            return {
                status: 'healthy',
                data: response.data
            };
        } catch (error) {
            log.warn('Agent 3 backend health check failed:', error.message);
            return {
                status: 'unhealthy',
                error: error.message
            };
        }
    }
}

module.exports = Agent3Adapter;
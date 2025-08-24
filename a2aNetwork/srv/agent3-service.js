/**
 * Agent 3 Vector Processing Service Implementation
 * Bridges SAP CAP OData service with Python FastAPI backend
 */

const cds = require('@sap/cds');
const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');

// Import the adapter for REST/OData conversion
const Agent3Adapter = require('./agent3-adapter');

const log = cds.log('agent3-service');

// Initialize the adapter
const adapter = new Agent3Adapter();

// Backend configuration
const AGENT3_BASE_URL = process.env.AGENT3_BASE_URL || 'http://localhost:8002';

module.exports = cds.service.impl(async function() {
    
    const { VectorProcessingTasks, VectorCollections, VectorSimilarityResults, VectorProcessingJobs } = this.entities;
    
    // =================================
    // CRUD Operations for VectorProcessingTasks
    // =================================
    
    this.on('READ', VectorProcessingTasks, async (req) => {
        try {
            log.info('Reading VectorProcessingTasks');
            
            // Get data from Python backend
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/tasks`, {
                timeout: 30000
            });
            
            // Convert to OData format using adapter
            const odataResults = response.data.map(task => adapter.convertTaskToOData(task));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading VectorProcessingTasks:', error.message);
            req.error(503, `Agent 3 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', VectorProcessingTasks, async (req) => {
        try {
            log.info('Creating VectorProcessingTask');
            
            // Convert OData to REST format
            const restData = adapter.convertTaskToRest(req.data);
            
            // Send to Python backend
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/tasks`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            // Convert response back to OData format
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error creating VectorProcessingTask:', error.message);
            req.error(500, `Failed to create task: ${error.message}`);
        }
    });
    
    this.on('UPDATE', VectorProcessingTasks, async (req) => {
        try {
            log.info('Updating VectorProcessingTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            const restData = adapter.convertTaskToRest(req.data);
            
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/tasks/${taskId}`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error updating VectorProcessingTask:', error.message);
            req.error(500, `Failed to update task: ${error.message}`);
        }
    });
    
    this.on('DELETE', VectorProcessingTasks, async (req) => {
        try {
            log.info('Deleting VectorProcessingTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            
            await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/tasks/${taskId}`, {
                timeout: 30000
            });
            
            return req.params[0];
        } catch (error) {
            log.error('Error deleting VectorProcessingTask:', error.message);
            req.error(500, `Failed to delete task: ${error.message}`);
        }
    });
    
    // =================================
    // Action Handlers for VectorProcessingTasks
    // =================================
    
    this.on('startProcessing', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Starting vector processing for task: ${ID}`);
            
            // Get task details from database
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            // Start processing via backend
            const result = await adapter.startProcessing(task.agent3TaskId);
            
            // Update task status
            await UPDATE(VectorProcessingTasks)
                .set({ status: 'RUNNING', startedAt: new Date().toISOString() })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error starting vector processing:', error.message);
            req.error(500, `Failed to start processing: ${error.message}`);
        }
    });
    
    this.on('pauseProcessing', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Pausing vector processing for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.pauseProcessing(task.agent3TaskId);
            
            await UPDATE(VectorProcessingTasks)
                .set({ status: 'PAUSED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error pausing vector processing:', error.message);
            req.error(500, `Failed to pause processing: ${error.message}`);
        }
    });
    
    this.on('resumeProcessing', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Resuming vector processing for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.resumeProcessing(task.agent3TaskId);
            
            await UPDATE(VectorProcessingTasks)
                .set({ status: 'RUNNING' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error resuming vector processing:', error.message);
            req.error(500, `Failed to resume processing: ${error.message}`);
        }
    });
    
    this.on('cancelProcessing', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Cancelling vector processing for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.cancelProcessing(task.agent3TaskId);
            
            await UPDATE(VectorProcessingTasks)
                .set({ status: 'CANCELLED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error cancelling vector processing:', error.message);
            req.error(500, `Failed to cancel processing: ${error.message}`);
        }
    });
    
    this.on('runSimilaritySearch', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { queryType, query, vectorQuery, topK, includeMetadata, includeDistance, filters } = req.data;
            
            log.info(`Running similarity search for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runSimilaritySearch(task.agent3TaskId, {
                queryType,
                query,
                vectorQuery,
                topK,
                includeMetadata,
                includeDistance,
                filters
            });
            
            return result;
        } catch (error) {
            log.error('Error running similarity search:', error.message);
            req.error(500, `Failed to run similarity search: ${error.message}`);
        }
    });
    
    this.on('optimizeIndex', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { indexType, parameters } = req.data;
            
            log.info(`Optimizing index for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.optimizeIndex(task.agent3TaskId, {
                indexType,
                parameters
            });
            
            return result;
        } catch (error) {
            log.error('Error optimizing index:', error.message);
            req.error(500, `Failed to optimize index: ${error.message}`);
        }
    });
    
    this.on('exportVectors', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { format, includeMetadata, compression, chunkSize } = req.data;
            
            log.info(`Exporting vectors for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.exportVectors(task.agent3TaskId, {
                format,
                includeMetadata,
                compression,
                chunkSize
            });
            
            return result;
        } catch (error) {
            log.error('Error exporting vectors:', error.message);
            req.error(500, `Failed to export vectors: ${error.message}`);
        }
    });
    
    this.on('getVisualizationData', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { method, perplexity, dimensions, sampleSize } = req.data;
            
            log.info(`Getting visualization data for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.getVisualizationData(task.agent3TaskId, {
                method,
                perplexity,
                dimensions,
                sampleSize
            });
            
            return result;
        } catch (error) {
            log.error('Error getting visualization data:', error.message);
            req.error(500, `Failed to get visualization data: ${error.message}`);
        }
    });
    
    this.on('runClusterAnalysis', VectorProcessingTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { algorithm, numClusters, minClusterSize } = req.data;
            
            log.info(`Running cluster analysis for task: ${ID}`);
            
            const task = await SELECT.one.from(VectorProcessingTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runClusterAnalysis(task.agent3TaskId, {
                algorithm,
                numClusters,
                minClusterSize
            });
            
            return result;
        } catch (error) {
            log.error('Error running cluster analysis:', error.message);
            req.error(500, `Failed to run cluster analysis: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for VectorCollections
    // =================================
    
    this.on('READ', VectorCollections, async (req) => {
        try {
            log.info('Reading VectorCollections');
            
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/collections`, {
                timeout: 30000
            });
            
            const odataResults = response.data.map(collection => adapter.convertCollectionToOData(collection));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading VectorCollections:', error.message);
            req.error(503, `Agent 3 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', VectorCollections, async (req) => {
        try {
            log.info('Creating VectorCollection');
            
            const restData = adapter.convertCollectionToRest(req.data);
            
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/collections`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            const odataResult = adapter.convertCollectionToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error creating VectorCollection:', error.message);
            req.error(500, `Failed to create collection: ${error.message}`);
        }
    });
    
    // =================================
    // Service-level Actions
    // =================================
    
    this.on('batchVectorProcessing', async (req) => {
        try {
            const { taskIds, parallel, useGPU, priority } = req.data;
            
            log.info(`Starting batch vector processing for ${taskIds.length} tasks`);
            
            const result = await adapter.batchVectorProcessing({
                taskIds,
                parallel,
                useGPU,
                priority
            });
            
            return result;
        } catch (error) {
            log.error('Error starting batch vector processing:', error.message);
            req.error(500, `Failed to start batch processing: ${error.message}`);
        }
    });
    
    this.on('executeVectorSearch', async (req) => {
        try {
            const { query, collection, topK, threshold, filters } = req.data;
            
            log.info(`Executing vector search in collection: ${collection}`);
            
            const result = await adapter.executeVectorSearch({
                query,
                collection,
                topK,
                threshold,
                filters
            });
            
            return result;
        } catch (error) {
            log.error('Error executing vector search:', error.message);
            req.error(500, `Failed to execute vector search: ${error.message}`);
        }
    });
    
    this.on('getModelComparison', async (req) => {
        try {
            log.info('Getting model comparison data');
            
            const result = await adapter.getModelComparison();
            
            return result;
        } catch (error) {
            log.error('Error getting model comparison:', error.message);
            req.error(500, `Failed to get model comparison: ${error.message}`);
        }
    });
    
    this.on('getCollections', async (req) => {
        try {
            log.info('Getting vector collections');
            
            const result = await adapter.getCollections();
            
            return result;
        } catch (error) {
            log.error('Error getting collections:', error.message);
            req.error(500, `Failed to get collections: ${error.message}`);
        }
    });
    
    this.on('createCollection', async (req) => {
        try {
            const { name, description, vectorDatabase, embeddingModel, dimensions, distanceMetric, indexType } = req.data;
            
            log.info(`Creating vector collection: ${name}`);
            
            const result = await adapter.createCollection({
                name,
                description,
                vectorDatabase,
                embeddingModel,
                dimensions,
                distanceMetric,
                indexType
            });
            
            return result;
        } catch (error) {
            log.error('Error creating collection:', error.message);
            req.error(500, `Failed to create collection: ${error.message}`);
        }
    });
    
    this.on('generateEmbeddings', async (req) => {
        try {
            const { texts, model, normalize } = req.data;
            
            log.info(`Generating embeddings for ${texts.length} texts`);
            
            const result = await adapter.generateEmbeddings({
                texts,
                model,
                normalize
            });
            
            return result;
        } catch (error) {
            log.error('Error generating embeddings:', error.message);
            req.error(500, `Failed to generate embeddings: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for VectorSimilarityResults
    // =================================
    
    this.on('READ', VectorSimilarityResults, async (req) => {
        try {
            log.info('Reading VectorSimilarityResults');
            
            const response = await blockchainClient.sendMessage(`${AGENT3_BASE_URL}/a2a/agent3/v1/similarity-results`, {
                timeout: 30000
            });
            
            const odataResults = response.data.map(result => adapter.convertSimilarityResultToOData(result));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading VectorSimilarityResults:', error.message);
            req.error(503, `Agent 3 Backend Error: ${error.message}`);
        }
    });
    
    // Error handling middleware
    this.on('error', (err, req) => {
        log.error('Service error:', err);
        
        // Provide user-friendly error messages
        switch (err.code) {
            case 'ECONNREFUSED':
                req.error(503, 'Agent 3 vector processing service is currently unavailable');
                break;
            case 'ETIMEDOUT':
                req.error(408, 'Request to Agent 3 service timed out');
                break;
            default:
                req.error(500, 'Internal server error in Agent 3 service');
        }
    });
    
    log.info('Agent 3 Vector Processing Service initialized successfully');
});
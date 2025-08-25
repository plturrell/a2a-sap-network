const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
const Agent2Adapter = require('./agent2-adapter');

/**
 * Agent 2 AI Preparation Service Implementation
 * Bridges SAP CAP OData service with Agent 2 Python FastAPI backend
 */
module.exports = cds.service.impl(async function() {

    const { AIPreparationTasks, AIPreparationFeatures } = this.entities;
    const adapter = new Agent2Adapter();

    // Entity handlers for AIPreparationTasks

    // CREATE handler - Create new AI preparation task
    this.before('CREATE', AIPreparationTasks, async (req) => {
        const task = req.data;

        // Validate required fields
        if (!task.taskName || !task.datasetName || !task.modelType || !task.dataType) {
            req.error(400, 'Missing required fields: taskName, datasetName, modelType, dataType');
        }

        // Set defaults and computed fields
        task.status = 'DRAFT';
        task.progressPercent = 0;
        task.priority = task.priority || 'MEDIUM';

        try {
            // Create task in Agent 2 backend using adapter
            const backendTask = await adapter.createTask(task);

            // Store backend task ID in the database for future reference
            task.agent2TaskId = backendTask.agent2TaskId;

        } catch (error) {
            req.error(500, `Failed to create task in Agent 2 backend: ${error.message}`);
        }
    });

    // READ handler - Fetch task status from Agent 2 backend
    this.after('READ', AIPreparationTasks, async (tasks, req) => {
        if (!Array.isArray(tasks)) tasks = [tasks];

        for (const task of tasks.filter(t => t)) {
            if (task.agent2TaskId) {
                try {
                    // Get real-time status from Agent 2 backend using adapter
                    const backendTask = await adapter.getTask(task.agent2TaskId);

                    // Update task with real-time data
                    Object.assign(task, {
                        status: backendTask.status,
                        progressPercent: backendTask.progressPercent,
                        currentStage: backendTask.currentStage,
                        processingTime: backendTask.processingTime,
                        errorDetails: backendTask.errorDetails,
                        resultsSummary: backendTask.resultsSummary
                    });

                } catch (error) {
                    console.warn(`Failed to get real-time status for task ${task.ID}:`, error.message);
                    // Continue with cached data from database
                }
            }
        }
    });

    // Action implementations

    // Start AI preparation process
    this.on('startPreparation', AIPreparationTasks, async (req) => {
        const { ID } = req.params[0];

        try {
            const task = await SELECT.one.from(AIPreparationTasks).where({ ID });
            if (!task) throw new Error('Task not found');

            if (!task.agent2TaskId) {
                throw new Error('Task not properly initialized with Agent 2 backend');
            }

            // Start preparation in Agent 2 backend using adapter
            const result = await adapter.startPreparation(task.agent2TaskId);

            // Update task status in database
            await UPDATE(AIPreparationTasks).set({
                status: 'RUNNING',
                startedAt: new Date().toISOString(),
                currentStage: 'INITIALIZATION'
            }).where({ ID });

            return JSON.stringify({
                success: true,
                jobId: result.job_id,
                estimatedTime: result.estimated_time,
                message: 'AI data preparation started successfully'
            });

        } catch (error) {
            await UPDATE(AIPreparationTasks).set({
                status: 'FAILED',
                errorDetails: error.message
            }).where({ ID });

            throw new Error(`Failed to start preparation: ${error.message}`);
        }
    });

    // Analyze features
    this.on('analyzeFeatures', AIPreparationTasks, async (req) => {
        const { ID } = req.params[0];

        try {
            const task = await SELECT.one.from(AIPreparationTasks).where({ ID });
            if (!task || !task.agent2TaskId) throw new Error('Task not found or not initialized');

            // Call Agent 2 backend for feature analysis using adapter
            const analysisResult = await adapter.analyzeFeatures(task.agent2TaskId);

            // Store feature analysis results
            if (analysisResult.features) {
                await this.handleFeatureAnalysisResults(ID, analysisResult.features);
            }

            return JSON.stringify(analysisResult);

        } catch (error) {
            throw new Error(`Feature analysis failed: ${error.message}`);
        }
    });

    // Generate embeddings
    this.on('generateEmbeddings', AIPreparationTasks, async (req) => {
        const { ID } = req.params[0];
        const { model, dimensions, normalization, batchSize, useGPU } = req.data;

        try {
            const task = await SELECT.one.from(AIPreparationTasks).where({ ID });
            if (!task || !task.agent2TaskId) throw new Error('Task not found or not initialized');

            const embeddingConfig = {
                model: model || 'text-embedding-ada-002',
                dimensions: dimensions || 768,
                normalization: normalization !== false,
                batchSize: batchSize || 32,
                useGPU: useGPU || false
            };

            const result = await adapter.generateEmbeddings(task.agent2TaskId, embeddingConfig);

            return JSON.stringify(result);

        } catch (error) {
            throw new Error(`Failed to generate embeddings: ${error.message}`);
        }
    });

    // Export prepared data
    this.on('exportPreparedData', AIPreparationTasks, async (req) => {
        const { ID } = req.params[0];
        const { format, includeMetadata, splitData, compression } = req.data;

        try {
            const task = await SELECT.one.from(AIPreparationTasks).where({ ID });
            if (!task || !task.agent2TaskId) throw new Error('Task not found or not initialized');

            const exportConfig = {
                format: format || 'TENSORFLOW',
                include_metadata: includeMetadata !== false,
                split_data: splitData !== false,
                compression: compression || 'GZIP'
            };

            const result = await callAgent2Backend(`/tasks/${task.agent2TaskId}/export`, 'POST', exportConfig);

            return JSON.stringify(result);

        } catch (error) {
            throw new Error(`Export failed: ${error.message}`);
        }
    });

    // Optimize hyperparameters
    this.on('optimizeHyperparameters', AIPreparationTasks, async (req) => {
        const { ID } = req.params[0];
        const { method, trials, timeout, earlyStop } = req.data;

        try {
            const task = await SELECT.one.from(AIPreparationTasks).where({ ID });
            if (!task || !task.agent2TaskId) throw new Error('Task not found or not initialized');

            const optimizationConfig = {
                method: method || 'BAYESIAN',
                trials: trials || 100,
                timeout: timeout || 3600,
                early_stop: earlyStop !== false
            };

            const result = await callAgent2Backend(`/tasks/${task.agent2TaskId}/optimize`, 'POST', optimizationConfig);

            return JSON.stringify(result);

        } catch (error) {
            throw new Error(`Hyperparameter optimization failed: ${error.message}`);
        }
    });

    // Service-level actions

    // Get data profile
    this.on('getDataProfile', async (req) => {
        try {
            const profile = await adapter.getDataProfile();
            return JSON.stringify(profile);
        } catch (error) {
            throw new Error(`Failed to get data profile: ${error.message}`);
        }
    });

    // Batch prepare multiple tasks
    this.on('batchPrepare', async (req) => {
        const { taskIds, parallel, gpuAcceleration } = req.data;

        try {
            const batchConfig = {
                taskIds: taskIds,
                parallel: parallel !== false,
                gpuAcceleration: gpuAcceleration !== false
            };

            const result = await adapter.batchPrepare(batchConfig);

            // Update all tasks to running status
            await UPDATE(AIPreparationTasks).set({
                status: 'RUNNING',
                startedAt: new Date().toISOString()
            }).where({ ID: { in: taskIds } });

            return JSON.stringify(result);

        } catch (error) {
            throw new Error(`Batch preparation failed: ${error.message}`);
        }
    });

    // Start AutoML wizard
    this.on('startAutoML', async (req) => {
        const { dataset, problemType, targetColumn, evaluationMetric, timeLimit, maxModels, includeEnsemble, crossValidation } = req.data;

        try {
            const autoMLConfig = {
                dataset,
                problem_type: problemType,
                target_column: targetColumn,
                evaluation_metric: evaluationMetric,
                time_limit: timeLimit || 60,
                max_models: maxModels || 10,
                include_ensemble: includeEnsemble !== false,
                cross_validation: crossValidation || 5
            };

            const result = await callAgent2Backend('/automl', 'POST', autoMLConfig);

            return JSON.stringify(result);

        } catch (error) {
            throw new Error(`AutoML failed: ${error.message}`);
        }
    });

    // Helper functions

    async function handleFeatureAnalysisResults(taskId, features) {
        // Delete existing features for this task
        await DELETE.from(AIPreparationFeatures).where({ task_ID: taskId });

        // Features are already converted to OData format by the adapter
        const featureRecords = features.map(feature => ({
            ...feature,
            task_ID: taskId
        }));

        await INSERT.into(AIPreparationFeatures).entries(featureRecords);
    }

    // Health check for Agent 2 backend connectivity
    this.on('healthCheck', async () => {
        return await adapter.healthCheck();
    });
});
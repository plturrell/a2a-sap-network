const cds = require('@sap/cds');
const axios = require('axios');

/**
 * Agent 2 OData to REST API Adapter
 * Converts between SAP OData format and Agent 2 Python FastAPI REST format
 */
class Agent2Adapter {
    constructor() {
        this.baseURL = process.env.AGENT2_BASE_URL || 'http://localhost:8001';
        this.apiBase = `${this.baseURL}/a2a/agent2/v1`;
        this.log = cds.log('agent2-adapter');
    }

    // Convert OData query to Agent 2 REST parameters
    convertODataQuery(odataQuery) {
        const params = {};
        
        // Handle $select
        if (odataQuery.$select) {
            params.fields = odataQuery.$select;
        }
        
        // Handle $filter
        if (odataQuery.$filter) {
            // Convert OData filter to simple query parameters
            // This is a simplified conversion - extend as needed
            const filter = odataQuery.$filter;
            if (filter.includes('status eq')) {
                const status = filter.match(/status eq '([^']+)'/)?.[1];
                if (status) params.status = status.toLowerCase();
            }
            if (filter.includes('modelType eq')) {
                const modelType = filter.match(/modelType eq '([^']+)'/)?.[1];
                if (modelType) params.model_type = modelType.toLowerCase();
            }
        }
        
        // Handle $orderby
        if (odataQuery.$orderby) {
            params.sort = odataQuery.$orderby.replace(/ desc$/, ':desc').replace(/ asc$/, ':asc');
        }
        
        // Handle $top and $skip
        if (odataQuery.$top) {
            params.limit = parseInt(odataQuery.$top);
        }
        if (odataQuery.$skip) {
            params.offset = parseInt(odataQuery.$skip);
        }
        
        return params;
    }

    // Convert Agent 2 REST task to OData format
    convertTaskToOData(restTask) {
        return {
            ID: restTask.id || cds.utils.uuid(),
            taskName: restTask.task_name || restTask.name,
            description: restTask.description,
            datasetName: restTask.dataset_name,
            modelType: this.mapModelType(restTask.model_type),
            dataType: this.mapDataType(restTask.data_type),
            framework: this.mapFramework(restTask.framework),
            splitRatio: restTask.split_ratio || 80,
            validationStrategy: this.mapValidationStrategy(restTask.validation_strategy),
            randomSeed: restTask.random_seed || 42,
            featureSelection: restTask.feature_selection !== false,
            autoFeatureEngineering: restTask.auto_feature_engineering !== false,
            optimizationMetric: this.mapOptimizationMetric(restTask.optimization_metric),
            useGPU: restTask.use_gpu || false,
            distributed: restTask.distributed || false,
            memoryOptimized: restTask.memory_optimized || false,
            cacheResults: restTask.cache_results !== false,
            status: this.mapStatus(restTask.status),
            priority: this.mapPriority(restTask.priority),
            progressPercent: restTask.progress || 0,
            currentStage: restTask.current_stage,
            processingTime: restTask.processing_time,
            resultsSummary: restTask.results ? JSON.stringify(restTask.results) : null,
            errorDetails: restTask.error_details,
            startedAt: restTask.started_at,
            completedAt: restTask.completed_at,
            createdAt: restTask.created_at,
            modifiedAt: restTask.modified_at,
            agent2TaskId: restTask.id
        };
    }

    // Convert OData task to Agent 2 REST format
    convertTaskFromOData(odataTask) {
        return {
            task_name: odataTask.taskName,
            description: odataTask.description,
            dataset_name: odataTask.datasetName,
            model_type: odataTask.modelType?.toLowerCase(),
            data_type: odataTask.dataType?.toLowerCase(),
            framework: odataTask.framework?.toLowerCase() || 'tensorflow',
            split_ratio: odataTask.splitRatio || 80,
            validation_strategy: odataTask.validationStrategy?.toLowerCase() || 'kfold',
            random_seed: odataTask.randomSeed || 42,
            feature_selection: odataTask.featureSelection !== false,
            auto_feature_engineering: odataTask.autoFeatureEngineering !== false,
            optimization_metric: odataTask.optimizationMetric?.toLowerCase() || 'auto',
            use_gpu: odataTask.useGPU || false,
            distributed: odataTask.distributed || false,
            memory_optimized: odataTask.memoryOptimized || false,
            cache_results: odataTask.cacheResults !== false,
            priority: odataTask.priority?.toLowerCase() || 'medium'
        };
    }

    // Convert Agent 2 feature to OData format
    convertFeatureToOData(restFeature, taskId) {
        return {
            ID: restFeature.id || cds.utils.uuid(),
            task_ID: taskId,
            name: restFeature.name,
            type: this.mapFeatureType(restFeature.type),
            dataType: restFeature.data_type,
            isTarget: restFeature.is_target || false,
            isSelected: restFeature.is_selected !== false,
            importance: restFeature.importance || 0,
            missingPercent: restFeature.missing_percent || 0,
            uniqueValues: restFeature.unique_values,
            meanValue: restFeature.mean_value,
            stdDev: restFeature.std_dev,
            minValue: restFeature.min_value,
            maxValue: restFeature.max_value,
            engineeringApplied: restFeature.engineering_applied
        };
    }

    // Mapping functions for enum values
    mapModelType(restType) {
        const map = {
            'classification': 'CLASSIFICATION',
            'regression': 'REGRESSION',
            'clustering': 'CLUSTERING',
            'embedding': 'EMBEDDING',
            'llm': 'LLM',
            'time_series': 'TIME_SERIES',
            'recommendation': 'RECOMMENDATION',
            'anomaly': 'ANOMALY'
        };
        return map[restType?.toLowerCase()] || 'CLASSIFICATION';
    }

    mapDataType(restType) {
        const map = {
            'tabular': 'TABULAR',
            'text': 'TEXT',
            'image': 'IMAGE',
            'audio': 'AUDIO',
            'video': 'VIDEO',
            'time_series': 'TIME_SERIES',
            'graph': 'GRAPH'
        };
        return map[restType?.toLowerCase()] || 'TABULAR';
    }

    mapFramework(restFramework) {
        const map = {
            'tensorflow': 'TENSORFLOW',
            'pytorch': 'PYTORCH',
            'scikit_learn': 'SCIKIT_LEARN',
            'xgboost': 'XGBOOST',
            'huggingface': 'HUGGINGFACE',
            'auto': 'AUTO'
        };
        return map[restFramework?.toLowerCase()] || 'TENSORFLOW';
    }

    mapValidationStrategy(restStrategy) {
        const map = {
            'kfold': 'KFOLD',
            'holdout': 'HOLDOUT'
        };
        return map[restStrategy?.toLowerCase()] || 'KFOLD';
    }

    mapOptimizationMetric(restMetric) {
        const map = {
            'auto': 'AUTO',
            'accuracy': 'ACCURACY',
            'auc': 'AUC',
            'f1': 'F1',
            'mse': 'MSE',
            'mae': 'MAE',
            'perplexity': 'PERPLEXITY'
        };
        return map[restMetric?.toLowerCase()] || 'AUTO';
    }

    mapStatus(restStatus) {
        const map = {
            'draft': 'DRAFT',
            'pending': 'PENDING',
            'running': 'RUNNING',
            'completed': 'COMPLETED',
            'failed': 'FAILED',
            'paused': 'PAUSED'
        };
        return map[restStatus?.toLowerCase()] || 'DRAFT';
    }

    mapPriority(restPriority) {
        const map = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'urgent': 'URGENT'
        };
        return map[restPriority?.toLowerCase()] || 'MEDIUM';
    }

    mapFeatureType(restType) {
        const map = {
            'numerical': 'NUMERICAL',
            'categorical': 'CATEGORICAL',
            'text': 'TEXT',
            'datetime': 'DATETIME',
            'boolean': 'BOOLEAN'
        };
        return map[restType?.toLowerCase()] || 'NUMERICAL';
    }

    // API call wrapper with error handling
    async callAgent2API(endpoint, method = 'GET', data = null, params = null) {
        try {
            const config = {
                method,
                url: `${this.apiBase}${endpoint}`,
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            if (data) config.data = data;
            if (params) config.params = params;

            this.log.debug(`Agent 2 API Call: ${method} ${config.url}`, { params, data });
            const response = await axios(config);
            return response.data;
        } catch (error) {
            this.log.error(`Agent 2 API Error (${endpoint}):`, error.message);
            if (error.response) {
                throw new Error(`Agent 2 Backend: ${error.response.status} - ${error.response.data?.error || error.response.statusText}`);
            }
            throw new Error(`Agent 2 Backend Connection Failed: ${error.message}`);
        }
    }

    // High-level API methods
    async getTasks(odataQuery = {}) {
        const params = this.convertODataQuery(odataQuery);
        const restTasks = await this.callAgent2API('/tasks', 'GET', null, params);
        
        // Handle both array and single object responses
        const tasksArray = Array.isArray(restTasks) ? restTasks : [restTasks];
        return tasksArray.map(task => this.convertTaskToOData(task));
    }

    async getTask(taskId) {
        const restTask = await this.callAgent2API(`/tasks/${taskId}`);
        return this.convertTaskToOData(restTask);
    }

    async createTask(odataTask) {
        const restTaskData = this.convertTaskFromOData(odataTask);
        const restTask = await this.callAgent2API('/tasks', 'POST', restTaskData);
        return this.convertTaskToOData(restTask);
    }

    async updateTask(taskId, odataTask) {
        const restTaskData = this.convertTaskFromOData(odataTask);
        const restTask = await this.callAgent2API(`/tasks/${taskId}`, 'PUT', restTaskData);
        return this.convertTaskToOData(restTask);
    }

    async deleteTask(taskId) {
        await this.callAgent2API(`/tasks/${taskId}`, 'DELETE');
        return true;
    }

    async startPreparation(taskId) {
        const result = await this.callAgent2API(`/tasks/${taskId}/prepare`, 'POST');
        return result;
    }

    async analyzeFeatures(taskId) {
        const result = await this.callAgent2API(`/tasks/${taskId}/analyze-features`, 'POST');
        
        // Convert features to OData format
        if (result.features) {
            result.features = result.features.map(feature => 
                this.convertFeatureToOData(feature, taskId)
            );
        }
        
        return result;
    }

    async generateEmbeddings(taskId, config) {
        const restConfig = {
            model: config.model,
            dimensions: config.dimensions,
            normalization: config.normalization,
            batch_size: config.batchSize,
            use_gpu: config.useGPU
        };
        
        return await this.callAgent2API(`/tasks/${taskId}/generate-embeddings`, 'POST', restConfig);
    }

    async exportPreparedData(taskId, config) {
        const restConfig = {
            format: config.format,
            include_metadata: config.includeMetadata,
            split_data: config.splitData,
            compression: config.compression
        };
        
        return await this.callAgent2API(`/tasks/${taskId}/export`, 'POST', restConfig);
    }

    async optimizeHyperparameters(taskId, config) {
        const restConfig = {
            method: config.method,
            trials: config.trials,
            timeout: config.timeout,
            early_stop: config.earlyStop
        };
        
        return await this.callAgent2API(`/tasks/${taskId}/optimize`, 'POST', restConfig);
    }

    async getDataProfile() {
        return await this.callAgent2API('/data-profile');
    }

    async batchPrepare(config) {
        const restConfig = {
            task_ids: config.taskIds,
            parallel: config.parallel,
            gpu_acceleration: config.gpuAcceleration
        };
        
        return await this.callAgent2API('/batch-prepare', 'POST', restConfig);
    }

    async startAutoML(config) {
        const restConfig = {
            dataset: config.dataset,
            problem_type: config.problemType,
            target_column: config.targetColumn,
            evaluation_metric: config.evaluationMetric,
            time_limit: config.timeLimit,
            max_models: config.maxModels,
            include_ensemble: config.includeEnsemble,
            cross_validation: config.crossValidation
        };
        
        return await this.callAgent2API('/automl', 'POST', restConfig);
    }

    async healthCheck() {
        try {
            await this.callAgent2API('/health');
            return { status: 'healthy', backend: 'connected' };
        } catch (error) {
            return { status: 'unhealthy', backend: 'disconnected', error: error.message };
        }
    }
}

module.exports = Agent2Adapter;
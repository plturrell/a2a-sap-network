/**
 * Agent 14 Adapter - Embedding Fine-Tuner Agent
 * Handles communication between service layer and Python backend SDK
 * Implements embedding model fine-tuning, optimization, and deployment functionality
 */

const BaseAdapter = require('./base-adapter');
const { v4: uuidv4 } = require('uuid');

class Agent14Adapter extends BaseAdapter {
    constructor() {
        super();
        this.agentName = 'embedding-fine-tuner-agent';
        this.agentId = 14;
    }

    // ===== EMBEDDING MODEL MANAGEMENT =====
    async getEmbeddingModels(query = {}) {
        try {
            const filters = this._buildFilters(query);
            const response = await this.callPythonBackend('list_embedding_models', { filters });
            return this._transformModelsResponse(response);
        } catch (error) {
            throw new Error(`Failed to get embedding models: ${error.message}`);
        }
    }

    async createEmbeddingModel(modelData) {
        try {
            const payload = {
                model_name: modelData.name,
                description: modelData.description,
                base_model: modelData.baseModel || 'sentence-transformers/all-MiniLM-L6-v2',
                model_type: modelData.modelType || 'sentence_transformer',
                optimization_strategy: modelData.optimizationStrategy || 'contrastive_learning',
                config_overrides: modelData.configOverrides || {}
            };

            const response = await this.callPythonBackend('create_embedding_model', payload);
            return this._transformModelResponse(response);
        } catch (error) {
            throw new Error(`Failed to create embedding model: ${error.message}`);
        }
    }

    // ===== FINE-TUNING OPERATIONS =====
    async startFineTuning(modelId, trainingData, validationData = null, customConfig = null) {
        try {
            const payload = {
                model_id: modelId,
                training_data: trainingData,
                validation_data: validationData,
                custom_config: customConfig
            };

            const response = await this.callPythonBackend('start_fine_tuning', payload);
            return {
                job_id: response.job_id,
                model_id: response.model_id,
                status: response.status,
                estimated_duration_minutes: response.estimated_duration_minutes
            };
        } catch (error) {
            throw new Error(`Failed to start fine-tuning: ${error.message}`);
        }
    }

    async stopFineTuning(jobId) {
        try {
            const response = await this.callPythonBackend('stop_fine_tuning', { 
                job_id: jobId 
            });
            return { status: 'stopped', job_id: jobId };
        } catch (error) {
            throw new Error(`Failed to stop fine-tuning: ${error.message}`);
        }
    }

    // ===== TRAINING MONITORING =====
    async getTrainingStatus(jobId) {
        try {
            const response = await this.callPythonBackend('get_training_status', { 
                job_id: jobId 
            });
            return this._transformStatusResponse(response);
        } catch (error) {
            throw new Error(`Failed to get training status: ${error.message}`);
        }
    }

    async getTrainingMetrics(jobId, metricType = 'all') {
        try {
            const payload = {
                job_id: jobId,
                metric_type: metricType
            };

            const response = await this.callPythonBackend('get_training_metrics', payload);
            return {
                job_id: jobId,
                metrics: response.metrics || {},
                history: response.history || [],
                charts: response.charts || {}
            };
        } catch (error) {
            throw new Error(`Failed to get training metrics: ${error.message}`);
        }
    }

    // ===== MODEL EVALUATION =====
    async evaluateEmbeddingModel(modelId, evaluationDataset, evaluationTasks = null) {
        try {
            const payload = {
                model_id: modelId,
                evaluation_dataset: evaluationDataset,
                evaluation_tasks: evaluationTasks
            };

            const response = await this.callPythonBackend('evaluate_embedding_model', payload);
            return {
                model_id: response.model_id,
                evaluation_results: response.evaluation_results,
                overall_score: response.overall_score,
                evaluated_at: response.evaluated_at
            };
        } catch (error) {
            throw new Error(`Failed to evaluate embedding model: ${error.message}`);
        }
    }

    async compareModels(modelIds, comparisonMetrics = []) {
        try {
            const payload = {
                model_ids: modelIds,
                comparison_metrics: comparisonMetrics
            };

            const response = await this.callPythonBackend('compare_models', payload);
            return {
                comparison_id: response.comparison_id,
                models: response.models,
                comparison_results: response.comparison_results,
                recommendations: response.recommendations
            };
        } catch (error) {
            throw new Error(`Failed to compare models: ${error.message}`);
        }
    }

    // ===== MODEL OPTIMIZATION =====
    async optimizeEmbeddingModel(modelId, optimizationTechniques, optimizationConfig = null) {
        try {
            const payload = {
                model_id: modelId,
                optimization_techniques: optimizationTechniques,
                optimization_config: optimizationConfig
            };

            const response = await this.callPythonBackend('optimize_embedding_model', payload);
            return {
                model_id: response.model_id,
                optimization_results: response.optimization_results,
                optimized_at: response.optimized_at
            };
        } catch (error) {
            throw new Error(`Failed to optimize embedding model: ${error.message}`);
        }
    }

    // ===== MODEL DEPLOYMENT =====
    async deployEmbeddingModel(modelId, deploymentConfig) {
        try {
            const payload = {
                model_id: modelId,
                deployment_config: deploymentConfig
            };

            const response = await this.callPythonBackend('deploy_embedding_model', payload);
            return {
                model_id: response.model_id,
                deployment_id: response.deployment_id,
                endpoint_url: response.endpoint_url,
                deployment_status: response.deployment_status,
                deployed_at: response.deployed_at
            };
        } catch (error) {
            throw new Error(`Failed to deploy embedding model: ${error.message}`);
        }
    }

    async undeployModel(deploymentId) {
        try {
            const response = await this.callPythonBackend('undeploy_model', { 
                deployment_id: deploymentId 
            });
            return { status: 'undeployed', deployment_id: deploymentId };
        } catch (error) {
            throw new Error(`Failed to undeploy model: ${error.message}`);
        }
    }

    // ===== TRAINING DATA MANAGEMENT =====
    async uploadTrainingData(datasetName, dataContent, dataFormat, metadata = {}) {
        try {
            const payload = {
                dataset_name: datasetName,
                data_content: dataContent,
                data_format: dataFormat,
                metadata: metadata
            };

            const response = await this.callPythonBackend('upload_training_data', payload);
            return {
                dataset_id: response.dataset_id,
                dataset_name: datasetName,
                record_count: response.record_count,
                file_size: response.file_size,
                upload_status: 'completed'
            };
        } catch (error) {
            throw new Error(`Failed to upload training data: ${error.message}`);
        }
    }

    async validateTrainingData(datasetId, validationRules = {}) {
        try {
            const payload = {
                dataset_id: datasetId,
                validation_rules: validationRules
            };

            const response = await this.callPythonBackend('validate_training_data', payload);
            return {
                dataset_id: datasetId,
                is_valid: response.is_valid,
                validation_errors: response.validation_errors || [],
                warnings: response.warnings || [],
                statistics: response.statistics || {}
            };
        } catch (error) {
            throw new Error(`Failed to validate training data: ${error.message}`);
        }
    }

    // ===== MODEL ANALYTICS =====
    async getModelAnalytics(modelId, analyticsType = 'performance', timeRange = '30d') {
        try {
            const payload = {
                model_id: modelId,
                analytics_type: analyticsType,
                time_range: timeRange
            };

            const response = await this.callPythonBackend('get_model_analytics', payload);
            return {
                model_id: modelId,
                analytics_type: analyticsType,
                data: response.analytics_data || {},
                charts: response.charts || [],
                insights: response.insights || []
            };
        } catch (error) {
            throw new Error(`Failed to get model analytics: ${error.message}`);
        }
    }

    async generateModelReport(modelId, reportType = 'comprehensive', includeMetrics = true) {
        try {
            const payload = {
                model_id: modelId,
                report_type: reportType,
                include_metrics: includeMetrics
            };

            const response = await this.callPythonBackend('generate_model_report', payload);
            return {
                report_id: response.report_id,
                model_id: modelId,
                report_type: reportType,
                download_url: response.download_url,
                generated_at: new Date().toISOString()
            };
        } catch (error) {
            throw new Error(`Failed to generate model report: ${error.message}`);
        }
    }

    // ===== HYPERPARAMETER TUNING =====
    async startHyperparameterTuning(modelId, tuningConfig, searchStrategy = 'random') {
        try {
            const payload = {
                model_id: modelId,
                tuning_config: tuningConfig,
                search_strategy: searchStrategy
            };

            const response = await this.callPythonBackend('start_hyperparameter_tuning', payload);
            return {
                tuning_job_id: response.tuning_job_id,
                model_id: modelId,
                search_strategy: searchStrategy,
                status: 'started',
                estimated_duration: response.estimated_duration
            };
        } catch (error) {
            throw new Error(`Failed to start hyperparameter tuning: ${error.message}`);
        }
    }

    // ===== TRAINING JOBS =====
    async getTrainingJobs(query = {}) {
        try {
            const filters = this._buildFilters(query);
            const response = await this.callPythonBackend('list_training_jobs', { filters });
            return response.training_jobs || [];
        } catch (error) {
            throw new Error(`Failed to get training jobs: ${error.message}`);
        }
    }

    async deleteTrainingJob(jobId) {
        try {
            await this.callPythonBackend('delete_training_job', { job_id: jobId });
            return { success: true };
        } catch (error) {
            throw new Error(`Failed to delete training job: ${error.message}`);
        }
    }

    // ===== TRANSFORMATION HELPERS =====
    _transformModelsResponse(response) {
        if (!response.models) return [];
        
        return response.models.map(model => ({
            ID: model.id,
            name: model.name,
            description: model.description,
            baseModel: model.base_model,
            modelType: model.model_type,
            status: model.status,
            createdAt: model.created_at,
            updatedAt: model.updated_at,
            metrics: model.metrics || {}
        }));
    }

    _transformModelResponse(response) {
        return {
            ID: response.model_id,
            status: response.status,
            baseModel: response.base_model,
            modelType: response.model_type,
            optimizationStrategy: response.optimization_strategy
        };
    }

    _transformStatusResponse(response) {
        return {
            job_id: response.job_id,
            model_id: response.model_id,
            status: response.status,
            progress: {
                percentage: response.progress?.percentage || 0,
                current_epoch: response.progress?.current_epoch || 0,
                total_epochs: response.progress?.total_epochs || 0,
                current_step: response.progress?.current_step || 0,
                total_steps: response.progress?.total_steps || 0
            },
            metrics: {
                current_loss: response.metrics?.current_loss,
                evaluation_metrics: response.metrics?.evaluation_metrics || {}
            },
            timing: response.timing || {},
            error_message: response.error_message
        };
    }

    _buildFilters(query) {
        const filters = {};
        
        if (query.status) filters.status = query.status;
        if (query.model_type) filters.model_type = query.model_type;
        if (query.created_after) filters.created_after = query.created_after;
        if (query.created_before) filters.created_before = query.created_before;
        
        return filters;
    }

    async callPythonBackend(method, payload) {
        const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');
        const baseUrl = process.env.AGENT14_BASE_URL || 'http://localhost:8014';
        
        try {
            let response;
            
            switch (method) {
                case 'list_embedding_models':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/embedding-models`, {
                        params: { filters: JSON.stringify(payload.filters || {}) }
                    });
                    return response.data;
                    
                case 'create_embedding_model':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/embedding-models`, payload);
                    return response.data;
                    
                case 'start_fine_tuning':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/fine-tuning/start`, payload);
                    return response.data;
                    
                case 'stop_fine_tuning':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/fine-tuning/stop`, null, {
                        params: { job_id: payload.job_id }
                    });
                    return response.data;
                    
                case 'get_training_status':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/fine-tuning/status`, {
                        params: { job_id: payload.job_id }
                    });
                    return response.data;
                    
                case 'get_training_metrics':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/fine-tuning/metrics`, payload);
                    return response.data;
                    
                case 'evaluate_embedding_model':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/embedding-models/evaluate`, payload);
                    return response.data;
                    
                case 'deploy_embedding_model':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/embedding-models/deploy`, payload);
                    return response.data;
                    
                case 'generate_embeddings':
                    response = await blockchainClient.sendMessage(`${baseUrl}/api/v1/embedding-models/generate-embeddings`, {
                        model_id: payload.model_id,
                        texts: payload.texts
                    });
                    return response.data;
                    
                default:
                    throw new Error(`Unknown method: ${method}`);
            }
        } catch (error) {
            console.error(`Agent 14 backend call failed:`, error.message);
            throw error;
        }
    }

    // Removed mock response method - now using real backend
}

module.exports = Agent14Adapter;
/**
 * Agent 14 Service Implementation - Embedding Fine-Tuner Agent
 * Implements business logic for embedding model fine-tuning, training monitoring,
 * model optimization, and deployment management
 */

const cds = require('@sap/cds');

const { LoggerFactory } = require('../../shared/logging/structured-logger');
const logger = LoggerFactory.createLogger('agent14-service');
const { v4: uuidv4 } = require('uuid');
const Agent14Adapter = require('../adapters/agent14-adapter');

class Agent14Service extends cds.ApplicationService {
    async init() {
        const db = await cds.connect.to('db');
        this.adapter = new Agent14Adapter();
        
        // Entity references
        const {
            EmbeddingModels,
            TrainingJobs,
            ModelEvaluations,
            OptimizationResults,
            DeploymentRecords,
            TrainingMetrics
        } = db.entities;

        // ===== EMBEDDING MODEL MANAGEMENT =====
        this.on('READ', 'EmbeddingModels', async (req) => {
            try {
                const models = await this.adapter.getEmbeddingModels(req.query);
                return models;
            } catch (error) {
                req.error(500, `Failed to read embedding models: ${error.message}`);
            }
        });

        this.on('CREATE', 'EmbeddingModels', async (req) => {
            try {
                const model = await this.adapter.createEmbeddingModel(req.data);
                
                // Emit model creation event
                await this.emit('EmbeddingModelCreated', {
                    modelId: model.ID,
                    modelName: model.name,
                    baseModel: model.baseModel,
                    modelType: model.modelType,
                    timestamp: new Date()
                });
                
                return model;
            } catch (error) {
                req.error(500, `Failed to create embedding model: ${error.message}`);
            }
        });

        // ===== FINE-TUNING OPERATIONS =====
        this.on('startFineTuning', async (req) => {
            try {
                const { modelId, trainingData, validationData, customConfig } = req.data;
                const training = await this.adapter.startFineTuning(
                    modelId, 
                    trainingData, 
                    validationData, 
                    customConfig
                );
                
                // Create training job record
                const trainingRecord = await INSERT.into(TrainingJobs).entries({
                    ID: training.job_id,
                    modelId: modelId,
                    status: 'PREPARING',
                    startTime: new Date(),
                    estimatedDurationMinutes: training.estimated_duration_minutes,
                    configuration: JSON.stringify(customConfig || {}),
                    createdAt: new Date(),
                    createdBy: req.user.id
                });
                
                await this.emit('FineTuningStarted', {
                    jobId: training.job_id,
                    modelId: modelId,
                    estimatedDuration: training.estimated_duration_minutes,
                    timestamp: new Date()
                });
                
                return training;
            } catch (error) {
                req.error(500, `Failed to start fine-tuning: ${error.message}`);
            }
        });

        this.on('stopFineTuning', async (req) => {
            try {
                const { jobId } = req.data;
                const result = await this.adapter.stopFineTuning(jobId);
                
                // Update training job status
                await UPDATE(TrainingJobs)
                    .set({ status: 'CANCELLED', endTime: new Date(), updatedAt: new Date() })
                    .where({ ID: jobId });
                
                await this.emit('FineTuningStopped', {
                    jobId,
                    timestamp: new Date()
                });
                
                return result;
            } catch (error) {
                req.error(500, `Failed to stop fine-tuning: ${error.message}`);
            }
        });

        // ===== TRAINING MONITORING =====
        this.on('getTrainingStatus', async (req) => {
            try {
                const { jobId } = req.data;
                const status = await this.adapter.getTrainingStatus(jobId);
                
                // Update training job record with latest status
                if (status.progress) {
                    await UPDATE(TrainingJobs)
                        .set({
                            status: status.status.toUpperCase(),
                            progress: status.progress.percentage,
                            currentEpoch: status.progress.current_epoch,
                            currentStep: status.progress.current_step,
                            totalSteps: status.progress.total_steps,
                            currentLoss: status.metrics?.current_loss,
                            evaluationMetrics: JSON.stringify(status.metrics?.evaluation_metrics || {}),
                            updatedAt: new Date()
                        })
                        .where({ ID: jobId });
                }
                
                return status;
            } catch (error) {
                req.error(500, `Failed to get training status: ${error.message}`);
            }
        });

        this.on('getTrainingMetrics', async (req) => {
            try {
                const { jobId, metricType } = req.data;
                const metrics = await this.adapter.getTrainingMetrics(jobId, metricType);
                return metrics;
            } catch (error) {
                req.error(500, `Failed to get training metrics: ${error.message}`);
            }
        });

        // ===== MODEL EVALUATION =====
        this.on('evaluateModel', async (req) => {
            try {
                const { modelId, evaluationDataset, evaluationTasks } = req.data;
                const evaluation = await this.adapter.evaluateEmbeddingModel(
                    modelId, 
                    evaluationDataset, 
                    evaluationTasks
                );
                
                // Store evaluation results
                const evaluationRecord = await INSERT.into(ModelEvaluations).entries({
                    ID: uuidv4(),
                    modelId: modelId,
                    evaluationResults: JSON.stringify(evaluation.evaluation_results),
                    overallScore: evaluation.overall_score,
                    evaluatedAt: evaluation.evaluated_at,
                    evaluationTasks: JSON.stringify(evaluationTasks || []),
                    createdAt: new Date(),
                    createdBy: req.user.id
                });
                
                await this.emit('ModelEvaluated', {
                    modelId,
                    overallScore: evaluation.overall_score,
                    evaluationId: evaluationRecord.ID,
                    timestamp: new Date()
                });
                
                return evaluation;
            } catch (error) {
                req.error(500, `Failed to evaluate model: ${error.message}`);
            }
        });

        this.on('compareModels', async (req) => {
            try {
                const { modelIds, comparisonMetrics } = req.data;
                const comparison = await this.adapter.compareModels(modelIds, comparisonMetrics);
                return comparison;
            } catch (error) {
                req.error(500, `Failed to compare models: ${error.message}`);
            }
        });

        // ===== MODEL OPTIMIZATION =====
        this.on('optimizeModel', async (req) => {
            try {
                const { modelId, optimizationTechniques, optimizationConfig } = req.data;
                const optimization = await this.adapter.optimizeEmbeddingModel(
                    modelId, 
                    optimizationTechniques, 
                    optimizationConfig
                );
                
                // Store optimization results
                const optimizationRecord = await INSERT.into(OptimizationResults).entries({
                    ID: uuidv4(),
                    modelId: modelId,
                    techniques: JSON.stringify(optimizationTechniques),
                    results: JSON.stringify(optimization.optimization_results),
                    optimizedAt: optimization.optimized_at,
                    createdAt: new Date(),
                    createdBy: req.user.id
                });
                
                await this.emit('ModelOptimized', {
                    modelId,
                    techniques: optimizationTechniques,
                    optimizationId: optimizationRecord.ID,
                    timestamp: new Date()
                });
                
                return optimization;
            } catch (error) {
                req.error(500, `Failed to optimize model: ${error.message}`);
            }
        });

        // ===== MODEL DEPLOYMENT =====
        this.on('deployModel', async (req) => {
            try {
                const { modelId, deploymentConfig } = req.data;
                const deployment = await this.adapter.deployEmbeddingModel(modelId, deploymentConfig);
                
                // Create deployment record
                const deploymentRecord = await INSERT.into(DeploymentRecords).entries({
                    ID: deployment.deployment_id,
                    modelId: modelId,
                    deploymentStatus: deployment.deployment_status,
                    endpointUrl: deployment.endpoint_url,
                    deploymentConfig: JSON.stringify(deploymentConfig),
                    deployedAt: deployment.deployed_at,
                    createdAt: new Date(),
                    createdBy: req.user.id
                });
                
                await this.emit('ModelDeployed', {
                    modelId,
                    deploymentId: deployment.deployment_id,
                    endpointUrl: deployment.endpoint_url,
                    timestamp: new Date()
                });
                
                return deployment;
            } catch (error) {
                req.error(500, `Failed to deploy model: ${error.message}`);
            }
        });

        this.on('undeployModel', async (req) => {
            try {
                const { deploymentId } = req.data;
                const result = await this.adapter.undeployModel(deploymentId);
                
                // Update deployment record
                await UPDATE(DeploymentRecords)
                    .set({ 
                        deploymentStatus: 'UNDEPLOYED',
                        undeployedAt: new Date(),
                        updatedAt: new Date()
                    })
                    .where({ ID: deploymentId });
                
                await this.emit('ModelUndeployed', {
                    deploymentId,
                    timestamp: new Date()
                });
                
                return result;
            } catch (error) {
                req.error(500, `Failed to undeploy model: ${error.message}`);
            }
        });

        // ===== TRAINING DATA MANAGEMENT =====
        this.on('uploadTrainingData', async (req) => {
            try {
                const { datasetName, dataContent, dataFormat, metadata } = req.data;
                const upload = await this.adapter.uploadTrainingData(
                    datasetName, 
                    dataContent, 
                    dataFormat, 
                    metadata
                );
                
                await this.emit('TrainingDataUploaded', {
                    datasetName,
                    datasetId: upload.dataset_id,
                    recordCount: upload.record_count,
                    timestamp: new Date()
                });
                
                return upload;
            } catch (error) {
                req.error(500, `Failed to upload training data: ${error.message}`);
            }
        });

        this.on('validateTrainingData', async (req) => {
            try {
                const { datasetId, validationRules } = req.data;
                const validation = await this.adapter.validateTrainingData(datasetId, validationRules);
                return validation;
            } catch (error) {
                req.error(500, `Failed to validate training data: ${error.message}`);
            }
        });

        // ===== MODEL ANALYTICS =====
        this.on('getModelAnalytics', async (req) => {
            try {
                const { modelId, analyticsType, timeRange } = req.data;
                const analytics = await this.adapter.getModelAnalytics(modelId, analyticsType, timeRange);
                return analytics;
            } catch (error) {
                req.error(500, `Failed to get model analytics: ${error.message}`);
            }
        });

        this.on('generateModelReport', async (req) => {
            try {
                const { modelId, reportType, includeMetrics } = req.data;
                const report = await this.adapter.generateModelReport(modelId, reportType, includeMetrics);
                return report;
            } catch (error) {
                req.error(500, `Failed to generate model report: ${error.message}`);
            }
        });

        // ===== HYPERPARAMETER TUNING =====
        this.on('startHyperparameterTuning', async (req) => {
            try {
                const { modelId, tuningConfig, searchStrategy } = req.data;
                const tuning = await this.adapter.startHyperparameterTuning(
                    modelId, 
                    tuningConfig, 
                    searchStrategy
                );
                
                await this.emit('HyperparameterTuningStarted', {
                    modelId,
                    tuningJobId: tuning.tuning_job_id,
                    searchStrategy,
                    timestamp: new Date()
                });
                
                return tuning;
            } catch (error) {
                req.error(500, `Failed to start hyperparameter tuning: ${error.message}`);
            }
        });

        // ===== TRAINING JOBS CRUD =====
        this.on('READ', 'TrainingJobs', async (req) => {
            try {
                const jobs = await this.adapter.getTrainingJobs(req.query);
                return jobs;
            } catch (error) {
                req.error(500, `Failed to read training jobs: ${error.message}`);
            }
        });

        this.on('DELETE', 'TrainingJobs', async (req) => {
            try {
                const jobId = req.params[0];
                await this.adapter.deleteTrainingJob(jobId);
                
                await this.emit('TrainingJobDeleted', {
                    jobId,
                    timestamp: new Date()
                });
                
            } catch (error) {
                req.error(500, `Failed to delete training job: ${error.message}`);
            }
        });

        // Initialize adapter
        await super.init();
        logger.info('Agent 14 Service (Embedding Fine-Tuner) initialized successfully');
    }

    // ===== HELPER METHODS =====
    async _updateTrainingMetrics(jobId, metrics) {
        try {
            const metricsData = {
                ID: uuidv4(),
                trainingJobId: jobId,
                epoch: metrics.epoch,
                step: metrics.step,
                loss: metrics.loss,
                learningRate: metrics.learning_rate,
                evaluationMetrics: JSON.stringify(metrics.eval_metrics || {}),
                timestamp: new Date(),
                createdAt: new Date()
            };
            
            await INSERT.into('TrainingMetrics').entries(metricsData);
        } catch (error) {
            logger.error('Failed to update training metrics:', { error: error });
        }
    }

    async _checkModelReadiness(modelId) {
        try {
            const model = await SELECT.one.from('EmbeddingModels').where({ ID: modelId });
            return model && model.status === 'COMPLETED';
        } catch (error) {
            logger.error('Failed to check model readiness:', { error: error });
            return false;
        }
    }
}

module.exports = Agent14Service;
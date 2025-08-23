/**
 * Agent 10 Adapter - Calculation Engine
 * Converts between REST API and OData formats for calculation tasks, statistical analysis,
 * formula evaluation, self-healing calculations, and error correction operations
 */

const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

class Agent10Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT10_BASE_URL || 'http://localhost:8010';
        this.apiVersion = 'v1';
        this.timeout = 60000; // Longer timeout for complex calculations
    }

    // ===== CALCULATION TASKS =====
    async getCalculationTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'CalculationTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCalculationTask(data) {
        try {
            const restData = this._convertODataCalculationTaskToREST(data);
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTCalculationTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateCalculationTask(id, data) {
        try {
            const restData = this._convertODataCalculationTaskToREST(data);
            const response = await axios.put(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTCalculationTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteCalculationTask(id) {
        try {
            await axios.delete(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CALCULATION OPERATIONS =====
    async startCalculation(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/start`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                taskName: response.data.task_name,
                calculationType: response.data.calculation_type?.toUpperCase(),
                calculationMethod: response.data.calculation_method?.toUpperCase(),
                sessionId: response.data.session_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async pauseCalculation(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/pause`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                pausedAt: response.data.paused_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resumeCalculation(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/resume`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                resumedAt: response.data.resumed_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async cancelCalculation(taskId) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/cancel`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                cancelledAt: response.data.cancelled_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateFormula(formula) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/validate-formula`, {
                formula
            }, {
                timeout: this.timeout
            });
            
            return {
                isValid: response.data.is_valid,
                errors: response.data.errors,
                warnings: response.data.warnings,
                variables: response.data.variables,
                complexity: response.data.complexity
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async previewCalculation(formula, sampleData) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/preview-calculation`, {
                formula,
                sample_data: sampleData
            }, {
                timeout: this.timeout
            });
            
            return {
                preview: response.data.preview,
                steps: response.data.steps,
                expectedResult: response.data.expected_result,
                warnings: response.data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportResults(taskId, options) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/export-results/${taskId}`, {
                format: options.format,
                include_steps: options.includeSteps,
                include_statistics: options.includeStatistics
            }, {
                timeout: this.timeout
            });
            
            return {
                downloadUrl: response.data.download_url,
                fileName: response.data.file_name,
                format: response.data.format,
                size: response.data.size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== MAIN CALCULATION METHODS =====
    async performCalculation(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/calculate`, {
                formula: data.formula,
                input_data: data.inputData,
                calculation_type: data.calculationType?.toLowerCase(),
                method: data.method?.toLowerCase(),
                precision: data.precision?.toLowerCase(),
                enable_self_healing: data.enableSelfHealing
            }, {
                timeout: this.timeout
            });
            
            return {
                result: response.data.result,
                executionTime: response.data.execution_time,
                accuracy: response.data.accuracy,
                steps: response.data.steps,
                corrections: response.data.corrections,
                selfHealingLog: response.data.self_healing_log,
                performanceMetrics: response.data.performance_metrics,
                error: response.data.error
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async performStatisticalAnalysis(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/statistical-analysis`, {
                data: data.data,
                analysis_type: data.analysisType?.toLowerCase(),
                confidence_level: data.confidenceLevel,
                options: data.options
            }, {
                timeout: this.timeout
            });
            
            return {
                datasetName: response.data.dataset_name,
                sampleSize: response.data.sample_size,
                statistics: {
                    mean: response.data.mean,
                    median: response.data.median,
                    mode: response.data.mode,
                    standardDeviation: response.data.standard_deviation,
                    variance: response.data.variance,
                    min: response.data.min_value,
                    max: response.data.max_value,
                    confidenceInterval: response.data.confidence_interval,
                    pValue: response.data.p_value,
                    correlation: response.data.correlation_coefficient,
                    rSquared: response.data.r_squared
                },
                additionalMetrics: response.data.additional_metrics,
                visualizationData: response.data.visualization_data,
                interpretation: response.data.interpretation,
                executionTime: response.data.execution_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async batchCalculate(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/batch-calculate`, {
                calculations: data.calculations,
                parallel: data.parallel,
                priority: data.priority?.toLowerCase()
            }, {
                timeout: this.timeout * 2 // Double timeout for batch operations
            });
            
            return {
                batchId: response.data.batch_id,
                totalCalculations: response.data.total_calculations,
                completed: response.data.completed,
                failed: response.data.failed,
                results: response.data.results,
                executionTime: response.data.execution_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async evaluateCustomFormula(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/evaluate-formula`, {
                formula: data.formula,
                variables: data.variables,
                verify: data.verify
            }, {
                timeout: this.timeout
            });
            
            return {
                result: response.data.result,
                verification: response.data.verification,
                executionTime: response.data.execution_time,
                warnings: response.data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CONFIGURATION METHODS =====
    async getCalculationMethods() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/calculation-methods`, {
                timeout: this.timeout
            });
            
            return response.data.methods;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getStatisticalMethods() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/statistical-methods`, {
                timeout: this.timeout
            });
            
            return response.data.methods;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getSelfHealingStrategies() {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/self-healing-strategies`, {
                timeout: this.timeout
            });
            
            return response.data.strategies;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async configurePrecision(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/configure-precision`, {
                type: data.type?.toLowerCase(),
                accuracy: data.accuracy
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                currentPrecision: response.data.current_precision,
                accuracy: response.data.accuracy
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async configureParallelProcessing(data) {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/configure-parallel-processing`, {
                max_threads: data.maxThreads,
                chunk_size: data.chunkSize
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                maxThreads: response.data.max_threads,
                chunkSize: response.data.chunk_size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== HISTORY AND METRICS =====
    async getCalculationHistory(options) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/calculation-history`, {
                params: {
                    limit: options.limit,
                    offset: options.offset,
                    filter: JSON.stringify(options.filter)
                },
                timeout: this.timeout
            });
            
            return {
                total: response.data.total,
                items: response.data.items,
                page: response.data.page,
                pageSize: response.data.page_size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getPerformanceMetrics(taskId) {
        try {
            const response = await axios.get(`${this.baseUrl}/api/${this.apiVersion}/performance-metrics/${taskId}`, {
                timeout: this.timeout
            });
            
            return response.data.metrics;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async clearCache() {
        try {
            const response = await axios.post(`${this.baseUrl}/api/${this.apiVersion}/clear-cache`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                clearedItems: response.data.cleared_items,
                freedSpace: response.data.freed_space
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CONVERSION UTILITIES =====
    _convertODataToREST(query) {
        const params = {};
        
        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$orderby) params.sort = query.$orderby;
        if (query.$search) params.search = query.$search;
        
        return params;
    }

    _parseODataFilter(filter) {
        // Simple OData filter parser - can be enhanced
        return filter
            .replace(/eq/g, '=')
            .replace(/ne/g, '!=')
            .replace(/gt/g, '>')
            .replace(/lt/g, '<')
            .replace(/ge/g, '>=')
            .replace(/le/g, '<=')
            .replace(/and/g, '&&')
            .replace(/or/g, '||');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        if (entityType === 'CalculationTask') {
            return this._convertRESTCalculationTaskToOData(item);
        }
        return item;
    }

    _convertODataCalculationTaskToREST(data) {
        return {
            task_name: data.taskName,
            description: data.description,
            calculation_type: data.calculationType?.toLowerCase(),
            formula: data.formula,
            input_parameters: data.inputParameters ? JSON.parse(data.inputParameters) : {},
            calculation_method: data.calculationMethod?.toLowerCase(),
            precision_type: data.precisionType?.toLowerCase(),
            required_accuracy: data.requiredAccuracy,
            max_iterations: data.maxIterations,
            timeout: data.timeout,
            enable_self_healing: data.enableSelfHealing,
            verification_rounds: data.verificationRounds,
            use_parallel_processing: data.useParallelProcessing,
            cache_results: data.cacheResults,
            priority: data.priority?.toLowerCase(),
            metadata: data.metadata ? JSON.parse(data.metadata) : {}
        };
    }

    _convertRESTCalculationTaskToOData(item) {
        return {
            ID: item.id || uuidv4(),
            taskName: item.task_name,
            description: item.description,
            calculationType: item.calculation_type?.toUpperCase(),
            formula: item.formula,
            inputParameters: JSON.stringify(item.input_parameters || {}),
            calculationMethod: item.calculation_method?.toUpperCase(),
            precisionType: item.precision_type?.toUpperCase() || 'DECIMAL64',
            requiredAccuracy: item.required_accuracy || 0.000001,
            maxIterations: item.max_iterations || 1000,
            timeout: item.timeout || 60000,
            enableSelfHealing: item.enable_self_healing !== false,
            verificationRounds: item.verification_rounds || 3,
            useParallelProcessing: item.use_parallel_processing !== false,
            cacheResults: item.cache_results !== false,
            priority: item.priority?.toUpperCase() || 'MEDIUM',
            status: item.status?.toUpperCase() || 'PENDING',
            progress: item.progress || 0,
            startTime: item.start_time,
            endTime: item.end_time,
            executionTime: item.execution_time,
            result: item.result ? JSON.stringify(item.result) : null,
            errorMessage: item.error_message,
            selfHealingLog: item.self_healing_log ? JSON.stringify(item.self_healing_log) : null,
            performanceMetrics: item.performance_metrics ? JSON.stringify(item.performance_metrics) : null,
            metadata: item.metadata ? JSON.stringify(item.metadata) : null,
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.message || error.response.statusText;
            const details = error.response.data?.details || null;
            
            return new Error(`Agent 10 Error (${status}): ${message}${details ? ` - ${JSON.stringify(details)}` : ''}`);
        } else if (error.request) {
            return new Error('Agent 10 Connection Error: No response from calculation service');
        } else {
            return new Error(`Agent 10 Error: ${error.message}`);
        }
    }
}

module.exports = Agent10Adapter;
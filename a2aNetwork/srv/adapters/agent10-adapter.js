/**
 * Agent 10 Adapter - Calculation Engine
 * Converts between REST API and OData formats for calculation tasks, statistical analysis,
 * formula evaluation, self-healing calculations, and error correction operations
 */

const fetch = require('node-fetch');
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
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CalculationTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCalculationTask(data) {
        try {
            const restData = this._convertODataCalculationTaskToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTCalculationTaskToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateCalculationTask(id, data) {
        try {
            const restData = this._convertODataCalculationTaskToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTCalculationTaskToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteCalculationTask(id) {
        try {
            await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CALCULATION OPERATIONS =====
    async startCalculation(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                taskName: data.task_name,
                calculationType: data.calculation_type?.toUpperCase(),
                calculationMethod: data.calculation_method?.toUpperCase(),
                sessionId: data.session_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async pauseCalculation(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/pause`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message,
                pausedAt: data.paused_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resumeCalculation(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/resume`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message,
                resumedAt: data.resumed_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async cancelCalculation(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-tasks/${taskId}/cancel`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message,
                cancelledAt: data.cancelled_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateFormula(formula) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/validate-formula`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                formula
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                isValid: data.is_valid,
                errors: data.errors,
                warnings: data.warnings,
                variables: data.variables,
                complexity: data.complexity
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async previewCalculation(formula, sampleData) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/preview-calculation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                formula,
                sample_data: sampleData
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                preview: data.preview,
                steps: data.steps,
                expectedResult: data.expected_result,
                warnings: data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportResults(taskId, options) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/export-results/${taskId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                format: options.format,
                include_steps: options.includeSteps,
                include_statistics: options.includeStatistics
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                downloadUrl: data.download_url,
                fileName: data.file_name,
                format: data.format,
                size: data.size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== MAIN CALCULATION METHODS =====
    async performCalculation(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                formula: data.formula,
                input_data: data.inputData,
                calculation_type: data.calculationType?.toLowerCase(),
                method: data.method?.toLowerCase(),
                precision: data.precision?.toLowerCase(),
                enable_self_healing: data.enableSelfHealing
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                result: data.result,
                executionTime: data.execution_time,
                accuracy: data.accuracy,
                steps: data.steps,
                corrections: data.corrections,
                selfHealingLog: data.self_healing_log,
                performanceMetrics: data.performance_metrics,
                error: data.error
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async performStatisticalAnalysis(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/statistical-analysis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                data: data.data,
                analysis_type: data.analysisType?.toLowerCase(),
                confidence_level: data.confidenceLevel,
                options: data.options
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                datasetName: data.dataset_name,
                sampleSize: data.sample_size,
                statistics: {
                    mean: data.mean,
                    median: data.median,
                    mode: data.mode,
                    standardDeviation: data.standard_deviation,
                    variance: data.variance,
                    min: data.min_value,
                    max: data.max_value,
                    confidenceInterval: data.confidence_interval,
                    pValue: data.p_value,
                    correlation: data.correlation_coefficient,
                    rSquared: data.r_squared
                },
                additionalMetrics: data.additional_metrics,
                visualizationData: data.visualization_data,
                interpretation: data.interpretation,
                executionTime: data.execution_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async batchCalculate(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/batch-calculate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                calculations: data.calculations,
                parallel: data.parallel,
                priority: data.priority?.toLowerCase()
            }, {
                timeout: this.timeout * 2 // Double timeout for batch operations
            })
            });
            const data = await response.json();
            return {
                batchId: data.batch_id,
                totalCalculations: data.total_calculations,
                completed: data.completed,
                failed: data.failed,
                results: data.results,
                executionTime: data.execution_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async evaluateCustomFormula(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/evaluate-formula`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                formula: data.formula,
                variables: data.variables,
                verify: data.verify
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                result: data.result,
                verification: data.verification,
                executionTime: data.execution_time,
                warnings: data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CONFIGURATION METHODS =====
    async getCalculationMethods() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-methods`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data.methods;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getStatisticalMethods() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/statistical-methods`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data.methods;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getSelfHealingStrategies() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/self-healing-strategies`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data.strategies;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async configurePrecision(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/configure-precision`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                type: data.type?.toLowerCase(),
                accuracy: data.accuracy
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success,
                currentPrecision: data.current_precision,
                accuracy: data.accuracy
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async configureParallelProcessing(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/configure-parallel-processing`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                max_threads: data.maxThreads,
                chunk_size: data.chunkSize
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success,
                maxThreads: data.max_threads,
                chunkSize: data.chunk_size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== HISTORY AND METRICS =====
    async getCalculationHistory(options) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/calculation-history`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                params: {
                    limit: options.limit,
                    offset: options.offset,
                    filter: JSON.stringify(options.filter)
                },
                timeout: this.timeout
            })
            });
            const data = await response.json();
            return {
                total: data.total,
                items: data.items,
                page: data.page,
                pageSize: data.page_size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getPerformanceMetrics(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/performance-metrics/${taskId}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data.metrics;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async clearCache() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/clear-cache`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                clearedItems: data.cleared_items,
                freedSpace: data.freed_space
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
            const message = error.data?.message || error.response.statusText;
            const details = error.data?.details || null;

            return new Error(`Agent 10 Error (${status}): ${message}${details ? ` - ${JSON.stringify(details)}` : ''}`);
        } else if (error.request) {
            return new Error('Agent 10 Connection Error: No response from calculation service');
        } else {
            return new Error(`Agent 10 Error: ${error.message}`);
        }
    }
}

module.exports = Agent10Adapter;
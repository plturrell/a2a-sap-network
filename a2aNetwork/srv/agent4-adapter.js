/**
 * Agent 4 Calculation Validation Adapter
 * Converts between REST API format (Python backend) and OData format (SAP CAP)
 */

const cds = require('@sap/cds');
const { BlockchainClient } = require('../core/blockchain-client');

const log = cds.log('agent4-adapter');

class Agent4Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT4_BASE_URL || 'http://localhost:8003';
        log.info(`Agent 4 Adapter initialized with base URL: ${this.baseUrl}`);
    }

    // =================================
    // Data Conversion Methods
    // =================================

    /**
     * Convert REST calculation validation task to OData format
     */
    convertTaskToOData(restTask) {
        return {
            ID: restTask.id || cds.utils.uuid(),
            taskName: restTask.task_name,
            description: restTask.description,
            expression: restTask.expression,
            inputVariables: JSON.stringify(restTask.input_variables || {}),
            expectedResult: restTask.expected_result,
            validationMethod: this.mapValidationMethod(restTask.validation_method),
            precisionLevel: this.mapPrecisionLevel(restTask.precision_level),
            tolerance: restTask.tolerance || 0.0001,
            useSymbolicMath: restTask.use_symbolic_math || false,
            useNumericalMethods: restTask.use_numerical_methods !== false,
            useStatisticalAnalysis: restTask.use_statistical_analysis || false,
            useAIValidation: restTask.use_ai_validation || false,
            useBlockchainConsensus: restTask.use_blockchain_consensus || false,
            aiModel: this.mapAIModel(restTask.ai_model),
            consensusValidators: restTask.consensus_validators || 3,
            consensusThreshold: restTask.consensus_threshold || 0.67,
            status: this.mapStatus(restTask.status),
            priority: this.mapPriority(restTask.priority),
            progressPercent: restTask.progress || 0,
            currentStage: restTask.current_stage,
            validationTime: restTask.validation_time,
            computedResult: restTask.computed_result,
            validationScore: restTask.validation_score,
            confidenceLevel: restTask.confidence_level,
            errorDetails: restTask.error_details,
            methodsUsed: JSON.stringify(restTask.methods_used || []),
            intermediateSteps: JSON.stringify(restTask.intermediate_steps || []),
            startedAt: restTask.started_at,
            completedAt: restTask.completed_at,
            createdAt: restTask.created_at || new Date().toISOString(),
            modifiedAt: restTask.modified_at || new Date().toISOString(),
            // Store the backend task ID for future reference
            agent4TaskId: restTask.id
        };
    }

    /**
     * Convert OData calculation validation task to REST format
     */
    convertTaskToRest(odataTask) {
        return {
            task_name: odataTask.taskName,
            description: odataTask.description,
            expression: odataTask.expression,
            input_variables: JSON.parse(odataTask.inputVariables || '{}'),
            expected_result: odataTask.expectedResult,
            validation_method: this.unmapValidationMethod(odataTask.validationMethod),
            precision_level: this.unmapPrecisionLevel(odataTask.precisionLevel),
            tolerance: odataTask.tolerance || 0.0001,
            use_symbolic_math: odataTask.useSymbolicMath || false,
            use_numerical_methods: odataTask.useNumericalMethods !== false,
            use_statistical_analysis: odataTask.useStatisticalAnalysis || false,
            use_ai_validation: odataTask.useAIValidation || false,
            use_blockchain_consensus: odataTask.useBlockchainConsensus || false,
            ai_model: this.unmapAIModel(odataTask.aiModel),
            consensus_validators: odataTask.consensusValidators || 3,
            consensus_threshold: odataTask.consensusThreshold || 0.67,
            priority: this.unmapPriority(odataTask.priority)
        };
    }

    /**
     * Convert REST validation result to OData format
     */
    convertResultToOData(restResult) {
        return {
            ID: restResult.id || cds.utils.uuid(),
            method: restResult.method,
            result: restResult.result,
            isCorrect: restResult.is_correct,
            confidenceScore: restResult.confidence_score,
            processingTime: restResult.processing_time,
            errorMessage: restResult.error_message,
            details: JSON.stringify(restResult.details || {}),
            validatedAt: restResult.validated_at || new Date().toISOString(),
            agent4ResultId: restResult.id
        };
    }

    /**
     * Convert REST validation template to OData format
     */
    convertTemplateToOData(restTemplate) {
        return {
            ID: restTemplate.id || cds.utils.uuid(),
            templateName: restTemplate.template_name,
            description: restTemplate.description,
            category: this.mapTemplateCategory(restTemplate.category),
            expressionTemplate: restTemplate.expression_template,
            variableDefinitions: JSON.stringify(restTemplate.variable_definitions || {}),
            defaultValidationMethod: restTemplate.default_validation_method,
            recommendedPrecision: restTemplate.recommended_precision,
            exampleUsage: restTemplate.example_usage,
            isActive: restTemplate.is_active !== false,
            usageCount: restTemplate.usage_count || 0,
            successRate: restTemplate.success_rate,
            createdAt: restTemplate.created_at || new Date().toISOString(),
            modifiedAt: restTemplate.modified_at || new Date().toISOString(),
            agent4TemplateId: restTemplate.id
        };
    }

    /**
     * Convert OData validation template to REST format
     */
    convertTemplateToRest(odataTemplate) {
        return {
            template_name: odataTemplate.templateName,
            description: odataTemplate.description,
            category: this.unmapTemplateCategory(odataTemplate.category),
            expression_template: odataTemplate.expressionTemplate,
            variable_definitions: JSON.parse(odataTemplate.variableDefinitions || '{}'),
            default_validation_method: odataTemplate.defaultValidationMethod,
            recommended_precision: odataTemplate.recommendedPrecision,
            example_usage: odataTemplate.exampleUsage
        };
    }

    // =================================
    // Field Mapping Methods
    // =================================

    mapValidationMethod(restMethod) {
        const methodMap = {
            'symbolic': 'SYMBOLIC',
            'numerical': 'NUMERICAL',
            'statistical': 'STATISTICAL',
            'ai_powered': 'AI_POWERED',
            'blockchain_consensus': 'BLOCKCHAIN_CONSENSUS',
            'hybrid': 'HYBRID'
        };
        return methodMap[restMethod] || restMethod?.toUpperCase() || 'NUMERICAL';
    }

    unmapValidationMethod(odataMethod) {
        const methodMap = {
            'SYMBOLIC': 'symbolic',
            'NUMERICAL': 'numerical',
            'STATISTICAL': 'statistical',
            'AI_POWERED': 'ai_powered',
            'BLOCKCHAIN_CONSENSUS': 'blockchain_consensus',
            'HYBRID': 'hybrid'
        };
        return methodMap[odataMethod] || odataMethod?.toLowerCase() || 'numerical';
    }

    mapPrecisionLevel(restLevel) {
        const levelMap = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'ultra_high': 'ULTRA_HIGH'
        };
        return levelMap[restLevel] || restLevel?.toUpperCase() || 'MEDIUM';
    }

    unmapPrecisionLevel(odataLevel) {
        const levelMap = {
            'LOW': 'low',
            'MEDIUM': 'medium',
            'HIGH': 'high',
            'ULTRA_HIGH': 'ultra_high'
        };
        return levelMap[odataLevel] || odataLevel?.toLowerCase() || 'medium';
    }

    mapAIModel(restModel) {
        const modelMap = {
            'grok': 'GROK',
            'gpt4': 'GPT4',
            'claude': 'CLAUDE',
            'gemini': 'GEMINI',
            'custom': 'CUSTOM'
        };
        return modelMap[restModel] || restModel?.toUpperCase() || 'GROK';
    }

    unmapAIModel(odataModel) {
        const modelMap = {
            'GROK': 'grok',
            'GPT4': 'gpt4',
            'CLAUDE': 'claude',
            'GEMINI': 'gemini',
            'CUSTOM': 'custom'
        };
        return modelMap[odataModel] || odataModel?.toLowerCase() || 'grok';
    }

    mapTemplateCategory(restCategory) {
        const categoryMap = {
            'arithmetic': 'ARITHMETIC',
            'algebra': 'ALGEBRA',
            'calculus': 'CALCULUS',
            'statistics': 'STATISTICS',
            'geometry': 'GEOMETRY',
            'trigonometry': 'TRIGONOMETRY',
            'linear_algebra': 'LINEAR_ALGEBRA',
            'differential_equations': 'DIFFERENTIAL_EQUATIONS',
            'financial': 'FINANCIAL',
            'physics': 'PHYSICS',
            'chemistry': 'CHEMISTRY'
        };
        return categoryMap[restCategory] || restCategory?.toUpperCase() || 'ARITHMETIC';
    }

    unmapTemplateCategory(odataCategory) {
        const categoryMap = {
            'ARITHMETIC': 'arithmetic',
            'ALGEBRA': 'algebra',
            'CALCULUS': 'calculus',
            'STATISTICS': 'statistics',
            'GEOMETRY': 'geometry',
            'TRIGONOMETRY': 'trigonometry',
            'LINEAR_ALGEBRA': 'linear_algebra',
            'DIFFERENTIAL_EQUATIONS': 'differential_equations',
            'FINANCIAL': 'financial',
            'PHYSICS': 'physics',
            'CHEMISTRY': 'chemistry'
        };
        return categoryMap[odataCategory] || odataCategory?.toLowerCase() || 'arithmetic';
    }

    mapStatus(restStatus) {
        const statusMap = {
            'draft': 'DRAFT',
            'pending': 'PENDING',
            'validating': 'VALIDATING',
            'completed': 'COMPLETED',
            'failed': 'FAILED',
            'cancelled': 'CANCELLED'
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

    async startValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/validate`, {}, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Calculation validation started successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start validation:', error.message);
            throw new Error(`Failed to start validation: ${error.message}`);
        }
    }

    async pauseValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/pause`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Calculation validation paused successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to pause validation:', error.message);
            throw new Error(`Failed to pause validation: ${error.message}`);
        }
    }

    async resumeValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/resume`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Calculation validation resumed successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to resume validation:', error.message);
            throw new Error(`Failed to resume validation: ${error.message}`);
        }
    }

    async cancelValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/cancel`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Calculation validation cancelled successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to cancel validation:', error.message);
            throw new Error(`Failed to cancel validation: ${error.message}`);
        }
    }

    async runSymbolicValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/symbolic`, {}, {
                timeout: 120000
            });
            return {
                success: true,
                message: 'Symbolic validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run symbolic validation:', error.message);
            throw new Error(`Failed to run symbolic validation: ${error.message}`);
        }
    }

    async runNumericalValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/numerical`, {}, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Numerical validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run numerical validation:', error.message);
            throw new Error(`Failed to run numerical validation: ${error.message}`);
        }
    }

    async runStatisticalValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/statistical`, {}, {
                timeout: 90000
            });
            return {
                success: true,
                message: 'Statistical validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run statistical validation:', error.message);
            throw new Error(`Failed to run statistical validation: ${error.message}`);
        }
    }

    async runAIValidation(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/ai-validation`, {}, {
                timeout: 180000 // AI validation can take longer
            });
            return {
                success: true,
                message: 'AI validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run AI validation:', error.message);
            throw new Error(`Failed to run AI validation: ${error.message}`);
        }
    }

    async runBlockchainConsensus(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/blockchain-consensus`, {}, {
                timeout: 300000 // Blockchain consensus can take much longer
            });
            return {
                success: true,
                message: 'Blockchain consensus validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to run blockchain consensus:', error.message);
            throw new Error(`Failed to run blockchain consensus: ${error.message}`);
        }
    }

    async exportValidationReport(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/tasks/${taskId}/export`, {
                format: options.format,
                include_steps: options.includeSteps,
                include_confidence: options.includeConfidence
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Validation report exported successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to export validation report:', error.message);
            throw new Error(`Failed to export validation report: ${error.message}`);
        }
    }

    async validateFromTemplate(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/templates/${options.templateId}/apply`, {
                task_id: taskId,
                variables: JSON.parse(options.variables || '{}')
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Template validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to validate from template:', error.message);
            throw new Error(`Failed to validate from template: ${error.message}`);
        }
    }

    async batchValidateCalculations(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/batch/validate`, {
                task_ids: options.taskIds,
                validation_method: options.validationMethod,
                parallel: options.parallel,
                priority: options.priority
            }, {
                timeout: 120000
            });
            return {
                success: true,
                message: 'Batch validation started',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start batch validation:', error.message);
            throw new Error(`Failed to start batch validation: ${error.message}`);
        }
    }

    async validateExpression(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/expression/evaluate`, {
                expression: options.expression,
                variables: JSON.parse(options.variables || '{}'),
                method: options.method,
                precision: options.precision
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Expression validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to validate expression:', error.message);
            throw new Error(`Failed to validate expression: ${error.message}`);
        }
    }

    async getValidationMethods() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/methods`, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Validation methods retrieved',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get validation methods:', error.message);
            throw new Error(`Failed to get validation methods: ${error.message}`);
        }
    }

    async getCalculationTemplates() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/templates`, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Calculation templates retrieved',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get calculation templates:', error.message);
            throw new Error(`Failed to get calculation templates: ${error.message}`);
        }
    }

    async createTemplate(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/templates`, {
                name: options.name,
                category: options.category,
                expression: options.expression,
                variables: JSON.parse(options.variables || '{}'),
                default_method: options.defaultMethod
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Template created successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to create template:', error.message);
            throw new Error(`Failed to create template: ${error.message}`);
        }
    }

    async benchmarkMethods(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/methods/benchmark`, {
                expression: options.expression,
                variables: JSON.parse(options.variables || '{}'),
                iterations: options.iterations
            }, {
                timeout: 180000
            });
            return {
                success: true,
                message: 'Method benchmarking completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to benchmark methods:', error.message);
            throw new Error(`Failed to benchmark methods: ${error.message}`);
        }
    }

    async configureAIModel(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/ai/configure`, {
                model: options.model,
                parameters: JSON.parse(options.parameters || '{}')
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'AI model configured successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to configure AI model:', error.message);
            throw new Error(`Failed to configure AI model: ${error.message}`);
        }
    }

    async configureBlockchainConsensus(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/blockchain/configure`, {
                validators: options.validators,
                threshold: options.threshold,
                timeout: options.timeout
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Blockchain consensus configured successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to configure blockchain consensus:', error.message);
            throw new Error(`Failed to configure blockchain consensus: ${error.message}`);
        }
    }

    // =================================
    // Health Check
    // =================================

    async checkHealth() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent4/v1/health`, {
                timeout: 5000
            });
            return {
                status: 'healthy',
                data: response.data
            };
        } catch (error) {
            log.warn('Agent 4 backend health check failed:', error.message);
            return {
                status: 'unhealthy',
                error: error.message
            };
        }
    }
}

module.exports = Agent4Adapter;
/**
 * Agent 1 Data Standardization Adapter
 * Converts between REST API format (Python backend) and OData format (SAP CAP)
 */

const cds = require('@sap/cds');
const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');

const log = cds.log('agent1-adapter');

class Agent1Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT1_BASE_URL || 'http://localhost:8001';
        log.info(`Agent 1 Adapter initialized with base URL: ${this.baseUrl}`);
    }

    // =================================
    // Data Conversion Methods
    // =================================

    /**
     * Convert REST standardization task to OData format
     */
    convertTaskToOData(restTask) {
        return {
            ID: restTask.id || cds.utils.uuid(),
            taskName: restTask.task_name,
            description: restTask.description,
            sourceFormat: this.mapSourceFormat(restTask.source_format),
            targetFormat: this.mapTargetFormat(restTask.target_format),
            schemaTemplateId: restTask.schema_template_id,
            schemaValidation: restTask.schema_validation !== false,
            dataTypeValidation: restTask.data_type_validation !== false,
            formatValidation: restTask.format_validation !== false,
            processingMode: restTask.processing_mode?.toUpperCase() || 'FULL',
            batchSize: restTask.batch_size || 1000,
            status: this.mapStatus(restTask.status),
            priority: this.mapPriority(restTask.priority),
            progressPercent: restTask.progress || 0,
            currentStage: restTask.current_stage,
            processingTime: restTask.processing_time,
            recordsProcessed: restTask.records_processed || 0,
            recordsTotal: restTask.records_total || 0,
            errorCount: restTask.error_count || 0,
            validationResults: JSON.stringify(restTask.validation_results || {}),
            errorDetails: restTask.error_details,
            startedAt: restTask.started_at,
            completedAt: restTask.completed_at,
            createdAt: restTask.created_at || new Date().toISOString(),
            modifiedAt: restTask.modified_at || new Date().toISOString(),
            // Store the backend task ID for future reference
            agent1TaskId: restTask.id
        };
    }

    /**
     * Convert OData standardization task to REST format
     */
    convertTaskToRest(odataTask) {
        return {
            task_name: odataTask.taskName,
            description: odataTask.description,
            source_format: this.unmapSourceFormat(odataTask.sourceFormat),
            target_format: this.unmapTargetFormat(odataTask.targetFormat),
            schema_template_id: odataTask.schemaTemplateId,
            schema_validation: odataTask.schemaValidation,
            data_type_validation: odataTask.dataTypeValidation,
            format_validation: odataTask.formatValidation,
            processing_mode: odataTask.processingMode?.toLowerCase() || 'full',
            batch_size: odataTask.batchSize || 1000,
            priority: this.unmapPriority(odataTask.priority)
        };
    }

    /**
     * Convert REST standardization rule to OData format
     */
    convertRuleToOData(restRule) {
        return {
            ID: restRule.id || cds.utils.uuid(),
            name: restRule.rule_name,
            type: this.mapRuleType(restRule.rule_type),
            sourceField: restRule.source_field,
            targetField: restRule.target_field,
            transformation: restRule.transformation,
            isActive: restRule.is_active !== false,
            executionOrder: restRule.execution_order || 0,
            agent1RuleId: restRule.id
        };
    }

    /**
     * Convert OData standardization rule to REST format
     */
    convertRuleToRest(odataRule) {
        return {
            rule_name: odataRule.name,
            rule_type: this.unmapRuleType(odataRule.type),
            source_field: odataRule.sourceField,
            target_field: odataRule.targetField,
            transformation: odataRule.transformation,
            is_active: odataRule.isActive,
            execution_order: odataRule.executionOrder || 0
        };
    }

    // =================================
    // Field Mapping Methods
    // =================================

    mapSourceFormat(restFormat) {
        const formatMap = {
            'csv': 'CSV',
            'json': 'JSON',
            'xml': 'XML',
            'excel': 'EXCEL',
            'fixed_width': 'FIXED_WIDTH',
            'avro': 'AVRO',
            'parquet': 'PARQUET'
        };
        return formatMap[restFormat] || restFormat?.toUpperCase() || 'CSV';
    }

    unmapSourceFormat(odataFormat) {
        const formatMap = {
            'CSV': 'csv',
            'JSON': 'json',
            'XML': 'xml',
            'EXCEL': 'excel',
            'FIXED_WIDTH': 'fixed_width',
            'AVRO': 'avro',
            'PARQUET': 'parquet'
        };
        return formatMap[odataFormat] || odataFormat?.toLowerCase() || 'csv';
    }

    mapTargetFormat(restFormat) {
        const formatMap = {
            'csv': 'CSV',
            'json': 'JSON',
            'xml': 'XML',
            'parquet': 'PARQUET',
            'avro': 'AVRO'
        };
        return formatMap[restFormat] || restFormat?.toUpperCase() || 'CSV';
    }

    unmapTargetFormat(odataFormat) {
        const formatMap = {
            'CSV': 'csv',
            'JSON': 'json',
            'XML': 'xml',
            'PARQUET': 'parquet',
            'AVRO': 'avro'
        };
        return formatMap[odataFormat] || odataFormat?.toLowerCase() || 'csv';
    }

    mapStatus(restStatus) {
        const statusMap = {
            'draft': 'DRAFT',
            'pending': 'PENDING',
            'running': 'RUNNING',
            'completed': 'COMPLETED',
            'failed': 'FAILED',
            'paused': 'PAUSED',
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

    mapRuleType(restType) {
        const typeMap = {
            'field_mapping': 'FIELD_MAPPING',
            'data_type_conversion': 'DATA_TYPE_CONVERSION',
            'value_transformation': 'VALUE_TRANSFORMATION',
            'validation_rule': 'VALIDATION_RULE',
            'enrichment_rule': 'ENRICHMENT_RULE'
        };
        return typeMap[restType] || restType?.toUpperCase() || 'FIELD_MAPPING';
    }

    unmapRuleType(odataType) {
        const typeMap = {
            'FIELD_MAPPING': 'field_mapping',
            'DATA_TYPE_CONVERSION': 'data_type_conversion',
            'VALUE_TRANSFORMATION': 'value_transformation',
            'VALIDATION_RULE': 'validation_rule',
            'ENRICHMENT_RULE': 'enrichment_rule'
        };
        return typeMap[odataType] || odataType?.toLowerCase() || 'field_mapping';
    }

    // =================================
    // Backend API Methods
    // =================================

    async startStandardization(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/standardize`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Standardization started successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start standardization:', error.message);
            throw new Error(`Failed to start standardization: ${error.message}`);
        }
    }

    async pauseStandardization(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/pause`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Standardization paused successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to pause standardization:', error.message);
            throw new Error(`Failed to pause standardization: ${error.message}`);
        }
    }

    async resumeStandardization(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/resume`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Standardization resumed successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to resume standardization:', error.message);
            throw new Error(`Failed to resume standardization: ${error.message}`);
        }
    }

    async cancelStandardization(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/cancel`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Standardization cancelled successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to cancel standardization:', error.message);
            throw new Error(`Failed to cancel standardization: ${error.message}`);
        }
    }

    async validateFormat(taskId, sampleData, validationRules) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/validate`, {
                sample_data: sampleData,
                validation_rules: validationRules
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Format validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to validate format:', error.message);
            throw new Error(`Failed to validate format: ${error.message}`);
        }
    }

    async analyzeDataQuality(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/analyze-quality`, {}, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Data quality analysis completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to analyze data quality:', error.message);
            throw new Error(`Failed to analyze data quality: ${error.message}`);
        }
    }

    async exportResults(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/export`, {
                format: options.format,
                include_metadata: options.includeMetadata,
                compression: options.compression
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Export completed successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to export results:', error.message);
            throw new Error(`Failed to export results: ${error.message}`);
        }
    }

    async previewTransformation(taskId, options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/tasks/${taskId}/preview`, {
                sample_size: options.sampleSize,
                rules: options.rules
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Transformation preview generated',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to generate transformation preview:', error.message);
            throw new Error(`Failed to generate transformation preview: ${error.message}`);
        }
    }

    async getFormatStatistics() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/format-statistics`, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Format statistics retrieved',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to get format statistics:', error.message);
            throw new Error(`Failed to get format statistics: ${error.message}`);
        }
    }

    async batchStandardize(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/batch-process`, {
                task_ids: options.taskIds,
                parallel: options.parallel,
                priority: options.priority
            }, {
                timeout: 60000
            });
            return {
                success: true,
                message: 'Batch standardization started',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to start batch standardization:', error.message);
            throw new Error(`Failed to start batch standardization: ${error.message}`);
        }
    }

    async importSchema(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/schema/import`, {
                schema_data: options.schemaData,
                format: options.format,
                template_name: options.templateName
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Schema imported successfully',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to import schema:', error.message);
            throw new Error(`Failed to import schema: ${error.message}`);
        }
    }

    async validateSchemaTemplate(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/schema/validate`, {
                template_id: options.templateId,
                source_data: options.sourceData
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Schema template validation completed',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to validate schema template:', error.message);
            throw new Error(`Failed to validate schema template: ${error.message}`);
        }
    }

    async generateStandardizationRules(options) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/rules/generate`, {
                source_format: options.sourceFormat,
                target_format: options.targetFormat,
                sample_data: options.sampleData
            }, {
                timeout: 30000
            });
            return {
                success: true,
                message: 'Standardization rules generated',
                data: response.data
            };
        } catch (error) {
            log.error('Failed to generate standardization rules:', error.message);
            throw new Error(`Failed to generate standardization rules: ${error.message}`);
        }
    }

    // =================================
    // Health Check
    // =================================

    async checkHealth() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/a2a/agent1/v1/health`, {
                timeout: 5000
            });
            return {
                status: 'healthy',
                data: response.data
            };
        } catch (error) {
            log.warn('Agent 1 backend health check failed:', error.message);
            return {
                status: 'unhealthy',
                error: error.message
            };
        }
    }
}

module.exports = Agent1Adapter;
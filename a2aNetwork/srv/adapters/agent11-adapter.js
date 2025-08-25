/**
 * Agent 11 Adapter - SQL Engine
 * Converts between REST API and OData formats for SQL query tasks, natural language processing,
 * query optimization, execution, and database operations
 */

const fetch = require('node-fetch');
// const { v4: uuidv4 } = require('uuid');

class Agent11Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT11_BASE_URL || 'http://localhost:8011';
        this.apiVersion = 'v1';
        this.timeout = 60000; // Longer timeout for complex SQL operations
    }

    // ===== SQL QUERY TASKS =====
    async getSQLQueryTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'SQLQueryTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createSQLQueryTask(data) {
        try {
            const restData = this._convertODataSQLQueryTaskToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTSQLQueryTaskToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateSQLQueryTask(id, data) {
        try {
            const restData = this._convertODataSQLQueryTaskToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTSQLQueryTaskToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteSQLQueryTask(id) {
        try {
            await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== SQL QUERY OPERATIONS =====
    async executeQuery(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${taskId}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout * 2 // Double timeout for execution

            });
            const data = await response.json();

            return {
                success: data.success,
                queryName: data.query_name,
                database: data.database,
                executionTime: data.execution_time,
                rowsAffected: data.rows_affected,
                resultRowCount: data.result_row_count,
                results: data.results,
                executionPlan: data.execution_plan,
                performanceMetrics: data.performance_metrics,
                memoryUsed: data.memory_used,
                cpuTime: data.cpu_time,
                details: data.details,
                error: data.error
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateSQL(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${taskId}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                isValid: data.is_valid,
                errors: data.errors,
                warnings: data.warnings,
                suggestions: data.suggestions,
                dialect: data.dialect
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeQuery(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${taskId}/optimize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                originalSQL: data.original_sql,
                optimizedSQL: data.optimized_sql,
                optimizations: data.optimizations?.map(opt => ({
                    type: opt.optimization_type?.toUpperCase(),
                    originalQuery: opt.original_query,
                    optimizedQuery: opt.optimized_query,
                    reason: opt.reason,
                    improvement: opt.performance_improvement,
                    costReduction: opt.cost_reduction,
                    benefit: opt.estimated_benefit,
                    details: opt.details
                })) || [],
                averageImprovement: data.average_improvement,
                estimatedCostSaving: data.estimated_cost_saving
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async generateFromNaturalLanguage(taskId, naturalLanguage) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${taskId}/generate-from-nl`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                natural_language: naturalLanguage
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                sql: data.generated_sql,
                confidence: data.confidence_score,
                status: data.processing_status?.toUpperCase(),
                intent: data.intent_recognition,
                entities: data.entity_extraction,
                schemaMapping: data.schema_mapping,
                ambiguities: data.ambiguities_found,
                clarifications: data.clarification_questions,
                context: data.context_used,
                processingTime: data.processing_time,
                modelVersion: data.model_version,
                alternatives: data.alternative_sqls,
                validation: data.validation_results,
                language: data.language
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async explainQuery(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sql-queries/${taskId}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                executionPlan: data.execution_plan,
                explanation: data.explanation,
                cost: data.estimated_cost,
                steps: data.execution_steps,
                recommendations: data.recommendations
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
                include_metadata: options.includeMetadata
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
                size: data.size,
                expiresAt: data.expires_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== MAIN SQL OPERATIONS =====
    async executeSQL(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/execute-sql`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                sql: data.sql,
                parameters: data.parameters,
                database: data.database,
                timeout: data.timeout
            }, {
                timeout: data.timeout + 5000 // Add buffer to axios timeout
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                success: data.success,
                results: data.results,
                executionTime: data.execution_time,
                rowsAffected: data.rows_affected,
                resultRowCount: data.result_row_count,
                executionPlan: data.execution_plan,
                performanceMetrics: data.performance_metrics,
                error: data.error
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async translateNaturalLanguage(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/translate-nl`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                natural_language: data.naturalLanguage,
                context: data.context,
                database: data.database
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                sql: data.generated_sql,
                confidence: data.confidence_score,
                processingTime: data.processing_time,
                language: data.language,
                intent: data.intent,
                entities: data.entities,
                warnings: data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeSQL(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/optimize-sql`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                sql: data.sql,
                database: data.database,
                explain: data.explain
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                originalSQL: data.original_sql,
                optimizedSQL: data.optimized_sql,
                improvements: data.improvements,
                executionPlan: data.execution_plan,
                recommendations: data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateSQL(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/validate-sql`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                sql: data.sql,
                dialect: data.dialect
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
                suggestions: data.suggestions,
                dialect: data.dialect
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async explainExecutionPlan(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/explain-plan`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                sql: data.sql,
                database: data.database
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                executionPlan: data.execution_plan,
                textPlan: data.text_plan,
                cost: data.cost,
                operations: data.operations,
                bottlenecks: data.bottlenecks
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== SCHEMA AND DATABASE OPERATIONS =====
    async getSchemaInfo(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/schema-info/${data.database}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                params: { schema: data.schema },
                timeout: this.timeout
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getTableInfo(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/table-info/${data.database}/${data.table}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async suggestIndexes(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/suggest-indexes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                table: data.table,
                database: data.database
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== ADDITIONAL OPERATIONS =====
    async analyzeQueryPerformance(queryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/performance-analysis/${queryId}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getQueryHistory(options) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/query-history`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                params: {
                    database: options.database,
                    limit: options.limit,
                    offset: options.offset
                },
                timeout: this.timeout
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createQueryTemplate(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/query-templates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                name: data.name,
                sql: data.sql,
                parameters: data.parameters
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async manageDatabaseConnection(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/connections`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                operation: data.operation,
                connection_config: data.connectionConfig
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async backupQuery(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/backup-query/${data.queryId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                include_results: data.includeResults
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async restoreQuery(backupId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/restore-query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                backup_id: backupId
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return data;
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
        if (entityType === 'SQLQueryTask') {
            return this._convertRESTSQLQueryTaskToOData(item);
        }
        return item;
    }

    _convertODataSQLQueryTaskToREST(data) {
        return {
            query_name: data.queryName,
            description: data.description,
            query_type: data.queryType?.toLowerCase(),
            natural_language_query: data.naturalLanguageQuery,
            generated_sql: data.generatedSQL,
            original_sql: data.originalSQL,
            optimized_sql: data.optimizedSQL,
            database_connection: data.databaseConnection,
            sql_dialect: data.sqlDialect?.toLowerCase() || 'hana',
            query_parameters: data.queryParameters ? JSON.parse(data.queryParameters) : {},
            execution_context: data.executionContext ? JSON.parse(data.executionContext) : {},
            priority: data.priority?.toLowerCase() || 'medium',
            requires_approval: data.requiresApproval,
            metadata: data.metadata ? JSON.parse(data.metadata) : {}
        };
    }

    _convertRESTSQLQueryTaskToOData(item) {
        return {
            ID: item.id || uuidv4(),
            queryName: item.query_name,
            description: item.description,
            queryType: item.query_type?.toUpperCase(),
            naturalLanguageQuery: item.natural_language_query,
            generatedSQL: item.generated_sql,
            originalSQL: item.original_sql,
            optimizedSQL: item.optimized_sql,
            databaseConnection: item.database_connection,
            sqlDialect: item.sql_dialect?.toUpperCase() || 'HANA',
            queryParameters: item.query_parameters ? JSON.stringify(item.query_parameters) : null,
            executionContext: item.execution_context ? JSON.stringify(item.execution_context) : null,
            priority: item.priority?.toUpperCase() || 'MEDIUM',
            status: item.status?.toUpperCase() || 'DRAFT',
            executionTime: item.execution_time,
            rowsAffected: item.rows_affected,
            resultRowCount: item.result_row_count,
            isOptimized: item.is_optimized !== false,
            autoGenerated: item.auto_generated !== false,
            requiresApproval: item.requires_approval !== false,
            isApproved: item.is_approved !== false,
            approvedBy: item.approved_by,
            approvalTimestamp: item.approval_timestamp,
            startTime: item.start_time,
            endTime: item.end_time,
            errorMessage: item.error_message,
            queryResults: item.query_results ? JSON.stringify(item.query_results) : null,
            executionPlan: item.execution_plan ? JSON.stringify(item.execution_plan) : null,
            performanceMetrics: item.performance_metrics ? JSON.stringify(item.performance_metrics) : null,
            securityContext: item.security_context ? JSON.stringify(item.security_context) : null,
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

            return new Error(`Agent 11 Error (${status}): ${message}${details ? ` - ${JSON.stringify(details)}` : ''}`);
        } else if (error.request) {
            return new Error('Agent 11 Connection Error: No response from SQL service');
        } else {
            return new Error(`Agent 11 Error: ${error.message}`);
        }
    }
}

module.exports = Agent11Adapter;
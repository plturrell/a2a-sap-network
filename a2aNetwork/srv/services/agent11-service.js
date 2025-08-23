/**
 * Agent 11 Service Implementation - SQL Engine
 * Implements business logic for SQL query tasks, natural language to SQL translation,
 * query optimization, execution, and database operations
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');
const Agent11Adapter = require('../adapters/agent11-adapter');

class Agent11Service extends cds.ApplicationService {
    async init() {
        const db = await cds.connect.to('db');
        this.adapter = new Agent11Adapter();
        
        // Entity references
        const {
            SQLQueryTasks,
            QueryOptimizations,
            QueryExecutionHistory,
            SchemaReferences,
            NLProcessingResults
        } = db.entities;

        // ===== SQL QUERY TASKS CRUD OPERATIONS =====
        this.on('read', 'SQLQueryTasks', async (req) => {
            try {
                const tasks = await this.adapter.getSQLQueryTasks(req.query);
                return tasks;
            } catch (error) {
                req.error(500, `Failed to read SQL query tasks: ${error.message}`);
            }
        });

        this.on('CREATE', 'SQLQueryTasks', async (req) => {
            try {
                const task = await this.adapter.createSQLQueryTask(req.data);
                
                // Emit task creation event
                await this.emit('QueryExecuted', {
                    queryId: task.ID,
                    queryName: task.queryName,
                    database: task.databaseConnection,
                    executionTime: 0,
                    rowsAffected: 0,
                    status: 'CREATED',
                    timestamp: new Date()
                });
                
                return task;
            } catch (error) {
                req.error(500, `Failed to create SQL query task: ${error.message}`);
            }
        });

        this.on('UPDATE', 'SQLQueryTasks', async (req) => {
            try {
                const task = await this.adapter.updateSQLQueryTask(req.params[0], req.data);
                return task;
            } catch (error) {
                req.error(500, `Failed to update SQL query task: ${error.message}`);
            }
        });

        this.on('DELETE', 'SQLQueryTasks', async (req) => {
            try {
                await this.adapter.deleteSQLQueryTask(req.params[0]);
            } catch (error) {
                req.error(500, `Failed to delete SQL query task: ${error.message}`);
            }
        });

        // ===== SQL QUERY TASK ACTIONS =====
        this.on('executeQuery', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const result = await this.adapter.executeQuery(taskId);
                
                // Update task status
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        status: result.success ? 'COMPLETED' : 'FAILED',
                        startTime: new Date(),
                        endTime: new Date(),
                        executionTime: result.executionTime,
                        rowsAffected: result.rowsAffected,
                        resultRowCount: result.resultRowCount,
                        queryResults: JSON.stringify(result.results),
                        executionPlan: JSON.stringify(result.executionPlan),
                        performanceMetrics: JSON.stringify(result.performanceMetrics),
                        errorMessage: result.error
                    })
                    .where({ ID: taskId });
                
                // Record execution history
                if (result.success) {
                    await INSERT.into(QueryExecutionHistory).entries({
                        ID: uuidv4(),
                        task_ID: taskId,
                        executionTimestamp: new Date(),
                        executionDuration: result.executionTime,
                        rowsReturned: result.resultRowCount,
                        rowsAffected: result.rowsAffected,
                        memoryUsed: result.memoryUsed,
                        cpuTime: result.cpuTime,
                        executionStatus: 'SUCCESS',
                        performanceRating: this._calculatePerformanceRating(result.executionTime),
                        executionDetails: JSON.stringify(result.details)
                    });
                }
                
                await this.emit('QueryExecuted', {
                    queryId: taskId,
                    queryName: result.queryName,
                    database: result.database,
                    executionTime: result.executionTime,
                    rowsAffected: result.rowsAffected,
                    status: result.success ? 'SUCCESS' : 'FAILED',
                    timestamp: new Date()
                });
                
                return JSON.stringify(result);
            } catch (error) {
                await this.emit('QueryError', {
                    queryId: req.params[0],
                    errorCode: 'SQL001',
                    errorMessage: error.message,
                    database: 'unknown',
                    timestamp: new Date()
                });
                req.error(500, `Failed to execute query: ${error.message}`);
            }
        });

        this.on('validateSQL', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const result = await this.adapter.validateSQL(taskId);
                
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        status: result.isValid ? 'READY' : 'DRAFT',
                        errorMessage: result.errors ? result.errors.join('; ') : null
                    })
                    .where({ ID: taskId });
                
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to validate SQL: ${error.message}`);
            }
        });

        this.on('optimizeQuery', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const result = await this.adapter.optimizeQuery(taskId);
                
                // Update task with optimized SQL
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        optimizedSQL: result.optimizedSQL,
                        isOptimized: true
                    })
                    .where({ ID: taskId });
                
                // Record optimization
                if (result.optimizations && result.optimizations.length > 0) {
                    const optimizationEntries = result.optimizations.map(opt => ({
                        ID: uuidv4(),
                        task_ID: taskId,
                        optimizationType: opt.type,
                        originalQuery: opt.originalQuery,
                        optimizedQuery: opt.optimizedQuery,
                        optimizationReason: opt.reason,
                        performanceImprovement: opt.improvement,
                        costReduction: opt.costReduction,
                        estimatedBenefit: opt.benefit,
                        isApplied: true,
                        applicationTime: new Date(),
                        validationStatus: 'APPLIED',
                        optimizationDetails: JSON.stringify(opt.details)
                    }));
                    await INSERT.into(QueryOptimizations).entries(optimizationEntries);
                    
                    await this.emit('QueryOptimized', {
                        queryId: taskId,
                        originalSQL: result.originalSQL,
                        optimizedSQL: result.optimizedSQL,
                        improvementPercent: result.averageImprovement,
                        optimizationType: result.optimizations[0].type,
                        timestamp: new Date()
                    });
                }
                
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to optimize query: ${error.message}`);
            }
        });

        this.on('generateFromNL', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const { naturalLanguage } = req.data;
                const result = await this.adapter.generateFromNaturalLanguage(taskId, naturalLanguage);
                
                // Update task with generated SQL
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        naturalLanguageQuery: naturalLanguage,
                        generatedSQL: result.sql,
                        autoGenerated: true
                    })
                    .where({ ID: taskId });
                
                // Record NL processing results
                await INSERT.into(NLProcessingResults).entries({
                    ID: uuidv4(),
                    task_ID: taskId,
                    originalQuery: naturalLanguage,
                    intentRecognition: JSON.stringify(result.intent),
                    entityExtraction: JSON.stringify(result.entities),
                    schemaMapping: JSON.stringify(result.schemaMapping),
                    generatedSQL: result.sql,
                    confidenceScore: result.confidence,
                    processingStatus: result.status,
                    ambiguitiesFound: JSON.stringify(result.ambiguities),
                    clarificationQuestions: JSON.stringify(result.clarifications),
                    contextUsed: JSON.stringify(result.context),
                    processingTime: result.processingTime,
                    modelVersion: result.modelVersion,
                    alternativeSQLs: JSON.stringify(result.alternatives),
                    validationResults: JSON.stringify(result.validation),
                    language: result.language || 'en'
                });
                
                await this.emit('NLQueryTranslated', {
                    originalText: naturalLanguage,
                    generatedSQL: result.sql,
                    confidenceScore: result.confidence,
                    language: result.language || 'en',
                    processingTime: result.processingTime,
                    timestamp: new Date()
                });
                
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to generate SQL from natural language: ${error.message}`);
            }
        });

        this.on('explainQuery', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const result = await this.adapter.explainQuery(taskId);
                
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        executionPlan: JSON.stringify(result.executionPlan)
                    })
                    .where({ ID: taskId });
                
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to explain query: ${error.message}`);
            }
        });

        this.on('approveQuery', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const userId = req.user.id;
                
                await UPDATE(SQLQueryTasks)
                    .set({ 
                        isApproved: true,
                        approvedBy: userId,
                        approvalTimestamp: new Date(),
                        status: 'READY'
                    })
                    .where({ ID: taskId });
                
                return JSON.stringify({ success: true, approvedBy: userId });
            } catch (error) {
                req.error(500, `Failed to approve query: ${error.message}`);
            }
        });

        this.on('exportResults', 'SQLQueryTasks', async (req) => {
            try {
                const taskId = req.params[0];
                const { format, includeMetadata } = req.data;
                const result = await this.adapter.exportResults(taskId, { format, includeMetadata });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to export results: ${error.message}`);
            }
        });

        // ===== MAIN SQL OPERATIONS =====
        this.on('executeSQL', async (req) => {
            try {
                const { sql, parameters, database, timeout } = req.data;
                
                // Create SQL query task
                const taskId = uuidv4();
                await INSERT.into(SQLQueryTasks).entries({
                    ID: taskId,
                    queryName: `Ad-hoc Query - ${new Date().toISOString()}`,
                    queryType: this._detectQueryType(sql),
                    generatedSQL: sql,
                    originalSQL: sql,
                    databaseConnection: database,
                    queryParameters: parameters,
                    status: 'EXECUTING',
                    startTime: new Date()
                });

                // Execute SQL via adapter
                const result = await this.adapter.executeSQL({
                    sql,
                    parameters: parameters ? JSON.parse(parameters) : {},
                    database,
                    timeout: timeout || 30000
                });

                // Update task with results
                await UPDATE(SQLQueryTasks)
                    .set({
                        status: result.success ? 'COMPLETED' : 'FAILED',
                        endTime: new Date(),
                        executionTime: result.executionTime,
                        rowsAffected: result.rowsAffected,
                        resultRowCount: result.resultRowCount,
                        queryResults: JSON.stringify(result.results),
                        executionPlan: JSON.stringify(result.executionPlan),
                        performanceMetrics: JSON.stringify(result.performanceMetrics),
                        errorMessage: result.error
                    })
                    .where({ ID: taskId });

                await this.emit('QueryExecuted', {
                    queryId: taskId,
                    queryName: 'Ad-hoc Query',
                    database,
                    executionTime: result.executionTime,
                    rowsAffected: result.rowsAffected,
                    status: result.success ? 'SUCCESS' : 'FAILED',
                    timestamp: new Date()
                });

                return JSON.stringify(result);
            } catch (error) {
                await this.emit('QueryError', {
                    queryId: 'unknown',
                    errorCode: 'SQL002',
                    errorMessage: error.message,
                    database: req.data.database || 'unknown',
                    timestamp: new Date()
                });
                req.error(500, `Failed to execute SQL: ${error.message}`);
            }
        });

        this.on('translateNaturalLanguage', async (req) => {
            try {
                const { naturalLanguage, context, database } = req.data;
                const result = await this.adapter.translateNaturalLanguage({
                    naturalLanguage,
                    context: context ? JSON.parse(context) : {},
                    database
                });

                await this.emit('NLQueryTranslated', {
                    originalText: naturalLanguage,
                    generatedSQL: result.sql,
                    confidenceScore: result.confidence,
                    language: result.language || 'en',
                    processingTime: result.processingTime,
                    timestamp: new Date()
                });

                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to translate natural language: ${error.message}`);
            }
        });

        this.on('optimizeSQL', async (req) => {
            try {
                const { sql, database, explain } = req.data;
                const result = await this.adapter.optimizeSQL({
                    sql,
                    database,
                    explain: explain !== false
                });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to optimize SQL: ${error.message}`);
            }
        });

        this.on('validateSQL', async (req) => {
            try {
                const { sql, dialect } = req.data;
                const result = await this.adapter.validateSQL({
                    sql,
                    dialect: dialect || 'HANA'
                });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to validate SQL: ${error.message}`);
            }
        });

        this.on('explainExecutionPlan', async (req) => {
            try {
                const { sql, database } = req.data;
                const result = await this.adapter.explainExecutionPlan({ sql, database });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to explain execution plan: ${error.message}`);
            }
        });

        // ===== SCHEMA AND DATABASE OPERATIONS =====
        this.on('getSchemaInfo', async (req) => {
            try {
                const { database, schema } = req.data;
                const result = await this.adapter.getSchemaInfo({ database, schema });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to get schema info: ${error.message}`);
            }
        });

        this.on('getTableInfo', async (req) => {
            try {
                const { database, table } = req.data;
                const result = await this.adapter.getTableInfo({ database, table });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to get table info: ${error.message}`);
            }
        });

        this.on('suggestIndexes', async (req) => {
            try {
                const { table, database } = req.data;
                const result = await this.adapter.suggestIndexes({ table, database });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to suggest indexes: ${error.message}`);
            }
        });

        // ===== ADDITIONAL OPERATIONS =====
        this.on('analyzeQueryPerformance', async (req) => {
            try {
                const { queryId } = req.data;
                const result = await this.adapter.analyzeQueryPerformance(queryId);
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to analyze query performance: ${error.message}`);
            }
        });

        this.on('getQueryHistory', async (req) => {
            try {
                const { database, limit, offset } = req.data;
                const result = await this.adapter.getQueryHistory({
                    database,
                    limit: limit || 100,
                    offset: offset || 0
                });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to get query history: ${error.message}`);
            }
        });

        this.on('createQueryTemplate', async (req) => {
            try {
                const { name, sql, parameters } = req.data;
                const result = await this.adapter.createQueryTemplate({ name, sql, parameters });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to create query template: ${error.message}`);
            }
        });

        this.on('manageDatabaseConnection', async (req) => {
            try {
                const { operation, connectionConfig } = req.data;
                const result = await this.adapter.manageDatabaseConnection({
                    operation,
                    connectionConfig: JSON.parse(connectionConfig)
                });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to manage database connection: ${error.message}`);
            }
        });

        this.on('backupQuery', async (req) => {
            try {
                const { queryId, includeResults } = req.data;
                const result = await this.adapter.backupQuery({ queryId, includeResults });
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to backup query: ${error.message}`);
            }
        });

        this.on('restoreQuery', async (req) => {
            try {
                const { backupId } = req.data;
                const result = await this.adapter.restoreQuery(backupId);
                return JSON.stringify(result);
            } catch (error) {
                req.error(500, `Failed to restore query: ${error.message}`);
            }
        });

        // Initialize parent service
        return super.init();
    }

    // ===== HELPER METHODS =====
    _detectQueryType(sql) {
        const sqlUpper = sql.trim().toUpperCase();
        if (sqlUpper.startsWith('SELECT')) return 'SELECT';
        if (sqlUpper.startsWith('INSERT')) return 'INSERT';
        if (sqlUpper.startsWith('UPDATE')) return 'UPDATE';
        if (sqlUpper.startsWith('DELETE')) return 'DELETE';
        if (sqlUpper.startsWith('CREATE')) return 'CREATE';
        if (sqlUpper.startsWith('DROP')) return 'DROP';
        if (sqlUpper.startsWith('ALTER')) return 'ALTER';
        if (sqlUpper.startsWith('MERGE')) return 'MERGE';
        if (sqlUpper.startsWith('CALL')) return 'CALL';
        return 'EXECUTE';
    }

    _calculatePerformanceRating(executionTime) {
        if (executionTime < 100) return 'EXCELLENT';
        if (executionTime < 1000) return 'GOOD';
        if (executionTime < 5000) return 'AVERAGE';
        if (executionTime < 10000) return 'POOR';
        return 'CRITICAL';
    }
}

module.exports = Agent11Service;
/**
 * Agent 4 Calculation Validation Service Implementation
 * Bridges SAP CAP OData service with Python FastAPI backend
 */

const cds = require('@sap/cds');
const axios = require('axios');

// Import the adapter for REST/OData conversion
const Agent4Adapter = require('./agent4-adapter');

const log = cds.log('agent4-service');

// Initialize the adapter
const adapter = new Agent4Adapter();

// Backend configuration
const AGENT4_BASE_URL = process.env.AGENT4_BASE_URL || 'http://localhost:8003';

module.exports = cds.service.impl(async function() {
    
    const { CalcValidationTasks, CalcValidationResults, CalcValidationTemplates } = this.entities;
    
    // =================================
    // CRUD Operations for CalcValidationTasks
    // =================================
    
    this.on('READ', CalcValidationTasks, async (req) => {
        try {
            log.info('Reading CalcValidationTasks');
            
            // Get data from Python backend
            const response = await axios.get(`${AGENT4_BASE_URL}/a2a/agent4/v1/tasks`, {
                timeout: 30000
            });
            
            // Convert to OData format using adapter
            const odataResults = response.data.map(task => adapter.convertTaskToOData(task));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading CalcValidationTasks:', error.message);
            req.error(503, `Agent 4 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', CalcValidationTasks, async (req) => {
        try {
            log.info('Creating CalcValidationTask');
            
            // Convert OData to REST format
            const restData = adapter.convertTaskToRest(req.data);
            
            // Send to Python backend
            const response = await axios.post(`${AGENT4_BASE_URL}/a2a/agent4/v1/tasks`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            // Convert response back to OData format
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error creating CalcValidationTask:', error.message);
            req.error(500, `Failed to create task: ${error.message}`);
        }
    });
    
    this.on('UPDATE', CalcValidationTasks, async (req) => {
        try {
            log.info('Updating CalcValidationTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            const restData = adapter.convertTaskToRest(req.data);
            
            const response = await axios.put(`${AGENT4_BASE_URL}/a2a/agent4/v1/tasks/${taskId}`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error updating CalcValidationTask:', error.message);
            req.error(500, `Failed to update task: ${error.message}`);
        }
    });
    
    this.on('DELETE', CalcValidationTasks, async (req) => {
        try {
            log.info('Deleting CalcValidationTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            
            await axios.delete(`${AGENT4_BASE_URL}/a2a/agent4/v1/tasks/${taskId}`, {
                timeout: 30000
            });
            
            return req.params[0];
        } catch (error) {
            log.error('Error deleting CalcValidationTask:', error.message);
            req.error(500, `Failed to delete task: ${error.message}`);
        }
    });
    
    // =================================
    // Action Handlers for CalcValidationTasks
    // =================================
    
    this.on('startValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Starting validation for task: ${ID}`);
            
            // Get task details from database
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            // Start validation via backend
            const result = await adapter.startValidation(task.agent4TaskId);
            
            // Update task status
            await UPDATE(CalcValidationTasks)
                .set({ status: 'VALIDATING', startedAt: new Date().toISOString() })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error starting validation:', error.message);
            req.error(500, `Failed to start validation: ${error.message}`);
        }
    });
    
    this.on('pauseValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Pausing validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.pauseValidation(task.agent4TaskId);
            
            await UPDATE(CalcValidationTasks)
                .set({ status: 'PAUSED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error pausing validation:', error.message);
            req.error(500, `Failed to pause validation: ${error.message}`);
        }
    });
    
    this.on('resumeValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Resuming validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.resumeValidation(task.agent4TaskId);
            
            await UPDATE(CalcValidationTasks)
                .set({ status: 'VALIDATING' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error resuming validation:', error.message);
            req.error(500, `Failed to resume validation: ${error.message}`);
        }
    });
    
    this.on('cancelValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Cancelling validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.cancelValidation(task.agent4TaskId);
            
            await UPDATE(CalcValidationTasks)
                .set({ status: 'CANCELLED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error cancelling validation:', error.message);
            req.error(500, `Failed to cancel validation: ${error.message}`);
        }
    });
    
    this.on('runSymbolicValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Running symbolic validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runSymbolicValidation(task.agent4TaskId);
            
            return result;
        } catch (error) {
            log.error('Error running symbolic validation:', error.message);
            req.error(500, `Failed to run symbolic validation: ${error.message}`);
        }
    });
    
    this.on('runNumericalValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Running numerical validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runNumericalValidation(task.agent4TaskId);
            
            return result;
        } catch (error) {
            log.error('Error running numerical validation:', error.message);
            req.error(500, `Failed to run numerical validation: ${error.message}`);
        }
    });
    
    this.on('runStatisticalValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Running statistical validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runStatisticalValidation(task.agent4TaskId);
            
            return result;
        } catch (error) {
            log.error('Error running statistical validation:', error.message);
            req.error(500, `Failed to run statistical validation: ${error.message}`);
        }
    });
    
    this.on('runAIValidation', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Running AI validation for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runAIValidation(task.agent4TaskId);
            
            return result;
        } catch (error) {
            log.error('Error running AI validation:', error.message);
            req.error(500, `Failed to run AI validation: ${error.message}`);
        }
    });
    
    this.on('runBlockchainConsensus', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Running blockchain consensus for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.runBlockchainConsensus(task.agent4TaskId);
            
            return result;
        } catch (error) {
            log.error('Error running blockchain consensus:', error.message);
            req.error(500, `Failed to run blockchain consensus: ${error.message}`);
        }
    });
    
    this.on('exportValidationReport', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { format, includeSteps, includeConfidence } = req.data;
            
            log.info(`Exporting validation report for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.exportValidationReport(task.agent4TaskId, {
                format,
                includeSteps,
                includeConfidence
            });
            
            return result;
        } catch (error) {
            log.error('Error exporting validation report:', error.message);
            req.error(500, `Failed to export validation report: ${error.message}`);
        }
    });
    
    this.on('validateFromTemplate', CalcValidationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { templateId, variables } = req.data;
            
            log.info(`Validating from template for task: ${ID}`);
            
            const task = await SELECT.one.from(CalcValidationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.validateFromTemplate(task.agent4TaskId, {
                templateId,
                variables
            });
            
            return result;
        } catch (error) {
            log.error('Error validating from template:', error.message);
            req.error(500, `Failed to validate from template: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for CalcValidationResults
    // =================================
    
    this.on('READ', CalcValidationResults, async (req) => {
        try {
            log.info('Reading CalcValidationResults');
            
            const response = await axios.get(`${AGENT4_BASE_URL}/a2a/agent4/v1/results`, {
                timeout: 30000
            });
            
            const odataResults = response.data.map(result => adapter.convertResultToOData(result));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading CalcValidationResults:', error.message);
            req.error(503, `Agent 4 Backend Error: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for CalcValidationTemplates
    // =================================
    
    this.on('READ', CalcValidationTemplates, async (req) => {
        try {
            log.info('Reading CalcValidationTemplates');
            
            const response = await axios.get(`${AGENT4_BASE_URL}/a2a/agent4/v1/templates`, {
                timeout: 30000
            });
            
            const odataResults = response.data.map(template => adapter.convertTemplateToOData(template));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading CalcValidationTemplates:', error.message);
            req.error(503, `Agent 4 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', CalcValidationTemplates, async (req) => {
        try {
            log.info('Creating CalcValidationTemplate');
            
            const restData = adapter.convertTemplateToRest(req.data);
            
            const response = await axios.post(`${AGENT4_BASE_URL}/a2a/agent4/v1/templates`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            const odataResult = adapter.convertTemplateToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error creating CalcValidationTemplate:', error.message);
            req.error(500, `Failed to create template: ${error.message}`);
        }
    });
    
    // =================================
    // Service-level Actions
    // =================================
    
    this.on('batchValidateCalculations', async (req) => {
        try {
            const { taskIds, validationMethod, parallel, priority } = req.data;
            
            log.info(`Starting batch validation for ${taskIds.length} tasks`);
            
            const result = await adapter.batchValidateCalculations({
                taskIds,
                validationMethod,
                parallel,
                priority
            });
            
            return result;
        } catch (error) {
            log.error('Error starting batch validation:', error.message);
            req.error(500, `Failed to start batch validation: ${error.message}`);
        }
    });
    
    this.on('validateExpression', async (req) => {
        try {
            const { expression, variables, method, precision } = req.data;
            
            log.info(`Validating expression: ${expression}`);
            
            const result = await adapter.validateExpression({
                expression,
                variables,
                method,
                precision
            });
            
            return result;
        } catch (error) {
            log.error('Error validating expression:', error.message);
            req.error(500, `Failed to validate expression: ${error.message}`);
        }
    });
    
    this.on('getValidationMethods', async (req) => {
        try {
            log.info('Getting validation methods');
            
            const result = await adapter.getValidationMethods();
            
            return result;
        } catch (error) {
            log.error('Error getting validation methods:', error.message);
            req.error(500, `Failed to get validation methods: ${error.message}`);
        }
    });
    
    this.on('getCalculationTemplates', async (req) => {
        try {
            log.info('Getting calculation templates');
            
            const result = await adapter.getCalculationTemplates();
            
            return result;
        } catch (error) {
            log.error('Error getting calculation templates:', error.message);
            req.error(500, `Failed to get calculation templates: ${error.message}`);
        }
    });
    
    this.on('createTemplate', async (req) => {
        try {
            const { name, category, expression, variables, defaultMethod } = req.data;
            
            log.info(`Creating template: ${name}`);
            
            const result = await adapter.createTemplate({
                name,
                category,
                expression,
                variables,
                defaultMethod
            });
            
            return result;
        } catch (error) {
            log.error('Error creating template:', error.message);
            req.error(500, `Failed to create template: ${error.message}`);
        }
    });
    
    this.on('benchmarkMethods', async (req) => {
        try {
            const { expression, variables, iterations } = req.data;
            
            log.info(`Benchmarking methods for expression: ${expression}`);
            
            const result = await adapter.benchmarkMethods({
                expression,
                variables,
                iterations
            });
            
            return result;
        } catch (error) {
            log.error('Error benchmarking methods:', error.message);
            req.error(500, `Failed to benchmark methods: ${error.message}`);
        }
    });
    
    this.on('configureAIModel', async (req) => {
        try {
            const { model, parameters } = req.data;
            
            log.info(`Configuring AI model: ${model}`);
            
            const result = await adapter.configureAIModel({
                model,
                parameters
            });
            
            return result;
        } catch (error) {
            log.error('Error configuring AI model:', error.message);
            req.error(500, `Failed to configure AI model: ${error.message}`);
        }
    });
    
    this.on('configureBlockchainConsensus', async (req) => {
        try {
            const { validators, threshold, timeout } = req.data;
            
            log.info(`Configuring blockchain consensus: ${validators} validators`);
            
            const result = await adapter.configureBlockchainConsensus({
                validators,
                threshold,
                timeout
            });
            
            return result;
        } catch (error) {
            log.error('Error configuring blockchain consensus:', error.message);
            req.error(500, `Failed to configure blockchain consensus: ${error.message}`);
        }
    });
    
    // Error handling middleware
    this.on('error', (err, req) => {
        log.error('Service error:', err);
        
        // Provide user-friendly error messages
        switch (err.code) {
            case 'ECONNREFUSED':
                req.error(503, 'Agent 4 calculation validation service is currently unavailable');
                break;
            case 'ETIMEDOUT':
                req.error(408, 'Request to Agent 4 service timed out');
                break;
            default:
                req.error(500, 'Internal server error in Agent 4 service');
        }
    });
    
    log.info('Agent 4 Calculation Validation Service initialized successfully');
});
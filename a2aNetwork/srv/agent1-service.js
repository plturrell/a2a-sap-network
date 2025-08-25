/**
 * Agent 1 Data Standardization Service Implementation
 * Bridges SAP CAP OData service with Python FastAPI backend
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
const { BlockchainClient } = require('../core/blockchain-client');

// Import the adapter for REST/OData conversion
const Agent1Adapter = require('./agent1-adapter');

const log = cds.log('agent1-service');

// Initialize the adapter
const adapter = new Agent1Adapter();

// Backend configuration
const AGENT1_BASE_URL = process.env.AGENT1_BASE_URL || 'http://localhost:8001';

module.exports = cds.service.impl(async function() {
    
    const { StandardizationTasks, StandardizationRules } = this.entities;
    
    // =================================
    // CRUD Operations for StandardizationTasks
    // =================================
    
    this.on('READ', StandardizationTasks, async (req) => {
        try {
            log.info('Reading StandardizationTasks');
            
            // Get data from Python backend
            const response = await blockchainClient.sendMessage(`${AGENT1_BASE_URL}/a2a/agent1/v1/tasks`, {
                timeout: 30000
            });
            
            // Convert to OData format using adapter
            const odataResults = response.data.map(task => adapter.convertTaskToOData(task));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading StandardizationTasks:', error.message);
            req.error(503, `Agent 1 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', StandardizationTasks, async (req) => {
        try {
            log.info('Creating StandardizationTask');
            
            // Convert OData to REST format
            const restData = adapter.convertTaskToRest(req.data);
            
            // Send to Python backend
            const response = await blockchainClient.sendMessage(`${AGENT1_BASE_URL}/a2a/agent1/v1/tasks`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            // Convert response back to OData format
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error creating StandardizationTask:', error.message);
            req.error(500, `Failed to create task: ${error.message}`);
        }
    });
    
    this.on('UPDATE', StandardizationTasks, async (req) => {
        try {
            log.info('Updating StandardizationTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            const restData = adapter.convertTaskToRest(req.data);
            
            const response = await blockchainClient.sendMessage(`${AGENT1_BASE_URL}/a2a/agent1/v1/tasks/${taskId}`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            const odataResult = adapter.convertTaskToOData(response.data);
            
            return odataResult;
        } catch (error) {
            log.error('Error updating StandardizationTask:', error.message);
            req.error(500, `Failed to update task: ${error.message}`);
        }
    });
    
    this.on('DELETE', StandardizationTasks, async (req) => {
        try {
            log.info('Deleting StandardizationTask:', req.params[0].ID);
            
            const taskId = req.params[0].ID;
            
            await blockchainClient.sendMessage(`${AGENT1_BASE_URL}/a2a/agent1/v1/tasks/${taskId}`, {
                timeout: 30000
            });
            
            return req.params[0];
        } catch (error) {
            log.error('Error deleting StandardizationTask:', error.message);
            req.error(500, `Failed to delete task: ${error.message}`);
        }
    });
    
    // =================================
    // Action Handlers for StandardizationTasks
    // =================================
    
    this.on('startStandardization', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Starting standardization for task: ${ID}`);
            
            // Get task details from database
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            // Start standardization via backend
            const result = await adapter.startStandardization(task.agent1TaskId);
            
            // Update task status
            await UPDATE(StandardizationTasks)
                .set({ status: 'RUNNING', startedAt: new Date().toISOString() })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error starting standardization:', error.message);
            req.error(500, `Failed to start standardization: ${error.message}`);
        }
    });
    
    this.on('pauseStandardization', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Pausing standardization for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.pauseStandardization(task.agent1TaskId);
            
            await UPDATE(StandardizationTasks)
                .set({ status: 'PAUSED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error pausing standardization:', error.message);
            req.error(500, `Failed to pause standardization: ${error.message}`);
        }
    });
    
    this.on('resumeStandardization', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Resuming standardization for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.resumeStandardization(task.agent1TaskId);
            
            await UPDATE(StandardizationTasks)
                .set({ status: 'RUNNING' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error resuming standardization:', error.message);
            req.error(500, `Failed to resume standardization: ${error.message}`);
        }
    });
    
    this.on('cancelStandardization', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Cancelling standardization for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.cancelStandardization(task.agent1TaskId);
            
            await UPDATE(StandardizationTasks)
                .set({ status: 'CANCELLED' })
                .where({ ID });
            
            return result;
        } catch (error) {
            log.error('Error cancelling standardization:', error.message);
            req.error(500, `Failed to cancel standardization: ${error.message}`);
        }
    });
    
    this.on('validateFormat', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { sampleData, validationRules } = req.data;
            
            log.info(`Validating format for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.validateFormat(task.agent1TaskId, sampleData, validationRules);
            
            return result;
        } catch (error) {
            log.error('Error validating format:', error.message);
            req.error(500, `Failed to validate format: ${error.message}`);
        }
    });
    
    this.on('analyzeDataQuality', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            log.info(`Analyzing data quality for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.analyzeDataQuality(task.agent1TaskId);
            
            return result;
        } catch (error) {
            log.error('Error analyzing data quality:', error.message);
            req.error(500, `Failed to analyze data quality: ${error.message}`);
        }
    });
    
    this.on('exportResults', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { format, includeMetadata, compression } = req.data;
            
            log.info(`Exporting results for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.exportResults(task.agent1TaskId, {
                format,
                includeMetadata,
                compression
            });
            
            return result;
        } catch (error) {
            log.error('Error exporting results:', error.message);
            req.error(500, `Failed to export results: ${error.message}`);
        }
    });
    
    this.on('previewTransformation', StandardizationTasks, async (req) => {
        try {
            const { ID } = req.params[0];
            const { sampleSize, rules } = req.data;
            
            log.info(`Generating transformation preview for task: ${ID}`);
            
            const task = await SELECT.one.from(StandardizationTasks).where({ ID });
            if (!task) {
                req.error(404, 'Task not found');
                return;
            }
            
            const result = await adapter.previewTransformation(task.agent1TaskId, {
                sampleSize,
                rules
            });
            
            return result;
        } catch (error) {
            log.error('Error generating transformation preview:', error.message);
            req.error(500, `Failed to generate preview: ${error.message}`);
        }
    });
    
    // =================================
    // Service-level Actions
    // =================================
    
    this.on('getFormatStatistics', async (req) => {
        try {
            log.info('Getting format statistics');
            
            const result = await adapter.getFormatStatistics();
            
            return result;
        } catch (error) {
            log.error('Error getting format statistics:', error.message);
            req.error(500, `Failed to get format statistics: ${error.message}`);
        }
    });
    
    this.on('batchStandardize', async (req) => {
        try {
            const { taskIds, parallel, priority } = req.data;
            
            log.info(`Starting batch standardization for ${taskIds.length} tasks`);
            
            const result = await adapter.batchStandardize({
                taskIds,
                parallel,
                priority
            });
            
            return result;
        } catch (error) {
            log.error('Error starting batch standardization:', error.message);
            req.error(500, `Failed to start batch standardization: ${error.message}`);
        }
    });
    
    this.on('importSchema', async (req) => {
        try {
            const { schemaData, format, templateName } = req.data;
            
            log.info(`Importing schema template: ${templateName}`);
            
            const result = await adapter.importSchema({
                schemaData,
                format,
                templateName
            });
            
            return result;
        } catch (error) {
            log.error('Error importing schema:', error.message);
            req.error(500, `Failed to import schema: ${error.message}`);
        }
    });
    
    this.on('validateSchemaTemplate', async (req) => {
        try {
            const { templateId, sourceData } = req.data;
            
            log.info(`Validating schema template: ${templateId}`);
            
            const result = await adapter.validateSchemaTemplate({
                templateId,
                sourceData
            });
            
            return result;
        } catch (error) {
            log.error('Error validating schema template:', error.message);
            req.error(500, `Failed to validate schema template: ${error.message}`);
        }
    });
    
    this.on('generateStandardizationRules', async (req) => {
        try {
            const { sourceFormat, targetFormat, sampleData } = req.data;
            
            log.info(`Generating standardization rules: ${sourceFormat} -> ${targetFormat}`);
            
            const result = await adapter.generateStandardizationRules({
                sourceFormat,
                targetFormat,
                sampleData
            });
            
            return result;
        } catch (error) {
            log.error('Error generating standardization rules:', error.message);
            req.error(500, `Failed to generate rules: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for StandardizationRules
    // =================================
    
    this.on('READ', StandardizationRules, async (req) => {
        try {
            log.info('Reading StandardizationRules');
            
            const response = await blockchainClient.sendMessage(`${AGENT1_BASE_URL}/a2a/agent1/v1/rules`, {
                timeout: 30000
            });
            
            const odataResults = response.data.map(rule => adapter.convertRuleToOData(rule));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading StandardizationRules:', error.message);
            req.error(503, `Agent 1 Backend Error: ${error.message}`);
        }
    });
    
    // Error handling middleware
    this.on('error', (err, req) => {
        log.error('Service error:', err);
        
        // Provide user-friendly error messages
        switch (err.code) {
            case 'ECONNREFUSED':
                req.error(503, 'Agent 1 data standardization service is currently unavailable');
                break;
            case 'ETIMEDOUT':
                req.error(408, 'Request to Agent 1 service timed out');
                break;
            default:
                req.error(500, 'Internal server error in Agent 1 service');
        }
    });
    
    log.info('Agent 1 Data Standardization Service initialized successfully');
});
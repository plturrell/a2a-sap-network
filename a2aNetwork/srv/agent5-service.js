/**
 * Agent 5 QA Validation Service Implementation
 * Bridges SAP CAP OData service with Python FastAPI backend
 */

const cds = require('@sap/cds');
const axios = require('axios');

// Import the adapter for REST/OData conversion
const Agent5Adapter = require('./agent5-adapter');

const log = cds.log('agent5-service');

// Initialize the adapter
const adapter = new Agent5Adapter();

// Backend configuration
const AGENT5_BASE_URL = process.env.AGENT5_BASE_URL || 'http://localhost:8004';

module.exports = cds.service.impl(async function() {
    
    const { QaValidationTasks, QaValidationRules, QaTestResults, QaApprovalWorkflows } = this.entities;
    
    // =================================
    // CRUD Operations for QaValidationTasks
    // =================================
    
    this.on('READ', QaValidationTasks, async (req) => {
        try {
            log.info('Reading QaValidationTasks');
            
            // Get data from Python backend
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks`, {
                timeout: 60000,
                params: req.query
            });
            
            // Convert to OData format using adapter
            const odataResults = response.data.map(task => adapter.convertTaskToOData(task));
            
            return odataResults;
        } catch (error) {
            log.error('Error reading QaValidationTasks:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', QaValidationTasks, async (req) => {
        try {
            log.info('Creating QaValidationTask');
            
            // Convert OData to REST format
            const restData = adapter.convertODataToTask(req.data);
            
            // Create task in Python backend
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks`, restData, {
                timeout: 60000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            // Convert response back to OData format
            return adapter.convertTaskToOData(response.data);
        } catch (error) {
            log.error('Error creating QaValidationTask:', error.message);
            req.error(500, `Failed to create QA validation task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('UPDATE', QaValidationTasks, async (req) => {
        try {
            log.info(`Updating QaValidationTask: ${req.params[0]}`);
            
            // Convert OData to REST format
            const restData = adapter.convertODataToTask(req.data);
            
            // Update task in Python backend
            const response = await axios.put(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}`, restData, {
                timeout: 60000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertTaskToOData(response.data);
        } catch (error) {
            log.error('Error updating QaValidationTask:', error.message);
            req.error(500, `Failed to update QA validation task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('DELETE', QaValidationTasks, async (req) => {
        try {
            log.info(`Deleting QaValidationTask: ${req.params[0]}`);
            
            await axios.delete(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}`, {
                timeout: 30000
            });
            
            return req.params[0];
        } catch (error) {
            log.error('Error deleting QaValidationTask:', error.message);
            req.error(500, `Failed to delete QA validation task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // Custom Actions for QaValidationTasks
    // =================================
    
    this.on('validateTask', QaValidationTasks, async (req) => {
        try {
            log.info(`Validating QaValidationTask: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}/validate`, req.data, {
                timeout: 300000, // 5 minutes for validation
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertValidationResultToOData(response.data);
        } catch (error) {
            log.error('Error validating task:', error.message);
            req.error(500, `Failed to validate task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('pauseTask', QaValidationTasks, async (req) => {
        try {
            log.info(`Pausing QaValidationTask: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}/pause`, {}, {
                timeout: 30000
            });
            
            return adapter.convertTaskToOData(response.data);
        } catch (error) {
            log.error('Error pausing task:', error.message);
            req.error(500, `Failed to pause task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('resumeTask', QaValidationTasks, async (req) => {
        try {
            log.info(`Resuming QaValidationTask: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}/resume`, {}, {
                timeout: 30000
            });
            
            return adapter.convertTaskToOData(response.data);
        } catch (error) {
            log.error('Error resuming task:', error.message);
            req.error(500, `Failed to resume task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('cancelTask', QaValidationTasks, async (req) => {
        try {
            log.info(`Cancelling QaValidationTask: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks/${req.params[0]}/cancel`, {}, {
                timeout: 30000
            });
            
            return adapter.convertTaskToOData(response.data);
        } catch (error) {
            log.error('Error cancelling task:', error.message);
            req.error(500, `Failed to cancel task: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for QaValidationRules
    // =================================
    
    this.on('READ', QaValidationRules, async (req) => {
        try {
            log.info('Reading QaValidationRules');
            
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/rules`, {
                timeout: 30000,
                params: req.query
            });
            
            const odataResults = response.data.map(rule => adapter.convertRuleToOData(rule));
            return odataResults;
        } catch (error) {
            log.error('Error reading QaValidationRules:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', QaValidationRules, async (req) => {
        try {
            log.info('Creating QaValidationRule');
            
            const restData = adapter.convertODataToRule(req.data);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/rules`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertRuleToOData(response.data);
        } catch (error) {
            log.error('Error creating QaValidationRule:', error.message);
            req.error(500, `Failed to create validation rule: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('UPDATE', QaValidationRules, async (req) => {
        try {
            log.info(`Updating QaValidationRule: ${req.params[0]}`);
            
            const restData = adapter.convertODataToRule(req.data);
            
            const response = await axios.put(`${AGENT5_BASE_URL}/a2a/agent5/v1/rules/${req.params[0]}`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertRuleToOData(response.data);
        } catch (error) {
            log.error('Error updating QaValidationRule:', error.message);
            req.error(500, `Failed to update validation rule: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('DELETE', QaValidationRules, async (req) => {
        try {
            log.info(`Deleting QaValidationRule: ${req.params[0]}`);
            
            await axios.delete(`${AGENT5_BASE_URL}/a2a/agent5/v1/rules/${req.params[0]}`, {
                timeout: 30000
            });
            
            return req.params[0];
        } catch (error) {
            log.error('Error deleting QaValidationRule:', error.message);
            req.error(500, `Failed to delete validation rule: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // Custom Actions for QaValidationRules
    // =================================
    
    this.on('testRule', QaValidationRules, async (req) => {
        try {
            log.info(`Testing QaValidationRule: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/rules/${req.params[0]}/test`, req.data, {
                timeout: 60000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertRuleTestResultToOData(response.data);
        } catch (error) {
            log.error('Error testing rule:', error.message);
            req.error(500, `Failed to test rule: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for QaTestResults
    // =================================
    
    this.on('READ', QaTestResults, async (req) => {
        try {
            log.info('Reading QaTestResults');
            
            // Extract task ID from query or context
            const taskId = req.query?.taskId || req.params?.[0];
            if (!taskId) {
                req.error(400, 'Task ID is required for reading test results');
                return;
            }
            
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/tests/${taskId}/results`, {
                timeout: 30000,
                params: req.query
            });
            
            const odataResults = response.data.map(result => adapter.convertTestResultToOData(result));
            return odataResults;
        } catch (error) {
            log.error('Error reading QaTestResults:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    // =================================
    // CRUD Operations for QaApprovalWorkflows
    // =================================
    
    this.on('READ', QaApprovalWorkflows, async (req) => {
        try {
            log.info('Reading QaApprovalWorkflows');
            
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/approvals`, {
                timeout: 30000,
                params: req.query
            });
            
            const odataResults = response.data.map(approval => adapter.convertApprovalToOData(approval));
            return odataResults;
        } catch (error) {
            log.error('Error reading QaApprovalWorkflows:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    this.on('CREATE', QaApprovalWorkflows, async (req) => {
        try {
            log.info('Creating QaApprovalWorkflow');
            
            const restData = adapter.convertODataToApproval(req.data);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/approvals`, restData, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertApprovalToOData(response.data);
        } catch (error) {
            log.error('Error creating QaApprovalWorkflow:', error.message);
            req.error(500, `Failed to create approval workflow: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // Custom Actions for QaApprovalWorkflows
    // =================================
    
    this.on('approveWorkflow', QaApprovalWorkflows, async (req) => {
        try {
            log.info(`Approving QaApprovalWorkflow: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/approvals/${req.params[0]}/approve`, req.data, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertApprovalToOData(response.data);
        } catch (error) {
            log.error('Error approving workflow:', error.message);
            req.error(500, `Failed to approve workflow: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('rejectWorkflow', QaApprovalWorkflows, async (req) => {
        try {
            log.info(`Rejecting QaApprovalWorkflow: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/approvals/${req.params[0]}/reject`, req.data, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertApprovalToOData(response.data);
        } catch (error) {
            log.error('Error rejecting workflow:', error.message);
            req.error(500, `Failed to reject workflow: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('escalateWorkflow', QaApprovalWorkflows, async (req) => {
        try {
            log.info(`Escalating QaApprovalWorkflow: ${req.params[0]}`);
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/approvals/${req.params[0]}/escalate`, req.data, {
                timeout: 30000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertApprovalToOData(response.data);
        } catch (error) {
            log.error('Error escalating workflow:', error.message);
            req.error(500, `Failed to escalate workflow: ${error.response?.data?.message || error.message}`);
        }
    });
    
    // =================================
    // Utility Functions and Actions
    // =================================
    
    this.on('generateSimpleQATests', async (req) => {
        try {
            log.info('Generating SimpleQA tests');
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/tests/simpleqa`, req.data, {
                timeout: 180000, // 3 minutes for test generation
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertTestGenerationResultToOData(response.data);
        } catch (error) {
            log.error('Error generating SimpleQA tests:', error.message);
            req.error(500, `Failed to generate SimpleQA tests: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('discoverORDRegistry', async (req) => {
        try {
            log.info('Discovering ORD registry');
            
            const response = await axios.post(`${AGENT5_BASE_URL}/a2a/agent5/v1/ord/discover`, req.data, {
                timeout: 120000, // 2 minutes for discovery
                headers: { 'Content-Type': 'application/json' }
            });
            
            return adapter.convertORDDiscoveryResultToOData(response.data);
        } catch (error) {
            log.error('Error discovering ORD registry:', error.message);
            req.error(500, `Failed to discover ORD registry: ${error.response?.data?.message || error.message}`);
        }
    });
    
    this.on('getQAMetrics', async (req) => {
        try {
            log.info('Getting QA metrics');
            
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/analytics/metrics`, {
                timeout: 30000,
                params: req.query
            });
            
            return adapter.convertMetricsToOData(response.data);
        } catch (error) {
            log.error('Error getting QA metrics:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    this.on('getQATrends', async (req) => {
        try {
            log.info('Getting QA trends');
            
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/analytics/trends`, {
                timeout: 30000,
                params: req.query
            });
            
            return adapter.convertTrendsToOData(response.data);
        } catch (error) {
            log.error('Error getting QA trends:', error.message);
            req.error(503, `Agent 5 Backend Error: ${error.message}`);
        }
    });
    
    // Health check endpoint
    this.on('getHealth', async (req) => {
        try {
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/health`, {
                timeout: 10000
            });
            
            return {
                status: 'healthy',
                backend: response.data,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            log.error('Health check failed:', error.message);
            return {
                status: 'unhealthy',
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    });
    
    log.info('Agent 5 QA Validation Service initialized successfully');
});
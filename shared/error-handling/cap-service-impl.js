/**
 * CAP Service Implementation with A2A Error Handling
 * Example of how to use the error handler in CAP services
 */

const cds = require('@sap/cds');
const { A2ACapErrorHandler, ErrorMiddleware, ErrorHandlerFactory } = require('./cap-error-handler');

/**
 * Base service class with error handling
 */
class A2ABaseService extends cds.ApplicationService {
  async init() {
    // Initialize error handler
    this.errorHandler = ErrorHandlerFactory.createNetworkErrorHandler(
      process.env.NODE_ENV || 'development'
    );
    
    // Register error handlers
    this.errorHandler.registerHandlers(this);
    
    // Add common middleware
    this.before('*', ErrorMiddleware.checkRateLimit);
    this.before(['READ', 'UPDATE', 'DELETE'], ErrorMiddleware.validateEntityExists);
    this.before(['CREATE', 'UPDATE'], ErrorMiddleware.validateDataQuality);
    
    // Call parent init
    return super.init();
  }

  /**
   * Helper method to handle async operations with error handling
   */
  async executeWithErrorHandling(operation, context) {
    try {
      return await operation();
    } catch (error) {
      // Log and transform error
      const processedError = this.errorHandler.processError(error, context);
      
      // Throw CAP-formatted error
      this.reject(
        this.getHttpStatusCode(processedError.code),
        processedError.message,
        {
          code: processedError.code,
          severity: processedError.severity,
          details: processedError.details,
          retryable: processedError.retryable
        }
      );
    }
  }

  /**
   * Get HTTP status code for error code
   */
  getHttpStatusCode(errorCode) {
    const statusMap = {
      'A2A_1001': 404, // AGENT_NOT_FOUND
      'A2A_1002': 503, // AGENT_UNAVAILABLE
      'A2A_1003': 408, // AGENT_TIMEOUT
      'A2A_1007': 401, // AGENT_AUTHENTICATION_FAILED
      'A2A_2001': 502, // NETWORK_CONNECTION_ERROR
      'A2A_2002': 408, // NETWORK_TIMEOUT
      'A2A_2004': 429, // NETWORK_RATE_LIMIT_EXCEEDED
      'A2A_3001': 400, // DATA_VALIDATION_ERROR
      'A2A_3002': 404, // DATA_NOT_FOUND
      'A2A_3004': 400, // DATA_FORMAT_ERROR
      'A2A_3005': 413, // DATA_SIZE_LIMIT_EXCEEDED
      'A2A_3006': 422, // DATA_QUALITY_CHECK_FAILED
      'A2A_3007': 403, // DATA_ACCESS_DENIED
      'A2A_9001': 500, // INTERNAL_SERVER_ERROR
      'A2A_9002': 503, // SERVICE_UNAVAILABLE
      'A2A_9003': 500  // DATABASE_ERROR
    };
    
    return statusMap[errorCode] || 500;
  }
}

/**
 * Example: Agent Service Implementation
 */
class AgentService extends A2ABaseService {
  async init() {
    await super.init();
    
    // Agent-specific handlers
    this.on('READ', 'Agents', this.onReadAgents);
    this.on('CREATE', 'Agents', this.onCreateAgent);
    this.on('UPDATE', 'Agents', this.onUpdateAgent);
    this.on('executeTask', this.onExecuteTask);
    
    return this;
  }

  async onReadAgents(req) {
    return this.executeWithErrorHandling(async () => {
      // Simulate agent retrieval
      const agents = await cds.run(req.query);
      
      if (!agents || agents.length === 0) {
        throw new Error('No agents found');
      }
      
      return agents;
    }, req);
  }

  async onCreateAgent(req) {
    return this.executeWithErrorHandling(async () => {
      // Validate agent data
      this.validateAgentData(req.data);
      
      // Create agent
      const agent = await cds.run(
        INSERT.into('Agents').entries(req.data)
      );
      
      return agent;
    }, req);
  }

  async onUpdateAgent(req) {
    return this.executeWithErrorHandling(async () => {
      const { ID } = req.params[0];
      
      // Check if agent exists
      const existing = await cds.run(
        SELECT.one.from('Agents').where({ ID })
      );
      
      if (!existing) {
        const error = new Error('Agent not found');
        error.code = 'A2A_1001';
        throw error;
      }
      
      // Update agent
      await cds.run(
        UPDATE('Agents').set(req.data).where({ ID })
      );
      
      return req.data;
    }, req);
  }

  async onExecuteTask(req) {
    return this.executeWithErrorHandling(async () => {
      const { agentId, task } = req.data;
      
      // Simulate task execution with potential errors
      if (!agentId) {
        const error = new Error('Agent ID is required');
        error.code = 'A2A_3001';
        throw error;
      }
      
      // Simulate timeout
      if (task.type === 'long-running') {
        const error = new Error('Task execution timeout');
        error.code = 'A2A_1003';
        throw error;
      }
      
      // Simulate successful execution
      return {
        status: 'completed',
        result: { processed: true }
      };
    }, req);
  }

  validateAgentData(data) {
    const required = ['name', 'type', 'status'];
    const missing = required.filter(field => !data[field]);
    
    if (missing.length > 0) {
      const error = new Error(`Missing required fields: ${missing.join(', ')}`);
      error.code = 'A2A_3001';
      error.details = missing.map(field => ({
        field,
        message: `${field} is required`
      }));
      throw error;
    }
  }
}

/**
 * Example: Workflow Service Implementation
 */
class WorkflowService extends A2ABaseService {
  async init() {
    await super.init();
    
    // Override error handler for workflow-specific handling
    this.errorHandler = ErrorHandlerFactory.createWorkflowErrorHandler(
      process.env.NODE_ENV || 'development'
    );
    this.errorHandler.registerHandlers(this);
    
    // Workflow handlers
    this.on('READ', 'Workflows', this.onReadWorkflows);
    this.on('executeWorkflow', this.onExecuteWorkflow);
    this.on('validateWorkflow', this.onValidateWorkflow);
    
    return this;
  }

  async onReadWorkflows(req) {
    return this.executeWithErrorHandling(async () => {
      const workflows = await cds.run(req.query);
      return workflows;
    }, req);
  }

  async onExecuteWorkflow(req) {
    return this.executeWithErrorHandling(async () => {
      const { workflowId, parameters } = req.data;
      
      // Get workflow definition
      const workflow = await cds.run(
        SELECT.one.from('Workflows').where({ ID: workflowId })
      );
      
      if (!workflow) {
        const error = new Error('Workflow not found');
        error.code = 'A2A_4001';
        throw error;
      }
      
      // Validate workflow state
      if (workflow.status !== 'active') {
        const error = new Error('Workflow is not active');
        error.code = 'A2A_4003';
        error.details = [{
          field: 'status',
          value: workflow.status,
          message: 'Workflow must be in active state'
        }];
        throw error;
      }
      
      // Execute workflow (simplified)
      try {
        const result = await this.executeWorkflowSteps(workflow, parameters);
        return result;
      } catch (error) {
        error.code = error.code || 'A2A_4002';
        throw error;
      }
    }, req);
  }

  async onValidateWorkflow(req) {
    return this.executeWithErrorHandling(async () => {
      const { workflowDefinition } = req.data;
      
      // Validate workflow structure
      const validationErrors = [];
      
      if (!workflowDefinition.name) {
        validationErrors.push({
          field: 'name',
          message: 'Workflow name is required'
        });
      }
      
      if (!workflowDefinition.steps || workflowDefinition.steps.length === 0) {
        validationErrors.push({
          field: 'steps',
          message: 'Workflow must have at least one step'
        });
      }
      
      if (validationErrors.length > 0) {
        const error = new Error('Workflow validation failed');
        error.code = 'A2A_4006';
        error.details = validationErrors;
        throw error;
      }
      
      return { valid: true };
    }, req);
  }

  async executeWorkflowSteps(workflow, parameters) {
    // Simplified workflow execution
    const results = [];
    
    for (const step of workflow.steps) {
      try {
        const stepResult = await this.executeStep(step, parameters);
        results.push(stepResult);
      } catch (error) {
        // Handle step failure
        throw new Error(`Step ${step.name} failed: ${error.message}`);
      }
    }
    
    return {
      workflowId: workflow.ID,
      status: 'completed',
      results
    };
  }

  async executeStep(step, parameters) {
    // Simulate step execution
    return {
      stepId: step.ID,
      status: 'completed',
      output: {}
    };
  }
}

/**
 * Service registration helper
 */
function registerA2AServices() {
  // Register services with error handling
  cds.serve('AgentService').with(AgentService);
  cds.serve('WorkflowService').with(WorkflowService);
  
  // Global error handling for uncaught errors
  process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    // Log to monitoring system
  });
  
  process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    // Log to monitoring system
  });
}

module.exports = {
  A2ABaseService,
  AgentService,
  WorkflowService,
  registerA2AServices
};
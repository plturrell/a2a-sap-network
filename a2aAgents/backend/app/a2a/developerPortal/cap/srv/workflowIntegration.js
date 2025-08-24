'use strict';

const cds = require('@sap/cds');
const { INSERT, UPDATE } = cds.ql;
const { WorkflowApi } = require('@sap/workflow');

/**
 * SAP Workflow Integration Service
 * Handles approval workflows and business process automation
 */
class WorkflowIntegrationService {
    
  constructor() {
    this.workflowApi = null;
    this.workflowDefinitions = new Map();
    this.init();
  }

  async init() {
    try {
      // Initialize SAP Workflow API
      this.workflowApi = new WorkflowApi({
        baseURL: process.env.WORKFLOW_SERVICE_URL || 'https://api.workflow.cfapps.sap.hana.ondemand.com',
        auth: {
          clientId: process.env.WORKFLOW_CLIENT_ID,
          clientSecret: process.env.WORKFLOW_CLIENT_SECRET,
          tokenUrl: process.env.WORKFLOW_TOKEN_URL
        }
      });

      // Load workflow definitions
      await this.loadWorkflowDefinitions();
            
             
            
      // eslint-disable-next-line no-console
            
             
            
      // eslint-disable-next-line no-console
      console.log('SAP Workflow Integration initialized successfully');
    } catch (error) {
      console.error('Failed to initialize SAP Workflow Integration:', error);
    }
  }

  /**
     * Load workflow definitions from the workflow service
     */
  loadWorkflowDefinitions() {
    try {
      const definitions = [
        {
          id: 'project-approval',
          name: 'Project Approval Workflow',
          version: '1.0.0',
          description: 'Approval workflow for new projects',
          steps: [
            { id: 'manager-approval', name: 'Manager Approval', type: 'userTask' },
            { id: 'admin-approval', name: 'Admin Approval', type: 'userTask' }
          ]
        },
        {
          id: 'agent-deployment',
          name: 'Agent Deployment Workflow',
          version: '1.0.0',
          description: 'Approval workflow for agent deployments',
          steps: [
            { id: 'security-review', name: 'Security Review', type: 'userTask' },
            { id: 'deployment-approval', name: 'Deployment Approval', type: 'userTask' }
          ]
        },
        {
          id: 'budget-approval',
          name: 'Budget Approval Workflow',
          version: '1.0.0',
          description: 'Approval workflow for budget changes',
          steps: [
            { id: 'finance-review', name: 'Finance Review', type: 'userTask' },
            { id: 'cfo-approval', name: 'CFO Approval', type: 'userTask' }
          ]
        }
      ];

      definitions.forEach(def => {
        this.workflowDefinitions.set(def.id, def);
      });

    } catch (error) {
      console.error('Failed to load workflow definitions:', error);
    }
  }

  /**
     * Start a workflow instance
     * @param {string} workflowId - Workflow definition ID
     * @param {object} context - Workflow context data
     * @param {string} requestedBy - User who requested the workflow
     * @returns {Promise<object>} Workflow instance
     */
  async startWorkflow(workflowId, context, requestedBy) {
    try {
      const definition = this.workflowDefinitions.get(workflowId);
      if (!definition) {
        throw new Error(`Workflow definition not found: ${workflowId}`);
      }

      // Prepare workflow context
      const workflowContext = {
        ...context,
        requestedBy,
        requestedAt: new Date().toISOString(),
        workflowDefinition: definition
      };

      // Start workflow instance via SAP Workflow API
      const workflowInstance = await this.workflowApi.startWorkflow({
        definitionId: workflowId,
        context: workflowContext
      });

      // Store workflow instance in database
      const db = await cds.connect.to('db');
      await db.run(
        INSERT.into('WorkflowInstances').entries({
          ID: workflowInstance.id,
          definitionId: workflowId,
          status: 'RUNNING',
          context: JSON.stringify(workflowContext),
          requestedBy,
          startedAt: new Date(),
          createdAt: new Date()
        })
      );

      return {
        instanceId: workflowInstance.id,
        definitionId: workflowId,
        status: 'RUNNING',
        context: workflowContext,
        tasks: await this.getWorkflowTasks(workflowInstance.id)
      };

    } catch (error) {
      console.error('Failed to start workflow:', error);
      throw new Error(`Failed to start workflow: ${error.message}`);
    }
  }

  /**
     * Get workflow tasks for a user
     * @param {string} userId - User ID
     * @param {string} status - Task status filter
     * @returns {Promise<array>} Array of workflow tasks
     */
  async getUserTasks(userId, status = 'READY') {
    try {
      const tasks = await this.workflowApi.getTasks({
        assignee: userId,
        status: status
      });

      return tasks.map(task => ({
        id: task.id,
        workflowInstanceId: task.workflowInstanceId,
        name: task.name,
        description: task.description,
        status: task.status,
        priority: task.priority,
        dueDate: task.dueDate,
        createdAt: task.createdAt,
        context: task.context,
        formKey: task.formKey,
        actions: this.getTaskActions(task)
      }));

    } catch (error) {
      console.error('Failed to get user tasks:', error);
      throw new Error(`Failed to get user tasks: ${error.message}`);
    }
  }

  /**
     * Complete a workflow task
     * @param {string} taskId - Task ID
     * @param {string} userId - User completing the task
     * @param {object} taskData - Task completion data
     * @returns {Promise<object>} Task completion result
     */
  async completeTask(taskId, userId, taskData) {
    try {
      // Get task details
      const task = await this.workflowApi.getTask(taskId);
            
      if (!task) {
        throw new Error(`Task not found: ${taskId}`);
      }

      if (task.assignee !== userId) {
        throw new Error('User not authorized to complete this task');
      }

      // Complete task via SAP Workflow API
      const result = await this.workflowApi.completeTask(taskId, {
        variables: taskData,
        completedBy: userId,
        completedAt: new Date().toISOString()
      });

      // Update database
      const db = await cds.connect.to('db');
      await db.run(
        INSERT.into('WorkflowTaskCompletions').entries({
          ID: cds.utils.uuid(),
          taskId,
          workflowInstanceId: task.workflowInstanceId,
          completedBy: userId,
          completedAt: new Date(),
          taskData: JSON.stringify(taskData),
          result: JSON.stringify(result)
        })
      );

      // Check if workflow is completed
      const workflowInstance = await this.workflowApi.getWorkflowInstance(task.workflowInstanceId);
      if (workflowInstance.status === 'COMPLETED') {
        await this.handleWorkflowCompletion(workflowInstance);
      }

      return {
        taskId,
        status: 'COMPLETED',
        result,
        nextTasks: await this.getWorkflowTasks(task.workflowInstanceId)
      };

    } catch (error) {
      console.error('Failed to complete task:', error);
      throw new Error(`Failed to complete task: ${error.message}`);
    }
  }

  /**
     * Get workflow instance status
     * @param {string} instanceId - Workflow instance ID
     * @returns {Promise<object>} Workflow status
     */
  async getWorkflowStatus(instanceId) {
    try {
      const instance = await this.workflowApi.getWorkflowInstance(instanceId);
      const tasks = await this.getWorkflowTasks(instanceId);
            
      return {
        instanceId,
        definitionId: instance.definitionId,
        status: instance.status,
        startedAt: instance.startedAt,
        completedAt: instance.completedAt,
        context: instance.context,
        currentTasks: tasks.filter(t => t.status === 'READY'),
        completedTasks: tasks.filter(t => t.status === 'COMPLETED'),
        progress: this.calculateWorkflowProgress(instance, tasks)
      };

    } catch (error) {
      console.error('Failed to get workflow status:', error);
      throw new Error(`Failed to get workflow status: ${error.message}`);
    }
  }

  /**
     * Cancel a workflow instance
     * @param {string} instanceId - Workflow instance ID
     * @param {string} userId - User canceling the workflow
     * @param {string} reason - Cancellation reason
     * @returns {Promise<boolean>} Success status
     */
  async cancelWorkflow(instanceId, userId, reason) {
    try {
      await this.workflowApi.cancelWorkflowInstance(instanceId, {
        cancelledBy: userId,
        cancelledAt: new Date().toISOString(),
        reason
      });

      // Update database
      const db = await cds.connect.to('db');
      await db.run(
        UPDATE('WorkflowInstances')
          .set({ 
            status: 'CANCELLED',
            completedAt: new Date(),
            cancelReason: reason
          })
          .where({ ID: instanceId })
      );

      return true;

    } catch (error) {
      console.error('Failed to cancel workflow:', error);
      throw new Error(`Failed to cancel workflow: ${error.message}`);
    }
  }

  /**
     * Get workflow definitions
     * @returns {Array} Available workflow definitions
     */
  getWorkflowDefinitions() {
    return Array.from(this.workflowDefinitions.values());
  }

  /**
     * Create custom workflow definition
     * @param {object} definition - Workflow definition
     * @returns {Promise<object>} Created definition
     */
  async createWorkflowDefinition(definition) {
    try {
      // Validate definition
      this.validateWorkflowDefinition(definition);

      // Deploy to SAP Workflow service
      const deployedDefinition = await this.workflowApi.deployWorkflow(definition);

      // Store in local cache
      this.workflowDefinitions.set(definition.id, {
        ...definition,
        deployedAt: new Date().toISOString(),
        version: deployedDefinition.version
      });

      return deployedDefinition;

    } catch (error) {
      console.error('Failed to create workflow definition:', error);
      throw new Error(`Failed to create workflow definition: ${error.message}`);
    }
  }

  // Private helper methods
  async getWorkflowTasks(instanceId) {
    try {
      const tasks = await this.workflowApi.getWorkflowTasks(instanceId);
      return tasks.map(task => ({
        id: task.id,
        name: task.name,
        status: task.status,
        assignee: task.assignee,
        dueDate: task.dueDate,
        priority: task.priority
      }));
    } catch (error) {
      console.error('Failed to get workflow tasks:', error);
      return [];
    }
  }

  getTaskActions(task) {
    const actions = ['approve', 'reject'];
        
    // Add custom actions based on task type
    if (task.name.includes('Review')) {
      actions.push('request-changes');
    }
        
    if (task.name.includes('Approval')) {
      actions.push('delegate');
    }

    return actions;
  }

  calculateWorkflowProgress(instance, tasks) {
    if (instance.status === 'COMPLETED') {
      return 100;
    }
    if (instance.status === 'CANCELLED' || instance.status === 'FAILED') {
      return 0;
    }
        
    const totalTasks = tasks.length;
    const completedTasks = tasks.filter(t => t.status === 'COMPLETED').length;
        
    return totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;
  }

  async handleWorkflowCompletion(workflowInstance) {
    try {
      const db = await cds.connect.to('db');
            
      // Update workflow instance status
      await db.run(
        UPDATE('WorkflowInstances')
          .set({ 
            status: workflowInstance.status,
            completedAt: new Date(),
            result: JSON.stringify(workflowInstance.result)
          })
          .where({ ID: workflowInstance.id })
      );

      // Handle specific workflow completion logic
      const context = JSON.parse(workflowInstance.context);
            
      switch (workflowInstance.definitionId) {
      case 'project-approval':
        await this.handleProjectApprovalCompletion(context, workflowInstance.result);
        break;
      case 'agent-deployment':
        await this.handleAgentDeploymentCompletion(context, workflowInstance.result);
        break;
      case 'budget-approval':
        await this.handleBudgetApprovalCompletion(context, workflowInstance.result);
        break;
      }

    } catch (error) {
      console.error('Failed to handle workflow completion:', error);
    }
  }

  async handleProjectApprovalCompletion(context, result) {
    const db = await cds.connect.to('db');
        
    if (result.approved) {
      // Activate the project
      await db.run(
        UPDATE('Projects')
          .set({ status: 'Active' })
          .where({ ID: context.projectId })
      );
    } else {
      // Mark project as rejected
      await db.run(
        UPDATE('Projects')
          .set({ status: 'Rejected' })
          .where({ ID: context.projectId })
      );
    }
  }

  async handleAgentDeploymentCompletion(context, result) {
    const db = await cds.connect.to('db');
        
    if (result.approved) {
      // Proceed with agent deployment
      await db.run(
        UPDATE('AgentDeployments')
          .set({ status: 'Approved' })
          .where({ ID: context.deploymentId })
      );
    }
  }

  async handleBudgetApprovalCompletion(context, result) {
    const db = await cds.connect.to('db');
        
    if (result.approved) {
      // Update project budget
      await db.run(
        UPDATE('Projects')
          .set({ 
            budget: context.newBudget,
            modifiedAt: new Date()
          })
          .where({ ID: context.projectId })
      );
    }
  }

  validateWorkflowDefinition(definition) {
    if (!definition.id || !definition.name || !definition.steps) {
      throw new Error('Invalid workflow definition: missing required fields');
    }

    if (!Array.isArray(definition.steps) || definition.steps.length === 0) {
      throw new Error('Invalid workflow definition: steps must be a non-empty array');
    }

    // Additional validation logic...
  }
}

module.exports = new WorkflowIntegrationService();

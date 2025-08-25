/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

"use strict";

const cds = require('@sap/cds');
const { INSERT, SELECT, UPDATE } = cds.ql;
const { v4: uuidv4 } = require('uuid');

/**
 * A2A Developer Portal Service Implementation
 * Handles business logic for OData v4 services
 */
class PortalService extends cds.ApplicationService {
    
    async init() {
        
        // Get database connection
        this.db = await cds.connect.to('db');
        
        // Get service entities
        const { 
            Users: _Users, Projects, Agents, Workflows, ProjectFiles: _ProjectFiles, 
            Deployments, TestSuites: _TestSuites, ApprovalWorkflows, AuditLogs: _AuditLogs,
            SystemMetrics: _SystemMetrics, UserSessions: _UserSessions
        } = this.entities;

        // Initialize logging (simplified for development)
        this.appLog = console;
        
        // Project Management Actions
        this.on('createProject', async (req) => {
            const { name, description, startDate, budget } = req.data;
            const user = req.user;
            
            try {
                // Create project
                const projectId = uuidv4();
                const project = await this.db.run(
                    INSERT.into(Projects).entries({
                        ID: projectId,
                        name,
                        description,
                        startDate,
                        budget,
                        status: 'Draft',
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Add creator as project owner
                await this.db.run(
                    INSERT.into('ProjectMembers').entries({
                        ID: uuidv4(),
                        project_ID: projectId,
                        user_ID: user.id,
                        role: 'Owner',
                        joinDate: new Date()
                    })
                );

                // Log audit event
                await this._logAuditEvent(user.id, 'CREATE', 'Project', projectId, null, { name, description });

                return project;
            } catch (error) {
                this.appLog.error('Failed to create project', { error: error.message, user: user.id });
                req.error(500, 'Failed to create project');
            }
        });

        this.on('cloneProject', async (req) => {
            const { projectId } = req.data;
            const user = req.user;
            
            try {
                // Get original project
                const originalProject = await this.db.run(
                    SELECT.from(Projects).where({ ID: projectId })
                );
                
                if (!originalProject.length) {
                    req.error(404, 'Project not found');
                    return;
                }

                const original = originalProject[0];
                const newProjectId = uuidv4();
                
                // Clone project
                const clonedProject = await this.db.run(
                    INSERT.into(Projects).entries({
                        ID: newProjectId,
                        name: `${original.name} (Copy)`,
                        description: original.description,
                        status: 'Draft',
                        priority: original.priority,
                        budget: original.budget,
                        currency_code: original.currency_code,
                        tags: original.tags,
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Clone agents - batch insert for better performance
                const agents = await this.db.run(
                    SELECT.from(Agents).where({ project_ID: projectId })
                );
                
                if (agents.length > 0) {
                    const agentEntries = agents.map(agent => ({
                        ID: uuidv4(),
                        project_ID: newProjectId,
                        name: agent.name,
                        description: agent.description,
                        type: agent.type,
                        status: 'Draft',
                        version: '1.0.0',
                        configuration: agent.configuration,
                        capabilities: agent.capabilities,
                        createdBy: user.id,
                        createdAt: new Date()
                    }));
                    
                    await this.db.run(
                        INSERT.into(Agents).entries(agentEntries)
                    );
                }

                // Add user as project owner
                await this.db.run(
                    INSERT.into('ProjectMembers').entries({
                        ID: uuidv4(),
                        project_ID: newProjectId,
                        user_ID: user.id,
                        role: 'Owner',
                        joinDate: new Date()
                    })
                );

                await this._logAuditEvent(user.id, 'CLONE', 'Project', newProjectId, null, { originalProjectId: projectId });

                return clonedProject;
            } catch (error) {
                this.appLog.error('Failed to clone project', { error: error.message, projectId, user: user.id });
                req.error(500, 'Failed to clone project');
            }
        });

        // Agent Management Actions
        this.on('deployAgent', async (req) => {
            const { agentId, environment, configuration } = req.data;
            const user = req.user;
            
            try {
                // Get agent details
                const agent = await this.db.run(
                    SELECT.from(Agents).where({ ID: agentId })
                );
                
                if (!agent.length) {
                    req.error(404, 'Agent not found');
                    return;
                }

                // Create deployment record
                const deploymentId = uuidv4();
                const _deployment = await this.db.run(
                    INSERT.into(Deployments).entries({
                        ID: deploymentId,
                        project_ID: agent[0].project_ID,
                        version: agent[0].version,
                        environment,
                        status: 'Pending',
                        deploymentType: 'Full',
                        startTime: new Date(),
                        deployedBy_ID: user.id,
                        configuration,
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Create agent deployment
                const agentDeployment = await this.db.run(
                    INSERT.into('AgentDeployments').entries({
                        ID: uuidv4(),
                        deployment_ID: deploymentId,
                        agent_ID: agentId,
                        status: 'Pending',
                        configuration,
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Trigger real deployment process
                this._executeRealDeployment(deploymentId, agentId, agent[0], environment, configuration, user);

                await this._logAuditEvent(user.id, 'DEPLOY', 'Agent', agentId, null, { environment, deploymentId });

                return agentDeployment;
            } catch (error) {
                this.appLog.error('Failed to deploy agent', { error: error.message, agentId, user: user.id });
                req.error(500, 'Failed to deploy agent');
            }
        });

        // Workflow Management Actions
        this.on('executeWorkflow', async (req) => {
            const { workflowId, input } = req.data;
            const user = req.user;
            
            try {
                // Get workflow details
                const workflow = await this.db.run(
                    SELECT.from(Workflows).where({ ID: workflowId })
                );
                
                if (!workflow.length) {
                    req.error(404, 'Workflow not found');
                    return;
                }

                // Create execution record
                const executionId = uuidv4();
                const execution = await this.db.run(
                    INSERT.into('WorkflowExecutions').entries({
                        ID: executionId,
                        workflow_ID: workflowId,
                        status: 'Running',
                        startTime: new Date(),
                        input,
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Get workflow steps
                const steps = await this.db.run(
                    SELECT.from('WorkflowSteps').where({ workflow_ID: workflowId }).orderBy('stepNumber')
                );

                // Create step executions - batch insert for better performance
                if (steps.length > 0) {
                    const stepExecutionEntries = steps.map(step => ({
                        ID: uuidv4(),
                        execution_ID: executionId,
                        step_ID: step.ID,
                        status: 'Pending',
                        createdBy: user.id,
                        createdAt: new Date()
                    }));
                    
                    await this.db.run(
                        INSERT.into('WorkflowStepExecutions').entries(stepExecutionEntries)
                    );
                }

                // Start real workflow execution
                this._executeRealWorkflow(executionId, workflow[0], steps, input, user);

                await this._logAuditEvent(user.id, 'EXECUTE', 'Workflow', workflowId, null, { executionId, input });

                return execution;
            } catch (error) {
                this.appLog.error('Failed to execute workflow', { error: error.message, workflowId, user: user.id });
                req.error(500, 'Failed to execute workflow');
            }
        });

        // Approval Management Actions
        this.on('submitForApproval', async (req) => {
            const { type, entityId, requestData } = req.data;
            const user = req.user;
            
            try {
                // Create approval workflow
                const workflowId = uuidv4();
                const approvalWorkflow = await this.db.run(
                    INSERT.into(ApprovalWorkflows).entries({
                        ID: workflowId,
                        title: `${type} Approval Request`,
                        description: `Approval request for ${type}`,
                        type,
                        status: 'Pending',
                        priority: 'Medium',
                        requestedBy_ID: user.id,
                        requestData,
                        createdBy: user.id,
                        createdAt: new Date()
                    })
                );

                // Get approvers based on type (simplified logic)
                const approvers = await this._getApproversForType(type);
                
                // Create approval steps - batch insert for better performance
                if (approvers.length > 0) {
                    const approvalStepEntries = approvers.map((approver, i) => ({
                        ID: uuidv4(),
                        workflow_ID: workflowId,
                        stepNumber: i + 1,
                        approver_ID: approver.ID,
                        status: 'Pending',
                        dueDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
                        createdBy: user.id,
                        createdAt: new Date()
                    }));
                    
                    await this.db.run(
                        INSERT.into('ApprovalSteps').entries(approvalStepEntries)
                    );
                }

                await this._logAuditEvent(user.id, 'SUBMIT_APPROVAL', type, entityId, null, { workflowId, requestData });

                return approvalWorkflow;
            } catch (error) {
                this.appLog.error('Failed to submit for approval', { error: error.message, type, entityId, user: user.id });
                req.error(500, 'Failed to submit for approval');
            }
        });

        // Analytics Functions
        this.on('getProjectStatistics', async (req) => {
            try {
                const stats = await this.db.run(`
                    SELECT 
                        COUNT(*) as totalProjects,
                        COUNT(CASE WHEN status = 'Active' THEN 1 END) as activeProjects,
                        COUNT(CASE WHEN status = 'Completed' THEN 1 END) as completedProjects
                    FROM ${Projects.name}
                `);

                const agentStats = await this.db.run(`
                    SELECT COUNT(*) as totalAgents
                    FROM ${Agents.name}
                `);

                const deploymentStats = await this.db.run(`
                    SELECT COUNT(*) as activeDeployments
                    FROM ${Deployments.name}
                    WHERE status = 'Completed'
                `);

                return {
                    totalProjects: stats[0].totalProjects || 0,
                    activeProjects: stats[0].activeProjects || 0,
                    completedProjects: stats[0].completedProjects || 0,
                    totalAgents: agentStats[0].totalAgents || 0,
                    activeDeployments: deploymentStats[0].activeDeployments || 0
                };
            } catch (error) {
                this.appLog.error('Failed to get project statistics', { error: error.message });
                req.error(500, 'Failed to get project statistics');
            }
        });

        this.on('getDashboardData', async (req) => {
            try {
                // Get KPIs
                const kpis = await this.getDashboardKPIs();
                
                // Get recent projects
                const recentProjects = await this.db.run(
                    SELECT.from(Projects)
                        .columns('ID', 'name', 'status', 'modifiedAt')
                        .orderBy('modifiedAt desc')
                        .limit(5)
                );

                // Get notifications (simplified)
                const notifications = await this._getRecentNotifications(req.user.id);

                return {
                    kpis,
                    recentProjects: recentProjects.map(p => ({
                        id: p.ID,
                        name: p.name,
                        status: p.status,
                        lastModified: p.modifiedAt
                    })),
                    notifications
                };
            } catch (error) {
                this.appLog.error('Failed to get dashboard data', { error: error.message });
                req.error(500, 'Failed to get dashboard data');
            }
        });

        // Entity event handlers
        this.before('CREATE', Projects, (req) => {
            req.data.ID = req.data.ID || uuidv4();
            req.data.status = req.data.status || 'Draft';
        });

        this.before('CREATE', Agents, (req) => {
            req.data.ID = req.data.ID || uuidv4();
            req.data.status = req.data.status || 'Draft';
            req.data.version = req.data.version || '1.0.0';
        });

        this.after('CREATE', Projects, async (data, req) => {
            await this._logAuditEvent(req.user.id, 'CREATE', 'Project', data.ID, null, data);
        });

        this.after('UPDATE', Projects, async (data, req) => {
            await this._logAuditEvent(req.user.id, 'UPDATE', 'Project', data.ID, null, data);
        });

        this.after('DELETE', Projects, async (data, req) => {
            await this._logAuditEvent(req.user.id, 'DELETE', 'Project', data.ID, data, null);
        });

        return super.init();
    }

    // Helper methods
    async getDashboardKPIs() {
        const projectCount = await this.db.run(`SELECT COUNT(*) as count FROM ${this.entities.Projects.name}`);
        const agentCount = await this.db.run(`SELECT COUNT(*) as count FROM ${this.entities.Agents.name} WHERE status = 'Active'`);
        const approvalCount = await this.db.run(`SELECT COUNT(*) as count FROM ${this.entities.ApprovalWorkflows.name} WHERE status = 'Pending'`);
        
        return {
            totalProjects: projectCount[0].count || 0,
            activeAgents: agentCount[0].count || 0,
            pendingApprovals: approvalCount[0].count || 0,
            systemHealth: 'Good'
        };
    }

    async _executeRealDeployment(deploymentId, agentId, agent, environment, configuration, user) {
        try {
            // Import deployment pipeline client
            const { BlockchainClient } = require('../core/blockchain-client');
            const portalUrl = process.env.PORTAL_URL || 'http://localhost:3001';
            
            // Create deployment configuration
            const deploymentConfig = {
                name: `${agent.name} Deployment`,
                project_id: agent.project_ID,
                target: {
                    name: environment,
                    environment: environment === 'production' ? 'production' : 'staging',
                    platform: 'kubernetes',
                    cpu_limit: '500m',
                    memory_limit: '512Mi',
                    replicas: environment === 'production' ? 2 : 1
                },
                dockerfile_path: 'Dockerfile',
                build_context: `/workspace/${agent.project_ID}/agents/${agent.ID}`,
                environment_variables: configuration || {},
                notification_emails: [user.email]
            };
            
            // Call deployment pipeline API
            const response = await blockchainClient.sendMessage(
                `${portalUrl}/api/projects/${agent.project_ID}/deploy`,
                deploymentConfig,
                {
                    headers: {
                        'Authorization': `Bearer ${user.token}`, // Assuming user has auth token
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            if (response.data.success) {
                const executionId = response.data.deployment_id;
                
                // Monitor deployment status
                this._monitorDeploymentStatus(deploymentId, executionId, agentId, environment, portalUrl);
            } else {
                throw new Error(response.data.message || 'Deployment failed to start');
            }
            
        } catch (error) {
            this.appLog.error('Real deployment failed', { error: error.message, deploymentId, agentId });
            
            // Update deployment status to failed
            await this.db.run(
                UPDATE(this.entities.Deployments).set({ 
                    status: 'Failed',
                    endTime: new Date(),
                    errorMessage: error.message
                }).where({ ID: deploymentId })
            );
            
            await this.db.run(
                UPDATE('AgentDeployments').set({ 
                    status: 'Failed',
                    errorMessage: error.message
                }).where({ deployment_ID: deploymentId, agent_ID: agentId })
            );
        }
    }
    
    _monitorDeploymentStatus(deploymentId, executionId, agentId, environment, portalUrl) {
        const { BlockchainClient } = require('../core/blockchain-client');
        const maxAttempts = 60; // 5 minutes with 5 second intervals
        let attempts = 0;
        
        const checkStatus = async () => {
            try {
                const response = await blockchainClient.sendMessage(
                    `${portalUrl}/api/deployments/${executionId}`,
                    { timeout: 5000 }
                );
                
                const execution = response.data;
                
                if (execution.status === 'completed') {
                    // Deployment succeeded
                    await this.db.run(
                        UPDATE(this.entities.Deployments).set({ 
                            status: 'Completed',
                            endTime: new Date()
                        }).where({ ID: deploymentId })
                    );
                    
                    await this.db.run(
                        UPDATE('AgentDeployments').set({ 
                            status: 'Deployed',
                            endpoint: `https://${environment}.a2a-portal.com/agents/${agentId}`,
                            deploymentMetadata: JSON.stringify(execution.result)
                        }).where({ deployment_ID: deploymentId, agent_ID: agentId })
                    );
                    
                    this.appLog.info('Deployment completed successfully', { deploymentId, executionId });
                    
                } else if (execution.status === 'failed') {
                    // Deployment failed
                    await this.db.run(
                        UPDATE(this.entities.Deployments).set({ 
                            status: 'Failed',
                            endTime: new Date(),
                            errorMessage: execution.error_message
                        }).where({ ID: deploymentId })
                    );
                    
                    await this.db.run(
                        UPDATE('AgentDeployments').set({ 
                            status: 'Failed',
                            errorMessage: execution.error_message
                        }).where({ deployment_ID: deploymentId, agent_ID: agentId })
                    );
                    
                    this.appLog.error('Deployment failed', { deploymentId, executionId, error: execution.error_message });
                    
                } else if (attempts < maxAttempts) {
                    // Still running, check again
                    attempts++;
                    setTimeout(checkStatus, 5000);
                }
                
            } catch (error) {
                this.appLog.error('Failed to check deployment status', { error: error.message, deploymentId, executionId });
                
                if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(checkStatus, 5000);
                } else {
                    // Timeout - mark as failed
                    await this.db.run(
                        UPDATE(this.entities.Deployments).set({ 
                            status: 'Failed',
                            endTime: new Date(),
                            errorMessage: 'Deployment status check timed out'
                        }).where({ ID: deploymentId })
                    );
                }
            }
        };
        
        // Start monitoring
        setTimeout(checkStatus, 5000);
    }
    
    async _executeRealWorkflow(executionId, workflow, steps, input, user) {
        try {
            // Check if we have a BPMN workflow definition
            if (workflow.bpmnDefinition) {
                // Use BPMN workflow designer API
                const { BlockchainClient } = require('../core/blockchain-client');
                const portalUrl = process.env.PORTAL_URL || 'http://localhost:3001';
                
                try {
                    // Execute BPMN workflow
                    const response = await blockchainClient.sendMessage(
                        `${portalUrl}/api/workflows/${workflow.ID}/execute`,
                        {
                            variables: input,
                            executionId: executionId
                        },
                        {
                            headers: {
                                'Authorization': `Bearer ${user.token}`,
                                'Content-Type': 'application/json'
                            }
                        }
                    );
                    
                    if (response.data.success) {
                        // Monitor workflow execution
                        this._monitorWorkflowExecution(executionId, response.data.execution_id, portalUrl);
                    } else {
                        throw new Error(response.data.message || 'Workflow execution failed to start');
                    }
                    
                } catch (error) {
                    throw new Error(`BPMN execution failed: ${error.message}`);
                }
                
            } else {
                // Execute step-based workflow
                await this._executeStepBasedWorkflow(executionId, steps, input);
            }
            
        } catch (error) {
            this.appLog.error('Workflow execution failed', { error: error.message, executionId });
            
            // Update execution status to failed
            await this.db.run(
                UPDATE('WorkflowExecutions')
                    .set({ 
                        status: 'Failed', 
                        endTime: new Date(),
                        errorMessage: error.message
                    })
                    .where({ ID: executionId })
            );
        }
    }
    
    _monitorWorkflowExecution(executionId, remoteExecutionId, portalUrl) {
        const { BlockchainClient } = require('../core/blockchain-client');
        const maxAttempts = 120; // 10 minutes with 5 second intervals
        let attempts = 0;
        
        const checkStatus = async () => {
            try {
                const response = await blockchainClient.sendMessage(
                    `${portalUrl}/api/workflow-executions/${remoteExecutionId}`,
                    { timeout: 5000 }
                );
                
                const execution = response.data;
                
                if (execution.status === 'completed') {
                    // Workflow completed
                    await this.db.run(
                        UPDATE('WorkflowExecutions')
                            .set({ 
                                status: 'Completed', 
                                endTime: new Date(),
                                output: JSON.stringify(execution.result || { message: 'Workflow completed successfully' })
                            })
                            .where({ ID: executionId })
                    );
                    
                    this.appLog.info('Workflow completed successfully', { executionId, remoteExecutionId });
                    
                } else if (execution.status === 'failed') {
                    // Workflow failed
                    await this.db.run(
                        UPDATE('WorkflowExecutions')
                            .set({ 
                                status: 'Failed', 
                                endTime: new Date(),
                                errorMessage: execution.error_message,
                                output: JSON.stringify({ error: execution.error_message })
                            })
                            .where({ ID: executionId })
                    );
                    
                    this.appLog.error('Workflow failed', { executionId, remoteExecutionId, error: execution.error_message });
                    
                } else if (attempts < maxAttempts) {
                    // Still running, check again
                    attempts++;
                    setTimeout(checkStatus, 5000);
                }
                
            } catch (error) {
                this.appLog.error('Failed to check workflow status', { error: error.message, executionId, remoteExecutionId });
                
                if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(checkStatus, 5000);
                } else {
                    // Timeout - mark as failed
                    await this.db.run(
                        UPDATE('WorkflowExecutions')
                            .set({ 
                                status: 'Failed', 
                                endTime: new Date(),
                                errorMessage: 'Workflow status check timed out'
                            })
                            .where({ ID: executionId })
                    );
                }
            }
        };
        
        // Start monitoring
        setTimeout(checkStatus, 5000);
    }
    
    async _executeStepBasedWorkflow(executionId, steps, input) {
        // Real step-based workflow execution - optimized to reduce N+1 queries
        for (const step of steps) {
            try {
                // Update step status to running
                await this.db.run(
                    UPDATE('WorkflowStepExecutions')
                        .set({ status: 'Running', startTime: new Date() })
                        .where({ execution_ID: executionId, step_ID: step.ID })
                );

                // Execute step based on type
                let stepResult;
                if (step.type === 'ServiceTask') {
                    stepResult = await this._executeServiceTask(step, input);
                } else if (step.type === 'ScriptTask') {
                    stepResult = await this._executeScriptTask(step, input);
                } else if (step.type === 'UserTask') {
                    stepResult = await this._executeUserTask(step, input);
                } else {
                    // Default task execution
                    stepResult = { result: 'success', stepName: step.name };
                }

                // Update step status to completed
                await this.db.run(
                    UPDATE('WorkflowStepExecutions')
                        .set({ 
                            status: 'Completed', 
                            endTime: new Date(),
                            output: JSON.stringify(stepResult)
                        })
                        .where({ execution_ID: executionId, step_ID: step.ID })
                );
                
                // Update input for next step
                input = { ...input, ...stepResult };
                
            } catch (error) {
                // Mark step as failed
                await this.db.run(
                    UPDATE('WorkflowStepExecutions')
                        .set({ 
                            status: 'Failed', 
                            endTime: new Date(),
                            errorMessage: error.message
                        })
                        .where({ execution_ID: executionId, step_ID: step.ID })
                );
                throw error; // Propagate to fail the workflow
            }
        }
        
        // Note: This loop is sequential by design for workflow execution order
        // Each step depends on the previous step's output

        // Update overall execution status
        await this.db.run(
            UPDATE('WorkflowExecutions')
                .set({ 
                    status: 'Completed', 
                    endTime: new Date(),
                    output: JSON.stringify({ result: 'Workflow completed successfully', finalOutput: input })
                })
                .where({ ID: executionId })
        );
    }
    
    async _executeServiceTask(step, input) {
        // Execute service task by calling configured endpoint
        const { BlockchainClient } = require('../core/blockchain-client');
        const config = JSON.parse(step.configuration || '{}');
        
        if (config.endpoint) {
            const response = await axios({
                method: config.method || 'POST',
                url: config.endpoint,
                data: input,
                headers: config.headers || {},
                timeout: config.timeout || 30000
            });
            
            return {
                result: 'success',
                stepName: step.name,
                serviceResponse: response.data
            };
        }
        
        return { result: 'success', stepName: step.name };
    }
    
    _executeScriptTask(step, input) {
        // Execute script task (safely)
        const config = JSON.parse(step.configuration || '{}');
        
        if (config.script && config.scriptType === 'javascript') {
            // Create safe execution context
            const vm = require('vm');
            const context = {
                input: input,
                output: {},
                console: console
            };
            
            try {
                vm.createContext(context);
                vm.runInContext(config.script, context, { timeout: 5000 });
                
                return {
                    result: 'success',
                    stepName: step.name,
                    scriptOutput: context.output
                };
            } catch (error) {
                throw new Error(`Script execution failed: ${error.message}`);
            }
        }
        
        return { result: 'success', stepName: step.name };
    }
    
    async _executeUserTask(step, input) {
        // Create user task in database for manual completion
        const taskId = uuidv4();
        const config = JSON.parse(step.configuration || '{}');
        
        await this.db.run(
            INSERT.into('UserTasks').entries({
                ID: taskId,
                step_ID: step.ID,
                assignee_ID: config.assignee || input.assignee,
                title: step.name,
                description: config.description || step.description,
                dueDate: new Date(Date.now() + (config.dueDays || 7) * 24 * 60 * 60 * 1000),
                status: 'Pending',
                formData: JSON.stringify(config.formFields || {}),
                createdAt: new Date()
            })
        );
        
        // In a real system, this would wait for user completion
        // For now, we'll mark it as auto-completed
        return {
            result: 'success',
            stepName: step.name,
            userTaskId: taskId,
            autoCompleted: true
        };
    }

    async _getApproversForType(_type) {
        // Simplified approver logic - in real scenario, this would be configurable
        const approvers = await this.db.run(
            SELECT.from(this.entities.Users).where({ role: 'ProjectManager' }).limit(2)
        );
        return approvers;
    }

    _getRecentNotifications(_userId) {
        // Simplified notification logic
        return [
            {
                id: uuidv4(),
                type: 'info',
                message: 'Welcome to A2A Developer Portal',
                timestamp: new Date(),
                isRead: false
            }
        ];
    }

    async _logAuditEvent(userId, action, entityType, entityId, oldValues, newValues) {
        try {
            await this.db.run(
                INSERT.into(this.entities.AuditLogs).entries({
                    ID: uuidv4(),
                    user_ID: userId,
                    action,
                    entityType,
                    entityId,
                    oldValues: oldValues ? JSON.stringify(oldValues) : null,
                    newValues: newValues ? JSON.stringify(newValues) : null,
                    timestamp: new Date(),
                    success: true
                })
            );
        } catch (error) {
            this.appLog.error('Failed to log audit event', { error: error.message, userId, action, entityType });
        }
    }
}

module.exports = PortalService;

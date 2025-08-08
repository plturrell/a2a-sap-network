using {com.sap.a2a.developerportal as db} from '../db/schema';

/**
 * A2A Developer Portal Services
 * OData v4 services for UI5 consumption
 */

// Main Portal Service
@path: '/api/v4/portal'
service PortalService @(requires: 'authenticated-user') {

    // User Management
    @odata.draft.enabled
    @cds.redirection.target
    entity Users as projection on db.Users {
        *,
        projects : redirected to ProjectMembers
    } excluding { 
        preferences // Exclude sensitive data from main view
    };

    @readonly
    entity UserProfiles as projection on db.Users {
        ID,
        email,
        firstName,
        lastName,
        displayName,
        role,
        department,
        isActive,
        lastLogin,
        preferences
    };

    // Project Management
    @odata.draft.enabled
    entity Projects as projection on db.Projects {
        *,
        members : redirected to ProjectMembers,
        agents : redirected to Agents,
        workflows : redirected to Workflows,
        files : redirected to ProjectFiles,
        deployments : redirected to Deployments
    };

    entity ProjectMembers as projection on db.ProjectMembers;

    // Agent Management
    entity Agents as projection on db.Agents {
        *,
        project : redirected to Projects,
        workflows : redirected to WorkflowSteps,
        deployments : redirected to AgentDeployments,
        testResults : redirected to TestResults
    };

    // Workflow Management
    entity Workflows as projection on db.Workflows {
        *,
        project : redirected to Projects,
        steps : redirected to WorkflowSteps,
        executions : redirected to WorkflowExecutions
    };

    entity WorkflowSteps as projection on db.WorkflowSteps;
    entity WorkflowExecutions as projection on db.WorkflowExecutions;
    entity WorkflowStepExecutions as projection on db.WorkflowStepExecutions;

    // File Management
    entity ProjectFiles as projection on db.ProjectFiles {
        *,
        project : redirected to Projects
    } excluding {
        content // Exclude binary content from main view
    };

    // Deployment Management
    entity Deployments as projection on db.Deployments {
        *,
        project : redirected to Projects,
        deployedBy : redirected to Users,
        agentDeployments : redirected to AgentDeployments,
        logs : redirected to DeploymentLogs
    };

    entity AgentDeployments as projection on db.AgentDeployments;

    // Testing Framework
    entity TestSuites as projection on db.TestSuites {
        *,
        project : redirected to Projects,
        testCases : redirected to TestCases,
        executions : redirected to TestExecutions
    };

    entity TestCases as projection on db.TestCases;
    entity TestExecutions as projection on db.TestExecutions;
    entity TestResults as projection on db.TestResults;

    // Session Management (Admin only)
    @restrict: [
        { grant: ['READ'], to: 'Admin' },
        { grant: ['*'], to: 'Admin' }
    ]
    entity UserSessions as projection on db.UserSessions;

    // Approval Workflows
    entity ApprovalWorkflows as projection on db.ApprovalWorkflows {
        *,
        requestedBy : redirected to Users,
        approvers : redirected to ApprovalSteps,
        project : redirected to Projects,
        agent : redirected to Agents,
        deployment : redirected to Deployments
    };

    entity ApprovalSteps as projection on db.ApprovalSteps;

    // Monitoring (Read-only)
    @readonly
    entity SystemMetrics as projection on db.SystemMetrics;

    @readonly
    entity DeploymentLogs as projection on db.DeploymentLogs;

    // Audit Logging (Admin only)
    @restrict: [
        { grant: ['READ'], to: 'Admin' }
    ]
    @readonly
    entity AuditLogs as projection on db.AuditLogs;

    // Custom Actions and Functions
    
    // Project Actions
    action createProject(
        name: String,
        description: String,
        startDate: Date,
        budget: Decimal(15,2)
    ) returns Projects;

    action cloneProject(projectId: UUID) returns Projects;
    action archiveProject(projectId: UUID) returns Boolean;
    
    // Agent Actions
    action deployAgent(
        agentId: UUID,
        environment: String,
        configuration: String
    ) returns AgentDeployments;

    action testAgent(
        agentId: UUID,
        testData: String
    ) returns TestResults;

    // Workflow Actions
    action executeWorkflow(
        workflowId: UUID,
        input: String
    ) returns WorkflowExecutions;

    action pauseWorkflow(executionId: UUID) returns Boolean;
    action resumeWorkflow(executionId: UUID) returns Boolean;
    action cancelWorkflow(executionId: UUID) returns Boolean;

    // Approval Actions
    action submitForApproval(
        type: String,
        entityId: UUID,
        requestData: String
    ) returns ApprovalWorkflows;

    action approveRequest(
        workflowId: UUID,
        stepId: UUID,
        comments: String
    ) returns Boolean;

    action rejectRequest(
        workflowId: UUID,
        stepId: UUID,
        comments: String
    ) returns Boolean;

    // File Actions
    action uploadFile(
        projectId: UUID,
        fileName: String,
        content: LargeBinary
    ) returns ProjectFiles;

    action downloadFile(fileId: UUID) returns LargeBinary;

    // Analytics Functions
    function getProjectStatistics() returns {
        totalProjects: Integer;
        activeProjects: Integer;
        completedProjects: Integer;
        totalAgents: Integer;
        activeDeployments: Integer;
    };

    function getUserStatistics(userId: UUID) returns {
        projectsOwned: Integer;
        projectsContributed: Integer;
        agentsCreated: Integer;
        deploymentsCompleted: Integer;
        testsPassed: Integer;
    };

    function getSystemHealth() returns {
        status: String;
        uptime: Integer;
        activeUsers: Integer;
        runningWorkflows: Integer;
        systemLoad: Decimal(5,2);
        memoryUsage: Decimal(5,2);
    };

    // Dashboard Functions
    function getDashboardData() returns {
        kpis: {
            totalProjects: Integer;
            activeAgents: Integer;
            pendingApprovals: Integer;
            systemHealth: String;
        };
        recentProjects: array of {
            id: UUID;
            name: String;
            status: String;
            lastModified: Timestamp;
        };
        notifications: array of {
            id: UUID;
            type: String;
            message: String;
            timestamp: Timestamp;
            isRead: Boolean;
        };
    };
}

// Workflow Service (Separate for workflow-specific operations)
@path: '/api/v4/workflow'
service WorkflowService @(requires: 'authenticated-user') {
    
    entity Workflows as projection on db.Workflows;
    entity WorkflowExecutions as projection on db.WorkflowExecutions;
    entity WorkflowSteps as projection on db.WorkflowSteps;
    entity WorkflowStepExecutions as projection on db.WorkflowStepExecutions;

    // Workflow-specific actions
    action startWorkflow(
        workflowId: UUID,
        input: String
    ) returns WorkflowExecutions;

    action getWorkflowStatus(executionId: UUID) returns {
        status: String;
        progress: Integer;
        currentStep: String;
        estimatedCompletion: Timestamp;
    };

    function getWorkflowTemplates() returns array of {
        id: String;
        name: String;
        description: String;
        category: String;
        complexity: String;
    };
}

// Admin Service (Admin-only operations)
@path: '/api/v4/admin'
service AdminService @(requires: 'Admin') {
    
    entity Users as projection on db.Users;
    entity UserSessions as projection on db.UserSessions;
    entity AuditLogs as projection on db.AuditLogs;
    entity SystemMetrics as projection on db.SystemMetrics;

    // Admin actions
    action createUser(
        email: String,
        firstName: String,
        lastName: String,
        role: String
    ) returns Users;

    action deactivateUser(userId: UUID) returns Boolean;
    action resetUserPassword(userId: UUID) returns String;

    action cleanupSessions() returns Integer;
    action generateSystemReport() returns LargeBinary;

    // System management
    function getSystemStatus() returns {
        version: String;
        environment: String;
        database: {
            status: String;
            connections: Integer;
            size: String;
        };
        services: array of {
            name: String;
            status: String;
            uptime: Integer;
        };
    };
}

using {sap.a2a.portal as db} from '../db/schema';

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
        documents : redirected to ProjectDocuments,
        deployments : redirected to Deployments
    };

    entity ProjectMembers as projection on db.ProjectMembers;

    // Agent Management
    entity Agents as projection on db.ProjectAgents {
        *,
        project : redirected to Projects,
        executions : redirected to AgentExecutions
    };

    // Workflow Management
    entity Workflows as projection on db.ProjectWorkflows {
        *,
        project : redirected to Projects,
        executions : redirected to WorkflowExecutions
    };

    entity WorkflowExecutions as projection on db.WorkflowExecutions;
    entity TaskExecutions as projection on db.TaskExecutions;

    // Document Management
    entity ProjectDocuments as projection on db.ProjectDocuments {
        *,
        project : redirected to Projects
    } excluding {
        storageLocation // Exclude storage details from main view
    };

    // Deployment Management
    entity Deployments as projection on db.Deployments {
        *,
        project : redirected to Projects,
        deployedBy : redirected to Users,
        testResults : redirected to TestExecutions
    };

    entity AgentExecutions as projection on db.AgentExecutions;

    // Testing Framework
    entity TestExecutions as projection on db.TestExecutions {
        *,
        project : redirected to Projects,
        deployment : redirected to Deployments
    };

    // Session Management (Admin only)
    @restrict: [
        { grant: ['READ'], to: 'Admin' },
        { grant: ['*'], to: 'Admin' }
    ]
    entity SessionLogs as projection on db.SessionLogs;

    // Notifications
    entity Notifications as projection on db.Notifications {
        *,
        recipient : redirected to Users
    };

    // Monitoring (Read-only)
    @readonly
    entity PerformanceMetrics as projection on db.PerformanceMetrics;

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
    ) returns Deployments;

    action testAgent(
        agentId: UUID,
        testData: String
    ) returns TestExecutions;

    // Workflow Actions
    action executeWorkflow(
        workflowId: UUID,
        input: String
    ) returns WorkflowExecutions;

    action pauseWorkflow(executionId: UUID) returns Boolean;
    action resumeWorkflow(executionId: UUID) returns Boolean;
    action cancelWorkflow(executionId: UUID) returns Boolean;

    // Notification Actions
    action sendNotification(
        recipientId: UUID,
        title: String,
        message: String,
        type: String
    ) returns Notifications;

    // Document Actions
    action uploadDocument(
        projectId: UUID,
        fileName: String,
        content: LargeBinary
    ) returns ProjectDocuments;

    action downloadDocument(documentId: UUID) returns LargeBinary;

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
@requires: 'authenticated-user'
service WorkflowService {
    
    entity Workflows as projection on db.ProjectWorkflows;
    entity WorkflowExecutions as projection on db.WorkflowExecutions;
    entity TaskExecutions as projection on db.TaskExecutions;

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
@requires: 'Admin'
service AdminService {
    
    entity Users as projection on db.Users;
    entity SessionLogs as projection on db.SessionLogs;
    entity AuditLogs as projection on db.AuditLogs;
    entity PerformanceMetrics as projection on db.PerformanceMetrics;

    // Admin actions
    action createUser(
        email: String,
        firstName: String,
        lastName: String,
        role: String
    ) returns Users;

    action deactivateUser(userId: UUID) returns Boolean;
    @readonly action resetUserPassword(userId: UUID) returns String;

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

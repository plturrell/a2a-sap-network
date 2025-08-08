namespace com.sap.a2a.developerportal;

using {
    Currency,
    managed,
    sap,
    cuid
} from '@sap/cds/common';

/**
 * Core Business Objects for A2A Developer Portal
 */

// User Management
entity Users : managed {
    key ID          : UUID;
        email       : String(255) @mandatory;
        firstName   : String(100);
        lastName    : String(100);
        displayName : String(200);
        role        : String(50) @assert.enum: ['Developer', 'ProjectManager', 'Admin'];
        department  : String(100);
        isActive    : Boolean default true;
        lastLogin   : Timestamp;
        preferences : LargeString; // JSON string for user preferences
        
        // Associations
        projects    : Association to many ProjectMembers on projects.user = $self;
        sessions    : Composition of many UserSessions on sessions.user = $self;
        auditLogs   : Association to many AuditLogs on auditLogs.user = $self;
}

// Project Management
entity Projects : managed {
    key ID              : UUID;
        name            : String(200) @mandatory;
        description     : LargeString;
        status          : String(20) @assert.enum: ['Draft', 'Active', 'Suspended', 'Completed', 'Archived'] default 'Draft';
        priority        : String(10) @assert.enum: ['Low', 'Medium', 'High', 'Critical'] default 'Medium';
        startDate       : Date;
        endDate         : Date;
        budget          : Decimal(15,2);
        currency        : Currency;
        tags            : String(500); // Comma-separated tags
        metadata        : LargeString; // JSON metadata
        
        // Project structure
        parentProject   : Association to Projects;
        subProjects     : Composition of many Projects on subProjects.parentProject = $self;
        
        // Associations
        members         : Composition of many ProjectMembers on members.project = $self;
        agents          : Composition of many Agents on agents.project = $self;
        workflows       : Composition of many Workflows on workflows.project = $self;
        files           : Composition of many ProjectFiles on files.project = $self;
        deployments     : Composition of many Deployments on deployments.project = $self;
        approvals       : Association to many ApprovalWorkflows on approvals.project = $self;
}

entity ProjectMembers : managed {
    key ID      : UUID;
        project : Association to Projects @mandatory;
        user    : Association to Users @mandatory;
        role    : String(50) @assert.enum: ['Owner', 'Developer', 'Tester', 'Viewer'] default 'Developer';
        joinDate: Date default $now;
        isActive: Boolean default true;
}

// Agent Management
entity Agents : managed {
    key ID              : UUID;
        name            : String(200) @mandatory;
        description     : LargeString;
        type            : String(50) @assert.enum: ['ChatBot', 'WorkflowAgent', 'DataProcessor', 'APIAgent'] default 'ChatBot';
        status          : String(20) @assert.enum: ['Draft', 'Testing', 'Active', 'Inactive', 'Deprecated'] default 'Draft';
        version         : String(20) default '1.0.0';
        configuration   : LargeString; // JSON configuration
        capabilities    : String(1000); // Comma-separated capabilities
        
        // Relationships
        project         : Association to Projects @mandatory;
        workflows       : Association to many WorkflowSteps on workflows.agent = $self;
        deployments     : Association to many AgentDeployments on deployments.agent = $self;
        testResults     : Composition of many TestResults on testResults.agent = $self;
}

// Workflow Management
entity Workflows : managed {
    key ID              : UUID;
        name            : String(200) @mandatory;
        description     : LargeString;
        type            : String(50) @assert.enum: ['Sequential', 'Parallel', 'Conditional', 'Loop'] default 'Sequential';
        status          : String(20) @assert.enum: ['Draft', 'Active', 'Paused', 'Completed', 'Failed'] default 'Draft';
        bpmnDefinition  : LargeString; // BPMN XML
        
        // Relationships
        project         : Association to Projects @mandatory;
        steps           : Composition of many WorkflowSteps on steps.workflow = $self;
        executions      : Composition of many WorkflowExecutions on executions.workflow = $self;
}

entity WorkflowSteps : managed {
    key ID          : UUID;
        workflow    : Association to Workflows @mandatory;
        stepNumber  : Integer @mandatory;
        name        : String(200) @mandatory;
        type        : String(50) @assert.enum: ['Start', 'Task', 'Decision', 'End'] default 'Task';
        agent       : Association to Agents;
        configuration: LargeString; // JSON configuration
        nextSteps   : String(500); // Comma-separated step IDs
}

entity WorkflowExecutions : managed {
    key ID          : UUID;
        workflow    : Association to Workflows @mandatory;
        status      : String(20) @assert.enum: ['Running', 'Completed', 'Failed', 'Cancelled'] default 'Running';
        startTime   : Timestamp;
        endTime     : Timestamp;
        input       : LargeString; // JSON input data
        output      : LargeString; // JSON output data
        errorMessage: String(1000);
        
        // Execution steps
        stepExecutions: Composition of many WorkflowStepExecutions on stepExecutions.execution = $self;
}

entity WorkflowStepExecutions : managed {
    key ID          : UUID;
        execution   : Association to WorkflowExecutions @mandatory;
        step        : Association to WorkflowSteps @mandatory;
        status      : String(20) @assert.enum: ['Pending', 'Running', 'Completed', 'Failed', 'Skipped'] default 'Pending';
        startTime   : Timestamp;
        endTime     : Timestamp;
        input       : LargeString;
        output      : LargeString;
        errorMessage: String(1000);
}

// File Management
entity ProjectFiles : managed {
    key ID          : UUID;
        project     : Association to Projects @mandatory;
        fileName    : String(255) @mandatory;
        filePath    : String(1000) @mandatory;
        fileSize    : Integer;
        mimeType    : String(100);
        content     : LargeBinary; // File content
        checksum    : String(64); // SHA-256 checksum
        version     : String(20) default '1.0';
        
        // File metadata
        isPublic    : Boolean default false;
        tags        : String(500);
        metadata    : LargeString; // JSON metadata
}

// Deployment Management
entity Deployments : managed {
    key ID              : UUID;
        project         : Association to Projects @mandatory;
        version         : String(20) @mandatory;
        environment     : String(50) @assert.enum: ['Development', 'Testing', 'Staging', 'Production'] default 'Development';
        status          : String(20) @assert.enum: ['Pending', 'InProgress', 'Completed', 'Failed', 'RolledBack'] default 'Pending';
        deploymentType  : String(50) @assert.enum: ['Full', 'Incremental', 'Hotfix'] default 'Full';
        
        // Deployment details
        startTime       : Timestamp;
        endTime         : Timestamp;
        deployedBy      : Association to Users;
        releaseNotes    : LargeString;
        configuration   : LargeString; // JSON configuration
        
        // Associations
        agentDeployments: Composition of many AgentDeployments on agentDeployments.deployment = $self;
        logs            : Composition of many DeploymentLogs on logs.deployment = $self;
}

entity AgentDeployments : managed {
    key ID          : UUID;
        deployment  : Association to Deployments @mandatory;
        agent       : Association to Agents @mandatory;
        status      : String(20) @assert.enum: ['Pending', 'Deployed', 'Failed', 'RolledBack'] default 'Pending';
        endpoint    : String(500); // Deployment endpoint URL
        configuration: LargeString; // JSON configuration
}

// Testing Framework
entity TestSuites : managed {
    key ID          : UUID;
        name        : String(200) @mandatory;
        description : LargeString;
        project     : Association to Projects @mandatory;
        type        : String(50) @assert.enum: ['Unit', 'Integration', 'Performance', 'Security'] default 'Unit';
        
        // Test cases
        testCases   : Composition of many TestCases on testCases.testSuite = $self;
        executions  : Composition of many TestExecutions on executions.testSuite = $self;
}

entity TestCases : managed {
    key ID          : UUID;
        testSuite   : Association to TestSuites @mandatory;
        name        : String(200) @mandatory;
        description : LargeString;
        testData    : LargeString; // JSON test data
        expectedResult: LargeString; // JSON expected result
        agent       : Association to Agents;
        priority    : String(10) @assert.enum: ['Low', 'Medium', 'High', 'Critical'] default 'Medium';
}

entity TestExecutions : managed {
    key ID          : UUID;
        testSuite   : Association to TestSuites @mandatory;
        status      : String(20) @assert.enum: ['Running', 'Passed', 'Failed', 'Skipped'] default 'Running';
        startTime   : Timestamp;
        endTime     : Timestamp;
        
        // Results
        results     : Composition of many TestResults on results.execution = $self;
}

entity TestResults : managed {
    key ID          : UUID;
        execution   : Association to TestExecutions @mandatory;
        testCase    : Association to TestCases @mandatory;
        agent       : Association to Agents;
        status      : String(20) @assert.enum: ['Passed', 'Failed', 'Skipped'] default 'Passed';
        actualResult: LargeString; // JSON actual result
        errorMessage: String(1000);
        duration    : Integer; // Execution time in milliseconds
}

// Session Management
entity UserSessions : managed {
    key ID          : UUID;
        user        : Association to Users @mandatory;
        sessionId   : String(128) @mandatory;
        ipAddress   : String(45);
        userAgent   : String(500);
        startTime   : Timestamp;
        lastActivity: Timestamp;
        endTime     : Timestamp;
        isActive    : Boolean default true;
        metadata    : LargeString; // JSON session metadata
}

// Approval Workflows
entity ApprovalWorkflows : managed {
    key ID              : UUID;
        title           : String(200) @mandatory;
        description     : LargeString;
        type            : String(50) @assert.enum: ['ProjectApproval', 'DeploymentApproval', 'AgentApproval'] @mandatory;
        status          : String(20) @assert.enum: ['Pending', 'Approved', 'Rejected', 'Cancelled'] default 'Pending';
        priority        : String(10) @assert.enum: ['Low', 'Medium', 'High', 'Critical'] default 'Medium';
        
        // Request details
        requestedBy     : Association to Users @mandatory;
        requestData     : LargeString; // JSON request data
        
        // Approval chain
        approvers       : Composition of many ApprovalSteps on approvers.workflow = $self;
        
        // Related entities
        project         : Association to Projects;
        agent           : Association to Agents;
        deployment      : Association to Deployments;
}

entity ApprovalSteps : managed {
    key ID          : UUID;
        workflow    : Association to ApprovalWorkflows @mandatory;
        stepNumber  : Integer @mandatory;
        approver    : Association to Users @mandatory;
        status      : String(20) @assert.enum: ['Pending', 'Approved', 'Rejected'] default 'Pending';
        comments    : LargeString;
        approvedAt  : Timestamp;
        dueDate     : Timestamp;
}

// Monitoring and Logging
entity SystemMetrics : managed {
    key ID          : UUID;
        metricName  : String(100) @mandatory;
        metricValue : Decimal(15,4);
        unit        : String(20);
        timestamp   : Timestamp;
        source      : String(100);
        tags        : String(500); // Comma-separated tags
        metadata    : LargeString; // JSON metadata
}

entity DeploymentLogs : managed {
    key ID          : UUID;
        deployment  : Association to Deployments @mandatory;
        level       : String(10) @assert.enum: ['DEBUG', 'INFO', 'WARN', 'ERROR'] default 'INFO';
        message     : LargeString @mandatory;
        timestamp   : Timestamp;
        component   : String(100);
        metadata    : LargeString; // JSON metadata
}

// Audit Logging
entity AuditLogs : managed {
    key ID          : UUID;
        user        : Association to Users;
        action      : String(100) @mandatory;
        entityType  : String(50) @mandatory;
        entityId    : UUID;
        oldValues   : LargeString; // JSON old values
        newValues   : LargeString; // JSON new values
        timestamp   : Timestamp;
        ipAddress   : String(45);
        userAgent   : String(500);
        sessionId   : String(128);
        success     : Boolean default true;
        errorMessage: String(1000);
}

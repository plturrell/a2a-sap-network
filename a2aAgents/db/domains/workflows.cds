namespace a2a.platform;

using { a2a.platform.Identifiable, a2a.platform.Manageable, a2a.platform.StatusTracking, a2a.platform.BusinessMetadata, a2a.platform.ProcessingMetadata, a2a.platform.ConfigurationSettings } from '../aspects/common';

// Workflow and orchestration entities
entity Workflows : Identifiable, Manageable, StatusTracking, BusinessMetadata, ProcessingMetadata {
    workflowType : String(100) not null; // batch, streaming, realtime
    triggerType : String(50); // manual, scheduled, event
    cronExpression : String(100); // for scheduled workflows
    
    // Execution settings
    maxRetries : Integer default 3;
    timeout : Integer; // seconds
    parallelism : Integer default 1;
    
    // Relationships
    steps : Composition of many WorkflowSteps on steps.workflow = $self;
    executions : Composition of many WorkflowExecutions on executions.workflow = $self;
    schedules : Composition of many WorkflowSchedules on schedules.workflow = $self;
}

entity WorkflowSteps : Identifiable, StatusTracking, ConfigurationSettings {
    workflow : Association to Workflows;
    stepName : String(255) not null;
    stepOrder : Integer not null;
    agentType : String(100); // which agent handles this step
    
    // Dependencies
    dependsOn : LargeString; // JSON array of step IDs
    conditions : LargeString; // JSON conditions for execution
    
    // Configuration
    inputMapping : LargeString; // JSON input transformation
    outputMapping : LargeString; // JSON output transformation
    errorHandling : String(50) default 'RETRY'; // RETRY, SKIP, ABORT
}

entity WorkflowExecutions : Identifiable, StatusTracking, ProcessingMetadata {
    workflow : Association to Workflows;
    executionId : String(100) not null unique;
    triggeredBy : String(100); // user, schedule, event
    
    // Execution context
    inputData : LargeString; // JSON input
    outputData : LargeString; // JSON output
    errorMessage : String(5000);
    
    // Relationships
    stepExecutions : Composition of many StepExecutions on stepExecutions.workflowExecution = $self;
}

entity StepExecutions : Identifiable, StatusTracking, ProcessingMetadata {
    workflowExecution : Association to WorkflowExecutions;
    workflowStep : Association to WorkflowSteps;
    agent : Association to Agents;
    
    executionOrder : Integer;
    attempts : Integer default 1;
    
    // Step-specific data
    inputData : LargeString; // JSON
    outputData : LargeString; // JSON
    errorDetails : LargeString; // JSON error info
    logs : LargeString; // Execution logs
}

entity WorkflowSchedules : Identifiable, Manageable, ConfigurationSettings {
    workflow : Association to Workflows;
    scheduleType : String(50) not null; // CRON, INTERVAL, ONE_TIME
    cronExpression : String(100);
    intervalMinutes : Integer;
    scheduledTime : Timestamp;
    
    // Execution window
    validFrom : Timestamp;
    validTo : Timestamp;
    timezone : String(50) default 'UTC';
    
    nextExecutionTime : Timestamp;
    lastExecutionTime : Timestamp;
}
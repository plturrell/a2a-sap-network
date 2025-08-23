namespace a2a.platform;

// Import common aspects
using from './aspects/common';

// Import domain-specific entities
using from './domains/agents';
using from './domains/data-products';
using from './domains/workflows';
using from './domains/security';

// Cross-domain views and projections
view AgentPerformanceView as select from Agents {
    agentId,
    name,
    type,
    status,
    lastHeartbeat,
    responseTime,
    successRate,
    throughput
} where status = 'ACTIVE';

view DataProductQualityView as select from DataProducts {
    ID,
    name,
    type,
    status,
    qualityScore,
    completenessScore,
    accuracyScore,
    consistencyScore,
    timelinessScore
} where status = 'ACTIVE';

view WorkflowExecutionSummary as select from WorkflowExecutions {
    workflow.name as workflowName,
    count(*) as totalExecutions : Integer,
    sum(case when status = 'COMPLETED' then 1 else 0 end) as successfulExecutions : Integer,
    sum(case when status = 'FAILED' then 1 else 0 end) as failedExecutions : Integer,
    avg(duration) as avgDuration : Integer
} group by workflow.ID, workflow.name;

view SecurityDashboard as select from AuditLogs {
    user.userId,
    user.email,
    action,
    entityType,
    timestamp,
    outcome
} where timestamp >= $now - 30; // Last 30 days

// Calculated fields and projections
extend Agents with {
    virtual calculatedUptime : Decimal(5,4) = case 
        when lastHeartbeat is null then 0
        when $now - lastHeartbeat < 300 then successRate  // 5 minutes
        else successRate * 0.5
    end;
}

extend DataProducts with {
    virtual overallQuality : Decimal(5,4) = (
        coalesce(qualityScore, 0) + 
        coalesce(completenessScore, 0) + 
        coalesce(accuracyScore, 0) + 
        coalesce(consistencyScore, 0) + 
        coalesce(timelinessScore, 0)
    ) / 5;
}

extend Workflows with {
    virtual isScheduled : Boolean = schedules is not initial;
    virtual nextRun : Timestamp = schedules[1].nextExecutionTime;
}

// Annotations for UI generation
annotate AgentPerformanceView with @(
    UI.HeaderInfo: {
        TypeName: 'Agent',
        TypeNamePlural: 'Agents',
        Title: { Value: name }
    },
    UI.LineItem: [
        { Value: agentId, Label: 'Agent ID' },
        { Value: name, Label: 'Name' },
        { Value: type, Label: 'Type' },
        { Value: status, Label: 'Status' },
        { Value: successRate, Label: 'Success Rate' }
    ]
);

annotate DataProductQualityView with @(
    UI.HeaderInfo: {
        TypeName: 'Data Product',
        TypeNamePlural: 'Data Products',
        Title: { Value: name }
    },
    UI.LineItem: [
        { Value: name, Label: 'Product Name' },
        { Value: type, Label: 'Type' },
        { Value: status, Label: 'Status' },
        { Value: qualityScore, Label: 'Quality Score' }
    ]
);

/*
Modular Schema Benefits:
1. Separation of concerns - each domain has its own file
2. Reusable aspects reduce duplication
3. Clear relationships between domains
4. Easier maintenance and evolution
5. Better collaboration across teams
6. Consistent patterns through aspects

Structure:
- aspects/common.cds - Reusable aspects and mixins
- domains/agents.cds - Agent-related entities  
- domains/data-products.cds - Data product entities
- domains/workflows.cds - Workflow orchestration
- domains/security.cds - Security and audit entities
- schema-modular.cds - Main schema with cross-domain views
*/
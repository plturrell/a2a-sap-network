/**
 * @fileoverview SAP CAP Analytics Service for A2A Workflow Templates
 * @since 1.0.0
 * @module analyticsService
 * 
 * Provides analytics data using pure SAP CAP capabilities
 * Following SAP CAP best practices for analytical views
 */

using { cuid, managed } from '@sap/cds/common';
using { a2a.workflow.WorkflowInstances, a2a.workflow.ExecutionHistory } from './workflowTemplateService';

namespace a2a.analytics;

/**
 * Analytics Service for Workflow Template Metrics
 * Provides aggregated data for Fiori analytical apps and charts
 */
@requires: 'authenticated-user'
service AnalyticsService @(
    path: '/analytics',
    impl: './analyticsService.js'
) {
    
    /**
     * Workflow Template Performance Metrics
     */
    @readonly
    entity TemplatePerformance as projection on analytics.TemplatePerformanceView;
    
    /**
     * Agent Utilization Metrics
     */
    @readonly
    entity AgentUtilization as projection on analytics.AgentUtilizationView;
    
    /**
     * Execution Trends over time
     */
    @readonly
    entity ExecutionTrends as projection on analytics.ExecutionTrendsView;
    
    /**
     * Quality Metrics
     */
    @readonly
    entity QualityMetrics as projection on analytics.QualityMetricsView;
    
    /**
     * Cost Analysis
     */
    @readonly
    entity CostAnalysis as projection on analytics.CostAnalysisView;
    
    /**
     * User Adoption Metrics
     */
    @readonly
    entity UserAdoption as projection on analytics.UserAdoptionView;
    
    /**
     * Functions for dynamic analytics
     */
    
    /**
     * Get workflow performance by date range
     */
    function getPerformanceByDateRange(
        startDate: Date,
        endDate: Date,
        templateId: String
    ) returns array of {
        date: Date;
        executionCount: Integer;
        successRate: Decimal(5,2);
        avgDuration: Decimal(8,2);
        errorRate: Decimal(5,2);
    };
    
    /**
     * Get agent utilization heatmap data
     */
    function getAgentUtilizationHeatmap(
        dateRange: String // 'week', 'month', 'quarter'
    ) returns array of {
        agentId: Integer;
        agentName: String;
        utilizationData: array of {
            date: Date;
            utilizationPercent: Decimal(5,2);
            taskCount: Integer;
        };
    };
    
    /**
     * Get cost breakdown by template
     */
    function getCostBreakdown(
        startDate: Date,
        endDate: Date
    ) returns array of {
        templateId: String;
        templateName: String;
        totalCost: Decimal(10,2);
        executionCount: Integer;
        avgCostPerExecution: Decimal(8,2);
        costByAgent: array of {
            agentId: Integer;
            cost: Decimal(8,2);
        };
    };
    
    /**
     * Get predictive analytics for template usage
     */
    function getPredictiveAnalytics(
        templateId: String,
        forecastDays: Integer
    ) returns {
        historicalTrend: array of {
            date: Date;
            value: Decimal;
        };
        predictedTrend: array of {
            date: Date;
            predictedValue: Decimal;
            confidenceInterval: {
                lower: Decimal;
                upper: Decimal;
            };
        };
        insights: array of String;
    };
}

/**
 * Analytics Views and Calculations
 */
namespace analytics;

/**
 * Template Performance aggregated view
 */
view TemplatePerformanceView as select from ExecutionHistory {
    instance.template.ID as templateId,
    instance.template.name as templateName,
    instance.template.category.name as categoryName,
    count(*) as totalExecutions : Integer,
    sum(case when status = 'completed' then 1 else 0 end) as successfulExecutions : Integer,
    avg(duration) as avgDurationSeconds : Decimal(8,2),
    min(duration) as minDurationSeconds : Decimal(8,2),
    max(duration) as maxDurationSeconds : Decimal(8,2),
    sum(tasksTotal) as totalTasks : Integer,
    sum(tasksCompleted) as completedTasks : Integer,
    sum(tasksFailed) as failedTasks : Integer,
    cast(sum(case when status = 'completed' then 1 else 0 end) as Decimal(5,2)) / 
    cast(count(*) as Decimal(5,2)) * 100 as successRate : Decimal(5,2),
    cast(sum(tasksFailed) as Decimal(5,2)) / 
    cast(sum(tasksTotal) as Decimal(5,2)) * 100 as taskFailureRate : Decimal(5,2),
    startTime,
    endTime
} 
group by 
    instance.template.ID,
    instance.template.name,
    instance.template.category.name,
    startTime,
    endTime;

/**
 * Agent Utilization metrics
 */
view AgentUtilizationView as select from ExecutionHistory {
    JSON_VALUE(agentMetrics, '$.agentId') as agentId : Integer,
    count(*) as executionCount : Integer,
    avg(cast(JSON_VALUE(agentMetrics, '$.responseTime') as Decimal)) as avgResponseTime : Decimal(8,2),
    sum(cast(JSON_VALUE(agentMetrics, '$.taskCount') as Integer)) as totalTasks : Integer,
    cast(startTime as Date) as executionDate : Date,
    cast(sum(case when status = 'completed' then 1 else 0 end) as Decimal(5,2)) / 
    cast(count(*) as Decimal(5,2)) * 100 as successRate : Decimal(5,2)
}
where agentMetrics is not null
group by 
    JSON_VALUE(agentMetrics, '$.agentId'),
    cast(startTime as Date);

/**
 * Execution trends over time
 */
view ExecutionTrendsView as select from ExecutionHistory {
    cast(startTime as Date) as executionDate : Date,
    extract(hour from startTime) as executionHour : Integer,
    extract(day from startTime) as dayOfMonth : Integer,
    extract(week from startTime) as weekOfYear : Integer,
    extract(month from startTime) as month : Integer,
    extract(year from startTime) as year : Integer,
    count(*) as executionCount : Integer,
    avg(duration) as avgDuration : Decimal(8,2),
    sum(case when status = 'completed' then 1 else 0 end) as successCount : Integer,
    sum(case when status = 'failed' then 1 else 0 end) as failureCount : Integer,
    instance.template.category.name as categoryName
}
group by 
    cast(startTime as Date),
    extract(hour from startTime),
    extract(day from startTime),
    extract(week from startTime),
    extract(month from startTime),
    extract(year from startTime),
    instance.template.category.name;

/**
 * Quality metrics aggregation
 */
view QualityMetricsView as select from ExecutionHistory {
    instance.template.ID as templateId,
    instance.template.name as templateName,
    cast(startTime as Date) as measurementDate : Date,
    avg(cast(JSON_VALUE(results, '$.qualityScore') as Decimal)) as avgQualityScore : Decimal(5,2),
    avg(cast(JSON_VALUE(results, '$.accuracy') as Decimal)) as avgAccuracy : Decimal(5,2),
    avg(cast(JSON_VALUE(results, '$.completeness') as Decimal)) as avgCompleteness : Decimal(5,2),
    avg(cast(JSON_VALUE(results, '$.consistency') as Decimal)) as avgConsistency : Decimal(5,2),
    count(*) as measurementCount : Integer
}
where results is not null
group by 
    instance.template.ID,
    instance.template.name,
    cast(startTime as Date);

/**
 * Cost analysis view
 */
view CostAnalysisView as select from ExecutionHistory {
    instance.template.ID as templateId,
    instance.template.name as templateName,
    instance.template.category.name as categoryName,
    cast(startTime as Date) as costDate : Date,
    count(*) as executionCount : Integer,
    sum(duration) as totalDurationSeconds : Integer,
    // Simplified cost calculation: $0.001 per second
    cast(sum(duration) as Decimal(10,2)) * 0.001 as totalCost : Decimal(10,2),
    cast(sum(duration) as Decimal(10,2)) * 0.001 / count(*) as avgCostPerExecution : Decimal(8,2),
    sum(tasksTotal) as totalTasks : Integer,
    cast(sum(tasksTotal) as Decimal(10,2)) * 0.0001 as taskProcessingCost : Decimal(8,2)
}
group by 
    instance.template.ID,
    instance.template.name,
    instance.template.category.name,
    cast(startTime as Date);

/**
 * User adoption metrics
 */
view UserAdoptionView as select from WorkflowInstances {
    createdBy as userId : String,
    template.category.name as categoryName,
    cast(createdAt as Date) as adoptionDate : Date,
    count(*) as workflowsCreated : Integer,
    count(distinct template_ID) as uniqueTemplatesUsed : Integer,
    sum(executionCount) as totalExecutions : Integer,
    avg(cast(successCount as Decimal) / cast(executionCount as Decimal)) * 100 as avgSuccessRate : Decimal(5,2)
}
where executionCount > 0
group by 
    createdBy,
    template.category.name,
    cast(createdAt as Date);

/**
 * Real-time dashboard metrics
 */
view DashboardMetricsView as select from ExecutionHistory {
    count(*) as totalExecutions : Integer,
    count(distinct instance.template_ID) as activeTemplates : Integer,
    sum(case when status = 'running' then 1 else 0 end) as currentlyRunning : Integer,
    sum(case when status = 'completed' and startTime >= current_date then 1 else 0 end) as completedToday : Integer,
    sum(case when status = 'failed' and startTime >= current_date then 1 else 0 end) as failedToday : Integer,
    avg(case when status = 'completed' then duration else null end) as avgCompletionTime : Decimal(8,2),
    max(startTime) as lastExecutionTime : DateTime
};

/**
 * UI annotations for Fiori analytical charts and tables
 */
annotate TemplatePerformanceView with @(
    UI: {
        Chart: {
            ChartType: #Column,
            Dimensions: [templateName],
            Measures: [successRate, avgDurationSeconds],
            MeasureAttributes: [
                {Measure: successRate, Role: #Axis1, DataPoint: '@UI.DataPoint#SuccessRate'},
                {Measure: avgDurationSeconds, Role: #Axis1, DataPoint: '@UI.DataPoint#Duration'}
            ]
        },
        DataPoint#SuccessRate: {
            Value: successRate,
            Title: 'Success Rate %',
            TargetValue: 95,
            Criticality: #Good
        },
        DataPoint#Duration: {
            Value: avgDurationSeconds,
            Title: 'Avg Duration (sec)',
            Criticality: #Neutral
        }
    }
);

annotate AgentUtilizationView with @(
    UI: {
        Chart: {
            ChartType: #Line,
            Dimensions: [executionDate],
            DynamicMeasures: '@Analytics.AggregatedProperties',
            Measures: [executionCount, successRate]
        },
        LineItem: [
            {Value: agentId, Label: 'Agent ID'},
            {Value: executionCount, Label: 'Executions'},
            {Value: avgResponseTime, Label: 'Avg Response Time'},
            {Value: successRate, Label: 'Success Rate %'}
        ]
    }
);

annotate ExecutionTrendsView with @(
    UI: {
        Chart: {
            ChartType: #Line,
            Dimensions: [executionDate],
            Measures: [executionCount, successCount, failureCount]
        }
    }
);
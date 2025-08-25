/**
 * @fileoverview SAP CAP Workflow Template Service
 * @since 1.0.0
 * @module workflowTemplateService
 * 
 * Provides workflow template management for A2A document grounding
 * Following SAP CAP best practices and integrated with existing services
 */

using { cuid, managed, Country } from '@sap/cds/common';
using { a2a.launchpad.AgentMetadata } from './launchpadService';

/**
 * Core Workflow Template Service
 * Manages workflow templates, marketplace, and execution
 */
@requires: 'authenticated-user'
service WorkflowTemplateService @(
    path: '/workflow-templates',
    impl: './workflowTemplateService.js'
) {
    
    /**
     * Workflow Templates - Pre-built templates for common use cases
     */
    @odata.draft.enabled
    @cds.redirection.target
    entity Templates as projection on WorkflowTemplates;
    
    /**
     * Template Categories for marketplace organization
     */
    @readonly
    entity Categories as projection on TemplateCategories;
    
    /**
     * User's workflow instances
     */
    entity Instances as projection on WorkflowInstances;
    
    /**
     * Template marketplace with ratings and usage
     */
    @readonly
    entity Marketplace as projection on MarketplaceView;
    
    /**
     * Workflow execution history
     */
    @readonly
    entity History as projection on ExecutionHistory;
    
    /**
     * Actions for template management
     */
    
    /**
     * Create workflow from template
     */
    action createWorkflowFromTemplate(
        templateId: UUID,
        name: String(100),
        description: String(500),
        parameters: String  // JSON string of parameters
    ) returns Instances;
    
    /**
     * Execute workflow instance
     */
    action executeWorkflow(
        instanceId: UUID,
        executionContext: String  // JSON context
    ) returns {
        executionId: UUID;
        status: String;
        message: String;
    };
    
    /**
     * Generate template from natural language
     */
    action generateTemplateFromNL(
        description: String(1000),
        category: String(50)
    ) returns Templates;
    
    /**
     * Validate workflow template
     */
    action validateTemplate(
        templateId: UUID
    ) returns {
        isValid: Boolean;
        errors: array of String;
        warnings: array of String;
    };
    
    /**
     * Get workflow execution metrics
     */
    function getWorkflowMetrics(
        instanceId: UUID
    ) returns {
        executionCount: Integer;
        successRate: Decimal(5,2);
        avgDuration: Integer;
        costEstimate: Decimal(10,2);
        agentUtilization: array of {
            agentId: Integer;
            taskCount: Integer;
            avgResponseTime: Integer;
        };
    };
    
    /**
     * Search templates by use case
     */
    function searchTemplates(
        query: String,
        category: String,
        minRating: Decimal(3,2)
    ) returns array of Marketplace;
}


/**
 * Workflow template entity
 */
entity WorkflowTemplates : cuid, managed {
    name: String(100) @mandatory;
    description: String(500);
    category: Association to TemplateCategories;
    version: String(20) default '1.0.0';
    
    // Template definition
    definition: LargeString @Core.MediaType: 'application/json';
    
    // Metadata
    author: String(100);
    tags: array of String(50);
    icon: String(100) default 'sap-icon://workflow-tasks';
    
    // Usage tracking
    usageCount: Integer default 0;
    rating: Decimal(3,2) default 0.0;
    ratingCount: Integer default 0;
    status: String(20) enum {
        draft = 'draft';
        published = 'published';
        archived = 'archived';
    } default 'draft';
    
    // Template configuration
    parameters: LargeString @Core.MediaType: 'application/json';
    requiredAgents: array of Integer;
    estimatedDuration: Integer; // minutes
    
    // Publishing
    isPublic: Boolean default false;
    isOfficial: Boolean default false;
    publishedAt: DateTime;
    
    // Relations
    instances: Composition of many WorkflowInstances on instances.template = $self;
    reviews: Composition of many TemplateReviews on reviews.template = $self;
}

/**
 * Template categories
 */
entity TemplateCategories : cuid {
    name: String(50) @mandatory;
    description: String(200);
    icon: String(100);
    parentCategory: Association to TemplateCategories;
    sortOrder: Integer default 0;
}

/**
 * Workflow instances created from templates
 */
entity WorkflowInstances : cuid, managed {
    name: String(100) @mandatory;
    description: String(500);
    template: Association to WorkflowTemplates;
    
    // Instance configuration
    configuration: LargeString @Core.MediaType: 'application/json';
    parameters: LargeString @Core.MediaType: 'application/json';
    
    // Status tracking
    status: String(20) enum {
        draft = 'draft';
        ready = 'ready';
        running = 'running';
        completed = 'completed';
        failed = 'failed';
        cancelled = 'cancelled';
    } default 'draft';
    
    // Execution details
    lastExecutionId: UUID;
    executionCount: Integer default 0;
    successCount: Integer default 0;
    failureCount: Integer default 0;
    
    // Relations
    executions: Composition of many ExecutionHistory on executions.instance = $self;
}

/**
 * Workflow execution history
 */
entity ExecutionHistory : cuid {
    instance: Association to WorkflowInstances;
    
    // Execution details
    startTime: DateTime @mandatory;
    endTime: DateTime;
    duration: Integer; // seconds
    status: String(20) enum {
        running = 'running';
        completed = 'completed';
        failed = 'failed';
        cancelled = 'cancelled';
    };
    
    // Context and results
    executionContext: LargeString @Core.MediaType: 'application/json';
    results: LargeString @Core.MediaType: 'application/json';
    errorDetails: LargeString;
    
    // Metrics
    tasksTotal: Integer default 0;
    tasksCompleted: Integer default 0;
    tasksFailed: Integer default 0;
    
    // Agent utilization
    agentMetrics: LargeString @Core.MediaType: 'application/json';
}

/**
 * Template reviews and ratings
 */
entity TemplateReviews : cuid, managed {
    template: Association to WorkflowTemplates;
    rating: Integer @assert.range: [1, 5];
    comment: String(1000);
    helpful: Integer default 0;
    notHelpful: Integer default 0;
}

/**
 * Marketplace view combining templates with metrics
 */
@readonly
view MarketplaceView as select from WorkflowTemplates {
    *,
    category.name as categoryName,
    category.icon as categoryIcon,
    case
        when rating >= 4.5 then 'Excellent'
        when rating >= 3.5 then 'Good'
        when rating >= 2.5 then 'Average'
        else 'Below Average'
    end as ratingBadge : String,
    case
        when usageCount >= 1000 then 'Popular'
        when usageCount >= 100 then 'Trending'
        else 'New'
    end as popularityBadge : String
} where isPublic = true;

/**
 * Pre-defined workflow templates
 */
annotate WorkflowTemplates with @(
    UI: {
        SelectionFields: [name, category_ID, rating],
        LineItem: [
            {Value: name, Label: 'Template Name'},
            {Value: category.name, Label: 'Category'},
            {Value: rating, Label: 'Rating'},
            {Value: usageCount, Label: 'Usage Count'},
            {Value: status, Label: 'Status'}
        ],
        HeaderInfo: {
            TypeName: 'Workflow Template',
            TypeNamePlural: 'Workflow Templates',
            Title: {Value: name},
            Description: {Value: description}
        }
    }
);

/**
 * Standard templates initialization data
 */
annotate WorkflowTemplateService.Categories with @cds.odata.valuelist;
annotate WorkflowTemplateService.Categories with {
    ID @Common.Text: name @Common.TextArrangement: #TextOnly;
};

/**
 * Pre-seeded categories
 */
annotate TemplateCategories with @cds.persistence.exists;
annotate TemplateCategories with @cds.seed: [
    {ID: '1', name: 'Document Processing', icon: 'sap-icon://document', sortOrder: 1},
    {ID: '2', name: 'Data Integration', icon: 'sap-icon://connected', sortOrder: 2},
    {ID: '3', name: 'AI/ML Workflows', icon: 'sap-icon://artificial-intelligence', sortOrder: 3},
    {ID: '4', name: 'Quality Assurance', icon: 'sap-icon://quality-issue', sortOrder: 4},
    {ID: '5', name: 'RAG Pipelines', icon: 'sap-icon://process', sortOrder: 5}
];
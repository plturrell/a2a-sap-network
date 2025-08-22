namespace a2a.network;

using { 
    cuid, 
    managed, 
    temporal, 
    Currency,
    Country,
    Language,
    sap.common.CodeList
} from '@sap/cds/common';

// using { a2a.network.tracking } from './change_tracking'; // Disabled for now
using { a2a.network.aspects } from './sap_aspects';
// using { a2a.network } from './blockchainSchema'; // Temporarily disabled

// Custom Types for Type Safety
type BlockchainAddress : String(42);
type TransactionHash : String(66);
type PublicKey : String(130);
type SemVer : String(20);
type IPAddress : String(45);
type URL : String(500);
type Email : String(254);

// Core Agent entities with performance indexes
@cds.persistence.index: [
    { unique: true, elements: ['address'] },
    { elements: ['isActive', 'reputation'] },
    { elements: ['country_code', 'isActive'] }
]
@Analytics: { DCL: 'AGENT' }
@PersonalData: { EntitySemantics: 'DataSubject' }
entity Agents : cuid, managed {
    @Core.Immutable
    @Common.Label: 'Blockchain Address'
    @assert.format: '^0x[a-fA-F0-9]{40}$'
    address      : String(42) @mandatory @assert.unique;
    
    @Common.Label: 'Agent Name'
    @PersonalData: { FieldSemantics: 'Name' }
    name         : localized String(100) @mandatory;
    
    @Common.Label: 'Service Endpoint'
    @assert.format: '^https?://[^\s]+$'
    endpoint     : String(500);
    
    @Common.Label: 'Reputation Score'
    @assert.range: [0, 1000]
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #AVG
    reputation   : Integer default 100;
    
    @Common.Label: 'Active Status'
    @UI.Hidden: false
    isActive     : Boolean default true;
    
    @Common.Label: 'Country'
    @PersonalData: { FieldSemantics: 'DataSubjectCountry' }
    country      : Country;
    
    @Common.Label: 'Agent Capabilities'
    capabilities : Composition of many AgentCapabilities on capabilities.agent = $self;
    
    @Common.Label: 'Provided Services'
    services     : Composition of many Services on services.provider = $self;
    
    @Common.Label: 'Sent Messages'
    messages     : Association to many Messages on messages.sender = $self;
    
    @Common.Label: 'Performance Metrics'
    performance  : Composition of one AgentPerformance on performance.agent = $self;
    
    @Common.Label: 'Reputation Transactions'
    reputationTransactions : Composition of many ReputationTransactions on reputationTransactions.agent = $self;
    
    @Common.Label: 'Endorsements Received'
    endorsementsReceived   : Composition of many PeerEndorsements on endorsementsReceived.toAgent = $self;
    
    @Common.Label: 'Endorsements Given'
    endorsementsGiven      : Composition of many PeerEndorsements on endorsementsGiven.fromAgent = $self;
    
    @Common.Label: 'Reputation Milestones'
    milestones            : Composition of many ReputationMilestones on milestones.agent = $self;
    
    @Common.Label: 'Recovery Programs'
    recoveryPrograms      : Composition of many ReputationRecovery on recoveryPrograms.agent = $self;
    
    // Computed reputation fields
    @Common.Label: 'Current Badge'
    @Core.Computed
    virtual currentBadge          : String(20);
    
    @Common.Label: 'Endorsement Power'
    @Core.Computed
    virtual endorsementPower      : Integer;
    
    @Common.Label: 'Reputation Trend'
    @Core.Computed
    virtual reputationTrend       : String(10);
}


@cds.persistence.index: [
    { elements: ['agent_ID', 'capability_ID'] },
    { elements: ['status', 'registeredAt'] }
]
entity AgentCapabilities : cuid {
    agent        : Association to Agents;
    capability   : Association to Capabilities;
    registeredAt : DateTime;
    version      : String(20);
    status       : String(20) enum { active; deprecated; sunset; };
}

// Capability Registry with search optimization
@cds.persistence.index: [
    { elements: ['category_code', 'status_code'] },
    { elements: ['version', 'status_code'] }
]
@Analytics: { DCL: 'CAPABILITY' }
@UI.HeaderInfo: {
    TypeName: 'Capability',
    TypeNamePlural: 'Capabilities',
    Title: { Value: name }
}
entity Capabilities : cuid, managed {
    @Common.Label: 'Capability Name'
    @Search.defaultSearchElement: true
    name         : localized String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description  : localized String(1000);
    
    @Common.Label: 'Category'
    @assert.integrity
    category     : Association to CapabilityCategories;
    
    @Common.Label: 'Tags'
    tags         : array of String;
    
    @Common.Label: 'Input Types'
    inputTypes   : array of String;
    
    @Common.Label: 'Output Types'
    outputTypes  : array of String;
    
    @Common.Label: 'Version'
    @assert.format: '^\d+\.\d+\.\d+$'
    version      : String(20) default '1.0.0';
    
    @Common.Label: 'Status'
    @assert.integrity
    status       : Association to StatusCodes default 'active';
    
    @Common.Label: 'Dependencies'
    dependencies : array of String;
    
    @Common.Label: 'Conflicts'
    conflicts    : array of String;
    
    @Common.Label: 'Agents with Capability'
    agents       : Association to many AgentCapabilities on agents.capability = $self;
}

entity CapabilityCategories : CodeList {
    key code : Integer enum { 
        COMPUTATION = 0; 
        STORAGE = 1; 
        ANALYSIS = 2; 
        COMMUNICATION = 3; 
        GOVERNANCE = 4; 
    };
    name : localized String(100);
    description : localized String(500);
}

entity StatusCodes : CodeList {
    key code : String(20) enum { active; deprecated; sunset; };
    name : localized String(50);
    description : localized String(200);
}

// Service Marketplace with performance indexes
@cds.persistence.index: [
    { elements: ['provider_ID', 'isActive'] },
    { elements: ['category', 'isActive', 'averageRating'] },
    { elements: ['pricePerCall', 'currency_code'] }
]
@Analytics: { DCL: 'SERVICE' }
@UI.HeaderInfo: {
    TypeName: 'Service',
    TypeNamePlural: 'Services',
    Title: { Value: name }
}
entity Services : cuid, managed {
    @Common.Label: 'Service Provider'
    @assert.integrity
    provider        : Association to Agents;
    
    @Common.Label: 'Service Name'
    @Common.FieldControl: #Mandatory
    name           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description    : String(1000);
    
    @Common.Label: 'Category'
    category       : String(50);
    
    @Common.Label: 'Price per Call'
    @Measures.Unit: currency
    @assert.range: [0, 999999.9999]
    pricePerCall   : Decimal(10,4);
    
    @Common.Label: 'Currency'
    currency       : Currency default 'EUR';
    
    @Common.Label: 'Minimum Reputation Required'
    @assert.range: [0, 1000]
    minReputation  : Integer default 0;
    
    @Common.Label: 'Max Calls per Day'
    @assert.range: [1, 1000000]
    maxCallsPerDay : Integer default 1000;
    
    @Common.Label: 'Active Status'
    isActive       : Boolean default true;
    
    @Common.Label: 'Total Calls'
    @readonly
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #SUM
    totalCalls     : Integer default 0;
    
    @Common.Label: 'Average Rating'
    @assert.range: [0, 5]
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #AVG
    averageRating  : Decimal(3,2) default 0;
    
    @Common.Label: 'Escrow Amount'
    @Measures.Unit: currency
    escrowAmount   : Decimal(10,4) default 0;
    
    @Common.Label: 'Service Orders'
    serviceOrders  : Composition of many ServiceOrders on serviceOrders.service = $self;
}

@cds.persistence.index: [
    { elements: ['service_ID', 'status'] },
    { elements: ['consumer_ID', 'createdAt'] },
    { elements: ['status', 'completedAt'] }
]
@Analytics: { DCL: 'ORDER' }
entity ServiceOrders : cuid, managed {
    @Common.Label: 'Service'
    @assert.integrity
    service         : Association to Services;
    
    @Common.Label: 'Consumer Agent'
    @assert.integrity
    consumer        : Association to Agents;
    
    @Common.Label: 'Order Status'
    status          : String(20) enum { pending; active; completed; cancelled; disputed; };
    
    @Common.Label: 'Call Count'
    @assert.range: [0, 2147483647]
    callCount       : Integer default 0;
    
    @Common.Label: 'Total Amount'
    @Measures.Unit: currency
    totalAmount     : Decimal(10,4);
    
    @Common.Label: 'Escrow Released'
    escrowReleased  : Boolean default false;
    
    @Common.Label: 'Completed At'
    completedAt     : DateTime;
    
    @Common.Label: 'Rating'
    @assert.range: [1, 5]
    rating          : Integer;
    
    @Common.Label: 'Feedback'
    @UI.MultiLineText: true
    feedback        : String(500);
}

// Message Routing with delivery optimization
@cds.persistence.index: [
    { unique: true, elements: ['messageHash'] },
    { elements: ['sender_ID', 'status'] },
    { elements: ['recipient_ID', 'status'] },
    { elements: ['priority', 'createdAt'] }
]
@Analytics: { DCL: 'MESSAGE' }
entity Messages : cuid, managed {
    @Common.Label: 'Sender Agent'
    @assert.integrity
    sender      : Association to Agents;
    
    @Common.Label: 'Recipient Agent'
    @assert.integrity
    recipient   : Association to Agents;
    
    @Core.Immutable
    @Common.Label: 'Message Hash'
    @assert.format: '^0x[a-fA-F0-9]{64}$'
    messageHash : String(66) @mandatory @assert.unique;
    
    @Common.Label: 'Protocol'
    protocol    : String(50);
    
    @Common.Label: 'Priority'
    @assert.range: [1, 10]
    priority    : Integer default 1;
    
    @Common.Label: 'Delivery Status'
    status      : String(20) enum { pending; sent; delivered; failed; };
    
    @Common.Label: 'Retry Count'
    @assert.range: [0, 10]
    retryCount  : Integer default 0;
    
    @Common.Label: 'Gas Used'
    @Analytics.Measure: true
    gasUsed     : Integer;
    
    @Common.Label: 'Delivered At'
    deliveredAt : DateTime;
}

// Performance & Reputation
@Analytics: { DCL: 'PERFORMANCE' }
entity AgentPerformance : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent               : Association to one Agents;
    
    @Common.Label: 'Total Tasks'
    @readonly
    @Analytics.Measure: true
    totalTasks          : Integer default 0;
    
    @Common.Label: 'Successful Tasks'
    @readonly
    @Analytics.Measure: true
    successfulTasks     : Integer default 0;
    
    @Common.Label: 'Failed Tasks'
    @readonly
    @Analytics.Measure: true
    failedTasks         : Integer default 0;
    
    @Common.Label: 'Average Response Time (ms)'
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #AVG
    averageResponseTime : Integer; // milliseconds
    
    @Common.Label: 'Average Gas Usage'
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #AVG
    averageGasUsage     : Integer;
    
    @Common.Label: 'Reputation Score'
    @assert.range: [0, 1000]
    reputationScore     : Integer default 100;
    
    @Common.Label: 'Trust Score'
    @assert.range: [0.0, 5.0]
    trustScore          : Decimal(3,2) default 1.0;
    
    @Common.Label: 'Last Updated'
    @readonly
    lastUpdated         : DateTime;
    
    @Common.Label: 'Performance History'
    performanceHistory  : Composition of many PerformanceSnapshots on performanceHistory.performance = $self;
}

entity PerformanceSnapshots : cuid {
    performance    : Association to AgentPerformance;
    timestamp      : DateTime;
    taskCount      : Integer;
    successRate    : Decimal(5,2);
    responseTime   : Integer;
    gasUsage       : Integer;
    reputationDelta: Integer;
}

// Workflows
@Analytics: { DCL: 'WORKFLOW' }
@UI.HeaderInfo: {
    TypeName: 'Workflow',
    TypeNamePlural: 'Workflows',
    Title: { Value: name }
}
entity Workflows : cuid, managed {
    @Common.Label: 'Workflow Name'
    @Search.defaultSearchElement: true
    name        : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description : String(1000);
    
    @Common.Label: 'Workflow Definition'
    @Core.MediaType: 'application/json'
    definition  : LargeString; // JSON workflow definition
    
    @Common.Label: 'Active Status'
    isActive    : Boolean default true;
    
    @Common.Label: 'Category'
    category    : String(50);
    
    @Common.Label: 'Owner'
    @assert.integrity
    owner       : Association to Agents;
    
    @Common.Label: 'Executions'
    executions  : Composition of many WorkflowExecutions on executions.workflow = $self;
}

entity WorkflowExecutions : cuid, managed {
    workflow     : Association to Workflows;
    executionId  : String(66);
    status       : String(20) enum { running; completed; failed; cancelled; };
    startedAt    : DateTime;
    completedAt  : DateTime;
    gasUsed      : Integer;
    result       : LargeString; // JSON result
    error        : String(1000);
    steps        : Composition of many WorkflowSteps on steps.execution = $self;
}

entity WorkflowSteps : cuid {
    @Common.Label: 'Execution'
    execution    : Association to WorkflowExecutions;
    
    @Common.Label: 'Step Number'
    @assert.range: [1, 1000]
    stepNumber   : Integer;
    
    @Common.Label: 'Agent Address'
    @assert.format: '^0x[a-fA-F0-9]{40}$'
    agentAddress : String(42);
    
    @Common.Label: 'Action'
    action       : String(100);
    
    @Common.Label: 'Input Data'
    @Core.MediaType: 'application/json'
    input        : LargeString; // JSON
    
    @Common.Label: 'Output Data'
    @Core.MediaType: 'application/json'
    output       : LargeString; // JSON
    
    @Common.Label: 'Step Status'
    status       : String(20) enum { pending; running; completed; failed; };
    
    @Common.Label: 'Gas Used'
    gasUsed      : Integer;
    
    @Common.Label: 'Started At'
    startedAt    : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt  : DateTime;
}

// Cross-chain Support
entity ChainBridges : cuid, managed {
    sourceChain  : String(50) @mandatory;
    targetChain  : String(50) @mandatory;
    bridgeAddress: String(42) @mandatory;
    isActive     : Boolean default true;
    transfers    : Composition of many CrossChainTransfers on transfers.bridge = $self;
}

entity CrossChainTransfers : cuid, managed {
    bridge       : Association to ChainBridges;
    fromAgent    : String(42);
    toAgent      : String(42);
    messageHash  : String(66);
    status       : String(20) enum { initiated; pending; completed; failed; };
    sourceBlock  : Integer;
    targetBlock  : Integer;
    gasUsed      : Integer;
}

// Privacy Features
entity PrivateChannels : cuid, managed {
    participants : array of String(42);
    publicKey    : String(130);
    isActive     : Boolean default true;
    messages     : Composition of many PrivateMessages on messages.channel = $self;
}

entity PrivateMessages : cuid, managed {
    channel      : Association to PrivateChannels;
    sender       : String(42);
    encryptedData: LargeString;
    zkProof      : String(500);
    timestamp    : DateTime;
}

// Network Analytics with temporal versioning
@cds.autoexpose
@Analytics: { DCL: 'NETWORK_STATS' }
@UI.Chart: {
    Title: 'Network Statistics',
    ChartType: #Line,
    Dimensions: ['validFrom'],
    Measures: ['totalAgents', 'activeAgents', 'networkLoad']
}
entity NetworkStats : temporal, cuid {
    @Common.Label: 'Total Agents'
    @Analytics.Measure: true
    totalAgents      : Integer;
    
    @Common.Label: 'Active Agents'
    @Analytics.Measure: true
    activeAgents     : Integer;
    
    @Common.Label: 'Total Services'
    @Analytics.Measure: true
    totalServices    : Integer;
    
    @Common.Label: 'Total Capabilities'
    @Analytics.Measure: true
    totalCapabilities: Integer;
    
    @Common.Label: 'Total Messages'
    @Analytics.Measure: true
    totalMessages    : Integer;
    
    @Common.Label: 'Total Transactions'
    @Analytics.Measure: true
    totalTransactions: Integer;
    
    @Common.Label: 'Average Reputation'
    @Analytics.Measure: true
    @Analytics.DefaultAggregation: #AVG
    averageReputation: Decimal(5,2);
    
    @Common.Label: 'Network Load %'
    @assert.range: [0, 100]
    @Analytics.Measure: true
    networkLoad      : Decimal(5,2);
    
    @Common.Label: 'Gas Price (Gwei)'
    @Analytics.Measure: true
    gasPrice         : Decimal(10,4);
}

// Configuration
entity NetworkConfig : cuid, managed {
    configKey   : String(100) @mandatory;
    value       : String(1000);
    description : String(500);
    isActive    : Boolean default true;
}

// Request Management
entity Requests : cuid, managed {
    @Common.Label: 'Request Title'
    title : String(200) @mandatory;
    
    @Common.Label: 'Description'
    description : String(2000);
    
    @Common.Label: 'Request Type'
    requestType : String(50) default 'SERVICE_REQUEST';
    
    @Common.Label: 'Priority'
    priority : String(10) default 'MEDIUM';
    
    @Common.Label: 'Status'
    status : String(20) default 'PENDING';
    
    @Common.Label: 'Requester'
    requester : Association to Agents;
    
    @Common.Label: 'Assigned Agent'
    assignedAgent : Association to Agents;
    
    @Common.Label: 'Due Date'
    dueDate : DateTime;
    
    @Common.Label: 'Responses'
    responses : Composition of many Responses on responses.request = $self;
}

// Response Management
entity Responses : cuid, managed {
    @Common.Label: 'Response Content'
    content : String(5000) @mandatory;
    
    @Common.Label: 'Response Type'
    responseType : String(50) default 'TEXT';
    
    @Common.Label: 'Status'
    status : String(20) default 'DRAFT';
    
    @Common.Label: 'Request'
    request : Association to Requests @mandatory;
    
    @Common.Label: 'Responder'
    responder : Association to Agents;
    
    @Common.Label: 'Response Priority'
    priority : String(10) default 'NORMAL';
    
    @Common.Label: 'Is Final Response'
    isFinalResponse : Boolean default false;
}

// Views for Analytics
view TopAgents as select from Agents {
    ID,
    address,
    name,
    reputation,
    performance.successfulTasks as completedTasks,
    performance.averageResponseTime as avgResponseTime
} where isActive = true order by reputation desc;

view ActiveServices as select from Services {
    ID,
    name,
    provider.name as providerName,
    category,
    pricePerCall,
    totalCalls,
    averageRating
} where isActive = true;

view RecentWorkflows as select from WorkflowExecutions {
    ID,
    workflow.name as workflowName,
    status,
    startedAt,
    completedAt,
    gasUsed
} order by startedAt desc;

// Multi-tenancy support
@multitenancy: {
    kind: 'discriminator'
}
entity TenantSettings : managed {
    key tenant : String(36);
    settings : {
        maxAgents : Integer default 1000;
        maxServices : Integer default 100;
        maxWorkflows : Integer default 50;
        features : {
            blockchain : Boolean default true;
            ai : Boolean default true;
            analytics : Boolean default true;
        };
    };
}

// Audit Log
@PersonalData: { EntitySemantics: 'DataSubjectDetails' }
@cds.persistence.index: [
    { elements: ['tenant', 'createdAt'] },
    { elements: ['entity', 'action', 'createdAt'] }
]
entity AuditLog : managed {
    key ID : UUID;
    
    @Common.Label: 'Tenant ID'
    tenant : String(36);
    
    @Common.Label: 'User'
    @PersonalData: { FieldSemantics: 'UserID' }
    user : String(100);
    
    @Common.Label: 'Action'
    action : String(50);
    
    @Common.Label: 'Entity Type'
    entity : String(100);
    
    @Common.Label: 'Entity Key'
    entityKey : String(100);
    
    @Common.Label: 'Old Value'
    @Core.MediaType: 'application/json'
    oldValue : LargeString;
    
    @Common.Label: 'New Value'
    @Core.MediaType: 'application/json'
    newValue : LargeString;
    
    @Common.Label: 'IP Address'
    @PersonalData: { FieldSemantics: 'SystemUserID' }
    ip : String(45);
    
    @Common.Label: 'User Agent'
    userAgent : String(500);
}

// Feature Toggles
entity FeatureToggles : managed {
    key feature : String(50);
    enabled : Boolean default false;
    description : localized String(200);
    validFrom : Date;
    validTo : Date;
    tenant : String(36);
}

// Extensibility - Custom Fields
@Extensible
entity ExtensionFields : managed {
    key entity : String(100);
    key field : String(50);
    key tenant : String(36);
    dataType : String(20) enum { String; Integer; Decimal; Boolean; Date; };
    label : localized String(100);
    defaultValue : String(100);
    mandatory : Boolean default false;
    visible : Boolean default true;
}

// ================================
// AGENT 2 - AI PREPARATION ENTITIES
// ================================

// AI Preparation Tasks with performance indexes
@cds.persistence.index: [
    { elements: ['status', 'createdAt'] },
    { elements: ['modelType', 'dataType'] },
    { elements: ['priority', 'status'] }
]
@Analytics: { DCL: 'AI_PREP_TASK' }
@UI.HeaderInfo: {
    TypeName: 'AI Preparation Task',
    TypeNamePlural: 'AI Preparation Tasks',
    Title: { Value: taskName }
}
entity AIPreparationTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName        : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description     : String(1000);
    
    @Common.Label: 'Dataset Name'
    datasetName     : String(100) @mandatory;
    
    @Common.Label: 'Model Type'
    modelType       : String(50) @mandatory enum { 
        CLASSIFICATION; REGRESSION; CLUSTERING; EMBEDDING; 
        LLM; TIME_SERIES; RECOMMENDATION; ANOMALY; 
    };
    
    @Common.Label: 'Data Type'
    dataType        : String(50) @mandatory enum { 
        TABULAR; TEXT; IMAGE; AUDIO; VIDEO; TIME_SERIES; GRAPH; 
    };
    
    @Common.Label: 'ML Framework'
    framework       : String(50) enum { 
        TENSORFLOW; PYTORCH; SCIKIT_LEARN; XGBOOST; HUGGINGFACE; AUTO; 
    } default 'TENSORFLOW';
    
    @Common.Label: 'Train/Test Split Ratio'
    @assert.range: [50, 90]
    splitRatio      : Integer default 80;
    
    @Common.Label: 'Validation Strategy'
    validationStrategy : String(20) enum { KFOLD; HOLDOUT; } default 'KFOLD';
    
    @Common.Label: 'Random Seed'
    randomSeed      : Integer default 42;
    
    @Common.Label: 'Feature Selection Enabled'
    featureSelection : Boolean default true;
    
    @Common.Label: 'Auto Feature Engineering'
    autoFeatureEngineering : Boolean default true;
    
    @Common.Label: 'Optimization Metric'
    optimizationMetric : String(20) enum { 
        AUTO; ACCURACY; AUC; F1; MSE; MAE; PERPLEXITY; 
    } default 'AUTO';
    
    @Common.Label: 'Use GPU Acceleration'
    useGPU          : Boolean default false;
    
    @Common.Label: 'Enable Distributed Processing'
    distributed     : Boolean default false;
    
    @Common.Label: 'Memory Optimized'
    memoryOptimized : Boolean default false;
    
    @Common.Label: 'Cache Intermediate Results'
    cacheResults    : Boolean default true;
    
    @Common.Label: 'Task Status'
    status          : String(20) enum { 
        DRAFT; PENDING; RUNNING; COMPLETED; FAILED; PAUSED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority        : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage    : String(50);
    
    @Common.Label: 'Processing Time (seconds)'
    processingTime  : Integer;
    
    @Common.Label: 'Results Summary'
    @Core.MediaType: 'application/json'
    resultsSummary  : LargeString;
    
    @Common.Label: 'Error Details'
    errorDetails    : String(1000);
    
    @Common.Label: 'Started At'
    startedAt       : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt     : DateTime;
    
    @Common.Label: 'Agent'
    agent           : Association to Agents;
    
    @Common.Label: 'Features'
    features        : Composition of many AIPreparationFeatures on features.task = $self;
}

// AI Preparation Features
entity AIPreparationFeatures : cuid {
    @Common.Label: 'Task'
    task            : Association to AIPreparationTasks;
    
    @Common.Label: 'Feature Name'
    name            : String(100) @mandatory;
    
    @Common.Label: 'Feature Type'
    type            : String(20) enum { NUMERICAL; CATEGORICAL; TEXT; DATETIME; BOOLEAN; };
    
    @Common.Label: 'Data Type'
    dataType        : String(20);
    
    @Common.Label: 'Is Target'
    isTarget        : Boolean default false;
    
    @Common.Label: 'Is Selected'
    isSelected      : Boolean default true;
    
    @Common.Label: 'Importance Score'
    @assert.range: [0.0, 1.0]
    importance      : Decimal(5,4);
    
    @Common.Label: 'Missing Percentage'
    @assert.range: [0.0, 100.0]
    missingPercent  : Decimal(5,2);
    
    @Common.Label: 'Unique Values'
    uniqueValues    : Integer;
    
    @Common.Label: 'Mean Value'
    meanValue       : Decimal(15,6);
    
    @Common.Label: 'Standard Deviation'
    stdDev          : Decimal(15,6);
    
    @Common.Label: 'Min Value'
    minValue        : Decimal(15,6);
    
    @Common.Label: 'Max Value'
    maxValue        : Decimal(15,6);
    
    @Common.Label: 'Feature Engineering Applied'
    engineeringApplied : String(500);
}

// ================================
// REPUTATION SYSTEM ENTITIES
// ================================

// Reputation transaction tracking
entity ReputationTransactions : cuid, managed {
    @Common.Label: 'Agent'
    @assert.integrity
    agent              : Association to Agents;
    
    @Common.Label: 'Transaction Type'
    transactionType    : String(50) not null enum {
        TASK_COMPLETION = 'TASK_COMPLETION';
        SERVICE_RATING = 'SERVICE_RATING';
        PEER_ENDORSEMENT = 'PEER_ENDORSEMENT';
        QUALITY_BONUS = 'QUALITY_BONUS';
        PENALTY = 'PENALTY';
        MILESTONE_BONUS = 'MILESTONE_BONUS';
        RECOVERY_REWARD = 'RECOVERY_REWARD';
        WORKFLOW_PARTICIPATION = 'WORKFLOW_PARTICIPATION';
        COLLABORATION_BONUS = 'COLLABORATION_BONUS';
    };
    
    @Common.Label: 'Amount'
    @assert.range: [-50, 50]
    amount            : Integer not null;
    
    @Common.Label: 'Reason'
    reason            : String(200);
    
    @Common.Label: 'Context'
    @Core.MediaType: 'application/json'
    context           : LargeString;
    
    @Common.Label: 'Is Automated'
    isAutomated       : Boolean default false;
    
    @Common.Label: 'Created By Agent'
    createdByAgent    : Association to Agents;
    
    @Common.Label: 'Related Service Order'
    serviceOrder      : Association to ServiceOrders;
    
    @Common.Label: 'Related Workflow'
    workflow          : Association to Workflows;
}

// Peer-to-peer endorsements
entity PeerEndorsements : cuid, managed {
    @Common.Label: 'From Agent'
    @assert.integrity
    fromAgent         : Association to Agents not null;
    
    @Common.Label: 'To Agent'
    @assert.integrity
    toAgent           : Association to Agents not null;
    
    @Common.Label: 'Endorsement Amount'
    @assert.range: [1, 10]
    amount            : Integer not null;
    
    @Common.Label: 'Endorsement Reason'
    reason            : String(50) not null enum {
        EXCELLENT_COLLABORATION = 'EXCELLENT_COLLABORATION';
        TIMELY_ASSISTANCE = 'TIMELY_ASSISTANCE';
        HIGH_QUALITY_WORK = 'HIGH_QUALITY_WORK';
        KNOWLEDGE_SHARING = 'KNOWLEDGE_SHARING';
        PROBLEM_SOLVING = 'PROBLEM_SOLVING';
        INNOVATION = 'INNOVATION';
        MENTORING = 'MENTORING';
        RELIABILITY = 'RELIABILITY';
    };
    
    @Common.Label: 'Context'
    @Core.MediaType: 'application/json'
    context           : LargeString;
    
    @Common.Label: 'Related Workflow'
    workflow          : Association to Workflows;
    
    @Common.Label: 'Related Service Order'
    serviceOrder      : Association to ServiceOrders;
    
    @Common.Label: 'Expires At'
    expiresAt         : DateTime;
    
    @Common.Label: 'Is Reciprocal'
    @readonly
    isReciprocal      : Boolean default false;
    
    @Common.Label: 'Verification Hash'
    verificationHash  : String(64);
    
    @Common.Label: 'Blockchain Transaction'
    blockchainTx      : String(66);
}

// Reputation milestones and badges
entity ReputationMilestones : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent             : Association to Agents;
    
    @Common.Label: 'Milestone'
    @assert.range: [50, 100, 150, 200]
    milestone         : Integer not null;
    
    @Common.Label: 'Badge Name'
    badgeName         : String(20) not null enum {
        NEWCOMER = 'NEWCOMER';
        ESTABLISHED = 'ESTABLISHED';
        TRUSTED = 'TRUSTED';
        EXPERT = 'EXPERT';
        LEGENDARY = 'LEGENDARY';
    };
    
    @Common.Label: 'Achieved At'
    achievedAt        : DateTime not null;
    
    @Common.Label: 'Badge Metadata'
    @Core.MediaType: 'application/json'
    badgeMetadata     : String(500);
}

// Reputation recovery programs
entity ReputationRecovery : cuid, managed {
    @Common.Label: 'Agent'
    @assert.integrity
    agent             : Association to Agents;
    
    @Common.Label: 'Recovery Type'
    recoveryType      : String(30) not null enum {
        PROBATION_TASKS = 'PROBATION_TASKS';
        PEER_VOUCHING = 'PEER_VOUCHING';
        TRAINING_COMPLETION = 'TRAINING_COMPLETION';
        COMMUNITY_SERVICE = 'COMMUNITY_SERVICE';
    };
    
    @Common.Label: 'Status'
    status            : String(20) enum {
        PENDING = 'PENDING';
        IN_PROGRESS = 'IN_PROGRESS';
        COMPLETED = 'COMPLETED';
        FAILED = 'FAILED';
    } default 'PENDING';
    
    @Common.Label: 'Requirements'
    @Core.MediaType: 'application/json'
    requirements      : LargeString;
    
    @Common.Label: 'Progress'
    @Core.MediaType: 'application/json'
    progress          : LargeString;
    
    @Common.Label: 'Reputation Reward'
    reputationReward  : Integer default 20;
    
    @Common.Label: 'Started At'
    startedAt         : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt       : DateTime;
}

// Daily reputation limits tracking
entity DailyReputationLimits : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent             : Association to Agents;
    
    @Common.Label: 'Date'
    date              : Date not null;
    
    @Common.Label: 'Endorsements Given'
    endorsementsGiven : Integer default 0;
    
    @Common.Label: 'Points Given'
    pointsGiven       : Integer default 0;
    
    @Common.Label: 'Max Daily Limit'
    maxDailyLimit     : Integer default 50;
}

// Reputation analytics aggregation
@Analytics: { DCL: 'REPUTATION_ANALYTICS' }
entity ReputationAnalytics : cuid {
    @Common.Label: 'Agent'
    @assert.integrity
    agent                  : Association to Agents;
    
    @Common.Label: 'Period Start'
    periodStart           : Date not null;
    
    @Common.Label: 'Period End'
    periodEnd             : Date not null;
    
    @Common.Label: 'Starting Reputation'
    startingReputation    : Integer;
    
    @Common.Label: 'Ending Reputation'
    endingReputation      : Integer;
    
    @Common.Label: 'Total Earned'
    @Analytics.Measure: true
    totalEarned           : Integer default 0;
    
    @Common.Label: 'Total Lost'
    @Analytics.Measure: true
    totalLost             : Integer default 0;
    
    @Common.Label: 'Endorsements Received'
    @Analytics.Measure: true
    endorsementsReceived  : Integer default 0;
    
    @Common.Label: 'Endorsements Given'
    @Analytics.Measure: true
    endorsementsGiven     : Integer default 0;
    
    @Common.Label: 'Unique Endorsers'
    @Analytics.Measure: true
    uniqueEndorsers       : Integer default 0;
    
    @Common.Label: 'Average Transaction'
    @Analytics.Measure: true
    averageTransaction    : Decimal(5,2);
    
    @Common.Label: 'Task Success Rate'
    @Analytics.Measure: true
    taskSuccessRate       : Decimal(5,2);
    
    @Common.Label: 'Service Rating Average'
    @Analytics.Measure: true
    serviceRatingAverage  : Decimal(3,2);
}

// ================================
// AGENT 1 - DATA STANDARDIZATION ENTITIES
// ================================

// Data Standardization Tasks with performance indexes
@cds.persistence.index: [
    { elements: ['status', 'createdAt'] },
    { elements: ['sourceFormat', 'targetFormat'] },
    { elements: ['priority', 'status'] }
]
@Analytics: { DCL: 'STANDARDIZATION_TASK' }
@UI.HeaderInfo: {
    TypeName: 'Standardization Task',
    TypeNamePlural: 'Standardization Tasks',
    Title: { Value: taskName }
}
entity StandardizationTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Source Format'
    sourceFormat       : String(50) @mandatory enum { 
        CSV; JSON; XML; EXCEL; FIXED_WIDTH; AVRO; PARQUET; 
    };
    
    @Common.Label: 'Target Format'
    targetFormat       : String(50) @mandatory enum { 
        CSV; JSON; XML; PARQUET; AVRO; 
    };
    
    @Common.Label: 'Schema Template ID'
    schemaTemplateId   : String(50) enum { 
        RETAIL_PRODUCT; FINANCIAL_TRANSACTION; CUSTOMER_DATA; INVENTORY; SALES_ORDER; 
    };
    
    @Common.Label: 'Schema Validation'
    schemaValidation   : Boolean default true;
    
    @Common.Label: 'Data Type Validation'
    dataTypeValidation : Boolean default true;
    
    @Common.Label: 'Format Validation'
    formatValidation   : Boolean default true;
    
    @Common.Label: 'Processing Mode'
    processingMode     : String(20) enum { FULL; BATCH; } default 'FULL';
    
    @Common.Label: 'Batch Size'
    batchSize          : Integer default 1000;
    
    @Common.Label: 'Task Status'
    status             : String(20) enum { 
        DRAFT; PENDING; RUNNING; COMPLETED; FAILED; PAUSED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage       : String(50);
    
    @Common.Label: 'Processing Time (seconds)'
    processingTime     : Integer;
    
    @Common.Label: 'Records Processed'
    recordsProcessed   : Integer default 0;
    
    @Common.Label: 'Records Total'
    recordsTotal       : Integer default 0;
    
    @Common.Label: 'Error Count'
    errorCount         : Integer default 0;
    
    @Common.Label: 'Validation Results'
    @Core.MediaType: 'application/json'
    validationResults  : LargeString;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Agent'
    agent              : Association to Agents;
    
    @Common.Label: 'Standardization Rules'
    rules              : Composition of many StandardizationRules on rules.task = $self;
}

// Standardization Rules
entity StandardizationRules : cuid {
    @Common.Label: 'Task'
    task           : Association to StandardizationTasks;
    
    @Common.Label: 'Rule Name'
    name           : String(100) @mandatory;
    
    @Common.Label: 'Rule Type'
    type           : String(20) enum { 
        FIELD_MAPPING; DATA_TYPE_CONVERSION; VALUE_TRANSFORMATION; 
        VALIDATION_RULE; ENRICHMENT_RULE; 
    };
    
    @Common.Label: 'Source Field'
    sourceField    : String(100);
    
    @Common.Label: 'Target Field'
    targetField    : String(100);
    
    @Common.Label: 'Transformation'
    transformation : String(500);
    
    @Common.Label: 'Is Active'
    isActive       : Boolean default true;
    
    @Common.Label: 'Execution Order'
    executionOrder : Integer;
}

// ================================
// AGENT 3 - VECTOR PROCESSING ENTITIES
// ================================

// Vector Processing Tasks with performance indexes
@cds.persistence.index: [
    { elements: ['status', 'createdAt'] },
    { elements: ['dataType', 'embeddingModel'] },
    { elements: ['priority', 'status'] }
]
@Analytics: { DCL: 'VECTOR_TASK' }
@UI.HeaderInfo: {
    TypeName: 'Vector Processing Task',
    TypeNamePlural: 'Vector Processing Tasks',
    Title: { Value: taskName }
}
entity VectorProcessingTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Data Source'
    dataSource         : String(500) @mandatory;
    
    @Common.Label: 'Data Type'
    dataType           : String(50) @mandatory enum { 
        TEXT; IMAGE; AUDIO; VIDEO; DOCUMENT; CODE; 
    };
    
    @Common.Label: 'Embedding Model'
    embeddingModel     : String(100) @mandatory;
    
    @Common.Label: 'Model Provider'
    modelProvider      : String(50) enum { 
        OPENAI; HUGGINGFACE; COHERE; ANTHROPIC; GOOGLE; CUSTOM; 
    } default 'OPENAI';
    
    @Common.Label: 'Vector Database'
    vectorDatabase     : String(50) enum { 
        PINECONE; WEAVIATE; MILVUS; CHROMA; QDRANT; PGVECTOR; 
    } default 'PINECONE';
    
    @Common.Label: 'Index Type'
    indexType          : String(20) enum { 
        HNSW; IVF; FLAT; LSH; 
    } default 'HNSW';
    
    @Common.Label: 'Distance Metric'
    distanceMetric     : String(20) enum { 
        COSINE; EUCLIDEAN; DOT_PRODUCT; MANHATTAN; 
    } default 'COSINE';
    
    @Common.Label: 'Vector Dimensions'
    @assert.range: [128, 4096]
    dimensions         : Integer default 1536;
    
    @Common.Label: 'Chunk Size'
    @assert.range: [100, 8192]
    chunkSize          : Integer default 512;
    
    @Common.Label: 'Chunk Overlap'
    @assert.range: [0, 200]
    chunkOverlap       : Integer default 50;
    
    @Common.Label: 'Enable Normalization'
    normalization      : Boolean default true;
    
    @Common.Label: 'Use GPU Acceleration'
    useGPU             : Boolean default false;
    
    @Common.Label: 'Batch Size'
    batchSize          : Integer default 100;
    
    @Common.Label: 'Task Status'
    status             : String(20) enum { 
        DRAFT; PENDING; RUNNING; COMPLETED; FAILED; PAUSED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage       : String(50);
    
    @Common.Label: 'Processing Time (seconds)'
    processingTime     : Integer;
    
    @Common.Label: 'Vectors Generated'
    vectorsGenerated   : Integer default 0;
    
    @Common.Label: 'Chunks Processed'
    chunksProcessed    : Integer default 0;
    
    @Common.Label: 'Total Chunks'
    totalChunks        : Integer default 0;
    
    @Common.Label: 'Collection Name'
    collectionName     : String(100);
    
    @Common.Label: 'Index Size (MB)'
    indexSize          : Decimal(10,2) default 0;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Agent'
    agent              : Association to Agents;
    
    @Common.Label: 'Vector Collection'
    collection         : Association to VectorCollections;
    
    @Common.Label: 'Similarity Results'
    similarityResults  : Composition of many VectorSimilarityResults on similarityResults.task = $self;
}

// Vector Collections
entity VectorCollections : cuid, managed {
    @Common.Label: 'Collection Name'
    @Search.defaultSearchElement: true
    name               : String(100) @mandatory @assert.unique;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Vector Database'
    vectorDatabase     : String(50) @mandatory;
    
    @Common.Label: 'Embedding Model'
    embeddingModel     : String(100) @mandatory;
    
    @Common.Label: 'Vector Dimensions'
    dimensions         : Integer @mandatory;
    
    @Common.Label: 'Distance Metric'
    distanceMetric     : String(20) @mandatory;
    
    @Common.Label: 'Index Type'
    indexType          : String(20) @mandatory;
    
    @Common.Label: 'Total Vectors'
    totalVectors       : Integer default 0;
    
    @Common.Label: 'Index Size (MB)'
    indexSize          : Decimal(10,2) default 0;
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Is Optimized'
    isOptimized        : Boolean default false;
    
    @Common.Label: 'Last Optimized'
    lastOptimized      : DateTime;
    
    @Common.Label: 'Metadata Schema'
    @Core.MediaType: 'application/json'
    metadataSchema     : LargeString;
    
    @Common.Label: 'Processing Tasks'
    tasks              : Composition of many VectorProcessingTasks on tasks.collection = $self;
}

// Vector Similarity Search Results
entity VectorSimilarityResults : cuid {
    @Common.Label: 'Task'
    task               : Association to VectorProcessingTasks;
    
    @Common.Label: 'Query Text'
    queryText          : String(1000);
    
    @Common.Label: 'Query Vector'
    @Core.MediaType: 'application/json'
    queryVector        : LargeString;
    
    @Common.Label: 'Result Vector ID'
    resultVectorId     : String(100);
    
    @Common.Label: 'Similarity Score'
    @assert.range: [0.0, 1.0]
    similarityScore    : Decimal(5,4);
    
    @Common.Label: 'Distance'
    distance           : Decimal(10,6);
    
    @Common.Label: 'Result Content'
    @UI.MultiLineText: true
    resultContent      : String(2000);
    
    @Common.Label: 'Result Metadata'
    @Core.MediaType: 'application/json'
    resultMetadata     : LargeString;
    
    @Common.Label: 'Rank'
    rank               : Integer;
    
    @Common.Label: 'Search Timestamp'
    searchTimestamp    : DateTime;
}

// Vector Processing Jobs (for batch operations)
entity VectorProcessingJobs : cuid, managed {
    @Common.Label: 'Job Name'
    jobName            : String(100) @mandatory;
    
    @Common.Label: 'Job Type'
    jobType            : String(50) enum { 
        BATCH_EMBEDDING; INDEX_OPTIMIZATION; SIMILARITY_SEARCH; CLUSTERING; 
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum { 
        PENDING; RUNNING; COMPLETED; FAILED; CANCELLED; 
    } default 'PENDING';
    
    @Common.Label: 'Task IDs'
    @Core.MediaType: 'application/json'
    taskIds            : LargeString;
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Estimated Vectors'
    estimatedVectors   : Integer;
    
    @Common.Label: 'Use GPU'
    useGPU             : Boolean default false;
    
    @Common.Label: 'Parallel Processing'
    parallel           : Boolean default true;
    
    @Common.Label: 'Priority'
    priority           : String(10) default 'MEDIUM';
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Result Data'
    @Core.MediaType: 'application/json'
    resultData         : LargeString;
}

// Agent 4 - Calculation Validation Tasks
entity CalcValidationTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Calculation Expression'
    @UI.MultiLineText: true
    @mandatory
    expression         : String(2000);
    
    @Common.Label: 'Input Variables'
    @Core.MediaType: 'application/json'
    inputVariables     : LargeString;
    
    @Common.Label: 'Expected Result'
    expectedResult     : String(500);
    
    @Common.Label: 'Validation Method'
    validationMethod   : String(50) @mandatory enum { 
        SYMBOLIC; NUMERICAL; STATISTICAL; AI_POWERED; BLOCKCHAIN_CONSENSUS; HYBRID; 
    };
    
    @Common.Label: 'Precision Level'
    precisionLevel     : String(20) enum { 
        LOW; MEDIUM; HIGH; ULTRA_HIGH; 
    } default 'MEDIUM';
    
    @Common.Label: 'Tolerance'
    tolerance          : Decimal(10,8) default 0.0001;
    
    @Common.Label: 'Use Symbolic Math'
    useSymbolicMath    : Boolean default false;
    
    @Common.Label: 'Use Numerical Methods'
    useNumericalMethods : Boolean default true;
    
    @Common.Label: 'Use Statistical Analysis'
    useStatisticalAnalysis : Boolean default false;
    
    @Common.Label: 'Use AI Validation'
    useAIValidation    : Boolean default false;
    
    @Common.Label: 'Use Blockchain Consensus'
    useBlockchainConsensus : Boolean default false;
    
    @Common.Label: 'AI Model'
    aiModel            : String(100) enum { 
        GROK; GPT4; CLAUDE; GEMINI; CUSTOM; 
    } default 'GROK';
    
    @Common.Label: 'Consensus Validators'
    consensusValidators : Integer default 3;
    
    @Common.Label: 'Consensus Threshold'
    @assert.range: [0.5, 1.0]
    consensusThreshold : Decimal(3,2) default 0.67;
    
    @Common.Label: 'Status'
    status             : String(20) enum { 
        DRAFT; PENDING; VALIDATING; COMPLETED; FAILED; CANCELLED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage       : String(100);
    
    @Common.Label: 'Validation Time (ms)'
    validationTime     : Integer;
    
    @Common.Label: 'Computed Result'
    computedResult     : String(500);
    
    @Common.Label: 'Validation Score'
    @assert.range: [0.0, 1.0]
    validationScore    : Decimal(5,4);
    
    @Common.Label: 'Confidence Level'
    @assert.range: [0.0, 1.0]
    confidenceLevel    : Decimal(5,4);
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Validation Methods Used'
    @Core.MediaType: 'application/json'
    methodsUsed        : LargeString;
    
    @Common.Label: 'Intermediate Steps'
    @Core.MediaType: 'application/json'
    intermediateSteps  : LargeString;
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Agent'
    agent              : Association to Agents;
    
    @Common.Label: 'Validation Results'
    validationResults  : Composition of many CalcValidationResults on validationResults.task = $self;
}

// Calculation Validation Results (for detailed breakdown)
entity CalcValidationResults : cuid {
    @Common.Label: 'Task'
    task               : Association to CalcValidationTasks;
    
    @Common.Label: 'Method'
    method             : String(50) @mandatory;
    
    @Common.Label: 'Result'
    result             : String(500);
    
    @Common.Label: 'Is Correct'
    isCorrect          : Boolean;
    
    @Common.Label: 'Confidence Score'
    @assert.range: [0.0, 1.0]
    confidenceScore    : Decimal(5,4);
    
    @Common.Label: 'Processing Time (ms)'
    processingTime     : Integer;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(500);
    
    @Common.Label: 'Details'
    @Core.MediaType: 'application/json'
    details            : LargeString;
    
    @Common.Label: 'Validation Timestamp'
    validatedAt        : DateTime;
}

// Calculation Templates (reusable validation patterns)
entity CalcValidationTemplates : cuid, managed {
    @Common.Label: 'Template Name'
    @Search.defaultSearchElement: true
    templateName       : String(100) @mandatory @assert.unique;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Category'
    category           : String(50) enum { 
        ARITHMETIC; ALGEBRA; CALCULUS; STATISTICS; GEOMETRY; TRIGONOMETRY; 
        LINEAR_ALGEBRA; DIFFERENTIAL_EQUATIONS; FINANCIAL; PHYSICS; CHEMISTRY; 
    };
    
    @Common.Label: 'Expression Template'
    @UI.MultiLineText: true
    expressionTemplate : String(2000) @mandatory;
    
    @Common.Label: 'Variable Definitions'
    @Core.MediaType: 'application/json'
    variableDefinitions : LargeString;
    
    @Common.Label: 'Default Validation Method'
    defaultValidationMethod : String(50);
    
    @Common.Label: 'Recommended Precision'
    recommendedPrecision : String(20);
    
    @Common.Label: 'Example Usage'
    @UI.MultiLineText: true
    exampleUsage       : String(1000);
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Success Rate'
    @assert.range: [0.0, 1.0]
    successRate        : Decimal(5,4);
}

// Business Validations (implemented in service layer)
// 1. Agent reputation must be recalculated after each completed service order
// 2. Service orders cannot exceed provider's maxCallsPerDay limit
// 3. Messages must be routed based on agent availability and capabilities
// 4. Workflow executions must validate agent permissions before each step
// 5. Cross-chain transfers require bridge availability confirmation
// 6. Private channels require mutual consent from all participants
// 7. Reputation changes must be logged in ReputationTransactions
// 8. Peer endorsements must respect daily and weekly limits
// 9. Reputation recovery programs must track progress automatically

// Computed Fields (calculated in service layer)
// 1. Agent.performance.successRate = successfulTasks / totalTasks * 100
// 2. Service.utilizationRate = totalCalls / (maxCallsPerDay * daysSinceCreated)
// 3. WorkflowExecution.duration = completedAt - startedAt
// 4. NetworkStats.efficiency = (successfulMessages / totalMessages) * 100
// 5. Agent.currentBadge = getReputationBadge(reputation)
// 6. Agent.endorsementPower = getMaxEndorsementAmount(reputation)
// 7. Agent.reputationTrend = calculateTrend(last30DaysTransactions)
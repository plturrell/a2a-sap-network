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
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
using { a2a.reputation } from './reputationSchema';


// Import reputation entities into this namespace
using a2a.reputation.ReputationTransactions as ReputationTransactions;
using a2a.reputation.PeerEndorsements as PeerEndorsements;
using a2a.reputation.ReputationMilestones as ReputationMilestones;
using a2a.reputation.ReputationRecovery as ReputationRecovery;

// Custom Types for Type Safety
type BlockchainAddress : String(42);
type TransactionHash : String(66);
type PublicKey : String(130);
type SemVer : String(20);
type IPAddress : String(45);
type URL : String(500);
type Email : String(254);

// ================================
// REPUTATION SYSTEM ENTITIES - DEFINED FIRST FOR FORWARD REFERENCES
// ================================

// // Reputation transaction tracking
// entity ReputationTransactions : cuid, managed {
//     @Common.Label: 'Agent'
//     @assert.integrity
//     agent              : Association to Agents;
    
//     @Common.Label: 'Transaction Type'
//     transactionType    : String(50) not null enum {
//         TASK_COMPLETION = 'TASK_COMPLETION';
//         SERVICE_RATING = 'SERVICE_RATING';
//         PEER_ENDORSEMENT = 'PEER_ENDORSEMENT';
//         QUALITY_BONUS = 'QUALITY_BONUS';
//         PENALTY = 'PENALTY';
//         MILESTONE_BONUS = 'MILESTONE_BONUS';
//         RECOVERY_REWARD = 'RECOVERY_REWARD';
//         WORKFLOW_PARTICIPATION = 'WORKFLOW_PARTICIPATION';
//         COLLABORATION_BONUS = 'COLLABORATION_BONUS';
//     };
    
//     @Common.Label: 'Amount'
//     @assert.range: [-50, 50]
//     amount            : Integer not null;
    
//     @Common.Label: 'Reason'
//     reason            : String(200);
    
//     @Common.Label: 'Context'
//     @Core.MediaType: 'application/json'
//     context           : LargeString;
    
//     @Common.Label: 'Is Automated'
//     isAutomated       : Boolean default false;
    
//     @Common.Label: 'Created By Agent'
//     createdByAgent    : Association to Agents;
    
//     @Common.Label: 'Related Service Order'
//     serviceOrder      : Association to ServiceOrders;
    
//     @Common.Label: 'Related Workflow'
//     workflow          : Association to Workflows;
// }

// // Peer-to-peer endorsements
// entity PeerEndorsements : cuid, managed {
//     @Common.Label: 'From Agent'
//     @assert.integrity
//     fromAgent         : Association to Agents not null;
    
//     @Common.Label: 'To Agent'
//     @assert.integrity
//     toAgent           : Association to Agents not null;
    
//     @Common.Label: 'Endorsement Amount'
//     @assert.range: [1, 10]
//     amount            : Integer not null;
    
//     @Common.Label: 'Endorsement Reason'
//     reason            : String(50) not null enum {
//         EXCELLENT_COLLABORATION = 'EXCELLENT_COLLABORATION';
//         TIMELY_ASSISTANCE = 'TIMELY_ASSISTANCE';
//         HIGH_QUALITY_WORK = 'HIGH_QUALITY_WORK';
//         KNOWLEDGE_SHARING = 'KNOWLEDGE_SHARING';
//         PROBLEM_SOLVING = 'PROBLEM_SOLVING';
//         INNOVATION = 'INNOVATION';
//         MENTORING = 'MENTORING';
//         RELIABILITY = 'RELIABILITY';
//     };
    
//     @Common.Label: 'Context'
//     @Core.MediaType: 'application/json'
//     context           : LargeString;
    
//     @Common.Label: 'Related Workflow'
//     workflow          : Association to Workflows;
    
//     @Common.Label: 'Related Service Order'
//     serviceOrder      : Association to ServiceOrders;
    
//     @Common.Label: 'Expires At'
//     expiresAt         : DateTime;
    
//     @Common.Label: 'Is Reciprocal'
//     @readonly
//     isReciprocal      : Boolean default false;
    
//     @Common.Label: 'Verification Hash'
//     verificationHash  : String(64);
    
//     @Common.Label: 'Blockchain Transaction'
//     blockchainTx      : String(66);
// }

// // Reputation milestones and badges
// entity ReputationMilestones : cuid {
//     @Common.Label: 'Agent'
//     @assert.integrity
//     agent             : Association to Agents;
    
//     @Common.Label: 'Milestone'
//     @assert.range: [50, 100, 150, 200]
//     milestone         : Integer not null;
    
//     @Common.Label: 'Badge Name'
//     badgeName         : String(20) not null enum {
//         NEWCOMER = 'NEWCOMER';
//         ESTABLISHED = 'ESTABLISHED';
//         TRUSTED = 'TRUSTED';
//         EXPERT = 'EXPERT';
//         LEGENDARY = 'LEGENDARY';
//     };
    
//     @Common.Label: 'Achieved At'
//     achievedAt        : DateTime not null;
    
//     @Common.Label: 'Badge Metadata'
//     @Core.MediaType: 'application/json'
//     badgeMetadata     : String(500);
// }

// // Reputation recovery programs
// entity ReputationRecovery : cuid, managed {
//     @Common.Label: 'Agent'
//     @assert.integrity
//     agent             : Association to Agents;
    
//     @Common.Label: 'Recovery Type'
//     recoveryType      : String(30) not null enum {
//         PROBATION_TASKS = 'PROBATION_TASKS';
//         PEER_VOUCHING = 'PEER_VOUCHING';
//         TRAINING_COMPLETION = 'TRAINING_COMPLETION';
//         COMMUNITY_SERVICE = 'COMMUNITY_SERVICE';
//     };
    
//     @Common.Label: 'Status'
//     status            : String(20) enum {
//         PENDING = 'PENDING';
//         IN_PROGRESS = 'IN_PROGRESS';
//         COMPLETED = 'COMPLETED';
//         FAILED = 'FAILED';
//     } default 'PENDING';
    
//     @Common.Label: 'Requirements'
//     @Core.MediaType: 'application/json'
//     requirements      : LargeString;
    
//     @Common.Label: 'Progress'
//     @Core.MediaType: 'application/json'
//     progress          : LargeString;
    
//     @Common.Label: 'Reputation Reward'
//     reputationReward  : Integer default 20;
    
//     @Common.Label: 'Started At'
//     startedAt         : DateTime;
    
//     @Common.Label: 'Completed At'
//     completedAt       : DateTime;
// }

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
    workflowId.workflowName as workflowName,
    status,
    startTime as startedAt,
    endTime as completedAt,
    durationMinutes as gasUsed
} order by startTime desc;

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
    
        };
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
    
        };
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
    
        };
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
    
        };
    @Common.Label: 'Batch Size'
    batchSize          : Integer default 1000;
    
    @Common.Label: 'Task Status'
    status             : String(20) enum { 
        DRAFT; PENDING; RUNNING; COMPLETED; FAILED; PAUSED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
        };
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
    
        };
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
 RUNNING; COMPLETED; FAILED; CANCELLED; 
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
 MEDIUM; HIGH; ULTRA_HIGH;
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
    
        };
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

// Agent 5 - QA Validation Tasks
entity QaValidationTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Data Product ID'
    dataProductId      : String(100) @mandatory;
    
    @Common.Label: 'ORD Registry URL'
    ordRegistryUrl     : String(500);
    
    @Common.Label: 'Validation Type'
    validationType     : String(50) @mandatory enum { 
        FACTUALITY; QUALITY_ASSURANCE; COMPLIANCE; END_TO_END; REGRESSION; INTEGRATION; 
    };
    
    @Common.Label: 'QA Scope'
    qaScope            : String(50) enum { 
        DATA_INTEGRITY; BUSINESS_RULES; REGULATORY_COMPLIANCE; PERFORMANCE; SECURITY; COMPLETENESS;
    } default 'DATA_INTEGRITY';
    
    @Common.Label: 'Test Generation Method'
    testGenerationMethod : String(50) enum { 
        DYNAMIC_SIMPLEQA; STATIC_RULES; HYBRID; CUSTOM_TEMPLATE;
    } default 'DYNAMIC_SIMPLEQA';
    
    @Common.Label: 'SimpleQA Test Count'
    simpleQaTestCount  : Integer default 10;
    
    @Common.Label: 'Quality Threshold'
    @assert.range: [0.0, 1.0]
    qualityThreshold   : Decimal(3,2) default 0.95;
    
    @Common.Label: 'Enable Factuality Testing'
    enableFactualityTesting : Boolean default true;
    
    @Common.Label: 'Enable Compliance Check'
    enableComplianceCheck : Boolean default false;
    
    @Common.Label: 'Enable Vector Similarity'
    enableVectorSimilarity : Boolean default false;
    
    @Common.Label: 'Vector Similarity Threshold'
    @assert.range: [0.0, 1.0]
    vectorSimilarityThreshold : Decimal(3,2) default 0.85;
    
    @Common.Label: 'Approval Required'
    approvalRequired   : Boolean default false;
    
    @Common.Label: 'Auto Approve Threshold'
    @assert.range: [0.0, 1.0]
    autoApproveThreshold : Decimal(3,2) default 0.98;
    
    @Common.Label: 'Processing Pipeline'
    @Core.MediaType: 'application/json'
    processingPipeline : LargeString;
    
    @Common.Label: 'QA Rules'
    @Core.MediaType: 'application/json'
    qaRules            : LargeString;
    
    @Common.Label: 'Status'
    status             : String(20) enum { 
        DRAFT; PENDING; DISCOVERING; GENERATING_TESTS; VALIDATING; COMPLETED; FAILED; APPROVED; REJECTED; 
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum { LOW; MEDIUM; HIGH; URGENT; } default 'MEDIUM';
    
        };
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage       : String(100);
    
    @Common.Label: 'Validation Time (ms)'
    validationTime     : Integer;
    
    @Common.Label: 'Tests Generated'
    testsGenerated     : Integer default 0;
    
    @Common.Label: 'Tests Passed'
    testsPassed        : Integer default 0;
    
    @Common.Label: 'Tests Failed'
    testsFailed        : Integer default 0;
    
    @Common.Label: 'Quality Score'
    @assert.range: [0.0, 1.0]
    qualityScore       : Decimal(5,4);
    
    @Common.Label: 'Factuality Score'
    @assert.range: [0.0, 1.0]
    factualityScore    : Decimal(5,4);
    
    @Common.Label: 'Compliance Score'
    @assert.range: [0.0, 1.0]
    complianceScore    : Decimal(5,4);
    
    @Common.Label: 'Overall Score'
    @assert.range: [0.0, 1.0]
    overallScore       : Decimal(5,4);
    
    @Common.Label: 'Approval Status'
    approvalStatus     : String(20) enum { 
 APPROVED; REJECTED; AUTO_APPROVED; CONDITIONAL; 
    };
    
    @Common.Label: 'Approved By'
    approvedBy         : String(100);
    
    @Common.Label: 'Approved At'
    approvedAt         : DateTime;
    
    @Common.Label: 'Rejection Reason'
    rejectionReason    : String(1000);
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Validation Report'
    @Core.MediaType: 'application/json'
    validationReport   : LargeString;
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Agent'
    agent              : Association to Agents;
    
    @Common.Label: 'QA Test Results'
    qaTestResults      : Composition of many QaTestResults on qaTestResults.task = $self;
}

// QA Test Results (for individual test details)
entity QaTestResults : cuid {
    @Common.Label: 'Task'
    task               : Association to QaValidationTasks;
    
    @Common.Label: 'Test Type'
    testType           : String(50) @mandatory;
    
    @Common.Label: 'Test Name'
    testName           : String(200);
    
    @Common.Label: 'Test Question'
    @UI.MultiLineText: true
    testQuestion       : String(1000);
    
    @Common.Label: 'Expected Answer'
    expectedAnswer     : String(500);
    
    @Common.Label: 'Actual Answer'
    actualAnswer       : String(500);
    
    @Common.Label: 'Is Passed'
    isPassed           : Boolean;
    
    @Common.Label: 'Confidence Score'
    @assert.range: [0.0, 1.0]
    confidenceScore    : Decimal(5,4);
    
    @Common.Label: 'Processing Time (ms)'
    processingTime     : Integer;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(500);
    
    @Common.Label: 'Test Data'
    @Core.MediaType: 'application/json'
    testData           : LargeString;
    
    @Common.Label: 'Test Results'
    @Core.MediaType: 'application/json'
    testResults        : LargeString;
    
    @Common.Label: 'Executed At'
    executedAt         : DateTime;
}

// QA Validation Rules (reusable validation patterns)
entity QaValidationRules : cuid, managed {
    @Common.Label: 'Rule Name'
    @Search.defaultSearchElement: true
    ruleName           : String(100) @mandatory @assert.unique;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Rule Category'
    ruleCategory       : String(50) enum { 
        DATA_QUALITY; BUSINESS_LOGIC; REGULATORY; SECURITY; PERFORMANCE; COMPLETENESS; 
    };
    
    @Common.Label: 'Rule Type'
    ruleType           : String(50) enum { 
        SIMPLE_QA; SQL_QUERY; PYTHON_SCRIPT; REST_API_CHECK; REGEX_PATTERN; THRESHOLD_CHECK; 
    };
    
    @Common.Label: 'Rule Expression'
    @UI.MultiLineText: true
    ruleExpression     : String(2000) @mandatory;
    
    @Common.Label: 'Expected Result'
    expectedResult     : String(500);
    
    @Common.Label: 'Severity Level'
    severityLevel      : String(20) enum { 
 MEDIUM; HIGH; CRITICAL; 
    } default 'MEDIUM';
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Is Blocking'
    isBlocking         : Boolean default false;
    
    @Common.Label: 'Execution Order'
    executionOrder     : Integer default 100;
    
    @Common.Label: 'Timeout (seconds)'
    timeoutSeconds     : Integer default 30;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Success Rate'
    @assert.range: [0.0, 1.0]
    successRate        : Decimal(5,4);
    
    @Common.Label: 'Tags'
    tags               : String(500);
}

// QA Approval Workflows
entity QaApprovalWorkflows : cuid, managed {
    @Common.Label: 'Workflow Name'
    @Search.defaultSearchElement: true
    workflowName       : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Trigger Conditions'
    @Core.MediaType: 'application/json'
    triggerConditions  : LargeString;
    
    @Common.Label: 'Approval Steps'
    @Core.MediaType: 'application/json'
    approvalSteps      : LargeString;
    
    @Common.Label: 'Auto Approval Rules'
    @Core.MediaType: 'application/json'
    autoApprovalRules  : LargeString;
    
    @Common.Label: 'Escalation Rules'
    @Core.MediaType: 'application/json'
    escalationRules    : LargeString;
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Default Workflow'
    isDefault          : Boolean default false;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
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

// ============================================
// AGENT 6 - QUALITY CONTROL & WORKFLOW ROUTING
// ============================================

// Quality Control Tasks
entity QualityControlTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    taskName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Quality Gate'
    qualityGate        : String(100) @mandatory;
    
    @Common.Label: 'Data Source'
    dataSource         : String(200);
    
    @Common.Label: 'Processing Pipeline'
    processingPipeline : String(200);
    
    @Common.Label: 'Status'
    @assert.range
    status             : String(20) enum {
        DRAFT; PENDING; ASSESSING; ROUTING; COMPLETED; FAILED; CANCELLED;
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    @assert.range
    priority           : String(20) enum {
        NORMAL; HIGH; CRITICAL;
    } default 'NORMAL';
    
    @Common.Label: 'Overall Quality Score'
    @assert.range: [0, 100]
    overallQuality     : Decimal(5, 2);
    
    @Common.Label: 'Trust Score'
    @assert.range: [0, 100]
    trustScore         : Decimal(5, 2);
    
    @Common.Label: 'Issues Found'
    issuesFound        : Integer default 0;
    
    @Common.Label: 'Routing Decision'
    routingDecision    : String(20) enum {
        PROCEED; REROUTE; HOLD; ESCALATE; REJECT;
    };
    
    @Common.Label: 'Target Agent'
    targetAgent        : String(50);
    
    @Common.Label: 'Routing Confidence'
    @assert.range: [0, 100]
    routingConfidence  : Decimal(5, 2);
    
    @Common.Label: 'Assessment Duration'
    assessmentDuration : Integer;
    
    @Common.Label: 'Workflow Optimization'
    workflowOptimized  : Boolean default false;
    
    @Common.Label: 'Auto Routed'
    autoRouted         : Boolean default false;
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Quality Components'
    @Core.MediaType: 'application/json'
    qualityComponents  : LargeString;
    
    @Common.Label: 'Assessment Results'
    @Core.MediaType: 'application/json'
    assessmentResults  : LargeString;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Agent'
    agent              : Association to Agents;
    
    @Common.Label: 'Quality Metrics'
    qualityMetrics     : Composition of many QualityMetrics on qualityMetrics.task = $self;
    
    @Common.Label: 'Routing Rules'
    routingRules       : Composition of many RoutingRules on routingRules.task = $self;
    
    @Common.Label: 'Trust Verifications'
    trustVerifications : Composition of many TrustVerifications on trustVerifications.task = $self;
}

// Quality Metrics
entity QualityMetrics : cuid {
    @Common.Label: 'Task'
    task               : Association to QualityControlTasks not null;
    
    @Common.Label: 'Component'
    component          : String(50) @mandatory enum {
        COMPLIANCE; PERFORMANCE; SECURITY; RELIABILITY; 
        USABILITY; MAINTAINABILITY; DATA_QUALITY; COMPLETENESS;
    };
    
    @Common.Label: 'Score'
    @assert.range: [0, 100]
    score              : Decimal(5, 2) @mandatory;
    
    @Common.Label: 'Weight'
    @assert.range: [0, 100]
    weight             : Decimal(5, 2) default 100;
    
    @Common.Label: 'Issues Count'
    issuesCount        : Integer default 0;
    
    @Common.Label: 'Critical Issues'
    criticalIssues     : Integer default 0;
    
    @Common.Label: 'Assessment Details'
    @Core.MediaType: 'application/json'
    assessmentDetails  : LargeString;
    
    @Common.Label: 'Recommendations'
    @UI.MultiLineText: true
    recommendations    : String(2000);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime @cds.on.insert: $now;
}

// Routing Rules
entity RoutingRules : cuid, managed {
    @Common.Label: 'Rule Name'
    ruleName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Rule Type'
    ruleType           : String(50) enum {
        QUALITY_BASED; LOAD_BASED; CAPABILITY_BASED; 
        TRUST_BASED; COST_BASED; PRIORITY_BASED;
    } default 'QUALITY_BASED';
    
    @Common.Label: 'Condition Expression'
    @UI.MultiLineText: true
    conditionExpression : String(1000) @mandatory;
    
    @Common.Label: 'Target Agent'
    targetAgent        : String(50);
    
    @Common.Label: 'Priority'
    priority           : Integer default 100;
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Quality Threshold'
    @assert.range: [0, 100]
    qualityThreshold   : Decimal(5, 2);
    
    @Common.Label: 'Trust Threshold'
    @assert.range: [0, 100]
    trustThreshold     : Decimal(5, 2);
    
    @Common.Label: 'Success Rate'
    @assert.range: [0, 100]
    successRate        : Decimal(5, 2) default 0;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Last Applied'
    lastApplied        : DateTime;
    
    @Common.Label: 'Task'
    task               : Association to QualityControlTasks;
}

// Trust Verifications
entity TrustVerifications : cuid {
    @Common.Label: 'Task'
    task               : Association to QualityControlTasks not null;
    
    @Common.Label: 'Verification Type'
    @mandatory
    verificationType   : String(50) enum {
        BLOCKCHAIN; REPUTATION; INTEGRITY; CONSENSUS; 
        SIGNATURE; HISTORICAL; PEER_REVIEW;
    };
    
    @Common.Label: 'Verification Status'
    status             : String(20) enum {
        PENDING; VERIFIED; FAILED; SUSPICIOUS;
    } default 'PENDING';
    
    @Common.Label: 'Trust Score'
    @assert.range: [0, 100]
    trustScore         : Decimal(5, 2);
    
    @Common.Label: 'Blockchain Hash'
    blockchainHash     : String(100);
    
    @Common.Label: 'Consensus Participants'
    consensusParticipants : Integer;
    
    @Common.Label: 'Consensus Agreement'
    @assert.range: [0, 100]
    consensusAgreement : Decimal(5, 2);
    
    @Common.Label: 'Verification Details'
    @Core.MediaType: 'application/json'
    verificationDetails : LargeString;
    
    @Common.Label: 'Anomalies Detected'
    anomaliesDetected  : Boolean default false;
    
    @Common.Label: 'Risk Level'
    riskLevel          : String(20) enum {
        LOW; MEDIUM; HIGH; CRITICAL;
    } default 'LOW';
    
    @Common.Label: 'Verified At'
    verifiedAt         : DateTime;
    
    @Common.Label: 'Verified By'
    verifiedBy         : String(100);
}

// Quality Gates Configuration
entity QualityGates : cuid, managed {
    @Common.Label: 'Gate Name'
    @Search.defaultSearchElement: true
    gateName           : String(100) @mandatory;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Gate Type'
    gateType           : String(50) enum {
        ENTRY; EXIT; CHECKPOINT; MILESTONE;
    } default 'CHECKPOINT';
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Min Quality Score'
    @assert.range: [0, 100]
    minQualityScore    : Decimal(5, 2) default 80;
    
    @Common.Label: 'Max Issues Allowed'
    maxIssuesAllowed   : Integer default 5;
    
    @Common.Label: 'Min Trust Score'
    @assert.range: [0, 100]
    minTrustScore      : Decimal(5, 2) default 75;
    
    @Common.Label: 'Required Components'
    @Core.MediaType: 'application/json'
    requiredComponents : String(500);
    
    @Common.Label: 'Evaluation Criteria'
    @Core.MediaType: 'application/json'
    evaluationCriteria : LargeString;
    
    @Common.Label: 'Auto Escalate'
    autoEscalate       : Boolean default false;
    
    @Common.Label: 'Escalation Threshold'
    escalationThreshold : Integer default 3;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Success Rate'
    @assert.range: [0, 100]
    successRate        : Decimal(5, 2) default 0;
}

// Workflow Optimizations
entity WorkflowOptimizations : cuid, managed {
    @Common.Label: 'Optimization Name'
    optimizationName   : String(100) @mandatory;
    
    @Common.Label: 'Workflow Stage'
    workflowStage      : String(100);
    
    @Common.Label: 'Optimization Type'
    optimizationType   : String(50) enum {
        BOTTLENECK_REMOVAL; PARALLEL_PROCESSING; RESOURCE_ALLOCATION;
        ROUTE_OPTIMIZATION; CACHE_IMPLEMENTATION; BATCH_PROCESSING;
    };
    
    @Common.Label: 'Current Performance'
    currentPerformance : Decimal(10, 2);
    
    @Common.Label: 'Expected Improvement'
    expectedImprovement : Decimal(5, 2);
    
    @Common.Label: 'Actual Improvement'
    actualImprovement  : Decimal(5, 2);
    
    @Common.Label: 'Implementation Status'
    status             : String(20) enum {
        PROPOSED; APPROVED; IMPLEMENTING; COMPLETED; ROLLED_BACK;
    } default 'PROPOSED';
    
    @Common.Label: 'Risk Assessment'
    riskAssessment     : String(20) enum {
        LOW; MEDIUM; HIGH;
    } default 'MEDIUM';
    
    @Common.Label: 'Implementation Details'
    @Core.MediaType: 'application/json'
    implementationDetails : LargeString;
    
    @Common.Label: 'Impact Analysis'
    @Core.MediaType: 'application/json'
    impactAnalysis     : LargeString;
    
    @Common.Label: 'Applied At'
    appliedAt          : DateTime;
    
    @Common.Label: 'Rolled Back At'
    rolledBackAt       : DateTime;
    
    @Common.Label: 'Task'
    task               : Association to QualityControlTasks;
}

// Agent 7 Entities - Agent Management & Orchestration System
// Manages and coordinates multiple AI agents in the distributed network
entity RegisteredAgents : cuid, managed {
    @Common.Label: 'Agent Name'
    @Search.defaultSearchElement: true
    @mandatory
    agentName          : String(100);
    
    @Common.Label: 'Agent Type'
    @mandatory
    agentType          : String(50) enum {
        BLOCKCHAIN; SERVICE_MANAGEMENT; NETWORK_INTELLIGENCE;
        WORKFLOW_AUTOMATION; QUALITY_CONTROL; SECURITY_MONITOR;
        PERFORMANCE_OPTIMIZER; DATA_PROCESSOR; ML_MODEL; CUSTOM;
    };
    
    @Common.Label: 'Agent Version'
    @mandatory
    agentVersion       : String(20);
    
    @Common.Label: 'Endpoint URL'
    @mandatory
    endpointUrl        : String(500);
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        REGISTERING; ACTIVE; INACTIVE; SUSPENDED; MAINTENANCE; FAILED; DECOMMISSIONED;
    } default 'REGISTERING';
    
    @Common.Label: 'Health Status'
    healthStatus       : String(20) enum {
        HEALTHY; DEGRADED; UNHEALTHY; UNKNOWN;
    } default 'UNKNOWN';
    
    @Common.Label: 'Capabilities'
    @Core.MediaType: 'application/json'
    capabilities       : LargeString;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Performance Score'
    @assert.range: [0, 100]
    performanceScore   : Decimal(5, 2) default 0;
    
    @Common.Label: 'Response Time (ms)'
    responseTime       : Integer;
    
    @Common.Label: 'Throughput (req/s)'
    throughput         : Decimal(10, 2);
    
    @Common.Label: 'Error Rate'
    @assert.range: [0, 100]
    errorRate          : Decimal(5, 2) default 0;
    
    @Common.Label: 'Last Health Check'
    lastHealthCheck    : DateTime;
    
    @Common.Label: 'Registration Date'
    registrationDate   : DateTime;
    
    @Common.Label: 'Deactivation Date'
    deactivationDate   : DateTime;
    
    @Common.Label: 'Load Balance Weight'
    @assert.range: [0, 100]
    loadBalanceWeight  : Integer default 50;
    
    @Common.Label: 'Priority Level'
    priority           : Integer default 5;
    
    @Common.Label: 'Tags'
    @Core.MediaType: 'application/json'
    tags               : String(500);
    
    @Common.Label: 'Notes'
    @UI.MultiLineText: true
    notes              : String(2000);
    
    @Common.Label: 'Management Tasks'
    managementTasks    : Composition of many ManagementTasks on managementTasks.agent = $self;
    
    @Common.Label: 'Health Checks'
    healthChecks       : Composition of many AgentHealthChecks on healthChecks.agent = $self;
    
    @Common.Label: 'Performance Metrics'
    performanceMetrics : Composition of many AgentPerformanceMetrics on performanceMetrics.agent = $self;
}

// Management Tasks for agents
entity ManagementTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    @mandatory
    taskName           : String(100);
    
    @Common.Label: 'Task Type'
    @mandatory
    taskType           : String(50) enum {
        HEALTH_CHECK; PERFORMANCE_TEST; CONFIGURATION_UPDATE;
        RESTART; BACKUP; RESTORE; UPGRADE; DIAGNOSTIC;
        BULK_OPERATION; COORDINATION_TASK;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        SCHEDULED; PENDING; RUNNING; COMPLETED; FAILED; CANCELLED; TIMEOUT;
    } default 'SCHEDULED';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum {
        LOW; NORMAL; HIGH; CRITICAL;
    } default 'NORMAL';
    
    @Common.Label: 'Agent'
    agent              : Association to RegisteredAgents;
    
    @Common.Label: 'Target Agents'
    @Core.MediaType: 'application/json'
    targetAgents       : String(2000);
    
    @Common.Label: 'Parameters'
    @Core.MediaType: 'application/json'
    parameters         : LargeString;
    
    @Common.Label: 'Schedule Type'
    scheduleType       : String(20) enum {
        IMMEDIATE; SCHEDULED; RECURRING;
    } default 'IMMEDIATE';
    
    @Common.Label: 'Scheduled Time'
    scheduledTime      : DateTime;
    
    @Common.Label: 'Recurrence Pattern'
    recurrencePattern  : String(100);
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration (ms)'
    duration           : Integer;
    
    @Common.Label: 'Progress'
    @assert.range: [0, 100]
    progress           : Integer default 0;
    
    @Common.Label: 'Result'
    @Core.MediaType: 'application/json'
    result             : LargeString;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(2000);
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Notification Sent'
    notificationSent   : Boolean default false;
    
    @Common.Label: 'Rollback Available'
    rollbackAvailable  : Boolean default false;
}

// Agent Health Checks
entity AgentHealthChecks : cuid {
    @Common.Label: 'Check ID'
    @mandatory
    checkId            : String(50);
    
    @Common.Label: 'Agent'
    agent              : Association to RegisteredAgents not null;
    
    @Common.Label: 'Check Type'
    checkType          : String(50) enum {
        PING; HTTP_GET; HTTP_POST; HEARTBEAT; DIAGNOSTIC;
        MEMORY_CHECK; CPU_CHECK; DISK_CHECK; CUSTOM;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        PASS; FAIL; WARNING; TIMEOUT; ERROR;
    };
    
    @Common.Label: 'Response Time (ms)'
    responseTime       : Integer;
    
    @Common.Label: 'Status Code'
    statusCode         : Integer;
    
    @Common.Label: 'Check Details'
    @Core.MediaType: 'application/json'
    checkDetails       : LargeString;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(2000);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Alert Triggered'
    alertTriggered     : Boolean default false;
    
    @Common.Label: 'Recommendations'
    @Core.MediaType: 'application/json'
    recommendations    : String(2000);
}

// Agent Performance Metrics
entity AgentPerformanceMetrics : cuid {
    @Common.Label: 'Metric ID'
    @mandatory
    metricId           : String(50);
    
    @Common.Label: 'Agent'
    agent              : Association to RegisteredAgents not null;
    
    @Common.Label: 'Metric Type'
    metricType         : String(50) enum {
        RESPONSE_TIME; THROUGHPUT; ERROR_RATE; CPU_USAGE;
        MEMORY_USAGE; QUEUE_LENGTH; SUCCESS_RATE; LATENCY;
    };
    
    @Common.Label: 'Value'
    value              : Decimal(20, 5);
    
    @Common.Label: 'Unit'
    unit               : String(20);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Time Window'
    timeWindow         : String(20) enum {
        MINUTE; HOUR; DAY; WEEK; MONTH;
    } default 'MINUTE';
    
    @Common.Label: 'Min Value'
    minValue           : Decimal(20, 5);
    
    @Common.Label: 'Max Value'
    maxValue           : Decimal(20, 5);
    
    @Common.Label: 'Average Value'
    averageValue       : Decimal(20, 5);
    
    @Common.Label: 'Percentile 95'
    percentile95       : Decimal(20, 5);
    
    @Common.Label: 'Percentile 99'
    percentile99       : Decimal(20, 5);
    
    @Common.Label: 'Trend'
    trend              : String(20) enum {
        IMPROVING; STABLE; DEGRADING;
    };
    
    @Common.Label: 'Anomaly Detected'
    anomalyDetected    : Boolean default false;
    
    @Common.Label: 'Benchmark Comparison'
    @assert.range: [-100, 100]
    benchmarkComparison : Decimal(5, 2);
}

// Agent Coordination
entity AgentCoordination : cuid, managed {
    @Common.Label: 'Coordination Name'
    @Search.defaultSearchElement: true
    @mandatory
    coordinationName   : String(100);
    
    @Common.Label: 'Coordination Type'
    coordinationType   : String(50) enum {
        WORKFLOW; CHAIN; PARALLEL; CONDITIONAL; LOAD_BALANCE;
        FAILOVER; ROUND_ROBIN; CONSENSUS;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        DRAFT; ACTIVE; PAUSED; COMPLETED; FAILED;
    } default 'DRAFT';
    
    @Common.Label: 'Primary Agent'
    primaryAgent       : Association to RegisteredAgents;
    
    @Common.Label: 'Participating Agents'
    @Core.MediaType: 'application/json'
    participatingAgents : LargeString;
    
    @Common.Label: 'Coordination Rules'
    @Core.MediaType: 'application/json'
    coordinationRules  : LargeString;
    
    @Common.Label: 'Load Balance Strategy'
    loadBalanceStrategy : String(50);
    
    @Common.Label: 'Failover Config'
    @Core.MediaType: 'application/json'
    failoverConfig     : String(2000);
    
    @Common.Label: 'Performance Target'
    performanceTarget  : Decimal(10, 2);
    
    @Common.Label: 'Current Performance'
    currentPerformance : Decimal(10, 2);
    
    @Common.Label: 'Success Rate'
    @assert.range: [0, 100]
    successRate        : Decimal(5, 2) default 0;
    
    @Common.Label: 'Total Executions'
    totalExecutions    : Integer default 0;
    
    @Common.Label: 'Failed Executions'
    failedExecutions   : Integer default 0;
    
    @Common.Label: 'Average Duration'
    averageDuration    : Integer;
    
    @Common.Label: 'Last Execution'
    lastExecution      : DateTime;
    
    @Common.Label: 'Next Scheduled'
    nextScheduled      : DateTime;
}

// Bulk Operations Log
entity BulkOperations : cuid, managed {
    @Common.Label: 'Operation Name'
    @mandatory
    operationName      : String(100);
    
    @Common.Label: 'Operation Type'
    operationType      : String(50) enum {
        UPDATE_CONFIG; RESTART_AGENTS; HEALTH_CHECK;
        PERFORMANCE_TEST; BACKUP; UPGRADE; DEACTIVATE;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        PREPARING; EXECUTING; COMPLETED; PARTIALLY_COMPLETED; FAILED; ROLLED_BACK;
    } default 'PREPARING';
    
    @Common.Label: 'Target Count'
    targetCount        : Integer;
    
    @Common.Label: 'Successful Count'
    successfulCount    : Integer default 0;
    
    @Common.Label: 'Failed Count'
    failedCount        : Integer default 0;
    
    @Common.Label: 'Progress'
    @assert.range: [0, 100]
    progress           : Integer default 0;
    
    @Common.Label: 'Operation Details'
    @Core.MediaType: 'application/json'
    operationDetails   : LargeString;
    
    @Common.Label: 'Results'
    @Core.MediaType: 'application/json'
    results            : LargeString;
    
    @Common.Label: 'Rollback Data'
    @Core.MediaType: 'application/json'
    rollbackData       : LargeString;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration (ms)'
    duration           : Integer;
    
    @Common.Label: 'Initiated By'
    initiatedBy        : String(100);
    
    @Common.Label: 'Approval Required'
    approvalRequired   : Boolean default false;
    
    @Common.Label: 'Approved By'
    approvedBy         : String(100);
    
    @Common.Label: 'Approval Time'
    approvalTime       : DateTime;
}

// Agent 8 Entities - Data Management Agent
// Comprehensive data storage, caching, and management system
entity DataTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    @mandatory
    taskName           : String(100);
    
    @Common.Label: 'Task Type'
    @mandatory
    taskType           : String(50) enum {
        STORE; RETRIEVE; UPDATE; DELETE; BACKUP; RESTORE;
        MIGRATE; COMPRESS; CACHE; VALIDATE; OPTIMIZE; ARCHIVE;
    };
    
    @Common.Label: 'Data Source'
    dataSource         : String(500);
    
    @Common.Label: 'Data Target'
    dataTarget         : String(500);
    
    @Common.Label: 'Data Format'
    dataFormat         : String(50) enum {
        JSON; XML; CSV; PARQUET; AVRO; BINARY; TEXT; IMAGE; DOCUMENT;
    };
    
    @Common.Label: 'Storage Backend'
    @mandatory
    storageBackend     : String(50) enum {
        HANA; SQLITE; S3; REDIS; MEMORY; FILE_SYSTEM; DISTRIBUTED; HYBRID;
    };
    
    @Common.Label: 'Cache Strategy'
    cacheStrategy      : String(50) enum {
        NONE; MEMORY_ONLY; REDIS_ONLY; MULTI_TIER; WRITE_THROUGH; WRITE_BACK;
    };
    
    @Common.Label: 'Compression'
    compressionEnabled : Boolean default false;
    
    @Common.Label: 'Compression Type'
    compressionType    : String(30) enum {
        GZIP; LZ4; SNAPPY; BROTLI; ZSTD;
    };
    
    @Common.Label: 'Encryption Enabled'
    encryptionEnabled  : Boolean default false;
    
    @Common.Label: 'Encryption Algorithm'
    encryptionAlgorithm : String(30) enum {
        AES_256; AES_128; RSA; CHACHA20;
    };
    
    @Common.Label: 'Versioning Enabled'
    versioningEnabled  : Boolean default false;
    
    @Common.Label: 'Version Retention'
    versionRetention   : Integer default 10;
    
    @Common.Label: 'Partition Strategy'
    partitionStrategy  : String(50) enum {
        NONE; DATE_BASED; SIZE_BASED; HASH_BASED; RANGE_BASED;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        DRAFT; QUEUED; PROCESSING; COMPLETED; FAILED; CANCELLED; PAUSED;
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(10) enum {
 NORMAL; HIGH; CRITICAL;
    } default 'NORMAL';
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Data Size (bytes)'
    dataSize           : Integer64;
    
    @Common.Label: 'Processed Size (bytes)'
    processedSize      : Integer64 default 0;
    
    @Common.Label: 'Processing Speed (MB/s)'
    processingSpeed    : Decimal(10, 2);
    
    @Common.Label: 'Error Count'
    errorCount         : Integer default 0;
    
    @Common.Label: 'Warning Count'
    warningCount       : Integer default 0;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Performance Metrics'
    @Core.MediaType: 'application/json'
    performanceMetrics : LargeString;
    
    @Common.Label: 'Error Log'
    @Core.MediaType: 'application/json'
    errorLog           : LargeString;
    
    @Common.Label: 'Scheduled Time'
    scheduledTime      : DateTime;
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Estimated Completion'
    estimatedCompletion : DateTime;
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Checkpoint Data'
    @Core.MediaType: 'application/json'
    checkpointData     : LargeString;
    
    @Common.Label: 'Storage Utilization'
    storageUtilizations : Composition of many StorageUtilizations on storageUtilizations.task = $self;
    
    @Common.Label: 'Cache Operations'
    cacheOperations    : Composition of many CacheOperations on cacheOperations.task = $self;
    
    @Common.Label: 'Data Versions'
    dataVersions       : Composition of many DataVersions on dataVersions.task = $self;
}

// Storage Backend Management
entity StorageBackends : cuid, managed {
    @Common.Label: 'Backend Name'
    @Search.defaultSearchElement: true
    @mandatory
    backendName        : String(100);
    
    @Common.Label: 'Backend Type'
    @mandatory
    backendType        : String(50) enum {
        HANA; SQLITE; S3; REDIS; MEMORY; FILE_SYSTEM; DISTRIBUTED; HYBRID;
    };
    
    @Common.Label: 'Connection String'
    connectionString   : String(500);
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 INACTIVE; MAINTENANCE; ERROR; DEGRADED;
    } default 'ACTIVE';
    
    @Common.Label: 'Health Score'
    @assert.range: [0, 100]
    healthScore        : Decimal(5, 2) default 100;
    
    @Common.Label: 'Total Capacity (GB)'
    totalCapacity      : Decimal(15, 2);
    
    @Common.Label: 'Used Capacity (GB)'
    usedCapacity       : Decimal(15, 2) default 0;
    
    @Common.Label: 'Available Capacity (GB)'
    availableCapacity  : Decimal(15, 2);
    
    @Common.Label: 'Read Performance (IOPS)'
    readPerformance    : Integer;
    
    @Common.Label: 'Write Performance (IOPS)'
    writePerformance   : Integer;
    
    @Common.Label: 'Latency (ms)'
    latency            : Decimal(10, 3);
    
    @Common.Label: 'Throughput (MB/s)'
    throughput         : Decimal(10, 2);
    
    @Common.Label: 'Compression Ratio'
    compressionRatio   : Decimal(5, 2);
    
    @Common.Label: 'Encryption Enabled'
    encryptionEnabled  : Boolean default false;
    
    @Common.Label: 'Backup Enabled'
    backupEnabled      : Boolean default true;
    
    @Common.Label: 'Replication Factor'
    replicationFactor  : Integer default 1;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Connection Pool Size'
    connectionPoolSize : Integer default 10;
    
    @Common.Label: 'Last Health Check'
    lastHealthCheck    : DateTime;
    
    @Common.Label: 'Last Backup'
    lastBackup         : DateTime;
    
    @Common.Label: 'Maintenance Window'
    maintenanceWindow  : String(100);
    
    @Common.Label: 'Storage Utilizations'
    storageUtilizations : Composition of many StorageUtilizations on storageUtilizations.backend = $self;
}

// Storage Utilization Tracking
entity StorageUtilizations : cuid {
    @Common.Label: 'Task'
    task               : Association to DataTasks;
    
    @Common.Label: 'Backend'
    backend            : Association to StorageBackends not null;
    
    @Common.Label: 'Operation Type'
    operationType      : String(20) enum {
        READ; WRITE; DELETE; UPDATE; BACKUP; RESTORE;
    };
    
    @Common.Label: 'Data Size (bytes)'
    dataSize           : Integer64;
    
    @Common.Label: 'Storage Used (bytes)'
    storageUsed        : Integer64;
    
    @Common.Label: 'Storage Saved (bytes)'
    storageSaved       : Integer64 default 0;
    
    @Common.Label: 'Duration (ms)'
    duration           : Integer;
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Performance Metrics'
    @Core.MediaType: 'application/json'
    performanceMetrics : String(2000);
    
    @Common.Label: 'Success'
    success            : Boolean default true;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
}

// Cache Management
entity CacheConfigurations : cuid, managed {
    @Common.Label: 'Cache Name'
    @Search.defaultSearchElement: true
    @mandatory
    cacheName          : String(100);
    
    @Common.Label: 'Cache Type'
    @mandatory
    cacheType          : String(30) enum {
        MEMORY; REDIS; HYBRID; DISTRIBUTED;
    };
    
    @Common.Label: 'Cache Strategy'
    cacheStrategy      : String(50) enum {
        LRU; LFU; FIFO; TTL; CUSTOM;
    };
    
    @Common.Label: 'Max Size (MB)'
    maxSize            : Integer;
    
    @Common.Label: 'TTL (seconds)'
    ttl                : Integer default 3600;
    
    @Common.Label: 'Eviction Policy'
    evictionPolicy     : String(50);
    
    @Common.Label: 'Compression Enabled'
    compressionEnabled : Boolean default false;
    
    @Common.Label: 'Persistence Enabled'
    persistenceEnabled : Boolean default false;
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 INACTIVE; MAINTENANCE; ERROR;
    } default 'ACTIVE';
    
    @Common.Label: 'Current Size (MB)'
    currentSize        : Decimal(10, 2) default 0;
    
    @Common.Label: 'Hit Rate (%)'
    @assert.range: [0, 100]
    hitRate            : Decimal(5, 2) default 0;
    
    @Common.Label: 'Miss Rate (%)'
    @assert.range: [0, 100]
    missRate           : Decimal(5, 2) default 0;
    
    @Common.Label: 'Operations Per Second'
    operationsPerSecond : Integer default 0;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Cache Operations'
    cacheOperations    : Composition of many CacheOperations on cacheOperations.cacheConfig = $self;
}

// Cache Operations Tracking
entity CacheOperations : cuid {
    @Common.Label: 'Task'
    task               : Association to DataTasks;
    
    @Common.Label: 'Cache Configuration'
    cacheConfig        : Association to CacheConfigurations not null;
    
    @Common.Label: 'Operation Type'
    operationType      : String(20) enum {
        GET; PUT; DELETE; EVICT; FLUSH; INVALIDATE;
    };
    
    @Common.Label: 'Cache Key'
    cacheKey           : String(500);
    
    @Common.Label: 'Data Size (bytes)'
    dataSize           : Integer;
    
    @Common.Label: 'Hit/Miss'
    hitMiss            : String(10) enum {
        HIT; MISS; ERROR;
    };
    
    @Common.Label: 'Response Time (ms)'
    responseTime       : Decimal(10, 3);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Success'
    success            : Boolean default true;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(2000);
}

// Data Version Management
entity DataVersions : cuid, managed {
    @Common.Label: 'Task'
    task               : Association to DataTasks not null;
    
    @Common.Label: 'Version Number'
    @mandatory
    versionNumber      : String(50);
    
    @Common.Label: 'Version Type'
    versionType        : String(20) enum {
        MAJOR; MINOR; PATCH; SNAPSHOT; BACKUP;
    } default 'MINOR';
    
    @Common.Label: 'Data Checksum'
    dataChecksum       : String(128);
    
    @Common.Label: 'Data Size (bytes)'
    dataSize           : Integer64;
    
    @Common.Label: 'Compressed Size (bytes)'
    compressedSize     : Integer64;
    
    @Common.Label: 'Storage Location'
    storageLocation    : String(500);
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Tags'
    @Core.MediaType: 'application/json'
    tags               : String(500);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : LargeString;
    
    @Common.Label: 'Parent Version'
    parentVersion      : Association to DataVersions;
    
    @Common.Label: 'Is Current'
    isCurrent          : Boolean default false;
    
    @Common.Label: 'Is Deleted'
    isDeleted          : Boolean default false;
    
    @Common.Label: 'Retention Until'
    retentionUntil     : DateTime;
    
    @Common.Label: 'Access Count'
    accessCount        : Integer default 0;
    
    @Common.Label: 'Last Accessed'
    lastAccessed       : DateTime;
}

// Data Backup Management
entity DataBackups : cuid, managed {
    @Common.Label: 'Backup Name'
    @Search.defaultSearchElement: true
    @mandatory
    backupName         : String(100);
    
    @Common.Label: 'Source Data'
    sourceData         : String(500);
    
    @Common.Label: 'Backup Type'
    backupType         : String(30) enum {
        FULL; INCREMENTAL; DIFFERENTIAL; SNAPSHOT;
    };
    
    @Common.Label: 'Storage Backend'
    storageBackend     : Association to StorageBackends;
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 RUNNING; COMPLETED; FAILED; CANCELLED;
    } default 'SCHEDULED';
    
    @Common.Label: 'Schedule Type'
    scheduleType       : String(20) enum {
 HOURLY; DAILY; WEEKLY; MONTHLY;
    } default 'MANUAL';
    
    @Common.Label: 'Schedule Expression'
    scheduleExpression : String(100);
    
    @Common.Label: 'Data Size (bytes)'
    dataSize           : Integer64;
    
    @Common.Label: 'Compressed Size (bytes)'
    compressedSize     : Integer64;
    
    @Common.Label: 'Compression Ratio'
    compressionRatio   : Decimal(5, 2);
    
    @Common.Label: 'Encryption Enabled'
    encryptionEnabled  : Boolean default true;
    
    @Common.Label: 'Verification Status'
    verificationStatus : String(20) enum {
 VERIFIED; FAILED; SKIPPED;
    } default 'PENDING';
    
    @Common.Label: 'Retention Period (days)'
    retentionPeriod    : Integer default 90;
    
    @Common.Label: 'Started At'
    startedAt          : DateTime;
    
    @Common.Label: 'Completed At'
    completedAt        : DateTime;
    
    @Common.Label: 'Duration (seconds)'
    duration           : Integer;
    
    @Common.Label: 'Progress Percentage'
    @assert.range: [0, 100]
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Storage Location'
    storageLocation    : String(500);
    
    @Common.Label: 'Checksum'
    checksum           : String(128);
    
    @Common.Label: 'Next Scheduled'
    nextScheduled      : DateTime;
    
    @Common.Label: 'Error Log'
    @Core.MediaType: 'application/json'
    errorLog           : LargeString;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
}

// Data Performance Metrics
entity DataPerformanceMetrics : cuid {
    @Common.Label: 'Metric Name'
    @mandatory
    metricName         : String(100);
    
    @Common.Label: 'Metric Type'
    metricType         : String(50) enum {
        THROUGHPUT; LATENCY; IOPS; CPU_USAGE; MEMORY_USAGE;
        STORAGE_USAGE; CACHE_HIT_RATE; ERROR_RATE; QUEUE_DEPTH;
    };
    
    @Common.Label: 'Storage Backend'
    storageBackend     : Association to StorageBackends;
    
    @Common.Label: 'Cache Configuration'
    cacheConfig        : Association to CacheConfigurations;
    
    @Common.Label: 'Value'
    value              : Decimal(20, 5);
    
    @Common.Label: 'Unit'
    unit               : String(20);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Time Window'
    timeWindow         : String(20) enum {
        MINUTE; HOUR; DAY; WEEK; MONTH;
    } default 'MINUTE';
    
    @Common.Label: 'Min Value'
    minValue           : Decimal(20, 5);
    
    @Common.Label: 'Max Value'
    maxValue           : Decimal(20, 5);
    
    @Common.Label: 'Average Value'
    averageValue       : Decimal(20, 5);
    
    @Common.Label: 'Percentile 95'
    percentile95       : Decimal(20, 5);
    
    @Common.Label: 'Percentile 99'
    percentile99       : Decimal(20, 5);
    
    @Common.Label: 'Trend'
    trend              : String(20) enum {
        IMPROVING; STABLE; DEGRADING;
    };
    
    @Common.Label: 'Threshold'
    threshold          : Decimal(20, 5);
    
    @Common.Label: 'Alert Triggered'
    alertTriggered     : Boolean default false;
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(2000);
}

// Agent 9 Entities - Advanced Logical Reasoning and Decision-Making Agent

// Reasoning Tasks Management
entity ReasoningTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    @mandatory
    taskName           : String(100);
    
    @Common.Label: 'Description'
    description        : String(1000);
    
    @Common.Label: 'Reasoning Type'
    @mandatory
    reasoningType      : String(30) enum {
        DEDUCTIVE; INDUCTIVE; ABDUCTIVE; ANALOGICAL; 
        PROBABILISTIC; CAUSAL; TEMPORAL; MODAL;
    };
    
    @Common.Label: 'Problem Domain'
    @mandatory
    problemDomain      : String(50) enum {
        BUSINESS_LOGIC; SCIENTIFIC; MEDICAL; LEGAL; 
        TECHNICAL; FINANCIAL; EDUCATIONAL; GENERAL;
    };
    
    @Common.Label: 'Reasoning Engine'
    @mandatory
    reasoningEngine    : String(40) enum {
        FORWARD_CHAINING; BACKWARD_CHAINING; BAYESIAN_NETWORK;
        FUZZY_LOGIC; NEURAL_REASONING; CASE_BASED; EXPERT_SYSTEM;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        DRAFT; PENDING; PROCESSING; COMPLETED; FAILED; CANCELLED;
    } default 'DRAFT';
    
    @Common.Label: 'Priority'
    priority           : String(20) enum {
 NORMAL; HIGH; URGENT; CRITICAL;
    } default 'NORMAL';
    
    @Common.Label: 'Confidence Score'
    confidenceScore    : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Processing Time'
    processingTime     : Integer default 0; // in milliseconds
    
    @Common.Label: 'Conclusions Reached'
    conclusionsReached : Integer default 0;
    
    @Common.Label: 'Premises'
    @Core.MediaType: 'application/json'
    premises           : String(5000);
    
    @Common.Label: 'Goals'
    @Core.MediaType: 'application/json'
    goals              : String(2000);
    
    @Common.Label: 'Constraints'
    @Core.MediaType: 'application/json'
    constraints        : String(2000);
    
    @Common.Label: 'Knowledge Context'
    @Core.MediaType: 'application/json'
    knowledgeContext   : String(3000);
    
    @Common.Label: 'Reasoning Results'
    @Core.MediaType: 'application/json'
    reasoningResults   : String(5000);
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(2000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(2000);
    
    // Associations
    @Common.Label: 'Knowledge Base Elements'
    knowledgeElements  : Composition of many KnowledgeBaseElements on knowledgeElements.task = $self;
    
    @Common.Label: 'Generated Inferences'
    inferences         : Composition of many LogicalInferences on inferences.task = $self;
    
    @Common.Label: 'Decision Records'
    decisions          : Composition of many DecisionRecords on decisions.task = $self;
}

// Knowledge Base Elements (Facts, Rules, Ontologies)
entity KnowledgeBaseElements : cuid, managed {
    @Common.Label: 'Element Type'
    @mandatory
    elementType        : String(20) enum {
        FACT; RULE; ONTOLOGY; AXIOM; DEFINITION;
    };
    
    @Common.Label: 'Name'
    @Search.defaultSearchElement: true
    @mandatory
    elementName        : String(100);
    
    @Common.Label: 'Content'
    @mandatory
    content            : String(5000);
    
    @Common.Label: 'Domain'
    domain             : String(50);
    
    @Common.Label: 'Confidence Level'
    confidenceLevel    : Decimal(5, 2) default 1.0;
    
    @Common.Label: 'Priority Weight'
    priorityWeight     : Integer default 50;
    
    @Common.Label: 'Source'
    source             : String(200);
    
    @Common.Label: 'Is Active'
    isActive           : Boolean default true;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Last Used'
    lastUsed           : DateTime;
    
    @Common.Label: 'Tags'
    @Core.MediaType: 'application/json'
    tags               : String(500);
    
    @Common.Label: 'Dependencies'
    @Core.MediaType: 'application/json'
    dependencies       : String(1000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to ReasoningTasks;
}

// Logical Inferences Generated
entity LogicalInferences : cuid, managed {
    @Common.Label: 'Inference Type'
    @mandatory
    inferenceType      : String(30) enum {
        DEDUCTION; INDUCTION; ABDUCTION; GENERALIZATION;
        SPECIALIZATION; ANALOGY; CONTRADICTION_RESOLUTION;
    };
    
    @Common.Label: 'Statement'
    @mandatory
    statement          : String(2000);
    
    @Common.Label: 'Premises Used'
    @Core.MediaType: 'application/json'
    premisesUsed       : String(3000);
    
    @Common.Label: 'Rules Applied'
    @Core.MediaType: 'application/json'
    rulesApplied       : String(1000);
    
    @Common.Label: 'Confidence Score'
    confidenceScore    : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Derivation Steps'
    derivationSteps    : Integer default 1;
    
    @Common.Label: 'Is Valid'
    isValid            : Boolean default true;
    
    @Common.Label: 'Validation Method'
    validationMethod   : String(30) enum {
        LOGICAL_CONSISTENCY; EMPIRICAL_VERIFICATION; 
        EXPERT_REVIEW; AUTOMATED_CHECKING;
    };
    
    @Common.Label: 'Support Evidence'
    @Core.MediaType: 'application/json'
    supportEvidence    : String(2000);
    
    @Common.Label: 'Counter Evidence'
    @Core.MediaType: 'application/json'
    counterEvidence    : String(2000);
    
    @Common.Label: 'Explanation'
    explanation        : String(3000);
    
    @Common.Label: 'Generation Time'
    generationTime     : DateTime;
    
    @Common.Label: 'Processing Duration'
    processingDuration : Integer default 0; // in milliseconds
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to ReasoningTasks;
}

// Reasoning Engine Configurations
entity ReasoningEngines : cuid, managed {
    @Common.Label: 'Engine Name'
    @Search.defaultSearchElement: true
    @mandatory
    engineName         : String(100);
    
    @Common.Label: 'Engine Type'
    @mandatory
    engineType         : String(40) enum {
        FORWARD_CHAINING; BACKWARD_CHAINING; BAYESIAN_NETWORK;
        FUZZY_LOGIC; NEURAL_REASONING; CASE_BASED; EXPERT_SYSTEM;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 INACTIVE; MAINTENANCE; DEPRECATED;
    } default 'ACTIVE';
    
    @Common.Label: 'Performance Score'
    performanceScore   : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Accuracy Rate'
    accuracyRate       : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Average Processing Time'
    averageProcessingTime : Integer default 0; // in milliseconds
    
    @Common.Label: 'Supported Domains'
    @Core.MediaType: 'application/json'
    supportedDomains   : String(1000);
    
    @Common.Label: 'Capabilities'
    @Core.MediaType: 'application/json'
    capabilities       : String(2000);
    
    @Common.Label: 'Limitations'
    @Core.MediaType: 'application/json'
    limitations        : String(1000);
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(3000);
    
    @Common.Label: 'Optimization Settings'
    @Core.MediaType: 'application/json'
    optimizationSettings : String(2000);
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Last Used'
    lastUsed           : DateTime;
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
}

// Decision Records and Analysis
entity DecisionRecords : cuid, managed {
    @Common.Label: 'Decision Name'
    @Search.defaultSearchElement: true
    @mandatory
    decisionName       : String(100);
    
    @Common.Label: 'Decision Type'
    @mandatory
    decisionType       : String(30) enum {
        BINARY; MULTI_CHOICE; RANKING; OPTIMIZATION; 
        CLASSIFICATION; STRATEGIC; TACTICAL;
    };
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 ANALYZED; DECIDED; IMPLEMENTED; EVALUATED;
    } default 'PENDING';
    
    @Common.Label: 'Decision Criteria'
    @Core.MediaType: 'application/json'
    decisionCriteria   : String(3000);
    
    @Common.Label: 'Alternatives'
    @Core.MediaType: 'application/json'
    alternatives       : String(3000);
    
    @Common.Label: 'Recommended Option'
    recommendedOption  : String(500);
    
    @Common.Label: 'Confidence Score'
    confidenceScore    : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Risk Assessment'
    @Core.MediaType: 'application/json'
    riskAssessment     : String(2000);
    
    @Common.Label: 'Impact Analysis'
    @Core.MediaType: 'application/json'
    impactAnalysis     : String(2000);
    
    @Common.Label: 'Justification'
    justification      : String(3000);
    
    @Common.Label: 'Expected Outcome'
    expectedOutcome    : String(1000);
    
    @Common.Label: 'Decision Time'
    decisionTime       : DateTime;
    
    @Common.Label: 'Implementation Date'
    implementationDate : DateTime;
    
    @Common.Label: 'Evaluation Date'
    evaluationDate     : DateTime;
    
    @Common.Label: 'Actual Outcome'
    actualOutcome      : String(1000);
    
    @Common.Label: 'Success Rate'
    successRate        : Decimal(5, 2);
    
    @Common.Label: 'Lessons Learned'
    lessonsLearned     : String(2000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to ReasoningTasks;
}

// Problem Solving Records
entity ProblemSolvingRecords : cuid, managed {
    @Common.Label: 'Problem Name'
    @Search.defaultSearchElement: true
    @mandatory
    problemName        : String(100);
    
    @Common.Label: 'Problem Type'
    @mandatory
    problemType        : String(30) enum {
        ANALYTICAL; CREATIVE; OPTIMIZATION; DIAGNOSTIC;
        PLANNING; SCHEDULING; RESOURCE_ALLOCATION;
    };
    
    @Common.Label: 'Problem Description'
    @mandatory
    problemDescription : String(3000);
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        IDENTIFIED; ANALYZING; SOLVING; SOLVED; VERIFIED;
    } default 'IDENTIFIED';
    
    @Common.Label: 'Solving Strategy'
    solvingStrategy    : String(40) enum {
        DIVIDE_CONQUER; BACKTRACKING; DYNAMIC_PROGRAMMING;
        GREEDY; HEURISTIC; SIMULATION; CONSTRAINT_SATISFACTION;
    };
    
    @Common.Label: 'Initial State'
    @Core.MediaType: 'application/json'
    initialState       : String(2000);
    
    @Common.Label: 'Goal State'
    @Core.MediaType: 'application/json'
    goalState          : String(2000);
    
    @Common.Label: 'Constraints'
    @Core.MediaType: 'application/json'
    constraints        : String(2000);
    
    @Common.Label: 'Solution Steps'
    @Core.MediaType: 'application/json'
    solutionSteps      : String(5000);
    
    @Common.Label: 'Final Solution'
    @Core.MediaType: 'application/json'
    finalSolution      : String(3000);
    
    @Common.Label: 'Solution Quality Score'
    solutionQualityScore : Decimal(5, 2) default 0.0;
    
    @Common.Label: 'Time Complexity'
    timeComplexity     : String(50);
    
    @Common.Label: 'Space Complexity'
    spaceComplexity    : String(50);
    
    @Common.Label: 'Solving Time'
    solvingTime        : Integer default 0; // in milliseconds
    
    @Common.Label: 'Iterations'
    iterations         : Integer default 0;
    
    @Common.Label: 'Validation Results'
    @Core.MediaType: 'application/json'
    validationResults  : String(2000);
    
    @Common.Label: 'Alternative Solutions'
    @Core.MediaType: 'application/json'
    alternativeSolutions : String(3000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
}

// Reasoning Performance Metrics
entity ReasoningPerformanceMetrics : cuid, managed {
    @Common.Label: 'Metric Name'
    @Search.defaultSearchElement: true
    @mandatory
    metricName         : String(100);
    
    @Common.Label: 'Metric Type'
    @mandatory
    metricType         : String(30) enum {
        ACCURACY; PRECISION; RECALL; F1_SCORE; CONFIDENCE;
        PROCESSING_TIME; MEMORY_USAGE; THROUGHPUT; ERROR_RATE;
    };
    
    @Common.Label: 'Engine Type'
    engineType         : String(40) enum {
        FORWARD_CHAINING; BACKWARD_CHAINING; BAYESIAN_NETWORK;
        FUZZY_LOGIC; NEURAL_REASONING; CASE_BASED; EXPERT_SYSTEM;
    };
    
    @Common.Label: 'Problem Domain'
    problemDomain      : String(50);
    
    @Common.Label: 'Metric Value'
    metricValue        : Decimal(20, 5);
    
    @Common.Label: 'Unit'
    unit               : String(20);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    @Common.Label: 'Benchmark Value'
    benchmarkValue     : Decimal(20, 5);
    
    @Common.Label: 'Threshold'
    threshold          : Decimal(20, 5);
    
    @Common.Label: 'Performance Grade'
    performanceGrade   : String(10) enum {
        EXCELLENT; GOOD; AVERAGE; POOR; CRITICAL;
    };
    
    @Common.Label: 'Trend'
    trend              : String(20) enum {
        IMPROVING; STABLE; DEGRADING;
    };
    
    @Common.Label: 'Sample Size'
    sampleSize         : Integer default 1;
    
    @Common.Label: 'Confidence Interval'
    confidenceInterval : String(50);
    
    @Common.Label: 'Statistical Significance'
    statisticalSignificance : Boolean default false;
    
    @Common.Label: 'Context'
    @Core.MediaType: 'application/json'
    context            : String(1000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
}

// Agent 10: Calculation Tasks
entity CalculationTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @Search.defaultSearchElement: true
    @mandatory
    taskName           : String(100);
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Calculation Type'
    @mandatory
    calculationType    : String(50) enum {
        MATHEMATICAL; STATISTICAL; FINANCIAL; SCIENTIFIC; 
        CUSTOM_FORMULA; OPTIMIZATION; SIMULATION;
    };
    
    @Common.Label: 'Formula Expression'
    @UI.MultiLineText: true
    @mandatory
    formula            : String(2000);
    
    @Common.Label: 'Input Parameters'
    @Core.MediaType: 'application/json'
    inputParameters    : LargeString;
    
    @Common.Label: 'Calculation Method'
    @mandatory
    calculationMethod  : String(30) enum {
        DIRECT; ITERATIVE; MONTE_CARLO; GENETIC_ALGORITHM;
        NEURAL_NETWORK; SYMBOLIC; NUMERICAL;
    };
    
    @Common.Label: 'Precision Type'
    precisionType      : String(20) enum {
        DECIMAL32; DECIMAL64; DECIMAL128; ARBITRARY;
    } default 'DECIMAL64';
    
    @Common.Label: 'Required Accuracy'
    requiredAccuracy   : Decimal(10, 8) default 0.000001;
    
    @Common.Label: 'Max Iterations'
    maxIterations      : Integer default 1000;
    
    @Common.Label: 'Timeout (ms)'
    timeout            : Integer default 60000;
    
    @Common.Label: 'Enable Self Healing'
    enableSelfHealing  : Boolean default true;
    
    @Common.Label: 'Verification Rounds'
    verificationRounds : Integer default 3;
    
    @Common.Label: 'Use Parallel Processing'
    useParallelProcessing : Boolean default true;
    
    @Common.Label: 'Cache Results'
    cacheResults       : Boolean default true;
    
    @Common.Label: 'Priority'
    priority           : String(10) enum {
 MEDIUM; HIGH; CRITICAL;
    } default 'MEDIUM';
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 PROCESSING; COMPLETED; FAILED; CANCELLED;
    } default 'PENDING';
    
    @Common.Label: 'Progress'
    progress           : Integer default 0;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Execution Time (ms)'
    executionTime      : Integer;
    
    @Common.Label: 'Result'
    @Core.MediaType: 'application/json'
    result             : LargeString;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Self Healing Log'
    @Core.MediaType: 'application/json'
    selfHealingLog     : String(2000);
    
    @Common.Label: 'Performance Metrics'
    @Core.MediaType: 'application/json'
    performanceMetrics : String(1000);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Associations
    @Common.Label: 'Calculation Steps'
    calculationSteps   : Composition of many CalculationSteps on calculationSteps.task = $self;
    
    @Common.Label: 'Statistical Results'
    statisticalResults : Composition of many StatisticalAnalysisResults on statisticalResults.task = $self;
    
    @Common.Label: 'Error Corrections'
    errorCorrections   : Composition of many CalculationErrorCorrections on errorCorrections.task = $self;
}

// Calculation Steps
entity CalculationSteps : cuid, managed {
    @Common.Label: 'Step Number'
    @mandatory
    stepNumber         : Integer;
    
    @Common.Label: 'Step Name'
    @mandatory
    stepName           : String(100);
    
    @Common.Label: 'Operation'
    @mandatory
    operation          : String(50);
    
    @Common.Label: 'Input Values'
    @Core.MediaType: 'application/json'
    inputValues        : String(2000);
    
    @Common.Label: 'Intermediate Result'
    @Core.MediaType: 'application/json'
    intermediateResult : String(1000);
    
    @Common.Label: 'Processing Time (ms)'
    processingTime     : Integer;
    
    @Common.Label: 'Is Valid'
    isValid            : Boolean default true;
    
    @Common.Label: 'Verification Status'
    verificationStatus : String(20) enum {
        NOT_VERIFIED; VERIFIED; FAILED; CORRECTED;
    };
    
    @Common.Label: 'Error Details'
    errorDetails       : String(500);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(500);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to CalculationTasks;
}

// Statistical Analysis Results
entity StatisticalAnalysisResults : cuid, managed {
    @Common.Label: 'Analysis Type'
    @mandatory
    analysisType       : String(50) enum {
        DESCRIPTIVE; INFERENTIAL; PREDICTIVE; PRESCRIPTIVE;
        TIME_SERIES; REGRESSION; CORRELATION; HYPOTHESIS_TEST;
    };
    
    @Common.Label: 'Dataset Name'
    datasetName        : String(100);
    
    @Common.Label: 'Sample Size'
    sampleSize         : Integer;
    
    @Common.Label: 'Mean'
    mean               : Decimal(20, 8);
    
    @Common.Label: 'Median'
    median             : Decimal(20, 8);
    
    @Common.Label: 'Mode'
    mode               : Decimal(20, 8);
    
    @Common.Label: 'Standard Deviation'
    standardDeviation  : Decimal(20, 8);
    
    @Common.Label: 'Variance'
    variance           : Decimal(20, 8);
    
    @Common.Label: 'Min Value'
    minValue           : Decimal(20, 8);
    
    @Common.Label: 'Max Value'
    maxValue           : Decimal(20, 8);
    
    @Common.Label: 'Confidence Level'
    confidenceLevel    : Decimal(5, 2) default 95.0;
    
    @Common.Label: 'Confidence Interval'
    confidenceInterval : String(100);
    
    @Common.Label: 'P-Value'
    pValue             : Decimal(10, 8);
    
    @Common.Label: 'Correlation Coefficient'
    correlationCoefficient : Decimal(5, 4);
    
    @Common.Label: 'R-Squared'
    rSquared           : Decimal(5, 4);
    
    @Common.Label: 'Additional Metrics'
    @Core.MediaType: 'application/json'
    additionalMetrics  : String(2000);
    
    @Common.Label: 'Visualization Data'
    @Core.MediaType: 'application/json'
    visualizationData  : String(3000);
    
    @Common.Label: 'Interpretation'
    interpretation     : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to CalculationTasks;
}

// Calculation Error Corrections
entity CalculationErrorCorrections : cuid, managed {
    @Common.Label: 'Error Type'
    @mandatory
    errorType          : String(50) enum {
        OVERFLOW; UNDERFLOW; DIVISION_BY_ZERO; NAN;
        PRECISION_LOSS; CONVERGENCE_FAILURE; TIMEOUT;
        INVALID_INPUT; FORMULA_ERROR;
    };
    
    @Common.Label: 'Error Description'
    errorDescription   : String(500);
    
    @Common.Label: 'Detection Method'
    detectionMethod    : String(30) enum {
        BOUNDARY_CHECK; CONSISTENCY_CHECK; REDUNDANT_CALC;
        PATTERN_ANALYSIS; STATISTICAL_ANOMALY;
    };
    
    @Common.Label: 'Original Value'
    originalValue      : String(200);
    
    @Common.Label: 'Corrected Value'
    correctedValue     : String(200);
    
    @Common.Label: 'Correction Strategy'
    correctionStrategy : String(50) enum {
        PRECISION_ADJUSTMENT; SCALING; APPROXIMATION;
        ALTERNATIVE_METHOD; ITERATIVE_REFINEMENT;
    };
    
    @Common.Label: 'Correction Confidence'
    correctionConfidence : Decimal(5, 2);
    
    @Common.Label: 'Verification Status'
    verificationStatus : Boolean default false;
    
    @Common.Label: 'Impact Assessment'
    impactAssessment   : String(500);
    
    @Common.Label: 'Timestamp'
    timestamp          : DateTime;
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to CalculationTasks;
}

// Agent 11: SQL Query Tasks
entity SQLQueryTasks : cuid, managed {
    @Common.Label: 'Query Name'
    @Search.defaultSearchElement: true
    @mandatory
    queryName          : String(100);
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(1000);
    
    @Common.Label: 'Query Type'
    @mandatory
    queryType          : String(20) enum {
        SELECT; INSERT; UPDATE; DELETE; CREATE; DROP; ALTER;
        MERGE; UPSERT; TRUNCATE; CALL; EXECUTE;
    };
    
    @Common.Label: 'Natural Language Input'
    @UI.MultiLineText: true
    naturalLanguageQuery : String(2000);
    
    @Common.Label: 'Generated SQL'
    @UI.MultiLineText: true
    @mandatory
    generatedSQL       : LargeString;
    
    @Common.Label: 'Original SQL'
    @UI.MultiLineText: true
    originalSQL        : LargeString;
    
    @Common.Label: 'Optimized SQL'
    @UI.MultiLineText: true
    optimizedSQL       : LargeString;
    
    @Common.Label: 'Database Connection'
    @mandatory
    databaseConnection : String(100);
    
    @Common.Label: 'SQL Dialect'
    sqlDialect         : String(30) enum {
        HANA; POSTGRESQL; MYSQL; SQLITE; ORACLE; SQLSERVER; MARIADB;
    } default 'HANA';
    
    @Common.Label: 'Parameters'
    @Core.MediaType: 'application/json'
    queryParameters    : LargeString;
    
    @Common.Label: 'Execution Context'
    @Core.MediaType: 'application/json'
    executionContext   : String(1000);
    
    @Common.Label: 'Priority'
    priority           : String(10) enum {
 MEDIUM; HIGH; CRITICAL;
    } default 'MEDIUM';
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        DRAFT; VALIDATING; READY; EXECUTING; COMPLETED; FAILED; CANCELLED;
    } default 'DRAFT';
    
    @Common.Label: 'Execution Time (ms)'
    executionTime      : Integer;
    
    @Common.Label: 'Rows Affected'
    rowsAffected       : Integer;
    
    @Common.Label: 'Result Row Count'
    resultRowCount     : Integer;
    
    @Common.Label: 'Is Optimized'
    isOptimized        : Boolean default false;
    
    @Common.Label: 'Auto Generated'
    autoGenerated      : Boolean default false;
    
    @Common.Label: 'Requires Approval'
    requiresApproval   : Boolean default false;
    
    @Common.Label: 'Is Approved'
    isApproved         : Boolean default false;
    
    @Common.Label: 'Approved By'
    approvedBy         : String(100);
    
    @Common.Label: 'Approval Timestamp'
    approvalTimestamp  : DateTime;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Query Results'
    @Core.MediaType: 'application/json'
    queryResults       : LargeString;
    
    @Common.Label: 'Execution Plan'
    @Core.MediaType: 'application/json'
    executionPlan      : LargeString;
    
    @Common.Label: 'Performance Metrics'
    @Core.MediaType: 'application/json'
    performanceMetrics : String(1000);
    
    @Common.Label: 'Security Context'
    @Core.MediaType: 'application/json'
    securityContext    : String(500);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Associations
    @Common.Label: 'Query Optimizations'
    optimizations      : Composition of many QueryOptimizations on optimizations.task = $self;
    
    @Common.Label: 'Execution History'
    executionHistory   : Composition of many QueryExecutionHistory on executionHistory.task = $self;
    
    @Common.Label: 'Schema References'
    schemaReferences   : Composition of many SchemaReferences on schemaReferences.task = $self;
}

// Query Optimization Records
entity QueryOptimizations : cuid, managed {
    @Common.Label: 'Optimization Type'
    @mandatory
    optimizationType   : String(50) enum {
        INDEX_SUGGESTION; QUERY_REWRITE; JOIN_OPTIMIZATION;
        PARTITION_PRUNING; COST_REDUCTION; CACHE_STRATEGY;
    };
    
    @Common.Label: 'Original Query'
    @UI.MultiLineText: true
    originalQuery      : LargeString;
    
    @Common.Label: 'Optimized Query'
    @UI.MultiLineText: true
    optimizedQuery     : LargeString;
    
    @Common.Label: 'Optimization Reason'
    optimizationReason : String(500);
    
    @Common.Label: 'Performance Improvement'
    performanceImprovement : Decimal(5, 2);
    
    @Common.Label: 'Cost Reduction'
    costReduction      : Decimal(5, 2);
    
    @Common.Label: 'Estimated Benefit'
    estimatedBenefit   : String(200);
    
    @Common.Label: 'Applied'
    isApplied          : Boolean default false;
    
    @Common.Label: 'Application Time'
    applicationTime    : DateTime;
    
    @Common.Label: 'Validation Status'
    validationStatus   : String(20) enum {
 VALIDATED; REJECTED; APPLIED;
    } default 'PENDING';
    
    @Common.Label: 'Optimization Details'
    @Core.MediaType: 'application/json'
    optimizationDetails : String(2000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to SQLQueryTasks;
}

// Query Execution History
entity QueryExecutionHistory : cuid, managed {
    @Common.Label: 'Execution Timestamp'
    @mandatory
    executionTimestamp : DateTime;
    
    @Common.Label: 'Execution Duration (ms)'
    executionDuration  : Integer;
    
    @Common.Label: 'Rows Returned'
    rowsReturned       : Integer;
    
    @Common.Label: 'Rows Affected'
    rowsAffected       : Integer;
    
    @Common.Label: 'Memory Used (KB)'
    memoryUsed         : Integer;
    
    @Common.Label: 'CPU Time (ms)'
    cpuTime            : Integer;
    
    @Common.Label: 'I/O Operations'
    ioOperations       : Integer;
    
    @Common.Label: 'Cache Hits'
    cacheHits          : Integer;
    
    @Common.Label: 'Cache Misses'
    cacheMisses        : Integer;
    
    @Common.Label: 'Lock Wait Time (ms)'
    lockWaitTime       : Integer;
    
    @Common.Label: 'Execution Status'
    executionStatus    : String(20) enum {
        SUCCESS; FAILED; TIMEOUT; CANCELLED; PARTIAL;
    };
    
    @Common.Label: 'Error Code'
    errorCode          : String(20);
    
    @Common.Label: 'Error Message'
    errorMessage       : String(500);
    
    @Common.Label: 'User Context'
    userContext        : String(100);
    
    @Common.Label: 'Connection ID'
    connectionId       : String(50);
    
    @Common.Label: 'Session ID'
    sessionId          : String(50);
    
    @Common.Label: 'Query Hash'
    queryHash          : String(64);
    
    @Common.Label: 'Execution Plan Hash'
    executionPlanHash  : String(64);
    
    @Common.Label: 'Performance Rating'
    performanceRating  : String(10) enum {
        EXCELLENT; GOOD; AVERAGE; POOR; CRITICAL;
    };
    
    @Common.Label: 'Optimization Opportunities'
    @Core.MediaType: 'application/json'
    optimizationOpportunities : String(1000);
    
    @Common.Label: 'Execution Details'
    @Core.MediaType: 'application/json'
    executionDetails   : String(2000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to SQLQueryTasks;
}

// Database Schema References
entity SchemaReferences : cuid, managed {
    @Common.Label: 'Schema Name'
    @mandatory
    schemaName         : String(100);
    
    @Common.Label: 'Table Name'
    tableName          : String(100);
    
    @Common.Label: 'Column Name'
    columnName         : String(100);
    
    @Common.Label: 'Data Type'
    dataType           : String(50);
    
    @Common.Label: 'Is Primary Key'
    isPrimaryKey       : Boolean default false;
    
    @Common.Label: 'Is Foreign Key'
    isForeignKey       : Boolean default false;
    
    @Common.Label: 'Is Nullable'
    isNullable         : Boolean default true;
    
    @Common.Label: 'Is Indexed'
    isIndexed          : Boolean default false;
    
    @Common.Label: 'Reference Type'
    referenceType      : String(20) enum {
        TABLE; COLUMN; INDEX; CONSTRAINT; VIEW; PROCEDURE; FUNCTION;
    };
    
    @Common.Label: 'Usage Context'
    usageContext       : String(20) enum {
        SELECT; INSERT; UPDATE; DELETE; JOIN; WHERE; GROUP_BY; ORDER_BY;
    };
    
    @Common.Label: 'Access Frequency'
    accessFrequency    : Integer default 0;
    
    @Common.Label: 'Last Accessed'
    lastAccessed       : DateTime;
    
    @Common.Label: 'Default Value'
    defaultValue       : String(200);
    
    @Common.Label: 'Constraints'
    constraints        : String(500);
    
    @Common.Label: 'Comments'
    comments           : String(500);
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to SQLQueryTasks;
}

// Natural Language Processing Results
entity NLProcessingResults : cuid, managed {
    @Common.Label: 'Original Query'
    @UI.MultiLineText: true
    @mandatory
    originalQuery      : String(2000);
    
    @Common.Label: 'Intent Recognition'
    @Core.MediaType: 'application/json'
    intentRecognition  : String(1000);
    
    @Common.Label: 'Entity Extraction'
    @Core.MediaType: 'application/json'
    entityExtraction   : String(2000);
    
    @Common.Label: 'Schema Mapping'
    @Core.MediaType: 'application/json'
    schemaMapping      : String(2000);
    
    @Common.Label: 'Generated SQL'
    @UI.MultiLineText: true
    generatedSQL       : LargeString;
    
    @Common.Label: 'Confidence Score'
    confidenceScore    : Decimal(5, 2);
    
    @Common.Label: 'Processing Status'
    processingStatus   : String(20) enum {
        PROCESSING; COMPLETED; FAILED; AMBIGUOUS; CLARIFICATION_NEEDED;
    };
    
    @Common.Label: 'Ambiguities Found'
    @Core.MediaType: 'application/json'
    ambiguitiesFound   : String(1000);
    
    @Common.Label: 'Clarification Questions'
    @Core.MediaType: 'application/json'
    clarificationQuestions : String(1000);
    
    @Common.Label: 'Context Used'
    @Core.MediaType: 'application/json'
    contextUsed        : String(1000);
    
    @Common.Label: 'Processing Time (ms)'
    processingTime     : Integer;
    
    @Common.Label: 'Model Version'
    modelVersion       : String(50);
    
    @Common.Label: 'Alternative SQLs'
    @Core.MediaType: 'application/json'
    alternativeSQLs    : LargeString;
    
    @Common.Label: 'Validation Results'
    @Core.MediaType: 'application/json'
    validationResults  : String(1000);
    
    @Common.Label: 'Language'
    language           : String(10) default 'en';
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : String(1000);
    
    // Association
    @Common.Label: 'Associated Task'
    task               : Association to SQLQueryTasks;
}

//=====================================================
// Agent 12: Catalog Manager Entities
//=====================================================

// Main catalog entries for services, APIs, and resources
entity CatalogEntries : cuid, managed {
    @Common.Label: 'Entry Name'
    @mandatory
    entryName          : String(100) not null;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Category'
    category           : String(50) not null enum {
        SERVICE; API; DATABASE; WORKFLOW; AGENT; RESOURCE; TEMPLATE; CONNECTOR;
    };
    
    @Common.Label: 'Sub Category'
    subCategory        : String(50);
    
    @Common.Label: 'Version'
    version            : String(20) default '1.0.0';
    
    @Common.Label: 'Status'
    status             : String(20) not null enum {
        DRAFT; PUBLISHED; DEPRECATED; ARCHIVED; UNDER_REVIEW;
    } default 'DRAFT';
    
    @Common.Label: 'Visibility'
    visibility         : String(20) not null enum {
        PUBLIC; PRIVATE; RESTRICTED; INTERNAL;
    } default 'PRIVATE';
    
    @Common.Label: 'Entry Type'
    entryType          : String(30) not null enum {
        MICROSERVICE; REST_API; GRAPHQL_API; DATABASE_TABLE; ML_MODEL;
        DATA_SOURCE; WORKFLOW_TEMPLATE; AGENT_SERVICE; FILE_RESOURCE;
    };
    
    @Common.Label: 'Provider'
    provider           : String(100);
    
    @Common.Label: 'Owner'
    owner              : String(100);
    
    @Common.Label: 'Contact Email'
    contactEmail       : String(100);
    
    @Common.Label: 'Documentation URL'
    documentationUrl   : String(200);
    
    @Common.Label: 'Source URL'
    sourceUrl          : String(200);
    
    @Common.Label: 'API Endpoint'
    apiEndpoint        : String(200);
    
    @Common.Label: 'Health Check URL'
    healthCheckUrl     : String(200);
    
    @Common.Label: 'Tags'
    tags               : String(500); // Comma-separated tags
    
    @Common.Label: 'Keywords'
    keywords           : String(500); // Space-separated keywords
    
    @Common.Label: 'Rating'
    @assert.range: [0, 5]
    rating             : Decimal(2, 1) default 0.0;
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Download Count'
    downloadCount      : Integer default 0;
    
    @Common.Label: 'Is Featured'
    isFeatured         : Boolean default false;
    
    @Common.Label: 'Is Verified'
    isVerified         : Boolean default false;
    
    @Common.Label: 'Last Accessed'
    lastAccessed       : DateTime;
    
    @Common.Label: 'Metadata'
    @Core.MediaType: 'application/json'
    metadata           : LargeString;
    
    @Common.Label: 'Configuration Schema'
    @Core.MediaType: 'application/json'
    configurationSchema : LargeString;
    
    @Common.Label: 'Example Usage'
    @UI.MultiLineText: true
    exampleUsage       : LargeString;
    
    @Common.Label: 'License'
    license            : String(50);
    
    @Common.Label: 'Security Level'
    securityLevel      : String(20) enum {
        PUBLIC; INTERNAL; CONFIDENTIAL; RESTRICTED; TOP_SECRET;
    } default 'INTERNAL';
    
    // Associations
    dependencies       : Composition of many CatalogDependencies on dependencies.catalogEntry = $self;
    reviews            : Composition of many CatalogReviews on reviews.catalogEntry = $self;
    metadata_entries   : Composition of many CatalogMetadata on metadata_entries.catalogEntry = $self;
}

// Dependencies between catalog entries
entity CatalogDependencies : cuid, managed {
    @Common.Label: 'Catalog Entry'
    @assert.integrity
    catalogEntry       : Association to CatalogEntries not null;
    
    @Common.Label: 'Dependent Entry'
    @assert.integrity  
    dependentEntry     : Association to CatalogEntries not null;
    
    @Common.Label: 'Dependency Type'
    dependencyType     : String(30) not null enum {
        REQUIRES; RECOMMENDS; CONFLICTS; REPLACES; EXTENDS; IMPLEMENTS;
    };
    
    @Common.Label: 'Version Range'
    versionRange       : String(50);
    
    @Common.Label: 'Is Critical'
    isCritical         : Boolean default false;
    
    @Common.Label: 'Description'
    description        : String(200);
}

// User reviews and ratings for catalog entries
entity CatalogReviews : cuid, managed {
    @Common.Label: 'Catalog Entry'
    @assert.integrity
    catalogEntry       : Association to CatalogEntries not null;
    
    @Common.Label: 'Reviewer'
    reviewer           : String(100) not null;
    
    @Common.Label: 'Rating'
    @assert.range: [1, 5]
    rating             : Integer not null;
    
    @Common.Label: 'Title'
    title              : String(100);
    
    @Common.Label: 'Review Text'
    @UI.MultiLineText: true
    reviewText         : String(1000);
    
    @Common.Label: 'Pros'
    pros               : String(500);
    
    @Common.Label: 'Cons'
    cons               : String(500);
    
    @Common.Label: 'Recommended Use Case'
    recommendedUseCase : String(200);
    
    @Common.Label: 'Is Verified Review'
    isVerifiedReview   : Boolean default false;
    
    @Common.Label: 'Helpful Votes'
    helpfulVotes       : Integer default 0;
    
    @Common.Label: 'Review Status'
    reviewStatus       : String(20) enum {
 APPROVED; REJECTED; FLAGGED;
    } default 'PENDING';
}

// Extended metadata for catalog entries
entity CatalogMetadata : cuid, managed {
    @Common.Label: 'Catalog Entry'
    @assert.integrity
    catalogEntry       : Association to CatalogEntries not null;
    
    @Common.Label: 'Metadata Key'
    metadataKey        : String(50) not null;
    
    @Common.Label: 'Metadata Value'
    metadataValue      : String(500);
    
    @Common.Label: 'Value Type'
    valueType          : String(20) enum {
        STRING; NUMBER; BOOLEAN; DATE; JSON; URL; EMAIL;
    } default 'STRING';
    
    @Common.Label: 'Is Searchable'
    isSearchable       : Boolean default true;
    
    @Common.Label: 'Display Order'
    displayOrder       : Integer default 0;
    
    @Common.Label: 'Category'
    category           : String(30) enum {
        TECHNICAL; BUSINESS; OPERATIONAL; SECURITY; COMPLIANCE;
    } default 'TECHNICAL';
}

// Search and discovery sessions
entity CatalogSearches : cuid, managed {
    @Common.Label: 'Search Query'
    searchQuery        : String(200) not null;
    
    @Common.Label: 'Search Type'
    searchType         : String(20) enum {
        KEYWORD; CATEGORY; TAG; METADATA; ADVANCED; SEMANTIC;
 not null;
    
        };
    @Common.Label: 'Filters Applied'
    @Core.MediaType: 'application/json'
    filtersApplied     : String(1000);
    
    @Common.Label: 'Results Count'
    resultsCount       : Integer default 0;
    
    @Common.Label: 'Search Time (ms)'
    searchTime         : Integer;
    
    @Common.Label: 'User Agent'
    userAgent          : String(100);
    
    @Common.Label: 'IP Address'
    ipAddress          : String(45);
    
    @Common.Label: 'Session ID'
    sessionId          : String(50);
    
    @Common.Label: 'Search Results'
    @Core.MediaType: 'application/json'
    searchResults      : LargeString;
    
    @Common.Label: 'Selected Result'
    selectedResult     : Association to CatalogEntries;
}

// Registry management for different types of registries
entity RegistryManagement : cuid, managed {
    @Common.Label: 'Registry Name'
    registryName       : String(100) not null;
    
    @Common.Label: 'Registry Type'
    registryType       : String(30) not null enum {
        SERVICE_REGISTRY; API_REGISTRY; SCHEMA_REGISTRY; ARTIFACT_REGISTRY;
        MODEL_REGISTRY; WORKFLOW_REGISTRY; AGENT_REGISTRY;
    };
    
    @Common.Label: 'Registry URL'
    registryUrl        : String(200);
    
    @Common.Label: 'Status'
    status             : String(20) enum {
 INACTIVE; MAINTENANCE; ERROR; SYNCING;
    } default 'ACTIVE';
    
    @Common.Label: 'Last Sync'
    lastSync           : DateTime;
    
    @Common.Label: 'Sync Frequency'
    syncFrequency      : String(20) enum {
        REAL_TIME; HOURLY; DAILY; WEEKLY; MANUAL;
    } default 'DAILY';
    
    @Common.Label: 'Authentication Type'
    authenticationType : String(20) enum {
        NONE; BASIC; OAUTH; API_KEY; CERTIFICATE;
    } default 'NONE';
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(1000);
    
    @Common.Label: 'Health Check URL'
    healthCheckUrl     : String(200);
    
    @Common.Label: 'Total Entries'
    totalEntries       : Integer default 0;
    
    @Common.Label: 'Active Entries'
    activeEntries      : Integer default 0;
    
    @Common.Label: 'Last Error'
    lastError          : String(500);
    
    @Common.Label: 'Sync Statistics'
    @Core.MediaType: 'application/json'
    syncStatistics     : String(1000);
}

//=====================================================
// Agent 13: Agent Builder Entities
//=====================================================

// Agent templates for generating new agents
entity AgentTemplates : cuid, managed {
    @Common.Label: 'Template Name'
    @mandatory
    templateName       : String(100) not null;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Template Category'
    templateCategory   : String(50) not null enum {
        DATA_PROCESSING; ML_INFERENCE; INTEGRATION; UTILITY; SECURITY; ANALYTICS;
    };
    
    @Common.Label: 'Template Type'
    templateType       : String(30) not null enum {
        FULL_AGENT; MICROSERVICE; FUNCTION; WORKFLOW; COMPONENT;
    };
    
    @Common.Label: 'Version'
    version            : String(20) default '1.0.0';
    
    @Common.Label: 'Framework'
    framework          : String(30) enum {
        PYTHON_FASTAPI; PYTHON_FLASK; JAVASCRIPT_EXPRESS; JAVASCRIPT_FASTIFY;
        GO_NATIVE; JAVA_SPRING; CSHARP_DOTNET;
    } default 'PYTHON_FASTAPI';
    
    @Common.Label: 'Language'
    language           : String(20) enum {
        PYTHON; JAVASCRIPT; GO; JAVA; CSHARP; TYPESCRIPT;
    } default 'PYTHON';
    
    @Common.Label: 'Status'
    status             : String(20) not null enum {
        DRAFT; PUBLISHED; DEPRECATED; ARCHIVED;
    } default 'DRAFT';
    
    @Common.Label: 'Complexity Level'
    complexityLevel    : String(20) enum {
        BEGINNER; INTERMEDIATE; ADVANCED; EXPERT;
    } default 'INTERMEDIATE';
    
    @Common.Label: 'Author'
    author             : String(100);
    
    @Common.Label: 'Template Code'
    @UI.MultiLineText: true
    templateCode       : LargeString;
    
    @Common.Label: 'Configuration Schema'
    @Core.MediaType: 'application/json'
    configurationSchema : LargeString;
    
    @Common.Label: 'Dependencies'
    dependencies       : String(500); // Comma-separated
    
    @Common.Label: 'Required Capabilities'
    requiredCapabilities : String(500); // Comma-separated
    
    @Common.Label: 'Tags'
    tags               : String(300); // Comma-separated
    
    @Common.Label: 'Usage Count'
    usageCount         : Integer default 0;
    
    @Common.Label: 'Rating'
    @assert.range: [0, 5]
    rating             : Decimal(2, 1) default 0.0;
    
    @Common.Label: 'Is Featured'
    isFeatured         : Boolean default false;
    
    @Common.Label: 'Is Verified'
    isVerified         : Boolean default false;
    
    @Common.Label: 'Documentation URL'
    documentationUrl   : String(200);
    
    @Common.Label: 'Example Usage'
    @UI.MultiLineText: true
    exampleUsage       : LargeString;
    
    @Common.Label: 'Deployment Config'
    @Core.MediaType: 'application/json'
    deploymentConfig   : LargeString;
    
    @Common.Label: 'Test Configuration'
    @Core.MediaType: 'application/json'
    testConfiguration  : LargeString;
    
    @Common.Label: 'Security Requirements'
    @Core.MediaType: 'application/json'
    securityRequirements : String(1000);
    
    @Common.Label: 'Performance Requirements'
    @Core.MediaType: 'application/json'
    performanceRequirements : String(1000);
    
    // Associations
    builds             : Composition of many AgentBuilds on builds.template = $self;
    components         : Composition of many TemplateComponents on components.template = $self;
}

// Agent builds generated from templates
entity AgentBuilds : cuid, managed {
    @Common.Label: 'Build Name'
    @mandatory
    buildName          : String(100) not null;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Template'
    @assert.integrity
    template           : Association to AgentTemplates not null;
    
    @Common.Label: 'Build Status'
    buildStatus        : String(20) not null enum {
 IN_PROGRESS; COMPLETED; FAILED; CANCELLED;
    } default 'PENDING';
    
    @Common.Label: 'Build Type'
    buildType          : String(20) enum {
        DEVELOPMENT; TESTING; STAGING; PRODUCTION;
    } default 'DEVELOPMENT';
    
    @Common.Label: 'Generated Code'
    @UI.MultiLineText: true
    generatedCode      : LargeString;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Build Parameters'
    @Core.MediaType: 'application/json'
    buildParameters    : LargeString;
    
    @Common.Label: 'Build Log'
    @UI.MultiLineText: true
    buildLog           : LargeString;
    
    @Common.Label: 'Error Log'
    @UI.MultiLineText: true
    errorLog           : LargeString;
    
    @Common.Label: 'Build Start Time'
    buildStartTime     : DateTime;
    
    @Common.Label: 'Build End Time'
    buildEndTime       : DateTime;
    
    @Common.Label: 'Build Duration (ms)'
    buildDuration      : Integer;
    
    @Common.Label: 'Generated Agent ID'
    generatedAgentId   : String(50);
    
    @Common.Label: 'Deployment URL'
    deploymentUrl      : String(200);
    
    @Common.Label: 'Container Image'
    containerImage     : String(200);
    
    @Common.Label: 'Registry URL'
    registryUrl        : String(200);
    
    @Common.Label: 'Health Check URL'
    healthCheckUrl     : String(200);
    
    @Common.Label: 'API Documentation URL'
    apiDocumentationUrl : String(200);
    
    @Common.Label: 'Build Artifacts'
    @Core.MediaType: 'application/json'
    buildArtifacts     : LargeString;
    
    @Common.Label: 'Test Results'
    @Core.MediaType: 'application/json'
    testResults        : LargeString;
    
    @Common.Label: 'Security Scan Results'
    @Core.MediaType: 'application/json'
    securityScanResults : LargeString;
    
    @Common.Label: 'Quality Metrics'
    @Core.MediaType: 'application/json'
    qualityMetrics     : String(1000);
    
    @Common.Label: 'Performance Metrics'
    @Core.MediaType: 'application/json'
    performanceMetrics : String(1000);
    
    @Common.Label: 'Resource Requirements'
    @Core.MediaType: 'application/json'
    resourceRequirements : String(1000);
    
    @Common.Label: 'Environment Variables'
    @Core.MediaType: 'application/json'
    environmentVariables : String(1000);
    
    @Common.Label: 'Build Metadata'
    @Core.MediaType: 'application/json'
    buildMetadata      : String(1000);
    
    // Associations
    deployments        : Composition of many AgentDeployments on deployments.build = $self;
}

// Template components for modular development
entity TemplateComponents : cuid, managed {
    @Common.Label: 'Component Name'
    @mandatory
    componentName      : String(100) not null;
    
    @Common.Label: 'Template'
    @assert.integrity
    template           : Association to AgentTemplates not null;
    
    @Common.Label: 'Component Type'
    componentType      : String(30) not null enum {
        HANDLER; MIDDLEWARE; SERVICE; UTILITY; CONFIG; TEST;
    };
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Component Code'
    @UI.MultiLineText: true
    componentCode      : LargeString;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(1000);
    
    @Common.Label: 'Dependencies'
    dependencies       : String(300);
    
    @Common.Label: 'Is Required'
    isRequired         : Boolean default true;
    
    @Common.Label: 'Display Order'
    displayOrder       : Integer default 0;
    
    @Common.Label: 'Documentation'
    @UI.MultiLineText: true
    documentation      : LargeString;
    
    @Common.Label: 'Example Usage'
    @UI.MultiLineText: true
    exampleUsage       : String(1000);
}

// Agent deployments and runtime management
entity AgentDeployments : cuid, managed {
    @Common.Label: 'Deployment Name'
    @mandatory
    deploymentName     : String(100) not null;
    
    @Common.Label: 'Build'
    @assert.integrity
    build              : Association to AgentBuilds not null;
    
    @Common.Label: 'Deployment Status'
    deploymentStatus   : String(20) not null enum {
 DEPLOYING; DEPLOYED; FAILED; STOPPED; SCALING;
    } default 'PENDING';
    
    @Common.Label: 'Environment'
    environment        : String(20) enum {
        DEVELOPMENT; TESTING; STAGING; PRODUCTION;
    } default 'DEVELOPMENT';
    
    @Common.Label: 'Deployment Type'
    deploymentType     : String(20) enum {
        KUBERNETES; DOCKER; SERVERLESS; CONTAINER;
    } default 'KUBERNETES';
    
    @Common.Label: 'Agent ID'
    agentId            : String(50);
    
    @Common.Label: 'Agent Port'
    agentPort          : Integer;
    
    @Common.Label: 'Container ID'
    containerId        : String(100);
    
    @Common.Label: 'Image Tag'
    imageTag           : String(100);
    
    @Common.Label: 'Deployment URL'
    deploymentUrl      : String(200);
    
    @Common.Label: 'Health Status'
    healthStatus       : String(20) enum {
        HEALTHY; UNHEALTHY; UNKNOWN; STARTING;
    } default 'UNKNOWN';
    
    @Common.Label: 'CPU Usage (%)'
    cpuUsage           : Decimal(5, 2);
    
    @Common.Label: 'Memory Usage (MB)'
    memoryUsage        : Integer;
    
    @Common.Label: 'Request Count'
    requestCount       : Integer default 0;
    
    @Common.Label: 'Error Count'
    errorCount         : Integer default 0;
    
    @Common.Label: 'Average Response Time (ms)'
    averageResponseTime : Integer;
    
    @Common.Label: 'Uptime (seconds)'
    uptime             : Integer;
    
    @Common.Label: 'Last Health Check'
    lastHealthCheck    : DateTime;
    
    @Common.Label: 'Deployment Start Time'
    deploymentStartTime : DateTime;
    
    @Common.Label: 'Deployment End Time'
    deploymentEndTime   : DateTime;
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : LargeString;
    
    @Common.Label: 'Environment Variables'
    @Core.MediaType: 'application/json'
    environmentVariables : String(1000);
    
    @Common.Label: 'Resource Limits'
    @Core.MediaType: 'application/json'
    resourceLimits     : String(1000);
    
    @Common.Label: 'Scaling Configuration'
    @Core.MediaType: 'application/json'
    scalingConfiguration : String(1000);
    
    @Common.Label: 'Logs'
    @UI.MultiLineText: true
    logs               : LargeString;
    
    @Common.Label: 'Metrics'
    @Core.MediaType: 'application/json'
    metrics            : LargeString;
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Deployment Metadata'
    @Core.MediaType: 'application/json'
    deploymentMetadata : String(1000);
}

// Build pipeline configurations
entity BuildPipelines : cuid, managed {
    @Common.Label: 'Pipeline Name'
    @mandatory
    pipelineName       : String(100) not null;
    
    @Common.Label: 'Description'
    @UI.MultiLineText: true
    description        : String(500);
    
    @Common.Label: 'Pipeline Type'
    pipelineType       : String(20) enum {
        CONTINUOUS_INTEGRATION; CONTINUOUS_DEPLOYMENT; RELEASE; TESTING;
    } default 'CONTINUOUS_INTEGRATION';
    
    @Common.Label: 'Status'
    status             : String(20) enum {
        ACTIVE; INACTIVE; DRAFT; ARCHIVED;
    } default 'DRAFT';
    
    @Common.Label: 'Trigger Type'
    triggerType        : String(20) enum {
        MANUAL; SCHEDULED; WEBHOOK; CODE_CHANGE;
    } default 'MANUAL';
    
    @Common.Label: 'Pipeline Configuration'
    @Core.MediaType: 'application/json'
    pipelineConfiguration : LargeString;
    
    @Common.Label: 'Build Steps'
    @Core.MediaType: 'application/json'
    buildSteps         : LargeString;
    
    @Common.Label: 'Test Steps'
    @Core.MediaType: 'application/json'
    testSteps          : LargeString;
    
    @Common.Label: 'Deployment Steps'
    @Core.MediaType: 'application/json'
    deploymentSteps    : LargeString;
    
    @Common.Label: 'Quality Gates'
    @Core.MediaType: 'application/json'
    qualityGates       : String(1000);
    
    @Common.Label: 'Security Checks'
    @Core.MediaType: 'application/json'
    securityChecks     : String(1000);
    
    @Common.Label: 'Notification Settings'
    @Core.MediaType: 'application/json'
    notificationSettings : String(1000);
    
    @Common.Label: 'Environment Configuration'
    @Core.MediaType: 'application/json'
    environmentConfiguration : LargeString;
    
    @Common.Label: 'Resource Requirements'
    @Core.MediaType: 'application/json'
    resourceRequirements : String(1000);
    
    @Common.Label: 'Timeout (minutes)'
    timeoutMinutes     : Integer default 30;
    
    @Common.Label: 'Retry Attempts'
    retryAttempts      : Integer default 3;
    
    @Common.Label: 'Last Execution Time'
    lastExecutionTime  : DateTime;
    
    @Common.Label: 'Success Count'
    successCount       : Integer default 0;
    
    @Common.Label: 'Failure Count'
    failureCount       : Integer default 0;
    
    @Common.Label: 'Average Duration (minutes)'
    averageDuration    : Integer;
    
    @Common.Label: 'Pipeline Metadata'
    @Core.MediaType: 'application/json'
    pipelineMetadata   : String(1000);
}

// ============================================
// AGENT 14 - EMBEDDING FINE-TUNER ENTITIES
// ============================================

@Common.Label: 'Embedding Models'
entity EmbeddingModels : cuid, managed {
    @Common.Label: 'Model Name'
    @assert.notNull
    modelName          : String(200) @mandatory;
    
    @Common.Label: 'Base Model'
    baseModel          : String(200);
    
    @Common.Label: 'Model Type'
    modelType          : String(50) enum {
        SENTENCE_TRANSFORMER;
        OPENAI_EMBEDDING;
        BERT_BASE;
    } default 'SENTENCE_TRANSFORMER';
    
    @Common.Label: 'Domain'
    domain             : String(100);
    
    @Common.Label: 'Language'
    language           : String(20) default 'en';
    
    @Common.Label: 'Embedding Dimensions'
    embeddingDimensions: Integer default 768;
    
    @Common.Label: 'Model Status'
    status             : String(20) enum {
        DRAFT;
        TRAINING;
        TRAINED;
        EVALUATING;
        PRODUCTION;
    } default 'DRAFT';
    
    @Common.Label: 'Model Version'
    version            : String(20) default '1.0.0';
    
    @Common.Label: 'Model Path'
    modelPath          : String(500);
    
    @Common.Label: 'Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(2000);
    
    @Common.Label: 'Model Size (MB)'
    modelSizeMB        : Integer;
    
    @Common.Label: 'Parameter Count'
    parameterCount     : Integer;
    
    @Common.Label: 'Is Quantized'
    isQuantized        : Boolean default false;
    
    @Common.Label: 'Is ONNX Exported'
    isOnnxExported     : Boolean default false;
    
    @Common.Label: 'Training Runs'
    trainingRuns       : Composition of many TrainingRuns on trainingRuns.modelId = $self;
    
    @Common.Label: 'Model Evaluations'
    evaluations        : Composition of many ModelEvaluations on evaluations.modelId = $self;
    
    @Common.Label: 'Model Optimizations'
    optimizations      : Composition of many ModelOptimizations on optimizations.modelId = $self;
}

@Common.Label: 'Training Runs'
entity TrainingRuns : cuid, managed {
    @Common.Label: 'Model'
    modelId            : Association to EmbeddingModels;
    
    @Common.Label: 'Training Name'
    @assert.notNull
    trainingName       : String(200) @mandatory;
    
    @Common.Label: 'Training Strategy'
        CONTRASTIVE_LEARNING;
        TRIPLET_LOSS;
        MULTIPLE_NEGATIVES_RANKING;
        DOMAIN_ADAPTATION;
        CROSS_LINGUAL;
        FEW_SHOT_LEARNING;
    trainingStrategy   : String(50) default 'CONTRASTIVE_LEARNING';
    
    @Common.Label: 'Training Status'
        PREPARING;
    status             : String(20) default 'PREPARING';
    
    @Common.Label: 'Training Configuration'
    @Core.MediaType: 'application/json'
    trainingConfig     : String(3000);
    
    @Common.Label: 'Dataset Path'
    datasetPath        : String(500);
    
    @Common.Label: 'Training Data Size'
    trainingDataSize   : Integer;
    
    @Common.Label: 'Validation Split'
    validationSplit    : Decimal(3,2) default 0.2;
    
    @Common.Label: 'Batch Size'
    batchSize          : Integer default 32;
    
    @Common.Label: 'Learning Rate'
    learningRate       : Decimal(10,8) default 0.00002;
    
    @Common.Label: 'Epochs'
    epochs             : Integer default 10;
    
    @Common.Label: 'Current Epoch'
    currentEpoch       : Integer default 0;
    
    @Common.Label: 'Training Loss'
    trainingLoss       : Decimal(10,6);
    
    @Common.Label: 'Validation Loss'
    validationLoss     : Decimal(10,6);
    
    @Common.Label: 'Training Accuracy'
    trainingAccuracy   : Decimal(5,4);
    
    @Common.Label: 'Validation Accuracy'
    validationAccuracy : Decimal(5,4);
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration (minutes)'
    durationMinutes    : Integer;
    
    @Common.Label: 'GPU Usage'
    gpuUsage           : Boolean default false;
    
    @Common.Label: 'Memory Usage (MB)'
    memoryUsageMB      : Integer;
    
    @Common.Label: 'Training Logs'
    trainingLogs       : String(5000);
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Checkpoint Path'
    checkpointPath     : String(500);
}

@Common.Label: 'Model Evaluations'
entity ModelEvaluations : cuid, managed {
    @Common.Label: 'Model'
    modelId            : Association to EmbeddingModels;
    
    @Common.Label: 'Evaluation Name'
    @assert.notNull
    evaluationName     : String(200) @mandatory;
    
    @Common.Label: 'Evaluation Type'
        SIMILARITY_BENCHMARK;
        RETRIEVAL_ACCURACY;
        CLUSTERING_QUALITY;
        DOWNSTREAM_TASK;
        SPEED_BENCHMARK;
    evaluationType     : String(50) default 'SIMILARITY_BENCHMARK';
    
    @Common.Label: 'Benchmark Dataset'
    benchmarkDataset   : String(200);
    
    @Common.Label: 'Test Data Path'
    testDataPath       : String(500);
    
    @Common.Label: 'Evaluation Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Cosine Similarity Score'
    cosineSimilarity   : Decimal(5,4);
    
    @Common.Label: 'Precision@K'
    precisionAtK       : Decimal(5,4);
    
    @Common.Label: 'Recall@K'
    recallAtK          : Decimal(5,4);
    
    @Common.Label: 'F1 Score'
    f1Score            : Decimal(5,4);
    
    @Common.Label: 'Silhouette Score'
    silhouetteScore    : Decimal(5,4);
    
    @Common.Label: 'Davies-Bouldin Index'
    daviesBouldinIndex : Decimal(10,6);
    
    @Common.Label: 'Inference Speed (ms)'
    inferenceSpeedMs   : Integer;
    
    @Common.Label: 'Throughput (embeddings/sec)'
    throughput         : Integer;
    
    @Common.Label: 'Memory Footprint (MB)'
    memoryFootprintMB  : Integer;
    
    @Common.Label: 'Overall Score'
    overallScore       : Decimal(5,4);
    
    @Common.Label: 'Evaluation Results'
    @Core.MediaType: 'application/json'
    evaluationResults  : String(3000);
    
    @Common.Label: 'Comparison Baseline'
    comparisonBaseline : String(200);
    
    @Common.Label: 'Improvement Percentage'
    improvementPercent : Decimal(5,2);
    
    @Common.Label: 'Evaluation Time'
    evaluationTime     : DateTime;
    
    @Common.Label: 'Duration (minutes)'
    durationMinutes    : Integer;
}

@Common.Label: 'Model Optimizations'
entity ModelOptimizations : cuid, managed {
    @Common.Label: 'Model'
    modelId            : Association to EmbeddingModels;
    
    @Common.Label: 'Optimization Name'
    @assert.notNull
    optimizationName   : String(200) @mandatory;
    
    @Common.Label: 'Optimization Type'
        QUANTIZATION;
        PRUNING;
        KNOWLEDGE_DISTILLATION;
        DIMENSION_REDUCTION;
        ONNX_CONVERSION;
        TENSORRT_OPTIMIZATION;
        OPENVINO_OPTIMIZATION;
    optimizationType   : String(50) default 'QUANTIZATION';
    
    @Common.Label: 'Optimization Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Original Model Size (MB)'
    originalSizeMB     : Integer;
    
    @Common.Label: 'Optimized Model Size (MB)'
    optimizedSizeMB    : Integer;
    
    @Common.Label: 'Size Reduction (%)'
    sizeReductionPercent: Decimal(5,2);
    
    @Common.Label: 'Original Inference Time (ms)'
    originalInferenceMs: Integer;
    
    @Common.Label: 'Optimized Inference Time (ms)'
    optimizedInferenceMs: Integer;
    
    @Common.Label: 'Speed Improvement (%)'
    speedImprovementPercent: Decimal(5,2);
    
    @Common.Label: 'Accuracy Loss (%)'
    accuracyLossPercent: Decimal(5,2);
    
    @Common.Label: 'Optimization Configuration'
    @Core.MediaType: 'application/json'
    optimizationConfig : String(2000);
    
    @Common.Label: 'Optimized Model Path'
    optimizedModelPath : String(500);
    
    @Common.Label: 'Quality Score Before'
    qualityScoreBefore : Decimal(5,4);
    
    @Common.Label: 'Quality Score After'
    qualityScoreAfter  : Decimal(5,4);
    
    @Common.Label: 'Optimization Time'
    optimizationTime   : DateTime;
    
    @Common.Label: 'Duration (minutes)'
    durationMinutes    : Integer;
    
    @Common.Label: 'Optimization Logs'
    optimizationLogs   : String(2000);
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
}

@Common.Label: 'Fine-Tuning Tasks'
entity FineTuningTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @assert.notNull
    taskName           : String(200) @mandatory;
    
    @Common.Label: 'Base Model'
    baseModel          : String(200);
    
    @Common.Label: 'Target Domain'
    targetDomain       : String(100);
    
    @Common.Label: 'Task Type'
        DOMAIN_ADAPTATION;
        LANGUAGE_ADAPTATION;
        TASK_SPECIFIC;
        PERFORMANCE_OPTIMIZATION;
        MULTI_LINGUAL;
    taskType           : String(50) default 'DOMAIN_ADAPTATION';
    
    @Common.Label: 'Task Status'
        CREATED;
        DATA_PREPARATION;
        TRAINING;
        EVALUATION;
        OPTIMIZATION;
    status             : String(20) default 'CREATED';
    
    @Common.Label: 'Priority'
    priority           : String(20) default 'MEDIUM';
    
    @Common.Label: 'Training Data Size'
    trainingDataSize   : Integer;
    
    @Common.Label: 'Expected Duration (hours)'
    expectedDurationHours: Integer;
    
    @Common.Label: 'Actual Duration (hours)'
    actualDurationHours: Integer;
    
    @Common.Label: 'Progress Percentage'
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Stage'
    currentStage       : String(100);
    
    @Common.Label: 'Quality Target'
    qualityTarget      : Decimal(5,4) default 0.85;
    
    @Common.Label: 'Quality Achieved'
    qualityAchieved    : Decimal(5,4);
    
    @Common.Label: 'Performance Target (ms)'
    performanceTargetMs: Integer default 100;
    
    @Common.Label: 'Performance Achieved (ms)'
    performanceAchievedMs: Integer;
    
    @Common.Label: 'Resource Requirements'
    @Core.MediaType: 'application/json'
    resourceRequirements: String(1000);
    
    @Common.Label: 'Task Configuration'
    @Core.MediaType: 'application/json'
    taskConfiguration  : String(3000);
    
    @Common.Label: 'Generated Model'
    generatedModel     : Association to EmbeddingModels;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Created By'
    createdBy          : String(100);
    
    @Common.Label: 'Assigned To'
    assignedTo         : String(100);
    
    @Common.Label: 'Tags'
    tags               : String(500);
    
    @Common.Label: 'Notes'
    notes              : String(2000);
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
}

// ============================================
// AGENT 15 - ORCHESTRATOR ENTITIES
// ============================================

@Common.Label: 'Workflows'
entity Workflows : cuid, managed {
    @Common.Label: 'Workflow Name'
    @assert.notNull
    workflowName       : String(200) @mandatory;
    
    @Common.Label: 'Workflow Description'
    description        : String(1000);
    
    @Common.Label: 'Workflow Version'
    version            : String(20) default '1.0.0';
    
    @Common.Label: 'Workflow Status'
        DRAFT;
        ARCHIVED;
    status             : String(20) default 'DRAFT';
    
    @Common.Label: 'Workflow Type'
        DATA_PROCESSING;
        ML_PIPELINE;
        ETL;
        BATCH_PROCESSING;
        REAL_TIME;
    workflowType       : String(50) default 'DATA_PROCESSING';
    
    @Common.Label: 'Workflow Definition'
    @Core.MediaType: 'application/json'
    workflowDefinition : String(5000);
    
    @Common.Label: 'Schedule Configuration'
    @Core.MediaType: 'application/json'
    scheduleConfig     : String(1000);
    
    @Common.Label: 'Is Scheduled'
    isScheduled        : Boolean default false;
    
    @Common.Label: 'Priority Level'
    priority           : String(20) default 'MEDIUM';
    
    @Common.Label: 'Maximum Parallel Executions'
    maxParallelExecutions: Integer default 1;
    
    @Common.Label: 'Timeout Minutes'
    timeoutMinutes     : Integer default 60;
    
    @Common.Label: 'Retry Configuration'
    @Core.MediaType: 'application/json'
    retryConfig        : String(500);
    
    @Common.Label: 'Notification Configuration'
    @Core.MediaType: 'application/json'
    notificationConfig : String(1000);
    
    @Common.Label: 'Workflow Tags'
    tags               : String(500);
    
    @Common.Label: 'Workflow Category'
    category           : String(100);
    
    @Common.Label: 'Expected Duration (minutes)'
    expectedDurationMinutes: Integer;
    
    @Common.Label: 'SLA Minutes'
    slaMinutes         : Integer;
    
    @Common.Label: 'Is Public'
    isPublic           : Boolean default false;
    
    @Common.Label: 'Created By'
    createdBy          : String(100);
    
    @Common.Label: 'Last Modified By'
    lastModifiedBy     : String(100);
    
    @Common.Label: 'Total Executions'
    totalExecutions    : Integer default 0;
    
    @Common.Label: 'Successful Executions'
    successfulExecutions: Integer default 0;
    
    @Common.Label: 'Failed Executions'
    failedExecutions   : Integer default 0;
    
    @Common.Label: 'Average Duration (minutes)'
    averageDurationMinutes: Integer;
    
    @Common.Label: 'Last Execution Time'
    lastExecutionTime  : DateTime;
    
    @Common.Label: 'Next Scheduled Time'
    nextScheduledTime  : DateTime;
    
    @Common.Label: 'Workflow Executions'
    executions         : Composition of many WorkflowExecutions on executions.workflowId = $self;
    
    @Common.Label: 'Workflow Steps'
    steps              : Composition of many WorkflowSteps on steps.workflowId = $self;
}

@Common.Label: 'Workflow Executions'
entity WorkflowExecutions : cuid, managed {
    @Common.Label: 'Workflow'
    workflowId         : Association to Workflows;
    
    @Common.Label: 'Execution Name'
    executionName      : String(200);
    
    @Common.Label: 'Execution Status'
    status             : String(20) default 'QUEUED';
    
    @Common.Label: 'Execution Type'
    executionType      : String(20) default 'MANUAL';
    
    @Common.Label: 'Input Data'
    @Core.MediaType: 'application/json'
    inputData          : String(3000);
    
    @Common.Label: 'Output Data'
    @Core.MediaType: 'application/json'
    outputData         : String(3000);
    
    @Common.Label: 'Execution Context'
    @Core.MediaType: 'application/json'
    executionContext   : String(2000);
    
    @Common.Label: 'Progress Percentage'
    progressPercent    : Integer default 0;
    
    @Common.Label: 'Current Step'
    currentStep        : String(200);
    
    @Common.Label: 'Total Steps'
    totalSteps         : Integer;
    
    @Common.Label: 'Completed Steps'
    completedSteps     : Integer default 0;
    
    @Common.Label: 'Failed Steps'
    failedSteps        : Integer default 0;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Scheduled Time'
    scheduledTime      : DateTime;
    
    @Common.Label: 'Priority Override'
    priorityOverride   : String(20);
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(2000);
    
    @Common.Label: 'Error Details'
    @Core.MediaType: 'application/json'
    errorDetails       : String(3000);
    
    @Common.Label: 'Execution Logs'
    executionLogs      : String(5000);
    
    @Common.Label: 'Resource Usage'
    @Core.MediaType: 'application/json'
    resourceUsage      : String(1000);
    
    @Common.Label: 'Metrics'
    @Core.MediaType: 'application/json'
    metrics            : String(2000);
    
    @Common.Label: 'Triggered By'
    triggeredBy        : String(100);
    
    @Common.Label: 'Execution Environment'
    executionEnvironment: String(100);
    
    @Common.Label: 'Step Executions'
    stepExecutions     : Composition of many StepExecutions on stepExecutions.executionId = $self;
}

@Common.Label: 'Workflow Steps'
entity WorkflowSteps : cuid, managed {
    @Common.Label: 'Workflow'
    workflowId         : Association to Workflows;
    
    @Common.Label: 'Step Name'
    @assert.notNull
    stepName           : String(200) @mandatory;
    
    @Common.Label: 'Step Type'
    stepType           : String(50) default 'AGENT_CALL';
    
    @Common.Label: 'Step Order'
    stepOrder          : Integer;
    
    @Common.Label: 'Target Agent'
    targetAgent        : String(100);
    
    @Common.Label: 'Action Name'
    actionName         : String(100);
    
    @Common.Label: 'Step Configuration'
    @Core.MediaType: 'application/json'
    stepConfiguration  : String(2000);
    
    @Common.Label: 'Input Mapping'
    @Core.MediaType: 'application/json'
    inputMapping       : String(1000);
    
    @Common.Label: 'Output Mapping'
    @Core.MediaType: 'application/json'
    outputMapping      : String(1000);
    
    @Common.Label: 'Dependencies'
    dependencies       : String(500);
    
    @Common.Label: 'Conditions'
    @Core.MediaType: 'application/json'
    conditions         : String(1000);
    
    @Common.Label: 'Retry Policy'
    @Core.MediaType: 'application/json'
    retryPolicy        : String(500);
    
    @Common.Label: 'Timeout Minutes'
    timeoutMinutes     : Integer default 30;
    
    @Common.Label: 'Is Parallel'
    isParallel         : Boolean default false;
    
    @Common.Label: 'Parallel Group'
    parallelGroup      : String(100);
    
    @Common.Label: 'On Success'
    onSuccess          : String(200);
    
    @Common.Label: 'On Failure'
    onFailure          : String(200);
    
    @Common.Label: 'Is Optional'
    isOptional         : Boolean default false;
    
    @Common.Label: 'Description'
    description        : String(500);
    
    @Common.Label: 'Expected Duration (minutes)'
    expectedDurationMinutes: Integer;
    
    @Common.Label: 'Step Status'
    status             : String(20) default 'ACTIVE';
}

@Common.Label: 'Step Executions'
entity StepExecutions : cuid, managed {
    @Common.Label: 'Workflow Execution'
    executionId        : Association to WorkflowExecutions;
    
    @Common.Label: 'Workflow Step'
    stepId             : Association to WorkflowSteps;
    
    @Common.Label: 'Step Name'
    stepName           : String(200);
    
    @Common.Label: 'Execution Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Input Data'
    @Core.MediaType: 'application/json'
    inputData          : String(2000);
    
    @Common.Label: 'Output Data'
    @Core.MediaType: 'application/json'
    outputData         : String(2000);
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Error Details'
    @Core.MediaType: 'application/json'
    errorDetails       : String(2000);
    
    @Common.Label: 'Step Logs'
    stepLogs           : String(3000);
    
    @Common.Label: 'Retry Attempt'
    retryAttempt       : Integer default 0;
    
    @Common.Label: 'Agent Response Time (ms)'
    agentResponseTimeMs: Integer;
    
    @Common.Label: 'Resource Usage'
    @Core.MediaType: 'application/json'
    resourceUsage      : String(500);
    
    @Common.Label: 'Step Metrics'
    @Core.MediaType: 'application/json'
    stepMetrics        : String(1000);
}

@Common.Label: 'Orchestrator Tasks'
entity OrchestratorTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @assert.notNull
    taskName           : String(200) @mandatory;
    
    @Common.Label: 'Task Type'
    taskType           : String(50) default 'WORKFLOW_EXECUTION';
    
    @Common.Label: 'Task Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Priority'
    priority           : String(20) default 'MEDIUM';
    
    @Common.Label: 'Scheduled Time'
    scheduledTime      : DateTime;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Task Data'
    @Core.MediaType: 'application/json'
    taskData           : String(2000);
    
    @Common.Label: 'Task Result'
    @Core.MediaType: 'application/json'
    taskResult         : String(2000);
    
    @Common.Label: 'Target Agent'
    targetAgent        : String(100);
    
    @Common.Label: 'Execution Context'
    @Core.MediaType: 'application/json'
    executionContext   : String(1000);
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Task Logs'
    taskLogs           : String(2000);
    
    @Common.Label: 'Assigned Worker'
    assignedWorker     : String(100);
    
    @Common.Label: 'Queue Name'
    queueName          : String(100);
    
    @Common.Label: 'Parent Task'
    parentTaskId       : Association to OrchestratorTasks;
    
    @Common.Label: 'Related Workflow'
    workflowId         : Association to Workflows;
    
    @Common.Label: 'Related Execution'
    executionId        : Association to WorkflowExecutions;
}

@Common.Label: 'Pipeline Configurations'
entity PipelineConfigurations : cuid, managed {
    @Common.Label: 'Pipeline Name'
    @assert.notNull
    pipelineName       : String(200) @mandatory;
    
    @Common.Label: 'Pipeline Type'
        DATA_PIPELINE;
        ML_PIPELINE;
        ETL_PIPELINE;
        STREAMING_PIPELINE;
        BATCH_PIPELINE;
        HYBRID_PIPELINE;
    pipelineType       : String(50) default 'DATA_PIPELINE';
    
    @Common.Label: 'Pipeline Status'
        INACTIVE;
    status             : String(20) default 'ACTIVE';
    
    @Common.Label: 'Pipeline Configuration'
    @Core.MediaType: 'application/json'
    configuration      : String(3000);
    
    @Common.Label: 'Source Systems'
    sourceSystems      : String(500);
    
    @Common.Label: 'Target Systems'
    targetSystems      : String(500);
    
    @Common.Label: 'Processing Agents'
    processingAgents   : String(1000);
    
    @Common.Label: 'Schedule Expression'
    scheduleExpression : String(100);
    
    @Common.Label: 'Is Real Time'
    isRealTime         : Boolean default false;
    
    @Common.Label: 'Batch Size'
    batchSize          : Integer;
    
    @Common.Label: 'Parallelism Level'
    parallelismLevel   : Integer default 1;
    
    @Common.Label: 'Resource Requirements'
    @Core.MediaType: 'application/json'
    resourceRequirements: String(1000);
    
    @Common.Label: 'SLA Requirements'
    @Core.MediaType: 'application/json'
    slaRequirements    : String(500);
    
    @Common.Label: 'Monitoring Configuration'
    @Core.MediaType: 'application/json'
    monitoringConfig   : String(1000);
    
    @Common.Label: 'Alert Configuration'
    @Core.MediaType: 'application/json'
    alertConfig        : String(1000);
    
    @Common.Label: 'Data Quality Rules'
    @Core.MediaType: 'application/json'
    dataQualityRules   : String(2000);
    
    @Common.Label: 'Environment'
    environment        : String(50) default 'PRODUCTION';
    
    @Common.Label: 'Owner'
    owner              : String(100);
    
    @Common.Label: 'Created By'
    createdBy          : String(100);
    
    @Common.Label: 'Last Run Time'
    lastRunTime        : DateTime;
    
    @Common.Label: 'Next Run Time'
    nextRunTime        : DateTime;
    
    @Common.Label: 'Success Rate'
    successRate        : Decimal(5,4);
    
    @Common.Label: 'Average Runtime (minutes)'
    averageRuntimeMinutes: Integer;
    
    @Common.Label: 'Total Runs'
    totalRuns          : Integer default 0;
    
    @Common.Label: 'Successful Runs'
    successfulRuns     : Integer default 0;
    
    @Common.Label: 'Failed Runs'
    failedRuns         : Integer default 0;
}

// ============================================
// AGENT 0 - DATA PRODUCT AGENT ENTITIES
// ============================================

@Common.Label: 'Data Products'
entity DataProducts : cuid, managed {
    @Common.Label: 'Product Name'
    @assert.notNull
    productName        : String(200) @mandatory;
    
    @Common.Label: 'Product Description'
    description        : String(2000);
    
    @Common.Label: 'Product Type'
        DATASET;
        STREAM;
        API;
        FILE;
        DATABASE;
        HYBRID;
    productType        : String(50) default 'DATASET';
    
    @Common.Label: 'Product Status'
        DRAFT;
        INGESTING;
        PROCESSING;
        READY;
        ARCHIVED;
    status             : String(20) default 'DRAFT';
    
    @Common.Label: 'Data Source'
    dataSource         : String(500);
    
    @Common.Label: 'Data Format'
        JSON;
        XML;
        CSV;
        PARQUET;
        AVRO;
        EXCEL;
        PLAIN_TEXT;
        BINARY;
    dataFormat         : String(50) default 'JSON';
    
    @Common.Label: 'Data Size (MB)'
    dataSizeMB         : Integer;
    
    @Common.Label: 'Record Count'
    recordCount        : Integer;
    
    @Common.Label: 'Quality Score'
    qualityScore       : Decimal(5,4) default 0.0;
    
    @Common.Label: 'Is Public'
    isPublic           : Boolean default false;
    
    @Common.Label: 'Is Encrypted'
    isEncrypted        : Boolean default false;
    
    @Common.Label: 'Compression Type'
    compressionType    : String(50);
    
    @Common.Label: 'Schema Version'
    schemaVersion      : String(20) default '1.0.0';
    
    @Common.Label: 'Data Schema'
    @Core.MediaType: 'application/json'
    dataSchema         : String(3000);
    
    @Common.Label: 'Sample Data'
    @Core.MediaType: 'application/json'
    sampleData         : String(2000);
    
    @Common.Label: 'Tags'
    tags               : String(500);
    
    @Common.Label: 'Category'
    category           : String(100);
    
    @Common.Label: 'Owner'
    owner              : String(100);
    
    @Common.Label: 'Data Classification'
        PUBLIC;
        INTERNAL;
        CONFIDENTIAL;
        RESTRICTED;
        HIGHLY_RESTRICTED;
    dataClassification : String(50) default 'INTERNAL';
    
    @Common.Label: 'Retention Days'
    retentionDays      : Integer default 365;
    
    @Common.Label: 'Expiry Date'
    expiryDate         : Date;
    
    @Common.Label: 'Created By'
    createdBy          : String(100);
    
    @Common.Label: 'Last Accessed'
    lastAccessed       : DateTime;
    
    @Common.Label: 'Access Count'
    accessCount        : Integer default 0;
    
    @Common.Label: 'Processing Pipeline'
    processingPipeline : String(200);
    
    @Common.Label: 'Next Agent'
    nextAgent          : String(100) default 'dataStandardizationAgent';
    
    @Common.Label: 'Dublin Core Metadata'
    dublinCoreMetadata : Composition of one DublinCoreMetadata on dublinCoreMetadata.dataProduct = $self;
    
    @Common.Label: 'Ingestion Sessions'
    ingestionSessions  : Composition of many IngestionSessions on ingestionSessions.dataProduct = $self;
    
    @Common.Label: 'Quality Assessments'
    qualityAssessments : Composition of many QualityAssessments on qualityAssessments.dataProduct = $self;
    
    @Common.Label: 'Product Transformations'
    transformations    : Composition of many ProductTransformations on transformations.dataProduct = $self;
}

@Common.Label: 'Dublin Core Metadata'
entity DublinCoreMetadata : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct        : Association to DataProducts;
    
    @Common.Label: 'Title'
    @assert.notNull
    title              : String(500) @mandatory;
    
    @Common.Label: 'Creator'
    @assert.notNull
    creator            : String(200) @mandatory;
    
    @Common.Label: 'Subject'
    subject            : String(500);
    
    @Common.Label: 'Description'
    description        : String(2000);
    
    @Common.Label: 'Publisher'
    publisher          : String(200);
    
    @Common.Label: 'Contributor'
    contributor        : String(500);
    
    @Common.Label: 'Date'
    @assert.notNull
    date               : Date @mandatory;
    
    @Common.Label: 'Type'
    @assert.notNull
    type               : String(100) @mandatory;
    
    @Common.Label: 'Format'
    format             : String(100);
    
    @Common.Label: 'Identifier'
    identifier         : String(200);
    
    @Common.Label: 'Source'
    source             : String(500);
    
    @Common.Label: 'Language'
    language           : String(20) default 'en';
    
    @Common.Label: 'Relation'
    relation           : String(500);
    
    @Common.Label: 'Coverage'
    coverage           : String(500);
    
    @Common.Label: 'Rights'
    rights             : String(1000);
    
    @Common.Label: 'Additional Metadata'
    @Core.MediaType: 'application/json'
    additionalMetadata : String(2000);
    
    @Common.Label: 'Metadata Version'
    metadataVersion    : String(20) default '1.0';
    
    @Common.Label: 'Is Complete'
    isComplete         : Boolean default false;
    
    @Common.Label: 'Validation Status'
        VALID;
        PARTIALLY_VALID;
        INVALID;
        NOT_VALIDATED;
    validationStatus   : String(20) default 'NOT_VALIDATED';
}

@Common.Label: 'Ingestion Sessions'
entity IngestionSessions : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct        : Association to DataProducts;
    
    @Common.Label: 'Session Name'
    sessionName        : String(200);
    
    @Common.Label: 'Ingestion Type'
        BATCH;
        STREAMING;
        API_PULL;
        FILE_UPLOAD;
        DATABASE_EXTRACT;
    ingestionType      : String(50) default 'BATCH';
    
    @Common.Label: 'Session Status'
        STARTING;
    status             : String(20) default 'STARTING';
    
    @Common.Label: 'Source URL'
    sourceUrl          : String(500);
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Records Ingested'
    recordsIngested    : Integer default 0;
    
    @Common.Label: 'Records Failed'
    recordsFailed      : Integer default 0;
    
    @Common.Label: 'Data Volume (MB)'
    dataVolumeMB       : Integer;
    
    @Common.Label: 'Throughput (records/sec)'
    throughput         : Integer;
    
    @Common.Label: 'Error Count'
    errorCount         : Integer default 0;
    
    @Common.Label: 'Error Summary'
    errorSummary       : String(2000);
    
    @Common.Label: 'Ingestion Configuration'
    @Core.MediaType: 'application/json'
    ingestionConfig    : String(2000);
    
    @Common.Label: 'Transformation Rules'
    @Core.MediaType: 'application/json'
    transformationRules: String(2000);
    
    @Common.Label: 'Quality Checks Performed'
    qualityChecksPerformed: String(500);
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Checkpoint Data'
    @Core.MediaType: 'application/json'
    checkpointData     : String(1000);
}

@Common.Label: 'Quality Assessments'
entity QualityAssessments : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct        : Association to DataProducts;
    
    @Common.Label: 'Assessment Name'
    assessmentName     : String(200);
    
    @Common.Label: 'Assessment Type'
        COMPLETENESS;
        VALIDITY;
        CONSISTENCY;
        ACCURACY;
        TIMELINESS;
        UNIQUENESS;
        COMPREHENSIVE;
    assessmentType     : String(50) default 'COMPREHENSIVE';
    
    @Common.Label: 'Assessment Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Overall Score'
    overallScore       : Decimal(5,4) default 0.0;
    
    @Common.Label: 'Completeness Score'
    completenessScore  : Decimal(5,4);
    
    @Common.Label: 'Validity Score'
    validityScore      : Decimal(5,4);
    
    @Common.Label: 'Consistency Score'
    consistencyScore   : Decimal(5,4);
    
    @Common.Label: 'Accuracy Score'
    accuracyScore      : Decimal(5,4);
    
    @Common.Label: 'Timeliness Score'
    timelinessScore    : Decimal(5,4);
    
    @Common.Label: 'Uniqueness Score'
    uniquenessScore    : Decimal(5,4);
    
    @Common.Label: 'Issues Found'
    issuesFound        : Integer default 0;
    
    @Common.Label: 'Critical Issues'
    criticalIssues     : Integer default 0;
    
    @Common.Label: 'Assessment Details'
    @Core.MediaType: 'application/json'
    assessmentDetails  : String(3000);
    
    @Common.Label: 'Recommendations'
    recommendations    : String(2000);
    
    @Common.Label: 'Assessment Time'
    assessmentTime     : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Auto Rejection'
    autoRejection      : Boolean default false;
    
    @Common.Label: 'Rejection Reason'
    rejectionReason    : String(500);
}

@Common.Label: 'Product Transformations'
entity ProductTransformations : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct        : Association to DataProducts;
    
    @Common.Label: 'Transformation Name'
    transformationName : String(200);
    
    @Common.Label: 'Transformation Type'
        FORMAT_CONVERSION;
        SCHEMA_MAPPING;
        DATA_CLEANSING;
        ENRICHMENT;
        AGGREGATION;
        FILTERING;
        NORMALIZATION;
    transformationType : String(50) default 'FORMAT_CONVERSION';
    
    @Common.Label: 'Transformation Status'
    status             : String(20) default 'PENDING';
    
    @Common.Label: 'Source Format'
    sourceFormat       : String(50);
    
    @Common.Label: 'Target Format'
    targetFormat       : String(50);
    
    @Common.Label: 'Transformation Rules'
    @Core.MediaType: 'application/json'
    transformationRules: String(3000);
    
    @Common.Label: 'Records Processed'
    recordsProcessed   : Integer default 0;
    
    @Common.Label: 'Records Transformed'
    recordsTransformed : Integer default 0;
    
    @Common.Label: 'Records Failed'
    recordsFailed      : Integer default 0;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Output Path'
    outputPath         : String(500);
    
    @Common.Label: 'Transformation Logs'
    transformationLogs : String(2000);
    
    @Common.Label: 'Error Details'
    errorDetails       : String(1000);
    
    @Common.Label: 'Is Reversible'
    isReversible       : Boolean default false;
    
    @Common.Label: 'Rollback Available'
    rollbackAvailable  : Boolean default false;
}

@Common.Label: 'Data Product Tasks'
entity DataProductTasks : cuid, managed {
    @Common.Label: 'Task Name'
    @assert.notNull
    taskName           : String(200) @mandatory;
    
    @Common.Label: 'Task Type'
        CREATE_PRODUCT;
        INGEST_DATA;
        TRANSFORM_DATA;
        VALIDATE_QUALITY;
        ENRICH_METADATA;
        ROUTE_TO_AGENT;
        ARCHIVE_PRODUCT;
    taskType           : String(50) default 'CREATE_PRODUCT';
    
    @Common.Label: 'Task Status'
        CREATED;
    status             : String(20) default 'CREATED';
    
    @Common.Label: 'Priority'
    priority           : String(20) default 'MEDIUM';
    
    @Common.Label: 'Related Product'
    dataProduct        : Association to DataProducts;
    
    @Common.Label: 'Task Configuration'
    @Core.MediaType: 'application/json'
    taskConfiguration  : String(2000);
    
    @Common.Label: 'Input Data'
    @Core.MediaType: 'application/json'
    inputData          : String(2000);
    
    @Common.Label: 'Output Data'
    @Core.MediaType: 'application/json'
    outputData         : String(2000);
    
    @Common.Label: 'Scheduled Time'
    scheduledTime      : DateTime;
    
    @Common.Label: 'Start Time'
    startTime          : DateTime;
    
    @Common.Label: 'End Time'
    endTime            : DateTime;
    
    @Common.Label: 'Duration Minutes'
    durationMinutes    : Integer;
    
    @Common.Label: 'Retry Count'
    retryCount         : Integer default 0;
    
    @Common.Label: 'Max Retries'
    maxRetries         : Integer default 3;
    
    @Common.Label: 'Error Message'
    errorMessage       : String(1000);
    
    @Common.Label: 'Task Logs'
    taskLogs           : String(2000);
    
    @Common.Label: 'Created By'
    createdBy          : String(100);
    
    @Common.Label: 'Assigned To'
    assignedTo         : String(100);
}
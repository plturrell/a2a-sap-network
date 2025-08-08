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

// Core Agent entities
entity Agents : cuid, managed {
    address      : String(42) @mandatory;
    name         : localized String(100) @mandatory;
    endpoint     : String(500);
    reputation   : Integer default 100;
    isActive     : Boolean default true;
    country      : Country;
    capabilities : Composition of many AgentCapabilities on capabilities.agent = $self;
    services     : Composition of many Services on services.provider = $self;
    messages     : Association to many Messages on messages.sender = $self;
    performance  : Composition of one AgentPerformance on performance.agent = $self;
}


entity AgentCapabilities : cuid {
    agent        : Association to Agents;
    capability   : Association to Capabilities;
    registeredAt : DateTime;
    version      : String(20);
    status       : String(20) enum { active; deprecated; sunset; };
}

// Capability Registry
entity Capabilities : cuid, managed {
    name         : localized String(100) @mandatory;
    description  : localized String(1000);
    category     : Association to CapabilityCategories;
    tags         : array of String;
    inputTypes   : array of String;
    outputTypes  : array of String;
    version      : String(20) default '1.0.0';
    status       : Association to StatusCodes default 'active';
    dependencies : array of String;
    conflicts    : array of String;
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

// Service Marketplace
entity Services : cuid, managed {
    provider        : Association to Agents;
    name           : String(100) @mandatory;
    description    : String(1000);
    category       : String(50);
    pricePerCall   : Decimal(10,4);
    currency       : Currency default 'EUR';
    minReputation  : Integer default 0;
    maxCallsPerDay : Integer default 1000;
    isActive       : Boolean default true;
    totalCalls     : Integer default 0;
    averageRating  : Decimal(3,2) default 0;
    escrowAmount   : Decimal(10,4) default 0;
    serviceOrders  : Composition of many ServiceOrders on serviceOrders.service = $self;
}

entity ServiceOrders : cuid, managed {
    service         : Association to Services;
    consumer        : Association to Agents;
    status          : String(20) enum { pending; active; completed; cancelled; disputed; };
    callCount       : Integer default 0;
    totalAmount     : Decimal(10,4);
    escrowReleased  : Boolean default false;
    completedAt     : DateTime;
    rating          : Integer;
    feedback        : String(500);
}

// Message Routing
entity Messages : cuid, managed {
    sender      : Association to Agents;
    recipient   : Association to Agents;
    messageHash : String(66) @mandatory;
    protocol    : String(50);
    priority    : Integer default 1;
    status      : String(20) enum { pending; sent; delivered; failed; };
    retryCount  : Integer default 0;
    gasUsed     : Integer;
    deliveredAt : DateTime;
}

// Performance & Reputation
entity AgentPerformance : cuid {
    agent               : Association to one Agents;
    totalTasks          : Integer default 0;
    successfulTasks     : Integer default 0;
    failedTasks         : Integer default 0;
    averageResponseTime : Integer; // milliseconds
    averageGasUsage     : Integer;
    reputationScore     : Integer default 100;
    trustScore          : Decimal(3,2) default 1.0;
    lastUpdated         : DateTime;
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
entity Workflows : cuid, managed {
    name        : String(100) @mandatory;
    description : String(1000);
    definition  : LargeString; // JSON workflow definition
    isActive    : Boolean default true;
    category    : String(50);
    owner       : Association to Agents;
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
    execution    : Association to WorkflowExecutions;
    stepNumber   : Integer;
    agentAddress : String(42);
    action       : String(100);
    input        : LargeString; // JSON
    output       : LargeString; // JSON
    status       : String(20) enum { pending; running; completed; failed; };
    gasUsed      : Integer;
    startedAt    : DateTime;
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
entity NetworkStats : temporal, cuid {
    totalAgents      : Integer;
    activeAgents     : Integer;
    totalServices    : Integer;
    totalCapabilities: Integer;
    totalMessages    : Integer;
    totalTransactions: Integer;
    averageReputation: Decimal(5,2);
    networkLoad      : Decimal(5,2);
    gasPrice         : Decimal(10,4);
}

// Configuration
entity NetworkConfig : cuid, managed {
    configKey   : String(100) @mandatory;
    value       : String(1000);
    description : String(500);
    isActive    : Boolean default true;
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
entity AuditLog : managed {
    key ID : UUID;
    tenant : String(36);
    user : String(100);
    action : String(50);
    entity : String(100);
    entityKey : String(100);
    oldValue : LargeString;
    newValue : LargeString;
    ip : String(45);
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
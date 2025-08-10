namespace a2a.platform;

using { cuid, managed, temporal, Currency, Country } from '@sap/cds/common';

// Data Product Entities
entity DataProducts : cuid, managed {
    name            : String(255) not null;
    description     : String(5000);
    type            : String(100) not null; // financial_time_series, reference_data, etc.
    sourceSystem    : String(100) not null;
    dataFormat      : String(50);
    schemaVersion   : String(20);
    status          : String(50) default 'ACTIVE';
    
    // Business metadata
    assetClass      : String(100);
    frequency       : String(50);
    startDate       : Date;
    endDate         : Date;
    
    // Relationships
    validationRules : Composition of many ValidationRules on validationRules.dataProduct = $self;
    processingSteps : Composition of many ProcessingSteps on processingSteps.dataProduct = $self;
    qualityMetrics  : Composition of many QualityMetrics on qualityMetrics.dataProduct = $self;
}

entity ValidationRules : cuid {
    dataProduct     : Association to DataProducts;
    ruleType        : String(100) not null; // COMPLETENESS, ACCURACY, etc.
    threshold       : Decimal(5,4);
    parameters      : LargeString;
    isActive        : Boolean default true;
}

// Agent Processing
entity Agents : cuid {
    agentId         : String(50) not null unique;
    name            : String(255) not null;
    type            : String(100) not null; // data_product, standardization, etc.
    version         : String(20);
    status          : String(50) default 'ACTIVE';
    endpoint        : String(500);
    capabilities    : LargeString; // JSON array
    
    // Performance metrics
    lastHeartbeat   : Timestamp;
    responseTime    : Integer; // milliseconds
    successRate     : Decimal(5,4);
    throughput      : Integer; // messages per hour
    
    // Configuration
    configuration   : LargeString; // JSON configuration
    maxConcurrency  : Integer default 10;
    timeoutMs       : Integer default 30000;
}

entity ProcessingSteps : cuid, managed {
    dataProduct     : Association to DataProducts;
    agent           : Association to Agents;
    stepOrder       : Integer not null;
    status          : String(50); // PENDING, PROCESSING, COMPLETED, FAILED
    startTime       : Timestamp;
    endTime         : Timestamp;
    duration        : Integer; // milliseconds
    inputHash       : String(64);
    outputHash      : String(64);
    errorMessage    : String(1000);
    
    // Transformations applied
    transformations : Composition of many Transformations on transformations.processingStep = $self;
}

entity Transformations : cuid {
    processingStep  : Association to ProcessingSteps;
    transformationType : String(100);
    fieldName       : String(255);
    originalValue   : String(1000);
    transformedValue : String(1000);
    confidence      : Decimal(5,4);
}

// Quality Management
entity QualityMetrics : cuid, managed {
    dataProduct     : Association to DataProducts;
    metricType      : String(100); // COMPLETENESS, ACCURACY, CONSISTENCY, etc.
    value           : Decimal(15,10);
    threshold       : Decimal(5,4);
    status          : String(50); // PASS, FAIL, WARNING
    measurementDate : Timestamp;
    
    details         : LargeString; // JSON details
}

entity QualityIssues : cuid, managed {
    dataProduct     : Association to DataProducts;
    severity        : String(50); // CRITICAL, HIGH, MEDIUM, LOW
    category        : String(100);
    description     : String(5000);
    affectedFields  : LargeString; // JSON array
    resolution      : String(5000);
    status          : String(50) default 'OPEN';
    assignedTo      : String(255);
    dueDate         : Date;
}

// Workflow Management
entity Workflows : cuid, managed {
    name            : String(255) not null;
    description     : String(5000);
    definition      : LargeString not null; // BPMN XML or JSON
    version         : String(20);
    status          : String(50) default 'ACTIVE';
    
    // Execution statistics
    totalExecutions : Integer default 0;
    successfulExecutions : Integer default 0;
    avgExecutionTime : Integer; // milliseconds
    lastExecution   : Timestamp;
}

entity WorkflowInstances : cuid, managed {
    workflow        : Association to Workflows;
    status          : String(50); // RUNNING, COMPLETED, FAILED, CANCELLED
    startTime       : Timestamp;
    endTime         : Timestamp;
    duration        : Integer;
    currentStep     : String(255);
    
    // Context and variables
    inputData       : LargeString; // JSON
    outputData      : LargeString; // JSON
    variables       : LargeString; // JSON
    errorMessage    : String(1000);
    
    // Steps executed
    executionSteps  : Composition of many WorkflowSteps on executionSteps.workflowInstance = $self;
}

entity WorkflowSteps : cuid, managed {
    workflowInstance : Association to WorkflowInstances;
    stepName        : String(255) not null;
    stepType        : String(100); // AGENT_TASK, DECISION, PARALLEL, etc.
    agent           : Association to Agents;
    status          : String(50);
    startTime       : Timestamp;
    endTime         : Timestamp;
    duration        : Integer;
    inputData       : LargeString;
    outputData      : LargeString;
    errorMessage    : String(1000);
}

// Trust and Security
entity TrustRelationships : cuid, managed {
    sourceAgent     : Association to Agents;
    targetAgent     : Association to Agents;
    trustLevel      : Decimal(3,2); // 0.00 to 1.00
    relationshipType : String(100); // DIRECT, TRANSITIVE, DELEGATED
    validFrom       : Timestamp;
    validTo         : Timestamp;
    
    // Trust factors
    authenticationStrength : Decimal(3,2);
    behaviorScore   : Decimal(3,2);
    networkReputation : Decimal(3,2);
    complianceScore : Decimal(3,2);
    
    status          : String(50) default 'ACTIVE';
    lastVerified    : Timestamp;
}

entity SecurityEvents : cuid, managed {
    eventType       : String(100) not null;
    severity        : String(50); // CRITICAL, HIGH, MEDIUM, LOW, INFO
    source          : String(255);
    target          : String(255);
    description     : String(5000);
    
    // Event details
    userAgent       : String(500);
    ipAddress       : String(45);
    sessionId       : String(100);
    correlationId   : String(100);
    
    // Risk assessment
    riskScore       : Decimal(3,2);
    automatedResponse : String(1000);
    status          : String(50) default 'OPEN';
    investigatedBy  : String(255);
    resolvedAt      : Timestamp;
}

// Registry and Discovery
entity ServiceRegistry : cuid, managed {
    serviceId       : String(100) not null unique;
    name            : String(255) not null;
    description     : String(5000);
    version         : String(20);
    endpoint        : String(500) not null;
    
    // ORD compliance
    ordId           : String(255);
    ordType         : String(100); // API, Event, EntityType, etc.
    ordPackage      : String(255);
    ordVisibility   : String(50) default 'PUBLIC';
    
    // Service details
    protocol        : String(50); // REST, GraphQL, gRPC, etc.
    documentation   : String(1000);
    tags            : LargeString; // JSON array
    
    // Health and metrics
    healthStatus    : String(50); // HEALTHY, DEGRADED, UNHEALTHY
    lastHealthCheck : Timestamp;
    avgResponseTime : Integer;
    uptime          : Decimal(5,4);
    
    // Capabilities
    capabilities    : Composition of many ServiceCapabilities on capabilities.service = $self;
}

entity ServiceCapabilities : cuid {
    service         : Association to ServiceRegistry;
    capabilityType  : String(100);
    name            : String(255);
    description     : String(1000);
    inputSchema     : LargeString; // JSON Schema
    outputSchema    : LargeString; // JSON Schema
    slaGuarantee    : String(500);
}

// Financial Data Specific Entities
entity FinancialInstruments : cuid, managed {
    instrumentId    : String(50) not null unique;
    name            : String(255);
    type            : String(100); // EQUITY, BOND, DERIVATIVE, etc.
    assetClass      : String(100);
    currency        : Currency;
    country         : Country;
    
    // Market data
    isin            : String(12);
    cusip           : String(9);
    bloomberg       : String(20);
    reuters         : String(20);
    
    // Status
    status          : String(50) default 'ACTIVE';
    listingDate     : Date;
    maturityDate    : Date;
    
    // Relationships
    priceData       : Composition of many PriceData on priceData.instrument = $self;
    riskMetrics     : Composition of many RiskMetrics on riskMetrics.instrument = $self;
}

entity PriceData : cuid, managed, temporal {
    instrument      : Association to FinancialInstruments;
    priceType       : String(50); // LAST, BID, ASK, CLOSE, etc.
    price           : Decimal(15,4);
    currency        : Currency;
    volume          : Integer;
    timestamp       : Timestamp;
    
    // Market data quality
    source          : String(100);
    confidence      : Decimal(3,2);
    staleFlag       : Boolean default false;
}

entity RiskMetrics : cuid, managed {
    instrument      : Association to FinancialInstruments;
    metricType      : String(100); // VAR, CVaR, BETA, etc.
    value           : Decimal(15,8);
    confidence      : Decimal(3,2);
    timeHorizon     : Integer; // days
    calculationDate : Date;
    
    // Methodology
    methodology     : String(100);
    parameters      : LargeString; // JSON
    modelVersion    : String(20);
}

// Audit and Compliance
entity AuditLog : cuid, managed {
    eventType       : String(100) not null;
    objectType      : String(100);
    objectId        : String(100);
    action          : String(100);
    
    // User context
    userId          : String(255);
    sessionId       : String(100);
    correlationId   : String(100);
    
    // Request details
    endpoint        : String(500);
    httpMethod      : String(10);
    userAgent       : String(500);
    ipAddress       : String(45);
    
    // Change details
    oldValues       : LargeString; // JSON
    newValues       : LargeString; // JSON
    
    // Result
    status          : String(50); // SUCCESS, FAILURE
    errorMessage    : String(1000);
    duration        : Integer; // milliseconds
}

entity ComplianceRules : cuid, managed {
    name            : String(255) not null;
    description     : String(5000);
    ruleType        : String(100); // GDPR, SOX, BASEL, etc.
    category        : String(100);
    severity        : String(50);
    
    // Rule definition
    condition       : LargeString; // JSON or rule expression
    action          : LargeString; // JSON action definition
    
    // Status and lifecycle
    status          : String(50) default 'ACTIVE';
    effectiveFrom   : Date;
    effectiveTo     : Date;
    reviewDate      : Date;
    
    // Violations
    violations      : Composition of many ComplianceViolations on violations.rule = $self;
}

entity ComplianceViolations : cuid, managed {
    rule            : Association to ComplianceRules;
    severity        : String(50);
    description     : String(5000);
    
    // Context
    objectType      : String(100);
    objectId        : String(100);
    userId          : String(255);
    
    // Resolution
    status          : String(50) default 'OPEN';
    assignedTo      : String(255);
    resolution      : String(5000);
    resolvedAt      : Timestamp;
    dueDate         : Date;
}
namespace a2a.platform;

using { a2a.platform.Identifiable, a2a.platform.StatusTracking, a2a.platform.BusinessMetadata, a2a.platform.PerformanceMetrics, a2a.platform.ConfigurationSettings } from '../aspects/common';

// Agent-related entities
entity Agents : Identifiable, StatusTracking, BusinessMetadata, PerformanceMetrics {
    agentId : String(50) not null unique;
    endpoint : String(500);
    capabilities : LargeString; // JSON array
    
    // Relationships
    processingSteps : Composition of many ProcessingSteps on processingSteps.agent = $self;
    configurations : Composition of many AgentConfigurations on configurations.agent = $self;
    metrics : Composition of many AgentMetrics on metrics.agent = $self;
}

entity AgentConfigurations : Identifiable, ConfigurationSettings {
    agent : Association to Agents;
    configKey : String(100) not null;
    configValue : LargeString;
    environment : String(50); // dev, staging, production
}

entity AgentMetrics : Identifiable {
    agent : Association to Agents;
    metricName : String(100) not null;
    metricValue : Decimal(15,4);
    metricUnit : String(50);
    timestamp : Timestamp not null;
    tags : LargeString; // JSON key-value pairs
}

entity ProcessingSteps : Identifiable, StatusTracking, ProcessingMetadata {
    agent : Association to Agents;
    dataProduct : Association to DataProducts;
    workflow : Association to Workflows;
    stepOrder : Integer not null;
    stepType : String(100) not null; // validation, transformation, etc.
    input : LargeString; // JSON
    output : LargeString; // JSON
    errors : LargeString; // JSON array of errors
}
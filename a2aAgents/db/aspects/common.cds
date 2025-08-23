namespace a2a.platform;

using { cuid, managed, temporal } from '@sap/cds/common';

// Common aspects for reuse across entities
aspect Identifiable : cuid;
aspect Manageable : managed;
aspect Timestampable : temporal;

aspect StatusTracking {
    status : String(50) default 'ACTIVE';
    statusMessage : String(500);
    statusUpdatedAt : Timestamp;
}

aspect PerformanceMetrics {
    lastHeartbeat : Timestamp;
    responseTime : Integer; // milliseconds
    successRate : Decimal(5,4);
    throughput : Integer; // messages per hour
}

aspect BusinessMetadata {
    name : String(255) not null;
    description : String(5000);
    type : String(100) not null;
    version : String(20);
}

aspect ProcessingMetadata {
    startTime : Timestamp;
    endTime : Timestamp;
    duration : Integer; // milliseconds
    recordsProcessed : Integer;
    recordsSucceeded : Integer;
    recordsFailed : Integer;
}

aspect QualityTracking {
    qualityScore : Decimal(5,4);
    completenessScore : Decimal(5,4); 
    accuracyScore : Decimal(5,4);
    consistencyScore : Decimal(5,4);
    timelinessScore : Decimal(5,4);
}

aspect ConfigurationSettings {
    parameters : LargeString; // JSON configuration
    isActive : Boolean default true;
    priority : Integer default 0;
}
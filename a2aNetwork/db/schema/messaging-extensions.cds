using { a2a.network } from './schema';

namespace a2a.messaging;

/**
 * Message Persistence Layer Entities
 */
entity Messages {
    key messageId        : String(36);
    content             : LargeString;
    metadata            : LargeString;  // JSON metadata
    priority            : String(10) default 'normal';
    ttl                 : Integer;      // Time to live in seconds
    status              : String(20) default 'active';
    createdAt           : Timestamp;
    updatedAt           : Timestamp;
    expiresAt           : Timestamp;
    
    // Indexing for performance
    @cds.persistence.index: true
    agentId             : String(100);
    
    @cds.persistence.index: true  
    messageType         : String(50);
    
    @cds.persistence.index: true
    conversationId      : String(36);
}

entity MessageArchive {
    key archiveId           : String(36);
    messageId               : String(36);
    originalContent         : LargeString;
    compressedContent       : LargeBinary;  // Compressed message data
    metadata               : LargeString;   // JSON metadata including compression info
    archivedAt             : Timestamp;
    originalCreatedAt      : Timestamp;
    
    // Archive indexing
    @cds.persistence.index: true
    archiveDate            : Date;
    
    @cds.persistence.index: true
    messageType            : String(50);
}

entity MessageMetrics {
    key date               : Date;
    key operation          : String(20);   // persisted, retrieved, archived
    key priority           : String(10);
    count                  : Integer default 1;
    timestamp              : Timestamp;
}

/**
 * Dead Letter Queue Entities  
 */
namespace a2a.dlq;

entity FailedMessages {
    key dlqId              : String(36);
    messageId              : String(36);
    originalMessage        : LargeString;   // JSON of original message
    failureReason          : String(500);
    failureDetails         : LargeString;   // JSON error details
    retryCount             : Integer default 0;
    maxRetries             : Integer default 3;
    nextRetryAt            : Timestamp;
    addedAt                : Timestamp;
    lastAttemptAt          : Timestamp;
    
    // Classification fields
    agentId                : String(100);
    messageType            : String(50);
    priority               : String(10) default 'normal';
    status                 : String(20) default 'pending_retry'; // pending_retry, exhausted, poison
    poisonScore            : Integer default 0;
    
    // Indexing for DLQ operations
    @cds.persistence.index: true
    status                 : String(20);
    
    @cds.persistence.index: true
    nextRetryAt            : Timestamp;
    
    @cds.persistence.index: true
    agentId                : String(100);
}

entity RetryPolicies {
    key policyId           : String(36);
    messageType            : String(50);
    agentId                : String(100);   // null for global policy
    maxRetries             : Integer default 3;
    initialDelay           : Integer default 1000;  // milliseconds
    maxDelay               : Integer default 300000; // 5 minutes
    backoffMultiplier      : Decimal(3,2) default 2.0;
    active                 : Boolean default true;
    createdAt              : Timestamp;
    updatedAt              : Timestamp;
}

entity PoisonMessages {
    key poisonId           : String(36);
    originalDlqId          : String(36);
    messageId              : String(36);
    originalMessage        : LargeString;
    poisonReason           : String(500);
    markedAt               : Timestamp;
    agentId                : String(100);
    messageType            : String(50);
    totalRetries           : Integer;
    
    // Investigation fields
    investigated           : Boolean default false;
    investigatedBy         : String(100);
    investigatedAt         : Timestamp;
    resolution             : String(500);
    canReprocess           : Boolean default false;
}

entity FailureAnalytics {
    key date               : Date;
    key failureReason      : String(100);
    key agentId            : String(100);
    key messageType        : String(50);
    count                  : Integer default 1;
    timestamp              : Timestamp;
}

/**
 * Message Transformation Entities
 */
namespace a2a.transformation;

entity TransformationRules {
    key ruleId             : String(36);
    name                   : String(100);
    description            : String(500);
    sourceFormat           : String(20);
    targetFormat           : String(20);
    transformationLogic    : LargeString;   // JSON transformation logic
    validationSchema       : LargeString;   // JSON schema for validation
    active                 : Boolean default true;
    version                : String(10) default '1.0';
    createdBy              : String(100);
    createdAt              : Timestamp;
    updatedAt              : Timestamp;
    
    // Usage tracking
    usageCount             : Integer default 0;
    lastUsed               : Timestamp;
}

entity FormatSchemas {
    key schemaId           : String(36);
    name                   : String(100);
    description            : String(500);
    format                 : String(20);    // json, xml, a2a, etc.
    schema                 : LargeString;   // JSON Schema definition
    version                : String(10) default '1.0';
    active                 : Boolean default true;
    createdAt              : Timestamp;
    updatedAt              : Timestamp;
}

entity TransformationHistory {
    key transformationId   : String(36);
    messageId              : String(36);
    sourceFormat           : String(20);
    targetFormat           : String(20);
    transformationRule     : String(100);
    status                 : String(20);    // success, failed
    duration               : Integer;       // milliseconds
    error                  : String(500);
    metadata               : LargeString;   // JSON metadata
    createdAt              : Timestamp;
    
    // Performance indexing
    @cds.persistence.index: true
    createdAt              : Timestamp;
    
    @cds.persistence.index: true
    status                 : String(20);
}

entity TransformationMetrics {
    key date               : Date;
    key sourceFormat       : String(20);
    key targetFormat       : String(20);
    key status             : String(20);
    count                  : Integer default 1;
    totalDuration          : Integer default 0;
    avgDuration            : Integer default 0;
    minDuration            : Integer;
    maxDuration            : Integer;
    timestamp              : Timestamp;
}

/**
 * Service Definitions for Message Extensions
 */
service MessagePersistenceService {
    action persistMessage(
        messageId: String,
        content: String,
        metadata: String,
        priority: String,
        ttl: Integer
    ) returns {
        success: Boolean;
        messageId: String;
        location: String;
        ttl: Integer;
    };
    
    action retrieveMessage(
        messageId: String,
        includeMetadata: Boolean
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action searchMessages(
        query: String,
        filters: String,
        sortBy: String,
        sortOrder: String,
        limit: Integer,
        offset: Integer
    ) returns {
        success: Boolean;
        messages: array of String;
        pagination: String;
    };
    
    action archiveMessages(
        olderThanDays: Integer,
        batchSize: Integer
    ) returns {
        success: Boolean;
        archivedCount: Integer;
        cutoffDate: String;
    };
    
    action getStorageStats() returns {
        success: Boolean;
        stats: String;
    };
}

service DeadLetterQueueService {
    action addToDeadLetter(
        messageId: String,
        originalMessage: String,
        failureReason: String,
        failureDetails: String,
        retryCount: Integer,
        agentId: String,
        messageType: String,
        priority: String
    ) returns {
        success: Boolean;
        dlqId: String;
        nextRetryAt: String;
        status: String;
    };
    
    action retryMessage(
        dlqId: String,
        forceRetry: Boolean
    ) returns {
        success: Boolean;
        messageId: String;
        result: String;
    };
    
    action getFailedMessages(
        status: String,
        agentId: String,
        messageType: String,
        limit: Integer,
        offset: Integer
    ) returns {
        success: Boolean;
        messages: array of String;
        pagination: String;
    };
    
    action getDLQStats() returns {
        success: Boolean;
        stats: String;
    };
    
    entity FailedMessages as projection on a2a.dlq.FailedMessages;
    entity PoisonMessages as projection on a2a.dlq.PoisonMessages;
    entity FailureAnalytics as projection on a2a.dlq.FailureAnalytics;
}

service MessageTransformationService {
    action transformMessage(
        messageId: String,
        content: String,
        sourceFormat: String,
        targetFormat: String,
        transformationRule: String,
        validationSchema: String,
        enrichmentOptions: String,
        metadata: String
    ) returns {
        success: Boolean;
        transformationId: String;
        transformedContent: String;
        sourceFormat: String;
        targetFormat: String;
        duration: Integer;
        metadata: String;
    };
    
    action transformBatch(
        messages: array of String,
        batchOptions: String
    ) returns {
        success: Boolean;
        batchId: String;
        summary: String;
        results: array of String;
        errors: array of String;
    };
    
    action detectFormat(content: String) returns {
        success: Boolean;
        detectedFormat: String;
        confidence: Decimal(3,2);
    };
    
    action convertFormat(
        content: String,
        sourceFormat: String,
        targetFormat: String
    ) returns {
        success: Boolean;
        convertedContent: String;
        sourceFormat: String;
        targetFormat: String;
    };
    
    action getSupportedFormats() returns {
        success: Boolean;
        formats: array of String;
        transformers: array of String;
    };
    
    entity TransformationRules as projection on a2a.transformation.TransformationRules;
    entity TransformationHistory as projection on a2a.transformation.TransformationHistory;
    entity TransformationMetrics as projection on a2a.transformation.TransformationMetrics;
}

/**
 * Views for Analytics and Monitoring
 */
define view MessagePersistenceStats as select from a2a.messaging.Messages {
    priority,
    status,
    count(*) as messageCount,
    avg(ttl) as avgTTL
} group by priority, status;

define view DLQHealthView as select from a2a.dlq.FailedMessages {
    status,
    agentId,
    count(*) as messageCount,
    avg(retryCount) as avgRetryCount,
    max(addedAt) as lastFailure
} group by status, agentId;

define view TransformationPerformanceView as select from a2a.transformation.TransformationMetrics {
    sourceFormat,
    targetFormat,
    sum(count) as totalTransformations,
    avg(avgDuration) as avgDuration,
    (sum(count) - sum(case when status = 'failed' then count else 0 end)) / sum(count) * 100 as successRate : Decimal(5,2)
} group by sourceFormat, targetFormat;

/**
 * Annotations for UI and API documentation
 */
annotate MessagePersistenceService with @(
    title: 'A2A Message Persistence Service',
    description: 'Reliable message storage with multi-layer caching and archiving'
);

annotate DeadLetterQueueService with @(
    title: 'A2A Dead Letter Queue Service', 
    description: 'Advanced failed message handling with intelligent retry policies'
);

annotate MessageTransformationService with @(
    title: 'A2A Message Transformation Service',
    description: 'Comprehensive message format conversion and content transformation'
);
/**
 * @fileoverview SAP Change Tracking Aspects and Annotations
 * @since 1.0.0
 * @module change-tracking
 * 
 * Implements SAP-standard change tracking patterns for audit trails,
 * compliance, and data governance requirements
 */

namespace a2a.network.tracking;

using { a2a.network } from './schema';
using { cuid, managed, temporal } from '@sap/cds/common';

/**
 * Change tracking aspect for audit trail
 * @since 1.0.0
 */
aspect tracked {
    @cds.on.insert: $now
    @Common.Label: 'First Created At'
    @UI.HiddenFilter
    firstCreatedAt : Timestamp;
    
    @cds.on.insert: $user
    @Common.Label: 'First Created By'
    @UI.HiddenFilter
    firstCreatedBy : String(255);
    
    @Common.Label: 'Change Sequence Number'
    @Core.Computed
    changeSequence : Integer;
    
    @Common.Label: 'Previous Value Hash'
    @UI.Hidden
    previousHash   : String(64);
    
    @Common.Label: 'Change Reason'
    changeReason   : String(500);
    
    @Common.Label: 'Change Type'
    changeType     : String enum { 
        CREATE = 'CREATE';
        UPDATE = 'UPDATE'; 
        DELETE = 'DELETE';
        RESTORE = 'RESTORE';
        ARCHIVE = 'ARCHIVE';
    };
}

/**
 * Extended tracking with field-level changes
 * @since 1.0.0
 */
aspect fieldTracked : tracked {
    @Common.Label: 'Changed Fields'
    @UI.Hidden
    virtual changedFields : array of {
        fieldName    : String(100);
        oldValue     : String(1000);
        newValue     : String(1000);
        changeDate   : Timestamp;
        changedBy    : String(255);
    };
}

/**
 * Compliance tracking for regulatory requirements
 * @since 1.0.0
 */
aspect complianceTracked : tracked {
    @Common.Label: 'Data Classification'
    dataClassification : String enum {
        PUBLIC = 'PUBLIC';
        INTERNAL = 'INTERNAL';
        CONFIDENTIAL = 'CONFIDENTIAL';
        RESTRICTED = 'RESTRICTED';
    } default 'INTERNAL';
    
    @Common.Label: 'Retention Period (days)'
    retentionPeriod    : Integer default 2555; // 7 years
    
    @Common.Label: 'Legal Hold'
    legalHold          : Boolean default false;
    
    @Common.Label: 'Compliance Tags'
    complianceTags     : array of String;
    
    @Common.Label: 'Last Compliance Review'
    lastComplianceReview : Date;
    
    @Common.Label: 'Next Review Due'
    @Core.Computed
    nextReviewDue      : Date;
}

/**
 * Change history entity for audit trails
 * @since 1.0.0
 */
@cds.autoexpose
@Common.Label: 'Change History'
@Capabilities.DeleteRestrictions: { Deletable: false }
@Capabilities.UpdateRestrictions: { Updatable: false }
entity ChangeHistory : cuid, managed {
    @Common.Label: 'Entity Type'
    entityType     : String(100) @mandatory;
    
    @Common.Label: 'Entity ID'
    entityId       : String(36) @mandatory;
    
    @Common.Label: 'Change Type'
    changeType     : String enum { 
        CREATE; UPDATE; DELETE; RESTORE; ARCHIVE; MERGE; SPLIT;
    } @mandatory;
    
    @Common.Label: 'Changed Fields'
    changedFields  : array of {
        fieldName  : String(100);
        oldValue   : LargeString;
        newValue   : LargeString;
        valueType  : String(50);
    };
    
    @Common.Label: 'Change Reason'
    changeReason   : String(500);
    
    @Common.Label: 'Business Context'
    businessContext : {
        transactionId : String(36);
        workflowId    : String(36);
        correlationId : String(36);
        source        : String(100);
    };
    
    @Common.Label: 'Technical Context'
    technicalContext : {
        clientIP      : String(45);
        userAgent     : String(500);
        sessionId     : String(36);
        requestId     : String(36);
    };
    
    @Common.Label: 'Approval Info'
    approvalInfo   : {
        required      : Boolean;
        approvedBy    : String(255);
        approvedAt    : Timestamp;
        approvalId    : String(36);
    };
    
    @Common.Label: 'Snapshot Before'
    @Core.MediaType: 'application/json'
    snapshotBefore : LargeString;
    
    @Common.Label: 'Snapshot After'
    @Core.MediaType: 'application/json'
    snapshotAfter  : LargeString;
    
    @cds.persistence.index: [
        { elements: ['entityType', 'entityId', 'createdAt'] },
        { elements: ['createdBy', 'createdAt'] },
        { elements: ['changeType', 'createdAt'] }
    ]
}

/**
 * Data lineage tracking
 * @since 1.0.0
 */
@Common.Label: 'Data Lineage'
entity DataLineage : cuid {
    @Common.Label: 'Source Entity'
    sourceEntity   : String(100);
    
    @Common.Label: 'Source ID'
    sourceId       : String(36);
    
    @Common.Label: 'Target Entity'
    targetEntity   : String(100);
    
    @Common.Label: 'Target ID'
    targetId       : String(36);
    
    @Common.Label: 'Transformation'
    transformation : String(100);
    
    @Common.Label: 'Lineage Type'
    lineageType    : String enum {
        COPY; DERIVE; AGGREGATE; SPLIT; MERGE; TRANSFORM;
    };
    
    @Common.Label: 'Processing Time'
    processingTime : Timestamp;
    
    @Common.Label: 'Processing Duration (ms)'
    duration       : Integer;
}

/**
 * Extend existing entities with change tracking
 * @since 1.0.0
 */
extend entity network.Agents with tracked {
    @changelog: [
        { aspect: 'reputation', always: true },
        { aspect: 'isActive', always: true },
        { aspect: 'endpoint', always: true }
    ]
}

extend entity network.Services with fieldTracked {
    @changelog: [
        { aspect: 'pricePerCall', always: true },
        { aspect: 'isActive', always: true },
        { aspect: 'maxCallsPerDay', always: true }
    ]
}

extend entity network.Workflows with complianceTracked {
    @changelog: [
        { aspect: 'steps', always: true },
        { aspect: 'isPublished', always: true },
        { aspect: 'gas', always: true }
    ]
}

extend entity network.Messages with tracked {
    @changelog: [
        { aspect: 'status', always: true },
        { aspect: 'deliveryAttempts', always: true }
    ]
}

extend entity network.Transactions with fieldTracked {
    @changelog: [
        { aspect: 'status', always: true },
        { aspect: 'value', always: true },
        { aspect: 'gasUsed', always: true }
    ]
}

/**
 * Change tracking configuration
 * @since 1.0.0
 */
@Common.Label: 'Change Tracking Configuration'
@requires: 'Admin'
entity ChangeTrackingConfig : cuid {
    @Common.Label: 'Entity Type'
    entityType     : String(100) @mandatory;
    
    @Common.Label: 'Tracking Enabled'
    isEnabled      : Boolean default true;
    
    @Common.Label: 'Track All Fields'
    trackAllFields : Boolean default false;
    
    @Common.Label: 'Tracked Fields'
    trackedFields  : array of String(100);
    
    @Common.Label: 'Excluded Fields'
    excludedFields : array of String(100);
    
    @Common.Label: 'Retention Days'
    retentionDays  : Integer default 2555;
    
    @Common.Label: 'Require Reason'
    requireReason  : Boolean default false;
    
    @Common.Label: 'Require Approval'
    requireApproval: Boolean default false;
    
    @Common.Label: 'Notification Settings'
    notifications  : {
        enabled    : Boolean;
        recipients : array of String(255);
        events     : array of String(50);
    };
}

/**
 * Annotations for UI representation
 * @since 1.0.0
 */
annotate ChangeHistory with @(
    UI: {
        LineItem: [
            { Value: createdAt, Label: 'Change Date' },
            { Value: createdBy, Label: 'Changed By' },
            { Value: entityType, Label: 'Entity' },
            { Value: changeType, Label: 'Type' },
            { Value: changeReason, Label: 'Reason' }
        ],
        HeaderInfo: {
            TypeName: 'Change',
            TypeNamePlural: 'Changes',
            Title: { Value: entityType },
            Description: { Value: changeType }
        },
        Facets: [
            {
                $Type: 'UI.ReferenceFacet',
                Label: 'Change Details',
                Target: '@UI.FieldGroup#ChangeDetails'
            },
            {
                $Type: 'UI.ReferenceFacet',
                Label: 'Field Changes',
                Target: '@UI.FieldGroup#FieldChanges'
            },
            {
                $Type: 'UI.ReferenceFacet',
                Label: 'Context',
                Target: '@UI.FieldGroup#Context'
            }
        ],
        FieldGroup#ChangeDetails: {
            Data: [
                { Value: entityType },
                { Value: entityId },
                { Value: changeType },
                { Value: changeReason },
                { Value: createdAt },
                { Value: createdBy }
            ]
        },
        FieldGroup#FieldChanges: {
            Data: [
                { Value: changedFields }
            ]
        },
        FieldGroup#Context: {
            Data: [
                { Value: businessContext },
                { Value: technicalContext },
                { Value: approvalInfo }
            ]
        }
    }
);

/**
 * Service annotations for change tracking
 * @since 1.0.0
 */
annotate network.Agents with @(
    Common.SemanticKey: ['address'],
    Capabilities.ChangeTracking: {
        Supported: true,
        FilterableProperties: ['createdAt', 'createdBy', 'modifiedAt', 'modifiedBy'],
        ExpandableProperties: ['ChangeHistory']
    }
);

annotate network.Services with @(
    Common.SemanticKey: ['ID', 'provider.address'],
    Capabilities.ChangeTracking: {
        Supported: true,
        FilterableProperties: ['createdAt', 'createdBy', 'modifiedAt', 'modifiedBy'],
        ExpandableProperties: ['ChangeHistory']
    }
);
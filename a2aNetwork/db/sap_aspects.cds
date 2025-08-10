/**
 * @fileoverview SAP CDS Aspects for Reusable Patterns
 * @since 1.0.0
 * @module sap-aspects
 * 
 * Defines reusable CDS aspects following SAP best practices for
 * enterprise data modeling, extensibility, and governance
 */

namespace a2a.network.aspects;

using { cuid, managed, temporal } from '@sap/cds/common';
using { User } from '@sap/cds/common';

/**
 * Enhanced managed aspect with full audit trail
 * @since 1.0.0
 */
aspect sapManaged : managed {
    @Common.Label: 'Creation User Details'
    @Core.Computed
    virtual createdByDetails : {
        userId   : String(255);
        userName : String(255);
        email    : String(255);
        department : String(100);
    };
    
    @Common.Label: 'Modification User Details'
    @Core.Computed
    virtual modifiedByDetails : {
        userId   : String(255);
        userName : String(255);
        email    : String(255);
        department : String(100);
    };
}

/**
 * Extensibility aspect for custom fields and flexibility
 * @since 1.0.0
 */
aspect sapExtensible {
    @Common.Label: 'Custom Fields'
    @UI.Hidden
    customFields : {
        field1 : String(255);
        field2 : String(255);
        field3 : String(255);
        field4 : String(255);
        field5 : String(255);
    };
    
    @Common.Label: 'Extensions'
    @Core.MediaType: 'application/json'
    extensions   : LargeString;
    
    @Common.Label: 'Extension Schema Version'
    extensionSchemaVersion : String(10);
}

/**
 * Multi-tenancy aspect for SaaS scenarios
 * @since 1.0.0
 */
aspect sapMultiTenant {
    @cds.on.insert: $user.tenant
    @Common.Label: 'Tenant ID'
    @Core.Immutable
    tenant : String(36) @mandatory;
    
    @Common.Label: 'Tenant Name'
    @Core.Computed
    virtual tenantName : String(255);
    
    @Common.Label: 'Tenant Region'
    tenantRegion : String(10);
}

/**
 * Approval workflow aspect
 * @since 1.0.0
 */
aspect sapApprovable {
    @Common.Label: 'Approval Status'
    approvalStatus : String enum {
        DRAFT     = 'DRAFT';
        SUBMITTED = 'SUBMITTED'; 
        APPROVED  = 'APPROVED';
        REJECTED  = 'REJECTED';
        WITHDRAWN = 'WITHDRAWN';
    } default 'DRAFT';
    
    @Common.Label: 'Approved By'
    approvedBy     : String(255);
    
    @Common.Label: 'Approved At'
    approvedAt     : Timestamp;
    
    @Common.Label: 'Approval Comments'
    approvalComments : String(1000);
    
    @Common.Label: 'Approval Process ID'
    approvalProcessId : String(36);
    
    @Common.Label: 'Next Approvers'
    nextApprovers  : array of String(255);
}

/**
 * Versioning aspect for change management
 * @since 1.0.0
 */
aspect sapVersioned {
    @Common.Label: 'Version Number'
    @Core.Computed
    version        : String(20);
    
    @Common.Label: 'Major Version'
    majorVersion   : Integer default 1;
    
    @Common.Label: 'Minor Version'  
    minorVersion   : Integer default 0;
    
    @Common.Label: 'Patch Version'
    patchVersion   : Integer default 0;
    
    @Common.Label: 'Is Latest Version'
    @Core.Computed
    isLatest       : Boolean;
    
    @Common.Label: 'Parent Version'
    parentVersion  : String(36);
    
    @Common.Label: 'Version Notes'
    versionNotes   : String(2000);
}

/**
 * Lifecycle management aspect
 * @since 1.0.0
 */
aspect sapLifecycle {
    @Common.Label: 'Lifecycle Status'
    lifecycleStatus : String enum {
        DESIGN      = 'DESIGN';
        DEVELOPMENT = 'DEVELOPMENT';
        TESTING     = 'TESTING';
        PRODUCTION  = 'PRODUCTION';
        MAINTENANCE = 'MAINTENANCE';
        DEPRECATED  = 'DEPRECATED';
        RETIRED     = 'RETIRED';
    } default 'DESIGN';
    
    @Common.Label: 'Effective From'
    effectiveFrom  : Date;
    
    @Common.Label: 'Effective To'
    effectiveTo    : Date;
    
    @Common.Label: 'Retirement Date'
    retirementDate : Date;
    
    @Common.Label: 'Support Level'
    supportLevel   : String enum {
        FULL     = 'FULL';
        LIMITED  = 'LIMITED';
        SECURITY = 'SECURITY';
        NONE     = 'NONE';
    } default 'FULL';
}

/**
 * Performance monitoring aspect
 * @since 1.0.0
 */
aspect sapMonitored {
    @Common.Label: 'Performance Metrics'
    virtual performanceMetrics : {
        avgResponseTime    : Decimal(10,2);
        successRate        : Decimal(5,2);
        errorRate          : Decimal(5,2);
        lastHealthCheck    : Timestamp;
        healthStatus       : String enum { HEALTHY; DEGRADED; UNHEALTHY; };
    };
    
    @Common.Label: 'SLA Metrics'
    virtual slaMetrics : {
        uptimePercentage   : Decimal(5,2);
        availabilityTarget : Decimal(5,2);
        responseTimeTarget : Integer;
        lastSLABreach      : Timestamp;
    };
}

/**
 * Data quality aspect
 * @since 1.0.0
 */
aspect sapDataQuality {
    @Common.Label: 'Data Quality Score'
    @Analytics.Measure
    dataQualityScore : Decimal(5,2);
    
    @Common.Label: 'Completeness Score'
    completenessScore : Decimal(5,2);
    
    @Common.Label: 'Accuracy Score'
    accuracyScore    : Decimal(5,2);
    
    @Common.Label: 'Consistency Score'
    consistencyScore : Decimal(5,2);
    
    @Common.Label: 'Last Quality Check'
    lastQualityCheck : Timestamp;
    
    @Common.Label: 'Quality Issues'
    qualityIssues    : array of {
        type        : String(50);
        description : String(500);
        severity    : String enum { LOW; MEDIUM; HIGH; CRITICAL; };
        detectedAt  : Timestamp;
    };
}

/**
 * Localization aspect
 * @since 1.0.0
 */
aspect sapLocalizable {
    @Common.Label: 'Default Language'
    defaultLanguage : String(2) default 'EN';
    
    @Common.Label: 'Available Languages'
    availableLanguages : array of String(2);
    
    @Common.Label: 'Localized Content'
    @Core.MediaType: 'application/json'
    localizedContent : LargeString;
    
    @Common.Label: 'Translation Status'
    translationStatus : {
        totalItems     : Integer;
        translatedItems : Integer;
        pendingItems   : Integer;
        lastUpdate     : Timestamp;
    };
}

/**
 * Integration aspect for external systems
 * @since 1.0.0
 */
aspect sapIntegrable {
    @Common.Label: 'External System ID'
    externalSystemId : String(100);
    
    @Common.Label: 'External Reference'
    externalReference : String(255);
    
    @Common.Label: 'Last Sync Time'
    lastSyncTime     : Timestamp;
    
    @Common.Label: 'Sync Status'
    syncStatus       : String enum {
        SYNCED    = 'SYNCED';
        PENDING   = 'PENDING';
        FAILED    = 'FAILED';
        CONFLICT  = 'CONFLICT';
    };
    
    @Common.Label: 'Integration Metadata'
    @Core.MediaType: 'application/json'
    integrationMetadata : LargeString;
}

/**
 * Business object aspect for enterprise entities
 * @since 1.0.0
 */
aspect sapBusinessObject : cuid, sapManaged, sapExtensible, sapMultiTenant {
    @Common.Label: 'Business Object Type'
    @Core.Immutable
    businessObjectType : String(50) @mandatory;
    
    @Common.Label: 'Business Key'
    businessKey        : String(100);
    
    @Common.Label: 'Display Name'
    displayName        : localized String(255);
    
    @Common.Label: 'Description'
    description        : localized String(2000);
    
    @Common.Label: 'Status'
    status             : String enum {
        ACTIVE   = 'ACTIVE';
        INACTIVE = 'INACTIVE';
        BLOCKED  = 'BLOCKED';
        ARCHIVED = 'ARCHIVED';
    } default 'ACTIVE';
    
    @Common.Label: 'Tags'
    tags               : array of String(50);
    
    @Common.Label: 'Category'
    category           : String(100);
    
    @Common.Label: 'Priority'
    priority           : String enum {
        LOW    = 'LOW';
        MEDIUM = 'MEDIUM';
        HIGH   = 'HIGH';
        URGENT = 'URGENT';
    } default 'MEDIUM';
}

/**
 * Master data aspect for reference entities
 * @since 1.0.0
 */
aspect sapMasterData : sapBusinessObject, sapVersioned, sapLifecycle {
    @Common.Label: 'Code'
    @Core.Immutable
    code               : String(50) @mandatory;
    
    @Common.Label: 'Name'
    name               : localized String(255) @mandatory;
    
    @Common.Label: 'Short Description'
    shortDescription   : localized String(255);
    
    @Common.Label: 'Long Description'
    longDescription    : localized String(2000);
    
    @Common.Label: 'Valid From'
    validFrom          : Date @mandatory;
    
    @Common.Label: 'Valid To'
    validTo            : Date;
    
    @Common.Label: 'Sort Order'
    sortOrder          : Integer;
    
    @Common.Label: 'Parent'
    parent             : Association to one sapMasterData;
    
    @Common.Label: 'Children'
    children           : Composition of many sapMasterData on children.parent = $self;
}

/**
 * Transactional data aspect for operational entities
 * @since 1.0.0
 */
aspect sapTransactional : sapBusinessObject, sapApprovable, sapMonitored {
    @Common.Label: 'Transaction Date'
    transactionDate    : Date @mandatory;
    
    @Common.Label: 'Processing Status'
    processingStatus   : String enum {
        NEW        = 'NEW';
        PROCESSING = 'PROCESSING';
        PROCESSED  = 'PROCESSED';
        FAILED     = 'FAILED';
        CANCELLED  = 'CANCELLED';
    } default 'NEW';
    
    @Common.Label: 'Reference Number'
    referenceNumber    : String(100);
    
    @Common.Label: 'Amount'
    @Measures.Unit: currency
    amount             : Decimal(15,2);
    
    @Common.Label: 'Currency'
    currency           : String(3);
    
    @Common.Label: 'Quantity'
    @Measures.Unit: unit
    quantity           : Decimal(13,3);
    
    @Common.Label: 'Unit'
    unit               : String(3);
}

/**
 * Configuration aspect for system settings
 * @since 1.0.0
 */
aspect sapConfigurable {
    @Common.Label: 'Configuration Key'
    @Core.Immutable
    configKey          : String(255) @mandatory;
    
    @Common.Label: 'Configuration Value'
    configValue        : String(4000);
    
    @Common.Label: 'Value Type'
    valueType          : String enum {
        STRING  = 'STRING';
        NUMBER  = 'NUMBER';
        BOOLEAN = 'BOOLEAN';
        JSON    = 'JSON';
        XML     = 'XML';
    } default 'STRING';
    
    @Common.Label: 'Is Encrypted'
    isEncrypted        : Boolean default false;
    
    @Common.Label: 'Environment Scope'
    environmentScope   : String enum {
        ALL         = 'ALL';
        DEVELOPMENT = 'DEVELOPMENT';
        TEST        = 'TEST';
        PRODUCTION  = 'PRODUCTION';
    } default 'ALL';
    
    @Common.Label: 'Restart Required'
    restartRequired    : Boolean default false;
}

/**
 * UI annotations for common aspect usage
 */

// Business Object UI
annotate sapBusinessObject with @(
    UI.LineItem: [
        { Value: businessKey, Label: 'Business Key' },
        { Value: displayName, Label: 'Name' },
        { Value: status, Label: 'Status' },
        { Value: category, Label: 'Category' },
        { Value: priority, Label: 'Priority' }
    ],
    UI.HeaderInfo: {
        Title: { Value: displayName },
        Description: { Value: description }
    }
);

// Master Data UI
annotate sapMasterData with @(
    UI.LineItem: [
        { Value: code, Label: 'Code' },
        { Value: name, Label: 'Name' },
        { Value: validFrom, Label: 'Valid From' },
        { Value: validTo, Label: 'Valid To' },
        { Value: lifecycleStatus, Label: 'Status' }
    ]
);

// Transactional Data UI
annotate sapTransactional with @(
    UI.LineItem: [
        { Value: referenceNumber, Label: 'Reference' },
        { Value: transactionDate, Label: 'Date' },
        { Value: amount, Label: 'Amount' },
        { Value: processingStatus, Label: 'Status' },
        { Value: approvalStatus, Label: 'Approval' }
    ]
);
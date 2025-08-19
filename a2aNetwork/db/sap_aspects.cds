// SAP Standard Enterprise Aspects
// Common reusable aspects for enterprise data modeling

namespace a2a.network.aspects;

using from '@sap/cds/common';

/**
 * SAP Extensible Aspect
 * Provides extensibility fields for business objects
 */
aspect sapExtensible {
    // Extension fields for customer customization
    extensionField1 : String(255);
    extensionField2 : String(255);
    extensionField3 : String(255);
    extensionField4 : Integer;
    extensionField5 : Decimal(15,2);
    extensionField6 : Date;
    extensionField7 : DateTime;
    extensionField8 : Boolean;
    extensionField9 : String(1000);
    extensionField10 : LargeBinary;
}

/**
 * SAP Monitored Aspect
 * Provides monitoring and observability fields
 */
aspect sapMonitored {
    // Monitoring fields
    lastHealthCheck : DateTime;
    healthStatus : String(20) default 'UNKNOWN';
    performanceMetrics : String(5000); // JSON string
    alertsCount : Integer default 0;
    warningsCount : Integer default 0;
    errorsCount : Integer default 0;
    
    // Operational status
    operationalStatus : String(20) default 'ACTIVE';
    maintenanceMode : Boolean default false;
    lastMaintenanceDate : DateTime;
    nextMaintenanceDate : DateTime;
}

/**
 * SAP Data Quality Aspect
 * Provides data quality and validation fields
 */
aspect sapDataQuality {
    // Data quality indicators
    dataQualityScore : Decimal(5,2) default 0.00;
    validationStatus : String(20) default 'PENDING';
    validationErrors : String(5000); // JSON string
    validationWarnings : String(5000); // JSON string
    
    // Data lineage
    dataSource : String(100);
    dataSourceVersion : String(20);
    dataIngestionDate : DateTime;
    dataValidationDate : DateTime;
    
    // Compliance fields
    gdprCompliant : Boolean default false;
    dataRetentionDate : Date;
    dataClassification : String(20) default 'INTERNAL';
}

/**
 * SAP Audit Aspect
 * Provides comprehensive audit trail
 */
aspect sapAuditTrail {
    // Audit fields
    auditTrail : String(10000); // JSON string
    complianceStatus : String(20) default 'COMPLIANT';
    lastAuditDate : DateTime;
    nextAuditDate : DateTime;
    auditFindings : String(5000); // JSON string
    
    // Regulatory compliance
    regulatoryFramework : String(50);
    complianceVersion : String(20);
    certificationStatus : String(20);
    certificationExpiry : Date;
}

/**
 * SAP Security Aspect
 * Provides security-related fields
 */
aspect sapSecurity {
    // Security classification
    securityLevel : String(20) default 'STANDARD';
    accessControlList : String(5000); // JSON string
    encryptionStatus : String(20) default 'ENCRYPTED';
    
    // Access tracking
    lastAccessDate : DateTime;
    accessCount : Integer default 0;
    unauthorizedAccessAttempts : Integer default 0;
    
    // Security compliance
    securityScanDate : DateTime;
    vulnerabilityStatus : String(20) default 'CLEAN';
    securityFindings : String(5000); // JSON string
}

/**
 * SAP Versioned Aspect
 * Provides version control and change management
 */
aspect sapVersioned {
    // Version information
    version : String(20) default '1.0.0';
    majorVersion : Integer default 1;
    minorVersion : Integer default 0;
    patchVersion : Integer default 0;
    
    // Change tracking
    changeLog : String(5000); // JSON string
    changeReason : String(500);
    changeApprovedBy : String(256);
    changeApprovedDate : DateTime;
    
    // Version status
    versionStatus : String(20) default 'DRAFT';
    isCurrentVersion : Boolean default true;
    previousVersion : String(20);
    nextVersion : String(20);
}

/**
 * SAP Lifecycle Aspect
 * Provides lifecycle management capabilities
 */
aspect sapLifecycle {
    // Lifecycle status
    lifecycleStatus : String(20) default 'ACTIVE';
    lifecycleStage : String(20) default 'DEVELOPMENT';
    
    // Lifecycle dates
    plannedStartDate : Date;
    actualStartDate : Date;
    plannedEndDate : Date;
    actualEndDate : Date;
    
    // Lifecycle management
    lifecycleOwner : String(256);
    lifecycleApprover : String(256);
    lifecycleNotes : String(2000);
    
    // Status tracking
    isActive : Boolean default true;
    isDeprecated : Boolean default false;
    deprecationDate : Date;
    replacementReference : String(256);
}

/**
 * SAP Approvable Aspect
 * Provides approval workflow capabilities
 */
aspect sapApprovable {
    // Approval status
    approvalStatus : String(20) default 'PENDING';
    approvalLevel : Integer default 1;
    requiredApprovals : Integer default 1;
    receivedApprovals : Integer default 0;
    
    // Approval workflow
    approvalWorkflow : String(100);
    currentApprover : String(256);
    approvalDeadline : DateTime;
    
    // Approval history
    approvalHistory : String(5000); // JSON string
    lastApprovalDate : DateTime;
    lastApprover : String(256);
    approvalComments : String(2000);
    
    // Rejection handling
    rejectionReason : String(1000);
    rejectedBy : String(256);
    rejectedDate : DateTime;
    canResubmit : Boolean default true;
}

/**
 * SAP Master Data Aspect
 * Provides master data management capabilities
 */
aspect sapMasterData {
    // Master data classification
    masterDataType : String(50);
    masterDataCategory : String(50);
    businessKey : String(100);
    
    // Data governance
    dataOwner : String(256);
    dataSteward : String(256);
    dataGovernancePolicy : String(100);
    
    // Synchronization
    lastSyncDate : DateTime;
    syncStatus : String(20) default 'SYNCED';
    syncSource : String(100);
    syncTarget : String(100);
    
    // Master data quality
    masterDataScore : Decimal(5,2) default 100.00;
    qualityIssues : String(2000); // JSON string
    qualityCheckDate : DateTime;
}

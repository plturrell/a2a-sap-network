/**
 * @fileoverview Entity Extensions with SAP Aspects
 * @since 1.0.0
 * @module aspect-extensions
 * 
 * Extends existing entities with SAP enterprise aspects for enhanced
 * functionality, governance, and integration capabilities
 */

namespace a2a.network.extensions;

using { a2a.network } from './schema';
using { a2a.network.aspects } from './sap-aspects';

/**
 * Extend Agents with enterprise aspects
 * @since 1.0.0
 */
extend entity network.Agents with aspects.sapExtensible, aspects.sapMonitored, aspects.sapDataQuality {
    // Add business object metadata
    @Common.Label: 'Agent Type'
    agentType : String enum {
        INDIVIDUAL = 'INDIVIDUAL';
        CORPORATE  = 'CORPORATE';
        SYSTEM     = 'SYSTEM';
        SERVICE    = 'SERVICE';
    } default 'INDIVIDUAL';
    
    @Common.Label: 'Certification Level'
    certificationLevel : String enum {
        BASIC      = 'BASIC';
        ADVANCED   = 'ADVANCED';
        EXPERT     = 'EXPERT';
        CERTIFIED  = 'CERTIFIED';
    };
    
    @Common.Label: 'Compliance Status'
    complianceStatus : String enum {
        COMPLIANT     = 'COMPLIANT';
        NON_COMPLIANT = 'NON_COMPLIANT';
        UNDER_REVIEW  = 'UNDER_REVIEW';
        PENDING       = 'PENDING';
    };
    
    @Common.Label: 'Risk Rating'
    riskRating : String enum {
        LOW    = 'LOW';
        MEDIUM = 'MEDIUM';
        HIGH   = 'HIGH';
    } default 'MEDIUM';
}

/**
 * Extend Services with business aspects  
 * @since 1.0.0
 */
extend entity network.Services with aspects.sapVersioned, aspects.sapLifecycle, aspects.sapApprovable {
    @Common.Label: 'Service Level Agreement'
    sla : {
        uptimeGuarantee    : Decimal(5,2); // 99.95%
        responseTimeMax    : Integer;      // milliseconds
        supportHours       : String(50);   // "24x7" or "8x5"
        penaltyRate        : Decimal(5,2); // percentage
        escalationPath     : array of String(255);
    };
    
    @Common.Label: 'Quality Gates'
    qualityGates : {
        codeQuality        : Decimal(5,2);
        testCoverage       : Decimal(5,2);
        securityScan       : Boolean;
        performanceTest    : Boolean;
        documentationScore : Decimal(5,2);
    };
    
    @Common.Label: 'Integration Points'
    integrationPoints : array of {
        systemName : String(100);
        endpoint   : String(500);
        protocol   : String(20);
        authType   : String(20);
        status     : String enum { ACTIVE; INACTIVE; ERROR; };
    };
}

/**
 * Extend Workflows with versioning and approval
 * @since 1.0.0
 */
extend entity network.Workflows with aspects.sapVersioned, aspects.sapApprovable, aspects.sapLifecycle {
    @Common.Label: 'Workflow Category'
    workflowCategory : String enum {
        BUSINESS_PROCESS = 'BUSINESS_PROCESS';
        TECHNICAL        = 'TECHNICAL';
        INTEGRATION      = 'INTEGRATION';
        GOVERNANCE       = 'GOVERNANCE';
        COMPLIANCE       = 'COMPLIANCE';
    };
    
    @Common.Label: 'Complexity Level'
    complexityLevel : String enum {
        SIMPLE  = 'SIMPLE';
        MEDIUM  = 'MEDIUM';
        COMPLEX = 'COMPLEX';
    };
    
    @Common.Label: 'Business Impact'
    businessImpact : String enum {
        LOW      = 'LOW';
        MEDIUM   = 'MEDIUM'; 
        HIGH     = 'HIGH';
        CRITICAL = 'CRITICAL';
    };
    
    @Common.Label: 'Dependencies'
    dependencies : array of {
        workflowId   : String(36);
        dependencyType : String enum { PREREQUISITE; PARALLEL; SUCCESSOR; };
        isMandatory  : Boolean;
    };
}

/**
 * Extend Messages with integration aspects
 * @since 1.0.0
 */
extend entity network.Messages with aspects.sapIntegrable, aspects.sapMonitored {
    @Common.Label: 'Message Priority'
    messagePriority : String enum {
        LOW    = 'LOW';
        NORMAL = 'NORMAL';
        HIGH   = 'HIGH';
        URGENT = 'URGENT';
    } default 'NORMAL';
    
    @Common.Label: 'Delivery Options'
    deliveryOptions : {
        maxRetries     : Integer default 3;
        retryInterval  : Integer default 5000; // milliseconds
        timeToLive     : Integer default 300;  // seconds
        requireAck     : Boolean default true;
        compression    : Boolean default false;
        encryption     : Boolean default true;
    };
    
    @Common.Label: 'Routing Information'
    routingInfo : {
        routingKey     : String(255);
        exchange       : String(100);
        routingRules   : array of String(500);
        fallbackRoute  : String(255);
    };
}

/**
 * Extend NetworkConfig with configuration aspect
 * @since 1.0.0
 */
extend entity network.NetworkConfig with aspects.sapConfigurable, aspects.sapVersioned {
    @Common.Label: 'Configuration Category'
    configCategory : String enum {
        SYSTEM      = 'SYSTEM';
        BUSINESS    = 'BUSINESS';
        SECURITY    = 'SECURITY';
        INTEGRATION = 'INTEGRATION';
        UI          = 'UI';
    };
    
    @Common.Label: 'Validation Rules'
    validationRules : array of {
        ruleType    : String(50);
        expression  : String(1000);
        errorMessage : String(255);
    };
    
    @Common.Label: 'Dependencies'
    configDependencies : array of String(255);
}

/**
 * Create new master data entities using aspects
 * @since 1.0.0
 */

/**
 * Agent Categories master data
 * @since 1.0.0
 */
@Common.Label: 'Agent Categories'
entity AgentCategories : aspects.sapMasterData {
    @Common.Label: 'Icon'
    icon : String(100);
    
    @Common.Label: 'Color Code'
    colorCode : String(7); // hex color
    
    @Common.Label: 'Capabilities'
    allowedCapabilities : array of String(100);
    
    @Common.Label: 'Default Settings'
    defaultSettings : {
        maxReputation     : Integer default 1000;
        defaultPriority   : String(10) default 'MEDIUM';
        requireApproval   : Boolean default false;
    };
}

/**
 * Service Categories master data
 * @since 1.0.0
 */
@Common.Label: 'Service Categories'
entity ServiceCategories : aspects.sapMasterData {
    @Common.Label: 'Service Type'
    serviceType : String enum {
        COMPUTE    = 'COMPUTE';
        STORAGE    = 'STORAGE';
        NETWORK    = 'NETWORK';
        AI_ML      = 'AI_ML';
        ANALYTICS  = 'ANALYTICS';
        SECURITY   = 'SECURITY';
    };
    
    @Common.Label: 'Pricing Model'
    pricingModel : String enum {
        PAY_PER_USE   = 'PAY_PER_USE';
        SUBSCRIPTION  = 'SUBSCRIPTION';
        TIERED        = 'TIERED';
        AUCTION       = 'AUCTION';
    } default 'PAY_PER_USE';
    
    @Common.Label: 'SLA Template'
    slaTemplate : String(36); // reference to SLA template
}

/**
 * System Parameters for configuration
 * @since 1.0.0
 */
@Common.Label: 'System Parameters'
entity SystemParameters : aspects.sapConfigurable {
    @Common.Label: 'Parameter Group'
    parameterGroup : String(100);
    
    @Common.Label: 'Display Order'
    displayOrder : Integer;
    
    @Common.Label: 'User Configurable'
    userConfigurable : Boolean default false;
    
    @Common.Label: 'Admin Only'
    adminOnly : Boolean default true;
}

/**
 * Business Rules engine
 * @since 1.0.0
 */
@Common.Label: 'Business Rules'
entity BusinessRules : aspects.sapBusinessObject, aspects.sapVersioned {
    @Common.Label: 'Rule Type'
    ruleType : String enum {
        VALIDATION  = 'VALIDATION';
        CALCULATION = 'CALCULATION';
        ROUTING     = 'ROUTING';
        APPROVAL    = 'APPROVAL';
        NOTIFICATION = 'NOTIFICATION';
    };
    
    @Common.Label: 'Rule Expression'
    @Core.MediaType: 'application/json'
    ruleExpression : LargeString;
    
    @Common.Label: 'Execution Context'
    executionContext : String enum {
        BEFORE_SAVE = 'BEFORE_SAVE';
        AFTER_SAVE  = 'AFTER_SAVE';
        ON_REQUEST  = 'ON_REQUEST';
        SCHEDULED   = 'SCHEDULED';
    };
    
    @Common.Label: 'Applies To Entity'
    appliesTo : String(100);
    
    @Common.Label: 'Execution Order'
    executionOrder : Integer default 100;
}

/**
 * Audit Trail for comprehensive logging
 * @since 1.0.0
 */
@readonly
@Common.Label: 'Audit Trail'
entity AuditTrail : aspects.sapBusinessObject {
    @Common.Label: 'Entity Type'
    entityType : String(100) @mandatory;
    
    @Common.Label: 'Entity ID'
    entityId : String(36) @mandatory;
    
    @Common.Label: 'Action Type'
    actionType : String enum {
        CREATE = 'CREATE';
        READ   = 'READ';
        UPDATE = 'UPDATE';
        DELETE = 'DELETE';
        EXECUTE = 'EXECUTE';
        APPROVE = 'APPROVE';
        REJECT  = 'REJECT';
    } @mandatory;
    
    @Common.Label: 'Field Changes'
    fieldChanges : array of {
        fieldName : String(100);
        oldValue  : String(4000);
        newValue  : String(4000);
    };
    
    @Common.Label: 'Business Context'
    businessContext : {
        transactionId : String(36);
        workflowId    : String(36);
        correlationId : String(36);
        sessionId     : String(36);
    };
    
    @Common.Label: 'Technical Context'
    technicalContext : {
        ipAddress   : String(45);
        userAgent   : String(500);
        requestId   : String(36);
        apiVersion  : String(10);
    };
}

/**
 * Notifications with template support
 * @since 1.0.0
 */
@Common.Label: 'Notifications'
entity Notifications : aspects.sapTransactional {
    @Common.Label: 'Notification Type'
    notificationType : String enum {
        INFO    = 'INFO';
        WARNING = 'WARNING';
        ERROR   = 'ERROR';
        SUCCESS = 'SUCCESS';
    };
    
    @Common.Label: 'Recipient'
    recipient : String(255) @mandatory;
    
    @Common.Label: 'Subject'
    subject : String(255) @mandatory;
    
    @Common.Label: 'Message Body'
    messageBody : String(4000);
    
    @Common.Label: 'Template ID'
    templateId : String(36);
    
    @Common.Label: 'Template Variables'
    @Core.MediaType: 'application/json'
    templateVariables : LargeString;
    
    @Common.Label: 'Delivery Status'
    deliveryStatus : String enum {
        PENDING   = 'PENDING';
        SENT      = 'SENT';
        DELIVERED = 'DELIVERED';
        FAILED    = 'FAILED';
        READ      = 'READ';
    } default 'PENDING';
    
    @Common.Label: 'Delivery Attempts'
    deliveryAttempts : Integer default 0;
    
    @Common.Label: 'Next Retry'
    nextRetry : Timestamp;
}

/**
 * UI Annotations for new entities
 */

annotate AgentCategories with @(
    UI.LineItem: [
        { Value: code, Label: 'Code' },
        { Value: name, Label: 'Name' },
        { Value: serviceType, Label: 'Type' },
        { Value: lifecycleStatus, Label: 'Status' }
    ]
);

annotate BusinessRules with @(
    UI.LineItem: [
        { Value: displayName, Label: 'Rule Name' },
        { Value: ruleType, Label: 'Type' },
        { Value: appliesTo, Label: 'Applies To' },
        { Value: status, Label: 'Status' },
        { Value: version, Label: 'Version' }
    ]
);

annotate AuditTrail with @(
    UI.LineItem: [
        { Value: createdAt, Label: 'Timestamp' },
        { Value: createdBy, Label: 'User' },
        { Value: entityType, Label: 'Entity' },
        { Value: actionType, Label: 'Action' }
    ]
);
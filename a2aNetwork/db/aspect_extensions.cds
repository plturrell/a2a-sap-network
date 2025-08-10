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
using { a2a.network.aspects } from './sap_aspects';

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
        SERVICE    = 'SERVICE'
    } default 'INDIVIDUAL';
    
    @Common.Label: 'Certification Level'
    certificationLevel : String enum {
        BASIC      = 'BASIC';
        ADVANCED   = 'ADVANCED';
        EXPERT     = 'EXPERT';
        CERTIFIED  = 'CERTIFIED'
    };
    
    @Common.Label: 'Compliance Status'
    complianceStatus : String enum {
        COMPLIANT     = 'COMPLIANT';
        NON_COMPLIANT = 'NON_COMPLIANT';
        UNDER_REVIEW  = 'UNDER_REVIEW';
        PENDING       = 'PENDING'
    };
    
    @Common.Label: 'Risk Rating'
    riskRating : String enum {
        LOW    = 'LOW';
        MEDIUM = 'MEDIUM';
        HIGH   = 'HIGH'
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
        status     : String enum { ACTIVE = 'ACTIVE'; INACTIVE = 'INACTIVE'; ERROR = 'ERROR' };
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
        COMPLIANCE       = 'COMPLIANCE'
    };
    
    @Common.Label: 'Complexity Level'
    complexityLevel : String enum {
        SIMPLE  = 'SIMPLE';
        MEDIUM  = 'MEDIUM';
        COMPLEX = 'COMPLEX'
    };
    
    @Common.Label: 'Business Impact'
    businessImpact : String enum {
        LOW      = 'LOW';
        MEDIUM   = 'MEDIUM';
        HIGH     = 'HIGH';
        CRITICAL = 'CRITICAL'
    };
    
    @Common.Label: 'Dependencies'
    dependencies : array of {
        workflowId   : String(36);
        dependencyType : String enum { PREREQUISITE = 'PREREQUISITE'; PARALLEL = 'PARALLEL'; SUCCESSOR = 'SUCCESSOR' };
        isMandatory  : Boolean;
    };
}

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
        SECURITY   = 'SECURITY'
    };
    
    @Common.Label: 'Pricing Model'
    pricingModel : String enum {
        PAY_PER_USE   = 'PAY_PER_USE';
        SUBSCRIPTION  = 'SUBSCRIPTION';
        TIERED        = 'TIERED';
        AUCTION       = 'AUCTION'
    } default 'PAY_PER_USE';
    
    @Common.Label: 'SLA Template'
    slaTemplate : String(36); // reference to SLA template
}
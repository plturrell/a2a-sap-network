// Entity Extensions with SAP Aspects
// Extends existing entities with SAP enterprise aspects for enhanced
// functionality, governance, and integration capabilities

namespace a2a.network.extensions;



/**
 * Extend Agents with enterprise aspects
 * @since 1.0.0
 */
extend entity a2a.network.Agents with {
    @Common.Label: 'Agent Type'
    @cds.persistence.name: 'AGENTTYPE'
    agentType : String(20) default 'INDIVIDUAL';
    
    @Common.Label: 'Certification Level'
    certificationLevel : String(20);
    
    @Common.Label: 'Compliance Status'
    complianceStatus : String(20);
    
    @Common.Label: 'Risk Rating'
    riskRating : String(20) default 'MEDIUM';
}

/**
 * Extend Services with business aspects  
 * @since 1.0.0
 */
extend entity a2a.network.Services with {
    @Common.Label: 'Service Level Agreement'
    slaData : String(2000);
    
    @Common.Label: 'Quality Gates'
    qualityData : String(2000);
    
    @Common.Label: 'Integration Points'
    integrationData : String(2000);
}

/**
 * Extend Workflows with versioning and approval
 * @since 1.0.0
 */
extend entity a2a.network.Workflows with {
    @Common.Label: 'Workflow Category'
    workflowCategory : String(50);
    
    @Common.Label: 'Complexity Level'
    complexityLevel : String(20);
    
    @Common.Label: 'Business Impact'
    businessImpact : String(20);
    
    @Common.Label: 'Dependencies'
    dependenciesData : String(2000);
}

/**
 * Agent Categories master data
 * @since 1.0.0
 */
entity AgentCategories {
    key ID : UUID;
    
    @Common.Label: 'Category Name'
    name : String(100);
    
    @Common.Label: 'Description'
    description : String(500);
    
    @Common.Label: 'Icon'
    icon : String(100);
    
    @Common.Label: 'Color Code'
    colorCode : String(7);
    
    @Common.Label: 'Capabilities'
    capabilitiesData : String(2000);
    
    @Common.Label: 'Default Settings'
    defaultSettingsData : String(2000);
}

/**
 * Service Categories master data
 * @since 1.0.0
 */
entity ServiceCategories {
    key ID : UUID;
    
    @Common.Label: 'Category Name'
    name : String(100);
    
    @Common.Label: 'Description'
    description : String(500);
    
    @Common.Label: 'Service Type'
    serviceType : String(50);
    
    @Common.Label: 'Pricing Model'
    pricingModel : String(50) default 'PAY_PER_USE';
    
    @Common.Label: 'SLA Template'
    slaTemplate : String(36);
}
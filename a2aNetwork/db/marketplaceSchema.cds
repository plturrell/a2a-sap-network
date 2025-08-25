namespace a2a.marketplace;

using {
    cuid,
    managed,
    temporal,
    Currency,
    Country,
    Language,
    User,
    sap.common.CodeList
} from '@sap/cds/common';

using { a2a.network.aspects } from './sap_aspects';

// =======================================
// CUSTOM TYPES AND ENUMERATIONS
// =======================================

type UUID : String(36);
type BlockchainAddress : String(42);
type IPAddress : String(45);
type URL : String(2000);
type JSONString : LargeString;
type Hash : String(64);
type Rating : Decimal(3,2) @assert.range: [0.00, 5.00];
type Percentage : Decimal(5,2) @assert.range: [0.00, 100.00];

// Service-related enums
type ServiceStatus : String enum {
    DRAFT = 'DRAFT';
    ACTIVE = 'ACTIVE';
    PAUSED = 'PAUSED';
    DEPRECATED = 'DEPRECATED';
    SUSPENDED = 'SUSPENDED';
} @Common.Label: 'Service Status';

type ServiceType : String enum {
    ONE_TIME = 'ONE_TIME';
    SUBSCRIPTION = 'SUBSCRIPTION';
    ON_DEMAND = 'ON_DEMAND';
    ENTERPRISE = 'ENTERPRISE';
} @Common.Label: 'Service Type';

type PricingModel : String enum {
    FREE = 'FREE';
    FIXED = 'FIXED';
    USAGE_BASED = 'USAGE_BASED';
    TIERED = 'TIERED';
    DYNAMIC = 'DYNAMIC';
} @Common.Label: 'Pricing Model';

// Data product related enums
type DataFormat : String enum {
    CSV = 'CSV';
    JSON = 'JSON';
    XML = 'XML';
    PARQUET = 'PARQUET';
    AVRO = 'AVRO';
    API = 'API';
    STREAM = 'STREAM';
} @Common.Label: 'Data Format';

type DataUpdateFrequency : String enum {
    REAL_TIME = 'REAL_TIME';
    MINUTELY = 'MINUTELY';
    HOURLY = 'HOURLY';
    DAILY = 'DAILY';
    WEEKLY = 'WEEKLY';
    MONTHLY = 'MONTHLY';
    QUARTERLY = 'QUARTERLY';
    ANNUALLY = 'ANNUALLY';
    ON_DEMAND = 'ON_DEMAND';
} @Common.Label: 'Update Frequency';

// Request and integration enums
type RequestStatus : String enum {
    DRAFT = 'DRAFT';
    SUBMITTED = 'SUBMITTED';
    ACCEPTED = 'ACCEPTED';
    IN_PROGRESS = 'IN_PROGRESS';
    COMPLETED = 'COMPLETED';
    CANCELLED = 'CANCELLED';
    DISPUTED = 'DISPUTED';
    REFUNDED = 'REFUNDED';
} @Common.Label: 'Request Status';

type IntegrationType : String enum {
    DATA_PROCESSING = 'DATA_PROCESSING';
    ANALYTICS_ENHANCEMENT = 'ANALYTICS_ENHANCEMENT';
    AI_TRAINING = 'AI_TRAINING';
    REAL_TIME_FEED = 'REAL_TIME_FEED';
    BATCH_PROCESSING = 'BATCH_PROCESSING';
    ETL_PIPELINE = 'ETL_PIPELINE';
} @Common.Label: 'Integration Type';

type IntegrationStatus : String enum {
    CONFIGURING = 'CONFIGURING';
    ACTIVE = 'ACTIVE';
    PAUSED = 'PAUSED';
    ERROR = 'ERROR';
    TERMINATED = 'TERMINATED';
} @Common.Label: 'Integration Status';

// =======================================
// CODE LISTS AND MASTER DATA
// =======================================

@Common.Label: 'Service Categories'
@cds.odata.valuelist
entity ServiceCategories : CodeList {
    key code : String(20) @Common.Label: 'Category Code';
    @Common.Label: 'Parent Category'
    parent : Association to ServiceCategories;
    @Common.Label: 'Icon'
    icon : String(100);
    @Common.Label: 'Color'
    color : String(20);
    @Common.Label: 'Sort Order'
    sortOrder : Integer;
    @Common.Label: 'Is Active'
    isActive : Boolean default true;
    @Common.Label: 'Service Count'
    @Core.Computed : true
    serviceCount : Integer;
}

@Common.Label: 'Data Product Categories'
@cds.odata.valuelist
entity DataProductCategories : CodeList {
    key code : String(20) @Common.Label: 'Category Code';
    @Common.Label: 'Parent Category'
    parent : Association to DataProductCategories;
    @Common.Label: 'Icon'
    icon : String(100);
    @Common.Label: 'Color'  
    color : String(20);
    @Common.Label: 'Sort Order'
    sortOrder : Integer;
    @Common.Label: 'Is Active'
    isActive : Boolean default true;
    @Common.Label: 'Product Count'
    @Core.Computed : true
    productCount : Integer;
}

@Common.Label: 'Capability Definitions'
@cds.odata.valuelist
entity Capabilities : CodeList {
    key code : String(50) @Common.Label: 'Capability Code';
    @Common.Label: 'Category'
    category : String(50);
    @Common.Label: 'Icon'
    icon : String(100);
    @Common.Label: 'Complexity Level'
    complexityLevel : String enum {
        BASIC = 'BASIC';
        INTERMEDIATE = 'INTERMEDIATE';
        ADVANCED = 'ADVANCED';
        EXPERT = 'EXPERT';
    };
    @Common.Label: 'Is Technical'
    isTechnical : Boolean default true;
}

// =======================================
// CORE ENTITIES
// =======================================

@Common.Label: 'Services'
@cds.odata.valuelist
@odata.draft.enabled
entity Services : cuid, managed, aspects.Authorizations {
    @Common.Label: 'Service Name'
    @mandatory
    name : String(200) not null;

    @Common.Label: 'Description'
    @UI.MultiLineText
    description : String(2000);

    @Common.Label: 'Detailed Description'
    @UI.MultiLineText
    detailedDescription : LargeString;

    @Common.Label: 'Service Category'
    @mandatory
    category : Association to ServiceCategories;
    category_code : String(20) not null;

    @Common.Label: 'Provider Agent'
    @mandatory
    providerAgent : String(100) not null;

    @Common.Label: 'Provider Organization'
    providerOrganization : String(200);

    @Common.Label: 'Service Type'
    serviceType : ServiceType not null default 'ONE_TIME';

    @Common.Label: 'Status'
    status : ServiceStatus not null default 'DRAFT';

    @Common.Label: 'Pricing Model'
    pricing : PricingModel not null default 'FIXED';

    @Common.Label: 'Base Price'
    @Semantics.amount.currencyCode: 'currency_code'
    basePrice : Decimal(15,2) not null default 0.00;

    @Common.Label: 'Currency'
    currency : Currency not null default 'USD';
    currency_code : String(3) not null default 'USD';

    @Common.Label: 'Minimum Price'
    @Semantics.amount.currencyCode: 'currency_code'
    minPrice : Decimal(15,2);

    @Common.Label: 'Maximum Price'
    @Semantics.amount.currencyCode: 'currency_code'
    maxPrice : Decimal(15,2);

    @Common.Label: 'Price Per Unit'
    @Semantics.amount.currencyCode: 'currency_code'
    pricePerUnit : Decimal(15,6);

    @Common.Label: 'Unit of Measure'
    unitOfMeasure : String(50);

    // Service characteristics
    @Common.Label: 'Estimated Processing Time (minutes)'
    estimatedTimeMinutes : Integer;

    @Common.Label: 'Minimum Reputation Required'
    minReputationRequired : Decimal(4,2) default 0.00;

    @Common.Label: 'Maximum Concurrent Requests'
    maxConcurrentRequests : Integer default 10;

    @Common.Label: 'Current Active Requests'
    @Core.Computed : true
    currentActiveRequests : Integer default 0;

    @Common.Label: 'Service Endpoint'
    serviceEndpoint : URL;

    @Common.Label: 'API Documentation URL'
    apiDocumentationUrl : URL;

    @Common.Label: 'Terms of Service URL'
    termsOfServiceUrl : URL;

    @Common.Label: 'SLA Document URL'
    slaDocumentUrl : URL;

    // Technical specifications
    @Common.Label: 'Input Schema'
    @UI.MultiLineText
    inputSchema : JSONString;

    @Common.Label: 'Output Schema'
    @UI.MultiLineText
    outputSchema : JSONString;

    @Common.Label: 'Configuration Schema'
    @UI.MultiLineText
    configurationSchema : JSONString;

    // Media and presentation
    @Common.Label: 'Service Icon URL'
    iconUrl : URL;

    @Common.Label: 'Service Image URL'
    imageUrl : URL;

    @Common.Label: 'Demo Video URL'
    demoVideoUrl : URL;

    @Common.Label: 'Is Featured'
    isFeatured : Boolean default false;

    @Common.Label: 'Featured Until'
    featuredUntil : DateTime;

    @Common.Label: 'Tags'
    tags : String(1000);

    @Common.Label: 'Keywords'
    keywords : String(1000);

    // Quality and performance metrics
    @Common.Label: 'Quality Score'
    @Core.Computed : true
    qualityScore : Rating;

    @Common.Label: 'Reliability Score'
    @Core.Computed : true
    reliabilityScore : Percentage;

    @Common.Label: 'Performance Score'
    @Core.Computed : true
    performanceScore : Percentage;

    @Common.Label: 'Customer Satisfaction'
    @Core.Computed : true
    customerSatisfaction : Rating;

    // Blockchain integration
    @Common.Label: 'Blockchain Address'
    blockchainAddress : BlockchainAddress;

    @Common.Label: 'Smart Contract Hash'
    contractHash : Hash;

    // Compliance and certification
    @Common.Label: 'Certification Level'
    certificationLevel : String enum {
        BASIC = 'BASIC';
        STANDARD = 'STANDARD';
        PREMIUM = 'PREMIUM';
        ENTERPRISE = 'ENTERPRISE';
    };

    @Common.Label: 'Compliance Standards'
    complianceStandards : String(500);

    @Common.Label: 'Security Rating'
    securityRating : String enum {
        LOW = 'LOW';
        MEDIUM = 'MEDIUM';
        HIGH = 'HIGH';
        CRITICAL = 'CRITICAL';
    };

    // Availability and scheduling
    @Common.Label: 'Availability Schedule'
    @UI.MultiLineText
    availabilitySchedule : JSONString;

    @Common.Label: 'Time Zone'
    timeZone : String(50) default 'UTC';

    @Common.Label: 'Maintenance Windows'
    @UI.MultiLineText
    maintenanceWindows : JSONString;

    // Navigation properties
    @Common.Label: 'Service Capabilities'
    capabilities : Composition of many ServiceCapabilities on capabilities.service = $self;

    @Common.Label: 'Service Reviews'
    reviews : Composition of many ServiceReviews on reviews.service = $self;

    @Common.Label: 'Service Requests'
    serviceRequests : Composition of many ServiceRequests on serviceRequests.service = $self;

    @Common.Label: 'Pricing Tiers'
    pricingTiers : Composition of many ServicePricingTiers on pricingTiers.service = $self;

    @Common.Label: 'Service Dependencies'
    dependencies : Composition of many ServiceDependencies on dependencies.service = $self;

    @Common.Label: 'Usage Metrics'
    usageMetrics : Composition of many ServiceUsageMetrics on usageMetrics.service = $self;
}

@Common.Label: 'Data Products'
@cds.odata.valuelist
@odata.draft.enabled
entity DataProducts : cuid, managed, aspects.Authorizations {
    @Common.Label: 'Product Name'
    @mandatory
    name : String(200) not null;

    @Common.Label: 'Description'
    @UI.MultiLineText
    description : String(2000);

    @Common.Label: 'Detailed Description'
    @UI.MultiLineText
    detailedDescription : LargeString;

    @Common.Label: 'Data Category'
    @mandatory
    category : Association to DataProductCategories;
    category_code : String(20) not null;

    @Common.Label: 'Provider'
    @mandatory
    provider : String(200) not null;

    @Common.Label: 'Provider Organization'
    providerOrganization : String(200);

    @Common.Label: 'Data Format'
    format : DataFormat not null;

    @Common.Label: 'Alternative Formats'
    alternativeFormats : String(200);

    @Common.Label: 'Status'
    status : ServiceStatus not null default 'DRAFT';

    @Common.Label: 'Pricing Model'
    pricing : PricingModel not null default 'FIXED';

    @Common.Label: 'Base Price'
    @Semantics.amount.currencyCode: 'currency_code'
    price : Decimal(15,2) not null default 0.00;

    @Common.Label: 'Currency'
    currency : Currency not null default 'USD';
    currency_code : String(3) not null default 'USD';

    @Common.Label: 'Price Per GB'
    @Semantics.amount.currencyCode: 'currency_code'
    pricePerGB : Decimal(10,4);

    @Common.Label: 'Price Per Record'
    @Semantics.amount.currencyCode: 'currency_code'
    pricePerRecord : Decimal(10,6);

    // Data characteristics
    @Common.Label: 'Data Size (GB)'
    dataSizeGB : Decimal(15,3);

    @Common.Label: 'Record Count'
    recordCount : Integer64;

    @Common.Label: 'Column Count'
    columnCount : Integer;

    @Common.Label: 'Update Frequency'
    updateFrequency : DataUpdateFrequency not null;

    @Common.Label: 'Data Retention Period (days)'
    dataRetentionDays : Integer;

    @Common.Label: 'Historical Data Available (years)'
    historicalDataYears : Integer;

    // Quality and validation
    @Common.Label: 'Data Quality Score'
    @Core.Computed : true
    qualityScore : Rating;

    @Common.Label: 'Completeness Score'
    @Core.Computed : true
    completenessScore : Percentage;

    @Common.Label: 'Accuracy Score'
    @Core.Computed : true  
    accuracyScore : Percentage;

    @Common.Label: 'Freshness Score'
    @Core.Computed : true
    freshnessScore : Percentage;

    @Common.Label: 'Validation Rules'
    @UI.MultiLineText
    validationRules : JSONString;

    @Common.Label: 'Data Lineage'
    @UI.MultiLineText
    dataLineage : JSONString;

    // Technical specifications
    @Common.Label: 'Schema Definition'
    @UI.MultiLineText
    schemaDefinition : JSONString;

    @Common.Label: 'Sample Data'
    @UI.MultiLineText
    sampleData : LargeString;

    @Common.Label: 'Data Dictionary'
    @UI.MultiLineText
    dataDictionary : LargeString;

    @Common.Label: 'API Endpoint'
    apiEndpoint : URL;

    @Common.Label: 'Download URL'
    downloadUrl : URL;

    @Common.Label: 'Preview URL'
    previewUrl : URL;

    @Common.Label: 'Documentation URL'
    documentationUrl : URL;

    // Geographic and temporal coverage
    @Common.Label: 'Geographic Coverage'
    geographicCoverage : String(500);

    @Common.Label: 'Temporal Coverage Start'
    temporalCoverageStart : Date;

    @Common.Label: 'Temporal Coverage End'
    temporalCoverageEnd : Date;

    @Common.Label: 'Time Zone'
    timeZone : String(50) default 'UTC';

    // Usage and licensing
    @Common.Label: 'License Type'
    licenseType : String enum {
        OPEN_DATA = 'OPEN_DATA';
        COMMERCIAL = 'COMMERCIAL';
        RESTRICTED = 'RESTRICTED';
        CUSTOM = 'CUSTOM';
    } default 'COMMERCIAL';

    @Common.Label: 'License Terms'
    @UI.MultiLineText
    licenseTerms : LargeString;

    @Common.Label: 'Usage Restrictions'
    usageRestrictions : String(1000);

    @Common.Label: 'Attribution Required'
    attributionRequired : Boolean default false;

    @Common.Label: 'Commercial Use Allowed'
    commercialUseAllowed : Boolean default true;

    // Media and presentation
    @Common.Label: 'Product Icon URL'
    iconUrl : URL;

    @Common.Label: 'Product Image URL'
    imageUrl : URL;

    @Common.Label: 'Demo URL'
    demoUrl : URL;

    @Common.Label: 'Is Featured'
    isFeatured : Boolean default false;

    @Common.Label: 'Featured Until'
    featuredUntil : DateTime;

    @Common.Label: 'Tags'
    tags : String(1000);

    @Common.Label: 'Keywords'
    keywords : String(1000);

    // Compliance and security
    @Common.Label: 'Privacy Classification'
    privacyClassification : String enum {
        PUBLIC = 'PUBLIC';
        INTERNAL = 'INTERNAL';
        CONFIDENTIAL = 'CONFIDENTIAL';
        RESTRICTED = 'RESTRICTED';
    } default 'PUBLIC';

    @Common.Label: 'Security Classification'
    securityClassification : String enum {
        LOW = 'LOW';
        MEDIUM = 'MEDIUM';
        HIGH = 'HIGH';
        CRITICAL = 'CRITICAL';
    } default 'LOW';

    @Common.Label: 'Compliance Standards'
    complianceStandards : String(500);

    @Common.Label: 'Data Governance'
    @UI.MultiLineText
    dataGovernance : JSONString;

    // Navigation properties
    @Common.Label: 'Data Product Reviews'
    reviews : Composition of many DataProductReviews on reviews.dataProduct = $self;

    @Common.Label: 'Purchase History'
    purchases : Composition of many DataProductPurchases on purchases.dataProduct = $self;

    @Common.Label: 'Quality Metrics'
    qualityMetrics : Composition of many DataQualityMetrics on qualityMetrics.dataProduct = $self;

    @Common.Label: 'Usage Analytics'
    usageAnalytics : Composition of many DataUsageAnalytics on usageAnalytics.dataProduct = $self;
}

@Common.Label: 'Agent Listings'
@readonly
entity AgentListings : cuid, managed {
    @Common.Label: 'Agent Name'
    name : String(200) not null;

    @Common.Label: 'Agent ID'
    agentId : String(100) not null;

    @Common.Label: 'Owner'
    owner : String(200);

    @Common.Label: 'Organization'
    organization : String(200);

    @Common.Label: 'Description'
    @UI.MultiLineText
    description : String(2000);

    @Common.Label: 'Agent Category'
    category : Association to ServiceCategories;
    category_code : String(20);

    @Common.Label: 'Status'
    status : String enum {
        ONLINE = 'ONLINE';
        OFFLINE = 'OFFLINE';
        BUSY = 'BUSY';
        MAINTENANCE = 'MAINTENANCE';
    } default 'OFFLINE';

    @Common.Label: 'Endpoint URL'
    endpointUrl : URL;

    @Common.Label: 'Health Check URL'
    healthCheckUrl : URL;

    @Common.Label: 'Agent Version'
    version : String(20);

    @Common.Label: 'Last Heartbeat'
    lastHeartbeat : DateTime;

    @Common.Label: 'Reputation Score'
    reputationScore : Decimal(4,2);

    @Common.Label: 'Trust Score'
    trustScore : Decimal(4,2);

    @Common.Label: 'Performance Score'
    performanceScore : Percentage;

    @Common.Label: 'Uptime Percentage'
    uptimePercentage : Percentage;

    @Common.Label: 'Average Response Time (ms)'
    avgResponseTimeMs : Integer;

    @Common.Label: 'Total Services Provided'
    totalServicesProvided : Integer;

    @Common.Label: 'Active Services'
    activeServices : Integer;

    @Common.Label: 'Total Earnings'
    @Semantics.amount.currencyCode: 'currency_code'
    totalEarnings : Decimal(15,2);

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'Success Rate'
    successRate : Percentage;

    // Navigation properties
    @Common.Label: 'Agent Capabilities'
    capabilities : Composition of many AgentCapabilities on capabilities.agent = $self;

    @Common.Label: 'Provided Services'
    services : Association to many Services on services.providerAgent = agentId;
}

// =======================================
// TRANSACTIONAL ENTITIES
// =======================================

@Common.Label: 'Service Requests'
@odata.draft.enabled
entity ServiceRequests : cuid, managed, aspects.Authorizations {
    @Common.Label: 'Service'
    @mandatory
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Requester'
    requester : User not null;
    requester_ID : String(255) not null;

    @Common.Label: 'Provider Agent'
    providerAgent : String(100) not null;

    @Common.Label: 'Status'
    status : RequestStatus not null default 'DRAFT';

    @Common.Label: 'Priority'
    priority : String enum {
        LOW = 'LOW';
        NORMAL = 'NORMAL';
        HIGH = 'HIGH';
        URGENT = 'URGENT';
    } default 'NORMAL';

    @Common.Label: 'Request Type'
    requestType : String enum {
        IMMEDIATE = 'IMMEDIATE';
        SCHEDULED = 'SCHEDULED';
        RECURRING = 'RECURRING';
        BATCH = 'BATCH';
    } default 'IMMEDIATE';

    @Common.Label: 'Agreed Price'
    @Semantics.amount.currencyCode: 'currency_code'
    agreedPrice : Decimal(15,2) not null;

    @Common.Label: 'Escrow Amount'
    @Semantics.amount.currencyCode: 'currency_code'
    escrowAmount : Decimal(15,2);

    @Common.Label: 'Currency'
    currency_code : String(3) not null default 'USD';

    @Common.Label: 'Deadline'
    deadline : DateTime;

    @Common.Label: 'Scheduled Start'
    scheduledStart : DateTime;

    @Common.Label: 'Actual Start'
    actualStart : DateTime;

    @Common.Label: 'Estimated Completion'
    estimatedCompletion : DateTime;

    @Common.Label: 'Actual Completion'
    actualCompletion : DateTime;

    @Common.Label: 'Request Parameters'
    @UI.MultiLineText
    parameters : JSONString;

    @Common.Label: 'Configuration'
    @UI.MultiLineText
    configuration : JSONString;

    @Common.Label: 'Result Data'
    @UI.MultiLineText
    resultData : LargeString;

    @Common.Label: 'Result URL'
    resultUrl : URL;

    @Common.Label: 'Progress Percentage'
    @Core.Computed : true
    progressPercentage : Percentage;

    @Common.Label: 'Status Message'
    statusMessage : String(500);

    @Common.Label: 'Error Message'
    errorMessage : String(1000);

    @Common.Label: 'Cancellation Reason'
    cancellationReason : String(500);

    @Common.Label: 'Refund Amount'
    @Semantics.amount.currencyCode: 'currency_code'
    refundAmount : Decimal(15,2);

    @Common.Label: 'Transaction Hash'
    transactionHash : Hash;

    // Quality and feedback
    @Common.Label: 'Service Rating'
    serviceRating : Rating;

    @Common.Label: 'Quality Rating'
    qualityRating : Rating;

    @Common.Label: 'Timeliness Rating'
    timelinessRating : Rating;

    @Common.Label: 'Review Comments'
    @UI.MultiLineText
    reviewComments : String(2000);

    @Common.Label: 'Is Review Public'
    isReviewPublic : Boolean default true;

    // Compliance and audit
    @Common.Label: 'SLA Compliance'
    @Core.Computed : true
    slaCompliance : Boolean;

    @Common.Label: 'Audit Trail'
    @UI.MultiLineText
    auditTrail : JSONString;

    // Navigation properties
    @Common.Label: 'Request History'
    history : Composition of many ServiceRequestHistory on history.serviceRequest = $self;

    @Common.Label: 'Request Attachments'
    attachments : Composition of many ServiceRequestAttachments on attachments.serviceRequest = $self;
}

@Common.Label: 'Data Integrations'
@odata.draft.enabled
entity DataIntegrations : cuid, managed, aspects.Authorizations {
    @Common.Label: 'Integration Name'
    name : String(200);

    @Common.Label: 'Description'
    @UI.MultiLineText
    description : String(1000);

    @Common.Label: 'Agent'
    agent : Association to AgentListings;
    agent_ID : UUID not null;

    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID;

    @Common.Label: 'Data Product'
    dataProduct : Association to DataProducts;
    dataProduct_ID : UUID not null;

    @Common.Label: 'Integration Type'
    integrationType : IntegrationType not null;

    @Common.Label: 'Status'
    status : IntegrationStatus not null default 'CONFIGURING';

    @Common.Label: 'Configuration'
    @UI.MultiLineText
    configuration : JSONString;

    @Common.Label: 'Pipeline Configuration'
    @UI.MultiLineText
    pipelineConfiguration : JSONString;

    @Common.Label: 'Data Mapping'
    @UI.MultiLineText
    dataMapping : JSONString;

    @Common.Label: 'Transformation Rules'
    @UI.MultiLineText
    transformationRules : JSONString;

    @Common.Label: 'Frequency'
    frequency : DataUpdateFrequency not null;

    @Common.Label: 'Next Execution'
    nextExecution : DateTime;

    @Common.Label: 'Last Execution'
    lastExecution : DateTime;

    @Common.Label: 'Last Successful Execution'
    lastSuccessfulExecution : DateTime;

    @Common.Label: 'Execution Count'
    executionCount : Integer default 0;

    @Common.Label: 'Success Count'
    successCount : Integer default 0;

    @Common.Label: 'Error Count'
    errorCount : Integer default 0;

    @Common.Label: 'Total Records Processed'
    totalRecordsProcessed : Integer64 default 0;

    @Common.Label: 'Average Processing Time (ms)'
    avgProcessingTimeMs : Integer;

    @Common.Label: 'Total Cost'
    @Semantics.amount.currencyCode: 'currency_code'
    totalCost : Decimal(15,2) default 0.00;

    @Common.Label: 'Monthly Cost'
    @Semantics.amount.currencyCode: 'currency_code'
    monthlyCost : Decimal(15,2) default 0.00;

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'Health Score'
    @Core.Computed : true
    healthScore : Rating;

    @Common.Label: 'Data Quality Score'
    @Core.Computed : true
    dataQualityScore : Rating;

    @Common.Label: 'Performance Score'
    @Core.Computed : true
    performanceScore : Rating;

    @Common.Label: 'Last Error Message'
    lastErrorMessage : String(1000);

    @Common.Label: 'Error Details'
    @UI.MultiLineText
    errorDetails : JSONString;

    @Common.Label: 'Monitoring URL'
    monitoringUrl : URL;

    @Common.Label: 'Log URL'
    logUrl : URL;

    @Common.Label: 'Alert Configuration'
    @UI.MultiLineText
    alertConfiguration : JSONString;

    // Navigation properties
    @Common.Label: 'Integration History'
    history : Composition of many DataIntegrationHistory on history.integration = $self;

    @Common.Label: 'Integration Metrics'
    metrics : Composition of many DataIntegrationMetrics on metrics.integration = $self;
}

// =======================================
// SUPPORTING ENTITIES
// =======================================

@Common.Label: 'Service Capabilities'
entity ServiceCapabilities : cuid {
    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Capability'
    capability : Association to Capabilities;
    capability_code : String(50) not null;

    @Common.Label: 'Proficiency Level'
    proficiencyLevel : String enum {
        BASIC = 'BASIC';
        INTERMEDIATE = 'INTERMEDIATE';
        ADVANCED = 'ADVANCED';
        EXPERT = 'EXPERT';
    } default 'BASIC';

    @Common.Label: 'Is Primary'
    isPrimary : Boolean default false;

    @Common.Label: 'Sort Order'
    sortOrder : Integer default 0;
}

@Common.Label: 'Agent Capabilities'
entity AgentCapabilities : cuid {
    @Common.Label: 'Agent'
    agent : Association to AgentListings;
    agent_ID : UUID not null;

    @Common.Label: 'Capability'
    capability : Association to Capabilities;
    capability_code : String(50) not null;

    @Common.Label: 'Proficiency Level'
    proficiencyLevel : String enum {
        BASIC = 'BASIC';
        INTERMEDIATE = 'INTERMEDIATE';
        ADVANCED = 'ADVANCED';
        EXPERT = 'EXPERT';
    } default 'BASIC';

    @Common.Label: 'Is Primary'
    isPrimary : Boolean default false;

    @Common.Label: 'Years of Experience'
    yearsOfExperience : Decimal(4,1);

    @Common.Label: 'Certification'
    certification : String(200);

    @Common.Label: 'Last Validated'
    lastValidated : DateTime;
}

@Common.Label: 'Service Reviews'
entity ServiceReviews : cuid, managed {
    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Reviewer'
    reviewer : User not null;
    reviewer_ID : String(255) not null;

    @Common.Label: 'Service Request'
    serviceRequest : Association to ServiceRequests;
    serviceRequest_ID : UUID;

    @Common.Label: 'Overall Rating'
    overallRating : Rating not null;

    @Common.Label: 'Quality Rating'
    qualityRating : Rating;

    @Common.Label: 'Timeliness Rating'
    timelinessRating : Rating;

    @Common.Label: 'Communication Rating'
    communicationRating : Rating;

    @Common.Label: 'Value Rating'
    valueRating : Rating;

    @Common.Label: 'Review Title'
    title : String(200);

    @Common.Label: 'Review Text'
    @UI.MultiLineText
    reviewText : String(2000);

    @Common.Label: 'Pros'
    pros : String(1000);

    @Common.Label: 'Cons'
    cons : String(1000);

    @Common.Label: 'Recommendations'
    recommendations : String(1000);

    @Common.Label: 'Is Verified Purchase'
    isVerifiedPurchase : Boolean default false;

    @Common.Label: 'Is Public'
    isPublic : Boolean default true;

    @Common.Label: 'Helpful Votes'
    helpfulVotes : Integer default 0;

    @Common.Label: 'Total Votes'
    totalVotes : Integer default 0;

    @Common.Label: 'Provider Response'
    @UI.MultiLineText
    providerResponse : String(1000);

    @Common.Label: 'Provider Response Date'
    providerResponseDate : DateTime;
}

@Common.Label: 'Data Product Reviews'
entity DataProductReviews : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct : Association to DataProducts;
    dataProduct_ID : UUID not null;

    @Common.Label: 'Reviewer'
    reviewer : User not null;
    reviewer_ID : String(255) not null;

    @Common.Label: 'Purchase'
    purchase : Association to DataProductPurchases;
    purchase_ID : UUID;

    @Common.Label: 'Overall Rating'
    overallRating : Rating not null;

    @Common.Label: 'Quality Rating'
    qualityRating : Rating;

    @Common.Label: 'Accuracy Rating'
    accuracyRating : Rating;

    @Common.Label: 'Completeness Rating'
    completenessRating : Rating;

    @Common.Label: 'Freshness Rating'
    freshnessRating : Rating;

    @Common.Label: 'Usability Rating'
    usabilityRating : Rating;

    @Common.Label: 'Value Rating'
    valueRating : Rating;

    @Common.Label: 'Review Title'
    title : String(200);

    @Common.Label: 'Review Text'
    @UI.MultiLineText
    reviewText : String(2000);

    @Common.Label: 'Use Case Description'
    @UI.MultiLineText
    useCaseDescription : String(1000);

    @Common.Label: 'Is Verified Purchase'
    isVerifiedPurchase : Boolean default false;

    @Common.Label: 'Is Public'
    isPublic : Boolean default true;

    @Common.Label: 'Helpful Votes'
    helpfulVotes : Integer default 0;

    @Common.Label: 'Total Votes'
    totalVotes : Integer default 0;
}

// =======================================
// ANALYTICS AND METRICS
// =======================================

@Common.Label: 'Marketplace Statistics'
@readonly
entity MarketplaceStats : cuid, temporal {
    @Common.Label: 'Total Services'
    totalServices : Integer;

    @Common.Label: 'Active Services'
    activeServices : Integer;

    @Common.Label: 'Total Data Products'
    totalDataProducts : Integer;

    @Common.Label: 'Active Data Products'
    activeDataProducts : Integer;

    @Common.Label: 'Total Agents'
    totalAgents : Integer;

    @Common.Label: 'Active Agents'
    activeAgents : Integer;

    @Common.Label: 'Total Users'
    totalUsers : Integer;

    @Common.Label: 'Active Users'
    activeUsers : Integer;

    @Common.Label: 'Total Revenue'
    @Semantics.amount.currencyCode: 'currency_code'
    totalRevenue : Decimal(15,2);

    @Common.Label: 'Monthly Revenue'
    @Semantics.amount.currencyCode: 'currency_code'
    monthlyRevenue : Decimal(15,2);

    @Common.Label: 'Daily Revenue'
    @Semantics.amount.currencyCode: 'currency_code'
    dailyRevenue : Decimal(15,2);

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'Total Transactions'
    totalTransactions : Integer;

    @Common.Label: 'Successful Transactions'
    successfulTransactions : Integer;

    @Common.Label: 'Failed Transactions'
    failedTransactions : Integer;

    @Common.Label: 'Average Transaction Value'
    @Semantics.amount.currencyCode: 'currency_code'
    avgTransactionValue : Decimal(15,2);

    @Common.Label: 'Average Service Rating'
    avgServiceRating : Rating;

    @Common.Label: 'Average Data Product Rating'
    avgDataProductRating : Rating;

    @Common.Label: 'System Uptime'
    systemUptime : Percentage;

    @Common.Label: 'Average Response Time'
    avgResponseTime : Integer;

    @Common.Label: 'Error Rate'
    errorRate : Percentage;

    @Common.Label: 'User Growth Rate'
    userGrowthRate : Percentage;

    @Common.Label: 'Revenue Growth Rate'
    revenueGrowthRate : Percentage;

    @Common.Label: 'Service Growth Rate'
    serviceGrowthRate : Percentage;
}

// Supporting transactional entities for audit and history
@Common.Label: 'Service Request History'
entity ServiceRequestHistory : cuid, managed {
    @Common.Label: 'Service Request'
    serviceRequest : Association to ServiceRequests;
    serviceRequest_ID : UUID not null;

    @Common.Label: 'Status From'
    statusFrom : RequestStatus;

    @Common.Label: 'Status To'
    statusTo : RequestStatus not null;

    @Common.Label: 'Changed By'
    changedBy : User not null;
    changedBy_ID : String(255) not null;

    @Common.Label: 'Change Reason'
    changeReason : String(500);

    @Common.Label: 'Additional Data'
    @UI.MultiLineText
    additionalData : JSONString;
}

@Common.Label: 'Data Integration History'
entity DataIntegrationHistory : cuid, managed {
    @Common.Label: 'Integration'
    integration : Association to DataIntegrations;
    integration_ID : UUID not null;

    @Common.Label: 'Execution Status'
    executionStatus : String enum {
        STARTED = 'STARTED';
        SUCCESS = 'SUCCESS';
        FAILED = 'FAILED';
        PARTIAL = 'PARTIAL';
        CANCELLED = 'CANCELLED';
    } not null;

    @Common.Label: 'Records Processed'
    recordsProcessed : Integer;

    @Common.Label: 'Records Failed'
    recordsFailed : Integer;

    @Common.Label: 'Processing Time (ms)'
    processingTimeMs : Integer;

    @Common.Label: 'Data Quality Score'
    dataQualityScore : Rating;

    @Common.Label: 'Cost'
    @Semantics.amount.currencyCode: 'currency_code'
    cost : Decimal(10,4);

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'Error Message'
    errorMessage : String(1000);

    @Common.Label: 'Execution Details'
    @UI.MultiLineText
    executionDetails : JSONString;
}

// Additional supporting entities for comprehensive functionality
@Common.Label: 'Service Pricing Tiers'
entity ServicePricingTiers : cuid {
    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Tier Name'
    tierName : String(100) not null;

    @Common.Label: 'Tier Description'
    tierDescription : String(500);

    @Common.Label: 'Minimum Volume'
    minVolume : Integer;

    @Common.Label: 'Maximum Volume'
    maxVolume : Integer;

    @Common.Label: 'Price Per Unit'
    @Semantics.amount.currencyCode: 'currency_code'
    pricePerUnit : Decimal(15,6);

    @Common.Label: 'Fixed Fee'
    @Semantics.amount.currencyCode: 'currency_code'
    fixedFee : Decimal(15,2);

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'Sort Order'
    sortOrder : Integer default 0;

    @Common.Label: 'Is Active'
    isActive : Boolean default true;
}

@Common.Label: 'Data Product Purchases'
entity DataProductPurchases : cuid, managed {
    @Common.Label: 'Data Product'
    dataProduct : Association to DataProducts;
    dataProduct_ID : UUID not null;

    @Common.Label: 'Purchaser'
    purchaser : User not null;
    purchaser_ID : String(255) not null;

    @Common.Label: 'License Type'
    licenseType : String enum {
        SINGLE_USE = 'SINGLE_USE';
        SUBSCRIPTION = 'SUBSCRIPTION';
        ENTERPRISE = 'ENTERPRISE';
        ACADEMIC = 'ACADEMIC';
    } not null;

    @Common.Label: 'Purchase Price'
    @Semantics.amount.currencyCode: 'currency_code'
    purchasePrice : Decimal(15,2);

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';

    @Common.Label: 'License Key'
    licenseKey : String(100);

    @Common.Label: 'Download URL'
    downloadUrl : URL;

    @Common.Label: 'License Expires At'
    licenseExpiresAt : DateTime;

    @Common.Label: 'Download Count'
    downloadCount : Integer default 0;

    @Common.Label: 'Last Downloaded'
    lastDownloaded : DateTime;

    @Common.Label: 'Transaction Hash'
    transactionHash : Hash;

    @Common.Label: 'Is Active'
    isActive : Boolean default true;
}

@Common.Label: 'Service Dependencies'
entity ServiceDependencies : cuid {
    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Dependent Service'
    dependentService : Association to Services;
    dependentService_ID : UUID not null;

    @Common.Label: 'Dependency Type'
    dependencyType : String enum {
        REQUIRED = 'REQUIRED';
        OPTIONAL = 'OPTIONAL';
        RECOMMENDED = 'RECOMMENDED';
        ALTERNATIVE = 'ALTERNATIVE';
    } default 'REQUIRED';

    @Common.Label: 'Description'
    description : String(500);

    @Common.Label: 'Version Constraint'
    versionConstraint : String(50);

    @Common.Label: 'Is Active'
    isActive : Boolean default true;
}

// Metrics entities for analytics
@Common.Label: 'Service Usage Metrics'
entity ServiceUsageMetrics : cuid, temporal {
    @Common.Label: 'Service'
    service : Association to Services;
    service_ID : UUID not null;

    @Common.Label: 'Metric Date'
    metricDate : Date not null;

    @Common.Label: 'Request Count'
    requestCount : Integer default 0;

    @Common.Label: 'Success Count'
    successCount : Integer default 0;

    @Common.Label: 'Error Count'
    errorCount : Integer default 0;

    @Common.Label: 'Total Revenue'
    @Semantics.amount.currencyCode: 'currency_code'
    totalRevenue : Decimal(15,2) default 0.00;

    @Common.Label: 'Average Response Time'
    avgResponseTime : Integer;

    @Common.Label: 'Unique Users'
    uniqueUsers : Integer;

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';
}

@Common.Label: 'Data Usage Analytics'
entity DataUsageAnalytics : cuid, temporal {
    @Common.Label: 'Data Product'
    dataProduct : Association to DataProducts;
    dataProduct_ID : UUID not null;

    @Common.Label: 'Metric Date'
    metricDate : Date not null;

    @Common.Label: 'Download Count'
    downloadCount : Integer default 0;

    @Common.Label: 'API Requests'
    apiRequests : Integer default 0;

    @Common.Label: 'Data Volume (GB)'
    dataVolumeGB : Decimal(15,3) default 0.000;

    @Common.Label: 'Unique Users'
    uniqueUsers : Integer;

    @Common.Label: 'Total Revenue'
    @Semantics.amount.currencyCode: 'currency_code'
    totalRevenue : Decimal(15,2) default 0.00;

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';
}

@Common.Label: 'Data Quality Metrics'
entity DataQualityMetrics : cuid, temporal {
    @Common.Label: 'Data Product'
    dataProduct : Association to DataProducts;
    dataProduct_ID : UUID not null;

    @Common.Label: 'Metric Date'
    metricDate : Date not null;

    @Common.Label: 'Completeness Score'
    completenessScore : Rating;

    @Common.Label: 'Accuracy Score'
    accuracyScore : Rating;

    @Common.Label: 'Consistency Score'
    consistencyScore : Rating;

    @Common.Label: 'Freshness Score'
    freshnessScore : Rating;

    @Common.Label: 'Validity Score'
    validityScore : Rating;

    @Common.Label: 'Overall Quality Score'
    overallQualityScore : Rating;

    @Common.Label: 'Records Validated'
    recordsValidated : Integer;

    @Common.Label: 'Validation Errors'
    validationErrors : Integer;

    @Common.Label: 'Quality Details'
    @UI.MultiLineText
    qualityDetails : JSONString;
}

@Common.Label: 'Data Integration Metrics'
entity DataIntegrationMetrics : cuid, temporal {
    @Common.Label: 'Integration'
    integration : Association to DataIntegrations;
    integration_ID : UUID not null;

    @Common.Label: 'Metric Date'
    metricDate : Date not null;

    @Common.Label: 'Execution Count'
    executionCount : Integer default 0;

    @Common.Label: 'Success Count'
    successCount : Integer default 0;

    @Common.Label: 'Error Count'
    errorCount : Integer default 0;

    @Common.Label: 'Records Processed'
    recordsProcessed : Integer64 default 0;

    @Common.Label: 'Average Processing Time'
    avgProcessingTime : Integer;

    @Common.Label: 'Total Cost'
    @Semantics.amount.currencyCode: 'currency_code'
    totalCost : Decimal(15,2) default 0.00;

    @Common.Label: 'Data Quality Score'
    dataQualityScore : Rating;

    @Common.Label: 'Currency'
    currency_code : String(3) default 'USD';
}

@Common.Label: 'Service Request Attachments'
entity ServiceRequestAttachments : cuid, managed {
    @Common.Label: 'Service Request'
    serviceRequest : Association to ServiceRequests;
    serviceRequest_ID : UUID not null;

    @Common.Label: 'File Name'
    fileName : String(255) not null;

    @Common.Label: 'File Size'
    fileSize : Integer;

    @Common.Label: 'MIME Type'
    mimeType : String(100);

    @Common.Label: 'File URL'
    fileUrl : URL;

    @Common.Label: 'File Hash'
    fileHash : Hash;

    @Common.Label: 'Attachment Type'
    attachmentType : String enum {
        INPUT_DATA = 'INPUT_DATA';
        CONFIGURATION = 'CONFIGURATION';
        DOCUMENTATION = 'DOCUMENTATION';
        RESULT = 'RESULT';
        OTHER = 'OTHER';
    } default 'OTHER';

    @Common.Label: 'Description'
    description : String(500);

    @Common.Label: 'Is Public'
    isPublic : Boolean default false;
}
namespace sap.a2a.portal;

using { 
  cuid, 
  managed, 
  Currency, 
  Country, 
  sap.common.CodeList as CodeList 
} from '@sap/cds/common';

using { 
  sap.common.Currencies as Currencies,
  sap.common.Countries as Countries,
  sap.common.Languages as Languages
} from '@sap/cds/common';

// ========================================
// MASTER DATA ENTITIES
// ========================================

entity BusinessUnits : cuid, managed {
  code        : String(10) @title: 'Business Unit Code';
  name        : String(255) @title: 'Business Unit Name';
  description : String(1000) @title: 'Description';
  parentUnit  : Association to BusinessUnits @title: 'Parent Business Unit';
  costCenter  : String(20) @title: 'Cost Center';
  manager     : String(255) @title: 'Manager';
  country     : Country @title: 'Country';
  active      : Boolean default true @title: 'Active';
  
  // Relationships
  departments : Composition of many Departments on departments.businessUnit = $self;
  projects    : Composition of many Projects on projects.businessUnit = $self;
}

entity Departments : cuid, managed {
  code         : String(10) @title: 'Department Code';
  name         : String(255) @title: 'Department Name';
  description  : String(1000) @title: 'Description';
  businessUnit : Association to BusinessUnits @title: 'Business Unit';
  manager      : String(255) @title: 'Department Manager';
  costCenter   : String(20) @title: 'Cost Center';
  active       : Boolean default true @title: 'Active';
  
  // Relationships
  users    : Composition of many Users on users.department = $self;
  projects : Composition of many Projects on projects.department = $self;
}

// ========================================
// USER MANAGEMENT AND SECURITY
// ========================================

entity Users : cuid, managed {
  userId          : String(255) @title: 'User ID';
  email           : String(320) @title: 'Email Address' @assert.format: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$';
  firstName       : String(100) @title: 'First Name';
  lastName        : String(100) @title: 'Last Name';
  displayName     : String(255) @title: 'Display Name';
  department      : Association to Departments @title: 'Department';
  businessUnit    : Association to BusinessUnits @title: 'Business Unit';
  manager         : Association to Users @title: 'Manager';
  jobTitle        : String(255) @title: 'Job Title';
  phone           : String(50) @title: 'Phone Number';
  location        : String(255) @title: 'Location';
  timezone        : String(50) default 'UTC' @title: 'Timezone';
  language        : String(5) default 'en' @title: 'Language';
  lastLogin       : Timestamp @title: 'Last Login';
  active          : Boolean default true @title: 'Active';
  
  // Security attributes
  securityClearance    : String(20) @title: 'Security Clearance Level' @readonly;
  dataClassification   : String(20) @title: 'Data Classification Access' @readonly;
  s4hanaSystemAccess   : String(500) @title: 'SAP S/4HANA System Access' @readonly;
  
  // Relationships
  roleAssignments      : Composition of many UserRoleAssignments on roleAssignments.user = $self;
  projectMemberships   : Composition of many ProjectMembers on projectMemberships.user = $self;
  userSettings         : Composition of many UserSettings on userSettings.user = $self;
  auditLogs           : Composition of many AuditLogs on auditLogs.user = $self;
  notifications       : Composition of many Notifications on notifications.recipient = $self;
  sessionLogs         : Composition of many SessionLogs on sessionLogs.user = $self;
}

entity Roles : CodeList {
  key code : String(50) @title: 'Role Code' @readonly;
  category     : String(50) @title: 'Role Category';
  permissions  : String(2000) @title: 'Permissions (JSON)';
  description  : String(1000) @title: 'Role Description';
  active       : Boolean default true @title: 'Active';
  
  // Relationships
  userAssignments : Composition of many UserRoleAssignments on userAssignments.role = $self;
}

entity UserRoleAssignments : cuid, managed {
  user        : Association to Users @title: 'User';
  role        : Association to Roles @title: 'Role';
  assignedBy  : String(255) @title: 'Assigned By';
  validFrom   : Date @title: 'Valid From';
  validTo     : Date @title: 'Valid To';
  active      : Boolean default true @title: 'Active';
}

// ========================================
// PROJECT MANAGEMENT
// ========================================

entity Projects : cuid, managed {
  projectId       : String(50) @title: 'Project ID';
  name            : String(255) @title: 'Project Name';
  description     : String(2000) @title: 'Description';
  businessUnit    : Association to BusinessUnits @title: 'Business Unit';
  department      : Association to Departments @title: 'Department';
  projectManager  : Association to Users @title: 'Project Manager';
  status          : String(20) default 'DRAFT' @title: 'Status'; // DRAFT, ACTIVE, TESTING, DEPLOYED, ARCHIVED
  priority        : String(10) default 'MEDIUM' @title: 'Priority'; // LOW, MEDIUM, HIGH, CRITICAL
  startDate       : Date @title: 'Start Date';
  endDate         : Date @title: 'End Date';
  budget          : Decimal(15,2) @title: 'Budget';
  currency        : Currency @title: 'Currency';
  costCenter      : String(20) @title: 'Cost Center';
  
  // Technical metadata
  repositoryUrl   : String(500) @title: 'Repository URL';
  version         : String(20) @title: 'Version';
  lastBuild       : Timestamp @title: 'Last Build';
  buildStatus     : String(20) @title: 'Build Status';
  testCoverage    : Decimal(5,2) @title: 'Test Coverage %';
  
  // Compliance and governance
  dataClassification : String(20) @title: 'Data Classification';
  complianceStatus   : String(20) @title: 'Compliance Status';
  securityReview     : Boolean default false @title: 'Security Review Completed';
  
  // Relationships
  members         : Composition of many ProjectMembers on members.project = $self;
  agents          : Composition of many ProjectAgents on agents.project = $self;
  workflows       : Composition of many ProjectWorkflows on workflows.project = $self;
  deployments     : Composition of many Deployments on deployments.project = $self;
  documents       : Composition of many ProjectDocuments on documents.project = $self;
  testResults     : Composition of many TestExecutions on testResults.project = $self;
  auditTrail      : Composition of many ProjectAuditTrail on auditTrail.project = $self;
}

entity ProjectMembers : cuid, managed {
  project     : Association to Projects @title: 'Project';
  user        : Association to Users @title: 'User';
  role        : String(50) @title: 'Project Role'; // OWNER, DEVELOPER, TESTER, VIEWER
  permissions : String(1000) @title: 'Permissions (JSON)';
  joinedDate  : Date @title: 'Joined Date';
  active      : Boolean default true @title: 'Active';
}

// ========================================
// AGENT MANAGEMENT
// ========================================

entity AgentTemplates : cuid, managed {
  templateId      : String(50) @title: 'Template ID';
  name            : String(255) @title: 'Template Name';
  description     : String(2000) @title: 'Description';
  category        : String(50) @title: 'Category';
  version         : String(20) @title: 'Version';
  author          : String(255) @title: 'Author';
  
  // Template configuration
  configurationSchema : LargeString @title: 'Configuration Schema (JSON)';
  defaultConfiguration : LargeString @title: 'Default Configuration (JSON)';
  skillsRequired      : String(1000) @title: 'Required Skills (JSON Array)';
  integrationTypes    : String(1000) @title: 'Supported Integration Types';
  
  // Metadata
  sapCertified     : Boolean default false @title: 'SAP Certified';
  enterpriseReady  : Boolean default false @title: 'Enterprise Ready';
  supportLevel     : String(20) @title: 'Support Level';
  licenseType      : String(50) @title: 'License Type';
  
  // Relationships
  projectAgents : Composition of many ProjectAgents on projectAgents.template = $self;
}

entity ProjectAgents : cuid, managed {
  project         : Association to Projects @title: 'Project';
  template        : Association to AgentTemplates @title: 'Template';
  agentId         : String(50) @title: 'Agent ID';
  name            : String(255) @title: 'Agent Name';
  description     : String(1000) @title: 'Description';
  
  // Configuration
  configuration   : LargeString @title: 'Configuration (JSON)';
  environmentVars : LargeString @title: 'Environment Variables (JSON)';
  customSkills    : LargeString @title: 'Custom Skills (JSON)';
  
  // Status
  status          : String(20) default 'DRAFT' @title: 'Status'; // DRAFT, CONFIGURED, TESTING, DEPLOYED, FAILED
  lastDeployed    : Timestamp @title: 'Last Deployed';
  deploymentStatus : String(20) @title: 'Deployment Status';
  healthStatus    : String(20) @title: 'Health Status';
  
  // Performance metrics
  executionCount  : Integer default 0 @title: 'Execution Count';
  avgResponseTime : Decimal(10,3) @title: 'Average Response Time (ms)';
  successRate     : Decimal(5,2) @title: 'Success Rate %';
  lastExecution   : Timestamp @title: 'Last Execution';
  
  // Relationships
  executions     : Composition of many AgentExecutions on executions.agent = $self;
  integrations   : Composition of many AgentIntegrations on integrations.agent = $self;
}

entity AgentExecutions : cuid, managed {
  agent           : Association to ProjectAgents @title: 'Agent';
  executionId     : String(100) @title: 'Execution ID';
  startTime       : Timestamp @title: 'Start Time';
  endTime         : Timestamp @title: 'End Time';
  duration        : Integer @title: 'Duration (ms)';
  status          : String(20) @title: 'Status'; // RUNNING, SUCCESS, FAILED, TIMEOUT
  
  // Request/Response
  inputData       : LargeString @title: 'Input Data (JSON)';
  outputData      : LargeString @title: 'Output Data (JSON)';
  errorMessage    : String(2000) @title: 'Error Message';
  stackTrace      : LargeString @title: 'Stack Trace';
  
  // Context
  triggeredBy     : String(255) @title: 'Triggered By';
  correlationId   : String(100) @title: 'Correlation ID';
  sessionId       : String(100) @title: 'Session ID';
  
  // Performance metrics
  cpuUsage        : Decimal(5,2) @title: 'CPU Usage %';
  memoryUsage     : Integer @title: 'Memory Usage (MB)';
  networkCalls    : Integer @title: 'Network Calls';
}

// ========================================
// WORKFLOW MANAGEMENT
// ========================================

entity WorkflowTemplates : cuid, managed {
  templateId      : String(50) @title: 'Template ID';
  name            : String(255) @title: 'Template Name';
  description     : String(2000) @title: 'Description';
  category        : String(50) @title: 'Category';
  version         : String(20) @title: 'Version';
  
  // BPMN Definition
  bpmnDefinition  : LargeString @title: 'BPMN Definition (XML)';
  processKey      : String(100) @title: 'Process Key' @readonly;
  
  // Configuration
  configurationSchema : LargeString @title: 'Configuration Schema (JSON)';
  defaultConfiguration : LargeString @title: 'Default Configuration (JSON)';
  
  // Metadata
  sapCertified    : Boolean default false @title: 'SAP Certified';
  businessProcess : String(255) @title: 'Business Process';
  industryVertical : String(100) @title: 'Industry Vertical';
  
  // Relationships
  projectWorkflows : Composition of many ProjectWorkflows on projectWorkflows.template = $self;
}

entity ProjectWorkflows : cuid, managed {
  project         : Association to Projects @title: 'Project';
  template        : Association to WorkflowTemplates @title: 'Template';
  workflowId      : String(50) @title: 'Workflow ID';
  name            : String(255) @title: 'Workflow Name';
  description     : String(1000) @title: 'Description';
  
  // Definition
  bpmnDefinition  : LargeString @title: 'BPMN Definition (XML)';
  configuration   : LargeString @title: 'Configuration (JSON)';
  
  // Status
  status          : String(20) default 'DRAFT' @title: 'Status';
  version         : String(20) @title: 'Version';
  lastDeployed    : Timestamp @title: 'Last Deployed';
  
  // Relationships
  executions      : Composition of many WorkflowExecutions on executions.workflow = $self;
  integrations    : Composition of many WorkflowIntegrations on integrations.workflow = $self;
}

entity WorkflowExecutions : cuid, managed {
  workflow        : Association to ProjectWorkflows @title: 'Workflow';
  executionId     : String(100) @title: 'Execution ID';
  processInstanceId : String(100) @title: 'Process Instance ID';
  
  // Timing
  startTime       : Timestamp @title: 'Start Time';
  endTime         : Timestamp @title: 'End Time';
  duration        : Integer @title: 'Duration (ms)';
  
  // Status
  status          : String(20) @title: 'Status'; // RUNNING, COMPLETED, FAILED, SUSPENDED
  currentActivity : String(100) @title: 'Current Activity';
  
  // Data
  inputVariables  : LargeString @title: 'Input Variables (JSON)';
  outputVariables : LargeString @title: 'Output Variables (JSON)';
  
  // Context
  initiatedBy     : String(255) @title: 'Initiated By';
  correlationId   : String(100) @title: 'Correlation ID';
  businessKey     : String(100) @title: 'Business Key' @readonly;
  
  // Relationships
  taskExecutions  : Composition of many TaskExecutions on taskExecutions.workflowExecution = $self;
}

entity TaskExecutions : cuid, managed {
  workflowExecution : Association to WorkflowExecutions @title: 'Workflow Execution';
  taskId          : String(100) @title: 'Task ID';
  taskName        : String(255) @title: 'Task Name';
  taskType        : String(50) @title: 'Task Type'; // USER_TASK, SERVICE_TASK, SCRIPT_TASK, etc.
  
  // Timing
  startTime       : Timestamp @title: 'Start Time';
  endTime         : Timestamp @title: 'End Time';
  duration        : Integer @title: 'Duration (ms)';
  
  // Status
  status          : String(20) @title: 'Status';
  assignee        : String(255) @title: 'Assignee';
  
  // Data
  inputData       : LargeString @title: 'Input Data (JSON)';
  outputData      : LargeString @title: 'Output Data (JSON)';
  errorMessage    : String(2000) @title: 'Error Message';
}

// ========================================
// INTEGRATION MANAGEMENT
// ========================================

entity SystemConnections : cuid, managed {
  connectionId    : String(50) @title: 'Connection ID';
  name            : String(255) @title: 'Connection Name';
  systemType      : String(50) @title: 'System Type'; // S4HANA, SAC, ARIBA, etc.
  description     : String(1000) @title: 'Description';
  
  // Connection details
  baseUrl         : String(500) @title: 'Base URL';
  authType        : String(50) @title: 'Authentication Type' @readonly;
  connectionParams : LargeString @title: 'Connection Parameters (Encrypted JSON)' @readonly;
  
  // Status
  status          : String(20) default 'INACTIVE' @title: 'Status';
  lastTested      : Timestamp @title: 'Last Tested';
  testResult      : String(20) @title: 'Test Result';
  
  // Metadata
  businessUnit    : Association to BusinessUnits @title: 'Business Unit';
  owner           : Association to Users @title: 'Owner';
  
  // Relationships
  agentIntegrations     : Composition of many AgentIntegrations on agentIntegrations.systemConnection = $self;
  workflowIntegrations  : Composition of many WorkflowIntegrations on workflowIntegrations.systemConnection = $self;
}

entity AgentIntegrations : cuid, managed {
  agent              : Association to ProjectAgents @title: 'Agent';
  systemConnection   : Association to SystemConnections @title: 'System Connection';
  integrationId      : String(50) @title: 'Integration ID';
  name               : String(255) @title: 'Integration Name';
  description        : String(1000) @title: 'Description';
  
  // Configuration
  configuration      : LargeString @title: 'Configuration (JSON)';
  mappingRules       : LargeString @title: 'Mapping Rules (JSON)';
  
  // Status
  status             : String(20) default 'INACTIVE' @title: 'Status';
  lastSynced         : Timestamp @title: 'Last Synced';
  syncStatus         : String(20) @title: 'Sync Status';
  
  // Performance
  executionCount     : Integer default 0 @title: 'Execution Count';
  avgResponseTime    : Decimal(10,3) @title: 'Average Response Time (ms)';
  errorRate          : Decimal(5,2) @title: 'Error Rate %';
}

entity WorkflowIntegrations : cuid, managed {
  workflow           : Association to ProjectWorkflows @title: 'Workflow';
  systemConnection   : Association to SystemConnections @title: 'System Connection';
  integrationId      : String(50) @title: 'Integration ID';
  name               : String(255) @title: 'Integration Name';
  description        : String(1000) @title: 'Description';
  
  // Configuration
  configuration      : LargeString @title: 'Configuration (JSON)';
  mappingRules       : LargeString @title: 'Mapping Rules (JSON)';
  
  // Status
  status             : String(20) default 'INACTIVE' @title: 'Status';
  lastSynced         : Timestamp @title: 'Last Synced';
  syncStatus         : String(20) @title: 'Sync Status';
}

// ========================================
// DEPLOYMENT AND TESTING
// ========================================

entity Deployments : cuid, managed {
  project          : Association to Projects @title: 'Project';
  deploymentId     : String(100) @title: 'Deployment ID';
  name             : String(255) @title: 'Deployment Name';
  description      : String(1000) @title: 'Description';
  
  // Configuration
  environment      : String(20) @title: 'Environment'; // DEV, TEST, PROD
  version          : String(20) @title: 'Version';
  configuration    : LargeString @title: 'Configuration (JSON)';
  
  // Timing
  startTime        : Timestamp @title: 'Start Time';
  endTime          : Timestamp @title: 'End Time';
  duration         : Integer @title: 'Duration (seconds)';
  
  // Status
  status           : String(20) @title: 'Status'; // PENDING, RUNNING, SUCCESS, FAILED, ROLLED_BACK
  deployedBy       : Association to Users @title: 'Deployed By';
  
  // Resources
  cpuLimit         : String(20) @title: 'CPU Limit';
  memoryLimit      : String(20) @title: 'Memory Limit';
  replicas         : Integer @title: 'Replicas';
  
  // Results
  deploymentLogs   : LargeString @title: 'Deployment Logs';
  errorMessage     : String(2000) @title: 'Error Message';
  
  // Relationships
  testResults      : Composition of many TestExecutions on testResults.deployment = $self;
}

entity TestExecutions : cuid, managed {
  project          : Association to Projects @title: 'Project';
  deployment       : Association to Deployments @title: 'Deployment';
  testExecutionId  : String(100) @title: 'Test Execution ID';
  testSuite        : String(255) @title: 'Test Suite';
  testFramework    : String(50) @title: 'Test Framework'; // PYTEST, JEST, GO_TEST
  
  // Timing
  startTime        : Timestamp @title: 'Start Time';
  endTime          : Timestamp @title: 'End Time';
  duration         : Integer @title: 'Duration (ms)';
  
  // Results
  status           : String(20) @title: 'Status';
  testsTotal       : Integer @title: 'Total Tests';
  testsPassed      : Integer @title: 'Tests Passed';
  testsFailed      : Integer @title: 'Tests Failed';
  testsSkipped     : Integer @title: 'Tests Skipped';
  
  // Coverage
  codeCoverage     : Decimal(5,2) @title: 'Code Coverage %';
  branchCoverage   : Decimal(5,2) @title: 'Branch Coverage %';
  
  // Results
  testResults      : LargeString @title: 'Test Results (JSON)';
  coverageReport   : LargeString @title: 'Coverage Report (JSON)';
  
  // Context
  triggeredBy      : String(255) @title: 'Triggered By';
  buildId          : String(100) @title: 'Build ID';
  commitHash       : String(100) @title: 'Commit Hash';
}

// ========================================
// NOTIFICATIONS AND MESSAGING
// ========================================

entity Notifications : cuid, managed {
  recipient        : Association to Users @title: 'Recipient';
  notificationId   : String(100) @title: 'Notification ID';
  title            : String(255) @title: 'Title';
  message          : String(2000) @title: 'Message';
  type             : String(50) @title: 'Type'; // INFO, SUCCESS, WARNING, ERROR
  priority         : String(20) @title: 'Priority'; // LOW, MEDIUM, HIGH, CRITICAL
  category         : String(50) @title: 'Category'; // SYSTEM, PROJECT, DEPLOYMENT, SECURITY
  
  // Status
  status           : String(20) default 'UNREAD' @title: 'Status'; // UNREAD, READ, DISMISSED
  readAt           : Timestamp @title: 'Read At';
  dismissedAt      : Timestamp @title: 'Dismissed At';
  
  // Actions
  actions          : LargeString @title: 'Actions (JSON Array)';
  
  // Context
  sourceType       : String(50) @title: 'Source Type';
  sourceId         : String(100) @title: 'Source ID';
  correlationId    : String(100) @title: 'Correlation ID';
  
  // Scheduling
  scheduledFor     : Timestamp @title: 'Scheduled For';
  expiresAt        : Timestamp @title: 'Expires At';
  
  // Delivery
  deliveryChannels : String(200) @title: 'Delivery Channels';
  deliveryStatus   : String(20) @title: 'Delivery Status';
  deliveredAt      : Timestamp @title: 'Delivered At';
}

// ========================================
// AUDIT AND MONITORING
// ========================================

entity AuditLogs : cuid {
  user             : Association to Users @title: 'User';
  timestamp        : Timestamp @title: 'Timestamp';
  action           : String(100) @title: 'Action';
  resource         : String(100) @title: 'Resource';
  resourceId       : String(100) @title: 'Resource ID';
  
  // Request context
  sessionId        : String(100) @title: 'Session ID';
  correlationId    : String(100) @title: 'Correlation ID';
  ipAddress        : String(50) @title: 'IP Address';
  userAgent        : String(500) @title: 'User Agent';
  
  // Changes
  oldValue         : LargeString @title: 'Old Value (JSON)';
  newValue         : LargeString @title: 'New Value (JSON)';
  
  // Result
  status           : String(20) @title: 'Status'; // SUCCESS, FAILED, UNAUTHORIZED
  errorMessage     : String(1000) @title: 'Error Message';
  
  // Compliance
  dataClassification : String(20) @title: 'Data Classification';
  retentionPeriod    : Integer @title: 'Retention Period (days)';
}

entity SessionLogs : cuid {
  user             : Association to Users @title: 'User';
  sessionId        : String(100) @title: 'Session ID';
  startTime        : Timestamp @title: 'Start Time';
  endTime          : Timestamp @title: 'End Time';
  duration         : Integer @title: 'Duration (seconds)';
  
  // Session details
  ipAddress        : String(50) @title: 'IP Address';
  userAgent        : String(500) @title: 'User Agent';
  location         : String(255) @title: 'Location';
  
  // Activity
  pageViews        : Integer default 0 @title: 'Page Views';
  actionsPerformed : Integer default 0 @title: 'Actions Performed';
  lastActivity     : Timestamp @title: 'Last Activity';
  
  // Status
  status           : String(20) @title: 'Status'; // ACTIVE, EXPIRED, TERMINATED
  terminationReason : String(100) @title: 'Termination Reason';
}

entity ProjectAuditTrail : cuid {
  project          : Association to Projects @title: 'Project';
  timestamp        : Timestamp @title: 'Timestamp';
  user             : Association to Users @title: 'User';
  action           : String(100) @title: 'Action';
  
  // Changes
  fieldName        : String(100) @title: 'Field Name';
  oldValue         : String(2000) @title: 'Old Value';
  newValue         : String(2000) @title: 'New Value';
  
  // Context
  sessionId        : String(100) @title: 'Session ID';
  correlationId    : String(100) @title: 'Correlation ID';
  comment          : String(1000) @title: 'Comment';
}

// ========================================
// SYSTEM CONFIGURATION
// ========================================

entity UserSettings : cuid, managed {
  user             : Association to Users @title: 'User';
  category         : String(50) @title: 'Category';
  settingKey       : String(100) @title: 'Setting Key';
  settingValue     : LargeString @title: 'Setting Value (JSON)';
  dataType         : String(20) @title: 'Data Type';
  isEncrypted      : Boolean default false @title: 'Is Encrypted';
  description      : String(500) @title: 'Description';
}

entity SystemConfiguration : cuid, managed {
  configKey        : String(100) @title: 'Configuration Key';
  configValue      : LargeString @title: 'Configuration Value';
  category         : String(50) @title: 'Category';
  dataType         : String(20) @title: 'Data Type';
  isEncrypted      : Boolean default false @title: 'Is Encrypted';
  description      : String(1000) @title: 'Description';
  validFrom        : Date @title: 'Valid From';
  validTo          : Date @title: 'Valid To';
  environment      : String(20) @title: 'Environment';
}

entity ProjectDocuments : cuid, managed {
  project          : Association to Projects @title: 'Project';
  documentId       : String(100) @title: 'Document ID';
  name             : String(255) @title: 'Document Name';
  description      : String(1000) @title: 'Description';
  category         : String(50) @title: 'Category'; // SPECIFICATION, DOCUMENTATION, TEST_REPORT
  
  // File details
  fileName         : String(255) @title: 'File Name';
  fileSize         : Integer @title: 'File Size (bytes)';
  mimeType         : String(100) @title: 'MIME Type';
  
  // Storage
  storageLocation  : String(500) @title: 'Storage Location';
  checksum         : String(100) @title: 'Checksum';
  
  // Version control
  version          : String(20) @title: 'Version';
  
  // Security
  dataClassification : String(20) @title: 'Data Classification';
  accessRestriction  : String(100) @title: 'Access Restriction';
}

// ========================================
// PERFORMANCE MONITORING
// ========================================

entity PerformanceMetrics : cuid {
  timestamp        : Timestamp @title: 'Timestamp';
  metricType       : String(50) @title: 'Metric Type'; // CPU, MEMORY, RESPONSE_TIME, ERROR_RATE
  metricName       : String(100) @title: 'Metric Name';
  metricValue      : Decimal(15,6) @title: 'Metric Value';
  unit             : String(20) @title: 'Unit';
  
  // Context
  resourceType     : String(50) @title: 'Resource Type'; // AGENT, WORKFLOW, SYSTEM
  resourceId       : String(100) @title: 'Resource ID';
  environment      : String(20) @title: 'Environment';
  
  // Additional data
  tags             : String(1000) @title: 'Tags (JSON)';
  dimensions       : String(1000) @title: 'Dimensions (JSON)';
}

// ========================================
// BUSINESS VIEWS
// ========================================

define view ProjectOverview as select from Projects {
  key ID,
  projectId,
  name,
  description,
  status,
  priority,
  startDate,
  endDate,
  budget,
  currency,
  businessUnit.name as businessUnitName,
  department.name as departmentName,
  projectManager.displayName as projectManagerName,
  
  // Calculated fields
  case 
    when endDate < $now then 'OVERDUE'
    when startDate > $now then 'NOT_STARTED'
    else 'IN_PROGRESS'
  end as scheduleStatus : String(20),
  
  // Aggregations
  members.user.ID as memberCount,
  agents.ID as agentCount,
  workflows.ID as workflowCount
} where Projects.active = true;

define view UserDashboard as select from Users {
  key ID,
  userId,
  displayName,
  department.name as departmentName,
  businessUnit.name as businessUnitName,
  lastLogin,
  
  // Project involvement
  projectMemberships.project.ID as projectCount,
  
  // Notification summary
  notifications[status = 'UNREAD'].ID as unreadNotifications,
  notifications[priority = 'CRITICAL' and status = 'UNREAD'].ID as criticalNotifications
} where Users.active = true;
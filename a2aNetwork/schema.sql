
CREATE TABLE BlockchainService_BlockchainStats (
  ID NVARCHAR(36) NOT NULL,
  blockHeight INTEGER,
  gasPrice DECIMAL(10, 2),
  networkStatus NVARCHAR(255),
  totalTransactions INTEGER,
  averageBlockTime DECIMAL(8, 2),
  timestamp DATETIME_TEXT,
  PRIMARY KEY(ID)
);

CREATE TABLE ConfigurationService_NetworkSettings (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  network NVARCHAR(50),
  rpcUrl NVARCHAR(200),
  chainId INTEGER,
  contractAddress NVARCHAR(42),
  isActive BOOLEAN DEFAULT TRUE,
  version INTEGER DEFAULT 1,
  PRIMARY KEY(ID)
);

CREATE TABLE ConfigurationService_SecuritySettings (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  encryptionEnabled BOOLEAN DEFAULT TRUE,
  authRequired BOOLEAN DEFAULT TRUE,
  twoFactorEnabled BOOLEAN DEFAULT FALSE,
  sessionTimeout INTEGER DEFAULT 30,
  maxLoginAttempts INTEGER DEFAULT 5,
  passwordMinLength INTEGER DEFAULT 8,
  isActive BOOLEAN DEFAULT TRUE,
  version INTEGER DEFAULT 1,
  PRIMARY KEY(ID)
);

CREATE TABLE ConfigurationService_ApplicationSettings (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  environment NVARCHAR(20),
  logLevel NVARCHAR(10) DEFAULT 'info',
  enableMetrics BOOLEAN DEFAULT TRUE,
  enableTracing BOOLEAN DEFAULT TRUE,
  maintenanceMode BOOLEAN DEFAULT FALSE,
  maxConcurrentUsers INTEGER DEFAULT 1000,
  cacheEnabled BOOLEAN DEFAULT TRUE,
  cacheTTL INTEGER DEFAULT 300,
  isActive BOOLEAN DEFAULT TRUE,
  version INTEGER DEFAULT 1,
  PRIMARY KEY(ID)
);

CREATE TABLE ConfigurationService_SettingsAuditLog (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  settingType NVARCHAR(50),
  settingKey NVARCHAR(100),
  oldValue NVARCHAR(500),
  newValue NVARCHAR(500),
  changedBy NVARCHAR(100),
  changeReason NVARCHAR(200),
  timestamp DATETIME_TEXT,
  version INTEGER,
  PRIMARY KEY(ID)
);

CREATE TABLE ConfigurationService_AutoSavedSettings (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  settingsData NCLOB,
  settingsType NVARCHAR(50),
  userId NVARCHAR(100),
  timestamp DATETIME_TEXT,
  version INTEGER,
  isLatest BOOLEAN DEFAULT FALSE,
  PRIMARY KEY(ID)
);

CREATE TABLE OperationsService_Health (
  ID NVARCHAR(36) NOT NULL,
  status NVARCHAR(255),
  score INTEGER,
  timestamp TIMESTAMP_TEXT,
  issues NCLOB,
  metrics_cpu DECIMAL,
  metrics_memory DECIMAL,
  metrics_responseTime INTEGER,
  metrics_activeAlerts INTEGER,
  PRIMARY KEY(ID)
);

CREATE TABLE OperationsService_HealthComponent (
  component NVARCHAR(255) NOT NULL,
  health_ID NVARCHAR(36),
  status NVARCHAR(255),
  lastCheck TIMESTAMP_TEXT,
  details NCLOB,
  PRIMARY KEY(component),
  CONSTRAINT c__OperationsService_HealthComponent_health
  FOREIGN KEY(health_ID)
  REFERENCES OperationsService_Health(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE OperationsService_Metrics (
  name NVARCHAR(255) NOT NULL,
  value DECIMAL,
  unit NVARCHAR(255),
  timestamp TIMESTAMP_TEXT,
  tags NCLOB,
  PRIMARY KEY(name)
);

CREATE TABLE OperationsService_Alerts (
  ID NVARCHAR(36) NOT NULL,
  name NVARCHAR(255),
  severity NVARCHAR(255),
  status NVARCHAR(255),
  message NVARCHAR(255),
  timestamp TIMESTAMP_TEXT,
  acknowledgedBy NVARCHAR(255),
  acknowledgedAt TIMESTAMP_TEXT,
  resolvedBy NVARCHAR(255),
  resolvedAt TIMESTAMP_TEXT,
  metric_name NVARCHAR(255),
  metric_value DECIMAL,
  metric_threshold DECIMAL,
  PRIMARY KEY(ID)
);

CREATE TABLE OperationsService_Logs (
  ID NVARCHAR(36) NOT NULL,
  timestamp TIMESTAMP_TEXT,
  level NVARCHAR(255),
  logger NVARCHAR(255),
  message NCLOB,
  correlationId NVARCHAR(255),
  tenant NVARCHAR(255),
  user NVARCHAR(255),
  details NCLOB,
  PRIMARY KEY(ID)
);

CREATE TABLE OperationsService_Traces (
  traceId NVARCHAR(255) NOT NULL,
  spanId NVARCHAR(255),
  parentSpanId NVARCHAR(255),
  operationName NVARCHAR(255),
  serviceName NVARCHAR(255),
  startTime TIMESTAMP_TEXT,
  endTime TIMESTAMP_TEXT,
  duration INTEGER,
  status NVARCHAR(255),
  tags NCLOB,
  PRIMARY KEY(traceId)
);

CREATE TABLE OperationsService_Configuration (
  name NVARCHAR(255) NOT NULL,
  value NCLOB,
  type NVARCHAR(255),
  category NVARCHAR(255),
  description NVARCHAR(255),
  lastModified TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  PRIMARY KEY(name)
);

CREATE TABLE a2a_network_Agents (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  address NVARCHAR(42),
  name NVARCHAR(100),
  endpoint NVARCHAR(500),
  reputation INTEGER DEFAULT 100,
  isActive BOOLEAN DEFAULT TRUE,
  country_code NVARCHAR(3),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Agents_country
  FOREIGN KEY(country_code)
  REFERENCES sap_common_Countries(code)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_AgentCapabilities (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36),
  capability_ID NVARCHAR(36),
  registeredAt DATETIME_TEXT,
  version NVARCHAR(20),
  status NVARCHAR(20),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_AgentCapabilities_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_AgentCapabilities_capability
  FOREIGN KEY(capability_ID)
  REFERENCES a2a_network_Capabilities(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_Capabilities (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  name NVARCHAR(100),
  description NVARCHAR(1000),
  category_code INTEGER,
  tags NCLOB,
  inputTypes NCLOB,
  outputTypes NCLOB,
  version NVARCHAR(20) DEFAULT '1.0.0',
  status_code NVARCHAR(20) DEFAULT 'active',
  dependencies NCLOB,
  conflicts NCLOB,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Capabilities_category
  FOREIGN KEY(category_code)
  REFERENCES a2a_network_CapabilityCategories(code)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_Capabilities_status
  FOREIGN KEY(status_code)
  REFERENCES a2a_network_StatusCodes(code)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_CapabilityCategories (
  name NVARCHAR(100),
  descr NVARCHAR(1000),
  code INTEGER NOT NULL,
  description NVARCHAR(500),
  PRIMARY KEY(code)
);

CREATE TABLE a2a_network_StatusCodes (
  name NVARCHAR(50),
  descr NVARCHAR(1000),
  code NVARCHAR(20) NOT NULL,
  description NVARCHAR(200),
  PRIMARY KEY(code)
);

CREATE TABLE a2a_network_Services (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  provider_ID NVARCHAR(36),
  name NVARCHAR(100),
  description NVARCHAR(1000),
  category NVARCHAR(50),
  pricePerCall DECIMAL(10, 4),
  currency_code NVARCHAR(3) DEFAULT 'EUR',
  minReputation INTEGER DEFAULT 0,
  maxCallsPerDay INTEGER DEFAULT 1000,
  isActive BOOLEAN DEFAULT TRUE,
  totalCalls INTEGER DEFAULT 0,
  averageRating DECIMAL(3, 2) DEFAULT 0,
  escrowAmount DECIMAL(10, 4) DEFAULT 0,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Services_provider
  FOREIGN KEY(provider_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_Services_currency
  FOREIGN KEY(currency_code)
  REFERENCES sap_common_Currencies(code)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ServiceOrders (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  service_ID NVARCHAR(36),
  consumer_ID NVARCHAR(36),
  status NVARCHAR(20),
  callCount INTEGER DEFAULT 0,
  totalAmount DECIMAL(10, 4),
  escrowReleased BOOLEAN DEFAULT FALSE,
  completedAt DATETIME_TEXT,
  rating INTEGER,
  feedback NVARCHAR(500),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_ServiceOrders_service
  FOREIGN KEY(service_ID)
  REFERENCES a2a_network_Services(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_ServiceOrders_consumer
  FOREIGN KEY(consumer_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_Messages (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  sender_ID NVARCHAR(36),
  recipient_ID NVARCHAR(36),
  messageHash NVARCHAR(66),
  protocol NVARCHAR(50),
  priority INTEGER DEFAULT 1,
  status NVARCHAR(20),
  retryCount INTEGER DEFAULT 0,
  gasUsed INTEGER,
  deliveredAt DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Messages_sender
  FOREIGN KEY(sender_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_Messages_recipient
  FOREIGN KEY(recipient_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_AgentPerformance (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36),
  totalTasks INTEGER DEFAULT 0,
  successfulTasks INTEGER DEFAULT 0,
  failedTasks INTEGER DEFAULT 0,
  averageResponseTime INTEGER,
  averageGasUsage INTEGER,
  reputationScore INTEGER DEFAULT 100,
  trustScore DECIMAL(3, 2) DEFAULT 1.0,
  lastUpdated DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_AgentPerformance_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_PerformanceSnapshots (
  ID NVARCHAR(36) NOT NULL,
  performance_ID NVARCHAR(36),
  timestamp DATETIME_TEXT,
  taskCount INTEGER,
  successRate DECIMAL(5, 2),
  responseTime INTEGER,
  gasUsage INTEGER,
  reputationDelta INTEGER,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_PerformanceSnapshots_performance
  FOREIGN KEY(performance_ID)
  REFERENCES a2a_network_AgentPerformance(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_Workflows (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  name NVARCHAR(100),
  description NVARCHAR(1000),
  definition NCLOB,
  isActive BOOLEAN DEFAULT TRUE,
  category NVARCHAR(50),
  owner_ID NVARCHAR(36),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Workflows_owner
  FOREIGN KEY(owner_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_WorkflowExecutions (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  workflow_ID NVARCHAR(36),
  executionId NVARCHAR(66),
  status NVARCHAR(20),
  startedAt DATETIME_TEXT,
  completedAt DATETIME_TEXT,
  gasUsed INTEGER,
  result NCLOB,
  error NVARCHAR(1000),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_WorkflowExecutions_workflow
  FOREIGN KEY(workflow_ID)
  REFERENCES a2a_network_Workflows(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_WorkflowSteps (
  ID NVARCHAR(36) NOT NULL,
  execution_ID NVARCHAR(36),
  stepNumber INTEGER,
  agentAddress NVARCHAR(42),
  "action" NVARCHAR(100),
  input NCLOB,
  output NCLOB,
  status NVARCHAR(20),
  gasUsed INTEGER,
  startedAt DATETIME_TEXT,
  completedAt DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_WorkflowSteps_execution
  FOREIGN KEY(execution_ID)
  REFERENCES a2a_network_WorkflowExecutions(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ChainBridges (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  sourceChain NVARCHAR(50),
  targetChain NVARCHAR(50),
  bridgeAddress NVARCHAR(42),
  isActive BOOLEAN DEFAULT TRUE,
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_CrossChainTransfers (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  bridge_ID NVARCHAR(36),
  fromAgent NVARCHAR(42),
  toAgent NVARCHAR(42),
  messageHash NVARCHAR(66),
  status NVARCHAR(20),
  sourceBlock INTEGER,
  targetBlock INTEGER,
  gasUsed INTEGER,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_CrossChainTransfers_bridge
  FOREIGN KEY(bridge_ID)
  REFERENCES a2a_network_ChainBridges(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_PrivateChannels (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  participants NCLOB,
  publicKey NVARCHAR(130),
  isActive BOOLEAN DEFAULT TRUE,
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_PrivateMessages (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  channel_ID NVARCHAR(36),
  sender NVARCHAR(42),
  encryptedData NCLOB,
  zkProof NVARCHAR(500),
  timestamp DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_PrivateMessages_channel
  FOREIGN KEY(channel_ID)
  REFERENCES a2a_network_PrivateChannels(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_NetworkStats (
  validFrom TIMESTAMP_TEXT NOT NULL,
  validTo TIMESTAMP_TEXT,
  ID NVARCHAR(36) NOT NULL,
  totalAgents INTEGER,
  activeAgents INTEGER,
  totalServices INTEGER,
  totalCapabilities INTEGER,
  totalMessages INTEGER,
  totalTransactions INTEGER,
  averageReputation DECIMAL(5, 2),
  networkLoad DECIMAL(5, 2),
  gasPrice DECIMAL(10, 4),
  PRIMARY KEY(validFrom, ID)
);

CREATE TABLE a2a_network_NetworkConfig (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  configKey NVARCHAR(100),
  value NVARCHAR(1000),
  description NVARCHAR(500),
  isActive BOOLEAN DEFAULT TRUE,
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_Requests (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  title NVARCHAR(200),
  description NVARCHAR(2000),
  requestType NVARCHAR(50) DEFAULT 'SERVICE_REQUEST',
  priority NVARCHAR(10) DEFAULT 'MEDIUM',
  status NVARCHAR(20) DEFAULT 'PENDING',
  requester_ID NVARCHAR(36),
  assignedAgent_ID NVARCHAR(36),
  dueDate DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Requests_requester
  FOREIGN KEY(requester_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_Requests_assignedAgent
  FOREIGN KEY(assignedAgent_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_Responses (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  content NVARCHAR(5000),
  responseType NVARCHAR(50) DEFAULT 'TEXT',
  status NVARCHAR(20) DEFAULT 'DRAFT',
  request_ID NVARCHAR(36),
  responder_ID NVARCHAR(36),
  priority NVARCHAR(10) DEFAULT 'NORMAL',
  isFinalResponse BOOLEAN DEFAULT FALSE,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_Responses_request
  FOREIGN KEY(request_ID)
  REFERENCES a2a_network_Requests(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_Responses_responder
  FOREIGN KEY(responder_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_TenantSettings (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  tenant NVARCHAR(36) NOT NULL,
  settings_maxAgents INTEGER DEFAULT 1000,
  settings_maxServices INTEGER DEFAULT 100,
  settings_maxWorkflows INTEGER DEFAULT 50,
  settings_features_blockchain BOOLEAN DEFAULT TRUE,
  settings_features_ai BOOLEAN DEFAULT TRUE,
  settings_features_analytics BOOLEAN DEFAULT TRUE,
  PRIMARY KEY(tenant)
);

CREATE TABLE a2a_network_AuditLog (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  ID NVARCHAR(36) NOT NULL,
  tenant NVARCHAR(36),
  user NVARCHAR(100),
  "action" NVARCHAR(50),
  entity NVARCHAR(100),
  entityKey NVARCHAR(100),
  oldValue NCLOB,
  newValue NCLOB,
  ip NVARCHAR(45),
  userAgent NVARCHAR(500),
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_FeatureToggles (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  feature NVARCHAR(50) NOT NULL,
  enabled BOOLEAN DEFAULT FALSE,
  description NVARCHAR(200),
  validFrom DATE_TEXT,
  validTo DATE_TEXT,
  tenant NVARCHAR(36),
  PRIMARY KEY(feature)
);

CREATE TABLE a2a_network_ExtensionFields (
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  entity NVARCHAR(100) NOT NULL,
  field NVARCHAR(50) NOT NULL,
  tenant NVARCHAR(36) NOT NULL,
  dataType NVARCHAR(20),
  label NVARCHAR(100),
  defaultValue NVARCHAR(100),
  mandatory BOOLEAN DEFAULT FALSE,
  visible BOOLEAN DEFAULT TRUE,
  PRIMARY KEY(entity, field, tenant)
);

CREATE TABLE a2a_network_ReputationTransactions (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  agent_ID NVARCHAR(36),
  transactionType NVARCHAR(50) NOT NULL,
  amount INTEGER NOT NULL,
  reason NVARCHAR(200),
  context NCLOB,
  isAutomated BOOLEAN DEFAULT FALSE,
  createdByAgent_ID NVARCHAR(36),
  serviceOrder_ID NVARCHAR(36),
  workflow_ID NVARCHAR(36),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_ReputationTransactions_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_ReputationTransactions_createdByAgent
  FOREIGN KEY(createdByAgent_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_ReputationTransactions_serviceOrder
  FOREIGN KEY(serviceOrder_ID)
  REFERENCES a2a_network_ServiceOrders(ID)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_ReputationTransactions_workflow
  FOREIGN KEY(workflow_ID)
  REFERENCES a2a_network_Workflows(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_PeerEndorsements (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  fromAgent_ID NVARCHAR(36) NOT NULL,
  toAgent_ID NVARCHAR(36) NOT NULL,
  amount INTEGER NOT NULL,
  reason NVARCHAR(50) NOT NULL,
  context NCLOB,
  workflow_ID NVARCHAR(36),
  serviceOrder_ID NVARCHAR(36),
  expiresAt DATETIME_TEXT,
  isReciprocal BOOLEAN DEFAULT FALSE,
  verificationHash NVARCHAR(64),
  blockchainTx NVARCHAR(66),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_PeerEndorsements_fromAgent
  FOREIGN KEY(fromAgent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_PeerEndorsements_toAgent
  FOREIGN KEY(toAgent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_PeerEndorsements_workflow
  FOREIGN KEY(workflow_ID)
  REFERENCES a2a_network_Workflows(ID)
  DEFERRABLE INITIALLY DEFERRED,
  CONSTRAINT c__a2a_network_PeerEndorsements_serviceOrder
  FOREIGN KEY(serviceOrder_ID)
  REFERENCES a2a_network_ServiceOrders(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ReputationMilestones (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36),
  milestone INTEGER NOT NULL,
  badgeName NVARCHAR(20) NOT NULL,
  achievedAt DATETIME_TEXT NOT NULL,
  badgeMetadata NVARCHAR(500),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_ReputationMilestones_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ReputationRecovery (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  agent_ID NVARCHAR(36),
  recoveryType NVARCHAR(30) NOT NULL,
  status NVARCHAR(20) DEFAULT 'PENDING',
  requirements NCLOB,
  progress NCLOB,
  reputationReward INTEGER DEFAULT 20,
  startedAt DATETIME_TEXT,
  completedAt DATETIME_TEXT,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_ReputationRecovery_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_DailyReputationLimits (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36),
  date DATE_TEXT NOT NULL,
  endorsementsGiven INTEGER DEFAULT 0,
  pointsGiven INTEGER DEFAULT 0,
  maxDailyLimit INTEGER DEFAULT 50,
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_DailyReputationLimits_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ReputationAnalytics (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36),
  periodStart DATE_TEXT NOT NULL,
  periodEnd DATE_TEXT NOT NULL,
  startingReputation INTEGER,
  endingReputation INTEGER,
  totalEarned INTEGER DEFAULT 0,
  totalLost INTEGER DEFAULT 0,
  endorsementsReceived INTEGER DEFAULT 0,
  endorsementsGiven INTEGER DEFAULT 0,
  uniqueEndorsers INTEGER DEFAULT 0,
  averageTransaction DECIMAL(5, 2),
  taskSuccessRate DECIMAL(5, 2),
  serviceRatingAverage DECIMAL(3, 2),
  PRIMARY KEY(ID),
  CONSTRAINT c__a2a_network_ReputationAnalytics_agent
  FOREIGN KEY(agent_ID)
  REFERENCES a2a_network_Agents(ID)
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE sap_common_Countries (
  name NVARCHAR(255),
  descr NVARCHAR(1000),
  code NVARCHAR(3) NOT NULL,
  PRIMARY KEY(code)
);

CREATE TABLE sap_common_Currencies (
  name NVARCHAR(255),
  descr NVARCHAR(1000),
  code NVARCHAR(3) NOT NULL,
  symbol NVARCHAR(5),
  minorUnit SMALLINT,
  PRIMARY KEY(code)
);

CREATE TABLE a2a_network_draft_DraftAdministrativeData (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  DraftUUID NVARCHAR(36),
  CreationDateTime DATETIME_TEXT,
  CreatedByUser NVARCHAR(256),
  DraftIsCreatedByMe BOOLEAN DEFAULT FALSE,
  DraftIsProcessedByMe BOOLEAN DEFAULT FALSE,
  DraftIsKeptByMe BOOLEAN DEFAULT FALSE,
  EnqueueStartDateTime DATETIME_TEXT,
  DraftEntityCreationDateTime DATETIME_TEXT,
  DraftEntityLastChangeDateTime DATETIME_TEXT,
  HasActiveEntity BOOLEAN DEFAULT FALSE,
  HasDraftEntity BOOLEAN DEFAULT FALSE,
  ProcessingStartedByUser NVARCHAR(256),
  ProcessingStartedByUserDescription NVARCHAR(256),
  LastChangedByUser NVARCHAR(256),
  LastChangedByUserDescription NVARCHAR(256),
  LastChangeDateTime DATETIME_TEXT,
  InProcessByUser NVARCHAR(256),
  InProcessByUserDescription NVARCHAR(256),
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_RequestDrafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  title NVARCHAR(200),
  description NVARCHAR(2000),
  status NVARCHAR(20) DEFAULT 'DRAFT',
  priority NVARCHAR(10) DEFAULT 'MEDIUM',
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_ResponseDrafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  content NVARCHAR(5000),
  responseType NVARCHAR(50),
  status NVARCHAR(20) DEFAULT 'DRAFT',
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_AgentDrafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  name NVARCHAR(200),
  description NVARCHAR(1000),
  agentType NVARCHAR(50),
  status NVARCHAR(20) DEFAULT 'DRAFT',
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_ServiceDrafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  name NVARCHAR(200),
  description NVARCHAR(2000),
  serviceType NVARCHAR(50),
  status NVARCHAR(20) DEFAULT 'DRAFT',
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_WorkflowDrafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  name NVARCHAR(200),
  description NVARCHAR(2000),
  workflowType NVARCHAR(50),
  status NVARCHAR(20) DEFAULT 'DRAFT',
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_draft_DraftConflicts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT,
  createdBy NVARCHAR(255),
  modifiedAt TIMESTAMP_TEXT,
  modifiedBy NVARCHAR(255),
  conflictType NVARCHAR(50),
  conflictDescription NVARCHAR(2000),
  conflictStatus NVARCHAR(20) DEFAULT 'PENDING',
  resolutionStrategy NVARCHAR(50),
  resolutionNotes NVARCHAR(2000),
  resolvedBy NVARCHAR(256),
  resolvedAt DATETIME_TEXT,
  PRIMARY KEY(ID)
);

CREATE TABLE a2a_network_Agents_texts (
  locale NVARCHAR(14) NOT NULL,
  ID NVARCHAR(36) NOT NULL,
  name NVARCHAR(100),
  PRIMARY KEY(locale, ID),
  CONSTRAINT c__a2a_network_Agents_texts_texts
  FOREIGN KEY(ID)
  REFERENCES a2a_network_Agents(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_Capabilities_texts (
  locale NVARCHAR(14) NOT NULL,
  ID NVARCHAR(36) NOT NULL,
  name NVARCHAR(100),
  description NVARCHAR(1000),
  PRIMARY KEY(locale, ID),
  CONSTRAINT c__a2a_network_Capabilities_texts_texts
  FOREIGN KEY(ID)
  REFERENCES a2a_network_Capabilities(ID)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_CapabilityCategories_texts (
  locale NVARCHAR(14) NOT NULL,
  name NVARCHAR(100),
  descr NVARCHAR(1000),
  code INTEGER NOT NULL,
  description NVARCHAR(500),
  PRIMARY KEY(locale, code),
  CONSTRAINT c__a2a_network_CapabilityCategories_texts_texts
  FOREIGN KEY(code)
  REFERENCES a2a_network_CapabilityCategories(code)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_StatusCodes_texts (
  locale NVARCHAR(14) NOT NULL,
  name NVARCHAR(50),
  descr NVARCHAR(1000),
  code NVARCHAR(20) NOT NULL,
  description NVARCHAR(200),
  PRIMARY KEY(locale, code),
  CONSTRAINT c__a2a_network_StatusCodes_texts_texts
  FOREIGN KEY(code)
  REFERENCES a2a_network_StatusCodes(code)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_FeatureToggles_texts (
  locale NVARCHAR(14) NOT NULL,
  feature NVARCHAR(50) NOT NULL,
  description NVARCHAR(200),
  PRIMARY KEY(locale, feature),
  CONSTRAINT c__a2a_network_FeatureToggles_texts_texts
  FOREIGN KEY(feature)
  REFERENCES a2a_network_FeatureToggles(feature)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE a2a_network_ExtensionFields_texts (
  locale NVARCHAR(14) NOT NULL,
  entity NVARCHAR(100) NOT NULL,
  field NVARCHAR(50) NOT NULL,
  tenant NVARCHAR(36) NOT NULL,
  label NVARCHAR(100),
  PRIMARY KEY(locale, entity, field, tenant),
  CONSTRAINT c__a2a_network_ExtensionFields_texts_texts
  FOREIGN KEY(entity, field, tenant)
  REFERENCES a2a_network_ExtensionFields(entity, field, tenant)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE sap_common_Countries_texts (
  locale NVARCHAR(14) NOT NULL,
  name NVARCHAR(255),
  descr NVARCHAR(1000),
  code NVARCHAR(3) NOT NULL,
  PRIMARY KEY(locale, code),
  CONSTRAINT c__sap_common_Countries_texts_texts
  FOREIGN KEY(code)
  REFERENCES sap_common_Countries(code)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE sap_common_Currencies_texts (
  locale NVARCHAR(14) NOT NULL,
  name NVARCHAR(255),
  descr NVARCHAR(1000),
  code NVARCHAR(3) NOT NULL,
  PRIMARY KEY(locale, code),
  CONSTRAINT c__sap_common_Currencies_texts_texts
  FOREIGN KEY(code)
  REFERENCES sap_common_Currencies(code)
  ON DELETE CASCADE
  DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE DRAFT_DraftAdministrativeData (
  DraftUUID NVARCHAR(36) NOT NULL,
  CreationDateTime TIMESTAMP_TEXT,
  CreatedByUser NVARCHAR(256),
  DraftIsCreatedByMe BOOLEAN,
  LastChangeDateTime TIMESTAMP_TEXT,
  LastChangedByUser NVARCHAR(256),
  InProcessByUser NVARCHAR(256),
  DraftIsProcessedByMe BOOLEAN,
  PRIMARY KEY(DraftUUID)
);

CREATE TABLE A2ADraftService_Agents_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  name NVARCHAR(200) NULL,
  description NVARCHAR(1000) NULL,
  agentType NVARCHAR(50) NULL,
  status NVARCHAR(20) NULL DEFAULT 'DRAFT',
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_AgentCapabilities_drafts (
  ID NVARCHAR(36) NOT NULL,
  agent_ID NVARCHAR(36) NULL,
  capability_ID NVARCHAR(36) NULL,
  registeredAt DATETIME_TEXT NULL,
  version NVARCHAR(20) NULL,
  status NVARCHAR(20) NULL,
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_ServiceDrafts_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  name NVARCHAR(200) NULL,
  description NVARCHAR(2000) NULL,
  serviceType NVARCHAR(50) NULL,
  status NVARCHAR(20) NULL DEFAULT 'DRAFT',
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_ServiceOrders_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  service_ID NVARCHAR(36) NULL,
  consumer_ID NVARCHAR(36) NULL,
  status NVARCHAR(20) NULL,
  callCount INTEGER NULL DEFAULT 0,
  totalAmount DECIMAL(10, 4) NULL,
  escrowReleased BOOLEAN NULL DEFAULT FALSE,
  completedAt DATETIME_TEXT NULL,
  rating INTEGER NULL,
  feedback NVARCHAR(500) NULL,
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_Workflows_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  name NVARCHAR(200) NULL,
  description NVARCHAR(2000) NULL,
  workflowType NVARCHAR(50) NULL,
  status NVARCHAR(20) NULL DEFAULT 'DRAFT',
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_Requests_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  title NVARCHAR(200) NULL,
  description NVARCHAR(2000) NULL,
  status NVARCHAR(20) NULL DEFAULT 'DRAFT',
  priority NVARCHAR(10) NULL DEFAULT 'MEDIUM',
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE TABLE A2ADraftService_Responses_drafts (
  ID NVARCHAR(36) NOT NULL,
  createdAt TIMESTAMP_TEXT NULL,
  createdBy NVARCHAR(255) NULL,
  modifiedAt TIMESTAMP_TEXT NULL,
  modifiedBy NVARCHAR(255) NULL,
  content NVARCHAR(5000) NULL,
  responseType NVARCHAR(50) NULL DEFAULT 'TEXT',
  status NVARCHAR(20) NULL DEFAULT 'DRAFT',
  request_ID NVARCHAR(36) NULL,
  responder_ID NVARCHAR(36) NULL,
  priority NVARCHAR(10) NULL DEFAULT 'NORMAL',
  isFinalResponse BOOLEAN NULL DEFAULT FALSE,
  IsActiveEntity BOOLEAN,
  HasActiveEntity BOOLEAN,
  HasDraftEntity BOOLEAN,
  DraftAdministrativeData_DraftUUID NVARCHAR(36) NOT NULL,
  PRIMARY KEY(ID)
);

CREATE VIEW A2AService_Agents AS SELECT
  Agents_0.ID,
  Agents_0.createdAt,
  Agents_0.createdBy,
  Agents_0.modifiedAt,
  Agents_0.modifiedBy,
  Agents_0.address,
  Agents_0.name,
  Agents_0.endpoint,
  Agents_0.reputation,
  Agents_0.isActive,
  Agents_0.country_code
FROM a2a_network_Agents AS Agents_0;

CREATE VIEW A2AService_AgentCapabilities AS SELECT
  AgentCapabilities_0.ID,
  AgentCapabilities_0.agent_ID,
  AgentCapabilities_0.capability_ID,
  AgentCapabilities_0.registeredAt,
  AgentCapabilities_0.version,
  AgentCapabilities_0.status
FROM a2a_network_AgentCapabilities AS AgentCapabilities_0;

CREATE VIEW A2AService_AgentPerformance AS SELECT
  AgentPerformance_0.ID,
  AgentPerformance_0.agent_ID,
  AgentPerformance_0.totalTasks,
  AgentPerformance_0.successfulTasks,
  AgentPerformance_0.failedTasks,
  AgentPerformance_0.averageResponseTime,
  AgentPerformance_0.averageGasUsage,
  AgentPerformance_0.reputationScore,
  AgentPerformance_0.trustScore,
  AgentPerformance_0.lastUpdated
FROM a2a_network_AgentPerformance AS AgentPerformance_0;

CREATE VIEW A2AService_PerformanceSnapshots AS SELECT
  PerformanceSnapshots_0.ID,
  PerformanceSnapshots_0.performance_ID,
  PerformanceSnapshots_0.timestamp,
  PerformanceSnapshots_0.taskCount,
  PerformanceSnapshots_0.successRate,
  PerformanceSnapshots_0.responseTime,
  PerformanceSnapshots_0.gasUsage,
  PerformanceSnapshots_0.reputationDelta
FROM a2a_network_PerformanceSnapshots AS PerformanceSnapshots_0;

CREATE VIEW A2AService_Capabilities AS SELECT
  Capabilities_0.ID,
  Capabilities_0.createdAt,
  Capabilities_0.createdBy,
  Capabilities_0.modifiedAt,
  Capabilities_0.modifiedBy,
  Capabilities_0.name,
  Capabilities_0.description,
  Capabilities_0.category_code,
  Capabilities_0.tags,
  Capabilities_0.inputTypes,
  Capabilities_0.outputTypes,
  Capabilities_0.version,
  Capabilities_0.status_code,
  Capabilities_0.dependencies,
  Capabilities_0.conflicts
FROM a2a_network_Capabilities AS Capabilities_0;

CREATE VIEW A2AService_Services AS SELECT
  Services_0.ID,
  Services_0.createdAt,
  Services_0.createdBy,
  Services_0.modifiedAt,
  Services_0.modifiedBy,
  Services_0.provider_ID,
  Services_0.name,
  Services_0.description,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.currency_code,
  Services_0.minReputation,
  Services_0.maxCallsPerDay,
  Services_0.isActive,
  Services_0.totalCalls,
  Services_0.averageRating,
  Services_0.escrowAmount
FROM a2a_network_Services AS Services_0;

CREATE VIEW A2AService_ServiceOrders AS SELECT
  ServiceOrders_0.ID,
  ServiceOrders_0.createdAt,
  ServiceOrders_0.createdBy,
  ServiceOrders_0.modifiedAt,
  ServiceOrders_0.modifiedBy,
  ServiceOrders_0.service_ID,
  ServiceOrders_0.consumer_ID,
  ServiceOrders_0.status,
  ServiceOrders_0.callCount,
  ServiceOrders_0.totalAmount,
  ServiceOrders_0.escrowReleased,
  ServiceOrders_0.completedAt,
  ServiceOrders_0.rating,
  ServiceOrders_0.feedback
FROM a2a_network_ServiceOrders AS ServiceOrders_0;

CREATE VIEW A2AService_Messages AS SELECT
  Messages_0.ID,
  Messages_0.createdAt,
  Messages_0.createdBy,
  Messages_0.modifiedAt,
  Messages_0.modifiedBy,
  Messages_0.sender_ID,
  Messages_0.recipient_ID,
  Messages_0.messageHash,
  Messages_0.protocol,
  Messages_0.priority,
  Messages_0.status,
  Messages_0.retryCount,
  Messages_0.gasUsed,
  Messages_0.deliveredAt
FROM a2a_network_Messages AS Messages_0;

CREATE VIEW A2AService_Workflows AS SELECT
  Workflows_0.ID,
  Workflows_0.createdAt,
  Workflows_0.createdBy,
  Workflows_0.modifiedAt,
  Workflows_0.modifiedBy,
  Workflows_0.name,
  Workflows_0.description,
  Workflows_0.definition,
  Workflows_0.isActive,
  Workflows_0.category,
  Workflows_0.owner_ID
FROM a2a_network_Workflows AS Workflows_0;

CREATE VIEW A2AService_WorkflowExecutions AS SELECT
  WorkflowExecutions_0.ID,
  WorkflowExecutions_0.createdAt,
  WorkflowExecutions_0.createdBy,
  WorkflowExecutions_0.modifiedAt,
  WorkflowExecutions_0.modifiedBy,
  WorkflowExecutions_0.workflow_ID,
  WorkflowExecutions_0.executionId,
  WorkflowExecutions_0.status,
  WorkflowExecutions_0.startedAt,
  WorkflowExecutions_0.completedAt,
  WorkflowExecutions_0.gasUsed,
  WorkflowExecutions_0.result,
  WorkflowExecutions_0.error
FROM a2a_network_WorkflowExecutions AS WorkflowExecutions_0;

CREATE VIEW A2AService_WorkflowSteps AS SELECT
  WorkflowSteps_0.ID,
  WorkflowSteps_0.execution_ID,
  WorkflowSteps_0.stepNumber,
  WorkflowSteps_0.agentAddress,
  WorkflowSteps_0."action",
  WorkflowSteps_0.input,
  WorkflowSteps_0.output,
  WorkflowSteps_0.status,
  WorkflowSteps_0.gasUsed,
  WorkflowSteps_0.startedAt,
  WorkflowSteps_0.completedAt
FROM a2a_network_WorkflowSteps AS WorkflowSteps_0;

CREATE VIEW A2AService_ChainBridges AS SELECT
  ChainBridges_0.ID,
  ChainBridges_0.createdAt,
  ChainBridges_0.createdBy,
  ChainBridges_0.modifiedAt,
  ChainBridges_0.modifiedBy,
  ChainBridges_0.sourceChain,
  ChainBridges_0.targetChain,
  ChainBridges_0.bridgeAddress,
  ChainBridges_0.isActive
FROM a2a_network_ChainBridges AS ChainBridges_0;

CREATE VIEW A2AService_CrossChainTransfers AS SELECT
  CrossChainTransfers_0.ID,
  CrossChainTransfers_0.createdAt,
  CrossChainTransfers_0.createdBy,
  CrossChainTransfers_0.modifiedAt,
  CrossChainTransfers_0.modifiedBy,
  CrossChainTransfers_0.bridge_ID,
  CrossChainTransfers_0.fromAgent,
  CrossChainTransfers_0.toAgent,
  CrossChainTransfers_0.messageHash,
  CrossChainTransfers_0.status,
  CrossChainTransfers_0.sourceBlock,
  CrossChainTransfers_0.targetBlock,
  CrossChainTransfers_0.gasUsed
FROM a2a_network_CrossChainTransfers AS CrossChainTransfers_0;

CREATE VIEW A2AService_PrivateChannels AS SELECT
  PrivateChannels_0.ID,
  PrivateChannels_0.createdAt,
  PrivateChannels_0.createdBy,
  PrivateChannels_0.modifiedAt,
  PrivateChannels_0.modifiedBy,
  PrivateChannels_0.participants,
  PrivateChannels_0.publicKey,
  PrivateChannels_0.isActive
FROM a2a_network_PrivateChannels AS PrivateChannels_0;

CREATE VIEW A2AService_PrivateMessages AS SELECT
  PrivateMessages_0.ID,
  PrivateMessages_0.createdAt,
  PrivateMessages_0.createdBy,
  PrivateMessages_0.modifiedAt,
  PrivateMessages_0.modifiedBy,
  PrivateMessages_0.channel_ID,
  PrivateMessages_0.sender,
  PrivateMessages_0.encryptedData,
  PrivateMessages_0.zkProof,
  PrivateMessages_0.timestamp
FROM a2a_network_PrivateMessages AS PrivateMessages_0;

CREATE VIEW A2AService_NetworkStats AS SELECT
  NetworkStats_0.validFrom,
  NetworkStats_0.validTo,
  NetworkStats_0.ID,
  NetworkStats_0.totalAgents,
  NetworkStats_0.activeAgents,
  NetworkStats_0.totalServices,
  NetworkStats_0.totalCapabilities,
  NetworkStats_0.totalMessages,
  NetworkStats_0.totalTransactions,
  NetworkStats_0.averageReputation,
  NetworkStats_0.networkLoad,
  NetworkStats_0.gasPrice
FROM a2a_network_NetworkStats AS NetworkStats_0
WHERE (NetworkStats_0.validFrom < session_context( '$valid.to' ) AND NetworkStats_0.validTo > session_context( '$valid.from' ));

CREATE VIEW A2AService_ReputationTransactions AS SELECT
  ReputationTransactions_0.ID,
  ReputationTransactions_0.createdAt,
  ReputationTransactions_0.createdBy,
  ReputationTransactions_0.modifiedAt,
  ReputationTransactions_0.modifiedBy,
  ReputationTransactions_0.agent_ID,
  ReputationTransactions_0.transactionType,
  ReputationTransactions_0.amount,
  ReputationTransactions_0.reason,
  ReputationTransactions_0.context,
  ReputationTransactions_0.isAutomated,
  ReputationTransactions_0.createdByAgent_ID,
  ReputationTransactions_0.serviceOrder_ID,
  ReputationTransactions_0.workflow_ID
FROM a2a_network_ReputationTransactions AS ReputationTransactions_0;

CREATE VIEW A2AService_PeerEndorsements AS SELECT
  PeerEndorsements_0.ID,
  PeerEndorsements_0.createdAt,
  PeerEndorsements_0.createdBy,
  PeerEndorsements_0.modifiedAt,
  PeerEndorsements_0.modifiedBy,
  PeerEndorsements_0.fromAgent_ID,
  PeerEndorsements_0.toAgent_ID,
  PeerEndorsements_0.amount,
  PeerEndorsements_0.reason,
  PeerEndorsements_0.context,
  PeerEndorsements_0.workflow_ID,
  PeerEndorsements_0.serviceOrder_ID,
  PeerEndorsements_0.expiresAt,
  PeerEndorsements_0.isReciprocal,
  PeerEndorsements_0.verificationHash,
  PeerEndorsements_0.blockchainTx
FROM a2a_network_PeerEndorsements AS PeerEndorsements_0;

CREATE VIEW A2AService_ReputationMilestones AS SELECT
  ReputationMilestones_0.ID,
  ReputationMilestones_0.agent_ID,
  ReputationMilestones_0.milestone,
  ReputationMilestones_0.badgeName,
  ReputationMilestones_0.achievedAt,
  ReputationMilestones_0.badgeMetadata
FROM a2a_network_ReputationMilestones AS ReputationMilestones_0;

CREATE VIEW A2AService_ReputationRecovery AS SELECT
  ReputationRecovery_0.ID,
  ReputationRecovery_0.createdAt,
  ReputationRecovery_0.createdBy,
  ReputationRecovery_0.modifiedAt,
  ReputationRecovery_0.modifiedBy,
  ReputationRecovery_0.agent_ID,
  ReputationRecovery_0.recoveryType,
  ReputationRecovery_0.status,
  ReputationRecovery_0.requirements,
  ReputationRecovery_0.progress,
  ReputationRecovery_0.reputationReward,
  ReputationRecovery_0.startedAt,
  ReputationRecovery_0.completedAt
FROM a2a_network_ReputationRecovery AS ReputationRecovery_0;

CREATE VIEW A2AService_DailyReputationLimits AS SELECT
  DailyReputationLimits_0.ID,
  DailyReputationLimits_0.agent_ID,
  DailyReputationLimits_0.date,
  DailyReputationLimits_0.endorsementsGiven,
  DailyReputationLimits_0.pointsGiven,
  DailyReputationLimits_0.maxDailyLimit
FROM a2a_network_DailyReputationLimits AS DailyReputationLimits_0;

CREATE VIEW A2AService_ReputationAnalytics AS SELECT
  ReputationAnalytics_0.ID,
  ReputationAnalytics_0.agent_ID,
  ReputationAnalytics_0.periodStart,
  ReputationAnalytics_0.periodEnd,
  ReputationAnalytics_0.startingReputation,
  ReputationAnalytics_0.endingReputation,
  ReputationAnalytics_0.totalEarned,
  ReputationAnalytics_0.totalLost,
  ReputationAnalytics_0.endorsementsReceived,
  ReputationAnalytics_0.endorsementsGiven,
  ReputationAnalytics_0.uniqueEndorsers,
  ReputationAnalytics_0.averageTransaction,
  ReputationAnalytics_0.taskSuccessRate,
  ReputationAnalytics_0.serviceRatingAverage
FROM a2a_network_ReputationAnalytics AS ReputationAnalytics_0;

CREATE VIEW A2AService_NetworkConfig AS SELECT
  NetworkConfig_0.ID,
  NetworkConfig_0.createdAt,
  NetworkConfig_0.createdBy,
  NetworkConfig_0.modifiedAt,
  NetworkConfig_0.modifiedBy,
  NetworkConfig_0.configKey,
  NetworkConfig_0.value,
  NetworkConfig_0.description,
  NetworkConfig_0.isActive
FROM a2a_network_NetworkConfig AS NetworkConfig_0;

CREATE VIEW A2ADraftService_Agents AS SELECT
  AgentDrafts_0.ID,
  AgentDrafts_0.createdAt,
  AgentDrafts_0.createdBy,
  AgentDrafts_0.modifiedAt,
  AgentDrafts_0.modifiedBy,
  AgentDrafts_0.name,
  AgentDrafts_0.description,
  AgentDrafts_0.agentType,
  AgentDrafts_0.status
FROM a2a_network_draft_AgentDrafts AS AgentDrafts_0;

CREATE VIEW A2ADraftService_ServiceDrafts AS SELECT
  ServiceDrafts_0.ID,
  ServiceDrafts_0.createdAt,
  ServiceDrafts_0.createdBy,
  ServiceDrafts_0.modifiedAt,
  ServiceDrafts_0.modifiedBy,
  ServiceDrafts_0.name,
  ServiceDrafts_0.description,
  ServiceDrafts_0.serviceType,
  ServiceDrafts_0.status
FROM a2a_network_draft_ServiceDrafts AS ServiceDrafts_0;

CREATE VIEW A2ADraftService_Workflows AS SELECT
  WorkflowDrafts_0.ID,
  WorkflowDrafts_0.createdAt,
  WorkflowDrafts_0.createdBy,
  WorkflowDrafts_0.modifiedAt,
  WorkflowDrafts_0.modifiedBy,
  WorkflowDrafts_0.name,
  WorkflowDrafts_0.description,
  WorkflowDrafts_0.workflowType,
  WorkflowDrafts_0.status
FROM a2a_network_draft_WorkflowDrafts AS WorkflowDrafts_0;

CREATE VIEW A2ADraftService_Requests AS SELECT
  RequestDrafts_0.ID,
  RequestDrafts_0.createdAt,
  RequestDrafts_0.createdBy,
  RequestDrafts_0.modifiedAt,
  RequestDrafts_0.modifiedBy,
  RequestDrafts_0.title,
  RequestDrafts_0.description,
  RequestDrafts_0.status,
  RequestDrafts_0.priority
FROM a2a_network_draft_RequestDrafts AS RequestDrafts_0;

CREATE VIEW A2ADraftService_AgentCapabilities AS SELECT
  AgentCapabilities_0.ID,
  AgentCapabilities_0.agent_ID,
  AgentCapabilities_0.capability_ID,
  AgentCapabilities_0.registeredAt,
  AgentCapabilities_0.version,
  AgentCapabilities_0.status
FROM a2a_network_AgentCapabilities AS AgentCapabilities_0;

CREATE VIEW A2ADraftService_AgentPerformance AS SELECT
  AgentPerformance_0.ID,
  AgentPerformance_0.agent_ID,
  AgentPerformance_0.totalTasks,
  AgentPerformance_0.successfulTasks,
  AgentPerformance_0.failedTasks,
  AgentPerformance_0.averageResponseTime,
  AgentPerformance_0.averageGasUsage,
  AgentPerformance_0.reputationScore,
  AgentPerformance_0.trustScore,
  AgentPerformance_0.lastUpdated
FROM a2a_network_AgentPerformance AS AgentPerformance_0;

CREATE VIEW A2ADraftService_ServiceOrders AS SELECT
  ServiceOrders_0.ID,
  ServiceOrders_0.createdAt,
  ServiceOrders_0.createdBy,
  ServiceOrders_0.modifiedAt,
  ServiceOrders_0.modifiedBy,
  ServiceOrders_0.service_ID,
  ServiceOrders_0.consumer_ID,
  ServiceOrders_0.status,
  ServiceOrders_0.callCount,
  ServiceOrders_0.totalAmount,
  ServiceOrders_0.escrowReleased,
  ServiceOrders_0.completedAt,
  ServiceOrders_0.rating,
  ServiceOrders_0.feedback
FROM a2a_network_ServiceOrders AS ServiceOrders_0;

CREATE VIEW A2ADraftService_WorkflowExecutions AS SELECT
  WorkflowExecutions_0.ID,
  WorkflowExecutions_0.createdAt,
  WorkflowExecutions_0.createdBy,
  WorkflowExecutions_0.modifiedAt,
  WorkflowExecutions_0.modifiedBy,
  WorkflowExecutions_0.workflow_ID,
  WorkflowExecutions_0.executionId,
  WorkflowExecutions_0.status,
  WorkflowExecutions_0.startedAt,
  WorkflowExecutions_0.completedAt,
  WorkflowExecutions_0.gasUsed,
  WorkflowExecutions_0.result,
  WorkflowExecutions_0.error
FROM a2a_network_WorkflowExecutions AS WorkflowExecutions_0;

CREATE VIEW A2ADraftService_Responses AS SELECT
  Responses_0.ID,
  Responses_0.createdAt,
  Responses_0.createdBy,
  Responses_0.modifiedAt,
  Responses_0.modifiedBy,
  Responses_0.content,
  Responses_0.responseType,
  Responses_0.status,
  Responses_0.request_ID,
  Responses_0.responder_ID,
  Responses_0.priority,
  Responses_0.isFinalResponse
FROM a2a_network_Responses AS Responses_0;

CREATE VIEW A2ADraftService_DraftConflicts AS SELECT
  DraftConflicts_0.ID,
  DraftConflicts_0.createdAt,
  DraftConflicts_0.createdBy,
  DraftConflicts_0.modifiedAt,
  DraftConflicts_0.modifiedBy,
  DraftConflicts_0.conflictType,
  DraftConflicts_0.conflictDescription,
  DraftConflicts_0.conflictStatus,
  DraftConflicts_0.resolutionStrategy,
  DraftConflicts_0.resolutionNotes,
  DraftConflicts_0.resolvedBy,
  DraftConflicts_0.resolvedAt
FROM a2a_network_draft_DraftConflicts AS DraftConflicts_0;

CREATE VIEW a2a_network_TopAgents AS SELECT
  Agents_0.ID,
  Agents_0.address,
  Agents_0.name,
  Agents_0.reputation,
  performance_1.successfulTasks AS completedTasks,
  performance_1.averageResponseTime AS avgResponseTime
FROM (a2a_network_Agents AS Agents_0 LEFT JOIN a2a_network_AgentPerformance AS performance_1 ON (performance_1.agent_ID = Agents_0.ID))
WHERE Agents_0.isActive = TRUE
ORDER BY reputation DESC;

CREATE VIEW a2a_network_ActiveServices AS SELECT
  Services_0.ID,
  Services_0.name,
  provider_1.name AS providerName,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.totalCalls,
  Services_0.averageRating
FROM (a2a_network_Services AS Services_0 LEFT JOIN a2a_network_Agents AS provider_1 ON Services_0.provider_ID = provider_1.ID)
WHERE Services_0.isActive = TRUE;

CREATE VIEW a2a_network_RecentWorkflows AS SELECT
  WorkflowExecutions_0.ID,
  workflow_1.name AS workflowName,
  WorkflowExecutions_0.status,
  WorkflowExecutions_0.startedAt,
  WorkflowExecutions_0.completedAt,
  WorkflowExecutions_0.gasUsed
FROM (a2a_network_WorkflowExecutions AS WorkflowExecutions_0 LEFT JOIN a2a_network_Workflows AS workflow_1 ON WorkflowExecutions_0.workflow_ID = workflow_1.ID)
ORDER BY startedAt DESC;

CREATE VIEW A2AService_Countries AS SELECT
  Countries_0.name,
  Countries_0.descr,
  Countries_0.code
FROM sap_common_Countries AS Countries_0;

CREATE VIEW A2AService_CapabilityCategories AS SELECT
  CapabilityCategories_0.name,
  CapabilityCategories_0.descr,
  CapabilityCategories_0.code,
  CapabilityCategories_0.description
FROM a2a_network_CapabilityCategories AS CapabilityCategories_0;

CREATE VIEW A2AService_StatusCodes AS SELECT
  StatusCodes_0.name,
  StatusCodes_0.descr,
  StatusCodes_0.code,
  StatusCodes_0.description
FROM a2a_network_StatusCodes AS StatusCodes_0;

CREATE VIEW A2AService_Capabilities_texts AS SELECT
  texts_0.locale,
  texts_0.ID,
  texts_0.name,
  texts_0.description
FROM a2a_network_Capabilities_texts AS texts_0;

CREATE VIEW A2AService_Currencies AS SELECT
  Currencies_0.name,
  Currencies_0.descr,
  Currencies_0.code,
  Currencies_0.symbol,
  Currencies_0.minorUnit
FROM sap_common_Currencies AS Currencies_0;

CREATE VIEW A2ADraftService_PerformanceSnapshots AS SELECT
  PerformanceSnapshots_0.ID,
  PerformanceSnapshots_0.performance_ID,
  PerformanceSnapshots_0.timestamp,
  PerformanceSnapshots_0.taskCount,
  PerformanceSnapshots_0.successRate,
  PerformanceSnapshots_0.responseTime,
  PerformanceSnapshots_0.gasUsage,
  PerformanceSnapshots_0.reputationDelta
FROM a2a_network_PerformanceSnapshots AS PerformanceSnapshots_0;

CREATE VIEW A2ADraftService_Services AS SELECT
  Services_0.ID,
  Services_0.createdAt,
  Services_0.createdBy,
  Services_0.modifiedAt,
  Services_0.modifiedBy,
  Services_0.provider_ID,
  Services_0.name,
  Services_0.description,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.currency_code,
  Services_0.minReputation,
  Services_0.maxCallsPerDay,
  Services_0.isActive,
  Services_0.totalCalls,
  Services_0.averageRating,
  Services_0.escrowAmount
FROM a2a_network_Services AS Services_0;

CREATE VIEW A2ADraftService_WorkflowSteps AS SELECT
  WorkflowSteps_0.ID,
  WorkflowSteps_0.execution_ID,
  WorkflowSteps_0.stepNumber,
  WorkflowSteps_0.agentAddress,
  WorkflowSteps_0."action",
  WorkflowSteps_0.input,
  WorkflowSteps_0.output,
  WorkflowSteps_0.status,
  WorkflowSteps_0.gasUsed,
  WorkflowSteps_0.startedAt,
  WorkflowSteps_0.completedAt
FROM a2a_network_WorkflowSteps AS WorkflowSteps_0;

CREATE VIEW A2AService_Countries_texts AS SELECT
  texts_0.locale,
  texts_0.name,
  texts_0.descr,
  texts_0.code
FROM sap_common_Countries_texts AS texts_0;

CREATE VIEW A2AService_CapabilityCategories_texts AS SELECT
  texts_0.locale,
  texts_0.name,
  texts_0.descr,
  texts_0.code,
  texts_0.description
FROM a2a_network_CapabilityCategories_texts AS texts_0;

CREATE VIEW A2AService_StatusCodes_texts AS SELECT
  texts_0.locale,
  texts_0.name,
  texts_0.descr,
  texts_0.code,
  texts_0.description
FROM a2a_network_StatusCodes_texts AS texts_0;

CREATE VIEW A2AService_Currencies_texts AS SELECT
  texts_0.locale,
  texts_0.name,
  texts_0.descr,
  texts_0.code
FROM sap_common_Currencies_texts AS texts_0;

CREATE VIEW A2ADraftService_Currencies AS SELECT
  Currencies_0.name,
  Currencies_0.descr,
  Currencies_0.code,
  Currencies_0.symbol,
  Currencies_0.minorUnit
FROM sap_common_Currencies AS Currencies_0;

CREATE VIEW A2ADraftService_Currencies_texts AS SELECT
  texts_0.locale,
  texts_0.name,
  texts_0.descr,
  texts_0.code
FROM sap_common_Currencies_texts AS texts_0;

CREATE VIEW localized_a2a_network_Agents AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.address,
  coalesce(localized_1.name, L_0.name) AS name,
  L_0.endpoint,
  L_0.reputation,
  L_0.isActive,
  L_0.country_code
FROM (a2a_network_Agents AS L_0 LEFT JOIN a2a_network_Agents_texts AS localized_1 ON localized_1.ID = L_0.ID AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_Capabilities AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.description, L_0.description) AS description,
  L_0.category_code,
  L_0.tags,
  L_0.inputTypes,
  L_0.outputTypes,
  L_0.version,
  L_0.status_code,
  L_0.dependencies,
  L_0.conflicts
FROM (a2a_network_Capabilities AS L_0 LEFT JOIN a2a_network_Capabilities_texts AS localized_1 ON localized_1.ID = L_0.ID AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_CapabilityCategories AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_1.description, L_0.description) AS description
FROM (a2a_network_CapabilityCategories AS L_0 LEFT JOIN a2a_network_CapabilityCategories_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_StatusCodes AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_1.description, L_0.description) AS description
FROM (a2a_network_StatusCodes AS L_0 LEFT JOIN a2a_network_StatusCodes_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_FeatureToggles AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.feature,
  L_0.enabled,
  coalesce(localized_1.description, L_0.description) AS description,
  L_0.validFrom,
  L_0.validTo,
  L_0.tenant
FROM (a2a_network_FeatureToggles AS L_0 LEFT JOIN a2a_network_FeatureToggles_texts AS localized_1 ON localized_1.feature = L_0.feature AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_ExtensionFields AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.entity,
  L_0.field,
  L_0.tenant,
  L_0.dataType,
  coalesce(localized_1.label, L_0.label) AS label,
  L_0.defaultValue,
  L_0.mandatory,
  L_0.visible
FROM (a2a_network_ExtensionFields AS L_0 LEFT JOIN a2a_network_ExtensionFields_texts AS localized_1 ON localized_1.entity = L_0.entity AND localized_1.field = L_0.field AND localized_1.tenant = L_0.tenant AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_sap_common_Countries AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code
FROM (sap_common_Countries AS L_0 LEFT JOIN sap_common_Countries_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_sap_common_Currencies AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  L_0.symbol,
  L_0.minorUnit
FROM (sap_common_Currencies AS L_0 LEFT JOIN sap_common_Currencies_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = session_context( '$user.locale' ));

CREATE VIEW localized_a2a_network_AgentCapabilities AS SELECT
  L.ID,
  L.agent_ID,
  L.capability_ID,
  L.registeredAt,
  L.version,
  L.status
FROM a2a_network_AgentCapabilities AS L;

CREATE VIEW localized_a2a_network_Services AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.provider_ID,
  L.name,
  L.description,
  L.category,
  L.pricePerCall,
  L.currency_code,
  L.minReputation,
  L.maxCallsPerDay,
  L.isActive,
  L.totalCalls,
  L.averageRating,
  L.escrowAmount
FROM a2a_network_Services AS L;

CREATE VIEW localized_a2a_network_ServiceOrders AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.service_ID,
  L.consumer_ID,
  L.status,
  L.callCount,
  L.totalAmount,
  L.escrowReleased,
  L.completedAt,
  L.rating,
  L.feedback
FROM a2a_network_ServiceOrders AS L;

CREATE VIEW localized_a2a_network_Messages AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.sender_ID,
  L.recipient_ID,
  L.messageHash,
  L.protocol,
  L.priority,
  L.status,
  L.retryCount,
  L.gasUsed,
  L.deliveredAt
FROM a2a_network_Messages AS L;

CREATE VIEW localized_a2a_network_AgentPerformance AS SELECT
  L.ID,
  L.agent_ID,
  L.totalTasks,
  L.successfulTasks,
  L.failedTasks,
  L.averageResponseTime,
  L.averageGasUsage,
  L.reputationScore,
  L.trustScore,
  L.lastUpdated
FROM a2a_network_AgentPerformance AS L;

CREATE VIEW localized_a2a_network_Workflows AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.name,
  L.description,
  L.definition,
  L.isActive,
  L.category,
  L.owner_ID
FROM a2a_network_Workflows AS L;

CREATE VIEW localized_a2a_network_Requests AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.title,
  L.description,
  L.requestType,
  L.priority,
  L.status,
  L.requester_ID,
  L.assignedAgent_ID,
  L.dueDate
FROM a2a_network_Requests AS L;

CREATE VIEW localized_a2a_network_Responses AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.content,
  L.responseType,
  L.status,
  L.request_ID,
  L.responder_ID,
  L.priority,
  L.isFinalResponse
FROM a2a_network_Responses AS L;

CREATE VIEW localized_a2a_network_ReputationTransactions AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.agent_ID,
  L.transactionType,
  L.amount,
  L.reason,
  L.context,
  L.isAutomated,
  L.createdByAgent_ID,
  L.serviceOrder_ID,
  L.workflow_ID
FROM a2a_network_ReputationTransactions AS L;

CREATE VIEW localized_a2a_network_PeerEndorsements AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.fromAgent_ID,
  L.toAgent_ID,
  L.amount,
  L.reason,
  L.context,
  L.workflow_ID,
  L.serviceOrder_ID,
  L.expiresAt,
  L.isReciprocal,
  L.verificationHash,
  L.blockchainTx
FROM a2a_network_PeerEndorsements AS L;

CREATE VIEW localized_a2a_network_ReputationMilestones AS SELECT
  L.ID,
  L.agent_ID,
  L.milestone,
  L.badgeName,
  L.achievedAt,
  L.badgeMetadata
FROM a2a_network_ReputationMilestones AS L;

CREATE VIEW localized_a2a_network_ReputationRecovery AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.agent_ID,
  L.recoveryType,
  L.status,
  L.requirements,
  L.progress,
  L.reputationReward,
  L.startedAt,
  L.completedAt
FROM a2a_network_ReputationRecovery AS L;

CREATE VIEW localized_a2a_network_DailyReputationLimits AS SELECT
  L.ID,
  L.agent_ID,
  L.date,
  L.endorsementsGiven,
  L.pointsGiven,
  L.maxDailyLimit
FROM a2a_network_DailyReputationLimits AS L;

CREATE VIEW localized_a2a_network_ReputationAnalytics AS SELECT
  L.ID,
  L.agent_ID,
  L.periodStart,
  L.periodEnd,
  L.startingReputation,
  L.endingReputation,
  L.totalEarned,
  L.totalLost,
  L.endorsementsReceived,
  L.endorsementsGiven,
  L.uniqueEndorsers,
  L.averageTransaction,
  L.taskSuccessRate,
  L.serviceRatingAverage
FROM a2a_network_ReputationAnalytics AS L;

CREATE VIEW localized_A2ADraftService_Agents AS SELECT
  AgentDrafts_0.ID,
  AgentDrafts_0.createdAt,
  AgentDrafts_0.createdBy,
  AgentDrafts_0.modifiedAt,
  AgentDrafts_0.modifiedBy,
  AgentDrafts_0.name,
  AgentDrafts_0.description,
  AgentDrafts_0.agentType,
  AgentDrafts_0.status
FROM a2a_network_draft_AgentDrafts AS AgentDrafts_0;

CREATE VIEW localized_A2ADraftService_ServiceDrafts AS SELECT
  ServiceDrafts_0.ID,
  ServiceDrafts_0.createdAt,
  ServiceDrafts_0.createdBy,
  ServiceDrafts_0.modifiedAt,
  ServiceDrafts_0.modifiedBy,
  ServiceDrafts_0.name,
  ServiceDrafts_0.description,
  ServiceDrafts_0.serviceType,
  ServiceDrafts_0.status
FROM a2a_network_draft_ServiceDrafts AS ServiceDrafts_0;

CREATE VIEW localized_A2ADraftService_Requests AS SELECT
  RequestDrafts_0.ID,
  RequestDrafts_0.createdAt,
  RequestDrafts_0.createdBy,
  RequestDrafts_0.modifiedAt,
  RequestDrafts_0.modifiedBy,
  RequestDrafts_0.title,
  RequestDrafts_0.description,
  RequestDrafts_0.status,
  RequestDrafts_0.priority
FROM a2a_network_draft_RequestDrafts AS RequestDrafts_0;

CREATE VIEW localized_a2a_network_PerformanceSnapshots AS SELECT
  L.ID,
  L.performance_ID,
  L.timestamp,
  L.taskCount,
  L.successRate,
  L.responseTime,
  L.gasUsage,
  L.reputationDelta
FROM a2a_network_PerformanceSnapshots AS L;

CREATE VIEW localized_a2a_network_WorkflowExecutions AS SELECT
  L.ID,
  L.createdAt,
  L.createdBy,
  L.modifiedAt,
  L.modifiedBy,
  L.workflow_ID,
  L.executionId,
  L.status,
  L.startedAt,
  L.completedAt,
  L.gasUsed,
  L.result,
  L.error
FROM a2a_network_WorkflowExecutions AS L;

CREATE VIEW localized_A2ADraftService_Workflows AS SELECT
  WorkflowDrafts_0.ID,
  WorkflowDrafts_0.createdAt,
  WorkflowDrafts_0.createdBy,
  WorkflowDrafts_0.modifiedAt,
  WorkflowDrafts_0.modifiedBy,
  WorkflowDrafts_0.name,
  WorkflowDrafts_0.description,
  WorkflowDrafts_0.workflowType,
  WorkflowDrafts_0.status
FROM a2a_network_draft_WorkflowDrafts AS WorkflowDrafts_0;

CREATE VIEW localized_a2a_network_WorkflowSteps AS SELECT
  L.ID,
  L.execution_ID,
  L.stepNumber,
  L.agentAddress,
  L."action",
  L.input,
  L.output,
  L.status,
  L.gasUsed,
  L.startedAt,
  L.completedAt
FROM a2a_network_WorkflowSteps AS L;

CREATE VIEW A2ADraftService_DraftAdministrativeData AS SELECT
  DraftAdministrativeData.DraftUUID,
  DraftAdministrativeData.CreationDateTime,
  DraftAdministrativeData.CreatedByUser,
  DraftAdministrativeData.DraftIsCreatedByMe,
  DraftAdministrativeData.LastChangeDateTime,
  DraftAdministrativeData.LastChangedByUser,
  DraftAdministrativeData.InProcessByUser,
  DraftAdministrativeData.DraftIsProcessedByMe
FROM DRAFT_DraftAdministrativeData AS DraftAdministrativeData;

CREATE VIEW A2AService_TopAgents AS SELECT
  TopAgents_0.ID,
  TopAgents_0.address,
  TopAgents_0.name,
  TopAgents_0.reputation,
  TopAgents_0.completedTasks,
  TopAgents_0.avgResponseTime
FROM a2a_network_TopAgents AS TopAgents_0;

CREATE VIEW A2AService_ActiveServices AS SELECT
  ActiveServices_0.ID,
  ActiveServices_0.name,
  ActiveServices_0.providerName,
  ActiveServices_0.category,
  ActiveServices_0.pricePerCall,
  ActiveServices_0.totalCalls,
  ActiveServices_0.averageRating
FROM a2a_network_ActiveServices AS ActiveServices_0;

CREATE VIEW A2AService_RecentWorkflows AS SELECT
  RecentWorkflows_0.ID,
  RecentWorkflows_0.workflowName,
  RecentWorkflows_0.status,
  RecentWorkflows_0.startedAt,
  RecentWorkflows_0.completedAt,
  RecentWorkflows_0.gasUsed
FROM a2a_network_RecentWorkflows AS RecentWorkflows_0;

CREATE VIEW localized_A2AService_Agents AS SELECT
  Agents_0.ID,
  Agents_0.createdAt,
  Agents_0.createdBy,
  Agents_0.modifiedAt,
  Agents_0.modifiedBy,
  Agents_0.address,
  Agents_0.name,
  Agents_0.endpoint,
  Agents_0.reputation,
  Agents_0.isActive,
  Agents_0.country_code
FROM localized_a2a_network_Agents AS Agents_0;

CREATE VIEW localized_A2AService_Capabilities AS SELECT
  Capabilities_0.ID,
  Capabilities_0.createdAt,
  Capabilities_0.createdBy,
  Capabilities_0.modifiedAt,
  Capabilities_0.modifiedBy,
  Capabilities_0.name,
  Capabilities_0.description,
  Capabilities_0.category_code,
  Capabilities_0.tags,
  Capabilities_0.inputTypes,
  Capabilities_0.outputTypes,
  Capabilities_0.version,
  Capabilities_0.status_code,
  Capabilities_0.dependencies,
  Capabilities_0.conflicts
FROM localized_a2a_network_Capabilities AS Capabilities_0;

CREATE VIEW localized_A2ADraftService_AgentCapabilities AS SELECT
  AgentCapabilities_0.ID,
  AgentCapabilities_0.agent_ID,
  AgentCapabilities_0.capability_ID,
  AgentCapabilities_0.registeredAt,
  AgentCapabilities_0.version,
  AgentCapabilities_0.status
FROM localized_a2a_network_AgentCapabilities AS AgentCapabilities_0;

CREATE VIEW localized_A2ADraftService_AgentPerformance AS SELECT
  AgentPerformance_0.ID,
  AgentPerformance_0.agent_ID,
  AgentPerformance_0.totalTasks,
  AgentPerformance_0.successfulTasks,
  AgentPerformance_0.failedTasks,
  AgentPerformance_0.averageResponseTime,
  AgentPerformance_0.averageGasUsage,
  AgentPerformance_0.reputationScore,
  AgentPerformance_0.trustScore,
  AgentPerformance_0.lastUpdated
FROM localized_a2a_network_AgentPerformance AS AgentPerformance_0;

CREATE VIEW localized_A2ADraftService_ServiceOrders AS SELECT
  ServiceOrders_0.ID,
  ServiceOrders_0.createdAt,
  ServiceOrders_0.createdBy,
  ServiceOrders_0.modifiedAt,
  ServiceOrders_0.modifiedBy,
  ServiceOrders_0.service_ID,
  ServiceOrders_0.consumer_ID,
  ServiceOrders_0.status,
  ServiceOrders_0.callCount,
  ServiceOrders_0.totalAmount,
  ServiceOrders_0.escrowReleased,
  ServiceOrders_0.completedAt,
  ServiceOrders_0.rating,
  ServiceOrders_0.feedback
FROM localized_a2a_network_ServiceOrders AS ServiceOrders_0;

CREATE VIEW localized_A2ADraftService_Responses AS SELECT
  Responses_0.ID,
  Responses_0.createdAt,
  Responses_0.createdBy,
  Responses_0.modifiedAt,
  Responses_0.modifiedBy,
  Responses_0.content,
  Responses_0.responseType,
  Responses_0.status,
  Responses_0.request_ID,
  Responses_0.responder_ID,
  Responses_0.priority,
  Responses_0.isFinalResponse
FROM localized_a2a_network_Responses AS Responses_0;

CREATE VIEW localized_a2a_network_TopAgents AS SELECT
  Agents_0.ID,
  Agents_0.address,
  Agents_0.name,
  Agents_0.reputation,
  performance_1.successfulTasks AS completedTasks,
  performance_1.averageResponseTime AS avgResponseTime
FROM (localized_a2a_network_Agents AS Agents_0 LEFT JOIN localized_a2a_network_AgentPerformance AS performance_1 ON (performance_1.agent_ID = Agents_0.ID))
WHERE Agents_0.isActive = TRUE
ORDER BY reputation DESC;

CREATE VIEW localized_a2a_network_ActiveServices AS SELECT
  Services_0.ID,
  Services_0.name,
  provider_1.name AS providerName,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.totalCalls,
  Services_0.averageRating
FROM (localized_a2a_network_Services AS Services_0 LEFT JOIN localized_a2a_network_Agents AS provider_1 ON Services_0.provider_ID = provider_1.ID)
WHERE Services_0.isActive = TRUE;

CREATE VIEW localized_A2AService_Countries AS SELECT
  Countries_0.name,
  Countries_0.descr,
  Countries_0.code
FROM localized_sap_common_Countries AS Countries_0;

CREATE VIEW localized_A2AService_CapabilityCategories AS SELECT
  CapabilityCategories_0.name,
  CapabilityCategories_0.descr,
  CapabilityCategories_0.code,
  CapabilityCategories_0.description
FROM localized_a2a_network_CapabilityCategories AS CapabilityCategories_0;

CREATE VIEW localized_A2AService_StatusCodes AS SELECT
  StatusCodes_0.name,
  StatusCodes_0.descr,
  StatusCodes_0.code,
  StatusCodes_0.description
FROM localized_a2a_network_StatusCodes AS StatusCodes_0;

CREATE VIEW localized_A2AService_Currencies AS SELECT
  Currencies_0.name,
  Currencies_0.descr,
  Currencies_0.code,
  Currencies_0.symbol,
  Currencies_0.minorUnit
FROM localized_sap_common_Currencies AS Currencies_0;

CREATE VIEW localized_A2ADraftService_Services AS SELECT
  Services_0.ID,
  Services_0.createdAt,
  Services_0.createdBy,
  Services_0.modifiedAt,
  Services_0.modifiedBy,
  Services_0.provider_ID,
  Services_0.name,
  Services_0.description,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.currency_code,
  Services_0.minReputation,
  Services_0.maxCallsPerDay,
  Services_0.isActive,
  Services_0.totalCalls,
  Services_0.averageRating,
  Services_0.escrowAmount
FROM localized_a2a_network_Services AS Services_0;

CREATE VIEW localized_A2ADraftService_Currencies AS SELECT
  Currencies_0.name,
  Currencies_0.descr,
  Currencies_0.code,
  Currencies_0.symbol,
  Currencies_0.minorUnit
FROM localized_sap_common_Currencies AS Currencies_0;

CREATE VIEW localized_A2AService_AgentCapabilities AS SELECT
  AgentCapabilities_0.ID,
  AgentCapabilities_0.agent_ID,
  AgentCapabilities_0.capability_ID,
  AgentCapabilities_0.registeredAt,
  AgentCapabilities_0.version,
  AgentCapabilities_0.status
FROM localized_a2a_network_AgentCapabilities AS AgentCapabilities_0;

CREATE VIEW localized_A2AService_AgentPerformance AS SELECT
  AgentPerformance_0.ID,
  AgentPerformance_0.agent_ID,
  AgentPerformance_0.totalTasks,
  AgentPerformance_0.successfulTasks,
  AgentPerformance_0.failedTasks,
  AgentPerformance_0.averageResponseTime,
  AgentPerformance_0.averageGasUsage,
  AgentPerformance_0.reputationScore,
  AgentPerformance_0.trustScore,
  AgentPerformance_0.lastUpdated
FROM localized_a2a_network_AgentPerformance AS AgentPerformance_0;

CREATE VIEW localized_A2AService_Services AS SELECT
  Services_0.ID,
  Services_0.createdAt,
  Services_0.createdBy,
  Services_0.modifiedAt,
  Services_0.modifiedBy,
  Services_0.provider_ID,
  Services_0.name,
  Services_0.description,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.currency_code,
  Services_0.minReputation,
  Services_0.maxCallsPerDay,
  Services_0.isActive,
  Services_0.totalCalls,
  Services_0.averageRating,
  Services_0.escrowAmount
FROM localized_a2a_network_Services AS Services_0;

CREATE VIEW localized_A2AService_ServiceOrders AS SELECT
  ServiceOrders_0.ID,
  ServiceOrders_0.createdAt,
  ServiceOrders_0.createdBy,
  ServiceOrders_0.modifiedAt,
  ServiceOrders_0.modifiedBy,
  ServiceOrders_0.service_ID,
  ServiceOrders_0.consumer_ID,
  ServiceOrders_0.status,
  ServiceOrders_0.callCount,
  ServiceOrders_0.totalAmount,
  ServiceOrders_0.escrowReleased,
  ServiceOrders_0.completedAt,
  ServiceOrders_0.rating,
  ServiceOrders_0.feedback
FROM localized_a2a_network_ServiceOrders AS ServiceOrders_0;

CREATE VIEW localized_A2AService_Messages AS SELECT
  Messages_0.ID,
  Messages_0.createdAt,
  Messages_0.createdBy,
  Messages_0.modifiedAt,
  Messages_0.modifiedBy,
  Messages_0.sender_ID,
  Messages_0.recipient_ID,
  Messages_0.messageHash,
  Messages_0.protocol,
  Messages_0.priority,
  Messages_0.status,
  Messages_0.retryCount,
  Messages_0.gasUsed,
  Messages_0.deliveredAt
FROM localized_a2a_network_Messages AS Messages_0;

CREATE VIEW localized_A2AService_Workflows AS SELECT
  Workflows_0.ID,
  Workflows_0.createdAt,
  Workflows_0.createdBy,
  Workflows_0.modifiedAt,
  Workflows_0.modifiedBy,
  Workflows_0.name,
  Workflows_0.description,
  Workflows_0.definition,
  Workflows_0.isActive,
  Workflows_0.category,
  Workflows_0.owner_ID
FROM localized_a2a_network_Workflows AS Workflows_0;

CREATE VIEW localized_A2AService_ReputationTransactions AS SELECT
  ReputationTransactions_0.ID,
  ReputationTransactions_0.createdAt,
  ReputationTransactions_0.createdBy,
  ReputationTransactions_0.modifiedAt,
  ReputationTransactions_0.modifiedBy,
  ReputationTransactions_0.agent_ID,
  ReputationTransactions_0.transactionType,
  ReputationTransactions_0.amount,
  ReputationTransactions_0.reason,
  ReputationTransactions_0.context,
  ReputationTransactions_0.isAutomated,
  ReputationTransactions_0.createdByAgent_ID,
  ReputationTransactions_0.serviceOrder_ID,
  ReputationTransactions_0.workflow_ID
FROM localized_a2a_network_ReputationTransactions AS ReputationTransactions_0;

CREATE VIEW localized_A2AService_PeerEndorsements AS SELECT
  PeerEndorsements_0.ID,
  PeerEndorsements_0.createdAt,
  PeerEndorsements_0.createdBy,
  PeerEndorsements_0.modifiedAt,
  PeerEndorsements_0.modifiedBy,
  PeerEndorsements_0.fromAgent_ID,
  PeerEndorsements_0.toAgent_ID,
  PeerEndorsements_0.amount,
  PeerEndorsements_0.reason,
  PeerEndorsements_0.context,
  PeerEndorsements_0.workflow_ID,
  PeerEndorsements_0.serviceOrder_ID,
  PeerEndorsements_0.expiresAt,
  PeerEndorsements_0.isReciprocal,
  PeerEndorsements_0.verificationHash,
  PeerEndorsements_0.blockchainTx
FROM localized_a2a_network_PeerEndorsements AS PeerEndorsements_0;

CREATE VIEW localized_A2AService_ReputationMilestones AS SELECT
  ReputationMilestones_0.ID,
  ReputationMilestones_0.agent_ID,
  ReputationMilestones_0.milestone,
  ReputationMilestones_0.badgeName,
  ReputationMilestones_0.achievedAt,
  ReputationMilestones_0.badgeMetadata
FROM localized_a2a_network_ReputationMilestones AS ReputationMilestones_0;

CREATE VIEW localized_A2AService_ReputationRecovery AS SELECT
  ReputationRecovery_0.ID,
  ReputationRecovery_0.createdAt,
  ReputationRecovery_0.createdBy,
  ReputationRecovery_0.modifiedAt,
  ReputationRecovery_0.modifiedBy,
  ReputationRecovery_0.agent_ID,
  ReputationRecovery_0.recoveryType,
  ReputationRecovery_0.status,
  ReputationRecovery_0.requirements,
  ReputationRecovery_0.progress,
  ReputationRecovery_0.reputationReward,
  ReputationRecovery_0.startedAt,
  ReputationRecovery_0.completedAt
FROM localized_a2a_network_ReputationRecovery AS ReputationRecovery_0;

CREATE VIEW localized_A2AService_DailyReputationLimits AS SELECT
  DailyReputationLimits_0.ID,
  DailyReputationLimits_0.agent_ID,
  DailyReputationLimits_0.date,
  DailyReputationLimits_0.endorsementsGiven,
  DailyReputationLimits_0.pointsGiven,
  DailyReputationLimits_0.maxDailyLimit
FROM localized_a2a_network_DailyReputationLimits AS DailyReputationLimits_0;

CREATE VIEW localized_A2AService_ReputationAnalytics AS SELECT
  ReputationAnalytics_0.ID,
  ReputationAnalytics_0.agent_ID,
  ReputationAnalytics_0.periodStart,
  ReputationAnalytics_0.periodEnd,
  ReputationAnalytics_0.startingReputation,
  ReputationAnalytics_0.endingReputation,
  ReputationAnalytics_0.totalEarned,
  ReputationAnalytics_0.totalLost,
  ReputationAnalytics_0.endorsementsReceived,
  ReputationAnalytics_0.endorsementsGiven,
  ReputationAnalytics_0.uniqueEndorsers,
  ReputationAnalytics_0.averageTransaction,
  ReputationAnalytics_0.taskSuccessRate,
  ReputationAnalytics_0.serviceRatingAverage
FROM localized_a2a_network_ReputationAnalytics AS ReputationAnalytics_0;

CREATE VIEW localized_A2ADraftService_PerformanceSnapshots AS SELECT
  PerformanceSnapshots_0.ID,
  PerformanceSnapshots_0.performance_ID,
  PerformanceSnapshots_0.timestamp,
  PerformanceSnapshots_0.taskCount,
  PerformanceSnapshots_0.successRate,
  PerformanceSnapshots_0.responseTime,
  PerformanceSnapshots_0.gasUsage,
  PerformanceSnapshots_0.reputationDelta
FROM localized_a2a_network_PerformanceSnapshots AS PerformanceSnapshots_0;

CREATE VIEW localized_A2ADraftService_WorkflowExecutions AS SELECT
  WorkflowExecutions_0.ID,
  WorkflowExecutions_0.createdAt,
  WorkflowExecutions_0.createdBy,
  WorkflowExecutions_0.modifiedAt,
  WorkflowExecutions_0.modifiedBy,
  WorkflowExecutions_0.workflow_ID,
  WorkflowExecutions_0.executionId,
  WorkflowExecutions_0.status,
  WorkflowExecutions_0.startedAt,
  WorkflowExecutions_0.completedAt,
  WorkflowExecutions_0.gasUsed,
  WorkflowExecutions_0.result,
  WorkflowExecutions_0.error
FROM localized_a2a_network_WorkflowExecutions AS WorkflowExecutions_0;

CREATE VIEW localized_A2AService_PerformanceSnapshots AS SELECT
  PerformanceSnapshots_0.ID,
  PerformanceSnapshots_0.performance_ID,
  PerformanceSnapshots_0.timestamp,
  PerformanceSnapshots_0.taskCount,
  PerformanceSnapshots_0.successRate,
  PerformanceSnapshots_0.responseTime,
  PerformanceSnapshots_0.gasUsage,
  PerformanceSnapshots_0.reputationDelta
FROM localized_a2a_network_PerformanceSnapshots AS PerformanceSnapshots_0;

CREATE VIEW localized_A2AService_WorkflowExecutions AS SELECT
  WorkflowExecutions_0.ID,
  WorkflowExecutions_0.createdAt,
  WorkflowExecutions_0.createdBy,
  WorkflowExecutions_0.modifiedAt,
  WorkflowExecutions_0.modifiedBy,
  WorkflowExecutions_0.workflow_ID,
  WorkflowExecutions_0.executionId,
  WorkflowExecutions_0.status,
  WorkflowExecutions_0.startedAt,
  WorkflowExecutions_0.completedAt,
  WorkflowExecutions_0.gasUsed,
  WorkflowExecutions_0.result,
  WorkflowExecutions_0.error
FROM localized_a2a_network_WorkflowExecutions AS WorkflowExecutions_0;

CREATE VIEW localized_A2ADraftService_WorkflowSteps AS SELECT
  WorkflowSteps_0.ID,
  WorkflowSteps_0.execution_ID,
  WorkflowSteps_0.stepNumber,
  WorkflowSteps_0.agentAddress,
  WorkflowSteps_0."action",
  WorkflowSteps_0.input,
  WorkflowSteps_0.output,
  WorkflowSteps_0.status,
  WorkflowSteps_0.gasUsed,
  WorkflowSteps_0.startedAt,
  WorkflowSteps_0.completedAt
FROM localized_a2a_network_WorkflowSteps AS WorkflowSteps_0;

CREATE VIEW localized_A2AService_WorkflowSteps AS SELECT
  WorkflowSteps_0.ID,
  WorkflowSteps_0.execution_ID,
  WorkflowSteps_0.stepNumber,
  WorkflowSteps_0.agentAddress,
  WorkflowSteps_0."action",
  WorkflowSteps_0.input,
  WorkflowSteps_0.output,
  WorkflowSteps_0.status,
  WorkflowSteps_0.gasUsed,
  WorkflowSteps_0.startedAt,
  WorkflowSteps_0.completedAt
FROM localized_a2a_network_WorkflowSteps AS WorkflowSteps_0;

CREATE VIEW localized_A2AService_TopAgents AS SELECT
  TopAgents_0.ID,
  TopAgents_0.address,
  TopAgents_0.name,
  TopAgents_0.reputation,
  TopAgents_0.completedTasks,
  TopAgents_0.avgResponseTime
FROM localized_a2a_network_TopAgents AS TopAgents_0;

CREATE VIEW localized_A2AService_ActiveServices AS SELECT
  ActiveServices_0.ID,
  ActiveServices_0.name,
  ActiveServices_0.providerName,
  ActiveServices_0.category,
  ActiveServices_0.pricePerCall,
  ActiveServices_0.totalCalls,
  ActiveServices_0.averageRating
FROM localized_a2a_network_ActiveServices AS ActiveServices_0;


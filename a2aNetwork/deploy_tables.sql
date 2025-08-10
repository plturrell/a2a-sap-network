
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
FROM (a2a_network_Agents AS L_0 LEFT JOIN a2a_network_Agents_texts AS localized_1 ON localized_1.ID = L_0.ID AND localized_1.locale = 'en');

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
FROM (a2a_network_Capabilities AS L_0 LEFT JOIN a2a_network_Capabilities_texts AS localized_1 ON localized_1.ID = L_0.ID AND localized_1.locale = 'en');

CREATE VIEW localized_a2a_network_CapabilityCategories AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_1.description, L_0.description) AS description
FROM (a2a_network_CapabilityCategories AS L_0 LEFT JOIN a2a_network_CapabilityCategories_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = 'en');

CREATE VIEW localized_a2a_network_StatusCodes AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_1.description, L_0.description) AS description
FROM (a2a_network_StatusCodes AS L_0 LEFT JOIN a2a_network_StatusCodes_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = 'en');

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
FROM (a2a_network_FeatureToggles AS L_0 LEFT JOIN a2a_network_FeatureToggles_texts AS localized_1 ON localized_1.feature = L_0.feature AND localized_1.locale = 'en');

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
FROM (a2a_network_ExtensionFields AS L_0 LEFT JOIN a2a_network_ExtensionFields_texts AS localized_1 ON localized_1.entity = L_0.entity AND localized_1.field = L_0.field AND localized_1.tenant = L_0.tenant AND localized_1.locale = 'en');

CREATE VIEW localized_sap_common_Countries AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code
FROM (sap_common_Countries AS L_0 LEFT JOIN sap_common_Countries_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = 'en');

CREATE VIEW localized_sap_common_Currencies AS SELECT
  coalesce(localized_1.name, L_0.name) AS name,
  coalesce(localized_1.descr, L_0.descr) AS descr,
  L_0.code,
  L_0.symbol,
  L_0.minorUnit
FROM (sap_common_Currencies AS L_0 LEFT JOIN sap_common_Currencies_texts AS localized_1 ON localized_1.code = L_0.code AND localized_1.locale = 'en');

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

CREATE VIEW localized_de_a2a_network_Agents AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.address,
  coalesce(localized_de_1.name, L_0.name) AS name,
  L_0.endpoint,
  L_0.reputation,
  L_0.isActive,
  L_0.country_code
FROM (a2a_network_Agents AS L_0 LEFT JOIN a2a_network_Agents_texts AS localized_de_1 ON localized_de_1.ID = L_0.ID AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_Agents AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.address,
  coalesce(localized_fr_1.name, L_0.name) AS name,
  L_0.endpoint,
  L_0.reputation,
  L_0.isActive,
  L_0.country_code
FROM (a2a_network_Agents AS L_0 LEFT JOIN a2a_network_Agents_texts AS localized_fr_1 ON localized_fr_1.ID = L_0.ID AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_Capabilities AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  coalesce(localized_de_1.name, L_0.name) AS name,
  coalesce(localized_de_1.description, L_0.description) AS description,
  L_0.category_code,
  L_0.tags,
  L_0.inputTypes,
  L_0.outputTypes,
  L_0.version,
  L_0.status_code,
  L_0.dependencies,
  L_0.conflicts
FROM (a2a_network_Capabilities AS L_0 LEFT JOIN a2a_network_Capabilities_texts AS localized_de_1 ON localized_de_1.ID = L_0.ID AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_Capabilities AS SELECT
  L_0.ID,
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  coalesce(localized_fr_1.name, L_0.name) AS name,
  coalesce(localized_fr_1.description, L_0.description) AS description,
  L_0.category_code,
  L_0.tags,
  L_0.inputTypes,
  L_0.outputTypes,
  L_0.version,
  L_0.status_code,
  L_0.dependencies,
  L_0.conflicts
FROM (a2a_network_Capabilities AS L_0 LEFT JOIN a2a_network_Capabilities_texts AS localized_fr_1 ON localized_fr_1.ID = L_0.ID AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_CapabilityCategories AS SELECT
  coalesce(localized_de_1.name, L_0.name) AS name,
  coalesce(localized_de_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_de_1.description, L_0.description) AS description
FROM (a2a_network_CapabilityCategories AS L_0 LEFT JOIN a2a_network_CapabilityCategories_texts AS localized_de_1 ON localized_de_1.code = L_0.code AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_CapabilityCategories AS SELECT
  coalesce(localized_fr_1.name, L_0.name) AS name,
  coalesce(localized_fr_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_fr_1.description, L_0.description) AS description
FROM (a2a_network_CapabilityCategories AS L_0 LEFT JOIN a2a_network_CapabilityCategories_texts AS localized_fr_1 ON localized_fr_1.code = L_0.code AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_StatusCodes AS SELECT
  coalesce(localized_de_1.name, L_0.name) AS name,
  coalesce(localized_de_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_de_1.description, L_0.description) AS description
FROM (a2a_network_StatusCodes AS L_0 LEFT JOIN a2a_network_StatusCodes_texts AS localized_de_1 ON localized_de_1.code = L_0.code AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_StatusCodes AS SELECT
  coalesce(localized_fr_1.name, L_0.name) AS name,
  coalesce(localized_fr_1.descr, L_0.descr) AS descr,
  L_0.code,
  coalesce(localized_fr_1.description, L_0.description) AS description
FROM (a2a_network_StatusCodes AS L_0 LEFT JOIN a2a_network_StatusCodes_texts AS localized_fr_1 ON localized_fr_1.code = L_0.code AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_FeatureToggles AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.feature,
  L_0.enabled,
  coalesce(localized_de_1.description, L_0.description) AS description,
  L_0.validFrom,
  L_0.validTo,
  L_0.tenant
FROM (a2a_network_FeatureToggles AS L_0 LEFT JOIN a2a_network_FeatureToggles_texts AS localized_de_1 ON localized_de_1.feature = L_0.feature AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_FeatureToggles AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.feature,
  L_0.enabled,
  coalesce(localized_fr_1.description, L_0.description) AS description,
  L_0.validFrom,
  L_0.validTo,
  L_0.tenant
FROM (a2a_network_FeatureToggles AS L_0 LEFT JOIN a2a_network_FeatureToggles_texts AS localized_fr_1 ON localized_fr_1.feature = L_0.feature AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_ExtensionFields AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.entity,
  L_0.field,
  L_0.tenant,
  L_0.dataType,
  coalesce(localized_de_1.label, L_0.label) AS label,
  L_0.defaultValue,
  L_0.mandatory,
  L_0.visible
FROM (a2a_network_ExtensionFields AS L_0 LEFT JOIN a2a_network_ExtensionFields_texts AS localized_de_1 ON localized_de_1.entity = L_0.entity AND localized_de_1.field = L_0.field AND localized_de_1.tenant = L_0.tenant AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_a2a_network_ExtensionFields AS SELECT
  L_0.createdAt,
  L_0.createdBy,
  L_0.modifiedAt,
  L_0.modifiedBy,
  L_0.entity,
  L_0.field,
  L_0.tenant,
  L_0.dataType,
  coalesce(localized_fr_1.label, L_0.label) AS label,
  L_0.defaultValue,
  L_0.mandatory,
  L_0.visible
FROM (a2a_network_ExtensionFields AS L_0 LEFT JOIN a2a_network_ExtensionFields_texts AS localized_fr_1 ON localized_fr_1.entity = L_0.entity AND localized_fr_1.field = L_0.field AND localized_fr_1.tenant = L_0.tenant AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_sap_common_Countries AS SELECT
  coalesce(localized_de_1.name, L_0.name) AS name,
  coalesce(localized_de_1.descr, L_0.descr) AS descr,
  L_0.code
FROM (sap_common_Countries AS L_0 LEFT JOIN sap_common_Countries_texts AS localized_de_1 ON localized_de_1.code = L_0.code AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_sap_common_Countries AS SELECT
  coalesce(localized_fr_1.name, L_0.name) AS name,
  coalesce(localized_fr_1.descr, L_0.descr) AS descr,
  L_0.code
FROM (sap_common_Countries AS L_0 LEFT JOIN sap_common_Countries_texts AS localized_fr_1 ON localized_fr_1.code = L_0.code AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_sap_common_Currencies AS SELECT
  coalesce(localized_de_1.name, L_0.name) AS name,
  coalesce(localized_de_1.descr, L_0.descr) AS descr,
  L_0.code,
  L_0.symbol,
  L_0.minorUnit
FROM (sap_common_Currencies AS L_0 LEFT JOIN sap_common_Currencies_texts AS localized_de_1 ON localized_de_1.code = L_0.code AND localized_de_1.locale = 'de');

CREATE VIEW localized_fr_sap_common_Currencies AS SELECT
  coalesce(localized_fr_1.name, L_0.name) AS name,
  coalesce(localized_fr_1.descr, L_0.descr) AS descr,
  L_0.code,
  L_0.symbol,
  L_0.minorUnit
FROM (sap_common_Currencies AS L_0 LEFT JOIN sap_common_Currencies_texts AS localized_fr_1 ON localized_fr_1.code = L_0.code AND localized_fr_1.locale = 'fr');

CREATE VIEW localized_de_a2a_network_AgentCapabilities AS SELECT
  L.ID,
  L.agent_ID,
  L.capability_ID,
  L.registeredAt,
  L.version,
  L.status
FROM a2a_network_AgentCapabilities AS L;

CREATE VIEW localized_fr_a2a_network_AgentCapabilities AS SELECT
  L.ID,
  L.agent_ID,
  L.capability_ID,
  L.registeredAt,
  L.version,
  L.status
FROM a2a_network_AgentCapabilities AS L;

CREATE VIEW localized_de_a2a_network_Services AS SELECT
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

CREATE VIEW localized_fr_a2a_network_Services AS SELECT
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

CREATE VIEW localized_de_a2a_network_ServiceOrders AS SELECT
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

CREATE VIEW localized_fr_a2a_network_ServiceOrders AS SELECT
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

CREATE VIEW localized_de_a2a_network_Messages AS SELECT
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

CREATE VIEW localized_fr_a2a_network_Messages AS SELECT
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

CREATE VIEW localized_de_a2a_network_AgentPerformance AS SELECT
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

CREATE VIEW localized_fr_a2a_network_AgentPerformance AS SELECT
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

CREATE VIEW localized_de_a2a_network_Workflows AS SELECT
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

CREATE VIEW localized_fr_a2a_network_Workflows AS SELECT
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

CREATE VIEW localized_de_a2a_network_PerformanceSnapshots AS SELECT
  L.ID,
  L.performance_ID,
  L.timestamp,
  L.taskCount,
  L.successRate,
  L.responseTime,
  L.gasUsage,
  L.reputationDelta
FROM a2a_network_PerformanceSnapshots AS L;

CREATE VIEW localized_fr_a2a_network_PerformanceSnapshots AS SELECT
  L.ID,
  L.performance_ID,
  L.timestamp,
  L.taskCount,
  L.successRate,
  L.responseTime,
  L.gasUsage,
  L.reputationDelta
FROM a2a_network_PerformanceSnapshots AS L;

CREATE VIEW localized_de_a2a_network_WorkflowExecutions AS SELECT
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

CREATE VIEW localized_fr_a2a_network_WorkflowExecutions AS SELECT
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

CREATE VIEW localized_de_a2a_network_WorkflowSteps AS SELECT
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

CREATE VIEW localized_fr_a2a_network_WorkflowSteps AS SELECT
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

CREATE VIEW localized_de_a2a_network_TopAgents AS SELECT
  Agents_0.ID,
  Agents_0.address,
  Agents_0.name,
  Agents_0.reputation,
  performance_1.successfulTasks AS completedTasks,
  performance_1.averageResponseTime AS avgResponseTime
FROM (localized_de_a2a_network_Agents AS Agents_0 LEFT JOIN localized_de_a2a_network_AgentPerformance AS performance_1 ON (performance_1.agent_ID = Agents_0.ID))
WHERE Agents_0.isActive = TRUE
ORDER BY reputation DESC;

CREATE VIEW localized_fr_a2a_network_TopAgents AS SELECT
  Agents_0.ID,
  Agents_0.address,
  Agents_0.name,
  Agents_0.reputation,
  performance_1.successfulTasks AS completedTasks,
  performance_1.averageResponseTime AS avgResponseTime
FROM (localized_fr_a2a_network_Agents AS Agents_0 LEFT JOIN localized_fr_a2a_network_AgentPerformance AS performance_1 ON (performance_1.agent_ID = Agents_0.ID))
WHERE Agents_0.isActive = TRUE
ORDER BY reputation DESC;

CREATE VIEW localized_de_a2a_network_ActiveServices AS SELECT
  Services_0.ID,
  Services_0.name,
  provider_1.name AS providerName,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.totalCalls,
  Services_0.averageRating
FROM (localized_de_a2a_network_Services AS Services_0 LEFT JOIN localized_de_a2a_network_Agents AS provider_1 ON Services_0.provider_ID = provider_1.ID)
WHERE Services_0.isActive = TRUE;

CREATE VIEW localized_fr_a2a_network_ActiveServices AS SELECT
  Services_0.ID,
  Services_0.name,
  provider_1.name AS providerName,
  Services_0.category,
  Services_0.pricePerCall,
  Services_0.totalCalls,
  Services_0.averageRating
FROM (localized_fr_a2a_network_Services AS Services_0 LEFT JOIN localized_fr_a2a_network_Agents AS provider_1 ON Services_0.provider_ID = provider_1.ID)
WHERE Services_0.isActive = TRUE;


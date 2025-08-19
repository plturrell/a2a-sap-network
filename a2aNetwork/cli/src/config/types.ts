export interface EnvironmentConfig {
  // Core settings
  NODE_ENV: string;
  PORT: string;
  LOG_LEVEL: string;
  
  // Security
  JWT_SECRET: string;
  SESSION_SECRET: string;
  
  // Agent configuration
  AGENT_NAME: string;
  AGENT_TYPE: string;
  AGENT_VERSION: string;
  AGENT_HEALTH_CHECK_INTERVAL: string;
  
  // Registry
  A2A_REGISTRY_URL: string;
  A2A_REGISTRY_API_KEY: string;
  
  // Database
  DATABASE_TYPE: string;
  DATABASE_URL: string;
  
  // Optional blockchain settings
  BLOCKCHAIN_ENABLED?: string;
  BLOCKCHAIN_NETWORK?: string;
  BLOCKCHAIN_RPC_URL?: string;
  BLOCKCHAIN_CHAIN_ID?: string;
  BLOCKCHAIN_PRIVATE_KEY?: string;
  CONTRACT_AGENT_REGISTRY?: string;
  CONTRACT_SERVICE_MARKETPLACE?: string;
  CONTRACT_REPUTATION_SYSTEM?: string;
  BLOCKCHAIN_AUTO_MINE?: string;
  BLOCKCHAIN_GAS_PRICE?: string;
  BLOCKCHAIN_CONFIRMATIONS?: string;
  
  // Optional SAP settings
  SAP_ENABLED?: string;
  SAP_SYSTEM_URL?: string;
  SAP_CLIENT?: string;
  SAP_AUTH_TYPE?: string;
  SAP_USERNAME?: string;
  SAP_PASSWORD?: string;
  XSUAA_URL?: string;
  XSUAA_CLIENT_ID?: string;
  XSUAA_CLIENT_SECRET?: string;
  CDS_REQUIRES_AUTH?: string;
  CDS_REQUIRES_DB?: string;
  
  // Any additional custom settings
  [key: string]: string | undefined;
}

export interface ConfigTemplate {
  name: string;
  description: string;
  capabilities: string[];
  dependencies: {
    required: string[];
    optional: string[];
    dev: string[];
  };
  files: {
    [path: string]: string;
  };
}

export interface A2AConfig {
  name: string;
  version: string;
  type: string;
  agent: {
    type: string;
    capabilities: string[];
    discovery: {
      enabled: boolean;
      interval: number;
      timeout: number;
    };
    health: {
      enabled: boolean;
      endpoint: string;
      interval: number;
    };
  };
  development: {
    watch: boolean;
    mocks: {
      registry: boolean;
      blockchain: boolean;
      sap: boolean;
    };
    tools: {
      inspector: boolean;
      profiler: boolean;
      metrics: boolean;
    };
  };
  network: {
    retry: {
      attempts: number;
      delay: number;
      backoff: number;
    };
    circuitBreaker: {
      enabled: boolean;
      threshold: number;
      timeout: number;
    };
  };
  blockchain?: {
    enabled: boolean;
    networks: {
      [key: string]: {
        url: string;
        accounts: string[];
      };
    };
  };
  sap?: {
    enabled: boolean;
    cap: {
      root: string;
      model: string;
    };
  };
  logging: {
    level: string;
    format: string;
    transports: string[];
  };
}
export const DEFAULT_CONFIG = {
  // Transaction settings
  gasLimit: 500000,
  maxPriorityFeePerGas: 2000000000, // 2 gwei
  maxFeePerGas: 20000000000, // 20 gwei
  
  // Timeouts
  transactionTimeout: 60000, // 60 seconds
  connectionTimeout: 30000, // 30 seconds
  messageTimeout: 30000, // 30 seconds
  
  // Retry settings
  maxRetries: 3,
  retryDelay: 1000, // 1 second
  
  // WebSocket settings
  wsReconnectInterval: 5000, // 5 seconds
  wsMaxReconnectAttempts: 10,
  
  // Cache settings
  cacheEnabled: true,
  cacheTTL: 300000, // 5 minutes
  
  // Logging
  logLevel: 'info',
  
  // API settings
  apiVersion: 'v1',
  apiTimeout: 30000 // 30 seconds
} as const;

export const MESSAGE_TYPES = {
  DIRECT: 'direct',
  BROADCAST: 'broadcast',
  SYSTEM: 'system',
  ERROR: 'error'
} as const;

export const AGENT_STATUS = {
  ACTIVE: 'active',
  INACTIVE: 'inactive',
  SUSPENDED: 'suspended',
  BANNED: 'banned'
} as const;
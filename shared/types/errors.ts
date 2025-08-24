/**
 * Standardized Error Types for A2A Platform
 * Provides consistent error handling across all services
 */

export enum ErrorCode {
  // Agent Errors (1000-1999)
  AGENT_NOT_FOUND = 'A2A_1001',
  AGENT_UNAVAILABLE = 'A2A_1002',
  AGENT_TIMEOUT = 'A2A_1003',
  AGENT_CONFIGURATION_ERROR = 'A2A_1004',
  AGENT_VALIDATION_FAILED = 'A2A_1005',
  AGENT_PROCESSING_ERROR = 'A2A_1006',
  AGENT_AUTHENTICATION_FAILED = 'A2A_1007',

  // Network Errors (2000-2999)
  NETWORK_CONNECTION_ERROR = 'A2A_2001',
  NETWORK_TIMEOUT = 'A2A_2002',
  NETWORK_SERVICE_UNAVAILABLE = 'A2A_2003',
  NETWORK_RATE_LIMIT_EXCEEDED = 'A2A_2004',
  NETWORK_PROTOCOL_ERROR = 'A2A_2005',
  NETWORK_SECURITY_VIOLATION = 'A2A_2006',

  // Data Errors (3000-3999)
  DATA_VALIDATION_ERROR = 'A2A_3001',
  DATA_NOT_FOUND = 'A2A_3002',
  DATA_CORRUPTION_DETECTED = 'A2A_3003',
  DATA_FORMAT_ERROR = 'A2A_3004',
  DATA_SIZE_LIMIT_EXCEEDED = 'A2A_3005',
  DATA_QUALITY_CHECK_FAILED = 'A2A_3006',
  DATA_ACCESS_DENIED = 'A2A_3007',

  // Workflow Errors (4000-4999)
  WORKFLOW_NOT_FOUND = 'A2A_4001',
  WORKFLOW_EXECUTION_FAILED = 'A2A_4002',
  WORKFLOW_INVALID_STATE = 'A2A_4003',
  WORKFLOW_DEPENDENCY_ERROR = 'A2A_4004',
  WORKFLOW_TIMEOUT = 'A2A_4005',
  WORKFLOW_VALIDATION_ERROR = 'A2A_4006',

  // Blockchain Errors (5000-5999)
  BLOCKCHAIN_CONNECTION_ERROR = 'A2A_5001',
  BLOCKCHAIN_TRANSACTION_FAILED = 'A2A_5002',
  BLOCKCHAIN_INSUFFICIENT_FUNDS = 'A2A_5003',
  BLOCKCHAIN_CONTRACT_ERROR = 'A2A_5004',
  BLOCKCHAIN_NETWORK_ERROR = 'A2A_5005',
  BLOCKCHAIN_VALIDATION_ERROR = 'A2A_5006',

  // System Errors (9000-9999)
  INTERNAL_SERVER_ERROR = 'A2A_9001',
  SERVICE_UNAVAILABLE = 'A2A_9002',
  DATABASE_ERROR = 'A2A_9003',
  CONFIGURATION_ERROR = 'A2A_9004',
  RESOURCE_EXHAUSTED = 'A2A_9005',
  DEPENDENCY_ERROR = 'A2A_9006',
  SECURITY_ERROR = 'A2A_9007',
}

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum ErrorCategory {
  BUSINESS_LOGIC = 'business_logic',
  TECHNICAL = 'technical',
  SECURITY = 'security',
  PERFORMANCE = 'performance',
  INFRASTRUCTURE = 'infrastructure'
}

export interface ErrorContext {
  userId?: string;
  agentId?: string;
  workflowId?: string;
  requestId?: string;
  correlationId?: string;
  timestamp: Date;
  environment: string;
  service: string;
  operation: string;
  metadata?: Record<string, any>;
}

export interface ErrorDetails {
  field?: string;
  value?: any;
  constraint?: string;
  message: string;
}

export interface A2AError {
  code: ErrorCode;
  message: string;
  severity: ErrorSeverity;
  category: ErrorCategory;
  context: ErrorContext;
  details?: ErrorDetails[];
  cause?: Error | A2AError;
  retryable: boolean;
  userFriendlyMessage?: string;
  troubleshootingUrl?: string;
}

export interface ErrorResponse {
  error: A2AError;
  requestId: string;
  timestamp: string;
  path?: string;
  method?: string;
}

// Error mapping for HTTP status codes
export const ERROR_HTTP_STATUS_MAP: Record<ErrorCode, number> = {
  // 400 Bad Request
  [ErrorCode.AGENT_VALIDATION_FAILED]: 400,
  [ErrorCode.DATA_VALIDATION_ERROR]: 400,
  [ErrorCode.DATA_FORMAT_ERROR]: 400,
  [ErrorCode.WORKFLOW_VALIDATION_ERROR]: 400,
  [ErrorCode.BLOCKCHAIN_VALIDATION_ERROR]: 400,

  // 401 Unauthorized
  [ErrorCode.AGENT_AUTHENTICATION_FAILED]: 401,

  // 403 Forbidden
  [ErrorCode.DATA_ACCESS_DENIED]: 403,
  [ErrorCode.NETWORK_SECURITY_VIOLATION]: 403,
  [ErrorCode.SECURITY_ERROR]: 403,

  // 404 Not Found
  [ErrorCode.AGENT_NOT_FOUND]: 404,
  [ErrorCode.DATA_NOT_FOUND]: 404,
  [ErrorCode.WORKFLOW_NOT_FOUND]: 404,

  // 408 Request Timeout
  [ErrorCode.AGENT_TIMEOUT]: 408,
  [ErrorCode.NETWORK_TIMEOUT]: 408,
  [ErrorCode.WORKFLOW_TIMEOUT]: 408,

  // 413 Payload Too Large
  [ErrorCode.DATA_SIZE_LIMIT_EXCEEDED]: 413,

  // 422 Unprocessable Entity
  [ErrorCode.DATA_QUALITY_CHECK_FAILED]: 422,
  [ErrorCode.WORKFLOW_INVALID_STATE]: 422,

  // 429 Too Many Requests
  [ErrorCode.NETWORK_RATE_LIMIT_EXCEEDED]: 429,

  // 500 Internal Server Error
  [ErrorCode.INTERNAL_SERVER_ERROR]: 500,
  [ErrorCode.AGENT_PROCESSING_ERROR]: 500,
  [ErrorCode.WORKFLOW_EXECUTION_FAILED]: 500,
  [ErrorCode.DATABASE_ERROR]: 500,

  // 502 Bad Gateway
  [ErrorCode.NETWORK_CONNECTION_ERROR]: 502,
  [ErrorCode.BLOCKCHAIN_CONNECTION_ERROR]: 502,
  [ErrorCode.DEPENDENCY_ERROR]: 502,

  // 503 Service Unavailable
  [ErrorCode.AGENT_UNAVAILABLE]: 503,
  [ErrorCode.NETWORK_SERVICE_UNAVAILABLE]: 503,
  [ErrorCode.SERVICE_UNAVAILABLE]: 503,

  // 507 Insufficient Storage
  [ErrorCode.RESOURCE_EXHAUSTED]: 507,

  // Default mappings
  [ErrorCode.AGENT_CONFIGURATION_ERROR]: 500,
  [ErrorCode.NETWORK_PROTOCOL_ERROR]: 500,
  [ErrorCode.DATA_CORRUPTION_DETECTED]: 500,
  [ErrorCode.WORKFLOW_DEPENDENCY_ERROR]: 500,
  [ErrorCode.BLOCKCHAIN_TRANSACTION_FAILED]: 500,
  [ErrorCode.BLOCKCHAIN_INSUFFICIENT_FUNDS]: 402,
  [ErrorCode.BLOCKCHAIN_CONTRACT_ERROR]: 500,
  [ErrorCode.BLOCKCHAIN_NETWORK_ERROR]: 502,
  [ErrorCode.CONFIGURATION_ERROR]: 500,
};

// Retryable error codes
export const RETRYABLE_ERRORS = new Set([
  ErrorCode.AGENT_TIMEOUT,
  ErrorCode.AGENT_UNAVAILABLE,
  ErrorCode.NETWORK_CONNECTION_ERROR,
  ErrorCode.NETWORK_TIMEOUT,
  ErrorCode.NETWORK_SERVICE_UNAVAILABLE,
  ErrorCode.SERVICE_UNAVAILABLE,
  ErrorCode.DATABASE_ERROR,
  ErrorCode.RESOURCE_EXHAUSTED,
  ErrorCode.BLOCKCHAIN_CONNECTION_ERROR,
  ErrorCode.BLOCKCHAIN_NETWORK_ERROR,
  ErrorCode.INTERNAL_SERVER_ERROR,
]);

// User-friendly error messages
export const USER_FRIENDLY_MESSAGES: Record<ErrorCode, string> = {
  [ErrorCode.AGENT_NOT_FOUND]: 'The requested agent could not be found.',
  [ErrorCode.AGENT_UNAVAILABLE]: 'The agent is temporarily unavailable. Please try again later.',
  [ErrorCode.AGENT_TIMEOUT]: 'The agent operation timed out. Please try again.',
  [ErrorCode.DATA_NOT_FOUND]: 'The requested data could not be found.',
  [ErrorCode.DATA_VALIDATION_ERROR]: 'The provided data is invalid. Please check your input.',
  [ErrorCode.WORKFLOW_NOT_FOUND]: 'The requested workflow could not be found.',
  [ErrorCode.NETWORK_RATE_LIMIT_EXCEEDED]: 'Too many requests. Please wait before trying again.',
  [ErrorCode.INTERNAL_SERVER_ERROR]: 'An unexpected error occurred. Please try again later.',
  // Add more user-friendly messages as needed
} as Record<ErrorCode, string>;
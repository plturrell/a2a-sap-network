/**
 * Centralized Error Handler for A2A Platform
 * Provides consistent error processing, logging, and response formatting
 */

import { 
  A2AError, 
  ErrorCode, 
  ErrorSeverity, 
  ErrorCategory, 
  ErrorContext, 
  ErrorDetails,
  ErrorResponse,
  ERROR_HTTP_STATUS_MAP,
  RETRYABLE_ERRORS,
  USER_FRIENDLY_MESSAGES
} from '../types/errors';

export interface ErrorHandlerConfig {
  serviceName: string;
  environment: string;
  enableStackTrace: boolean;
  enableMetrics: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  troubleshootingBaseUrl?: string;
}

export interface MetricsCollector {
  incrementCounter(name: string, labels: Record<string, string>): void;
  observeHistogram(name: string, value: number, labels: Record<string, string>): void;
}

export interface Logger {
  debug(message: string, meta?: any): void;
  info(message: string, meta?: any): void;
  warn(message: string, meta?: any): void;
  error(message: string, meta?: any): void;
}

export class A2AErrorHandler {
  private config: ErrorHandlerConfig;
  private logger: Logger;
  private metricsCollector?: MetricsCollector;

  constructor(
    config: ErrorHandlerConfig,
    logger: Logger,
    metricsCollector?: MetricsCollector
  ) {
    this.config = config;
    this.logger = logger;
    this.metricsCollector = metricsCollector;
  }

  /**
   * Creates a standardized A2A error
   */
  createError(
    code: ErrorCode,
    message: string,
    context: Partial<ErrorContext>,
    details?: ErrorDetails[],
    cause?: Error | A2AError
  ): A2AError {
    const severity = this.determineSeverity(code);
    const category = this.determineCategory(code);
    const retryable = RETRYABLE_ERRORS.has(code);
    const userFriendlyMessage = USER_FRIENDLY_MESSAGES[code];
    const troubleshootingUrl = this.buildTroubleshootingUrl(code);

    const fullContext: ErrorContext = {
      timestamp: new Date(),
      environment: this.config.environment,
      service: this.config.serviceName,
      operation: context.operation || 'unknown',
      ...context
    };

    return {
      code,
      message,
      severity,
      category,
      context: fullContext,
      details,
      cause,
      retryable,
      userFriendlyMessage,
      troubleshootingUrl
    };
  }

  /**
   * Handles an error with logging and metrics
   */
  handleError(error: A2AError | Error, operation?: string): A2AError {
    let processedError: A2AError;

    if (this.isA2AError(error)) {
      processedError = error;
    } else {
      // Convert generic error to A2A error
      processedError = this.createError(
        ErrorCode.INTERNAL_SERVER_ERROR,
        error.message || 'Unknown error occurred',
        { operation: operation || 'unknown' },
        undefined,
        error
      );
    }

    // Log the error
    this.logError(processedError);

    // Collect metrics
    if (this.config.enableMetrics && this.metricsCollector) {
      this.collectMetrics(processedError);
    }

    return processedError;
  }

  /**
   * Converts an A2A error to HTTP response format
   */
  toHttpResponse(error: A2AError, requestId: string, path?: string, method?: string): ErrorResponse {
    return {
      error: {
        ...error,
        // Remove sensitive information for external responses
        context: {
          ...error.context,
          metadata: this.sanitizeMetadata(error.context.metadata)
        }
      },
      requestId,
      timestamp: new Date().toISOString(),
      path,
      method
    };
  }

  /**
   * Gets HTTP status code for an error
   */
  getHttpStatusCode(error: A2AError): number {
    return ERROR_HTTP_STATUS_MAP[error.code] || 500;
  }

  /**
   * Checks if an error is retryable
   */
  isRetryable(error: A2AError): boolean {
    return error.retryable && RETRYABLE_ERRORS.has(error.code);
  }

  /**
   * Creates a retry strategy for retryable errors
   */
  getRetryStrategy(error: A2AError): { shouldRetry: boolean; delayMs: number; maxAttempts: number } {
    if (!this.isRetryable(error)) {
      return { shouldRetry: false, delayMs: 0, maxAttempts: 0 };
    }

    // Exponential backoff with jitter
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const maxAttempts = 3;

    return {
      shouldRetry: true,
      delayMs: Math.min(baseDelay * Math.pow(2, Math.random()), maxDelay),
      maxAttempts
    };
  }

  private isA2AError(error: any): error is A2AError {
    return error && typeof error === 'object' && 'code' in error && 'context' in error;
  }

  private determineSeverity(code: ErrorCode): ErrorSeverity {
    const criticalErrors = [
      ErrorCode.SECURITY_ERROR,
      ErrorCode.DATA_CORRUPTION_DETECTED,
      ErrorCode.BLOCKCHAIN_CONTRACT_ERROR
    ];

    const highErrors = [
      ErrorCode.AGENT_AUTHENTICATION_FAILED,
      ErrorCode.DATA_ACCESS_DENIED,
      ErrorCode.WORKFLOW_EXECUTION_FAILED,
      ErrorCode.DATABASE_ERROR
    ];

    const mediumErrors = [
      ErrorCode.AGENT_UNAVAILABLE,
      ErrorCode.NETWORK_CONNECTION_ERROR,
      ErrorCode.DATA_VALIDATION_ERROR
    ];

    if (criticalErrors.includes(code)) return ErrorSeverity.CRITICAL;
    if (highErrors.includes(code)) return ErrorSeverity.HIGH;
    if (mediumErrors.includes(code)) return ErrorSeverity.MEDIUM;
    return ErrorSeverity.LOW;
  }

  private determineCategory(code: ErrorCode): ErrorCategory {
    const codeString = code.toString();
    
    if (codeString.startsWith('A2A_1') || codeString.startsWith('A2A_3') || codeString.startsWith('A2A_4')) {
      return ErrorCategory.BUSINESS_LOGIC;
    }
    
    if (codeString.startsWith('A2A_2') || codeString.startsWith('A2A_9')) {
      return ErrorCategory.TECHNICAL;
    }
    
    if (code === ErrorCode.SECURITY_ERROR || code === ErrorCode.AGENT_AUTHENTICATION_FAILED) {
      return ErrorCategory.SECURITY;
    }
    
    return ErrorCategory.TECHNICAL;
  }

  private buildTroubleshootingUrl(code: ErrorCode): string | undefined {
    if (!this.config.troubleshootingBaseUrl) return undefined;
    
    return `${this.config.troubleshootingBaseUrl}/errors/${code}`;
  }

  private logError(error: A2AError): void {
    const logData = {
      code: error.code,
      message: error.message,
      severity: error.severity,
      category: error.category,
      context: error.context,
      details: error.details,
      retryable: error.retryable,
      stack: this.config.enableStackTrace && error.cause instanceof Error ? error.cause.stack : undefined
    };

    switch (error.severity) {
      case ErrorSeverity.CRITICAL:
      case ErrorSeverity.HIGH:
        this.logger.error('A2A Error occurred', logData);
        break;
      case ErrorSeverity.MEDIUM:
        this.logger.warn('A2A Warning occurred', logData);
        break;
      case ErrorSeverity.LOW:
        this.logger.info('A2A Info occurred', logData);
        break;
    }
  }

  private collectMetrics(error: A2AError): void {
    if (!this.metricsCollector) return;

    const labels = {
      code: error.code,
      severity: error.severity,
      category: error.category,
      service: error.context.service,
      operation: error.context.operation,
      retryable: error.retryable.toString()
    };

    // Increment error counter
    this.metricsCollector.incrementCounter('a2a_errors_total', labels);

    // Track error by severity
    this.metricsCollector.incrementCounter(`a2a_errors_${error.severity}_total`, {
      service: error.context.service,
      operation: error.context.operation
    });
  }

  private sanitizeMetadata(metadata?: Record<string, any>): Record<string, any> | undefined {
    if (!metadata) return undefined;

    const sensitiveKeys = ['password', 'token', 'secret', 'key', 'authorization'];
    const sanitized = { ...metadata };

    for (const key of Object.keys(sanitized)) {
      if (sensitiveKeys.some(sensitive => key.toLowerCase().includes(sensitive))) {
        sanitized[key] = '[REDACTED]';
      }
    }

    return sanitized;
  }
}

/**
 * Error handler factory for different services
 */
export class ErrorHandlerFactory {
  static createAgentErrorHandler(
    agentId: string,
    environment: string,
    logger: Logger,
    metricsCollector?: MetricsCollector
  ): A2AErrorHandler {
    return new A2AErrorHandler(
      {
        serviceName: `agent-${agentId}`,
        environment,
        enableStackTrace: environment !== 'production',
        enableMetrics: true,
        logLevel: environment === 'production' ? 'error' : 'debug',
        troubleshootingBaseUrl: 'https://docs.a2a-platform.com'
      },
      logger,
      metricsCollector
    );
  }

  static createNetworkErrorHandler(
    environment: string,
    logger: Logger,
    metricsCollector?: MetricsCollector
  ): A2AErrorHandler {
    return new A2AErrorHandler(
      {
        serviceName: 'a2a-network',
        environment,
        enableStackTrace: environment !== 'production',
        enableMetrics: true,
        logLevel: environment === 'production' ? 'error' : 'debug',
        troubleshootingBaseUrl: 'https://docs.a2a-platform.com'
      },
      logger,
      metricsCollector
    );
  }

  static createWorkflowErrorHandler(
    environment: string,
    logger: Logger,
    metricsCollector?: MetricsCollector
  ): A2AErrorHandler {
    return new A2AErrorHandler(
      {
        serviceName: 'a2a-workflow',
        environment,
        enableStackTrace: environment !== 'production',
        enableMetrics: true,
        logLevel: environment === 'production' ? 'error' : 'debug',
        troubleshootingBaseUrl: 'https://docs.a2a-platform.com'
      },
      logger,
      metricsCollector
    );
  }
}
/**
 * SAP CAP Error Handler for A2A Platform
 * Provides consistent error handling for CAP services
 */

const cds = require('@sap/cds');

class A2ACapErrorHandler {
  constructor(config) {
    this.config = {
      serviceName: config.serviceName || 'a2a-service',
      environment: config.environment || 'development',
      enableStackTrace: config.enableStackTrace ?? (config.environment !== 'production'),
      enableMetrics: config.enableMetrics ?? true,
      logLevel: config.logLevel || 'info'
    };
    
    this.logger = cds.log(this.config.serviceName);
  }

  /**
   * Register error handlers for a CAP service
   */
  registerHandlers(srv) {
    // Before handler for request tracking
    srv.before('*', (req) => {
      req._.requestId = req.headers['x-request-id'] || cds.utils.uuid();
      req._.correlationId = req.headers['x-correlation-id'] || req._.requestId;
      req._.startTime = Date.now();
      
      this.logger.debug('Request started', {
        requestId: req._.requestId,
        event: req.event,
        entity: req.entity,
        user: req.user?.id
      });
    });

    // After handler for successful requests
    srv.after('*', (result, req) => {
      const duration = Date.now() - req._.startTime;
      
      this.logger.info('Request completed', {
        requestId: req._.requestId,
        event: req.event,
        entity: req.entity,
        duration,
        user: req.user?.id
      });
      
      // Add request ID to response headers if possible
      if (req._.res) {
        req._.res.set('x-request-id', req._.requestId);
        req._.res.set('x-correlation-id', req._.correlationId);
      }
    });

    // Error handler
    srv.on('error', (err, req) => {
      const error = this.processError(err, req);
      this.logError(error, req);
      
      // Enhance error with additional context
      err.code = error.code;
      err.severity = error.severity;
      err.requestId = req._.requestId;
      err.correlationId = req._.correlationId;
      
      if (this.config.enableStackTrace && err.stack) {
        err.technicalDetails = {
          stack: err.stack,
          service: this.config.serviceName,
          environment: this.config.environment
        };
      }
    });
  }

  /**
   * Process and categorize errors
   */
  processError(err, req) {
    // Map CAP errors to A2A error codes
    const errorMapping = {
      // CAP standard errors
      '400': 'A2A_3001', // DATA_VALIDATION_ERROR
      '401': 'A2A_1007', // AGENT_AUTHENTICATION_FAILED
      '403': 'A2A_3007', // DATA_ACCESS_DENIED
      '404': 'A2A_3002', // DATA_NOT_FOUND
      '409': 'A2A_4003', // WORKFLOW_INVALID_STATE
      '422': 'A2A_3006', // DATA_QUALITY_CHECK_FAILED
      '500': 'A2A_9001', // INTERNAL_SERVER_ERROR
      '502': 'A2A_2001', // NETWORK_CONNECTION_ERROR
      '503': 'A2A_9002', // SERVICE_UNAVAILABLE
      
      // Custom error codes
      'ENTITY_NOT_FOUND': 'A2A_3002',
      'MULTIPLE_ENTITIES_FOUND': 'A2A_3001',
      'UNIQUE_CONSTRAINT_VIOLATION': 'A2A_3001',
      'FOREIGN_KEY_VIOLATION': 'A2A_3001',
      'NOT_NULL_VIOLATION': 'A2A_3001',
      'VALUE_REQUIRED': 'A2A_3001',
      'INVALID_FORMAT': 'A2A_3004',
      'EXCEEDED_MAX_LENGTH': 'A2A_3005',
      'CONNECTION_ERROR': 'A2A_2001',
      'TIMEOUT': 'A2A_2002',
      'AUTH_FAILED': 'A2A_1007'
    };

    const code = errorMapping[err.code] || errorMapping[String(err.status)] || 'A2A_9001';
    
    return {
      code,
      severity: this.determineSeverity(code),
      category: this.determineCategory(code),
      message: err.message,
      details: this.extractErrorDetails(err),
      retryable: this.isRetryable(code)
    };
  }

  /**
   * Extract detailed error information
   */
  extractErrorDetails(err) {
    const details = [];
    
    // CAP validation errors
    if (err.details) {
      err.details.forEach(detail => {
        details.push({
          field: detail.element || detail.path,
          value: detail.value,
          constraint: detail.type || 'validation',
          message: detail.message
        });
      });
    }
    
    // Database constraint violations
    if (err.constraint) {
      details.push({
        field: err.column,
        constraint: err.constraint,
        message: `Database constraint violation: ${err.constraint}`
      });
    }
    
    return details;
  }

  /**
   * Log error with appropriate level
   */
  logError(error, req) {
    const logData = {
      code: error.code,
      message: error.message,
      severity: error.severity,
      category: error.category,
      requestId: req._.requestId,
      correlationId: req._.correlationId,
      user: req.user?.id,
      tenant: req.tenant,
      event: req.event,
      entity: req.entity,
      details: error.details,
      retryable: error.retryable,
      duration: Date.now() - req._.startTime
    };

    switch (error.severity) {
      case 'critical':
      case 'high':
        this.logger.error('A2A Error occurred', logData);
        break;
      case 'medium':
        this.logger.warn('A2A Warning occurred', logData);
        break;
      case 'low':
        this.logger.info('A2A Info occurred', logData);
        break;
    }
  }

  /**
   * Determine error severity
   */
  determineSeverity(code) {
    const severityMap = {
      'A2A_9007': 'critical', // SECURITY_ERROR
      'A2A_3003': 'critical', // DATA_CORRUPTION_DETECTED
      'A2A_5004': 'critical', // BLOCKCHAIN_CONTRACT_ERROR
      'A2A_1007': 'high',     // AGENT_AUTHENTICATION_FAILED
      'A2A_3007': 'high',     // DATA_ACCESS_DENIED
      'A2A_4002': 'high',     // WORKFLOW_EXECUTION_FAILED
      'A2A_9003': 'high',     // DATABASE_ERROR
      'A2A_1002': 'medium',   // AGENT_UNAVAILABLE
      'A2A_2001': 'medium',   // NETWORK_CONNECTION_ERROR
      'A2A_3001': 'medium'    // DATA_VALIDATION_ERROR
    };
    
    return severityMap[code] || 'low';
  }

  /**
   * Determine error category
   */
  determineCategory(code) {
    if (code.startsWith('A2A_1') || code.startsWith('A2A_3') || code.startsWith('A2A_4')) {
      return 'business_logic';
    }
    if (code.startsWith('A2A_2') || code.startsWith('A2A_9')) {
      return 'technical';
    }
    if (code === 'A2A_9007' || code === 'A2A_1007') {
      return 'security';
    }
    return 'technical';
  }

  /**
   * Check if error is retryable
   */
  isRetryable(code) {
    const retryableCodes = [
      'A2A_1003', // AGENT_TIMEOUT
      'A2A_1002', // AGENT_UNAVAILABLE
      'A2A_2001', // NETWORK_CONNECTION_ERROR
      'A2A_2002', // NETWORK_TIMEOUT
      'A2A_2003', // NETWORK_SERVICE_UNAVAILABLE
      'A2A_9002', // SERVICE_UNAVAILABLE
      'A2A_9003', // DATABASE_ERROR
      'A2A_9005', // RESOURCE_EXHAUSTED
      'A2A_5001', // BLOCKCHAIN_CONNECTION_ERROR
      'A2A_5005', // BLOCKCHAIN_NETWORK_ERROR
      'A2A_9001'  // INTERNAL_SERVER_ERROR
    ];
    
    return retryableCodes.includes(code);
  }
}

/**
 * Create custom error class for A2A
 */
class A2AError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = 'A2AError';
    this.code = code;
    this.details = details;
  }
}

/**
 * CAP Service extensions for error handling
 */
cds.Service.prototype.throwA2AError = function(code, message, details) {
  throw new A2AError(code, message, details);
};

/**
 * Middleware for specific error scenarios
 */
const ErrorMiddleware = {
  /**
   * Validate entity exists before operations
   */
  validateEntityExists: async function(req) {
    if (req.params && req.params.length > 0 && req.entity) {
      const key = req.params[0];
      const exists = await cds.run(
        SELECT.one.from(req.entity).where({ ID: key })
      );
      
      if (!exists) {
        req.reject(404, 'Entity not found', {
          code: 'A2A_3002',
          entity: req.entity,
          key: key
        });
      }
    }
  },

  /**
   * Check rate limits
   */
  checkRateLimit: async function(req) {
    // Implement rate limiting logic
    const userId = req.user?.id || req._.ip;
    const key = `rate_limit:${userId}:${req.event}`;
    
    // Example rate limit check (implement actual logic)
    const isLimited = false; // await checkRedisRateLimit(key);
    
    if (isLimited) {
      req.reject(429, 'Rate limit exceeded', {
        code: 'A2A_2004',
        retryAfter: 60
      });
    }
  },

  /**
   * Validate data quality
   */
  validateDataQuality: async function(req) {
    if (req.data && req.event === 'CREATE' || req.event === 'UPDATE') {
      // Implement data quality checks
      const qualityIssues = [];
      
      // Example: Check for required fields
      const requiredFields = ['name', 'type', 'status'];
      for (const field of requiredFields) {
        if (!req.data[field]) {
          qualityIssues.push({
            field,
            issue: 'Required field missing'
          });
        }
      }
      
      if (qualityIssues.length > 0) {
        req.reject(422, 'Data quality check failed', {
          code: 'A2A_3006',
          issues: qualityIssues
        });
      }
    }
  }
};

/**
 * Factory for creating error handlers
 */
class ErrorHandlerFactory {
  static createAgentErrorHandler(agentId, environment) {
    return new A2ACapErrorHandler({
      serviceName: `agent-${agentId}`,
      environment,
      enableStackTrace: environment !== 'production',
      enableMetrics: true,
      logLevel: environment === 'production' ? 'error' : 'debug'
    });
  }

  static createNetworkErrorHandler(environment) {
    return new A2ACapErrorHandler({
      serviceName: 'a2a-network',
      environment,
      enableStackTrace: environment !== 'production',
      enableMetrics: true,
      logLevel: environment === 'production' ? 'error' : 'debug'
    });
  }

  static createWorkflowErrorHandler(environment) {
    return new A2ACapErrorHandler({
      serviceName: 'a2a-workflow',
      environment,
      enableStackTrace: environment !== 'production',
      enableMetrics: true,
      logLevel: environment === 'production' ? 'error' : 'debug'
    });
  }
}

module.exports = {
  A2ACapErrorHandler,
  A2AError,
  ErrorMiddleware,
  ErrorHandlerFactory
};
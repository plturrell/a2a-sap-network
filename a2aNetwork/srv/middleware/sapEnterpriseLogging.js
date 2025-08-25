/**
 * Enterprise Logging Middleware for A2A Network
 * Implements SAP-compliant structured logging with audit trail
 */

const winston = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
const { v4: uuidv4 } = require('uuid');

class EnterpriseLogger {
  constructor() {
    this.logger = this.createLogger();
    this.auditLogger = this.createAuditLogger();
  }

  createLogger() {
    const logFormat = winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
      winston.format.errors({ stack: true }),
      winston.format.json(),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        return JSON.stringify({
          timestamp,
          level: level.toUpperCase(),
          message,
          component: 'a2a-network',
          environment: process.env.NODE_ENV || 'development',
          instance: process.env.CF_INSTANCE_INDEX || '0',
          correlation_id: meta.correlation_id,
          user_id: meta.user_id,
          tenant_id: meta.tenant_id,
          request_id: meta.request_id,
          ...meta
        });
      })
    );

    return winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: logFormat,
      defaultMeta: {
        service: 'a2a-network-srv',
        version: process.env.APP_VERSION || '1.0.0'
      },
      transports: [
        // Console transport for development
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          ),
          silent: process.env.NODE_ENV === 'production'
        }),

        // Application logs with daily rotation
        new DailyRotateFile({
          filename: 'logs/application-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxSize: '100m',
          maxFiles: '30d',
          level: 'info',
          format: logFormat
        }),

        // Error logs
        new DailyRotateFile({
          filename: 'logs/error-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxSize: '100m',
          maxFiles: '90d',
          level: 'error',
          format: logFormat
        }),

        // Security logs
        new DailyRotateFile({
          filename: 'logs/security-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxSize: '100m',
          maxFiles: '365d',
          level: 'warn',
          format: logFormat
        })
      ]
    });
  }

  createAuditLogger() {
    const auditFormat = winston.format.combine(
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
      winston.format.json(),
      winston.format.printf(({ timestamp, ...audit }) => {
        return JSON.stringify({
          timestamp,
          audit_type: 'DATA_ACCESS',
          ...audit
        });
      })
    );

    return winston.createLogger({
      level: 'info',
      format: auditFormat,
      transports: [
        new DailyRotateFile({
          filename: 'logs/audit-%DATE%.log',
          datePattern: 'YYYY-MM-DD',
          maxSize: '100m',
          maxFiles: '2555d', // 7 years retention for audit logs
          level: 'info'
        })
      ]
    });
  }

  /**
   * Request logging middleware
   */
  requestMiddleware() {
    return (req, res, next) => {
      const startTime = Date.now();
      const requestId = uuidv4();
      const correlationId = req.headers['x-correlation-id'] || uuidv4();

      // Add to request object for use in other middleware
      req.requestId = requestId;
      req.correlationId = correlationId;
      req.startTime = startTime;

      // Extract user context
      const userId = req.user?.sub || 'anonymous';
      const tenantId = req.user?.tenant || 'default';

      // Log request start
      this.logger.info('Request started', {
        request_id: requestId,
        correlation_id: correlationId,
        user_id: userId,
        tenant_id: tenantId,
        method: req.method,
        url: req.originalUrl,
        user_agent: req.headers['user-agent'],
        client_ip: this.getClientIP(req),
        content_length: req.headers['content-length'] || 0
      });

      // Use response event listeners instead of overriding res.end to avoid OpenTelemetry conflicts
      res.on('finish', () => {
        const duration = Date.now() - startTime;
        const contentLength = parseInt(res.get('Content-Length') || 0);

        // Log response
        const logLevel = res.statusCode >= 400 ? 'error' : 'info';
        this.logger[logLevel]('Request completed', {
          request_id: requestId,
          correlation_id: correlationId,
          user_id: userId,
          tenant_id: tenantId,
          method: req.method,
          url: req.originalUrl,
          status_code: res.statusCode,
          duration_ms: duration,
          response_size: contentLength
        });

        // Log slow requests as warnings
        if (duration > 5000) { // 5 seconds
          this.logger.warn('Slow request detected', {
            request_id: requestId,
            correlation_id: correlationId,
            duration_ms: duration,
            url: req.originalUrl
          });
        }
      });

      next();
    };
  }

  /**
   * Security event logging
   */
  logSecurityEvent(event, details, req) {
    const securityEvent = {
      event_type: event,
      severity: details.severity || 'medium',
      user_id: req?.user?.sub || 'anonymous',
      tenant_id: req?.user?.tenant || 'default',
      client_ip: req ? this.getClientIP(req) : 'unknown',
      user_agent: req?.headers['user-agent'] || 'unknown',
      request_id: req?.requestId,
      correlation_id: req?.correlationId,
      ...details
    };

    this.logger.warn('Security event', securityEvent);
  }

  /**
   * Audit trail for data operations
   */
  logAudit(operation, entity, entityId, changes, req) {
    const auditEntry = {
      operation, // CREATE, READ, UPDATE, DELETE
      entity_type: entity,
      entity_id: entityId,
      user_id: req?.user?.sub || 'system',
      tenant_id: req?.user?.tenant || 'default',
      timestamp: new Date().toISOString(),
      client_ip: req ? this.getClientIP(req) : null,
      user_agent: req?.headers['user-agent'],
      request_id: req?.requestId,
      correlation_id: req?.correlationId,
      changes: changes || null
    };

    this.auditLogger.info('Data operation audit', auditEntry);
  }

  /**
   * Business event logging
   */
  logBusinessEvent(event, details, req) {
    this.logger.info('Business event', {
      event_type: event,
      user_id: req?.user?.sub,
      tenant_id: req?.user?.tenant,
      request_id: req?.requestId,
      correlation_id: req?.correlationId,
      timestamp: new Date().toISOString(),
      ...details
    });
  }

  /**
   * Performance monitoring
   */
  logPerformance(operation, duration, metadata, req) {
    const perfLog = {
      operation,
      duration_ms: duration,
      user_id: req?.user?.sub,
      tenant_id: req?.user?.tenant,
      request_id: req?.requestId,
      correlation_id: req?.correlationId,
      ...metadata
    };

    if (duration > 1000) { // Log slow operations
      this.logger.warn('Slow operation detected', perfLog);
    } else {
      this.logger.info('Performance measurement', perfLog);
    }
  }

  /**
   * Error logging with context
   */
  logError(error, context, req) {
    const errorLog = {
      error_message: error.message,
      error_stack: error.stack,
      error_code: error.code || 'UNKNOWN',
      user_id: req?.user?.sub,
      tenant_id: req?.user?.tenant,
      request_id: req?.requestId,
      correlation_id: req?.correlationId,
      context: context || {}
    };

    this.logger.error('Application error', errorLog);
  }

  /**
   * Structured logging methods
   */
  info(message, meta = {}, req = null) {
    this.logger.info(message, this.enrichMeta(meta, req));
  }

  warn(message, meta = {}, req = null) {
    this.logger.warn(message, this.enrichMeta(meta, req));
  }

  error(message, meta = {}, req = null) {
    this.logger.error(message, this.enrichMeta(meta, req));
  }

  debug(message, meta = {}, req = null) {
    this.logger.debug(message, this.enrichMeta(meta, req));
  }

  /**
   * Enrich metadata with request context
   */
  enrichMeta(meta, req) {
    if (!req) return meta;

    return {
      ...meta,
      user_id: req.user?.sub,
      tenant_id: req.user?.tenant,
      request_id: req.requestId,
      correlation_id: req.correlationId
    };
  }

  /**
   * Extract client IP address
   */
  getClientIP(req) {
    return req.headers['x-forwarded-for'] ||
           req.headers['x-real-ip'] ||
           req.connection.remoteAddress ||
           req.socket.remoteAddress ||
           (req.connection.socket ? req.connection.socket.remoteAddress : null);
  }

  /**
   * Health check for logging system
   */
  async healthCheck() {
    try {
      this.logger.info('Health check');
      return { status: 'UP', component: 'logging' };
    } catch (error) {
      return {
        status: 'DOWN',
        component: 'logging',
        error: error.message
      };
    }
  }

  /**
   * Get logging statistics
   */
  getStats() {
    return {
      transports: this.logger.transports.length,
      level: this.logger.level,
      audit_transports: this.auditLogger.transports.length
    };
  }
}

// Export singleton instance
module.exports = new EnterpriseLogger();
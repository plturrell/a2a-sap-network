/**
 * Structured Logger for A2A Platform with SAP CAP Integration
 * Provides consistent logging with correlation IDs and performance metrics
 */

const cds = require('@sap/cds');
const winston = require('winston');
const { format } = winston;

/**
 * Custom log levels for A2A platform
 */
const A2A_LOG_LEVELS = {
  levels: {
    critical: 0,
    error: 1,
    warn: 2,
    info: 3,
    debug: 4,
    trace: 5
  },
  colors: {
    critical: 'red bold',
    error: 'red',
    warn: 'yellow',
    info: 'green',
    debug: 'blue',
    trace: 'gray'
  }
};

/**
 * Structured Logger class
 */
class A2AStructuredLogger {
  constructor(config = {}) {
    this.config = {
      serviceName: config.serviceName || 'a2a-service',
      environment: config.environment || process.env.NODE_ENV || 'development',
      logLevel: config.logLevel || process.env.LOG_LEVEL || 'info',
      enableConsole: config.enableConsole ?? true,
      enableFile: config.enableFile ?? (config.environment === 'production'),
      enableMetrics: config.enableMetrics ?? true,
      logDirectory: config.logDirectory || './logs',
      maxFileSize: config.maxFileSize || '10m',
      maxFiles: config.maxFiles || '7d',
      enableCorrelation: config.enableCorrelation ?? true
    };

    this.logger = this.createLogger();
    this.metricsCollector = this.createMetricsCollector();
    this.contextStorage = new Map();
  }

  /**
   * Create Winston logger instance
   */
  createLogger() {
    const transports = [];

    // Console transport
    if (this.config.enableConsole) {
      transports.push(new winston.transports.Console({
        format: format.combine(
          format.colorize(),
          format.timestamp(),
          this.config.environment === 'development' 
            ? format.printf(this.devLogFormat)
            : format.json()
        )
      }));
    }

    // File transport
    if (this.config.enableFile) {
      transports.push(new winston.transports.File({
        filename: `${this.config.logDirectory}/error.log`,
        level: 'error',
        maxsize: this.config.maxFileSize,
        maxFiles: this.config.maxFiles,
        format: format.json()
      }));

      transports.push(new winston.transports.File({
        filename: `${this.config.logDirectory}/combined.log`,
        maxsize: this.config.maxFileSize,
        maxFiles: this.config.maxFiles,
        format: format.json()
      }));
    }

    return winston.createLogger({
      levels: A2A_LOG_LEVELS.levels,
      level: this.config.logLevel,
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        this.enrichLogFormat()
      ),
      transports,
      exitOnError: false
    });
  }

  /**
   * Create metrics collector
   */
  createMetricsCollector() {
    return {
      logDuration: new Map(),
      logCounts: new Map(),
      
      recordLogMetric: (level, context) => {
        const key = `${level}:${context.service}:${context.operation || 'unknown'}`;
        const count = this.metricsCollector.logCounts.get(key) || 0;
        this.metricsCollector.logCounts.set(key, count + 1);
      },
      
      getMetrics: () => {
        const metrics = {
          logCounts: Object.fromEntries(this.metricsCollector.logCounts),
          timestamp: new Date().toISOString()
        };
        return metrics;
      }
    };
  }

  /**
   * Enrich log format with context
   */
  enrichLogFormat() {
    return format((info) => {
      // Add service metadata
      info.service = this.config.serviceName;
      info.environment = this.config.environment;
      info.version = process.env.npm_package_version || 'unknown';
      
      // Add Node.js metadata
      info.nodeVersion = process.version;
      info.pid = process.pid;
      
      // Add memory usage
      const memUsage = process.memoryUsage();
      info.memory = {
        heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024),
        heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024),
        rss: Math.round(memUsage.rss / 1024 / 1024),
        external: Math.round(memUsage.external / 1024 / 1024)
      };
      
      return info;
    })();
  }

  /**
   * Development log format
   */
  devLogFormat(info) {
    const { timestamp, level, message, ...meta } = info;
    const time = timestamp.slice(11, 19);
    
    let output = `${time} [${level}] ${message}`;
    
    if (meta.correlationId) {
      output += ` | CID: ${meta.correlationId.slice(0, 8)}`;
    }
    
    if (meta.duration) {
      output += ` | ${meta.duration}ms`;
    }
    
    if (meta.error) {
      output += `\n  Error: ${meta.error}`;
      if (meta.stack) {
        output += `\n  Stack: ${meta.stack}`;
      }
    }
    
    return output;
  }

  /**
   * Log with context
   */
  log(level, message, context = {}) {
    const enrichedContext = {
      ...context,
      timestamp: new Date().toISOString()
    };
    
    // Record metrics
    if (this.config.enableMetrics) {
      this.metricsCollector.recordLogMetric(level, enrichedContext);
    }
    
    this.logger.log(level, message, enrichedContext);
  }

  // Convenience methods
  critical(message, context) { this.log('critical', message, context); }
  error(message, context) { this.log('error', message, context); }
  warn(message, context) { this.log('warn', message, context); }
  info(message, context) { this.log('info', message, context); }
  debug(message, context) { this.log('debug', message, context); }
  trace(message, context) { this.log('trace', message, context); }

  /**
   * Create child logger with persistent context
   */
  child(context) {
    const childLogger = Object.create(this);
    childLogger.defaultContext = { ...this.defaultContext, ...context };
    return childLogger;
  }

  /**
   * Log performance metrics
   */
  logPerformance(operation, duration, context = {}) {
    const performanceContext = {
      ...context,
      operation,
      duration,
      performance: {
        duration,
        category: this.categorizePerformance(duration)
      }
    };
    
    this.info(`Performance: ${operation}`, performanceContext);
  }

  /**
   * Categorize performance based on duration
   */
  categorizePerformance(duration) {
    if (duration < 100) return 'fast';
    if (duration < 500) return 'normal';
    if (duration < 1000) return 'slow';
    return 'very_slow';
  }

  /**
   * Start performance timer
   */
  startTimer(operation) {
    const start = process.hrtime.bigint();
    
    return {
      end: (context = {}) => {
        const end = process.hrtime.bigint();
        const duration = Number(end - start) / 1000000; // Convert to milliseconds
        this.logPerformance(operation, Math.round(duration), context);
        return duration;
      }
    };
  }
}

/**
 * CAP Logger Integration
 */
class CapStructuredLogger extends A2AStructuredLogger {
  constructor(config) {
    super(config);
    this.setupCapIntegration();
  }

  /**
   * Setup CAP logging integration
   */
  setupCapIntegration() {
    // Override CAP's default logger
    const self = this;
    
    cds.log = function(module) {
      const logger = {
        _module: module,
        
        trace: (msg, ...args) => self.trace(self.formatCapMessage(msg, args), { module }),
        debug: (msg, ...args) => self.debug(self.formatCapMessage(msg, args), { module }),
        info: (msg, ...args) => self.info(self.formatCapMessage(msg, args), { module }),
        warn: (msg, ...args) => self.warn(self.formatCapMessage(msg, args), { module }),
        error: (msg, ...args) => self.error(self.formatCapMessage(msg, args), { module })
      };
      
      return logger;
    };
  }

  /**
   * Format CAP log messages
   */
  formatCapMessage(msg, args) {
    if (typeof msg === 'function') {
      msg = msg();
    }
    
    if (args.length > 0) {
      return `${msg} ${args.map(arg => JSON.stringify(arg)).join(' ')}`;
    }
    
    return msg;
  }

  /**
   * Create request logger middleware for CAP
   */
  createRequestLogger() {
    return (req, res, next) => {
      // Generate or extract correlation ID
      const correlationId = req.headers['x-correlation-id'] || 
                          req.headers['x-request-id'] || 
                          cds.utils.uuid();
      
      // Store in request context
      req.correlationId = correlationId;
      if (req._) {
        req._.correlationId = correlationId;
      }
      
      // Start timer
      const timer = this.startTimer('http_request');
      
      // Log request
      this.info('Request received', {
        correlationId,
        method: req.method,
        path: req.path,
        query: req.query,
        user: req.user?.id,
        tenant: req.tenant,
        headers: this.sanitizeHeaders(req.headers)
      });
      
      // Override res.end to log response
      const originalEnd = res.end;
      res.end = (...args) => {
        // Log response
        const duration = timer.end({
          correlationId,
          statusCode: res.statusCode,
          method: req.method,
          path: req.path
        });
        
        this.info('Request completed', {
          correlationId,
          method: req.method,
          path: req.path,
          statusCode: res.statusCode,
          duration,
          user: req.user?.id
        });
        
        // Call original end
        originalEnd.apply(res, args);
      };
      
      next();
    };
  }

  /**
   * Create service logger for CAP services
   */
  createServiceLogger(srv) {
    const self = this;
    
    // Before handler for logging
    srv.before('*', (req) => {
      const timer = self.startTimer(`service_${req.event}`);
      req._.timer = timer;
      
      self.debug('Service operation started', {
        correlationId: req._.correlationId,
        service: srv.name,
        event: req.event,
        entity: req.entity,
        user: req.user?.id,
        data: req.data
      });
    });
    
    // After handler for success logging
    srv.after('*', (result, req) => {
      const duration = req._.timer ? req._.timer.end() : 0;
      
      self.info('Service operation completed', {
        correlationId: req._.correlationId,
        service: srv.name,
        event: req.event,
        entity: req.entity,
        duration,
        resultCount: Array.isArray(result) ? result.length : 1
      });
    });
    
    // Error handler
    srv.on('error', (err, req) => {
      const duration = req._.timer ? req._.timer.end() : 0;
      
      self.error('Service operation failed', {
        correlationId: req._.correlationId,
        service: srv.name,
        event: req.event,
        entity: req.entity,
        duration,
        error: err.message,
        stack: err.stack,
        code: err.code
      });
    });
  }

  /**
   * Sanitize headers for logging
   */
  sanitizeHeaders(headers) {
    const sanitized = { ...headers };
    const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];
    
    sensitiveHeaders.forEach(header => {
      if (sanitized[header]) {
        sanitized[header] = '[REDACTED]';
      }
    });
    
    return sanitized;
  }

  /**
   * Log CAP database queries
   */
  logDatabaseQuery(query, duration, success = true) {
    const context = {
      query: this.sanitizeQuery(query),
      duration,
      success
    };
    
    if (duration > 1000) {
      this.warn('Slow database query detected', context);
    } else {
      this.debug('Database query executed', context);
    }
  }

  /**
   * Sanitize database query for logging
   */
  sanitizeQuery(query) {
    if (typeof query === 'string') {
      // Remove potential sensitive data patterns
      return query.replace(/password\s*=\s*'[^']*'/gi, "password='[REDACTED]'");
    }
    return query;
  }
}

/**
 * Logger factory for different services
 */
class LoggerFactory {
  static createAgentLogger(agentId, environment) {
    return new CapStructuredLogger({
      serviceName: `agent-${agentId}`,
      environment,
      logLevel: environment === 'production' ? 'info' : 'debug',
      enableFile: environment === 'production',
      enableMetrics: true
    });
  }

  static createNetworkLogger(environment) {
    return new CapStructuredLogger({
      serviceName: 'a2a-network',
      environment,
      logLevel: environment === 'production' ? 'info' : 'debug',
      enableFile: true,
      enableMetrics: true
    });
  }

  static createWorkflowLogger(environment) {
    return new CapStructuredLogger({
      serviceName: 'a2a-workflow',
      environment,
      logLevel: environment === 'production' ? 'info' : 'debug',
      enableFile: environment === 'production',
      enableMetrics: true
    });
  }
}

module.exports = {
  A2AStructuredLogger,
  CapStructuredLogger,
  LoggerFactory,
  A2A_LOG_LEVELS
};
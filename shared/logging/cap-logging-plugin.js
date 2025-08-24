/**
 * SAP CAP Logging Plugin for A2A Platform
 * Integrates structured logging and performance tracking into CAP services
 */

const cds = require('@sap/cds');
const { CapStructuredLogger, LoggerFactory } = require('./structured-logger');
const { PerformanceTracker, PerformanceCategory } = require('./performance-tracker');

/**
 * CAP Logging Plugin
 */
class CapLoggingPlugin {
  constructor(config = {}) {
    this.config = {
      environment: config.environment || process.env.NODE_ENV || 'development',
      enableRequestLogging: config.enableRequestLogging ?? true,
      enableQueryLogging: config.enableQueryLogging ?? true,
      enablePerformanceTracking: config.enablePerformanceTracking ?? true,
      enableErrorEnrichment: config.enableErrorEnrichment ?? true,
      logSlowQueries: config.logSlowQueries ?? true,
      slowQueryThreshold: config.slowQueryThreshold || 100, // ms
      sensitiveFields: config.sensitiveFields || ['password', 'token', 'secret', 'key']
    };
    
    this.logger = LoggerFactory.createNetworkLogger(this.config.environment);
    this.performanceTracker = new PerformanceTracker({
      enableMetrics: this.config.enablePerformanceTracking,
      enableTracing: this.config.environment !== 'production'
    });
  }

  /**
   * Initialize the plugin
   */
  async init() {
    // Setup request logging
    if (this.config.enableRequestLogging) {
      this.setupRequestLogging();
    }
    
    // Setup query logging
    if (this.config.enableQueryLogging) {
      this.setupQueryLogging();
    }
    
    // Setup service logging
    this.setupServiceLogging();
    
    // Setup error enrichment
    if (this.config.enableErrorEnrichment) {
      this.setupErrorEnrichment();
    }
    
    // Setup performance endpoints
    this.setupPerformanceEndpoints();
    
    this.logger.info('CAP Logging Plugin initialized', {
      environment: this.config.environment,
      features: {
        requestLogging: this.config.enableRequestLogging,
        queryLogging: this.config.enableQueryLogging,
        performanceTracking: this.config.enablePerformanceTracking,
        errorEnrichment: this.config.enableErrorEnrichment
      }
    });
  }

  /**
   * Setup request logging middleware
   */
  setupRequestLogging() {
    const self = this;
    
    cds.on('bootstrap', function(app) {
      // Add correlation ID middleware
      app.use(function(req, res, next) {
        const correlationId = req.get('x-correlation-id') || 
                            req.get('x-request-id') || 
                            cds.utils.uuid();
        
        req.correlationId = correlationId;
        res.set('x-correlation-id', correlationId);
        
        // Start performance tracking
        if (self.config.enablePerformanceTracking) {
          const tracker = self.performanceTracker.startOperation(correlationId, {
            category: PerformanceCategory.API,
            operation: `${req.method} ${req.path}`,
            method: req.method,
            path: req.path,
            user: req.user?.id
          });
          
          req.performanceTracker = tracker;
        }
        
        // Log request
        self.logger.info('HTTP Request', {
          correlationId,
          method: req.method,
          path: req.path,
          query: req.query,
          ip: req.ip,
          userAgent: req.get('user-agent'),
          user: req.user?.id,
          tenant: req.authInfo?.tenant
        });
        
        // Override res.end to log response
        const originalEnd = res.end;
        res.end = function(...args) {
          // End performance tracking
          if (req.performanceTracker) {
            const perfData = req.performanceTracker.end({
              statusCode: res.statusCode,
              error: res.statusCode >= 400
            });
            
            // Log slow requests
            if (perfData.duration > 1000) {
              self.logger.warn('Slow HTTP Request', {
                correlationId,
                duration: perfData.duration,
                method: req.method,
                path: req.path,
                statusCode: res.statusCode
              });
            }
          }
          
          // Log response
          self.logger.info('HTTP Response', {
            correlationId,
            method: req.method,
            path: req.path,
            statusCode: res.statusCode,
            duration: req.performanceTracker?.duration
          });
          
          originalEnd.apply(res, args);
        };
        
        next();
      });
    });
  }

  /**
   * Setup database query logging
   */
  setupQueryLogging() {
    const self = this;
    
    // Intercept database operations
    cds.on('connect', function(service) {
      if (service.constructor.name !== 'DatabaseService') return;
      
      const originalRun = service.run.bind(service);
      
      service.run = async function(query) {
        const queryId = cds.utils.uuid();
        const startTime = Date.now();
        
        // Start performance tracking
        let perfTracker;
        if (self.config.enablePerformanceTracking) {
          perfTracker = self.performanceTracker.startOperation(queryId, {
            category: PerformanceCategory.DATABASE,
            operation: self.getQueryOperation(query),
            entity: self.getQueryEntity(query)
          });
        }
        
        // Log query start
        self.logger.debug('Database Query Start', {
          queryId,
          operation: self.getQueryOperation(query),
          entity: self.getQueryEntity(query),
          query: self.sanitizeQuery(query)
        });
        
        try {
          const result = await originalRun(query);
          const duration = Date.now() - startTime;
          
          // End performance tracking
          if (perfTracker) {
            perfTracker.end({ success: true });
          }
          
          // Log query completion
          const logData = {
            queryId,
            operation: self.getQueryOperation(query),
            entity: self.getQueryEntity(query),
            duration,
            rowCount: Array.isArray(result) ? result.length : 1
          };
          
          if (duration > self.config.slowQueryThreshold) {
            self.logger.warn('Slow Database Query', {
              ...logData,
              query: self.sanitizeQuery(query)
            });
          } else {
            self.logger.debug('Database Query Complete', logData);
          }
          
          return result;
        } catch (error) {
          const duration = Date.now() - startTime;
          
          // End performance tracking
          if (perfTracker) {
            perfTracker.end({ success: false, error: error.message });
          }
          
          // Log query error
          self.logger.error('Database Query Error', {
            queryId,
            operation: self.getQueryOperation(query),
            entity: self.getQueryEntity(query),
            duration,
            error: error.message,
            query: self.sanitizeQuery(query)
          });
          
          throw error;
        }
      };
    });
  }

  /**
   * Setup service-level logging
   */
  setupServiceLogging() {
    const self = this;
    
    cds.on('serving', function(service) {
      // Skip technical services
      if (service.name.startsWith('cds.') || service.name === 'db') return;
      
      self.logger.info('Service registered', {
        service: service.name,
        model: service.model?.name,
        endpoints: Object.keys(service.entities || {})
      });
      
      // Add service logger
      self.logger.createServiceLogger(service);
      
      // Track service operations
      service.before('*', function(req) {
        if (self.config.enablePerformanceTracking && req._) {
          const opId = `${req._.correlationId || cds.utils.uuid()}-${req.event}`;
          const tracker = self.performanceTracker.startOperation(opId, {
            category: PerformanceCategory.SERVICE,
            operation: req.event,
            service: service.name,
            entity: req.entity,
            user: req.user?.id
          });
          
          req._.performanceTracker = tracker;
        }
      });
      
      // Log service operation results
      service.after('*', function(result, req) {
        if (req._.performanceTracker) {
          req._.performanceTracker.end({
            success: true,
            resultCount: Array.isArray(result) ? result.length : 1
          });
        }
      });
    });
  }

  /**
   * Setup error enrichment
   */
  setupErrorEnrichment() {
    const self = this;
    
    // Enhance errors with context
    cds.on('error', function(err, req) {
      // Add correlation ID
      if (req && req._ && req._.correlationId) {
        err.correlationId = req._.correlationId;
      }
      
      // Add request context
      if (req) {
        err.context = {
          service: req.service?.name,
          event: req.event,
          entity: req.entity,
          user: req.user?.id,
          tenant: req.tenant,
          timestamp: new Date().toISOString()
        };
      }
      
      // Log enhanced error
      self.logger.error('Service Error', {
        correlationId: err.correlationId,
        code: err.code || 'UNKNOWN',
        message: err.message,
        stack: err.stack,
        context: err.context,
        details: err.details
      });
    });
  }

  /**
   * Setup performance monitoring endpoints
   */
  setupPerformanceEndpoints() {
    const self = this;
    
    cds.on('bootstrap', (app) => {
      // Health check endpoint with performance data
      app.get('/health/performance', (req, res) => {
        const report = self.performanceTracker.getPerformanceReport();
        res.json({
          status: 'ok',
          timestamp: new Date().toISOString(),
          performance: report
        });
      });
      
      // Metrics endpoint
      app.get('/metrics', (req, res) => {
        const metrics = self.performanceTracker.getMetrics();
        const logs = self.logger.metricsCollector?.getMetrics() || {};
        
        res.json({
          timestamp: new Date().toISOString(),
          performance: metrics,
          logging: logs
        });
      });
      
      // Logging level endpoint
      app.post('/logging/level', (req, res) => {
        const { level } = req.body;
        if (level && ['trace', 'debug', 'info', 'warn', 'error'].includes(level)) {
          self.logger.logger.level = level;
          res.json({ 
            message: 'Logging level updated',
            level 
          });
        } else {
          res.status(400).json({ 
            error: 'Invalid logging level' 
          });
        }
      });
    });
  }

  /**
   * Get query operation type
   */
  getQueryOperation(query) {
    if (query.SELECT) return 'SELECT';
    if (query.INSERT) return 'INSERT';
    if (query.UPDATE) return 'UPDATE';
    if (query.DELETE) return 'DELETE';
    if (query.CREATE) return 'CREATE';
    if (query.DROP) return 'DROP';
    return 'UNKNOWN';
  }

  /**
   * Get query entity
   */
  getQueryEntity(query) {
    return query.SELECT?.from?.ref?.[0] ||
           query.INSERT?.into?.ref?.[0] ||
           query.UPDATE?.entity?.ref?.[0] ||
           query.DELETE?.from?.ref?.[0] ||
           'unknown';
  }

  /**
   * Sanitize query for logging
   */
  sanitizeQuery(query) {
    if (typeof query === 'string') {
      let sanitized = query;
      for (const field of this.config.sensitiveFields) {
        const regex = new RegExp(`${field}\\s*=\\s*['"][^'"]*['"]`, 'gi');
        sanitized = sanitized.replace(regex, `${field}='[REDACTED]'`);
      }
      return sanitized;
    }
    
    // For CQN objects, remove sensitive data
    const sanitized = JSON.parse(JSON.stringify(query));
    this.removeSensitiveData(sanitized);
    return sanitized;
  }

  /**
   * Remove sensitive data from objects
   */
  removeSensitiveData(obj) {
    if (!obj || typeof obj !== 'object') return;
    
    for (const key of Object.keys(obj)) {
      if (this.config.sensitiveFields.some(field => 
        key.toLowerCase().includes(field.toLowerCase())
      )) {
        obj[key] = '[REDACTED]';
      } else if (typeof obj[key] === 'object') {
        this.removeSensitiveData(obj[key]);
      }
    }
  }
}

/**
 * Create and initialize logging plugin
 */
async function setupLogging(config) {
  const plugin = new CapLoggingPlugin(config);
  await plugin.init();
  return plugin;
}

module.exports = {
  CapLoggingPlugin,
  setupLogging
};
const cds = require('@sap/cds');
const { ApplicationLogging } = require('@sap/logging');
const winston = require('winston');

/**
 * Application Logging Service Integration
 * Provides structured logging for monitoring and debugging
 */
class LoggingService {
    
    constructor() {
        this.logger = null;
        this.appLogging = null;
        this.init();
    }

    async init() {
        try {
            // Initialize SAP Application Logging Service
            this.appLogging = new ApplicationLogging({
                level: process.env.LOG_LEVEL || 'info',
                format: 'json',
                service: 'a2a-developer-portal',
                version: '1.0.0'
            });

            // Initialize Winston logger with custom configuration
            this.logger = winston.createLogger({
                level: process.env.LOG_LEVEL || 'info',
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.errors({ stack: true }),
                    winston.format.json(),
                    winston.format.printf(({ timestamp, level, message, ...meta }) => {
                        return JSON.stringify({
                            timestamp,
                            level,
                            message,
                            service: 'a2a-developer-portal',
                            correlationId: meta.correlationId || this.generateCorrelationId(),
                            userId: meta.userId,
                            sessionId: meta.sessionId,
                            component: meta.component || 'unknown',
                            operation: meta.operation,
                            duration: meta.duration,
                            metadata: meta
                        });
                    })
                ),
                transports: [
                    // Console transport for local development
                    new winston.transports.Console({
                        format: winston.format.combine(
                            winston.format.colorize(),
                            winston.format.simple()
                        )
                    }),
                    
                    // File transport for persistent logging
                    new winston.transports.File({
                        filename: 'logs/error.log',
                        level: 'error',
                        maxsize: 10485760, // 10MB
                        maxFiles: 5
                    }),
                    
                    new winston.transports.File({
                        filename: 'logs/combined.log',
                        maxsize: 10485760, // 10MB
                        maxFiles: 10
                    })
                ],
                
                // Exception and rejection handling
                exceptionHandlers: [
                    new winston.transports.File({ filename: 'logs/exceptions.log' })
                ],
                rejectionHandlers: [
                    new winston.transports.File({ filename: 'logs/rejections.log' })
                ]
            });

            // Add SAP Application Logging transport if available
            if (this.appLogging) {
                this.logger.add(new winston.transports.Stream({
                    stream: this.appLogging.getLogStream()
                }));
            }

            console.log('Application Logging Service initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Application Logging Service:', error);
        }
    }

    /**
     * Log info message
     * @param {string} message - Log message
     * @param {object} metadata - Additional metadata
     */
    info(message, metadata = {}) {
        this.logger.info(message, {
            ...metadata,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log warning message
     * @param {string} message - Log message
     * @param {object} metadata - Additional metadata
     */
    warn(message, metadata = {}) {
        this.logger.warn(message, {
            ...metadata,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log error message
     * @param {string} message - Log message
     * @param {object} metadata - Additional metadata
     */
    error(message, metadata = {}) {
        this.logger.error(message, {
            ...metadata,
            timestamp: new Date().toISOString(),
            stack: metadata.error?.stack
        });
    }

    /**
     * Log debug message
     * @param {string} message - Log message
     * @param {object} metadata - Additional metadata
     */
    debug(message, metadata = {}) {
        this.logger.debug(message, {
            ...metadata,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log business event
     * @param {string} event - Event name
     * @param {object} data - Event data
     * @param {object} context - Event context
     */
    logBusinessEvent(event, data, context = {}) {
        this.info(`Business Event: ${event}`, {
            eventType: 'business',
            event,
            data,
            ...context
        });
    }

    /**
     * Log performance metrics
     * @param {string} operation - Operation name
     * @param {number} duration - Duration in milliseconds
     * @param {object} metadata - Additional metadata
     */
    logPerformance(operation, duration, metadata = {}) {
        this.info(`Performance: ${operation}`, {
            eventType: 'performance',
            operation,
            duration,
            ...metadata
        });
    }

    /**
     * Log security event
     * @param {string} event - Security event
     * @param {object} details - Event details
     * @param {object} context - Security context
     */
    logSecurityEvent(event, details, context = {}) {
        this.warn(`Security Event: ${event}`, {
            eventType: 'security',
            event,
            details,
            ...context
        });
    }

    /**
     * Log API request/response
     * @param {object} req - Request object
     * @param {object} res - Response object
     * @param {number} duration - Request duration
     */
    logApiRequest(req, res, duration) {
        const logData = {
            eventType: 'api',
            method: req.method,
            url: req.url,
            statusCode: res.statusCode,
            duration,
            userAgent: req.get('User-Agent'),
            ip: req.ip,
            userId: req.user?.id,
            sessionId: req.session?.id
        };

        if (res.statusCode >= 400) {
            this.error(`API Error: ${req.method} ${req.url}`, logData);
        } else {
            this.info(`API Request: ${req.method} ${req.url}`, logData);
        }
    }

    /**
     * Log database operation
     * @param {string} operation - Database operation
     * @param {string} entity - Entity name
     * @param {number} duration - Operation duration
     * @param {object} metadata - Additional metadata
     */
    logDatabaseOperation(operation, entity, duration, metadata = {}) {
        this.debug(`Database: ${operation} ${entity}`, {
            eventType: 'database',
            operation,
            entity,
            duration,
            ...metadata
        });
    }

    /**
     * Log workflow event
     * @param {string} workflowId - Workflow ID
     * @param {string} event - Workflow event
     * @param {object} data - Event data
     */
    logWorkflowEvent(workflowId, event, data = {}) {
        this.info(`Workflow Event: ${event}`, {
            eventType: 'workflow',
            workflowId,
            event,
            data
        });
    }

    /**
     * Log deployment event
     * @param {string} deploymentId - Deployment ID
     * @param {string} event - Deployment event
     * @param {object} data - Event data
     */
    logDeploymentEvent(deploymentId, event, data = {}) {
        this.info(`Deployment Event: ${event}`, {
            eventType: 'deployment',
            deploymentId,
            event,
            data
        });
    }

    /**
     * Create child logger with context
     * @param {object} context - Logger context
     * @returns {object} Child logger
     */
    createChildLogger(context) {
        return {
            info: (message, metadata = {}) => this.info(message, { ...context, ...metadata }),
            warn: (message, metadata = {}) => this.warn(message, { ...context, ...metadata }),
            error: (message, metadata = {}) => this.error(message, { ...context, ...metadata }),
            debug: (message, metadata = {}) => this.debug(message, { ...context, ...metadata })
        };
    }

    /**
     * Start performance timer
     * @param {string} operation - Operation name
     * @returns {function} Timer end function
     */
    startTimer(operation) {
        const startTime = Date.now();
        
        return (metadata = {}) => {
            const duration = Date.now() - startTime;
            this.logPerformance(operation, duration, metadata);
            return duration;
        };
    }

    /**
     * Log with correlation ID
     * @param {string} correlationId - Correlation ID
     * @param {string} level - Log level
     * @param {string} message - Log message
     * @param {object} metadata - Additional metadata
     */
    logWithCorrelation(correlationId, level, message, metadata = {}) {
        this[level](message, {
            ...metadata,
            correlationId
        });
    }

    /**
     * Get log statistics
     * @param {string} timeframe - Timeframe (1h, 24h, 7d)
     * @returns {object} Log statistics
     */
    async getLogStatistics(timeframe = '24h') {
        try {
            // This would typically query the logging backend
            // For now, return mock statistics
            return {
                timeframe,
                totalLogs: 15420,
                errorCount: 23,
                warningCount: 156,
                infoCount: 14890,
                debugCount: 351,
                topErrors: [
                    { message: 'Database connection timeout', count: 8 },
                    { message: 'Authentication failed', count: 7 },
                    { message: 'Workflow execution failed', count: 5 }
                ],
                performanceMetrics: {
                    averageResponseTime: 245,
                    slowestOperations: [
                        { operation: 'project-search', avgDuration: 1200 },
                        { operation: 'agent-deployment', avgDuration: 890 },
                        { operation: 'workflow-execution', avgDuration: 650 }
                    ]
                }
            };
        } catch (error) {
            this.error('Failed to get log statistics', { error: error.message });
            throw error;
        }
    }

    /**
     * Search logs
     * @param {object} criteria - Search criteria
     * @returns {array} Log entries
     */
    async searchLogs(criteria) {
        try {
            const {
                level,
                component,
                userId,
                startTime,
                endTime,
                message,
                limit = 100
            } = criteria;

            // This would typically query the logging backend
            // For now, return mock results
            return [
                {
                    timestamp: new Date().toISOString(),
                    level: 'info',
                    message: 'Project created successfully',
                    component: 'project-service',
                    userId: 'user123',
                    metadata: { projectId: 'proj-456' }
                }
            ];
        } catch (error) {
            this.error('Failed to search logs', { error: error.message, criteria });
            throw error;
        }
    }

    /**
     * Configure log retention
     * @param {object} config - Retention configuration
     */
    configureRetention(config) {
        const {
            errorLogRetention = '30d',
            infoLogRetention = '7d',
            debugLogRetention = '1d'
        } = config;

        this.info('Log retention configured', {
            errorLogRetention,
            infoLogRetention,
            debugLogRetention
        });
    }

    /**
     * Export logs
     * @param {object} criteria - Export criteria
     * @returns {string} Export file path
     */
    async exportLogs(criteria) {
        try {
            const logs = await this.searchLogs(criteria);
            const exportPath = `logs/export_${Date.now()}.json`;
            
            // In a real implementation, this would write to file system or cloud storage
            this.info('Logs exported', { 
                exportPath, 
                logCount: logs.length,
                criteria 
            });
            
            return exportPath;
        } catch (error) {
            this.error('Failed to export logs', { error: error.message, criteria });
            throw error;
        }
    }

    // Private helper methods
    generateCorrelationId() {
        return `corr-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Flush logs (for graceful shutdown)
     */
    async flush() {
        return new Promise((resolve) => {
            this.logger.end(() => {
                console.log('Logger flushed successfully');
                resolve();
            });
        });
    }
}

// Create singleton instance
const loggingService = new LoggingService();

// Express middleware for request logging
const requestLoggingMiddleware = (req, res, next) => {
    const startTime = Date.now();
    
    // Generate correlation ID for request
    req.correlationId = loggingService.generateCorrelationId();
    res.set('X-Correlation-ID', req.correlationId);
    
    // Log request start
    loggingService.debug('Request started', {
        correlationId: req.correlationId,
        method: req.method,
        url: req.url,
        userAgent: req.get('User-Agent'),
        ip: req.ip
    });
    
    // Override res.end to log response
    const originalEnd = res.end;
    res.end = function(...args) {
        const duration = Date.now() - startTime;
        loggingService.logApiRequest(req, res, duration);
        originalEnd.apply(this, args);
    };
    
    next();
};

module.exports = {
    loggingService,
    requestLoggingMiddleware
};

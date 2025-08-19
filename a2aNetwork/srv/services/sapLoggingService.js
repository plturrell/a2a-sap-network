/**
 * Enterprise Logging Service
 * 
 * Provides structured logging with correlation IDs, different log levels,
 * and integration with SAP Cloud Logging service following enterprise standards.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */

const winston = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
const path = require('path');
const os = require('os');
const { v4: uuidv4 } = require('uuid');

class LoggingService {
    constructor() {
        this.logger = null;
        this.loggers = new Map();
        this.logBuffer = [];
        this.maxBufferSize = 1000;
        this.correlationStore = new Map();
        
        this._initializeLogger();
    \n        this.intervals = new Map(); // Track intervals for cleanup}

    /**
     * Initialize Winston logger with multiple transports
     * @private
     */
    _initializeLogger() {
        // Custom log format for enterprise applications
        const enterpriseFormat = winston.format.combine(
            winston.format.timestamp({
                format: 'YYYY-MM-DD HH:mm:ss.SSS'
            }),
            winston.format.errors({ stack: true }),
            winston.format.json(),
            winston.format.printf(info => {
                const logEntry = {
                    timestamp: info.timestamp,
                    level: info.level.toUpperCase(),
                    message: info.message,
                    correlationId: info.correlationId || 'no-correlation',
                    sessionId: info.sessionId || 'no-session',
                    userId: info.userId || 'anonymous',
                    component: info.component || 'a2a-network',
                    service: info.service || 'main',
                    environment: process.env.NODE_ENV || 'development',
                    hostname: os.hostname(),
                    pid: process.pid,
                    memory: info.includeMemory ? this._getMemoryUsage() : undefined,
                    duration: info.duration,
                    statusCode: info.statusCode,
                    method: info.method,
                    url: info.url,
                    userAgent: info.userAgent,
                    stack: info.stack,
                    metadata: info.metadata
                };
                
                // Remove undefined fields
                Object.keys(logEntry).forEach(key => {
                    if (logEntry[key] === undefined) {
                        delete logEntry[key];
                    }
                });
                
                return JSON.stringify(logEntry);
            })
        );

        // Create transports
        const transports = [
            // Console transport for development
            new winston.transports.Console({
                level: process.env.LOG_LEVEL || 'info',
                format: process.env.NODE_ENV === 'development' 
                    ? winston.format.combine(
                        winston.format.colorize(),
                        winston.format.timestamp({ format: 'HH:mm:ss' }),
                        winston.format.printf(info => 
                            `${info.timestamp} [${info.level}] ${info.correlationId ? `[${info.correlationId}] ` : ''}${info.message}`
                        )
                    )
                    : enterpriseFormat
            })
        ];

        // File transports for production
        if (process.env.NODE_ENV === 'production' || process.env.ENABLE_FILE_LOGGING === 'true') {
            // Application logs
            transports.push(new DailyRotateFile({
                filename: path.join('logs', 'a2a-application-%DATE%.log'),
                datePattern: 'YYYY-MM-DD',
                maxSize: '100m',
                maxFiles: '30d',
                level: 'info',
                format: enterpriseFormat,
                createSymlink: true,
                symlinkName: 'a2a-application-current.log'
            }));

            // Error logs
            transports.push(new DailyRotateFile({
                filename: path.join('logs', 'a2a-error-%DATE%.log'),
                datePattern: 'YYYY-MM-DD',
                maxSize: '100m',
                maxFiles: '90d',
                level: 'error',
                format: enterpriseFormat,
                createSymlink: true,
                symlinkName: 'a2a-error-current.log'
            }));

            // Security audit logs
            transports.push(new DailyRotateFile({
                filename: path.join('logs', 'a2a-security-%DATE%.log'),
                datePattern: 'YYYY-MM-DD',
                maxSize: '100m',
                maxFiles: '365d', // Keep security logs for 1 year
                level: 'warn',
                format: enterpriseFormat,
                createSymlink: true,
                symlinkName: 'a2a-security-current.log',
                // Only log security-related messages
                filter: info => info.component === 'security' || info.category === 'security'
            }));

            // Performance logs
            transports.push(new DailyRotateFile({
                filename: path.join('logs', 'a2a-performance-%DATE%.log'),
                datePattern: 'YYYY-MM-DD',
                maxSize: '50m',
                maxFiles: '14d',
                format: enterpriseFormat,
                createSymlink: true,
                symlinkName: 'a2a-performance-current.log',
                // Only log performance-related messages
                filter: info => info.category === 'performance' || info.duration !== undefined
            }));
        }

        // Create the main logger
        this.logger = winston.createLogger({
            level: process.env.LOG_LEVEL || 'info',
            transports: transports,
            exitOnError: false,
            handleExceptions: true,
            handleRejections: true
        });

        // Handle uncaught exceptions
        this.logger.exceptions.handle(
            new winston.transports.Console({
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.errors({ stack: true }),
                    winston.format.json()
                )
            })
        );
    }

    /**
     * Get memory usage information
     * @private
     * @returns {object} Memory usage stats
     */
    _getMemoryUsage() {
        const memUsage = process.memoryUsage();
        return {
            rss: Math.round(memUsage.rss / 1024 / 1024), // MB
            heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024), // MB
            heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024), // MB
            external: Math.round(memUsage.external / 1024 / 1024) // MB
        };
    }

    /**
     * Create a correlation context for tracking related operations
     * @param {string} [correlationId] Optional correlation ID
     * @returns {string} Correlation ID
     */
    createCorrelation(correlationId = null) {
        const id = correlationId || uuidv4();
        this.correlationStore.set(id, {
            id,
            created: Date.now(),
            requests: 0,
            errors: 0
        });
        return id;
    }

    /**
     * Log with correlation context
     * @param {string} level Log level
     * @param {string} message Log message
     * @param {object} context Additional context
     */
    log(level, message, context = {}) {
        const logContext = {
            ...context,
            correlationId: context.correlationId || 'no-correlation',
            timestamp: new Date().toISOString()
        };

        // Update correlation tracking
        if (logContext.correlationId !== 'no-correlation') {
            const correlation = this.correlationStore.get(logContext.correlationId);
            if (correlation) {
                correlation.requests++;
                if (level === 'error') {
                    correlation.errors++;
                }
            }
        }

        // Add to buffer for recent logs API
        this.logBuffer.push({
            timestamp: logContext.timestamp,
            level: level.toUpperCase(),
            message,
            ...logContext
        });

        // Maintain buffer size
        if (this.logBuffer.length > this.maxBufferSize) {
            this.logBuffer.shift();
        }

        this.logger.log(level, message, logContext);
    }

    /**
     * Log debug message
     * @param {string} message Log message
     * @param {object} context Additional context
     */
    debug(message, context = {}) {
        this.log('debug', message, context);
    }

    /**
     * Log info message
     * @param {string} message Log message
     * @param {object} context Additional context
     */
    info(message, context = {}) {
        this.log('info', message, context);
    }

    /**
     * Log warning message
     * @param {string} message Log message
     * @param {object} context Additional context
     */
    warn(message, context = {}) {
        this.log('warn', message, context);
    }

    /**
     * Log error message
     * @param {string} message Log message
     * @param {Error|object} context Error object or additional context
     */
    error(message, context = {}) {
        const errorContext = context instanceof Error ? {
            stack: context.stack,
            name: context.name,
            message: context.message,
            ...context
        } : context;

        this.log('error', message, errorContext);
    }

    /**
     * Log security audit message
     * @param {string} message Security message
     * @param {object} context Security context
     */
    security(message, context = {}) {
        this.log('warn', message, {
            ...context,
            component: 'security',
            category: 'security'
        });
    }

    /**
     * Log performance metrics
     * @param {string} operation Operation name
     * @param {number} duration Duration in milliseconds
     * @param {object} context Additional context
     */
    performance(operation, duration, context = {}) {
        this.log('info', `Performance: ${operation}`, {
            ...context,
            category: 'performance',
            operation,
            duration: `${duration}ms`,
            durationMs: duration
        });
    }

    /**
     * Log HTTP request
     * @param {object} req Express request object
     * @param {object} res Express response object
     * @param {number} duration Request duration in ms
     */
    httpRequest(req, res, duration) {
        const context = {
            method: req.method,
            url: req.originalUrl || req.url,
            statusCode: res.statusCode,
            userAgent: req.get('User-Agent'),
            ip: req.ip || req.connection.remoteAddress,
            correlationId: req.correlationId,
            sessionId: req.sessionID,
            userId: req.user?.id || req.user?.email,
            duration: `${duration}ms`,
            durationMs: duration,
            category: 'http'
        };

        const level = res.statusCode >= 400 ? 'error' : 'info';
        this.log(level, `HTTP ${req.method} ${req.url} ${res.statusCode}`, context);
    }

    /**
     * Log blockchain operation
     * @param {string} operation Blockchain operation
     * @param {string} transactionHash Transaction hash
     * @param {object} context Additional context
     */
    blockchain(operation, transactionHash, context = {}) {
        this.log('info', `Blockchain: ${operation}`, {
            ...context,
            category: 'blockchain',
            operation,
            transactionHash
        });
    }

    /**
     * Log business process event
     * @param {string} process Process name
     * @param {string} event Event name
     * @param {object} context Additional context
     */
    business(process, event, context = {}) {
        this.log('info', `Business Process: ${process} - ${event}`, {
            ...context,
            category: 'business',
            process,
            event
        });
    }

    /**
     * Get recent logs from buffer
     * @param {number} limit Number of logs to return
     * @param {string} level Minimum log level
     * @returns {Array} Recent log entries
     */
    getRecentLogs(limit = 100, level = null) {
        let logs = [...this.logBuffer];
        
        if (level) {
            const levels = ['debug', 'info', 'warn', 'error'];
            const minLevelIndex = levels.indexOf(level.toLowerCase());
            if (minLevelIndex !== -1) {
                const allowedLevels = levels.slice(minLevelIndex).map(l => l.toUpperCase());
                logs = logs.filter(log => allowedLevels.includes(log.level));
            }
        }
        
        return logs.slice(-limit).reverse();
    }

    /**
     * Get correlation statistics
     * @param {string} correlationId Correlation ID
     * @returns {object|null} Correlation stats
     */
    getCorrelationStats(correlationId) {
        return this.correlationStore.get(correlationId) || null;
    }

    /**
     * Clean up old correlations
     * @param {number} maxAge Maximum age in milliseconds (default: 1 hour)
     */
    cleanupCorrelations(maxAge = 3600000) {
        const cutoff = Date.now() - maxAge;
        for (const [id, correlation] of this.correlationStore) {
            if (correlation.created < cutoff) {
                this.correlationStore.delete(id);
            }
        }
    }

    /**
     * Create a child logger for a specific component
     * @param {string} component Component name
     * @returns {object} Child logger
     */
    child(component) {
        if (this.loggers.has(component)) {
            return this.loggers.get(component);
        }

        const childLogger = {
            debug: (message, context = {}) => this.debug(message, { ...context, component }),
            info: (message, context = {}) => this.info(message, { ...context, component }),
            warn: (message, context = {}) => this.warn(message, { ...context, component }),
            error: (message, context = {}) => this.error(message, { ...context, component }),
            security: (message, context = {}) => this.security(message, { ...context, component }),
            performance: (operation, duration, context = {}) => 
                this.performance(operation, duration, { ...context, component })
        };

        this.loggers.set(component, childLogger);
        return childLogger;
    }

    /**
     * Express middleware for request logging
     * @returns {Function} Express middleware
     */
    middleware() {
        return (req, res, next) => {
            const startTime = Date.now();
            
            // Ensure correlation ID exists
            if (!req.correlationId) {
                req.correlationId = this.createCorrelation();
            }

            // Log request start
            this.debug(`Request started: ${req.method} ${req.url}`, {
                correlationId: req.correlationId,
                method: req.method,
                url: req.url,
                userAgent: req.get('User-Agent'),
                ip: req.ip || req.connection.remoteAddress
            });

            // Use response event listeners instead of overriding res.end to avoid OpenTelemetry conflicts
            res.on('finish', () => {
                const duration = Date.now() - startTime;
                
                // Log request completion
                loggingService.httpRequest(req, res, duration);
            });

            next();
        };
    }
}

// Create singleton instance
const loggingService = new LoggingService();

// Clean up correlations every hour
this.intervals.set('interval_471', (function(intervalId) { this.intervals.add(intervalId); return intervalId; }).call(this, setInterval(() => {
    loggingService.cleanupCorrelations();
}, 3600000));

module.exports = loggingService;
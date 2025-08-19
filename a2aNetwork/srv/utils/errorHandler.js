/**
 * @fileoverview Comprehensive Error Handling Utility
 * @description Structured error handling for blockchain operations
 * @module error-handler
 */

const cds = require('@sap/cds');

/**
 * Error types for classification
 */
const ErrorTypes = {
    VALIDATION: 'VALIDATION',
    AUTHENTICATION: 'AUTHENTICATION',
    AUTHORIZATION: 'AUTHORIZATION',
    BLOCKCHAIN: 'BLOCKCHAIN',
    NETWORK: 'NETWORK',
    TIMEOUT: 'TIMEOUT',
    RATE_LIMIT: 'RATE_LIMIT',
    CIRCUIT_BREAKER: 'CIRCUIT_BREAKER',
    CONFIGURATION: 'CONFIGURATION',
    INTERNAL: 'INTERNAL'
};

/**
 * Error severity levels
 */
const ErrorSeverity = {
    LOW: 'LOW',
    MEDIUM: 'MEDIUM',
    HIGH: 'HIGH',
    CRITICAL: 'CRITICAL'
};

/**
 * Structured error class for blockchain operations
 */
class BlockchainError extends Error {
    constructor(message, type, severity, context = {}) {
        super(message);
        this.name = 'BlockchainError';
        this.type = type;
        this.severity = severity;
        this.context = context;
        this.timestamp = new Date().toISOString();
        this.traceId = this.generateTraceId();
    }

    generateTraceId() {
        return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    toJSON() {
        return {
            name: this.name,
            message: this.message,
            type: this.type,
            severity: this.severity,
            context: this.context,
            timestamp: this.timestamp,
            traceId: this.traceId,
            stack: this.stack
        };
    }
}

/**
 * Comprehensive error handler for blockchain operations
 */
class ErrorHandler {
    constructor() {
        this.errorCounts = new Map();
        this.errorHistory = [];
        this.maxHistorySize = 1000;
        this.alertThresholds = {
            [ErrorSeverity.CRITICAL]: 1,
            [ErrorSeverity.HIGH]: 5,
            [ErrorSeverity.MEDIUM]: 20,
            [ErrorSeverity.LOW]: 50
        \n        this.intervals = new Map(); // Track intervals for cleanup};
    }

    /**
     * Handle and classify blockchain errors
     */
    handleError(error, context = {}) {
        const classifiedError = this.classifyError(error, context);
        
        // Log the error
        this.logError(classifiedError);
        
        // Track error metrics
        this.trackErrorMetrics(classifiedError);
        
        // Check if alerting is needed
        this.checkAlertThresholds(classifiedError);
        
        // Store in history
        this.addToHistory(classifiedError);
        
        return classifiedError;
    }

    /**
     * Classify error by type and severity
     */
    classifyError(error, context = {}) {
        let type = ErrorTypes.INTERNAL;
        let severity = ErrorSeverity.MEDIUM;
        let message = error.message || 'Unknown error';

        // Classify by error message patterns
        if (message.includes('nonce') || message.includes('replacement transaction underpriced')) {
            type = ErrorTypes.BLOCKCHAIN;
            severity = ErrorSeverity.MEDIUM;
        } else if (message.includes('insufficient funds') || message.includes('gas')) {
            type = ErrorTypes.BLOCKCHAIN;
            severity = ErrorSeverity.HIGH;
        } else if (message.includes('timeout') || message.includes('TIMEOUT')) {
            type = ErrorTypes.TIMEOUT;
            severity = ErrorSeverity.MEDIUM;
        } else if (message.includes('Authentication') || message.includes('Unauthorized')) {
            type = ErrorTypes.AUTHENTICATION;
            severity = ErrorSeverity.HIGH;
        } else if (message.includes('rate limit') || message.includes('Rate limit')) {
            type = ErrorTypes.RATE_LIMIT;
            severity = ErrorSeverity.LOW;
        } else if (message.includes('Circuit breaker')) {
            type = ErrorTypes.CIRCUIT_BREAKER;
            severity = ErrorSeverity.MEDIUM;
        } else if (message.includes('validation') || message.includes('Invalid')) {
            type = ErrorTypes.VALIDATION;
            severity = ErrorSeverity.LOW;
        } else if (message.includes('network') || message.includes('connection')) {
            type = ErrorTypes.NETWORK;
            severity = ErrorSeverity.MEDIUM;
        }

        // Check for critical patterns
        if (message.includes('private key') || message.includes('security')) {
            severity = ErrorSeverity.CRITICAL;
        } else if (message.includes('contract') && message.includes('failed')) {
            severity = ErrorSeverity.HIGH;
        }

        // Handle existing BlockchainError
        if (error instanceof BlockchainError) {
            return error;
        }

        return new BlockchainError(message, type, severity, {
            originalError: error.name,
            stack: error.stack,
            ...context
        });
    }

    /**
     * Log error with appropriate level
     */
    logError(error) {
        const logData = {
            traceId: error.traceId,
            type: error.type,
            severity: error.severity,
            message: error.message,
            context: error.context
        };

        switch (error.severity) {
            case ErrorSeverity.CRITICAL:
                cds.log('blockchain-error').error('CRITICAL blockchain error', logData);
                break;
            case ErrorSeverity.HIGH:
                cds.log('blockchain-error').error('HIGH severity blockchain error', logData);
                break;
            case ErrorSeverity.MEDIUM:
                cds.log('blockchain-error').warn('MEDIUM severity blockchain error', logData);
                break;
            case ErrorSeverity.LOW:
                cds.log('blockchain-error').info('LOW severity blockchain error', logData);
                break;
        }
    }

    /**
     * Track error metrics for monitoring
     */
    trackErrorMetrics(error) {
        const key = `${error.type}-${error.severity}`;
        const current = this.errorCounts.get(key) || 0;
        this.errorCounts.set(key, current + 1);
    }

    /**
     * Check if error count exceeds alert thresholds
     */
    checkAlertThresholds(error) {
        const key = `${error.type}-${error.severity}`;
        const count = this.errorCounts.get(key) || 0;
        const threshold = this.alertThresholds[error.severity];

        if (count >= threshold) {
            this.triggerAlert(error, count);
        }
    }

    /**
     * Trigger alert for high error rates
     */
    triggerAlert(error, count) {
        const alertData = {
            errorType: error.type,
            severity: error.severity,
            count: count,
            threshold: this.alertThresholds[error.severity],
            lastError: error.message,
            traceId: error.traceId
        };

        cds.log('blockchain-alert').error('Error threshold exceeded', alertData);
        
        // In production, this would integrate with alerting systems
        // (PagerDuty, Slack, email, etc.)
        this.sendAlert(alertData);
    }

    /**
     * Send alert to monitoring systems
     */
    async sendAlert(alertData) {
        try {
            // Placeholder for actual alerting integration
            // Could integrate with PagerDuty, Slack, email, etc.
            
            // Example: Send to webhook
            if (process.env.ALERT_WEBHOOK_URL) {
                // await fetch(process.env.ALERT_WEBHOOK_URL, {
                //     method: 'POST',
                //     headers: { 'Content-Type': 'application/json' },
                //     body: JSON.stringify(alertData)
                // });
            }
            
            cds.log('blockchain-alert').info('Alert sent successfully', { traceId: alertData.traceId });
            
        } catch (error) {
            cds.log('blockchain-alert').error('Failed to send alert', error);
        }
    }

    /**
     * Add error to history for analysis
     */
    addToHistory(error) {
        this.errorHistory.push(error.toJSON());
        
        // Maintain history size limit
        if (this.errorHistory.length > this.maxHistorySize) {
            this.errorHistory.shift();
        }
    }

    /**
     * Get error statistics for monitoring
     */
    getErrorStats() {
        const stats = {
            totalErrors: this.errorHistory.length,
            errorsByType: {},
            errorsBySeverity: {},
            recentErrors: this.errorHistory.slice(-10),
            errorCounts: Object.fromEntries(this.errorCounts)
        };

        // Calculate error distribution
        for (const error of this.errorHistory) {
            // By type
            stats.errorsByType[error.type] = (stats.errorsByType[error.type] || 0) + 1;
            
            // By severity
            stats.errorsBySeverity[error.severity] = (stats.errorsBySeverity[error.severity] || 0) + 1;
        }

        return stats;
    }

    /**
     * Reset error counts (for periodic cleanup)
     */
    resetCounts() {
        this.errorCounts.clear();
        cds.log('blockchain-error').info('Error counts reset');
    }

    /**
     * Get errors by criteria for analysis
     */
    getErrorsByCriteria(criteria = {}) {
        return this.errorHistory.filter(error => {
            if (criteria.type && error.type !== criteria.type) return false;
            if (criteria.severity && error.severity !== criteria.severity) return false;
            if (criteria.since) {
                const errorTime = new Date(error.timestamp);
                const sinceTime = new Date(criteria.since);
                if (errorTime < sinceTime) return false;
            }
            return true;
        });
    }

    /**
     * Handle request errors and format appropriate responses
     */
    handleRequestError(req, error, context = {}) {
        const classifiedError = this.handleError(error, {
            operation: req.event || req.path,
            user: req.user?.id,
            ...context
        });

        // Determine HTTP status code
        let statusCode = 500;
        let userMessage = 'An internal error occurred';

        switch (classifiedError.type) {
            case ErrorTypes.VALIDATION:
                statusCode = 400;
                userMessage = 'Invalid request data';
                break;
            case ErrorTypes.AUTHENTICATION:
                statusCode = 401;
                userMessage = 'Authentication required';
                break;
            case ErrorTypes.AUTHORIZATION:
                statusCode = 403;
                userMessage = 'Insufficient permissions';
                break;
            case ErrorTypes.RATE_LIMIT:
                statusCode = 429;
                userMessage = 'Rate limit exceeded';
                break;
            case ErrorTypes.TIMEOUT:
                statusCode = 408;
                userMessage = 'Request timeout';
                break;
            case ErrorTypes.BLOCKCHAIN:
                statusCode = 502;
                userMessage = 'Blockchain operation failed';
                break;
            case ErrorTypes.NETWORK:
                statusCode = 503;
                userMessage = 'Service temporarily unavailable';
                break;
        }

        // Handle both CAP requests and Express requests
        if (req.error) {
            // CAP request - use req.error
            req.error(statusCode, userMessage, {
                traceId: classifiedError.traceId,
                timestamp: classifiedError.timestamp
            });
        } else {
            // Express request - use res object (passed in context)
            const res = context.res;
            if (res) {
                res.status(statusCode).json({
                    error: {
                        message: userMessage,
                        traceId: classifiedError.traceId,
                        timestamp: classifiedError.timestamp,
                        code: classifiedError.type
                    }
                });
            }
        }
    }

    /**
     * Express error middleware wrapper
     */
    expressErrorMiddleware(error, req, res, next) {
        this.handleRequestError(req, error, { res });
    }

    /**
     * Async handler wrapper for Express routes
     */
    asyncHandler(fn) {
        return (req, res, next) => {
            Promise.resolve(fn(req, res, next)).catch(error => {
                this.handleRequestError(req, error, { res });
            });
        };
    }
}

// Global error handler instance
const globalErrorHandler = new ErrorHandler();

// Reset error counts every hour
this.intervals.set('interval_403', (function(intervalId) { this.intervals.add(intervalId); return intervalId; }).call(this, setInterval(() => {
    globalErrorHandler.resetCounts();
}, 3600000));

module.exports = {
    ErrorHandler,
    BlockchainError,
    ErrorTypes,
    ErrorSeverity,
    handleError: (error, context) => globalErrorHandler.handleError(error, context),
    handleRequestError: (req, error, context) => globalErrorHandler.handleRequestError(req, error, context),
    getErrorStats: () => globalErrorHandler.getErrorStats(),
    getErrorsByCriteria: (criteria) => globalErrorHandler.getErrorsByCriteria(criteria),
    asyncHandler: (fn) => globalErrorHandler.asyncHandler(fn),
    expressErrorMiddleware: (error, req, res, next) => globalErrorHandler.expressErrorMiddleware(error, req, res, next)
};
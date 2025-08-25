/**
 * @fileoverview Comprehensive Error Handler with Trace Integration
 * @description Centralized error handling with full trace context
 * @module errorHandler
 */

const cds = require('@sap/cds');
const { addError, completeTrace } = require('./traceManager');

/**
 * Error Classification and Handling
 */
class ErrorHandler {
    constructor() {
        this.errorCodes = {
            // Authentication Errors
            AUTH_INVALID_TOKEN: { code: 'AUTH_001', status: 401, message: 'Invalid authentication token' },
            AUTH_TOKEN_EXPIRED: { code: 'AUTH_002', status: 401, message: 'Authentication token expired' },
            AUTH_INSUFFICIENT_PERMISSIONS: { code: 'AUTH_003', status: 403, message: 'Insufficient permissions' },

            // Validation Errors
            VALIDATION_INVALID_INPUT: { code: 'VAL_001', status: 400, message: 'Invalid input parameters' },
            VALIDATION_MISSING_FIELD: { code: 'VAL_002', status: 400, message: 'Required field missing' },
            VALIDATION_INVALID_FORMAT: { code: 'VAL_003', status: 400, message: 'Invalid data format' },

            // Business Logic Errors
            BUSINESS_RULE_VIOLATION: { code: 'BUS_001', status: 422, message: 'Business rule violation' },
            BUSINESS_WORKFLOW_ERROR: { code: 'BUS_002', status: 422, message: 'Workflow execution error' },

            // Database Errors
            DB_CONNECTION_ERROR: { code: 'DB_001', status: 503, message: 'Database connection error' },
            DB_QUERY_ERROR: { code: 'DB_002', status: 500, message: 'Database query error' },
            DB_CONSTRAINT_VIOLATION: { code: 'DB_003', status: 409, message: 'Database constraint violation' },

            // External Service Errors
            EXT_SERVICE_UNAVAILABLE: { code: 'EXT_001', status: 503, message: 'External service unavailable' },
            EXT_SERVICE_TIMEOUT: { code: 'EXT_002', status: 504, message: 'External service timeout' },
            EXT_BLOCKCHAIN_ERROR: { code: 'EXT_003', status: 502, message: 'Blockchain service error' },

            // System Errors
            SYS_INTERNAL_ERROR: { code: 'SYS_001', status: 500, message: 'Internal system error' },
            SYS_CONFIG_ERROR: { code: 'SYS_002', status: 500, message: 'System configuration error' },
            SYS_RESOURCE_EXHAUSTED: { code: 'SYS_003', status: 503, message: 'System resources exhausted' }
        };
    }

    /**
     * Main error handling middleware
     */
    handleError(error, req, res, next) {
        const traceId = req.traceId;
        const log = cds.log('error-handler');

        // Classify error
        const errorInfo = this.classifyError(error);

        // Add error to trace
        const errorId = addError(traceId, error, {
            component: 'error-handler',
            layer: 'middleware',
            operation: 'error-handling',
            severity: this.getSeverity(errorInfo.status),
            data: {
                url: req.originalUrl,
                method: req.method,
                userAgent: req.get('User-Agent')
            }
        });

        // Complete trace with error status
        completeTrace(traceId, errorInfo.status);

        // Log error with full context
        log.error('Request error occurred', {
            traceId,
            errorId,
            errorCode: errorInfo.code,
            statusCode: errorInfo.status,
            message: errorInfo.message,
            originalError: error.message,
            stack: error.stack,
            url: req.originalUrl,
            method: req.method,
            user: req.user?.id || 'anonymous'
        });

        // Send error response
        this.sendErrorResponse(res, errorInfo, traceId, errorId);
    }

    /**
     * Classify error based on type and content
     */
    classifyError(error) {
        // Check for specific error patterns
        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            return this.errorCodes.EXT_SERVICE_UNAVAILABLE;
        }

        if (error.code === 'ETIMEOUT') {
            return this.errorCodes.EXT_SERVICE_TIMEOUT;
        }

        if (error.message && error.message.includes('HANA')) {
            return this.errorCodes.DB_CONNECTION_ERROR;
        }

        if (error.statusCode === 401) {
            return this.errorCodes.AUTH_INVALID_TOKEN;
        }

        if (error.statusCode === 403) {
            return this.errorCodes.AUTH_INSUFFICIENT_PERMISSIONS;
        }

        if (error.statusCode === 400) {
            return this.errorCodes.VALIDATION_INVALID_INPUT;
        }

        if (error.name === 'ValidationError') {
            return this.errorCodes.VALIDATION_INVALID_INPUT;
        }

        if (error.name === 'SequelizeConstraintError') {
            return this.errorCodes.DB_CONSTRAINT_VIOLATION;
        }

        // Default to internal error
        return this.errorCodes.SYS_INTERNAL_ERROR;
    }

    /**
     * Get error severity
     */
    getSeverity(statusCode) {
        if (statusCode >= 500) return 'critical';
        if (statusCode >= 400) return 'error';
        if (statusCode >= 300) return 'warning';
        return 'info';
    }

    /**
     * Send standardized error response
     */
    sendErrorResponse(res, errorInfo, traceId, errorId) {
        const response = {
            error: {
                code: errorInfo.code,
                message: errorInfo.message,
                timestamp: new Date().toISOString(),
                traceId,
                errorId
            }
        };

        // Add additional info for development
        if (process.env.NODE_ENV === 'development') {
            response.error.debug = {
                stack: res.locals.error?.stack,
                details: 'Check trace logs for full context'
            };
        }

        res.status(errorInfo.status).json(response);
    }

    /**
     * Handle CDS service errors
     */
    handleCDSError(error, req) {
        const traceId = req.traceId;
        const log = cds.log('cds-error');

        let errorInfo;

        if (error.code === 'UNIQUE_CONSTRAINT') {
            errorInfo = this.errorCodes.DB_CONSTRAINT_VIOLATION;
        } else if (error.code === 'NOT_FOUND') {
            errorInfo = { code: 'CDS_001', status: 404, message: 'Resource not found' };
        } else if (error.code === 'FORBIDDEN') {
            errorInfo = this.errorCodes.AUTH_INSUFFICIENT_PERMISSIONS;
        } else {
            errorInfo = this.errorCodes.SYS_INTERNAL_ERROR;
        }

        const errorId = addError(traceId, error, {
            component: 'cds-service',
            layer: 'service',
            operation: 'cds-operation',
            severity: this.getSeverity(errorInfo.status)
        });

        log.error('CDS service error', {
            traceId,
            errorId,
            errorCode: errorInfo.code,
            cdsError: error.code,
            message: error.message
        });

        return { errorInfo, errorId };
    }

    /**
     * Handle database errors
     */
    handleDatabaseError(error, req, operation = 'database-operation') {
        const traceId = req.traceId;
        const log = cds.log('database-error');

        let errorInfo;

        if (error.code === 'ECONNREFUSED') {
            errorInfo = this.errorCodes.DB_CONNECTION_ERROR;
        } else if (error.name === 'SequelizeConnectionError') {
            errorInfo = this.errorCodes.DB_CONNECTION_ERROR;
        } else if (error.name === 'SequelizeValidationError') {
            errorInfo = this.errorCodes.VALIDATION_INVALID_INPUT;
        } else {
            errorInfo = this.errorCodes.DB_QUERY_ERROR;
        }

        const errorId = addError(traceId, error, {
            component: 'database',
            layer: 'database',
            operation,
            severity: this.getSeverity(errorInfo.status)
        });

        log.error('Database error', {
            traceId,
            errorId,
            errorCode: errorInfo.code,
            operation,
            dbError: error.name,
            message: error.message
        });

        return { errorInfo, errorId };
    }

    /**
     * Handle blockchain service errors
     */
    handleBlockchainError(error, req, operation = 'blockchain-operation') {
        const traceId = req.traceId;
        const log = cds.log('blockchain-error');

        const errorInfo = this.errorCodes.EXT_BLOCKCHAIN_ERROR;

        const errorId = addError(traceId, error, {
            component: 'blockchain-service',
            layer: 'external',
            operation,
            severity: 'error'
        });

        log.error('Blockchain service error', {
            traceId,
            errorId,
            errorCode: errorInfo.code,
            operation,
            message: error.message
        });

        return { errorInfo, errorId };
    }

    /**
     * Create a traced error
     */
    createTracedError(message, code, statusCode, req, context = {}) {
        const error = new Error(message);
        error.code = code;
        error.statusCode = statusCode;

        if (req && req.traceId) {
            const errorId = addError(req.traceId, error, {
                component: context.component || 'application',
                layer: context.layer || 'business',
                operation: context.operation || 'custom',
                severity: this.getSeverity(statusCode),
                data: context.data || {}
            });
            error.errorId = errorId;
        }

        return error;
    }
}

// Singleton instance
const errorHandler = new ErrorHandler();

module.exports = {
    ErrorHandler,
    errorHandler,

    // Middleware function
    handleError: (error, req, res, next) => errorHandler.handleError(error, req, res, next),
    handleCDSError: (error, req) => errorHandler.handleCDSError(error, req),
    handleDatabaseError: (error, req, operation) => errorHandler.handleDatabaseError(error, req, operation),
    handleBlockchainError: (error, req, operation) => errorHandler.handleBlockchainError(error, req, operation),
    createTracedError: (message, code, statusCode, req, context) =>
        errorHandler.createTracedError(message, code, statusCode, req, context)
};
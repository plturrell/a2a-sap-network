/**
 * @fileoverview Global Error Handler Middleware
 * @since 1.0.0
 * @module error-handler
 *
 * Provides standardized error handling across the A2A Network application
 * with proper logging, monitoring, and user-friendly error responses
 */

const cds = require('@sap/cds');

/**
 * Standard error response structure
 */
class StandardError extends Error {
  constructor(message, code = 'INTERNAL_ERROR', statusCode = 500, details = {}) {
    super(message);
    this.name = 'StandardError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    this.timestamp = new Date().toISOString();
  }
}

/**
 * Error classifications
 */
const ERROR_TYPES = {
  VALIDATION: {
    code: 'VALIDATION_ERROR',
    statusCode: 400,
    category: 'client'
  },
  AUTHENTICATION: {
    code: 'AUTHENTICATION_ERROR',
    statusCode: 401,
    category: 'client'
  },
  AUTHORIZATION: {
    code: 'AUTHORIZATION_ERROR',
    statusCode: 403,
    category: 'client'
  },
  NOT_FOUND: {
    code: 'NOT_FOUND',
    statusCode: 404,
    category: 'client'
  },
  CONFLICT: {
    code: 'CONFLICT_ERROR',
    statusCode: 409,
    category: 'client'
  },
  RATE_LIMIT: {
    code: 'RATE_LIMIT_EXCEEDED',
    statusCode: 429,
    category: 'client'
  },
  INTERNAL: {
    code: 'INTERNAL_ERROR',
    statusCode: 500,
    category: 'server'
  },
  SERVICE_UNAVAILABLE: {
    code: 'SERVICE_UNAVAILABLE',
    statusCode: 503,
    category: 'server'
  },
  DATABASE: {
    code: 'DATABASE_ERROR',
    statusCode: 500,
    category: 'server'
  },
  NETWORK: {
    code: 'NETWORK_ERROR',
    statusCode: 502,
    category: 'server'
  }
};

/**
 * Create a standardized error
 * @param {string} type - Error type from ERROR_TYPES
 * @param {string} message - Error message
 * @param {Object} details - Additional error details
 * @returns {StandardError} Standardized error
 */
function createError(type, message, details = {}) {
  const errorConfig = ERROR_TYPES[type] || ERROR_TYPES.INTERNAL;
  return new StandardError(message, errorConfig.code, errorConfig.statusCode, details);
}

/**
 * Express error handler middleware
 */
function errorHandler(error, req, res, next) {
  const log = cds.log('error-handler');

  // Convert to StandardError if not already
  let standardError;
  if (error instanceof StandardError) {
    standardError = error;
  } else {
    // Classify common error types
    if (error.name === 'ValidationError') {
      standardError = createError('VALIDATION', error.message, { validation: error.details });
    } else if (error.code === 'SQL_INJECTION_DETECTED') {
      standardError = createError('VALIDATION', 'Invalid input detected', { security: true });
    } else if (error.code === 'SQLITE_ERROR' || error.code === 'ENOTFOUND') {
      standardError = createError('DATABASE', 'Database operation failed');
    } else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      standardError = createError('NETWORK', 'Network connection failed');
    } else {
      standardError = createError('INTERNAL', error.message || 'An unexpected error occurred');
    }
  }

  // Log error appropriately
  const statusCode = standardError.statusCode || 500;
  if (statusCode >= 500) {
    log.error('Server error:', standardError.message, { error: standardError });
  } else {
    log.warn('Client error:', standardError.message, { error: standardError });
  }

  // Send response
  const clientError = {
    error: true,
    code: standardError.code,
    message: standardError.message,
    timestamp: standardError.timestamp
  };

  res.status(statusCode).json(clientError);
}

/**
 * Apply error handling middleware to Express app
 * @param {Object} app - Express application
 */
function applyErrorHandling(app) {
  // Global error handler (must be last)
  app.use(errorHandler);

  cds.log('error-handler').info('Global error handling middleware applied');
}

module.exports = {
  StandardError,
  ERROR_TYPES,
  createError,
  errorHandler,
  applyErrorHandling
};
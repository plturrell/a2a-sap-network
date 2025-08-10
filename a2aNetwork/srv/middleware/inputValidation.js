/**
 * @fileoverview Input Validation Middleware
 * @since 1.0.0
 * @module inputValidation
 * 
 * Comprehensive input validation middleware using Joi for schema validation
 * Provides validation schemas for all A2A Network API endpoints
 */

const Joi = require('joi');
const cds = require('@sap/cds');

// Common validation schemas
const commonSchemas = {
    id: Joi.string().uuid().required(),
    optionalId: Joi.string().uuid(),
    agentAddress: Joi.string().pattern(/^0x[a-fA-F0-9]{40}$/).required(),
    ethAddress: Joi.string().pattern(/^0x[a-fA-F0-9]{40}$/),
    name: Joi.string().min(3).max(100).trim(),
    description: Joi.string().max(1000).trim(),
    email: Joi.string().email(),
    url: Joi.string().uri(),
    positiveNumber: Joi.number().positive(),
    nonNegativeNumber: Joi.number().min(0),
    percentage: Joi.number().min(0).max(100),
    timestamp: Joi.date().iso(),
    status: Joi.string().valid('active', 'inactive', 'pending', 'completed', 'failed'),
    severity: Joi.string().valid('low', 'medium', 'high', 'critical'),
    pagination: {
        limit: Joi.number().integer().min(1).max(1000).default(100),
        offset: Joi.number().integer().min(0).default(0),
        sort: Joi.string().valid('asc', 'desc').default('desc')
    }
};

// API endpoint validation schemas
const validationSchemas = {
    // Network settings validation
    'PUT:/api/v1/settings/network': Joi.object({
        network: Joi.string().required().valid('localhost', 'testnet', 'mainnet'),
        rpcUrl: commonSchemas.url.required(),
        chainId: Joi.number().integer().positive().required(),
        contractAddress: commonSchemas.ethAddress.required()
    }),

    // Security settings validation
    'PUT:/api/v1/settings/security': Joi.object({
        encryptionEnabled: Joi.boolean().required(),
        authRequired: Joi.boolean().required(),
        twoFactorEnabled: Joi.boolean(),
        sessionTimeout: Joi.number().integer().min(300).max(86400), // 5 minutes to 24 hours
        maxLoginAttempts: Joi.number().integer().min(1).max(20)
    }),

    // Auto-save settings validation
    'POST:/api/v1/settings/autosave': Joi.object({
        settings: Joi.object().required(),
        timestamp: commonSchemas.timestamp
    }),

    // Connection test validation
    'POST:/api/v1/test-connection': Joi.object({
        rpcUrl: commonSchemas.url.required(),
        networkId: Joi.number().integer().positive()
    }),

    // System reconfiguration validation
    'POST:/api/v1/reconfigure': Joi.object({
        network: Joi.object({
            rpcUrl: commonSchemas.url,
            chainId: Joi.number().integer().positive(),
            gasLimit: Joi.number().integer().positive(),
            gasPrice: Joi.string().pattern(/^\d+$/)
        }),
        performance: Joi.object({
            cacheEnabled: Joi.boolean(),
            cacheTTL: Joi.number().integer().positive(),
            maxConnections: Joi.number().integer().positive(),
            timeout: Joi.number().integer().positive()
        }),
        security: Joi.object({
            rateLimitEnabled: Joi.boolean(),
            maxRequestsPerHour: Joi.number().integer().positive(),
            encryptionEnabled: Joi.boolean()
        })
    }).min(1), // At least one configuration section required

    // Operations restart validation
    'POST:/api/v1/operations/restart': Joi.object({
        service: Joi.string().valid('cache', 'monitoring', 'all').default('all')
    }),

    // Error reporting validation
    'POST:/api/v1/errors/report': Joi.object({
        message: Joi.string().required().max(1000),
        stack: Joi.string().max(5000),
        timestamp: commonSchemas.timestamp,
        url: Joi.string().max(500),
        userAgent: Joi.string().max(500),
        severity: commonSchemas.severity.default('error'),
        category: Joi.string().valid('javascript', 'network', 'validation', 'security', 'performance'),
        component: Joi.string().max(100),
        correlationId: Joi.string().uuid(),
        metadata: Joi.object()
    }),

    // Log query validation
    'GET:/api/v1/logs': Joi.object({
        limit: commonSchemas.pagination.limit,
        level: Joi.string().valid('error', 'warn', 'info', 'debug'),
        since: commonSchemas.timestamp,
        component: Joi.string().max(100),
        correlationId: Joi.string().uuid()
    }),

    // Error query validation
    'GET:/api/v1/errors': Joi.object({
        limit: commonSchemas.pagination.limit,
        severity: commonSchemas.severity,
        category: Joi.string().max(100),
        component: Joi.string().max(100),
        correlationId: Joi.string().uuid(),
        since: commonSchemas.timestamp
    }),

    // Cache invalidation validation
    'POST:/cache/invalidate': Joi.object({
        pattern: Joi.string().required().min(1).max(200)
    }),

    // Agent validation schemas
    agent: {
        create: Joi.object({
            name: commonSchemas.name.required(),
            address: commonSchemas.agentAddress,
            endpoint: commonSchemas.url,
            description: commonSchemas.description,
            capabilities: Joi.array().items(Joi.string().max(100)).max(20),
            metadata: Joi.object()
        }),
        update: Joi.object({
            name: commonSchemas.name,
            endpoint: commonSchemas.url,
            description: commonSchemas.description,
            isActive: Joi.boolean(),
            metadata: Joi.object()
        }).min(1)
    },

    // Service validation schemas
    service: {
        create: Joi.object({
            name: commonSchemas.name.required(),
            description: commonSchemas.description.required(),
            provider_ID: commonSchemas.id.required(),
            pricePerCall: commonSchemas.nonNegativeNumber.required(),
            category: Joi.string().required().max(50),
            endpoint: commonSchemas.url,
            minReputation: commonSchemas.nonNegativeNumber.default(0),
            maxCallsPerDay: Joi.number().integer().positive().max(1000000),
            metadata: Joi.object()
        }),
        update: Joi.object({
            name: commonSchemas.name,
            description: commonSchemas.description,
            pricePerCall: commonSchemas.nonNegativeNumber,
            endpoint: commonSchemas.url,
            minReputation: commonSchemas.nonNegativeNumber,
            maxCallsPerDay: Joi.number().integer().positive(),
            isActive: Joi.boolean(),
            metadata: Joi.object()
        }).min(1)
    },

    // Workflow validation schemas
    workflow: {
        create: Joi.object({
            name: commonSchemas.name.required(),
            description: commonSchemas.description.required(),
            definition: Joi.object().required(),
            category: Joi.string().required().max(50),
            version: Joi.string().pattern(/^\d+\.\d+\.\d+$/).default('1.0.0'),
            isPublic: Joi.boolean().default(false),
            metadata: Joi.object()
        }),
        execute: Joi.object({
            parameters: Joi.object().default({}),
            priority: Joi.number().integer().min(1).max(10).default(5),
            timeout: Joi.number().integer().positive().max(3600000), // Max 1 hour
            metadata: Joi.object()
        })
    }
};

/**
 * Validation middleware factory
 * @param {string} schemaKey - Key to identify validation schema
 * @returns {Function} Express middleware function
 */
function validateInput(schemaKey) {
    return (req, res, next) => {
        const log = cds.log('input-validation');
        
        // Get schema based on method and path or direct key
        const routeKey = `${req.method}:${req.path}`;
        let schema = validationSchemas[routeKey] || validationSchemas[schemaKey];
        
        if (!schema) {
            log.debug('No validation schema found for route', { route: routeKey, schemaKey });
            return next(); // Skip validation if no schema defined
        }

        // Validate request body for POST/PUT, query params for GET
        const dataToValidate = ['POST', 'PUT', 'PATCH'].includes(req.method) ? req.body : req.query;
        
        const { error, value } = schema.validate(dataToValidate, {
            abortEarly: false, // Return all validation errors
            stripUnknown: true, // Remove unknown fields
            convert: true // Convert types (string to number, etc.)
        });

        if (error) {
            log.warn('Input validation failed', {
                route: routeKey,
                errors: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message,
                    value: detail.context?.value
                }))
            });

            return res.status(400).json({
                error: 'Invalid input data',
                code: 'VALIDATION_ERROR',
                details: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message.replace(/"/g, "'"),
                    value: detail.context?.value
                }))
            });
        }

        // Replace original data with validated and converted data
        if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
            req.body = value;
        } else {
            req.query = value;
        }

        log.debug('Input validation passed', { route: routeKey });
        next();
    };
}

/**
 * Custom validation for CDS entities
 * @param {string} entityType - Type of entity (agent, service, workflow)
 * @param {string} operation - Operation type (create, update)
 * @returns {Function} CDS middleware function
 */
function validateEntity(entityType, operation) {
    return (req) => {
        const log = cds.log('entity-validation');
        const schemaPath = `${entityType}.${operation}`;
        const schema = validationSchemas[schemaPath];
        
        if (!schema) {
            log.debug('No validation schema found for entity', { entityType, operation });
            return;
        }

        const { error, value } = schema.validate(req.data, {
            abortEarly: false,
            stripUnknown: true,
            convert: true
        });

        if (error) {
            log.warn('Entity validation failed', {
                entityType,
                operation,
                errors: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message
                }))
            });

            req.error(400, 'VALIDATION_ERROR', {
                message: 'Invalid entity data',
                details: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message.replace(/"/g, "'")
                }))
            });
        }

        // Update request data with validated values
        req.data = value;
    };
}

module.exports = {
    validateInput,
    validateEntity,
    commonSchemas,
    validationSchemas
};
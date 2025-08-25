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
    },
    // Enhanced edge case validations
    safeString: Joi.string().pattern(/^[a-zA-Z0-9\s\-_.,!?()]+$/).max(1000),
    alphanumeric: Joi.string().pattern(/^[a-zA-Z0-9]+$/),
    safePath: Joi.string().pattern(/^[a-zA-Z0-9\-_\/\.]+$/).max(500),
    ipAddress: Joi.string().ip(),
    port: Joi.number().integer().min(1).max(65535),
    semver: Joi.string().pattern(/^\d+\.\d+\.\d+(-[a-zA-Z0-9\-]+)*$/),
    base64: Joi.string().base64(),
    json: Joi.string().custom((value, helpers) => {
        try {
            JSON.parse(value);
            return value;
        } catch (error) {
            return helpers.error('any.invalid');
        }
    }, 'JSON validation')
};

// API endpoint validation schemas
const validationSchemas = {
    // Tile Data Endpoints - Permissive validation for dashboard queries
    'GET:/api/v1/NetworkStats': Joi.object({
        id: Joi.string().valid(
            'overview_dashboard', 'dashboard_test', 'agent_visualization',
            'service_marketplace', 'blockchain_dashboard', 'notification_center',
            'network_analytics', 'network_health'
        ).required()
    }),

    'GET:/api/v1/Agents': Joi.object({
        id: Joi.string().valid(
            'agent_visualization', 'dashboard_test', 'agent_marketplace'
        ).required()
    }),

    'GET:/api/v1/Services': Joi.object({
        id: Joi.string().valid(
            'service_marketplace', 'dashboard_test'
        ).required()
    }),

    'GET:/api/v1/Notifications': Joi.object({
        id: Joi.string().valid(
            'notification_center', 'dashboard_test'
        ).optional()
    }),

    'GET:/api/v1/blockchain/stats': Joi.object({
        id: Joi.string().valid(
            'blockchain_dashboard', 'dashboard_test'
        ).required()
    }),

    'GET:/odata/v4/blockchain/BlockchainStats': Joi.object({
        id: Joi.string().valid(
            'blockchain_dashboard', 'dashboard_test'
        ).required()
    }),

    'GET:/api/v1/notifications/count': Joi.object({
        id: Joi.string().valid(
            'notification_center', 'dashboard_test'
        ).optional()
    }),

    'GET:/api/v1/network/analytics': Joi.object({
        id: Joi.string().valid(
            'network_analytics', 'dashboard_test'
        ).optional()
    }),

    'GET:/api/v1/network/health': Joi.object({
        id: Joi.string().valid(
            'network_health', 'dashboard_test'
        ).optional()
    }),

    'GET:/api/v1/metrics/current': Joi.object({}).allow({}),
    'GET:/api/v1/metrics/performance': Joi.object({}).allow({}),
    'GET:/api/v1/operations/status': Joi.object({}).allow({}),
    'GET:/api/v1/operations/logs': Joi.object({
        limit: Joi.number().integer().min(1).max(1000).default(100),
        level: Joi.string().valid('error', 'warn', 'info', 'debug').optional(),
        since: Joi.date().iso().optional()
    }),
    'GET:/api/v1/operations/logs/download': Joi.object({
        format: Joi.string().valid('txt', 'json').default('txt'),
        since: Joi.date().iso().optional(),
        until: Joi.date().iso().optional()
    }),
    'GET:/api/v1/monitoring/status': Joi.object({}).allow({}),
    'GET:/api/v1/blockchain/status': Joi.object({}).allow({}),
    'GET:/api/v1/debug/agents': Joi.object({}).allow({}),

    // Network settings validation
    'PUT:/api/v1/settings/network': Joi.object({
        network: Joi.string().required().valid(
            ...(process.env.NODE_ENV === 'development' ? ['localhost'] : []),
            'testnet', 'mainnet'
        ),
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

    // Agent validation schemas with enhanced edge case handling
    agent: {
        create: Joi.object({
            name: commonSchemas.name.required(),
            address: commonSchemas.agentAddress,
            endpoint: commonSchemas.url,
            description: commonSchemas.description,
            capabilities: Joi.array().items(commonSchemas.safeString.max(100)).max(20).unique(),
            metadata: Joi.object().max(10), // Limit metadata size
            version: commonSchemas.semver.optional(),
            tags: Joi.array().items(commonSchemas.alphanumeric.max(50)).max(10).unique().optional()
        }),
        update: Joi.object({
            name: commonSchemas.name,
            endpoint: commonSchemas.url,
            description: commonSchemas.description,
            isActive: Joi.boolean(),
            metadata: Joi.object().max(10),
            version: commonSchemas.semver.optional(),
            tags: Joi.array().items(commonSchemas.alphanumeric.max(50)).max(10).unique().optional()
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

        // Log request details for debugging (development only)
        if (req.path.includes('launchpad') || req.originalUrl.includes('launchpad')) {
            log.debug('Launchpad request validation', {
                method: req.method,
                path: req.path,
                originalUrl: req.originalUrl,
                schemaKey: schemaKey
            });
        }

        // Skip validation for tile API endpoints that need to return real backend data
        const tileApiEndpoints = [
            '/api/v1/NetworkStats',
            '/api/v1/Agents',
            '/api/v1/Services',  // Added for diagnostic test page
            '/api/v1/Notifications',  // Added for diagnostic test page
            '/api/v1/network/Agents',  // Added for launchpad
            '/api/v1/debug/agents',
            '/odata/v4/blockchain/BlockchainStats',
            '/api/v1/blockchain/stats',  // Added for blockchain dashboard tile
            '/api/v1/network/analytics',
            '/api/v1/notifications/count',
            '/api/v1/network/health'
        ];

        // Skip validation for static file paths (HTML, CSS, JS, images, etc.)
        const staticFilePaths = [
            '/app/',
            '/static/',
            '/assets/',
            '/resources/',
            '/shells/',
            '/common/'
        ];

        // Check if the request path matches any tile API endpoint (handles query parameters)
        const matchedTileEndpoint = tileApiEndpoints.find(endpoint => req.path.startsWith(endpoint));
        if (matchedTileEndpoint) {
            log.debug('Skipping validation for tile API endpoint', {
                path: req.path,
                matchedEndpoint: matchedTileEndpoint
            });
            return next();
        }

        // Skip validation for static file requests
        const matchedStaticPath = staticFilePaths.find(staticPath => req.path.startsWith(staticPath));
        if (matchedStaticPath) {
            log.debug('Skipping validation for static file request', {
                path: req.path,
                matchedPath: matchedStaticPath
            });
            return next();
        }

        // Debug validation proceeding for launchpad
        if (req.path.includes('launchpad') || req.originalUrl.includes('launchpad')) {
            log.debug('Validation proceeding for launchpad', {
                path: req.path,
                pathStartsWithApp: req.path.startsWith('/app/')
            });
        }

        // Get schema based on method and path or direct key
        const routeKey = `${req.method}:${req.path}`;
        const schema = validationSchemas[routeKey] || validationSchemas[schemaKey];

        // Debug logging for diagnostic endpoints
        if (req.path.includes('/api/v1/Services') || req.path.includes('/api/v1/Notifications')) {
            log.debug('Diagnostic endpoint validation', {
                path: req.path,
                routeKey: routeKey,
                schemaKey: schemaKey,
                hasSchema: !!schema
            });
        }

        if (!schema) {
            log.debug('No validation schema found for route', { route: routeKey, schemaKey });
            return next(); // Skip validation if no schema defined
        }

        // Validate request body for POST/PUT, query params for GET
        const dataToValidate = ['POST', 'PUT', 'PATCH'].includes(req.method) ? req.body : req.query;

        // First, check for edge cases and security threats
        const edgeValidation = validateEdgeCases(dataToValidate);
        if (!edgeValidation.isValid) {
            log.warn('Edge case validation failed', {
                route: routeKey,
                errors: edgeValidation.errors
            });

            return res.status(400).json({
                error: 'Invalid input detected',
                code: 'SECURITY_VALIDATION_ERROR',
                details: edgeValidation.errors.map(error => ({
                    message: error,
                    type: 'security'
                }))
            });
        }

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
                    message: detail.message.replace(/"/g, '\''),
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
 * Enhanced validation for edge cases and security threats
 * @param {*} data - Data to validate
 * @returns {Object} Validation result
 */
function validateEdgeCases(data) {
    const errors = [];

    function checkValue(value, path = '') {
        if (typeof value === 'string') {
            // Check for extremely long strings (potential DoS)
            if (value.length > 100000) {
                errors.push(`String too long at ${path}: ${value.length} characters`);
            }

            // Check for null bytes (potential injection)
            if (value.includes('\0')) {
                errors.push(`Null byte detected at ${path}`);
            }

            // Check for binary data masquerading as text
            if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(value)) {
                errors.push(`Binary/control characters detected at ${path}`);
            }

            // Check for potential path traversal
            if (value.includes('../') || value.includes('..\\')) {
                errors.push(`Path traversal attempt detected at ${path}`);
            }

            // Check for script tags or potential XSS
            if (/<script|javascript:|data:|vbscript:/i.test(value)) {
                errors.push(`Potential XSS detected at ${path}`);
            }
        } else if (typeof value === 'object' && value !== null) {
            // Check for prototype pollution attempts
            if (value.hasOwnProperty('__proto__') || value.hasOwnProperty('constructor') || value.hasOwnProperty('prototype')) {
                errors.push(`Prototype pollution attempt detected at ${path}`);
            }

            // Check for circular references (potential DoS)
            try {
                JSON.stringify(value);
            } catch (error) {
                if (error.message.includes('circular')) {
                    errors.push(`Circular reference detected at ${path}`);
                }
            }

            // Recursively check nested objects/arrays
            Object.keys(value).forEach(key => {
                checkValue(value[key], path ? `${path}.${key}` : key);
            });
        } else if (typeof value === 'number') {
            // Check for extreme numbers that might cause issues
            if (!Number.isFinite(value) || Math.abs(value) > Number.MAX_SAFE_INTEGER) {
                errors.push(`Unsafe number detected at ${path}: ${value}`);
            }
        }
    }

    checkValue(data);

    return {
        isValid: errors.length === 0,
        errors: errors
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
                    message: detail.message.replace(/"/g, '\'')
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
    validateEdgeCases,
    commonSchemas,
    validationSchemas
};
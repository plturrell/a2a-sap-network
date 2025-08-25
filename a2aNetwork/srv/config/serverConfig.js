/**
 * @fileoverview Server Configuration Module
 * @since 1.0.0
 * @module serverConfig
 *
 * Centralized server configuration and initialization utilities
 */

const cds = require('@sap/cds');
const path = require('path');

/**
 * Initialize and configure middleware stack
 * @param {object} app - Express application instance
 */
async function initializeMiddleware(app) {
    const log = cds.log('server-config');

    // Import middleware modules
    const { applySecurityMiddleware } = require('../middleware/security');
    const { applyAuthMiddleware, initializeXSUAAStrategy } = require('../middleware/auth');
    const { initializeEnvironmentValidation } = require('../middleware/envValidation');
    const { validateSQLMiddleware } = require('../middleware/sqlSecurity');
    const enterpriseLogger = require('../middleware/sapEnterpriseLogging');
    const cacheMiddleware = require('../middleware/sapCacheMiddleware');
    const securityHardening = require('../middleware/sapSecurityHardening');
    const monitoringIntegration = require('../middleware/sapMonitoringIntegration');
    const cors = require('cors');

    // Validate environment variables first
    try {
        initializeEnvironmentValidation();
        log.info('Environment validation completed successfully');
    } catch (error) {
        log.error('Environment validation failed:', error);
        throw error;
    }

    // Initialize enterprise components
    await cacheMiddleware.initialize();
    await monitoringIntegration.initialize();

    // Initialize i18n
    const { initializeI18n } = require('../i18n/sapI18nConfig');
    initializeI18n(app);

    // Initialize XSUAA authentication strategy
    initializeXSUAAStrategy();

    // Apply CORS configuration
    const corsOptions = {
        origin: function (origin, callback) {
            const allowedOrigins = process.env.ALLOWED_ORIGINS
                ? process.env.ALLOWED_ORIGINS.split(',')
                : ['http://localhost:3000', 'http://localhost:8080'];

            // SECURITY: Only allow requests with no origin in development mode
            if (!origin) {
                if (process.env.NODE_ENV === 'development') {
                    return callback(null, true);
                } else {
                    return callback(new Error('Origin header required in production'));
                }
            }

            if (allowedOrigins.indexOf(origin) !== -1) {
                callback(null, true);
            } else {
                callback(new Error('Not allowed by CORS'));
            }
        },
        credentials: true,
        optionsSuccessStatus: 200
    };
    app.use(cors(corsOptions));

    // Apply security hardening (must be early)
    app.use(securityHardening.securityHeaders());
    app.use(securityHardening.generateNonce());

    // Apply rate limiting
    const rateLimits = securityHardening.rateLimiting();
    app.use('/auth', rateLimits.auth);

    // Apply enterprise logging middleware
    app.use(enterpriseLogger.requestMiddleware());

    // Apply caching middleware
    app.use(cacheMiddleware.middleware());

    // Apply monitoring middleware
    app.use(monitoringIntegration.metricsMiddleware());

    // Apply SQL security middleware
    app.use(validateSQLMiddleware);

    log.info('Middleware stack initialized successfully');
}

/**
 * Configure static file serving
 * @param {object} app - Express application instance
 */
function configureStaticFiles(app) {
    const express = require('express');

    // Configure static file serving with proper MIME types
    const staticOptions = {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            } else if (filePath.endsWith('.html')) {
                res.setHeader('Content-Type', 'text/html');
            }
        }
    };

    // Serve common JavaScript components
    app.use('/common', express.static(path.join(__dirname, '../../common'), staticOptions));

    // Serve A2A Agents static files
    app.use('/a2aAgents', express.static(path.join(__dirname, '../../../a2aAgents'), staticOptions));

    // Serve A2A Fiori webapp
    app.use('/a2aFiori', express.static(path.join(__dirname, '../../a2aFiori'), staticOptions));

    // Serve A2A Fiori app at standard SAP UShell path
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../../app/a2aFiori'), staticOptions));

    // Serve launchpad static files
    app.use('/app', express.static(path.join(__dirname, '../../app'), staticOptions));

    // Serve shells directory for Fiori Sandbox configuration
    app.use('/shells', express.static(path.join(__dirname, '../../app/shells')));

    cds.log('server-config').info('Static file serving configured successfully');
}

/**
 * Setup health check endpoints
 * @param {object} app - Express application instance
 */
function setupHealthChecks(app) {
    const healthService = require('../services/sapHealthService');

    // Basic health check
    app.get('/health', async (req, res) => {
        try {
            const health = await healthService.getHealth();
            res.status(200).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

    // Detailed health check
    app.get('/health/detailed', async (req, res) => {
        try {
            const health = await healthService.getDetailedHealth();
            res.status(health.status === 'healthy' ? 200 : 503).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

    // Readiness probe
    app.get('/health/ready', async (req, res) => {
        try {
            const readiness = await healthService.getReadiness();
            res.status(readiness.status === 'ready' ? 200 : 503).json(readiness);
        } catch (error) {
            res.status(503).json({ status: 'not-ready', error: error.message });
        }
    });

    // Liveness probe
    app.get('/health/live', (req, res) => {
        const liveness = healthService.getLiveness();
        res.status(liveness.status === 'alive' ? 200 : 503).json(liveness);
    });

    // Metrics endpoint
    app.get('/metrics', (req, res) => {
        const metrics = healthService.getMetrics();
        res.status(200).json(metrics);
    });

    cds.log('server-config').info('Health check endpoints configured successfully');
}

/**
 * Setup admin and monitoring routes
 * @param {object} app - Express application instance
 */
function setupAdminRoutes(app) {
    const cacheMiddleware = require('../middleware/sapCacheMiddleware');
    const monitoringIntegration = require('../middleware/sapMonitoringIntegration');

    // Admin role check helper
    const requireAdmin = (req, res, next) => {
        if (!req.user ||
            (!req.user.scope?.includes('Admin') &&
             !req.user.roles?.includes('Admin') &&
             !req.user.sapRoles?.includes('Admin'))) {
            return res.status(403).json({ error: 'Admin role required' });
        }
        next();
    };

    // Cache management endpoints
    app.get('/cache/stats', requireAdmin, async (req, res) => {
        const stats = await cacheMiddleware.getStats();
        res.json(stats);
    });

    app.post('/cache/invalidate', requireAdmin, async (req, res) => {
        const { pattern } = req.body;
        if (!pattern) {
            return res.status(400).json({ error: 'Pattern required' });
        }
        await cacheMiddleware.invalidate(pattern);
        res.json({ success: true });
    });

    // Monitoring routes
    monitoringIntegration.setupRoutes(app);

    // Audit logs endpoint
    app.get('/logs/audit', requireAdmin, (req, res) => {
        res.json({ message: 'Audit logs endpoint - implement based on your storage' });
    });

    // Security events endpoint
    app.get('/security/events', requireAdmin, (req, res) => {
        res.json({ message: 'Security events endpoint - implement based on your requirements' });
    });

    cds.log('server-config').info('Admin routes configured successfully');
}

/**
 * Setup API routes
 * @param {object} app - Express application instance
 */
function setupAPIRoutes(app) {
    // Add API routes
    const apiRoutes = require('../apiRoutes');
    app.use(apiRoutes);

    // Apply security middleware
    const { applySecurityMiddleware } = require('../middleware/security');
    applySecurityMiddleware(app);

    // Apply authentication middleware
    const { applyAuthMiddleware } = require('../middleware/auth');
    applyAuthMiddleware(app);

    // Enhanced error handling middleware
    const { expressErrorMiddleware } = require('../utils/errorHandler');
    app.use(expressErrorMiddleware);

    cds.log('server-config').info('API routes configured successfully');
}

/**
 * Initialize SAP Fiori flexibility services
 * @param {object} app - Express application instance
 */
function setupSAPFioriServices(app) {
    // SAP Fiori flexibility services stub endpoints
    app.get('/sap/bc/lrep/flex/settings', (req, res) => {
        res.status(200).json({
            'isKeyUser': false,
            'isAtoAvailable': false,
            'isAtoEnabled': false,
            'isProductiveSystem': true,
            'isZeroDowntimeUpgradeRunning': false,
            'system': '',
            'client': ''
        });
    });

    app.get('/sap/bc/lrep/flex/data/:appId', (req, res) => {
        res.status(200).json({
            changes: [],
            ui2personalization: {},
            variants: []
        });
    });

    // Launchpad pages
    app.get('/launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../../app/launchpad.html'));
    });

    app.get('/fiori-launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../../app/fioriLaunchpad.html'));
    });

    cds.log('server-config').info('SAP Fiori services configured successfully');
}

module.exports = {
    initializeMiddleware,
    configureStaticFiles,
    setupHealthChecks,
    setupAdminRoutes,
    setupAPIRoutes,
    setupSAPFioriServices
};
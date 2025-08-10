/**
 * @fileoverview SAP CAP Server Configuration with Enterprise Middleware
 * @since 1.0.0
 * @module server
 * 
 * Main server configuration for the A2A Network service implementing
 * SAP enterprise patterns including security, monitoring, caching,
 * health checks, and distributed tracing
 */

const cds = require('@sap/cds');
const { applySecurityMiddleware } = require('./middleware/security');
const { applyAuthMiddleware, initializeXSUAAStrategy } = require('./middleware/auth');
const monitoring = require('./lib/monitoring');
const cloudALM = require('./lib/sapCloudALM');
const { tracing } = require('./lib/sapDistributedTracing');
const { initializeI18n } = require('./i18n/sapI18nConfig');
const i18nMiddleware = require('./i18n/sapI18nMiddleware');
const { initializeDatabase } = require('./lib/sapDbInit');
const networkStats = require('./lib/sapNetworkStats');

// Enterprise services
const healthService = require('./services/sapHealthService');
const loggingService = require('./services/sapLoggingService');
const errorReporting = require('./services/sapErrorReportingService');

// New enterprise middleware
const enterpriseLogger = require('./middleware/sapEnterpriseLogging');
const cacheMiddleware = require('./middleware/sapCacheMiddleware');
const securityHardening = require('./middleware/sapSecurityHardening');
const monitoringIntegration = require('./middleware/sapMonitoringIntegration');
const cors = require('cors');
const apiRoutes = require('./apiRoutes');
const { Server } = require('socket.io');
const http = require('http');

// Standard CAP server setup
module.exports = cds.server;

// Global database connection and WebSocket server
let dbConnection = null;
let io = null;
let httpServer = null;

// Initialize i18n middleware
i18nMiddleware.init();

// Helper function to check topic access based on user roles
function hasTopicAccess(user, topic) {
    if (!user || !user.roles) return false;
    
    const roles = user.sapRoles || user.roles || [];
    
    // Admin can access all topics
    if (roles.includes('Admin')) return true;
    
    // Topic-based access control
    const topicPermissions = {
        'agent.events': ['authenticated-user'],
        'service.events': ['authenticated-user'],
        'workflow.events': ['authenticated-user', 'WorkflowManager'],
        'network.events': ['authenticated-user'],
        'admin.events': ['Admin'],
        'monitoring.alerts': ['Admin', 'AgentManager', 'ServiceManager'],
        'system.health': ['Admin'],
        'blockchain.events': ['authenticated-user'],
        'performance.metrics': ['authenticated-user']
    };
    
    const allowedRoles = topicPermissions[topic];
    if (!allowedRoles) {
        // Default: allow all authenticated users for unknown topics
        return roles.includes('authenticated-user');
    }
    
    return allowedRoles.some(role => roles.includes(role));
}

// Apply security middleware and add health endpoint
cds.on('bootstrap', async (app) => {
    // Initialize enterprise components
    await cacheMiddleware.initialize();
    await monitoringIntegration.initialize();
    
    // Initialize i18n
    initializeI18n(app);
    
    // Initialize XSUAA authentication strategy
    initializeXSUAAStrategy();
    
    // Apply CORS with SAP-specific configuration
    app.use(cors());
    
    // Apply security hardening (must be early)
    app.use(securityHardening.securityHeaders());
    app.use(securityHardening.generateNonce());
    
    // Apply rate limiting
    const rateLimits = securityHardening.rateLimiting();
    app.use('/auth', rateLimits.auth);
    app.use('/api/v1/Agents', rateLimits.strict);
    app.use('/api/v1/Services', rateLimits.standard);
    app.use(rateLimits.standard); // Default rate limiting
    app.use(rateLimits.slowDown);
    
    // Apply enterprise logging middleware
    app.use(enterpriseLogger.requestMiddleware());
    
    // Apply caching middleware
    app.use(cacheMiddleware.middleware());
    
    // Apply monitoring middleware
    app.use(monitoringIntegration.metricsMiddleware());
    
    // Apply input validation and sanitization
    app.use(securityHardening.inputValidation());
    
    // Apply response filtering
    app.use(securityHardening.responseFilter());
    
    // Apply comprehensive security middleware
    applySecurityMiddleware(app);
    
    // Apply authentication middleware (always enabled)
    // Use USE_DEVELOPMENT_AUTH=true for development mode
    applyAuthMiddleware(app);
    
    // Apply security monitoring
    app.use(securityHardening.securityMonitoring());
    
    // Apply distributed tracing middleware
    app.use(tracing.instrumentHTTP());
    
    // Apply logging middleware
    app.use(loggingService.middleware());
    
    // Apply health service request tracking
    app.use((req, res, next) => {
        const startTime = Date.now();
        
        res.on('finish', () => {
            const duration = Date.now() - startTime;
            healthService.recordRequest(res.statusCode, duration);
        });
        
        next();
    });
    
    // Apply monitoring middleware
    app.use(monitoring.middleware());
    
    // Apply error reporting middleware
    app.use(errorReporting.middleware());
    
    // Add missing API routes
    app.use(apiRoutes);
    
    // Initialize WebSocket server after all middleware is configured
    httpServer = http.createServer(app);
    io = new Server(httpServer, {
        cors: {
            origin: process.env.WEBSOCKET_CORS_ORIGIN || "*",
            methods: ["GET", "POST"],
            credentials: process.env.NODE_ENV === 'production'
        },
        transports: ['websocket', 'polling'],
        pingTimeout: 60000,
        pingInterval: 25000
    });
    
    // Make io available globally for CDS services
    cds.io = io;
    
    // WebSocket authentication middleware
    io.use(async (socket, next) => {
        try {
            const token = socket.handshake.auth.token || socket.handshake.headers.authorization?.substring(7);
            
            if (!token) {
                if (process.env.USE_DEVELOPMENT_AUTH === 'true' && process.env.NODE_ENV !== 'production') {
                    // Development mode - allow connections without auth
                    socket.user = { id: 'dev-user', roles: ['authenticated-user'], isDevelopment: true };
                    return next();
                } else {
                    return next(new Error('Authentication token required'));
                }
            }
            
            // Use the same JWT validation as HTTP requests
            const { validateJWT } = require('./middleware/auth');
            const mockReq = { headers: { authorization: `Bearer ${token}` } };
            const mockRes = {
                status: () => mockRes,
                json: (data) => { throw new Error(data.error || 'Authentication failed'); }
            };
            
            validateJWT(mockReq, mockRes, (error) => {
                if (error) {
                    return next(error);
                }
                socket.user = mockReq.user;
                next();
            });
        } catch (error) {
            next(new Error('WebSocket authentication failed: ' + error.message));
        }
    });
    
    // WebSocket connection handling
    io.on('connection', (socket) => {
        const log = cds.log('websocket');
        log.info(`WebSocket client connected: ${socket.id}`, {
            userId: socket.user?.id,
            userAgent: socket.handshake.headers['user-agent']
        });
        
        // Join user to their personal room
        if (socket.user?.id) {
            socket.join(`user:${socket.user.id}`);
        }
        
        // Handle subscription management
        socket.on('subscribe', (topics) => {
            if (!Array.isArray(topics)) {
                socket.emit('error', { message: 'Topics must be an array' });
                return;
            }
            
            topics.forEach(topic => {
                // Validate topic access based on user roles
                if (hasTopicAccess(socket.user, topic)) {
                    socket.join(`topic:${topic}`);
                    log.debug(`User ${socket.user.id} subscribed to topic: ${topic}`);
                } else {
                    socket.emit('error', { message: `Access denied to topic: ${topic}` });
                }
            });
            
            socket.emit('subscribed', { topics: topics.filter(t => hasTopicAccess(socket.user, t)) });
        });
        
        socket.on('unsubscribe', (topics) => {
            if (!Array.isArray(topics)) {
                socket.emit('error', { message: 'Topics must be an array' });
                return;
            }
            
            topics.forEach(topic => {
                socket.leave(`topic:${topic}`);
                log.debug(`User ${socket.user.id} unsubscribed from topic: ${topic}`);
            });
            
            socket.emit('unsubscribed', { topics });
        });
        
        // Handle disconnection
        socket.on('disconnect', (reason) => {
            log.info(`WebSocket client disconnected: ${socket.id}`, {
                reason,
                userId: socket.user?.id
            });
        });
        
        // Send initial connection confirmation
        socket.emit('connected', {
            socketId: socket.id,
            userId: socket.user?.id,
            timestamp: new Date().toISOString()
        });
    });
    
    log.info('WebSocket server initialized successfully');
    
    // Health check endpoints
    app.get('/health', async (req, res) => {
        try {
            const health = await healthService.getHealth();
            res.status(200).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

    app.get('/health/detailed', async (req, res) => {
        try {
            const health = await healthService.getDetailedHealth();
            res.status(health.status === 'healthy' ? 200 : 503).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

    app.get('/health/ready', async (req, res) => {
        try {
            const readiness = await healthService.getReadiness();
            res.status(readiness.status === 'ready' ? 200 : 503).json(readiness);
        } catch (error) {
            res.status(503).json({ status: 'not-ready', error: error.message });
        }
    });

    app.get('/health/live', (req, res) => {
        const liveness = healthService.getLiveness();
        res.status(liveness.status === 'alive' ? 200 : 503).json(liveness);
    });

    app.get('/metrics', (req, res) => {
        const metrics = healthService.getMetrics();
        res.status(200).json(metrics);
    });

    // Error reporting endpoints
    app.post('/api/v1/errors/report', (req, res) => {
        try {
            // Validate basic structure or provide defaults
            const errorData = req.body || {};
            const sanitizedError = {
                message: errorData.message || 'Unknown error',
                stack: errorData.stack || '',
                timestamp: errorData.timestamp || new Date().toISOString(),
                url: errorData.url || req.get('referer') || '',
                userAgent: errorData.userAgent || req.get('user-agent') || '',
                severity: errorData.severity || 'error'
            };
            
            const errorId = errorReporting.reportClientError(sanitizedError);
            res.status(201).json({ errorId });
        } catch (error) {
            // Log the error but don't fail the request
            cds.log('server').warn('Error reporting failed:', error.message);
            res.status(200).json({ errorId: 'client-error-ignored' });
        }
    });

    app.get('/api/v1/errors', (req, res) => {
        const limit = parseInt(req.query.limit) || 50;
        const filters = {
            severity: req.query.severity,
            category: req.query.category,
            component: req.query.component,
            correlationId: req.query.correlationId,
            since: req.query.since
        };
        
        const errors = errorReporting.getRecentErrors(limit, filters);
        res.status(200).json(errors);
    });

    app.get('/api/v1/errors/stats', (req, res) => {
        const timeframe = req.query.timeframe || '24h';
        const stats = errorReporting.getErrorStats(timeframe);
        res.status(200).json(stats);
    });

    app.get('/api/v1/logs', (req, res) => {
        const limit = parseInt(req.query.limit) || 100;
        const level = req.query.level;
        const logs = loggingService.getRecentLogs(limit, level);
        res.status(200).json(logs);
    });
    
    // User API endpoints for BTP integration
    app.use('/user-api', require('./sapUserService'));
    
    // Serve UI5 app
    const path = require('path');
    const express = require('express');
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2aFiori/webapp')));
    app.use('/app/launchpad', express.static(path.join(__dirname, '../app/launchpad')));
    
    // Serve launchpad pages
    app.get('/launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/launchpad.html'));
    });
    app.get('/fiori-launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/fioriLaunchpad.html'));
    });
    app.get('/debug-launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, 'debugLaunchpad.html'));
    });
    app.get('/launchpad-simple.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/launchpadSimple.html'));
    });
    
    // SAP Fiori flexibility services stub endpoints
    app.get('/sap/bc/lrep/flex/settings', (req, res) => {
        res.status(200).json({
            "isKeyUser": false,
            "isAtoAvailable": false,
            "isAtoEnabled": false,
            "isProductiveSystem": true,
            "isZeroDowntimeUpgradeRunning": false,
            "system": "",
            "client": ""
        });
    });
    
    app.get('/sap/bc/lrep/flex/data/:appId', (req, res) => {
        res.status(200).json({
            changes: [],
            ui2personalization: {},
            variants: []
        });
    });
    
    // Setup monitoring routes (enhanced)
    monitoringIntegration.setupRoutes(app);
    
    // Cache management endpoints
    app.get('/cache/stats', async (req, res) => {
        if (!req.user || !req.user.scope?.includes('Admin')) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        const stats = await cacheMiddleware.getStats();
        res.json(stats);
    });
    
    app.post('/cache/invalidate', async (req, res) => {
        if (!req.user || !req.user.scope?.includes('Admin')) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        const { pattern } = req.body;
        if (!pattern) {
            return res.status(400).json({ error: 'Pattern required' });
        }
        await cacheMiddleware.invalidate(pattern);
        res.json({ success: true });
    });
    
    // Logging endpoints
    app.get('/logs/audit', (req, res) => {
        if (!req.user || !req.user.scope?.includes('Admin')) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        // Return recent audit logs (implementation depends on your storage)
        res.json({ message: 'Audit logs endpoint - implement based on your storage' });
    });
    
    // Security monitoring dashboard
    app.get('/security/events', (req, res) => {
        if (!req.user || !req.user.scope?.includes('Admin')) {
            return res.status(403).json({ error: 'Forbidden' });
        }
        // Return recent security events
        res.json({ message: 'Security events endpoint - implement based on your requirements' });
    });
    
    // Network statistics service status endpoint
    app.get('/network-stats/status', (req, res) => {
        const status = networkStats.getStatus();
        res.json(status);
    });

    // Comprehensive UI Health Check endpoint
    app.get('/health/ui', async (req, res) => {
        try {
            const UIHealthChecker = require('./lib/sapUiHealthCheck');
            const healthChecker = new UIHealthChecker('http://localhost:4004');
            
            cds.log('server').info('Starting comprehensive UI health check...');
            const results = await healthChecker.runFullHealthCheck();
            
            res.status(results.overall === 'HEALTHY' ? 200 : 
                      results.overall === 'WARNING' ? 202 : 503).json(results);
        } catch (error) {
            res.status(500).json({
                overall: 'ERROR',
                message: 'Health check system failed',
                error: error.message,
                timestamp: new Date().toISOString()
            });
        }
    });
});

// Graceful shutdown handling
process.on('SIGINT', () => {
    const log = cds.log('server');
    log.info('Received SIGINT, shutting down gracefully');
    
    // Stop network statistics service
    networkStats.stop();
    
    process.exit(0);
});

process.on('SIGTERM', () => {
    const log = cds.log('server');
    log.info('Received SIGTERM, shutting down gracefully');
    
    // Stop network statistics service
    networkStats.stop();
    
    process.exit(0);
});

// Start periodic tasks after server is listening
cds.on('listening', async (info) => {
    // Start the HTTP server with WebSocket if it's not already started by CAP
    if (httpServer && !httpServer.listening) {
        const port = info.port || process.env.PORT || 4004;
        httpServer.listen(port, () => {
            cds.log('server').info(`HTTP server with WebSocket listening on port ${port}`);
        });
    }
    const log = cds.log('jobs');
    
    // Initialize database connection
    try {
        dbConnection = await initializeDatabase();
        log.info('Database connection initialized successfully');
        
        // Warm up cache with frequently accessed data
        await cacheMiddleware.warmUp();
        log.info('Cache warm-up completed');
        
    } catch (error) {
        log.error('Failed to initialize database:', error.message);
        // Continue running without database for now
    }
    
    // Initialize tracing collector
    try {
        const operationsService = await cds.connect.to('OperationsService');
        const { OperationsServiceCollector } = require('./lib/sapDistributedTracing');
        tracing.addCollector(new OperationsServiceCollector(operationsService));
        log.info('Distributed tracing collector initialized');
    } catch (error) {
        log.warn('Failed to initialize tracing collector:', error.message);
    }
    
    // Log application startup to monitoring
    monitoring.log('info', 'A2A Network application started', {
        logger: 'server',
        version: process.env.APP_VERSION || '1.0.0',
        nodeVersion: process.version,
        environment: process.env.NODE_ENV || 'development'
    });
    
    // Start network statistics service
    try {
        await networkStats.start();
        log.info('Network statistics service started successfully');
    } catch (error) {
        log.error('Failed to start network statistics service:', error.message);
    }
    
    // Flush logs periodically
    setInterval(async () => {
        try {
            await monitoring.flushLogs();
        } catch (error) {
            log.error('Failed to flush logs:', error);
        }
    }, 5000);
    
    // Create Cloud ALM dashboard
    try {
        await cloudALM.createDashboard();
        log.info('Cloud ALM dashboard created');
    } catch (error) {
        log.error('Failed to create Cloud ALM dashboard:', error);
    }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    monitoring.log('info', 'Received SIGTERM signal, shutting down gracefully', {
        logger: 'server'
    });
    
    try {
        // Flush any remaining logs
        await monitoring.flushLogs();
        process.exit(0);
    } catch (error) {
        cds.log('server').error('Error during shutdown:', error);
        process.exit(1);
    }
});
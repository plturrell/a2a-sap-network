/**
 * @fileoverview SAP CAP Server Configuration with Enterprise Middleware
 * @since 1.0.0
 * @module server
 * 
 * Main server configuration for the A2A Network service implementing
 * SAP enterprise patterns including security, monitoring, caching,
 * health checks, and distributed tracing - SAP CAP Framework Compliant
 */

const cds = require('@sap/cds');
const express = require('express');
const path = require('path');
const { Server } = require('socket.io');
const { applySecurityMiddleware } = require('./middleware/security');
const { applyAuthMiddleware, initializeXSUAAStrategy } = require('./middleware/auth');
const { initializeEnvironmentValidation } = require('./middleware/envValidation');
const { validateSQLMiddleware } = require('./middleware/sqlSecurity');
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
const A2ANotificationService = require('./notificationService');
const A2AWebSocketDataService = require('./websocketDataService');

// New enterprise middleware
const enterpriseLogger = require('./middleware/sapEnterpriseLogging');
const cacheMiddleware = require('./middleware/sapCacheMiddleware');
const securityHardening = require('./middleware/sapSecurityHardening');
const monitoringIntegration = require('./middleware/sapMonitoringIntegration');
const cors = require('cors');
const apiRoutes = require('./apiRoutes');

// Initialize logging
const log = cds.log('server');

// Initialize enterprise services
let notificationService;
let websocketDataService;

// Standard CAP server setup - SAP Enterprise Standard
module.exports = cds.server;

// Global database connection and WebSocket server
let dbConnection = null;
let io = null;
let logFlushInterval = null; // Track log flush interval for cleanup

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

// SAP CAP Framework Compliant Bootstrap - Express App Configuration Only
cds.on('bootstrap', async (app) => {
    // CRITICAL: Validate environment variables first
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
    initializeI18n(app);
    
    // Initialize XSUAA authentication strategy
    initializeXSUAAStrategy();
    
    // Apply CORS with SAP-specific configuration
    const corsOptions = {
        origin: function (origin, callback) {
            const allowedOrigins = process.env.ALLOWED_ORIGINS 
                ? process.env.ALLOWED_ORIGINS.split(',') 
                : ['http://localhost:3000', 'http://localhost:8080'];
            
            // SECURITY FIX: Only allow requests with no origin in development mode
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
    
    // Apply rate limiting with exclusions for tile API endpoints
    const rateLimits = securityHardening.rateLimiting();
    app.use('/auth', rateLimits.auth);
    
    // SECURITY FIX: Apply rate limiting to all API endpoints in production
    app.use('/api/v1/Agents', (req, res, next) => {
        // Only skip rate limiting in development for specific dashboard queries
        if (process.env.NODE_ENV === 'development' && 
            req.query.id && 
            (req.query.id.includes('dashboard') || req.query.id.includes('visualization'))) {
            return next();
        }
        return rateLimits.strict(req, res, next);
    });
    
    app.use('/api/v1/Services', (req, res, next) => {
        // Only skip rate limiting in development for specific dashboard queries
        if (process.env.NODE_ENV === 'development' && 
            req.query.id && 
            req.query.id.includes('dashboard')) {
            return next();
        }
        return rateLimits.standard(req, res, next);
    });
    
    // Apply default rate limiting with static file exclusions only
    app.use((req, res, next) => {
        // SECURITY FIX: Only skip rate limiting for genuine static files, not API endpoints
        const staticFilePaths = ['/app/', '/static/', '/assets/', '/resources/', '/shells/', '/common/'];
        if (staticFilePaths.some(staticPath => req.path.startsWith(staticPath))) {
            return next();
        }
        
        // SECURITY FIX: Apply rate limiting to all API endpoints including NetworkStats
        return rateLimits.standard(req, res, next);
    });
    
    app.use(rateLimits.slowDown);
    
    // Apply enterprise logging middleware
    app.use(enterpriseLogger.requestMiddleware());
    
    // Apply caching middleware
    app.use(cacheMiddleware.middleware());
    
    // Apply monitoring middleware
    app.use(monitoringIntegration.metricsMiddleware());
    
    // Apply SQL security middleware
    app.use(validateSQLMiddleware);
    
    // Add static file serving for JavaScript resources - SAP Enterprise Standard (BEFORE input validation)
    
    // Serve common JavaScript components with proper MIME types
    app.use('/common', express.static(path.join(__dirname, '../common'), {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));
    
    // Serve A2A Agents static files with proper MIME types
    app.use('/a2aAgents', express.static(path.join(__dirname, '../../a2aAgents'), {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));
    
    // Serve A2A Fiori webapp with proper MIME types
    app.use('/a2aFiori', express.static(path.join(__dirname, '../a2aFiori'), {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));
    
    // Serve A2A Fiori app at the standard SAP UShell expected path
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2aFiori'), {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            }
        }
    }));
    
    // CRITICAL FIX: Serve launchpad static files BEFORE blocking middleware
    app.use('/app', express.static(path.join(__dirname, '../app'), {
        setHeaders: (res, filePath) => {
            if (filePath.endsWith('.js')) {
                res.setHeader('Content-Type', 'application/javascript');
            } else if (filePath.endsWith('.html')) {
                res.setHeader('Content-Type', 'text/html');
            }
        }
    }));
    
    // Serve shells directory for Fiori Sandbox configuration BEFORE blocking middleware
    app.use('/shells', express.static(path.join(__dirname, '../app/shells')));
    
    // CRITICAL FIX: Bypass validation for launchpad tile endpoints
    app.use((req, res, next) => {
        const bypassPaths = [
            '/api/v1/Agents',
            '/api/v1/Services', 
            '/api/v1/blockchain/stats',
            '/api/v1/network/health',
            '/api/v1/notifications/count',
            '/api/v1/NetworkStats'
        ];
        
        const shouldBypass = bypassPaths.some(path => req.path.startsWith(path));
        if (shouldBypass) {
            log.debug(`Bypassing validation for: ${req.path}`);
            return next();
        }
        
        // Apply input validation for all other endpoints
        securityHardening.inputValidation()(req, res, next);
    });
    
    // Apply response filtering
    app.use(securityHardening.responseFilter());
    
    // Add API routes BEFORE authentication middleware
    app.use(apiRoutes);
    
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
    
    // Apply enhanced error handling middleware
    const { expressErrorMiddleware } = require('./utils/errorHandler');
    app.use(expressErrorMiddleware);
    
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
    
    // Serve UI5 app (duplicate removed - now served earlier before blocking middleware)
    // app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2aFiori/webapp')));
    // app.use('/app/launchpad', express.static(path.join(__dirname, '../app/launchpad')));
    
    // Serve shells directory for Fiori Sandbox configuration (duplicate removed - now served earlier)
    // app.use('/shells', express.static(path.join(__dirname, '../app/shells')));
    
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
        // SECURITY FIX: Improve authorization check for Admin access
        if (!req.user || 
            (!req.user.scope?.includes('Admin') && 
             !req.user.roles?.includes('Admin') && 
             !req.user.sapRoles?.includes('Admin'))) {
            return res.status(403).json({ error: 'Admin role required' });
        }
        const stats = await cacheMiddleware.getStats();
        res.json(stats);
    });
    
    app.post('/cache/invalidate', async (req, res) => {
        // SECURITY FIX: Improve authorization check for Admin access
        if (!req.user || 
            (!req.user.scope?.includes('Admin') && 
             !req.user.roles?.includes('Admin') && 
             !req.user.sapRoles?.includes('Admin'))) {
            return res.status(403).json({ error: 'Admin role required' });
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
        // SECURITY FIX: Improve authorization check for Admin access
        if (!req.user || 
            (!req.user.scope?.includes('Admin') && 
             !req.user.roles?.includes('Admin') && 
             !req.user.sapRoles?.includes('Admin'))) {
            return res.status(403).json({ error: 'Admin role required' });
        }
        // Return recent audit logs (implementation depends on your storage)
        res.json({ message: 'Audit logs endpoint - implement based on your storage' });
    });
    
    // Security monitoring dashboard
    app.get('/security/events', (req, res) => {
        // SECURITY FIX: Improve authorization check for Admin access
        if (!req.user || 
            (!req.user.scope?.includes('Admin') && 
             !req.user.roles?.includes('Admin') && 
             !req.user.sapRoles?.includes('Admin'))) {
            return res.status(403).json({ error: 'Admin role required' });
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
            const port = process.env.PORT || 4004;
            const healthCheckUrl = process.env.UI_HEALTH_CHECK_URL || `http://localhost:${port}`;
            const healthChecker = new UIHealthChecker(healthCheckUrl);
            
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

    // NOTE: No HTTP server creation or return - CDS framework handles this
    log.info('SAP CAP server bootstrap completed successfully');
});

// SAP CAP Framework Compliant WebSocket Initialization - After Server is Listening
cds.on('listening', async (info) => {
    const log = cds.log('server');
    
    try {
        // Get the HTTP server created by CDS framework
        const httpServer = cds.app.server;
        
        // Initialize WebSocket server using CDS-managed HTTP server
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
                    // SECURITY FIX: Only allow development bypass in development environment
                    if (process.env.USE_DEVELOPMENT_AUTH === 'true' && process.env.NODE_ENV === 'development') {
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
        
    } catch (error) {
        log.error('Failed to initialize WebSocket server:', error.message);
    }
    
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
        environment: process.env.NODE_ENV || 'development',
        port: info.port
    });
    
    // Initialize enterprise services
    try {
        // Initialize notification service
        notificationService = new A2ANotificationService();
        log.info('✅ A2A Notification Service initialized successfully');
        
        // Initialize real-time data service
        websocketDataService = new A2AWebSocketDataService();
        log.info('✅ A2A Real-Time Data Service initialized successfully');
    } catch (error) {
        log.error('❌ Failed to initialize A2A Enterprise Services:', error);
    }
    
    // Start network statistics service
    try {
        await networkStats.start();
        log.info('Network statistics service started successfully');
    } catch (error) {
        log.error('Failed to start network statistics service:', error.message);
    }
    
    // Flush logs periodically with non-blocking async operation
    logFlushInterval = setInterval(() => {
        // Use setImmediate to avoid blocking the event loop
        setImmediate(async () => {
            try {
                await monitoring.flushLogs();
            } catch (error) {
                log.error('Failed to flush logs:', error);
            }
        });
    }, 5000);
    
    // Create Cloud ALM dashboard
    try {
        await cloudALM.createDashboard();
        log.info('Cloud ALM dashboard created');
    } catch (error) {
        log.error('Failed to create Cloud ALM dashboard:', error);
    }
    
    log.info(`SAP CAP server listening on port ${info.port} - OpenTelemetry monitoring: ENABLED`);
});

// Unified graceful shutdown handler
async function shutdown() {
    const log = cds.log('server');
    try {
        log.info('Shutting down gracefully...');
        
        // Clear log flush interval
        if (logFlushInterval) {
            clearInterval(logFlushInterval);
            log.info('Log flush interval cleared');
        }
        
        // Stop network statistics service
        if (networkStats && typeof networkStats.stop === 'function') {
            await networkStats.stop();
        }
        
        // Close WebSocket server
        if (io) {
            await new Promise((resolve) => {
                io.close(() => {
                    log.info('WebSocket server closed');
                    resolve();
                });
            });
        }
        
        // Close database connection if exists
        if (dbConnection && typeof dbConnection.close === 'function') {
            await dbConnection.close();
            log.info('Database connection closed');
        } else if (dbConnection) {
            log.info('Database connection exists but no close method available');
        }
        
        // Flush any remaining logs
        await monitoring.flushLogs();
        log.info('Logs flushed successfully');
        
        return 0;
    } catch (error) {
        log.error('Error during shutdown:', error);
        return 1;
    }
}

process.on('SIGINT', async () => {
    const exitCode = await shutdown();
    process.exit(exitCode);
});

process.on('SIGTERM', async () => {
    const exitCode = await shutdown();
    process.exit(exitCode);
});

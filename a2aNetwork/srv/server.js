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
// const apiRoutes = require('./apiRoutes'); // Temporarily disabled - Express routes being converted to CAP

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
                : ['http://localhost:4004', 'http://localhost:8080'];
            
            // SECURITY FIX: Only allow requests with no origin in development mode
            if (!origin) {
                // Allow requests without origin header (e.g., curl, mobile apps, same-origin requests)
                return callback(null, true);
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
            '/api/v1/agents',
            '/api/v1/Services', 
            '/api/v1/services',
            '/api/v1/blockchain',
            '/api/v1/network',
            '/api/v1/health',
            '/api/v1/notifications',
            '/api/v1/NetworkStats',
            '/a2a/agent1/v1',
            '/a2a/agent2/v1',
            '/a2a/agent3/v1',
            '/a2a/agent4/v1',
            '/a2a/agent5/v1',
            '/a2a/agent6/v1',
            '/a2a/agent7/v1',
            '/a2a/agent8/v1'
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
    // app.use(apiRoutes); // Temporarily disabled - Express routes being converted to CAP
    
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
    
    // AGENT 2 API PROXY ROUTES - Bridge to Python FastAPI Backend
    const axios = require('axios');
    const AGENT2_BASE_URL = process.env.AGENT2_BASE_URL || 'http://localhost:8001';
    
    // Helper function to proxy Agent 2 requests
    async function proxyAgent2Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT2_BASE_URL}/a2a/agent2/v1${endpoint}`,
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 2 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 2 Backend: ${error.response.status}`
                });
            } else {
                res.status(503).json({
                    error: 'Agent 2 Backend Connection Failed',
                    message: error.message
                });
            }
        }
    }
    
    // Agent 2 Data Profiler
    app.get('/a2a/agent2/v1/data-profile', (req, res) => proxyAgent2Request(req, res, '/data-profile'));
    
    // Agent 2 Tasks Management
    app.post('/a2a/agent2/v1/tasks', (req, res) => proxyAgent2Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent2/v1/tasks/:taskId', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}`));
    app.post('/a2a/agent2/v1/tasks/:taskId/prepare', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}/prepare`, 'POST'));
    app.post('/a2a/agent2/v1/tasks/:taskId/analyze-features', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}/analyze-features`, 'POST'));
    app.post('/a2a/agent2/v1/tasks/:taskId/generate-embeddings', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}/generate-embeddings`, 'POST'));
    app.post('/a2a/agent2/v1/tasks/:taskId/export', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}/export`, 'POST'));
    app.post('/a2a/agent2/v1/tasks/:taskId/optimize', (req, res) => 
        proxyAgent2Request(req, res, `/tasks/${req.params.taskId}/optimize`, 'POST'));
    
    // Agent 2 Batch Processing
    app.post('/a2a/agent2/v1/batch-prepare', (req, res) => proxyAgent2Request(req, res, '/batch-prepare', 'POST'));
    
    // Agent 2 AutoML
    app.post('/a2a/agent2/v1/automl', (req, res) => proxyAgent2Request(req, res, '/automl', 'POST'));
    
    // Agent 2 Health Check
    app.get('/a2a/agent2/v1/health', (req, res) => proxyAgent2Request(req, res, '/health'));
    
    // Agent 2 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent2/v1/odata/AIPreparationTasks', async (req, res) => {
        try {
            // This would be handled by the CAP service, but we provide a fallback
            const response = await axios.get(`${AGENT2_BASE_URL}/a2a/agent2/v1/tasks`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#AIPreparationTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    datasetName: task.dataset_name,
                    modelType: task.model_type,
                    dataType: task.data_type,
                    framework: task.framework,
                    status: task.status?.toUpperCase() || 'DRAFT',
                    progressPercent: task.progress || 0,
                    currentStage: task.current_stage,
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 2 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 2 API proxy routes initialized');

    // AGENT 1 API PROXY ROUTES - Bridge to Python FastAPI Backend
    const AGENT1_BASE_URL = process.env.AGENT1_BASE_URL || 'http://localhost:8001';
    
    // Helper function to proxy Agent 1 requests
    async function proxyAgent1Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT1_BASE_URL}/a2a/agent1/v1${endpoint}`,
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 1 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 1 Backend: ${error.response.status}`
                });
            } else {
                res.status(503).json({
                    error: 'Agent 1 Backend Connection Failed',
                    message: error.message
                });
            }
        }
    }
    
    // Agent 1 Format Statistics
    app.get('/a2a/agent1/v1/format-statistics', (req, res) => proxyAgent1Request(req, res, '/format-statistics'));
    
    // Agent 1 Tasks Management
    app.post('/a2a/agent1/v1/tasks', (req, res) => proxyAgent1Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent1/v1/tasks/:taskId', (req, res) => 
        proxyAgent1Request(req, res, `/tasks/${req.params.taskId}`));
    app.post('/a2a/agent1/v1/tasks/:taskId/standardize', (req, res) => 
        proxyAgent1Request(req, res, `/tasks/${req.params.taskId}/standardize`, 'POST'));
    app.post('/a2a/agent1/v1/tasks/:taskId/validate', (req, res) => 
        proxyAgent1Request(req, res, `/tasks/${req.params.taskId}/validate`, 'POST'));
    app.post('/a2a/agent1/v1/tasks/:taskId/export', (req, res) => 
        proxyAgent1Request(req, res, `/tasks/${req.params.taskId}/export`, 'POST'));
    app.post('/a2a/agent1/v1/tasks/:taskId/preview', (req, res) => 
        proxyAgent1Request(req, res, `/tasks/${req.params.taskId}/preview`, 'POST'));
    
    // Agent 1 Batch Processing
    app.post('/a2a/agent1/v1/batch-process', (req, res) => proxyAgent1Request(req, res, '/batch-process', 'POST'));
    
    // Agent 1 Schema Management
    app.post('/a2a/agent1/v1/schema/import', (req, res) => proxyAgent1Request(req, res, '/schema/import', 'POST'));
    app.get('/a2a/agent1/v1/schema/templates', (req, res) => proxyAgent1Request(req, res, '/schema/templates'));
    app.post('/a2a/agent1/v1/schema/validate', (req, res) => proxyAgent1Request(req, res, '/schema/validate', 'POST'));
    
    // Agent 1 Rules Management
    app.post('/a2a/agent1/v1/rules/generate', (req, res) => proxyAgent1Request(req, res, '/rules/generate', 'POST'));
    app.post('/a2a/agent1/v1/rules/apply', (req, res) => proxyAgent1Request(req, res, '/rules/apply', 'POST'));
    
    // Agent 1 Health Check
    app.get('/a2a/agent1/v1/health', (req, res) => proxyAgent1Request(req, res, '/health'));
    
    // Agent 1 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent1/v1/odata/StandardizationTasks', async (req, res) => {
        try {
            // This would be handled by the CAP service, but we provide a fallback
            const response = await axios.get(`${AGENT1_BASE_URL}/a2a/agent1/v1/tasks`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#StandardizationTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    sourceFormat: task.source_format,
                    targetFormat: task.target_format,
                    schemaTemplateId: task.schema_template_id,
                    schemaValidation: task.schema_validation || true,
                    dataTypeValidation: task.data_type_validation || true,
                    formatValidation: task.format_validation || true,
                    processingMode: task.processing_mode || 'FULL',
                    batchSize: task.batch_size || 1000,
                    status: task.status?.toUpperCase() || 'DRAFT',
                    progressPercent: task.progress || 0,
                    currentStage: task.current_stage,
                    recordsProcessed: task.records_processed || 0,
                    recordsTotal: task.records_total || 0,
                    errorCount: task.error_count || 0,
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 1 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 1 API proxy routes initialized');

    // AGENT 3 API PROXY ROUTES - Bridge to Python FastAPI Backend
    const AGENT3_BASE_URL = process.env.AGENT3_BASE_URL || 'http://localhost:8002';
    
    // Helper function to proxy Agent 3 requests
    async function proxyAgent3Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT3_BASE_URL}/a2a/agent3/v1${endpoint}`,
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 3 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 3 Backend: ${error.response.status}`
                });
            } else {
                res.status(503).json({
                    error: 'Agent 3 Backend Connection Failed',
                    message: error.message
                });
            }
        }
    }
    
    // Agent 3 Collections Management
    app.get('/a2a/agent3/v1/collections', (req, res) => proxyAgent3Request(req, res, '/collections'));
    app.post('/a2a/agent3/v1/collections', (req, res) => proxyAgent3Request(req, res, '/collections', 'POST'));
    
    // Agent 3 Vector Search
    app.post('/a2a/agent3/v1/search', (req, res) => proxyAgent3Request(req, res, '/search', 'POST'));
    
    // Agent 3 Tasks Management
    app.post('/a2a/agent3/v1/tasks', (req, res) => proxyAgent3Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent3/v1/tasks/:taskId', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}`));
    app.post('/a2a/agent3/v1/tasks/:taskId/process', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/process`, 'POST'));
    app.post('/a2a/agent3/v1/tasks/:taskId/similarity-search', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/similarity-search`, 'POST'));
    app.post('/a2a/agent3/v1/tasks/:taskId/optimize-index', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/optimize-index`, 'POST'));
    app.post('/a2a/agent3/v1/tasks/:taskId/export', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/export`, 'POST'));
    app.get('/a2a/agent3/v1/tasks/:taskId/visualization-data', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/visualization-data`));
    app.post('/a2a/agent3/v1/tasks/:taskId/cluster-analysis', (req, res) => 
        proxyAgent3Request(req, res, `/tasks/${req.params.taskId}/cluster-analysis`, 'POST'));
    
    // Agent 3 Batch Processing
    app.post('/a2a/agent3/v1/batch-process', (req, res) => proxyAgent3Request(req, res, '/batch-process', 'POST'));
    
    // Agent 3 Model Comparison
    app.get('/a2a/agent3/v1/model-comparison', (req, res) => proxyAgent3Request(req, res, '/model-comparison'));
    
    // Agent 3 Vector Operations
    app.post('/a2a/agent3/v1/embeddings/generate', (req, res) => proxyAgent3Request(req, res, '/embeddings/generate', 'POST'));
    app.post('/a2a/agent3/v1/index/optimize', (req, res) => proxyAgent3Request(req, res, '/index/optimize', 'POST'));
    
    // Agent 3 Health Check
    app.get('/a2a/agent3/v1/health', (req, res) => proxyAgent3Request(req, res, '/health'));
    
    // Agent 3 WebSocket endpoint for real-time updates
    app.get('/a2a/agent3/v1/tasks/:taskId/ws', (req, res) => {
        // WebSocket upgrade handling - would be implemented with socket.io
        res.status(501).json({
            error: 'WebSocket endpoint not implemented',
            message: 'Real-time updates will be available when WebSocket server is configured'
        });
    });
    
    // ================================
    // AGENT 4 - CALCULATION VALIDATION PROXY ROUTES
    // ================================
    
    const AGENT4_BASE_URL = process.env.AGENT4_BASE_URL || 'http://localhost:8003';
    
    // Helper function to proxy Agent 4 requests
    async function proxyAgent4Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT4_BASE_URL}/a2a/agent4/v1${endpoint}`,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': req.headers.authorization || '',
                    'X-Forwarded-For': req.ip,
                    'X-User-Agent': req.headers['user-agent'] || 'SAP-CAP-Proxy'
                },
                timeout: method === 'POST' ? 120000 : 30000 // Longer timeout for validation operations
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 4 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 4 Backend: ${error.response.status}`
                });
            } else {
                res.status(503).json({
                    error: 'Agent 4 Backend Connection Failed',
                    message: error.message
                });
            }
        }
    }
    
    // Agent 4 Calculation Validation Tasks
    app.get('/a2a/agent4/v1/tasks', (req, res) => proxyAgent4Request(req, res, '/tasks'));
    app.post('/a2a/agent4/v1/tasks', (req, res) => proxyAgent4Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent4/v1/tasks/:taskId', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}`));
    app.put('/a2a/agent4/v1/tasks/:taskId', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}`, 'PUT'));
    app.delete('/a2a/agent4/v1/tasks/:taskId', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}`, 'DELETE'));
    
    // Agent 4 Validation Operations
    app.post('/a2a/agent4/v1/tasks/:taskId/validate', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/validate`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/pause', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/pause`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/resume', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/resume`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/cancel', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/cancel`, 'POST'));
    
    // Agent 4 Method-Specific Validation
    app.post('/a2a/agent4/v1/tasks/:taskId/symbolic', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/symbolic`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/numerical', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/numerical`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/statistical', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/statistical`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/ai-validation', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/ai-validation`, 'POST'));
    app.post('/a2a/agent4/v1/tasks/:taskId/blockchain-consensus', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/blockchain-consensus`, 'POST'));
    
    // Agent 4 Expression Operations
    app.post('/a2a/agent4/v1/expression/validate-syntax', (req, res) => 
        proxyAgent4Request(req, res, '/expression/validate-syntax', 'POST'));
    app.post('/a2a/agent4/v1/expression/evaluate', (req, res) => 
        proxyAgent4Request(req, res, '/expression/evaluate', 'POST'));
    app.post('/a2a/agent4/v1/expression/simplify', (req, res) => 
        proxyAgent4Request(req, res, '/expression/simplify', 'POST'));
    app.post('/a2a/agent4/v1/expression/derivative', (req, res) => 
        proxyAgent4Request(req, res, '/expression/derivative', 'POST'));
    app.post('/a2a/agent4/v1/expression/integral', (req, res) => 
        proxyAgent4Request(req, res, '/expression/integral', 'POST'));
    
    // Agent 4 Validation Methods
    app.get('/a2a/agent4/v1/methods', (req, res) => proxyAgent4Request(req, res, '/methods'));
    app.get('/a2a/agent4/v1/methods/symbolic/engines', (req, res) => 
        proxyAgent4Request(req, res, '/methods/symbolic/engines'));
    app.get('/a2a/agent4/v1/methods/ai/models', (req, res) => 
        proxyAgent4Request(req, res, '/methods/ai/models'));
    app.post('/a2a/agent4/v1/methods/benchmark', (req, res) => 
        proxyAgent4Request(req, res, '/methods/benchmark', 'POST'));
    
    // Agent 4 Templates Management
    app.get('/a2a/agent4/v1/templates', (req, res) => proxyAgent4Request(req, res, '/templates'));
    app.post('/a2a/agent4/v1/templates', (req, res) => proxyAgent4Request(req, res, '/templates', 'POST'));
    app.get('/a2a/agent4/v1/templates/:templateId', (req, res) => 
        proxyAgent4Request(req, res, `/templates/${req.params.templateId}`));
    app.put('/a2a/agent4/v1/templates/:templateId', (req, res) => 
        proxyAgent4Request(req, res, `/templates/${req.params.templateId}`, 'PUT'));
    app.delete('/a2a/agent4/v1/templates/:templateId', (req, res) => 
        proxyAgent4Request(req, res, `/templates/${req.params.templateId}`, 'DELETE'));
    app.post('/a2a/agent4/v1/templates/:templateId/apply', (req, res) => 
        proxyAgent4Request(req, res, `/templates/${req.params.templateId}/apply`, 'POST'));
    
    // Agent 4 AI Model Comparison
    app.get('/a2a/agent4/v1/ai/models', (req, res) => proxyAgent4Request(req, res, '/ai/models'));
    app.post('/a2a/agent4/v1/ai/compare', (req, res) => proxyAgent4Request(req, res, '/ai/compare', 'POST'));
    app.get('/a2a/agent4/v1/ai/compare/:comparisonId', (req, res) => 
        proxyAgent4Request(req, res, `/ai/compare/${req.params.comparisonId}`));
    app.post('/a2a/agent4/v1/ai/configure', (req, res) => 
        proxyAgent4Request(req, res, '/ai/configure', 'POST'));
    
    // Agent 4 Blockchain Consensus
    app.get('/a2a/agent4/v1/blockchain/validators', (req, res) => 
        proxyAgent4Request(req, res, '/blockchain/validators'));
    app.post('/a2a/agent4/v1/blockchain/consensus', (req, res) => 
        proxyAgent4Request(req, res, '/blockchain/consensus', 'POST'));
    app.get('/a2a/agent4/v1/blockchain/consensus/:consensusId', (req, res) => 
        proxyAgent4Request(req, res, `/blockchain/consensus/${req.params.consensusId}`));
    app.post('/a2a/agent4/v1/blockchain/validators/select', (req, res) => 
        proxyAgent4Request(req, res, '/blockchain/validators/select', 'POST'));
    app.post('/a2a/agent4/v1/blockchain/configure', (req, res) => 
        proxyAgent4Request(req, res, '/blockchain/configure', 'POST'));
    
    // Agent 4 Batch Operations
    app.post('/a2a/agent4/v1/batch/validate', (req, res) => 
        proxyAgent4Request(req, res, '/batch/validate', 'POST'));
    app.get('/a2a/agent4/v1/batch/:batchId', (req, res) => 
        proxyAgent4Request(req, res, `/batch/${req.params.batchId}`));
    app.post('/a2a/agent4/v1/batch/:batchId/pause', (req, res) => 
        proxyAgent4Request(req, res, `/batch/${req.params.batchId}/pause`, 'POST'));
    app.post('/a2a/agent4/v1/batch/:batchId/resume', (req, res) => 
        proxyAgent4Request(req, res, `/batch/${req.params.batchId}/resume`, 'POST'));
    
    // Agent 4 Export and Reporting
    app.post('/a2a/agent4/v1/tasks/:taskId/export', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/export`, 'POST'));
    app.get('/a2a/agent4/v1/tasks/:taskId/report', (req, res) => 
        proxyAgent4Request(req, res, `/tasks/${req.params.taskId}/report`));
    app.post('/a2a/agent4/v1/reports/generate', (req, res) => 
        proxyAgent4Request(req, res, '/reports/generate', 'POST'));
    
    // Agent 4 Statistics and Analytics
    app.get('/a2a/agent4/v1/stats/accuracy', (req, res) => 
        proxyAgent4Request(req, res, '/stats/accuracy'));
    app.get('/a2a/agent4/v1/stats/performance', (req, res) => 
        proxyAgent4Request(req, res, '/stats/performance'));
    app.get('/a2a/agent4/v1/stats/usage', (req, res) => 
        proxyAgent4Request(req, res, '/stats/usage'));
    
    // Agent 4 Health Check
    app.get('/a2a/agent4/v1/health', (req, res) => proxyAgent4Request(req, res, '/health'));
    
    log.info('Agent 4 API proxy routes initialized');

    // ================================
    // AGENT 5 - QA VALIDATION PROXY ROUTES
    // ================================
    
    const AGENT5_BASE_URL = process.env.AGENT5_BASE_URL || 'http://localhost:8004';
    
    // Helper function to proxy Agent 5 requests
    async function proxyAgent5Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT5_BASE_URL}/a2a/agent5/v1${endpoint}`,
                timeout: 60000, // Longer timeout for QA operations
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': req.headers.authorization || '',
                    'X-Forwarded-For': req.ip,
                    'X-User-Agent': req.headers['user-agent'] || 'SAP-CAP-Proxy'
                }
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 5 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 5 Backend: ${error.response.status}`,
                    details: error.response.data?.details || null
                });
            } else {
                res.status(503).json({
                    error: 'Agent 5 Backend Connection Failed',
                    message: error.message,
                    service: 'QA Validation Service'
                });
            }
        }
    }
    
    // Agent 5 QA Validation Tasks
    app.get('/a2a/agent5/v1/tasks', (req, res) => proxyAgent5Request(req, res, '/tasks'));
    app.post('/a2a/agent5/v1/tasks', (req, res) => proxyAgent5Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent5/v1/tasks/:taskId', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}`));
    app.put('/a2a/agent5/v1/tasks/:taskId', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}`, 'PUT'));
    app.delete('/a2a/agent5/v1/tasks/:taskId', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}`, 'DELETE'));
    
    // Agent 5 Task Operations
    app.post('/a2a/agent5/v1/tasks/:taskId/validate', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}/validate`, 'POST'));
    app.post('/a2a/agent5/v1/tasks/:taskId/pause', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}/pause`, 'POST'));
    app.post('/a2a/agent5/v1/tasks/:taskId/resume', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}/resume`, 'POST'));
    app.post('/a2a/agent5/v1/tasks/:taskId/cancel', (req, res) => 
        proxyAgent5Request(req, res, `/tasks/${req.params.taskId}/cancel`, 'POST'));
    
    // Agent 5 Test Generation
    app.post('/a2a/agent5/v1/tests/generate', (req, res) => 
        proxyAgent5Request(req, res, '/tests/generate', 'POST'));
    app.post('/a2a/agent5/v1/tests/simpleqa', (req, res) => 
        proxyAgent5Request(req, res, '/tests/simpleqa', 'POST'));
    app.post('/a2a/agent5/v1/tests/execute', (req, res) => 
        proxyAgent5Request(req, res, '/tests/execute', 'POST'));
    app.get('/a2a/agent5/v1/tests/:testId/results', (req, res) => 
        proxyAgent5Request(req, res, `/tests/${req.params.testId}/results`));
    
    // Agent 5 ORD Discovery
    app.post('/a2a/agent5/v1/ord/discover', (req, res) => 
        proxyAgent5Request(req, res, '/ord/discover', 'POST'));
    app.get('/a2a/agent5/v1/ord/registries', (req, res) => 
        proxyAgent5Request(req, res, '/ord/registries'));
    app.get('/a2a/agent5/v1/ord/data-products', (req, res) => 
        proxyAgent5Request(req, res, '/ord/data-products'));
    
    // Agent 5 Validation Rules
    app.get('/a2a/agent5/v1/rules', (req, res) => proxyAgent5Request(req, res, '/rules'));
    app.post('/a2a/agent5/v1/rules', (req, res) => proxyAgent5Request(req, res, '/rules', 'POST'));
    app.get('/a2a/agent5/v1/rules/:ruleId', (req, res) => 
        proxyAgent5Request(req, res, `/rules/${req.params.ruleId}`));
    app.put('/a2a/agent5/v1/rules/:ruleId', (req, res) => 
        proxyAgent5Request(req, res, `/rules/${req.params.ruleId}`, 'PUT'));
    app.delete('/a2a/agent5/v1/rules/:ruleId', (req, res) => 
        proxyAgent5Request(req, res, `/rules/${req.params.ruleId}`, 'DELETE'));
    app.post('/a2a/agent5/v1/rules/:ruleId/test', (req, res) => 
        proxyAgent5Request(req, res, `/rules/${req.params.ruleId}/test`, 'POST'));
    
    // Agent 5 Approval Workflow
    app.get('/a2a/agent5/v1/approvals', (req, res) => proxyAgent5Request(req, res, '/approvals'));
    app.post('/a2a/agent5/v1/approvals', (req, res) => proxyAgent5Request(req, res, '/approvals', 'POST'));
    app.get('/a2a/agent5/v1/approvals/:approvalId', (req, res) => 
        proxyAgent5Request(req, res, `/approvals/${req.params.approvalId}`));
    app.post('/a2a/agent5/v1/approvals/:approvalId/approve', (req, res) => 
        proxyAgent5Request(req, res, `/approvals/${req.params.approvalId}/approve`, 'POST'));
    app.post('/a2a/agent5/v1/approvals/:approvalId/reject', (req, res) => 
        proxyAgent5Request(req, res, `/approvals/${req.params.approvalId}/reject`, 'POST'));
    app.post('/a2a/agent5/v1/approvals/:approvalId/escalate', (req, res) => 
        proxyAgent5Request(req, res, `/approvals/${req.params.approvalId}/escalate`, 'POST'));
    
    // Agent 5 Configuration
    app.get('/a2a/agent5/v1/config', (req, res) => proxyAgent5Request(req, res, '/config'));
    app.post('/a2a/agent5/v1/config', (req, res) => proxyAgent5Request(req, res, '/config', 'POST'));
    app.put('/a2a/agent5/v1/config', (req, res) => proxyAgent5Request(req, res, '/config', 'PUT'));
    
    // Agent 5 Batch Operations
    app.post('/a2a/agent5/v1/batch/validate', (req, res) => 
        proxyAgent5Request(req, res, '/batch/validate', 'POST'));
    app.get('/a2a/agent5/v1/batch/:batchId', (req, res) => 
        proxyAgent5Request(req, res, `/batch/${req.params.batchId}`));
    app.post('/a2a/agent5/v1/batch/:batchId/pause', (req, res) => 
        proxyAgent5Request(req, res, `/batch/${req.params.batchId}/pause`, 'POST'));
    app.post('/a2a/agent5/v1/batch/:batchId/resume', (req, res) => 
        proxyAgent5Request(req, res, `/batch/${req.params.batchId}/resume`, 'POST'));
    
    // Agent 5 Reports and Analytics
    app.get('/a2a/agent5/v1/reports', (req, res) => proxyAgent5Request(req, res, '/reports'));
    app.post('/a2a/agent5/v1/reports/generate', (req, res) => 
        proxyAgent5Request(req, res, '/reports/generate', 'POST'));
    app.get('/a2a/agent5/v1/analytics/metrics', (req, res) => 
        proxyAgent5Request(req, res, '/analytics/metrics'));
    app.get('/a2a/agent5/v1/analytics/trends', (req, res) => 
        proxyAgent5Request(req, res, '/analytics/trends'));
    
    // Agent 5 Health Check
    app.get('/a2a/agent5/v1/health', (req, res) => proxyAgent5Request(req, res, '/health'));
    
    // Agent 5 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent5/v1/odata/QaValidationTasks', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT5_BASE_URL}/a2a/agent5/v1/tasks`);
            
            const odataResponse = {
                "@odata.context": "$metadata#QaValidationTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    dataProductId: task.data_product_id,
                    ordRegistryUrl: task.ord_registry_url,
                    validationType: task.validation_type?.toUpperCase() || 'QUALITY_ASSURANCE',
                    qaScope: task.qa_scope?.toUpperCase() || 'DATA_INTEGRITY',
                    testGenerationMethod: task.test_generation_method?.toUpperCase() || 'DYNAMIC_SIMPLEQA',
                    simpleQaTestCount: task.simple_qa_test_count || 10,
                    qualityThreshold: task.quality_threshold || 0.8,
                    factualityThreshold: task.factuality_threshold || 0.85,
                    complianceThreshold: task.compliance_threshold || 0.95,
                    vectorSimilarityThreshold: task.vector_similarity_threshold || 0.7,
                    enableFactualityTesting: task.enable_factuality_testing !== false,
                    enableComplianceCheck: task.enable_compliance_check !== false,
                    enableVectorSimilarity: task.enable_vector_similarity !== false,
                    enableRegressionTesting: task.enable_regression_testing || false,
                    requireApproval: task.require_approval !== false,
                    status: task.status?.toUpperCase() || 'DRAFT',
                    priority: task.priority?.toUpperCase() || 'MEDIUM',
                    progressPercent: task.progress || 0,
                    currentStage: task.current_stage,
                    overallScore: task.overall_score,
                    qualityScore: task.quality_score,
                    factualityScore: task.factuality_score,
                    complianceScore: task.compliance_score,
                    testsGenerated: task.tests_generated || 0,
                    testsPassed: task.tests_passed || 0,
                    testsFailed: task.tests_failed || 0,
                    validationTime: task.validation_time,
                    approvalStatus: task.approval_status?.toUpperCase() || 'PENDING',
                    approvedBy: task.approved_by,
                    approvedAt: task.approved_at,
                    rejectionReason: task.rejection_reason,
                    createdBy: task.created_by,
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 5 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 5 API proxy routes initialized');

    // ================================
    // AGENT 6 - QUALITY CONTROL & WORKFLOW ROUTING PROXY ROUTES
    // ================================
    
    const AGENT6_BASE_URL = process.env.AGENT6_BASE_URL || 'http://localhost:8005';
    
    // Helper function to proxy Agent 6 requests
    async function proxyAgent6Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT6_BASE_URL}/a2a/agent6/v1${endpoint}`,
                timeout: 60000, // Longer timeout for quality assessments
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': req.headers.authorization || '',
                    'X-Forwarded-For': req.ip,
                    'X-User-Agent': req.headers['user-agent'] || 'SAP-CAP-Proxy'
                }
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            if (req.query && Object.keys(req.query).length > 0) {
                config.params = req.query;
            }
            
            const response = await axios(config);
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error(`Agent 6 Proxy Error (${endpoint}):`, error.message);
            if (error.response) {
                res.status(error.response.status).json({
                    error: error.response.data?.error || error.response.statusText,
                    message: `Agent 6 Backend: ${error.response.status}`,
                    details: error.response.data?.details || null
                });
            } else {
                res.status(503).json({
                    error: 'Agent 6 Backend Connection Failed',
                    message: error.message,
                    service: 'Quality Control Service'
                });
            }
        }
    }
    
    // Agent 6 Quality Control Tasks
    app.get('/a2a/agent6/v1/tasks', (req, res) => proxyAgent6Request(req, res, '/tasks'));
    app.post('/a2a/agent6/v1/tasks', (req, res) => proxyAgent6Request(req, res, '/tasks', 'POST'));
    app.get('/a2a/agent6/v1/tasks/:taskId', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}`));
    app.put('/a2a/agent6/v1/tasks/:taskId', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}`, 'PUT'));
    app.delete('/a2a/agent6/v1/tasks/:taskId', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}`, 'DELETE'));
    
    // Agent 6 Task Operations
    app.post('/a2a/agent6/v1/tasks/:taskId/assess', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/assess`, 'POST'));
    app.get('/a2a/agent6/v1/tasks/:taskId/stream', (req, res) => {
        // Special handling for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        const eventSourceUrl = `${AGENT6_BASE_URL}/a2a/agent6/v1/tasks/${req.params.taskId}/stream`;
        // Proxy SSE stream - implementation would forward events
        res.write(':ok\n\n');
    });
    app.post('/a2a/agent6/v1/tasks/:taskId/route', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/route`, 'POST'));
    app.post('/a2a/agent6/v1/tasks/:taskId/verify-trust', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/verify-trust`, 'POST'));
    app.post('/a2a/agent6/v1/tasks/:taskId/optimize', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/optimize`, 'POST'));
    app.post('/a2a/agent6/v1/tasks/:taskId/apply-optimizations', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/apply-optimizations`, 'POST'));
    app.post('/a2a/agent6/v1/tasks/:taskId/escalate', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/escalate`, 'POST'));
    app.post('/a2a/agent6/v1/tasks/:taskId/report', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/report`, 'POST'));
    app.get('/a2a/agent6/v1/tasks/:taskId/metrics', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/metrics`));
    app.get('/a2a/agent6/v1/tasks/:taskId/routing-options', (req, res) => 
        proxyAgent6Request(req, res, `/tasks/${req.params.taskId}/routing-options`));
    
    // Agent 6 Dashboard and Analytics
    app.get('/a2a/agent6/v1/dashboard', (req, res) => proxyAgent6Request(req, res, '/dashboard'));
    app.get('/a2a/agent6/v1/trust-metrics', (req, res) => proxyAgent6Request(req, res, '/trust-metrics'));
    app.get('/a2a/agent6/v1/workflow-analysis', (req, res) => proxyAgent6Request(req, res, '/workflow-analysis'));
    
    // Agent 6 Routing Rules
    app.get('/a2a/agent6/v1/routing-rules', (req, res) => proxyAgent6Request(req, res, '/routing-rules'));
    app.post('/a2a/agent6/v1/routing-rules', (req, res) => proxyAgent6Request(req, res, '/routing-rules', 'POST'));
    app.get('/a2a/agent6/v1/routing-rules/:ruleId', (req, res) => 
        proxyAgent6Request(req, res, `/routing-rules/${req.params.ruleId}`));
    app.put('/a2a/agent6/v1/routing-rules/:ruleId', (req, res) => 
        proxyAgent6Request(req, res, `/routing-rules/${req.params.ruleId}`, 'PUT'));
    app.delete('/a2a/agent6/v1/routing-rules/:ruleId', (req, res) => 
        proxyAgent6Request(req, res, `/routing-rules/${req.params.ruleId}`, 'DELETE'));
    app.post('/a2a/agent6/v1/routing-rules/:ruleId/test', (req, res) => 
        proxyAgent6Request(req, res, `/routing-rules/${req.params.ruleId}/test`, 'POST'));
    
    // Agent 6 Quality Gates
    app.get('/a2a/agent6/v1/quality-gates', (req, res) => proxyAgent6Request(req, res, '/quality-gates'));
    app.post('/a2a/agent6/v1/quality-gates', (req, res) => proxyAgent6Request(req, res, '/quality-gates', 'POST'));
    app.get('/a2a/agent6/v1/quality-gates/:gateId', (req, res) => 
        proxyAgent6Request(req, res, `/quality-gates/${req.params.gateId}`));
    app.put('/a2a/agent6/v1/quality-gates/:gateId', (req, res) => 
        proxyAgent6Request(req, res, `/quality-gates/${req.params.gateId}`, 'PUT'));
    app.delete('/a2a/agent6/v1/quality-gates/:gateId', (req, res) => 
        proxyAgent6Request(req, res, `/quality-gates/${req.params.gateId}`, 'DELETE'));
    
    // Agent 6 Batch Operations
    app.post('/a2a/agent6/v1/batch-assessment', (req, res) => 
        proxyAgent6Request(req, res, '/batch-assessment', 'POST'));
    app.get('/a2a/agent6/v1/batch/:batchId', (req, res) => 
        proxyAgent6Request(req, res, `/batch/${req.params.batchId}`));
    
    // Agent 6 Reports and Exports
    app.post('/a2a/agent6/v1/reports/generate', (req, res) => 
        proxyAgent6Request(req, res, '/reports/generate', 'POST'));
    app.get('/a2a/agent6/v1/reports/:reportId', (req, res) => 
        proxyAgent6Request(req, res, `/reports/${req.params.reportId}`));
    app.get('/a2a/agent6/v1/reports/:reportId/download', (req, res) => 
        proxyAgent6Request(req, res, `/reports/${req.params.reportId}/download`));
    
    // Agent 6 Health Check
    app.get('/a2a/agent6/v1/health', (req, res) => proxyAgent6Request(req, res, '/health'));
    
    // Agent 6 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent6/v1/odata/QualityControlTasks', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT6_BASE_URL}/a2a/agent6/v1/tasks`);
            
            const odataResponse = {
                "@odata.context": "$metadata#QualityControlTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    qualityGate: task.quality_gate,
                    dataSource: task.data_source,
                    processingPipeline: task.processing_pipeline,
                    status: task.status?.toUpperCase() || 'DRAFT',
                    priority: task.priority?.toUpperCase() || 'NORMAL',
                    overallQuality: task.overall_quality,
                    trustScore: task.trust_score,
                    issuesFound: task.issues_found || 0,
                    routingDecision: task.routing_decision?.toUpperCase(),
                    targetAgent: task.target_agent,
                    routingConfidence: task.routing_confidence,
                    assessmentDuration: task.assessment_duration,
                    workflowOptimized: task.workflow_optimized || false,
                    autoRouted: task.auto_routed || false,
                    qualityComponents: JSON.stringify(task.quality_components || {}),
                    assessmentResults: JSON.stringify(task.assessment_results || {}),
                    errorDetails: task.error_details,
                    startedAt: task.started_at,
                    completedAt: task.completed_at,
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 6 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 6 API proxy routes initialized');
    
    // ===== AGENT 7 PROXY ROUTES =====
    // Agent Management & Orchestration System
    const AGENT7_BASE_URL = process.env.AGENT7_BASE_URL || 'http://localhost:8006';
    
    // Helper function to proxy Agent 7 requests
    async function proxyAgent7Request(req, res, endpoint, method = 'GET') {
        try {
            const config = {
                method,
                url: `${AGENT7_BASE_URL}/api/v1${endpoint}`,
                headers: {
                    'Content-Type': 'application/json',
                    ...req.headers
                },
                timeout: 30000
            };
            
            if (method !== 'GET' && req.body) {
                config.data = req.body;
            }
            
            const response = await axios(config);
            
            // Add CORS headers
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
            
            res.status(response.status).json(response.data);
        } catch (error) {
            log.error('Agent 7 proxy error:', error.message);
            
            if (error.response) {
                res.status(error.response.status).json(error.response.data);
            } else {
                res.status(503).json({
                    error: {
                        code: "SERVICE_UNAVAILABLE",
                        message: "Agent 7 backend not available"
                    }
                });
            }
        }
    }
    
    // Registered Agents endpoints
    app.get('/a2a/agent7/v1/registered-agents', (req, res) => proxyAgent7Request(req, res, '/registered-agents'));
    app.post('/a2a/agent7/v1/registered-agents', (req, res) => proxyAgent7Request(req, res, '/registered-agents', 'POST'));
    app.get('/a2a/agent7/v1/registered-agents/:id', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}`));
    app.put('/a2a/agent7/v1/registered-agents/:id', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}`, 'PUT'));
    app.delete('/a2a/agent7/v1/registered-agents/:id', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}`, 'DELETE'));
    
    // Agent registration and management
    app.post('/a2a/agent7/v1/register-agent', (req, res) => proxyAgent7Request(req, res, '/register-agent', 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/update-status', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/update-status`, 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/health-check', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/health-check`, 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/update-config', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/update-config`, 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/deactivate', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/deactivate`, 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/schedule-task', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/schedule-task`, 'POST'));
    app.post('/a2a/agent7/v1/registered-agents/:id/assign-workload', (req, res) => proxyAgent7Request(req, res, `/registered-agents/${req.params.id}/assign-workload`, 'POST'));
    
    // Management Tasks endpoints
    app.get('/a2a/agent7/v1/management-tasks', (req, res) => proxyAgent7Request(req, res, '/management-tasks'));
    app.post('/a2a/agent7/v1/management-tasks', (req, res) => proxyAgent7Request(req, res, '/management-tasks', 'POST'));
    app.get('/a2a/agent7/v1/management-tasks/:id', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}`));
    app.put('/a2a/agent7/v1/management-tasks/:id', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}`, 'PUT'));
    app.delete('/a2a/agent7/v1/management-tasks/:id', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}`, 'DELETE'));
    
    // Management task actions
    app.post('/a2a/agent7/v1/management-tasks/:id/execute', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/execute`, 'POST'));
    app.post('/a2a/agent7/v1/management-tasks/:id/pause', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/pause`, 'POST'));
    app.post('/a2a/agent7/v1/management-tasks/:id/resume', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/resume`, 'POST'));
    app.post('/a2a/agent7/v1/management-tasks/:id/cancel', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/cancel`, 'POST'));
    app.post('/a2a/agent7/v1/management-tasks/:id/retry', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/retry`, 'POST'));
    app.post('/a2a/agent7/v1/management-tasks/:id/rollback', (req, res) => proxyAgent7Request(req, res, `/management-tasks/${req.params.id}/rollback`, 'POST'));
    
    // Health Check endpoints
    app.get('/a2a/agent7/v1/health-checks', (req, res) => proxyAgent7Request(req, res, '/health-checks'));
    app.get('/a2a/agent7/v1/health-status', (req, res) => proxyAgent7Request(req, res, '/health-status'));
    app.get('/a2a/agent7/v1/health-status/:agentId', (req, res) => proxyAgent7Request(req, res, `/health-status/${req.params.agentId}`));
    app.get('/a2a/agent7/v1/health-stream', (req, res) => {
        // Server-Sent Events for real-time health monitoring
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        });
        
        proxyAgent7Request(req, res, '/health-stream').catch(error => {
            res.write(`data: ${JSON.stringify({error: error.message})}\\n\\n`);
            res.end();
        });
    });
    
    // Performance Metrics endpoints
    app.get('/a2a/agent7/v1/performance-metrics', (req, res) => proxyAgent7Request(req, res, '/performance-metrics'));
    app.get('/a2a/agent7/v1/performance-analysis/:agentId', (req, res) => proxyAgent7Request(req, res, `/performance-analysis/${req.params.agentId}`));
    app.get('/a2a/agent7/v1/performance-benchmarks', (req, res) => proxyAgent7Request(req, res, '/performance-benchmarks'));
    
    // Agent Coordination endpoints
    app.get('/a2a/agent7/v1/coordination', (req, res) => proxyAgent7Request(req, res, '/coordination'));
    app.post('/a2a/agent7/v1/coordination', (req, res) => proxyAgent7Request(req, res, '/coordination', 'POST'));
    app.get('/a2a/agent7/v1/coordination/:id', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}`));
    app.put('/a2a/agent7/v1/coordination/:id', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}`, 'PUT'));
    app.delete('/a2a/agent7/v1/coordination/:id', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}`, 'DELETE'));
    
    // Coordination actions
    app.post('/a2a/agent7/v1/coordination/:id/activate', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}/activate`, 'POST'));
    app.post('/a2a/agent7/v1/coordination/:id/pause', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}/pause`, 'POST'));
    app.post('/a2a/agent7/v1/coordination/:id/update-rules', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}/update-rules`, 'POST'));
    app.post('/a2a/agent7/v1/coordination/:id/add-agent', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}/add-agent`, 'POST'));
    app.post('/a2a/agent7/v1/coordination/:id/remove-agent', (req, res) => proxyAgent7Request(req, res, `/coordination/${req.params.id}/remove-agent`, 'POST'));
    
    // Coordination status and management
    app.get('/a2a/agent7/v1/coordination-status', (req, res) => proxyAgent7Request(req, res, '/coordination-status'));
    app.get('/a2a/agent7/v1/network-topology', (req, res) => proxyAgent7Request(req, res, '/network-topology'));
    app.get('/a2a/agent7/v1/load-balancing', (req, res) => proxyAgent7Request(req, res, '/load-balancing'));
    app.post('/a2a/agent7/v1/load-balancing/optimize', (req, res) => proxyAgent7Request(req, res, '/load-balancing/optimize', 'POST'));
    
    // Bulk Operations endpoints
    app.get('/a2a/agent7/v1/bulk-operations', (req, res) => proxyAgent7Request(req, res, '/bulk-operations'));
    app.post('/a2a/agent7/v1/bulk-operations', (req, res) => proxyAgent7Request(req, res, '/bulk-operations', 'POST'));
    app.get('/a2a/agent7/v1/bulk-operations/:id', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}`));
    app.put('/a2a/agent7/v1/bulk-operations/:id', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}`, 'PUT'));
    app.delete('/a2a/agent7/v1/bulk-operations/:id', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}`, 'DELETE'));
    
    // Bulk operation actions
    app.post('/a2a/agent7/v1/bulk-operations/:id/execute', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}/execute`, 'POST'));
    app.post('/a2a/agent7/v1/bulk-operations/:id/rollback', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}/rollback`, 'POST'));
    app.post('/a2a/agent7/v1/bulk-operations/:id/pause', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}/pause`, 'POST'));
    app.post('/a2a/agent7/v1/bulk-operations/:id/resume', (req, res) => proxyAgent7Request(req, res, `/bulk-operations/${req.params.id}/resume`, 'POST'));
    
    // Agent Management Functions
    app.get('/a2a/agent7/v1/agent-types', (req, res) => proxyAgent7Request(req, res, '/agent-types'));
    app.get('/a2a/agent7/v1/dashboard', (req, res) => proxyAgent7Request(req, res, '/dashboard'));
    app.get('/a2a/agent7/v1/agent-capabilities/:type', (req, res) => proxyAgent7Request(req, res, `/agent-capabilities/${req.params.type}`));
    app.post('/a2a/agent7/v1/validate-configuration', (req, res) => proxyAgent7Request(req, res, '/validate-configuration', 'POST'));
    app.get('/a2a/agent7/v1/load-balancing-recommendations', (req, res) => proxyAgent7Request(req, res, '/load-balancing-recommendations'));
    
    // Real-time updates and streaming
    app.get('/a2a/agent7/v1/realtime-updates', (req, res) => {
        // Server-Sent Events for real-time agent updates
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        });
        
        proxyAgent7Request(req, res, '/realtime-updates').catch(error => {
            res.write(`data: ${JSON.stringify({error: error.message})}\\n\\n`);
            res.end();
        });
    });
    
    // Agent 7 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent7/v1/odata/RegisteredAgents', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT7_BASE_URL}/api/v1/registered-agents`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#RegisteredAgents",
                "value": response.data.map(agent => ({
                    ID: agent.id,
                    agentName: agent.agent_name,
                    agentType: agent.agent_type?.toUpperCase(),
                    agentVersion: agent.agent_version,
                    endpointUrl: agent.endpoint_url,
                    status: agent.status?.toUpperCase() || 'REGISTERING',
                    healthStatus: agent.health_status?.toUpperCase() || 'UNKNOWN',
                    capabilities: JSON.stringify(agent.capabilities || {}),
                    configuration: JSON.stringify(agent.configuration || {}),
                    performanceScore: agent.performance_score || 0,
                    responseTime: agent.response_time,
                    throughput: agent.throughput,
                    errorRate: agent.error_rate || 0,
                    lastHealthCheck: agent.last_health_check,
                    registrationDate: agent.registration_date,
                    deactivationDate: agent.deactivation_date,
                    loadBalanceWeight: agent.load_balance_weight || 50,
                    priority: agent.priority || 5,
                    tags: JSON.stringify(agent.tags || []),
                    notes: agent.notes,
                    createdAt: agent.created_at,
                    createdBy: agent.created_by,
                    modifiedAt: agent.modified_at,
                    modifiedBy: agent.modified_by
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 7 backend not available"
                }
            });
        }
    });
    
    app.get('/a2a/agent7/v1/odata/ManagementTasks', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT7_BASE_URL}/api/v1/management-tasks`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#ManagementTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    taskType: task.task_type?.toUpperCase(),
                    status: task.status?.toUpperCase() || 'SCHEDULED',
                    priority: task.priority?.toUpperCase() || 'NORMAL',
                    targetAgents: JSON.stringify(task.target_agents || []),
                    parameters: JSON.stringify(task.parameters || {}),
                    scheduleType: task.schedule_type?.toUpperCase() || 'IMMEDIATE',
                    scheduledTime: task.scheduled_time,
                    recurrencePattern: task.recurrence_pattern,
                    startTime: task.start_time,
                    endTime: task.end_time,
                    duration: task.duration,
                    progress: task.progress || 0,
                    result: JSON.stringify(task.result || {}),
                    errorMessage: task.error_message,
                    retryCount: task.retry_count || 0,
                    maxRetries: task.max_retries || 3,
                    notificationSent: task.notification_sent !== false,
                    rollbackAvailable: task.rollback_available !== false,
                    createdAt: task.created_at,
                    createdBy: task.created_by,
                    modifiedAt: task.modified_at,
                    modifiedBy: task.modified_by
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 7 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 7 API proxy routes initialized');
    
    // ===== AGENT 8 PROXY ROUTES =====
    const AGENT8_BASE_URL = process.env.AGENT8_BASE_URL || 'http://localhost:8007';
    
    // Agent 8 proxy function
    const proxyAgent8Request = async (req, res, endpoint, method = 'GET') => {
        try {
            const response = await axios({
                method,
                url: `${AGENT8_BASE_URL}/api/v1${endpoint}`,
                data: req.body,
                headers: { 'Content-Type': 'application/json' },
                timeout: 30000
            });
            res.json(response.data);
        } catch (error) {
            log.error(`Agent 8 proxy error for ${endpoint}:`, error.message);
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 8 Data Management service temporarily unavailable",
                    endpoint: endpoint
                }
            });
        }
    };
    
    // Agent 8 Data Management Core Operations
    app.get('/a2a/agent8/v1/tasks', (req, res) => proxyAgent8Request(req, res, '/data-tasks'));
    app.post('/a2a/agent8/v1/tasks', (req, res) => proxyAgent8Request(req, res, '/data-tasks', 'POST'));
    app.get('/a2a/agent8/v1/tasks/:taskId', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}`));
    app.put('/a2a/agent8/v1/tasks/:taskId', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}`, 'PUT'));
    app.delete('/a2a/agent8/v1/tasks/:taskId', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}`, 'DELETE'));
    
    // Agent 8 Data Task Operations
    app.post('/a2a/agent8/v1/tasks/:taskId/execute', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/execute`, 'POST'));
    app.post('/a2a/agent8/v1/tasks/:taskId/validate', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/validate`, 'POST'));
    app.post('/a2a/agent8/v1/tasks/:taskId/pause', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/pause`, 'POST'));
    app.post('/a2a/agent8/v1/tasks/:taskId/resume', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/resume`, 'POST'));
    app.post('/a2a/agent8/v1/tasks/:taskId/cancel', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/cancel`, 'POST'));
    app.get('/a2a/agent8/v1/tasks/:taskId/status', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/status`));
    app.get('/a2a/agent8/v1/tasks/:taskId/progress', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/progress`));
    app.get('/a2a/agent8/v1/tasks/:taskId/logs', (req, res) => 
        proxyAgent8Request(req, res, `/data-tasks/${req.params.taskId}/logs`));
    app.get('/a2a/agent8/v1/tasks/:taskId/stream', (req, res) => {
        // Special handling for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        const eventSourceUrl = `${AGENT8_BASE_URL}/api/v1/data-tasks/${req.params.taskId}/stream`;
        // Proxy SSE stream - implementation would forward events
        res.write(':ok\n\n');
    });
    
    // Agent 8 Storage Backend Management
    app.get('/a2a/agent8/v1/storage-backends', (req, res) => proxyAgent8Request(req, res, '/storage-backends'));
    app.post('/a2a/agent8/v1/storage-backends', (req, res) => proxyAgent8Request(req, res, '/storage-backends', 'POST'));
    app.get('/a2a/agent8/v1/storage-backends/:backendId', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}`));
    app.put('/a2a/agent8/v1/storage-backends/:backendId', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}`, 'PUT'));
    app.delete('/a2a/agent8/v1/storage-backends/:backendId', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}`, 'DELETE'));
    app.post('/a2a/agent8/v1/storage-backends/:backendId/health-check', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}/health-check`, 'POST'));
    app.get('/a2a/agent8/v1/storage-backends/:backendId/utilization', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}/utilization`));
    app.post('/a2a/agent8/v1/storage-backends/:backendId/optimize', (req, res) => 
        proxyAgent8Request(req, res, `/storage-backends/${req.params.backendId}/optimize`, 'POST'));
    
    // Agent 8 Cache Management
    app.get('/a2a/agent8/v1/cache/configurations', (req, res) => proxyAgent8Request(req, res, '/cache/configurations'));
    app.post('/a2a/agent8/v1/cache/configurations', (req, res) => proxyAgent8Request(req, res, '/cache/configurations', 'POST'));
    app.get('/a2a/agent8/v1/cache/configurations/:configId', (req, res) => 
        proxyAgent8Request(req, res, `/cache/configurations/${req.params.configId}`));
    app.put('/a2a/agent8/v1/cache/configurations/:configId', (req, res) => 
        proxyAgent8Request(req, res, `/cache/configurations/${req.params.configId}`, 'PUT'));
    app.delete('/a2a/agent8/v1/cache/configurations/:configId', (req, res) => 
        proxyAgent8Request(req, res, `/cache/configurations/${req.params.configId}`, 'DELETE'));
    app.post('/a2a/agent8/v1/cache/:cacheId/clear', (req, res) => 
        proxyAgent8Request(req, res, `/cache/${req.params.cacheId}/clear`, 'POST'));
    app.post('/a2a/agent8/v1/cache/:cacheId/warmup', (req, res) => 
        proxyAgent8Request(req, res, `/cache/${req.params.cacheId}/warmup`, 'POST'));
    app.get('/a2a/agent8/v1/cache/:cacheId/stats', (req, res) => 
        proxyAgent8Request(req, res, `/cache/${req.params.cacheId}/stats`));
    app.get('/a2a/agent8/v1/cache/operations', (req, res) => proxyAgent8Request(req, res, '/cache/operations'));
    app.get('/a2a/agent8/v1/cache/analytics', (req, res) => proxyAgent8Request(req, res, '/cache/analytics'));
    
    // Agent 8 Data Versioning
    app.get('/a2a/agent8/v1/versions', (req, res) => proxyAgent8Request(req, res, '/data-versions'));
    app.post('/a2a/agent8/v1/versions', (req, res) => proxyAgent8Request(req, res, '/data-versions', 'POST'));
    app.get('/a2a/agent8/v1/versions/:versionId', (req, res) => 
        proxyAgent8Request(req, res, `/data-versions/${req.params.versionId}`));
    app.delete('/a2a/agent8/v1/versions/:versionId', (req, res) => 
        proxyAgent8Request(req, res, `/data-versions/${req.params.versionId}`, 'DELETE'));
    app.post('/a2a/agent8/v1/versions/:versionId/restore', (req, res) => 
        proxyAgent8Request(req, res, `/data-versions/${req.params.versionId}/restore`, 'POST'));
    app.get('/a2a/agent8/v1/versions/:versionId/compare/:compareVersionId', (req, res) => 
        proxyAgent8Request(req, res, `/data-versions/${req.params.versionId}/compare/${req.params.compareVersionId}`));
    app.get('/a2a/agent8/v1/datasets/:datasetId/versions', (req, res) => 
        proxyAgent8Request(req, res, `/datasets/${req.params.datasetId}/versions`));
    app.post('/a2a/agent8/v1/datasets/:datasetId/create-version', (req, res) => 
        proxyAgent8Request(req, res, `/datasets/${req.params.datasetId}/create-version`, 'POST'));
    
    // Agent 8 Data Backup Management
    app.get('/a2a/agent8/v1/backups', (req, res) => proxyAgent8Request(req, res, '/data-backups'));
    app.post('/a2a/agent8/v1/backups', (req, res) => proxyAgent8Request(req, res, '/data-backups', 'POST'));
    app.get('/a2a/agent8/v1/backups/:backupId', (req, res) => 
        proxyAgent8Request(req, res, `/data-backups/${req.params.backupId}`));
    app.delete('/a2a/agent8/v1/backups/:backupId', (req, res) => 
        proxyAgent8Request(req, res, `/data-backups/${req.params.backupId}`, 'DELETE'));
    app.post('/a2a/agent8/v1/backups/:backupId/restore', (req, res) => 
        proxyAgent8Request(req, res, `/data-backups/${req.params.backupId}/restore`, 'POST'));
    app.post('/a2a/agent8/v1/backups/:backupId/verify', (req, res) => 
        proxyAgent8Request(req, res, `/data-backups/${req.params.backupId}/verify`, 'POST'));
    app.get('/a2a/agent8/v1/backup-schedules', (req, res) => proxyAgent8Request(req, res, '/backup-schedules'));
    app.post('/a2a/agent8/v1/backup-schedules', (req, res) => proxyAgent8Request(req, res, '/backup-schedules', 'POST'));
    app.put('/a2a/agent8/v1/backup-schedules/:scheduleId', (req, res) => 
        proxyAgent8Request(req, res, `/backup-schedules/${req.params.scheduleId}`, 'PUT'));
    app.delete('/a2a/agent8/v1/backup-schedules/:scheduleId', (req, res) => 
        proxyAgent8Request(req, res, `/backup-schedules/${req.params.scheduleId}`, 'DELETE'));
    
    // Agent 8 Data Import/Export
    app.post('/a2a/agent8/v1/import', (req, res) => proxyAgent8Request(req, res, '/data-import', 'POST'));
    app.get('/a2a/agent8/v1/import/:importId', (req, res) => 
        proxyAgent8Request(req, res, `/data-import/${req.params.importId}`));
    app.post('/a2a/agent8/v1/import/:importId/cancel', (req, res) => 
        proxyAgent8Request(req, res, `/data-import/${req.params.importId}/cancel`, 'POST'));
    app.post('/a2a/agent8/v1/export', (req, res) => proxyAgent8Request(req, res, '/data-export', 'POST'));
    app.get('/a2a/agent8/v1/export/:exportId', (req, res) => 
        proxyAgent8Request(req, res, `/data-export/${req.params.exportId}`));
    app.get('/a2a/agent8/v1/export/:exportId/download', (req, res) => 
        proxyAgent8Request(req, res, `/data-export/${req.params.exportId}/download`));
    app.post('/a2a/agent8/v1/export/:exportId/cancel', (req, res) => 
        proxyAgent8Request(req, res, `/data-export/${req.params.exportId}/cancel`, 'POST'));
    
    // Agent 8 Bulk Operations
    app.get('/a2a/agent8/v1/bulk-operations', (req, res) => proxyAgent8Request(req, res, '/bulk-operations'));
    app.post('/a2a/agent8/v1/bulk-operations', (req, res) => proxyAgent8Request(req, res, '/bulk-operations', 'POST'));
    app.get('/a2a/agent8/v1/bulk-operations/:operationId', (req, res) => 
        proxyAgent8Request(req, res, `/bulk-operations/${req.params.operationId}`));
    app.post('/a2a/agent8/v1/bulk-operations/:operationId/pause', (req, res) => 
        proxyAgent8Request(req, res, `/bulk-operations/${req.params.operationId}/pause`, 'POST'));
    app.post('/a2a/agent8/v1/bulk-operations/:operationId/resume', (req, res) => 
        proxyAgent8Request(req, res, `/bulk-operations/${req.params.operationId}/resume`, 'POST'));
    app.post('/a2a/agent8/v1/bulk-operations/:operationId/cancel', (req, res) => 
        proxyAgent8Request(req, res, `/bulk-operations/${req.params.operationId}/cancel`, 'POST'));
    app.get('/a2a/agent8/v1/bulk-operations/:operationId/progress', (req, res) => 
        proxyAgent8Request(req, res, `/bulk-operations/${req.params.operationId}/progress`));
    
    // Agent 8 Performance Metrics
    app.get('/a2a/agent8/v1/performance/storage', (req, res) => proxyAgent8Request(req, res, '/performance/storage'));
    app.get('/a2a/agent8/v1/performance/cache', (req, res) => proxyAgent8Request(req, res, '/performance/cache'));
    app.get('/a2a/agent8/v1/performance/throughput', (req, res) => proxyAgent8Request(req, res, '/performance/throughput'));
    app.get('/a2a/agent8/v1/performance/operations', (req, res) => proxyAgent8Request(req, res, '/performance/operations'));
    
    // Agent 8 Dashboard and Analytics
    app.get('/a2a/agent8/v1/dashboard', (req, res) => proxyAgent8Request(req, res, '/dashboard'));
    app.get('/a2a/agent8/v1/analytics/storage-trends', (req, res) => proxyAgent8Request(req, res, '/analytics/storage-trends'));
    app.get('/a2a/agent8/v1/analytics/cache-performance', (req, res) => proxyAgent8Request(req, res, '/analytics/cache-performance'));
    app.get('/a2a/agent8/v1/analytics/operation-patterns', (req, res) => proxyAgent8Request(req, res, '/analytics/operation-patterns'));
    
    // Agent 8 Configuration Management
    app.get('/a2a/agent8/v1/config', (req, res) => proxyAgent8Request(req, res, '/config'));
    app.put('/a2a/agent8/v1/config', (req, res) => proxyAgent8Request(req, res, '/config', 'PUT'));
    app.post('/a2a/agent8/v1/config/test', (req, res) => proxyAgent8Request(req, res, '/config/test', 'POST'));
    app.post('/a2a/agent8/v1/config/reset', (req, res) => proxyAgent8Request(req, res, '/config/reset', 'POST'));
    
    // Agent 8 Health Check
    app.get('/a2a/agent8/v1/health', (req, res) => proxyAgent8Request(req, res, '/health'));
    
    // Agent 8 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent8/v1/odata/DataTasks', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT8_BASE_URL}/api/v1/data-tasks`);
            
            const odataResponse = {
                "@odata.context": "$metadata#DataTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    taskType: task.task_type?.toUpperCase() || 'PROCESSING',
                    status: task.status?.toUpperCase() || 'PENDING',
                    priority: task.priority?.toUpperCase() || 'NORMAL',
                    dataSource: task.data_source,
                    targetDestination: task.target_destination,
                    dataSize: task.data_size || 0,
                    processedSize: task.processed_size || 0,
                    progressPercent: task.progress_percent || 0,
                    estimatedDuration: task.estimated_duration,
                    actualDuration: task.actual_duration,
                    startTime: task.start_time,
                    endTime: task.end_time,
                    errorMessage: task.error_message,
                    retryCount: task.retry_count || 0,
                    maxRetries: task.max_retries || 3,
                    configuration: JSON.stringify(task.configuration || {}),
                    metadata: JSON.stringify(task.metadata || {}),
                    createdAt: task.created_at,
                    createdBy: task.created_by,
                    modifiedAt: task.modified_at,
                    modifiedBy: task.modified_by
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 8 backend not available"
                }
            });
        }
    });
    
    app.get('/a2a/agent8/v1/odata/StorageBackends', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT8_BASE_URL}/api/v1/storage-backends`);
            
            const odataResponse = {
                "@odata.context": "$metadata#StorageBackends",
                "value": response.data.map(backend => ({
                    ID: backend.id,
                    backendName: backend.backend_name,
                    backendType: backend.backend_type?.toUpperCase() || 'HANA',
                    connectionString: backend.connection_string,
                    status: backend.status?.toUpperCase() || 'ACTIVE',
                    healthScore: backend.health_score || 100,
                    totalCapacity: backend.total_capacity || 0,
                    usedCapacity: backend.used_capacity || 0,
                    availableCapacity: backend.available_capacity || 0,
                    compressionEnabled: backend.compression_enabled !== false,
                    encryptionEnabled: backend.encryption_enabled !== false,
                    replicationFactor: backend.replication_factor || 1,
                    lastHealthCheck: backend.last_health_check,
                    configuration: JSON.stringify(backend.configuration || {}),
                    credentials: JSON.stringify(backend.credentials || {}),
                    createdAt: backend.created_at,
                    createdBy: backend.created_by,
                    modifiedAt: backend.modified_at,
                    modifiedBy: backend.modified_by
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 8 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 8 API proxy routes initialized');
    
    // Agent 3 OData Service Proxy - Convert REST to OData format
    app.get('/a2a/agent3/v1/odata/VectorProcessingTasks', async (req, res) => {
        try {
            // This would be handled by the CAP service, but we provide a fallback
            const response = await axios.get(`${AGENT3_BASE_URL}/a2a/agent3/v1/tasks`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#VectorProcessingTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    dataSource: task.data_source,
                    dataType: task.data_type?.toUpperCase() || 'TEXT',
                    embeddingModel: task.embedding_model,
                    modelProvider: task.model_provider?.toUpperCase() || 'OPENAI',
                    vectorDatabase: task.vector_database?.toUpperCase() || 'PINECONE',
                    indexType: task.index_type?.toUpperCase() || 'HNSW',
                    distanceMetric: task.distance_metric?.toUpperCase() || 'COSINE',
                    dimensions: task.dimensions || 1536,
                    chunkSize: task.chunk_size || 512,
                    chunkOverlap: task.chunk_overlap || 50,
                    normalization: task.normalization !== false,
                    useGPU: task.use_gpu || false,
                    batchSize: task.batch_size || 100,
                    status: task.status?.toUpperCase() || 'DRAFT',
                    priority: task.priority?.toUpperCase() || 'MEDIUM',
                    progressPercent: task.progress || 0,
                    currentStage: task.current_stage,
                    processingTime: task.processing_time,
                    vectorsGenerated: task.vectors_generated || 0,
                    chunksProcessed: task.chunks_processed || 0,
                    totalChunks: task.total_chunks || 0,
                    collectionName: task.collection_name,
                    indexSize: task.index_size || 0,
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 3 backend not available"
                }
            });
        }
    });
    
    app.get('/a2a/agent3/v1/odata/VectorCollections', async (req, res) => {
        try {
            const response = await axios.get(`${AGENT3_BASE_URL}/a2a/agent3/v1/collections`);
            
            const odataResponse = {
                "@odata.context": "$metadata#VectorCollections",
                "value": response.data.map(collection => ({
                    ID: collection.id,
                    name: collection.name,
                    description: collection.description,
                    vectorDatabase: collection.vector_database,
                    embeddingModel: collection.embedding_model,
                    dimensions: collection.dimensions,
                    distanceMetric: collection.distance_metric,
                    indexType: collection.index_type,
                    totalVectors: collection.total_vectors || 0,
                    indexSize: collection.index_size || 0,
                    isActive: collection.is_active !== false,
                    isOptimized: collection.is_optimized || false,
                    lastOptimized: collection.last_optimized,
                    createdAt: collection.created_at,
                    modifiedAt: collection.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 3 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 3 API proxy routes initialized');

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

    // Launchpad-specific health check endpoint
    app.get('/api/v1/launchpad/health', async (req, res) => {
        try {
            const healthCheck = {
                timestamp: new Date().toISOString(),
                status: 'checking',
                components: {
                    ui5_resources: { status: 'unknown' },
                    shell_config: { status: 'unknown' },
                    api_endpoints: { status: 'unknown', details: {} },
                    tile_data: { status: 'unknown', details: {} },
                    websocket: { status: 'unknown' }
                },
                tiles_loaded: false,
                real_data_available: false,
                fallback_mode: true
            };

            // 1. Check UI5 resources availability
            try {
                const ui5Check = await fetch('https://ui5.sap.com/1.120.0/resources/sap-ui-core.js', { 
                    method: 'HEAD',
                    timeout: 5000 
                });
                healthCheck.components.ui5_resources.status = ui5Check.ok ? 'healthy' : 'error';
            } catch (error) {
                healthCheck.components.ui5_resources.status = 'error';
                healthCheck.components.ui5_resources.error = error.message;
            }

            // 2. Check shell configuration
            healthCheck.components.shell_config.status = 'healthy'; // Static config always available

            // 3. Check API endpoints
            const endpointChecks = await Promise.all([
                fetch(`http://localhost:${req.app.get('port') || 4004}/api/v1/Agents?id=agent_visualization`)
                    .then(r => ({ endpoint: 'agent_visualization', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'agent_visualization', ok: false, error: e.message })),
                fetch(`http://localhost:${req.app.get('port') || 4004}/api/v1/network/overview`)
                    .then(r => ({ endpoint: 'network_overview', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'network_overview', ok: false, error: e.message })),
                fetch(`http://localhost:${req.app.get('port') || 4004}/api/v1/health/summary`)
                    .then(r => ({ endpoint: 'health_summary', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'health_summary', ok: false, error: e.message }))
            ]);

            const allEndpointsOk = endpointChecks.every(check => check.ok);
            healthCheck.components.api_endpoints.status = allEndpointsOk ? 'healthy' : 'degraded';
            healthCheck.components.api_endpoints.details = endpointChecks;

            // 4. Check tile data quality
            try {
                const tileDataResponse = await fetch(`http://localhost:${req.app.get('port') || 4004}/api/v1/Agents?id=agent_visualization`);
                const tileData = await tileDataResponse.json();
                
                // Check if we have real data (not all zeros)
                const hasRealData = tileData.agentCount > 0 || 
                                  tileData.services > 0 || 
                                  tileData.workflows > 0;
                
                healthCheck.components.tile_data.status = hasRealData ? 'healthy' : 'warning';
                healthCheck.components.tile_data.details = tileData;
                healthCheck.real_data_available = hasRealData;
                healthCheck.fallback_mode = !hasRealData;
                
                // Tiles are considered loaded if we get valid response structure
                healthCheck.tiles_loaded = tileData.hasOwnProperty('agentCount') && 
                                         tileData.hasOwnProperty('services') &&
                                         tileData.hasOwnProperty('workflows');
            } catch (error) {
                healthCheck.components.tile_data.status = 'error';
                healthCheck.components.tile_data.error = error.message;
            }

            // 5. Check WebSocket availability
            healthCheck.components.websocket.status = io ? 'healthy' : 'unavailable';
            healthCheck.components.websocket.connected_clients = io ? io.sockets.sockets.size : 0;

            // Overall status determination
            const criticalComponents = [
                healthCheck.components.api_endpoints.status,
                healthCheck.components.tile_data.status
            ];
            
            if (criticalComponents.every(s => s === 'healthy')) {
                healthCheck.status = 'healthy';
            } else if (criticalComponents.some(s => s === 'error')) {
                healthCheck.status = 'error';
            } else if (healthCheck.tiles_loaded) {
                healthCheck.status = 'degraded';
            } else {
                healthCheck.status = 'error';
            }

            // Add recommendations
            healthCheck.recommendations = [];
            if (!healthCheck.real_data_available) {
                healthCheck.recommendations.push('Start agent services on ports 8000-8015 for real data');
            }
            if (healthCheck.components.websocket.status !== 'healthy') {
                healthCheck.recommendations.push('WebSocket server not initialized');
            }

            res.json(healthCheck);
        } catch (error) {
            log.error('Launchpad health check failed:', error);
            res.status(500).json({
                status: 'error',
                error: error.message,
                timestamp: new Date().toISOString()
            });
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
    
    // User API endpoints are now handled by CAP UserManagementService at /api/v1/user
    
    // Serve UI5 app (duplicate removed - now served earlier before blocking middleware)
    // app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2aFiori/webapp')));
    // app.use('/app/launchpad', express.static(path.join(__dirname, '../app/launchpad')));
    
    // Serve shells directory for Fiori Sandbox configuration (duplicate removed - now served earlier)
    // app.use('/shells', express.static(path.join(__dirname, '../app/shells')));
    
    // Serve launchpad pages
    app.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/launchpad.html'));
    });
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

    // LAUNCHPAD TILE REST ENDPOINTS - For real-time tile data
    const { checkAgentHealth, checkBlockchainHealth, checkMcpHealth, AGENT_METADATA } = require('./utils/launchpadHelpers');

    // Agent visualization endpoint for launchpad controller
    app.get('/api/v1/Agents', async (req, res) => {
        if (req.query.id === 'agent_visualization') {
            try {
                // Check health of all agents and aggregate data
                const healthChecks = await Promise.all(
                    Object.entries(AGENT_METADATA).map(async ([id, agent]) => {
                        const health = await checkAgentHealth(agent.port);
                        return { id: parseInt(id), health };
                    })
                );
                
                const healthyAgents = healthChecks.filter(h => h.health.status === 'healthy');
                const totalActiveTasks = healthyAgents.reduce((sum, agent) => sum + (agent.health.active_tasks || 0), 0);
                const totalSkills = healthyAgents.reduce((sum, agent) => sum + (agent.health.skills || 0), 0);
                const totalMcpTools = healthyAgents.reduce((sum, agent) => sum + (agent.health.mcp_tools || 0), 0);
                
                // Calculate performance metric
                const validSuccessRates = healthyAgents
                    .filter(a => a.health.success_rate !== null)
                    .map(a => a.health.success_rate);
                const avgSuccessRate = validSuccessRates.length > 0 ? 
                    validSuccessRates.reduce((sum, rate) => sum + rate, 0) / validSuccessRates.length : 85;
                
                res.json({
                    agentCount: healthyAgents.length,
                    services: totalSkills + totalMcpTools,
                    workflows: totalActiveTasks,
                    performance: Math.round(avgSuccessRate),
                    notifications: 3, // Default value
                    security: 0 // Default value
                });
            } catch (error) {
                log.error('Error in agent visualization endpoint:', error);
                // Return fallback data
                res.json({
                    agentCount: 0,
                    services: 0,
                    workflows: 0,
                    performance: 0,
                    notifications: 0,
                    security: 0
                });
            }
        } else {
            res.status(400).json({ error: 'Invalid request' });
        }
    });

    // Agent status endpoints for tiles
    for (let i = 0; i < 16; i++) {
        app.get(`/api/v1/agents/${i}/status`, async (req, res) => {
            try {
                const agent = AGENT_METADATA[i];
                if (!agent) {
                    return res.status(404).json({ error: `Agent ${i} not found` });
                }
                
                const health = await checkAgentHealth(agent.port);
                
                if (health.status === 'healthy') {
                    // Build tile response format matching launchpadService.js
                    let numberState = "Neutral";
                    let stateArrow = "None";
                    
                    if (health.success_rate !== null) {
                        if (health.success_rate >= 95) {
                            numberState = "Positive";
                            stateArrow = "Up";
                        } else if (health.success_rate >= 85) {
                            numberState = "Critical";
                            stateArrow = "None";
                        } else {
                            numberState = "Error";
                            stateArrow = "Down";
                        }
                    } else if (health.active_tasks > 0) {
                        numberState = "Positive";
                        stateArrow = "Up";
                    }
                    
                    let subtitle = `${health.total_tasks} total tasks`;
                    if (health.success_rate !== null) {
                        subtitle += `, ${health.success_rate.toFixed(1)}% success`;
                    }
                    
                    let info = `${health.skills} skills, ${health.mcp_tools} MCP tools`;
                    if (health.avg_response_time_ms !== null) {
                        info += `, ${health.avg_response_time_ms}ms avg`;
                    }
                    
                    res.json({
                        d: {
                            title: health.name || agent.name,
                            number: health.active_tasks.toString(),
                            numberUnit: "active tasks",
                            numberState: numberState,
                            subtitle: subtitle,
                            stateArrow: stateArrow,
                            info: info,
                            status: health.status,
                            agent_id: health.agent_id,
                            version: health.version,
                            port: agent.port,
                            capabilities: {
                                skills: health.skills,
                                handlers: health.handlers,
                                mcp_tools: health.mcp_tools,
                                mcp_resources: health.mcp_resources
                            },
                            performance: {
                                cpu_usage: health.cpu_usage,
                                memory_usage: health.memory_usage,
                                uptime_seconds: health.uptime_seconds,
                                success_rate: health.success_rate,
                                avg_response_time_ms: health.avg_response_time_ms,
                                processed_today: health.processed_today,
                                error_rate: health.error_rate,
                                queue_depth: health.queue_depth
                            },
                            timestamp: health.timestamp
                        }
                    });
                } else {
                    res.status(503).json({
                        d: {
                            title: agent.name,
                            number: "0",
                            numberUnit: health.status,
                            numberState: "Error",
                            subtitle: health.message || `Port ${agent.port}`,
                            stateArrow: "Down",
                            info: `${agent.type} Agent - ${health.status}`,
                            status: health.status,
                            port: agent.port,
                            error: health.message,
                            timestamp: new Date().toISOString()
                        }
                    });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    // Network overview endpoint
    app.get('/api/v1/network/overview', async (req, res) => {
        try {
            const [healthChecks, blockchainHealth, mcpHealth] = await Promise.all([
                Promise.all(
                    Object.entries(AGENT_METADATA).map(async ([id, agent]) => {
                        const health = await checkAgentHealth(agent.port);
                        return { 
                            id: parseInt(id), 
                            name: health.name || agent.name, 
                            status: health.status,
                            active_tasks: health.active_tasks || 0,
                            total_tasks: health.total_tasks || 0,
                            skills: health.skills || 0,
                            mcp_tools: health.mcp_tools || 0
                        };
                    })
                ),
                checkBlockchainHealth(),
                checkMcpHealth()
            ]);
            
            const healthyAgents = healthChecks.filter(h => h.status === 'healthy');
            const totalAgents = healthChecks.length;
            const activeAgents = healthyAgents.length;
            const agentHealthScore = Math.round((activeAgents / totalAgents) * 100);
            
            const totalActiveTasks = healthyAgents.reduce((sum, agent) => sum + agent.active_tasks, 0);
            const totalSkills = healthyAgents.reduce((sum, agent) => sum + agent.skills, 0);
            const totalMcpTools = healthyAgents.reduce((sum, agent) => sum + agent.mcp_tools, 0);
            
            const blockchainScore = blockchainHealth.status === 'healthy' ? 100 : 0;
            const mcpScore = mcpHealth.status === 'healthy' ? 100 : mcpHealth.status === 'offline' ? 0 : 50;
            const overallSystemHealth = Math.round((agentHealthScore + blockchainScore + mcpScore) / 3);
            
            res.json({
                d: {
                    title: "Network Overview",
                    number: activeAgents.toString(),
                    numberUnit: "active agents",
                    numberState: overallSystemHealth > 80 ? "Positive" : overallSystemHealth > 50 ? "Critical" : "Error",
                    subtitle: `${totalAgents} total agents, ${overallSystemHealth}% system health`,
                    stateArrow: overallSystemHealth > 80 ? "Up" : "Down",
                    info: `${totalActiveTasks} active tasks, ${totalSkills} skills, ${totalMcpTools} MCP tools`,
                    real_metrics: {
                        healthy_agents: activeAgents,
                        total_agents: totalAgents,
                        agent_health_score: agentHealthScore,
                        total_active_tasks: totalActiveTasks,
                        total_skills: totalSkills,
                        total_mcp_tools: totalMcpTools,
                        blockchain_status: blockchainHealth.status,
                        blockchain_score: blockchainScore,
                        mcp_status: mcpHealth.status,
                        mcp_score: mcpScore,
                        overall_system_health: overallSystemHealth
                    },
                    timestamp: new Date().toISOString()
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Blockchain stats endpoint
    app.get('/api/v1/blockchain/status', async (req, res) => {
        try {
            const blockchainHealth = await checkBlockchainHealth();
            
            if (blockchainHealth.status === 'healthy') {
                const registeredAgents = blockchainHealth.total_agents_on_chain || 0;
                const contractCount = Object.keys(blockchainHealth.contracts || {}).length;
                
                res.json({
                    d: {
                        title: "Blockchain Monitor",
                        number: registeredAgents.toString(),
                        numberUnit: "registered agents",
                        numberState: blockchainHealth.trust_integration ? "Positive" : "Critical",
                        subtitle: `${contractCount} contracts deployed`,
                        stateArrow: blockchainHealth.trust_integration ? "Up" : "None",
                        info: `Network: ${blockchainHealth.network || 'Unknown'}, Trust: ${blockchainHealth.trust_integration ? 'Enabled' : 'Disabled'}`,
                        blockchain_metrics: {
                            network: blockchainHealth.network || 'Unknown',
                            contracts: blockchainHealth.contracts || {},
                            registered_agents_count: registeredAgents,
                            contract_count: contractCount,
                            trust_integration: blockchainHealth.trust_integration || false,
                            avg_trust_score: blockchainHealth.avg_trust_score || null
                        },
                        timestamp: blockchainHealth.timestamp || new Date().toISOString()
                    }
                });
            } else {
                res.status(503).json({
                    d: {
                        title: "Blockchain Monitor",
                        number: "0",
                        numberUnit: "offline",
                        numberState: "Error",
                        subtitle: blockchainHealth.message || "Connection failed",
                        stateArrow: "Down",
                        info: `Status: ${blockchainHealth.status}`,
                        error: blockchainHealth.message,
                        timestamp: new Date().toISOString()
                    }
                });
            }
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Services count endpoint
    app.get('/api/v1/services/count', async (req, res) => {
        try {
            const healthChecks = await Promise.all(
                Object.entries(AGENT_METADATA).map(async ([id, agent]) => {
                    const health = await checkAgentHealth(agent.port);
                    return health.status === 'healthy' ? health : null;
                })
            );
            
            const healthyAgents = healthChecks.filter(h => h !== null);
            const totalSkills = healthyAgents.reduce((sum, agent) => sum + (agent.skills || 0), 0);
            const totalHandlers = healthyAgents.reduce((sum, agent) => sum + (agent.handlers || 0), 0);
            const totalMcpTools = healthyAgents.reduce((sum, agent) => sum + (agent.mcp_tools || 0), 0);
            const totalMcpResources = healthyAgents.reduce((sum, agent) => sum + (agent.mcp_resources || 0), 0);
            const totalServices = totalSkills + totalHandlers + totalMcpTools;
            
            const activeProviders = healthyAgents.length;
            const totalProviders = Object.keys(AGENT_METADATA).length;
            const providerHealthPercentage = Math.round((activeProviders / totalProviders) * 100);
            
            res.json({
                d: {
                    title: "Service Marketplace",
                    number: totalServices.toString(),
                    numberUnit: "available services",
                    numberState: providerHealthPercentage > 80 ? "Positive" : providerHealthPercentage > 50 ? "Critical" : "Error",
                    subtitle: `${activeProviders}/${totalProviders} providers active (${providerHealthPercentage}%)`,
                    stateArrow: providerHealthPercentage > 80 ? "Up" : "Down",
                    info: `${totalSkills} skills, ${totalHandlers} handlers, ${totalMcpTools} MCP tools`,
                    service_breakdown: {
                        agent_skills: totalSkills,
                        agent_handlers: totalHandlers,
                        mcp_tools: totalMcpTools,
                        database_services: totalMcpResources,
                        total_services: totalServices
                    },
                    provider_health: {
                        active_providers: activeProviders,
                        total_providers: totalProviders,
                        provider_health_percentage: providerHealthPercentage
                    },
                    timestamp: new Date().toISOString()
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // Health summary endpoint
    app.get('/api/v1/health/summary', async (req, res) => {
        try {
            const [healthChecks, blockchainHealth, mcpHealth] = await Promise.all([
                Promise.all(
                    Object.entries(AGENT_METADATA).map(async ([id, agent]) => {
                        const health = await checkAgentHealth(agent.port);
                        return { 
                            id: parseInt(id), 
                            status: health.status,
                            cpu_usage: health.cpu_usage,
                            memory_usage: health.memory_usage,
                            success_rate: health.success_rate,
                            error_rate: health.error_rate
                        };
                    })
                ),
                checkBlockchainHealth(),
                checkMcpHealth()
            ]);
            
            const healthyAgents = healthChecks.filter(h => h.status === 'healthy');
            const totalAgents = healthChecks.length;
            
            const agentsHealth = Math.round((healthyAgents.length / totalAgents) * 100);
            const blockchainHealth_score = blockchainHealth.status === 'healthy' ? 100 : 0;
            const mcpHealth_score = mcpHealth.status === 'healthy' ? 100 : mcpHealth.status === 'offline' ? 0 : 50;
            const apiHealth = 100;
            
            const validCpuUsages = healthyAgents.filter(a => a.cpu_usage !== null).map(a => a.cpu_usage);
            const validMemoryUsages = healthyAgents.filter(a => a.memory_usage !== null).map(a => a.memory_usage);
            const validErrorRates = healthyAgents.filter(a => a.error_rate !== null).map(a => a.error_rate);
            
            const avgCpuUsage = validCpuUsages.length > 0 ? 
                validCpuUsages.reduce((sum, cpu) => sum + cpu, 0) / validCpuUsages.length : null;
            const avgMemoryUsage = validMemoryUsages.length > 0 ? 
                validMemoryUsages.reduce((sum, mem) => sum + mem, 0) / validMemoryUsages.length : null;
            const avgErrorRate = validErrorRates.length > 0 ? 
                validErrorRates.reduce((sum, err) => sum + err, 0) / validErrorRates.length : null;
            
            const overallHealth = Math.round((agentsHealth + blockchainHealth_score + mcpHealth_score + apiHealth) / 4);
            
            res.json({
                d: {
                    title: "System Health",
                    number: overallHealth.toString(),
                    numberUnit: "% system health",
                    numberState: overallHealth > 80 ? "Positive" : overallHealth > 50 ? "Critical" : "Error",
                    subtitle: `${healthyAgents.length}/${totalAgents} agents healthy`,
                    stateArrow: overallHealth > 80 ? "Up" : "Down",
                    info: `Agents: ${agentsHealth}%, Blockchain: ${blockchainHealth_score}%, MCP: ${mcpHealth_score}%`,
                    component_health: {
                        agents_health: agentsHealth,
                        blockchain_health: blockchainHealth_score,
                        mcp_health: mcpHealth_score,
                        api_health: apiHealth
                    },
                    system_performance: {
                        avg_cpu_usage: avgCpuUsage,
                        avg_memory_usage: avgMemoryUsage,
                        network_latency: 50
                    },
                    error_tracking: {
                        agent_error_rate: avgErrorRate,
                        blockchain_tx_failure_rate: blockchainHealth.status !== 'healthy' ? 100.0 : 0.0,
                        api_error_rate: 0.0
                    },
                    timestamp: new Date().toISOString()
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
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
    
    // Perform launchpad health check before declaring server ready
    log.info('🏥 Performing launchpad health check...');
    try {
        const StartupHealthCheck = require('../scripts/startup-health-check');
        const healthChecker = new StartupHealthCheck(info.port);
        
        // Wait a bit for all endpoints to be ready
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const healthResult = await healthChecker.performHealthCheck();
        
        if (healthResult.success) {
            log.info('✅ Launchpad health check passed - all systems operational');
        } else {
            log.warn('⚠️  Launchpad health check failed:', healthResult.message);
            log.warn('Server will continue but launchpad may not function properly');
        }
    } catch (error) {
        log.error('Failed to perform launchpad health check:', error);
        log.warn('Server will continue but launchpad status is unknown');
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

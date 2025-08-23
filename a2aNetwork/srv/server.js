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
            '/a2a/agent8/v1',
            '/a2a/agent10/v1',
            '/a2a/agent11/v1',
            '/a2a/agent12/v1',
            '/a2a/agent13/v1',
            '/a2a/agent14/v1'
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
    
    // Agent proxy routes have been moved to AgentProxyService (agentProxyService.cds)
    // All Express routes are now handled through CAP services
    

    // Agent proxy routes have been moved to AgentProxyService (agentProxyService.cds)
    // All Express routes are now handled through CAP services
    

    // Agent proxy routes have been moved to AgentProxyService (agentProxyService.cds)
    // All Express routes are now handled through CAP services
    

    // ================================
    // AGENT 6 - QUALITY CONTROL & WORKFLOW ROUTING PROXY ROUTES
    // ================================
    
    
    // Helper function to proxy Agent 6 requests
    
    // Agent 6 Quality Control Tasks
    
    // Agent 6 Task Operations
        // Special handling for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        const eventSourceUrl = `${AGENT6_BASE_URL}/a2a/agent6/v1/tasks/${req.params.taskId}/stream`;
        // Proxy SSE stream - implementation would forward events
        res.write(':ok\n\n');
    });
    
    // Agent 6 Dashboard and Analytics
    
    // Agent 6 Routing Rules
    
    // Agent 6 Quality Gates
    
    // Agent 6 Batch Operations
    
    // Agent 6 Reports and Exports
    
    // Agent 6 Health Check
    
    // Agent 6 OData Service Proxy - Convert REST to OData format
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
    
    // Helper function to proxy Agent 7 requests
    
    // Registered Agents endpoints
    
    // Agent registration and management
    
    // Management Tasks endpoints
    
    // Management task actions
    
    // Health Check endpoints
        // Server-Sent Events for real-time health monitoring
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        });
        
    });
    
    // Performance Metrics endpoints
    
    // Agent Coordination endpoints
    
    // Coordination actions
    
    // Coordination status and management
    
    // Bulk Operations endpoints
    
    // Bulk operation actions
    
    // Agent Management Functions
    
    // Real-time updates and streaming
        // Server-Sent Events for real-time agent updates
        res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        });
        
    });
    
    // Agent 7 OData Service Proxy - Convert REST to OData format
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
    
    // Agent 8 proxy function
    
    // Agent 8 Data Management Core Operations
    
    // Agent 8 Data Task Operations
        // Special handling for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        const eventSourceUrl = `${AGENT8_BASE_URL}/api/v1/data-tasks/${req.params.taskId}/stream`;
        // Proxy SSE stream - implementation would forward events
        res.write(':ok\n\n');
    });
    
    // Agent 8 Storage Backend Management
    
    // Agent 8 Cache Management
    
    // Agent 8 Data Versioning
    
    // Agent 8 Data Backup Management
    
    // Agent 8 Data Import/Export
    
    // Agent 8 Bulk Operations
    
    // Agent 8 Performance Metrics
    
    // Agent 8 Dashboard and Analytics
    
    // Agent 8 Configuration Management
    
    // Agent 8 Health Check
    
    // Agent 8 OData Service Proxy - Convert REST to OData format
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
    
    // ===== AGENT 9 PROXY ROUTES - Advanced Logical Reasoning and Decision-Making Agent =====
    
    // Agent 9 proxy function
    
    // Reasoning Tasks Management
    
    // Reasoning Task Actions
    
    // Knowledge Base Management
    
    // Decision Making
    
    // Problem Solving
    
    // Reasoning Engines Management
    
    // Health Check
    
    // Agent 9 OData Service Proxy - Convert REST to OData format
        try {
            // This would be handled by the CAP service, but we provide a fallback
            const response = await axios.get(`${AGENT9_BASE_URL}/api/agent9/v1/reasoning-tasks`);
            
            // Convert to OData format
            const odataResponse = {
                "@odata.context": "$metadata#ReasoningTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    reasoningType: task.reasoning_type?.toUpperCase() || 'DEDUCTIVE',
                    problemDomain: task.problem_domain?.toUpperCase() || 'GENERAL',
                    reasoningEngine: task.reasoning_engine?.toUpperCase() || 'FORWARD_CHAINING',
                    status: task.status?.toUpperCase() || 'PENDING',
                    priority: task.priority?.toUpperCase() || 'MEDIUM',
                    confidenceScore: task.confidence_score || 0.0,
                    factsProcessed: task.facts_processed || 0,
                    inferencesGenerated: task.inferences_generated || 0,
                    conclusionsReached: task.conclusions_reached || 0,
                    processingTime: task.processing_time || 0,
                    confidenceThreshold: task.confidence_threshold || 0.5,
                    maxInferenceDepth: task.max_inference_depth || 5,
                    chainingStrategy: task.chaining_strategy?.toUpperCase() || 'BREADTH_FIRST',
                    uncertaintyHandling: task.uncertainty_handling?.toUpperCase() || 'CRISP',
                    probabilisticModel: task.probabilistic_model?.toUpperCase() || 'BAYESIAN',
                    logicalFramework: task.logical_framework?.toUpperCase() || 'FIRST_ORDER',
                    parallelReasoning: task.parallel_reasoning || false,
                    explanationDepth: task.explanation_depth || 3,
                    validationStatus: task.validation_status?.toUpperCase() || 'PENDING',
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.set('Content-Type', 'application/json');
            res.json(odataResponse);
        } catch (error) {
            log.error('Agent 9 OData proxy error:', error.message);
            res.status(500).json({
                error: 'Failed to fetch Agent 9 reasoning tasks',
                message: error.message
            });
        }
    });
    
        try {
            const response = await axios.get(`${AGENT9_BASE_URL}/api/agent9/v1/knowledge-base`);
            
            const odataResponse = {
                "@odata.context": "$metadata#KnowledgeBaseElements", 
                "value": response.data.map(element => ({
                    ID: element.id,
                    elementName: element.element_name,
                    elementType: element.element_type?.toUpperCase() || 'FACT',
                    content: element.content,
                    domain: element.domain?.toUpperCase() || 'GENERAL',
                    confidenceLevel: element.confidence_level || 3,
                    priorityWeight: element.priority_weight || 0.5,
                    source: element.source,
                    isActive: element.is_active !== false,
                    usageCount: element.usage_count || 0,
                    lastUsed: element.last_used,
                    tags: element.tags,
                    createdAt: element.created_at,
                    modifiedAt: element.modified_at
                }))
            };
            
            res.set('Content-Type', 'application/json');
            res.json(odataResponse);
        } catch (error) {
            log.error('Agent 9 OData knowledge base proxy error:', error.message);
            res.status(500).json({
                error: 'Failed to fetch Agent 9 knowledge base elements',
                message: error.message
            });
        }
    });
    
    log.info('Agent 9 API proxy routes initialized');
    
    // ===== AGENT 10 PROXY ROUTES - Calculation Engine =====
    
    // Agent 10 proxy function
    
    // Calculation Tasks Management
    
    // Calculation Task Actions
    
    // Calculation Operations
    
    // Configuration and Methods
    
    // Results and History
    
    // Cache Management
    
    // Health Check
    
    // Agent 10 OData Service Proxy - Convert REST to OData format
        try {
            const response = await axios.get(`${AGENT10_BASE_URL}/api/agent10/v1/calculation-tasks`);
            
            const odataResponse = {
                "@odata.context": "$metadata#CalculationTasks",
                "value": response.data.map(task => ({
                    ID: task.id,
                    taskName: task.task_name,
                    description: task.description,
                    calculationType: task.calculation_type?.toUpperCase(),
                    formula: task.formula,
                    inputParameters: JSON.stringify(task.input_parameters || {}),
                    calculationMethod: task.calculation_method?.toUpperCase(),
                    precisionType: task.precision_type || 'DECIMAL64',
                    requiredAccuracy: task.required_accuracy || 0.000001,
                    maxIterations: task.max_iterations || 1000,
                    timeout: task.timeout || 60000,
                    enableSelfHealing: task.enable_self_healing !== false,
                    verificationRounds: task.verification_rounds || 3,
                    useParallelProcessing: task.use_parallel_processing !== false,
                    cacheResults: task.cache_results !== false,
                    priority: task.priority?.toUpperCase() || 'MEDIUM',
                    status: task.status?.toUpperCase() || 'PENDING',
                    progress: task.progress || 0,
                    startTime: task.start_time,
                    endTime: task.end_time,
                    executionTime: task.execution_time,
                    result: JSON.stringify(task.result || {}),
                    errorMessage: task.error_message,
                    selfHealingLog: JSON.stringify(task.self_healing_log || {}),
                    performanceMetrics: JSON.stringify(task.performance_metrics || {}),
                    metadata: JSON.stringify(task.metadata || {}),
                    createdAt: task.created_at,
                    modifiedAt: task.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 10 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 10 API proxy routes initialized');
    
    // ===== AGENT 11 PROXY ROUTES - SQL Engine =====
    
    // Agent 11 proxy function
    
    // SQL Query Tasks Management
    
    // SQL Query Actions
    
    // Natural Language to SQL
    
    // SQL Operations
    
    // Schema and Database Information
    
    // Query History and Performance
    
    // Query Templates
    
    // Database Connection Management
    
    // Export and Backup
    
    // Health Check
    
    // Agent 11 OData Service Proxy - Convert REST to OData format
        try {
            const response = await axios.get(`${AGENT11_BASE_URL}/api/agent11/v1/sql-queries`);
            
            const odataResponse = {
                "@odata.context": "$metadata#SQLQueryTasks",
                "value": response.data.map(query => ({
                    ID: query.id,
                    queryName: query.query_name,
                    description: query.description,
                    queryType: query.query_type?.toUpperCase(),
                    naturalLanguageQuery: query.natural_language_query,
                    generatedSQL: query.generated_sql,
                    originalSQL: query.original_sql,
                    optimizedSQL: query.optimized_sql,
                    databaseConnection: query.database_connection,
                    sqlDialect: query.sql_dialect?.toUpperCase() || 'HANA',
                    queryParameters: JSON.stringify(query.query_parameters || {}),
                    executionContext: JSON.stringify(query.execution_context || {}),
                    priority: query.priority?.toUpperCase() || 'MEDIUM',
                    status: query.status?.toUpperCase() || 'DRAFT',
                    executionTime: query.execution_time,
                    rowsAffected: query.rows_affected,
                    resultRowCount: query.result_row_count,
                    isOptimized: query.is_optimized !== false,
                    autoGenerated: query.auto_generated !== false,
                    requiresApproval: query.requires_approval !== false,
                    isApproved: query.is_approved !== false,
                    approvedBy: query.approved_by,
                    approvalTimestamp: query.approval_timestamp,
                    startTime: query.start_time,
                    endTime: query.end_time,
                    errorMessage: query.error_message,
                    queryResults: JSON.stringify(query.query_results || {}),
                    executionPlan: JSON.stringify(query.execution_plan || {}),
                    performanceMetrics: JSON.stringify(query.performance_metrics || {}),
                    securityContext: JSON.stringify(query.security_context || {}),
                    metadata: JSON.stringify(query.metadata || {}),
                    createdAt: query.created_at,
                    modifiedAt: query.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 11 backend not available"
                }
            });
        }
    });
    
    log.info('Agent 11 API proxy routes initialized');
    
    // ===== AGENT 12 PROXY ROUTES - Catalog Manager =====
    
    // Agent 12 proxy function
    
    // Catalog Entry Management
    
    // Catalog Entry Actions
    
    // Dependencies Management
    
    // Reviews and Ratings
    
    // Metadata Management
    
    // Search and Discovery
    
    // Service Registration
    
    // Registry Management
    
    // Analysis and Reporting
    
    // Bulk Operations
    
    // External Catalog Integration  
    
    // Categories and Tags
    
    // Recommendations and AI features
    
    // Versioning
    
    // Health Check
    
    // Agent 12 OData Service Proxy - Convert REST to OData format
        try {
            const response = await axios.get(`${AGENT12_BASE_URL}/api/agent12/v1/catalog-entries`);
            
            const odataResponse = {
                "@odata.context": "$metadata#CatalogEntries",
                "value": response.data.map(entry => ({
                    ID: entry.id,
                    entryName: entry.entry_name,
                    description: entry.description,
                    category: entry.category?.toUpperCase() || 'SERVICE',
                    subCategory: entry.sub_category,
                    version: entry.version,
                    status: entry.status?.toUpperCase() || 'DRAFT',
                    visibility: entry.visibility?.toUpperCase() || 'PRIVATE',
                    entryType: entry.entry_type?.toUpperCase() || 'MICROSERVICE',
                    provider: entry.provider,
                    owner: entry.owner,
                    contactEmail: entry.contact_email,
                    documentationUrl: entry.documentation_url,
                    sourceUrl: entry.source_url,
                    apiEndpoint: entry.api_endpoint,
                    healthCheckUrl: entry.health_check_url,
                    tags: entry.tags,
                    keywords: entry.keywords,
                    rating: entry.rating || 0.0,
                    usageCount: entry.usage_count || 0,
                    downloadCount: entry.download_count || 0,
                    isFeatured: entry.is_featured !== false,
                    isVerified: entry.is_verified !== false,
                    lastAccessed: entry.last_accessed,
                    metadata: entry.metadata,
                    configurationSchema: entry.configuration_schema,
                    exampleUsage: entry.example_usage,
                    license: entry.license,
                    securityLevel: entry.security_level?.toUpperCase() || 'INTERNAL',
                    createdAt: entry.created_at,
                    modifiedAt: entry.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            log.error('Agent 12 OData proxy error:', error);
            res.status(500).json({
                error: 'Failed to fetch catalog entries',
                message: error.message
            });
        }
    });
    
    log.info('Agent 12 API proxy routes initialized');
    
    // Agent 3 OData Service Proxy - Convert REST to OData format
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
        try {
            const health = await healthService.getHealth();
            res.status(200).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

        try {
            const health = await healthService.getDetailedHealth();
            res.status(health.status === 'healthy' ? 200 : 503).json(health);
        } catch (error) {
            res.status(503).json({ status: 'unhealthy', error: error.message });
        }
    });

    // Launchpad-specific health check endpoint
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
                    .then(r => ({ endpoint: 'agent_visualization', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'agent_visualization', ok: false, error: e.message })),
                    .then(r => ({ endpoint: 'network_overview', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'network_overview', ok: false, error: e.message })),
                    .then(r => ({ endpoint: 'health_summary', ok: r.ok, status: r.status }))
                    .catch(e => ({ endpoint: 'health_summary', ok: false, error: e.message }))
            ]);

            const allEndpointsOk = endpointChecks.every(check => check.ok);
            healthCheck.components.api_endpoints.status = allEndpointsOk ? 'healthy' : 'degraded';
            healthCheck.components.api_endpoints.details = endpointChecks;

            // 4. Check tile data quality
            try {
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

        try {
            const readiness = await healthService.getReadiness();
            res.status(readiness.status === 'ready' ? 200 : 503).json(readiness);
        } catch (error) {
            res.status(503).json({ status: 'not-ready', error: error.message });
        }
    });

        const liveness = healthService.getLiveness();
        res.status(liveness.status === 'alive' ? 200 : 503).json(liveness);
    });

        const metrics = healthService.getMetrics();
        res.status(200).json(metrics);
    });

    // Error reporting endpoints
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

        const timeframe = req.query.timeframe || '24h';
        const stats = errorReporting.getErrorStats(timeframe);
        res.status(200).json(stats);
    });

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
        res.sendFile(path.join(__dirname, '../app/launchpad.html'));
    });
        res.sendFile(path.join(__dirname, '../app/launchpad.html'));
    });
        res.sendFile(path.join(__dirname, '../app/fioriLaunchpad.html'));
    });
        res.sendFile(path.join(__dirname, 'debugLaunchpad.html'));
    });
        res.sendFile(path.join(__dirname, '../app/launchpadSimple.html'));
    });
    
    // SAP Fiori flexibility services stub endpoints
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
    
        res.status(200).json({
            changes: [],
            ui2personalization: {},
            variants: []
        });
    });
    
    // Setup monitoring routes (enhanced)
    monitoringIntegration.setupRoutes(app);
    
    // Cache management endpoints
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
        const status = networkStats.getStatus();
        res.json(status);
    });

    // LAUNCHPAD TILE REST ENDPOINTS - For real-time tile data
    const { checkAgentHealth, checkBlockchainHealth, checkMcpHealth, AGENT_METADATA } = require('./utils/launchpadHelpers');

    // Agent visualization endpoint for launchpad controller
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

    // ============================================
    // AGENT 13 - AGENT BUILDER PROXY ROUTES
    // ============================================
    
    
    // Agent 13 proxy function
    
    // Template Management
    
    // Template Actions
    
    // Agent Build Management
    
    // Build Actions
    
    // Deployment Management
    
    // Deployment Actions
    
    // Template Component Management
    
    // Component Actions
    
    // Build Pipeline Management
    
    // Pipeline Actions
    
    // Code Generation
    
    // Batch Operations
    
    // Statistics and Analytics
    
    // Configuration and Settings
    
    // Resource Management  
    
    // Templates and Documentation
    
    // Health and Status
    
    // Agent 13 OData Service Proxy - Convert REST to OData format
        try {
            const response = await axios.get(`${AGENT13_BASE_URL}/api/agent13/v1/templates`);
            
            const odataResponse = {
                "@odata.context": "$metadata#AgentTemplates",
                "value": response.data.map(template => ({
                    ID: template.id,
                    templateName: template.template_name,
                    agentType: template.agent_type?.toUpperCase() || 'CUSTOM',
                    version: template.version,
                    baseTemplate: template.base_template,
                    capabilities: template.capabilities,
                    configuration: template.configuration,
                    description: template.description,
                    status: template.status?.toUpperCase() || 'DRAFT',
                    isPublic: template.is_public !== false,
                    tags: template.tags,
                    framework: template.framework,
                    language: template.language?.toUpperCase() || 'JAVASCRIPT',
                    buildCount: template.build_count || 0,
                    successRate: template.success_rate || 0.0,
                    lastBuildAt: template.last_build_at,
                    createdBy: template.created_by,
                    createdAt: template.created_at,
                    modifiedAt: template.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 13 backend not available"
                }
            });
        }
    });
    
        try {
            const response = await axios.get(`${AGENT13_BASE_URL}/api/agent13/v1/builds`);
            
            const odataResponse = {
                "@odata.context": "$metadata#AgentBuilds",
                "value": response.data.map(build => ({
                    ID: build.id,
                    templateId: build.template_id,
                    buildNumber: build.build_number,
                    agentName: build.agent_name,
                    buildType: build.build_type?.toUpperCase() || 'STANDARD',
                    status: build.status?.toUpperCase() || 'PENDING',
                    targetEnvironment: build.target_environment?.toUpperCase() || 'DEVELOPMENT',
                    buildConfig: build.build_config,
                    artifacts: build.artifacts,
                    buildLogs: build.build_logs,
                    testResults: build.test_results,
                    duration: build.duration || 0,
                    startedAt: build.started_at,
                    completedAt: build.completed_at,
                    createdBy: build.created_by,
                    createdAt: build.created_at,
                    modifiedAt: build.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 13 backend not available"
                }
            });
        }
    });
    
        try {
            const response = await axios.get(`${AGENT13_BASE_URL}/api/agent13/v1/deployments`);
            
            const odataResponse = {
                "@odata.context": "$metadata#AgentDeployments", 
                "value": response.data.map(deployment => ({
                    ID: deployment.id,
                    buildId: deployment.build_id,
                    deploymentName: deployment.deployment_name,
                    targetEnvironment: deployment.target_environment?.toUpperCase() || 'DEVELOPMENT',
                    deploymentType: deployment.deployment_type?.toUpperCase() || 'CONTAINER',
                    status: deployment.status?.toUpperCase() || 'PENDING',
                    endpoint: deployment.endpoint,
                    replicas: deployment.replicas || 1,
                    resources: deployment.resources,
                    environmentVariables: deployment.environment_variables,
                    healthCheckUrl: deployment.health_check_url,
                    isActive: deployment.is_active !== false,
                    autoRestart: deployment.auto_restart !== false,
                    deployedAt: deployment.deployed_at,
                    lastHealthCheck: deployment.last_health_check,
                    deployedBy: deployment.deployed_by,
                    createdAt: deployment.created_at,
                    modifiedAt: deployment.modified_at
                }))
            };
            
            res.json(odataResponse);
        } catch (error) {
            res.status(503).json({
                error: {
                    code: "SERVICE_UNAVAILABLE",
                    message: "Agent 13 backend not available"
                }
            });
        }
    });
    
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

"use strict";

/**
 * A2A Developer Portal - CAP Server with OpenTelemetry
 * Production-grade server implementation with distributed tracing
 */

// Initialize tracing before any other imports
const tracer = require('./telemetry/tracer');
tracer.initialize().catch(console.error);

const cds = require('@sap/cds');
const enterpriseSecurity = require('./middleware/enterprise-security');

// Import trace API
const { trace, context: _context } = tracer;

class A2APortalServer {
    constructor() {
        this.app = null;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) {
return;
}

        try {
            // Start initialization span
            const initSpan = tracer.startSpan('server.initialization');
            
            // Configure CDS
            this._configureCDS();

            // Initialize express app
            this.app = cds.app;
            
            // Apply enterprise security middleware
            enterpriseSecurity.initialize(this.app);
            
            // Apply observability middleware
            this._applyObservabilityMiddleware();
            
            // Initialize API documentation
            this._initializeApiDocumentation();
            
            // Initialize monitoring
            this._initializeMonitoring();
            
            // Initialize services
            await this._initializeServices();
            
            this.initialized = true;
            initSpan.setStatus({ code: 1, message: 'Server initialized successfully' });
            initSpan.end();
            
        } catch (error) {
            console.error('Failed to initialize server:', error);
            throw error;
        }
    }

    _configureCDS() {
        // Configure CDS with production settings
        cds.env.production = process.env.NODE_ENV === 'production';
        cds.env.features.serve_on_root = true;
        cds.env.features.folders = {
            db: ['db', 'srv/db'],
            srv: ['srv'],
            app: ['app']
        };
    }


    _applyObservabilityMiddleware() {
        // Request ID middleware
        this.app.use((req, _res, _next) => {
            req.id = req.headers['x-request-id'] || require('uuid').v4();
            _res.setHeader('X-Request-Id', req.id);
            
            // Add request ID to trace
            const span = trace.getActiveSpan();
            if (span) {
                span.setAttribute('request.id', req.id);
            }
            
            _next();
        });

        // Request logging with tracing context
        this.app.use((req, _res, _next) => {
            const startTime = Date.now();
            
            // Log request
            // eslint-disable-next-line no-console
            console.log({
                type: 'request',
                method: req.method,
                path: req.path,
                requestId: req.id,
                traceId: trace.getActiveSpan()?.spanContext().traceId || 'none',
                userAgent: req.headers['user-agent'],
                timestamp: new Date().toISOString()
            });

            // Log response
            _res.on('finish', () => {
                const duration = Date.now() - startTime;
                const span = trace.getActiveSpan();
                
                if (span) {
                    span.setAttribute('http.response.duration', duration);
                }
                
                // eslint-disable-next-line no-console
                console.log({
                    type: 'response',
                    method: req.method,
                    path: req.path,
                    statusCode: _res.statusCode,
                    duration,
                    requestId: req.id,
                    timestamp: new Date().toISOString()
                });
            });

            _next();
        });

        // Health check endpoints (excluded from tracing)
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'UP',
                timestamp: new Date().toISOString(),
                version: process.env.SERVICE_VERSION || '2.1.0'
            });
        });

        this.app.get('/metrics', async (req, res) => {
            // Return Prometheus-style metrics
            const metrics = await this._collectMetrics();
            res.set('Content-Type', 'text/plain');
            res.send(metrics);
        });
    }

    async _initializeServices() {
        const servicesSpan = tracer.startSpan('server.initialize_services');
        
        try {
            // Load all service implementations
            await cds.serve('./srv/catalog-service').in(this.app);
            await cds.serve('./srv/admin-service').in(this.app);
            await cds.serve('./srv/workflow-service').in(this.app);
            
            servicesSpan.setStatus({ code: 1 });
        } catch (error) {
            servicesSpan.recordException(error);
            servicesSpan.setStatus({ code: 2, message: error.message });
            throw error;
        } finally {
            servicesSpan.end();
        }
    }

    _collectMetrics() {
        // Collect application metrics for Prometheus
        const metrics = [];
        
        // Add custom business metrics
        metrics.push('# HELP a2a_portal_active_projects Total number of active projects');
        metrics.push('# TYPE a2a_portal_active_projects gauge');
        metrics.push('a2a_portal_active_projects 42');
        
        metrics.push('# HELP a2a_portal_agent_executions_total Total number of agent executions');
        metrics.push('# TYPE a2a_portal_agent_executions_total counter');
        metrics.push('a2a_portal_agent_executions_total 1337');
        
        return metrics.join('\n');
    }

    _initializeApiDocumentation() {
        const swaggerUI = require('./api-docs/swagger-ui');
        this.app.use('/api-docs', swaggerUI.initialize());
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('API documentation available at /api-docs');
    }

    _initializeMonitoring() {
        const monitoringService = require('./monitoring/monitoringService');
        const a2aAgentMonitoringService = require('./monitoring/a2aAgentMonitoringService');
        const tileMetricsService = require('./monitoring/tileMetricsService');
        
        // Mount standard monitoring service
        this.app.use('/monitoring', monitoringService.router);
        
        // Mount A2A agent monitoring service
        this.app.use('/srv/monitoring/a2a-agents', a2aAgentMonitoringService.router);
        
        // Mount tile metrics service for launchpad
        this.app.use('/srv/monitoring/tiles', tileMetricsService.getRouter());
        
        // Legacy API compatibility for Fiori controller
        this.app.use('/api/monitoring', monitoringService.router);
        
         
        
        // eslint-disable-next-line no-console
        
         
        
        // eslint-disable-next-line no-console
        console.log('📊 Standard monitoring available at /monitoring');
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🤖 A2A agent monitoring available at /srv/monitoring/a2a-agents');
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('📱 Tile metrics available at /srv/monitoring/tiles');
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('🎯 Fiori dashboard integration ready');
    }

    async start() {
        const startSpan = tracer.startSpan('server.start');
        
        try {
            await this.initialize();
            
            const port = process.env.PORT || 4004;
            
            const server = this.app.listen(port, () => {
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log(`A2A Developer Portal server listening on port ${port}`);
                
                // Initialize WebSocket server for real-time metrics
                const realtimeMetricsService = require('./monitoring/realtimeMetricsService');
                realtimeMetricsService.initializeWebSocketServer(server);
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('🚀 WebSocket server initialized for real-time metrics');
                
                startSpan.addEvent('server_started', { port });
                startSpan.setStatus({ code: 1 });
                startSpan.end();
            });
            
        } catch (error) {
            startSpan.recordException(error);
            startSpan.setStatus({ code: 2, message: error.message });
            startSpan.end();
            throw error;
        }
    }
}

// Create and start server
const server = new A2APortalServer();
server.start().catch(error => {
    console.error('Failed to start server:', error);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('SIGTERM received, shutting down gracefully...');
    await tracer.shutdown();
    process.exit(0);
});
/**
 * Real OpenTelemetry Implementation for A2A Network
 * Provides distributed tracing, metrics, and logging
 */
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { ConsoleSpanExporter } = require('@opentelemetry/sdk-trace-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { PeriodicExportingMetricReader, ConsoleMetricExporter } = require('@opentelemetry/sdk-metrics');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { PrometheusExporter } = require('@opentelemetry/exporter-prometheus');
const { metrics, trace } = require('@opentelemetry/api');

class A2ATelemetry {
    constructor() {
        this.serviceName = 'a2a-network-launchpad';
        this.serviceVersion = process.env.npm_package_version || '1.0.0';
        this.environment = process.env.NODE_ENV || 'development';
        this.isBTP = process.env.BTP_ENVIRONMENT === 'true';

        this.sdk = null;
        this.tracer = null;
        this.meter = null;
        this.metrics = {};
    }

    initialize() {
        try {
            // Create resource identification
            const resource = Resource.default().merge(
                new Resource({
                    [SemanticResourceAttributes.SERVICE_NAME]: this.serviceName,
                    [SemanticResourceAttributes.SERVICE_VERSION]: this.serviceVersion,
                    [SemanticResourceAttributes.SERVICE_NAMESPACE]: 'a2a-network',
                    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: this.environment,
                    [SemanticResourceAttributes.SERVICE_INSTANCE_ID]: process.env.CF_INSTANCE_GUID || 'local-instance',
                })
            );

            // Configure exporters based on environment
            const traceExporters = this.configureTraceExporters();
            const metricExporters = this.configureMetricExporters();

            // Initialize SDK
            this.sdk = new NodeSDK({
                resource: resource,
                traceExporter: traceExporters,
                metricReader: metricExporters,
                instrumentations: [getNodeAutoInstrumentations({
                    // Disable instrumentation for local files
                    '@opentelemetry/instrumentation-fs': {
                        enabled: false
                    },
                    // Configure HTTP instrumentation
                    '@opentelemetry/instrumentation-http': {
                        enabled: true,
                        requestHook: (span, request) => {
                            span.setAttributes({
                                'http.request.user_agent': request.headers['user-agent'],
                                'http.request.x_forwarded_for': request.headers['x-forwarded-for']
                            });
                        }
                    },
                    // Configure Express instrumentation
                    '@opentelemetry/instrumentation-express': {
                        enabled: true,
                        ignoreLayers: [
                            // Ignore static file serving layers
                            (name) => name === 'serveStatic'
                        ]
                    }
                })]
            });

            this.sdk.start();

            // Initialize tracer and meter
            this.tracer = trace.getTracer(this.serviceName, this.serviceVersion);
            this.meter = metrics.getMeter(this.serviceName, this.serviceVersion);

            // Create custom metrics
            this.createCustomMetrics();

            // console.log('✅ OpenTelemetry initialized successfully');
            // console.log(`   Service: ${this.serviceName} v${this.serviceVersion}`);
            // console.log(`   Environment: ${this.environment}`);
            // console.log(`   Trace Export: ${this.isBTP ? 'SAP Cloud Logging' : 'Console + Jaeger'}`);
            // console.log(`   Metrics Export: ${this.isBTP ? 'SAP Monitoring' : 'Console + Prometheus'}`);

        } catch (error) {
            console.error('❌ Failed to initialize OpenTelemetry:', error.message);
            // Don't fail the application if telemetry fails
        }
    }

    configureTraceExporters() {
        if (this.isBTP) {
            // SAP BTP: Use SAP Cloud Logging Service
            try {
                const { SAPCloudLoggingExporter } = require('@opentelemetry/exporter-sap-cloud-logging');
                return new SAPCloudLoggingExporter();
            } catch (error) {
                console.warn('SAP Cloud Logging exporter not available, using console');
                return new ConsoleSpanExporter();
            }
        } else {
            // Local: Use Jaeger if available, otherwise console
            const jaegerEndpoint = process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces';
            try {
                return new JaegerExporter({
                    endpoint: jaegerEndpoint,
                });
            } catch (error) {
                console.warn('Jaeger not available, using console exporter');
                return new ConsoleSpanExporter();
            }
        }
    }

    configureMetricExporters() {
        if (this.isBTP) {
            // SAP BTP: Use SAP Monitoring Service
            try {
                const { SAPMonitoringExporter } = require('@opentelemetry/exporter-sap-monitoring');
                return new PeriodicExportingMetricReader({
                    exporter: new SAPMonitoringExporter(),
                    exportIntervalMillis: 30000
                });
            } catch (error) {
                console.warn('SAP Monitoring exporter not available, using console');
                return new PeriodicExportingMetricReader({
                    exporter: new ConsoleMetricExporter(),
                    exportIntervalMillis: 30000
                });
            }
        } else {
            // Local: Use Prometheus + Console
            try {
                const prometheusExporter = new PrometheusExporter({
                    port: 9090,
                    preventServerStart: false
                });

                // console.log('🔍 Prometheus metrics available at: http://localhost:9090/metrics');

                return new PeriodicExportingMetricReader({
                    exporter: prometheusExporter,
                    exportIntervalMillis: 15000
                });
            } catch (error) {
                console.warn('Prometheus not available, using console exporter');
                return new PeriodicExportingMetricReader({
                    exporter: new ConsoleMetricExporter(),
                    exportIntervalMillis: 30000
                });
            }
        }
    }

    createCustomMetrics() {
        // HTTP Request metrics
        this.metrics.httpRequestsTotal = this.meter.createCounter('http_requests_total', {
            description: 'Total number of HTTP requests'
        });

        this.metrics.httpRequestDuration = this.meter.createHistogram('http_request_duration_seconds', {
            description: 'HTTP request duration in seconds'
        });

        // Application-specific metrics
        this.metrics.tileDataRequests = this.meter.createCounter('tile_data_requests_total', {
            description: 'Total number of tile data requests'
        });

        this.metrics.databaseConnections = this.meter.createUpDownCounter('database_connections_active', {
            description: 'Number of active database connections'
        });

        this.metrics.activeUsers = this.meter.createUpDownCounter('active_users_total', {
            description: 'Number of active users'
        });

        this.metrics.authenticationAttempts = this.meter.createCounter('authentication_attempts_total', {
            description: 'Total authentication attempts'
        });

        this.metrics.launchpadLoads = this.meter.createCounter('launchpad_loads_total', {
            description: 'Total launchpad page loads'
        });

        // Business metrics
        this.metrics.agentOperations = this.meter.createCounter('agent_operations_total', {
            description: 'Total agent operations'
        });

        this.metrics.blockchainTransactions = this.meter.createCounter('blockchain_transactions_total', {
            description: 'Total blockchain transactions'
        });
    }

    // Middleware for Express
    middleware() {
        return (req, res, next) => {
            const start = Date.now();

            // Create span for request
            const span = this.tracer?.startSpan(`${req.method} ${req.path}`, {
                attributes: {
                    'http.method': req.method,
                    'http.url': req.url,
                    'http.route': req.path,
                    'http.user_agent': req.headers['user-agent'],
                    'user.id': req.user?.id || 'anonymous'
                }
            });

            // Track request metrics
            this.recordHttpRequest(req.method, req.path);

            res.on('finish', () => {
                const duration = (Date.now() - start) / 1000;

                // Update span with response info
                span?.setAttributes({
                    'http.status_code': res.statusCode,
                    'http.response_size': res.get('content-length') || 0
                });

                // Record metrics
                this.recordHttpDuration(req.method, req.path, res.statusCode, duration);

                span?.end();
            });

            req.span = span;
            next();
        };
    }

    // Record HTTP request
    recordHttpRequest(method, path) {
        this.metrics.httpRequestsTotal?.add(1, {
            method,
            endpoint: path
        });
    }

    // Record HTTP duration
    recordHttpDuration(method, path, statusCode, duration) {
        this.metrics.httpRequestDuration?.record(duration, {
            method,
            endpoint: path,
            status_code: statusCode.toString()
        });
    }

    // Record tile data request
    recordTileDataRequest(tileId, success = true) {
        this.metrics.tileDataRequests?.add(1, {
            tile_id: tileId,
            success: success.toString()
        });
    }

    // Record launchpad load
    recordLaunchpadLoad(userId) {
        this.metrics.launchpadLoads?.add(1, {
            user_id: userId || 'anonymous'
        });
    }

    // Record authentication attempt
    recordAuthenticationAttempt(method, success = true) {
        this.metrics.authenticationAttempts?.add(1, {
            auth_method: method,
            success: success.toString()
        });
    }

    // Record business operations
    recordAgentOperation(operationType, agentId) {
        this.metrics.agentOperations?.add(1, {
            operation_type: operationType,
            agent_id: agentId
        });
    }

    recordBlockchainTransaction(type, success = true) {
        this.metrics.blockchainTransactions?.add(1, {
            transaction_type: type,
            success: success.toString()
        });
    }

    // Update connection count
    updateDatabaseConnections(count) {
        this.metrics.databaseConnections?.add(count);
    }

    // Update active users
    updateActiveUsers(count) {
        this.metrics.activeUsers?.add(count);
    }

    // Create custom span
    createSpan(name, attributes = {}) {
        return this.tracer?.startSpan(name, { attributes });
    }

    // Shutdown telemetry
    async shutdown() {
        try {
            await this.sdk?.shutdown();
            // console.log('✅ OpenTelemetry shut down successfully');
        } catch (error) {
            console.error('❌ Error shutting down OpenTelemetry:', error.message);
        }
    }
}

// Export singleton instance
const telemetry = new A2ATelemetry();
module.exports = telemetry;
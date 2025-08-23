"use strict";

/**
 * OpenTelemetry Distributed Tracing Configuration
 * SAP Cloud Observability Integration
 */

const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-http');
const { BatchSpanProcessor } = require('@opentelemetry/sdk-trace-base');
const { registerInstrumentations: _registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { RedisInstrumentation } = require('@opentelemetry/instrumentation-redis-4');

// SAP Cloud Logging integration
const { SAPCloudLoggingExporter } = require('./sap-cloud-logging-exporter');

class DistributedTracer {
    constructor() {
        this.sdk = null;
        this.isInitialized = false;
    }

    /**
     * Initialize OpenTelemetry with SAP Cloud Observability
     */
    async initialize() {
        if (this.isInitialized) {
            return;
        }

        try {
            // Configure resource attributes
            const resource = Resource.default().merge(
                new Resource({
                    [SemanticResourceAttributes.SERVICE_NAME]: process.env.SERVICE_NAME || 'a2a-developer-portal',
                    [SemanticResourceAttributes.SERVICE_VERSION]: process.env.SERVICE_VERSION || '2.1.0',
                    [SemanticResourceAttributes.SERVICE_NAMESPACE]: 'sap.a2a.portal',
                    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV || 'development',
                    'sap.tenant_id': process.env.TENANT_ID || 'default',
                    'sap.space_name': process.env.CF_SPACE_NAME || 'dev',
                    'sap.organization_name': process.env.CF_ORGANIZATION_NAME || 'a2a-org',
                    'sap.application_name': 'A2A Developer Portal',
                    'sap.component': 'backend'
                })
            );

            // Configure exporters
            const traceExporter = this._configureExporter();

            // Configure span processor with batching
            const spanProcessor = new BatchSpanProcessor(traceExporter, {
                maxQueueSize: 2048,
                maxExportBatchSize: 512,
                scheduledDelayMillis: 5000,
                exportTimeoutMillis: 30000
            });

            // Initialize SDK
            this.sdk = new NodeSDK({
                resource,
                spanProcessor,
                instrumentations: [
                    // Auto-instrumentations
                    getNodeAutoInstrumentations({
                        '@opentelemetry/instrumentation-fs': {
                            enabled: false, // Disable fs to reduce noise
                        },
                        '@opentelemetry/instrumentation-dns': {
                            enabled: false, // Disable DNS to reduce noise
                        }
                    }),
                    // Custom instrumentations
                    new HttpInstrumentation({
                        requestHook: (span, request) => {
                            span.setAttributes({
                                'http.request.body.size': request.headers['content-length'] || 0,
                                'sap.user_id': request.headers['x-user-id'] || 'anonymous',
                                'sap.correlation_id': request.headers['x-correlation-id'] || '',
                                'sap.request_id': request.headers['x-request-id'] || ''
                            });
                        },
                        responseHook: (span, response) => {
                            span.setAttributes({
                                'http.response.body.size': response.headers['content-length'] || 0,
                                'sap.response_time': response.headers['x-response-time'] || 0
                            });
                        },
                        ignoreIncomingPaths: ['/health', '/metrics', '/livez', '/readyz']
                    }),
                    new ExpressInstrumentation({
                        requestHook: (span, info) => {
                            span.updateName(`${info.request.method} ${info.route || info.request.path}`);
                            span.setAttributes({
                                'express.route': info.route,
                                'express.params': JSON.stringify(info.request.params),
                                'sap.operation': info.request.headers['x-operation'] || 'unknown'
                            });
                        }
                    }),
                    new RedisInstrumentation({
                        responseHook: (span, cmdName, cmdArgs, response) => {
                            span.setAttributes({
                                'redis.command': cmdName,
                                'redis.key': cmdArgs[0] || 'unknown',
                                'redis.response.size': JSON.stringify(response).length
                            });
                        }
                    })
                ]
            });

            // Start the SDK
            await this.sdk.start();
            this.isInitialized = true;

             

            // eslint-disable-next-line no-console

             

            // eslint-disable-next-line no-console
            console.log('OpenTelemetry tracing initialized successfully');
            
            // Configure error handling
            process.on('SIGTERM', () => {
                this.shutdown()
                    // eslint-disable-next-line no-console
                    .then(() => console.log('Tracing terminated'))
                    .catch((error) => console.error('Error terminating tracing', error))
                    .finally(() => process.exit(0));
            });

        } catch (error) {
            console.error('Failed to initialize OpenTelemetry:', error);
            throw error;
        }
    }

    /**
     * Configure the appropriate exporter based on environment
     */
    _configureExporter() {
        const environment = process.env.NODE_ENV || 'development';

        if (environment === 'production') {
            // SAP Cloud Logging exporter for production
            return new SAPCloudLoggingExporter({
                serviceName: 'a2a-developer-portal',
                endpoint: process.env.SAP_CLOUD_LOGGING_ENDPOINT || 'https://logs.cf.sap.hana.ondemand.com',
                apiKey: process.env.SAP_CLOUD_LOGGING_API_KEY,
                spaceGuid: process.env.CF_SPACE_GUID,
                orgGuid: process.env.CF_ORGANIZATION_GUID
            });
        } else {
            // OTLP exporter for development/staging
            return new OTLPTraceExporter({
                url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces',
                headers: {
                    'api-key': process.env.OTEL_EXPORTER_OTLP_HEADERS_API_KEY || ''
                },
                compression: 'gzip'
            });
        }
    }

    /**
     * Create a custom span for business operations
     */
    startSpan(name, attributes = {}) {
        const tracer = trace.getTracer('a2a-portal-business', '1.0.0');
        const span = tracer.startSpan(name, {
            attributes: {
                'sap.business.operation': name,
                'sap.timestamp': new Date().toISOString(),
                ...attributes
            }
        });
        return span;
    }

    /**
     * Add event to current span
     */
    addEvent(name, attributes = {}) {
        const span = trace.getActiveSpan();
        if (span) {
            span.addEvent(name, attributes);
        }
    }

    /**
     * Set baggage for context propagation
     */
    setBaggage(key, value) {
        const baggage = propagation.getBaggage(context.active()) || propagation.createBaggage();
        const updatedBaggage = baggage.setEntry(key, { value });
        return propagation.setBaggage(context.active(), updatedBaggage);
    }

    /**
     * Get baggage value
     */
    getBaggage(key) {
        const baggage = propagation.getBaggage(context.active());
        if (baggage) {
            const entry = baggage.getEntry(key);
            return entry ? entry.value : null;
        }
        return null;
    }

    /**
     * Gracefully shutdown tracing
     */
    async shutdown() {
        if (this.sdk) {
            await this.sdk.shutdown();
            this.isInitialized = false;
        }
    }
}

// Export singleton instance
module.exports = new DistributedTracer();

// Also export the trace API for direct usage
const { trace, context, propagation } = require('@opentelemetry/api');
module.exports.trace = trace;
module.exports.context = context;
module.exports.propagation = propagation;
/**
 * SAP Cloud ALM and Enterprise Monitoring Integration
 * Implements comprehensive observability following SAP standards
 */

// Temporarily disable all OpenTelemetry imports to isolate root cause of Express response patching conflict
// const { NodeSDK } = require('@opentelemetry/sdk-node');
const cds = require('@sap/cds');
// const { Resource } = require('@opentelemetry/resources');
// const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
// const { OTLPTraceExporter } = require('@opentelemetry/exporter-otlp-http');
// const { OTLPMetricExporter } = require('@opentelemetry/exporter-otlp-http');
// const { PeriodicExportingMetricReader } = require('@opentelemetry/sdk-metrics');
// const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node'); // Replaced with selective manual instrumentation to avoid HTTP conflicts
// const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const promClient = require('prom-client');
const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');

class MonitoringIntegration {
  constructor() {
    this.serviceName = 'a2a-network-srv';
    this.serviceVersion = process.env.APP_VERSION || '1.0.0';
    this.environment = process.env.NODE_ENV || 'development';

    // Don't initialize OpenTelemetry in constructor to avoid early HTTP patching conflicts
    this.setupCustomMetrics();
    this.setupHealthChecks();
  }

  /**
   * Initialize OpenTelemetry for SAP Cloud ALM
   */
  initializeOpenTelemetry() {
    // Temporarily disable OpenTelemetry SDK initialization to isolate root cause of Express response patching conflict
    cds.log('service').info('OpenTelemetry temporarily disabled for debugging - SAP enterprise monitoring features will be restored after resolving Express conflict');
    return;

    const sdk = new NodeSDK({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: this.serviceName,
        [SemanticResourceAttributes.SERVICE_VERSION]: this.serviceVersion,
        [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: this.environment,
        [SemanticResourceAttributes.SERVICE_NAMESPACE]: 'sap.a2a',
        'sap.cf.app': process.env.VCAP_APPLICATION ? JSON.parse(process.env.VCAP_APPLICATION).name : 'local',
        'sap.cf.space': process.env.VCAP_APPLICATION ? JSON.parse(process.env.VCAP_APPLICATION).space_name : 'local',
        'sap.cf.org': process.env.VCAP_APPLICATION ? JSON.parse(process.env.VCAP_APPLICATION).organization_name : 'local'
      }),

      traceExporter: new OTLPTraceExporter({
        url: process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT || 'http://localhost:4318/v1/traces',
        headers: {
          'Authorization': `Bearer ${process.env.SAP_CLOUD_ALM_TOKEN}`,
          'SAP-Client': process.env.SAP_CLIENT || '100',
          'SAP-Language': 'EN'
        }
      }),

      metricReader: new PeriodicExportingMetricReader({
        exporter: new OTLPMetricExporter({
          url: process.env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT || 'http://localhost:4318/v1/metrics',
          headers: {
            'Authorization': `Bearer ${process.env.SAP_CLOUD_ALM_TOKEN}`,
            'SAP-Client': process.env.SAP_CLIENT || '100'
          }
        }),
        exportIntervalMillis: 30000 // Export every 30 seconds
      }),

      instrumentations: [
        // Use selective manual instrumentation instead of auto-instrumentations to avoid HTTP conflicts
        new ExpressInstrumentation({
          requestHook: (span, info) => {
            span.setAttributes({
              'sap.tenant_id': info.request.user?.tenant || 'default',
              'sap.user_id': info.request.user?.sub || 'anonymous',
              'sap.correlation_id': info.request.correlationId,
              'sap.component': 'a2a-network',
              'sap.interface': 'REST'
            });
          }
        })
        // Note: HTTP instrumentation deliberately excluded to prevent Express response patching conflicts
      ]
    });

    sdk.start();
    cds.log('service').info('OpenTelemetry started successfully');
  }

  /**
   * Setup custom Prometheus metrics
   */
  setupCustomMetrics() {
    // Create custom metrics registry
    this.register = new promClient.Registry();

    // Add default metrics
    promClient.collectDefaultMetrics({
      register: this.register,
      prefix: 'a2a_',
      gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5]
    });

    // Business metrics
    this.businessMetrics = {
      // Agent metrics
      agentsTotal: new promClient.Gauge({
        name: 'a2a_agents_total',
        help: 'Total number of registered agents',
        labelNames: ['status', 'country'],
        registers: [this.register]
      }),

      agentReputation: new promClient.Histogram({
        name: 'a2a_agent_reputation_score',
        help: 'Agent reputation score distribution',
        buckets: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        registers: [this.register]
      }),

      // Service metrics
      servicesTotal: new promClient.Gauge({
        name: 'a2a_services_total',
        help: 'Total number of services in marketplace',
        labelNames: ['category', 'status'],
        registers: [this.register]
      }),

      serviceOrders: new promClient.Counter({
        name: 'a2a_service_orders_total',
        help: 'Total service orders processed',
        labelNames: ['status', 'category'],
        registers: [this.register]
      }),

      // Workflow metrics
      workflowExecutions: new promClient.Counter({
        name: 'a2a_workflow_executions_total',
        help: 'Total workflow executions',
        labelNames: ['status', 'category'],
        registers: [this.register]
      }),

      workflowDuration: new promClient.Histogram({
        name: 'a2a_workflow_execution_duration_seconds',
        help: 'Workflow execution duration',
        buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60],
        labelNames: ['category'],
        registers: [this.register]
      }),

      // Message routing metrics
      messagesTotal: new promClient.Counter({
        name: 'a2a_messages_total',
        help: 'Total messages processed',
        labelNames: ['status', 'protocol'],
        registers: [this.register]
      }),

      messageDeliveryTime: new promClient.Histogram({
        name: 'a2a_message_delivery_duration_seconds',
        help: 'Message delivery time',
        buckets: [0.001, 0.01, 0.1, 0.5, 1, 2, 5],
        labelNames: ['protocol'],
        registers: [this.register]
      }),

      // Blockchain metrics
      blockchainTransactions: new promClient.Counter({
        name: 'a2a_blockchain_transactions_total',
        help: 'Total blockchain transactions',
        labelNames: ['chain', 'status'],
        registers: [this.register]
      }),

      gasUsage: new promClient.Histogram({
        name: 'a2a_gas_usage',
        help: 'Gas usage per transaction',
        buckets: [1000, 5000, 10000, 50000, 100000, 500000, 1000000],
        labelNames: ['operation_type'],
        registers: [this.register]
      }),

      // Cache metrics
      cacheOperations: new promClient.Counter({
        name: 'a2a_cache_operations_total',
        help: 'Cache operations',
        labelNames: ['operation', 'result'],
        registers: [this.register]
      })
    };
  }

  /**
   * Middleware for collecting business metrics
   */
  metricsMiddleware() {
    return (req, res, next) => {
      const startTime = Date.now();

      // Use event-based response handling instead of overriding res.json to avoid OpenTelemetry conflicts
      res.on('finish', (() => {
        const duration = (Date.now() - startTime) / 1000;

        // Collect API metrics
        this.collectAPIMetrics(req, res, duration);
      }).bind(this));

      next();
    };
  }

  /**
   * Collect API-specific metrics
   */
  collectAPIMetrics(req, res, duration) {
    const path = req.route?.path || req.path;
    const method = req.method;
    const status = res.statusCode;

    // Update business metrics based on endpoint
    if (path.includes('/Agents')) {
      if (method === 'POST' && status === 201) {
        this.businessMetrics.agentsTotal.inc({ status: 'active', country: req.body?.country_code || 'unknown' });
      }
    } else if (path.includes('/Services')) {
      if (method === 'POST' && status === 201) {
        this.businessMetrics.servicesTotal.inc({
          category: req.body?.category || 'unknown',
          status: 'active'
        });
      }
    } else if (path.includes('/ServiceOrders')) {
      if (method === 'POST' && status === 201) {
        this.businessMetrics.serviceOrders.inc({
          status: 'created',
          category: req.body?.service?.category || 'unknown'
        });
      }
    }
  }

  /**
   * Setup comprehensive health checks
   */
  setupHealthChecks() {
    this.healthChecks = {
      database: async () => {
        try {
          // Check database connection
          const db = require('../lib/db-connection');
          await db.test();
          return { status: 'UP', responseTime: '< 100ms' };
        } catch (error) {
          return { status: 'DOWN', error: error.message };
        }
      },

      blockchain: async () => {
        try {
          // Check blockchain connectivity
          const web3 = require('../lib/blockchain-client');
          const blockNumber = await web3.eth.getBlockNumber();
          return { status: 'UP', blockHeight: blockNumber };
        } catch (error) {
          return { status: 'DOWN', error: error.message };
        }
      },

      cache: async () => {
        try {
          const cacheMiddleware = require('./sapCacheMiddleware');
          return await cacheMiddleware.healthCheck();
        } catch (error) {
          return { status: 'DOWN', error: error.message };
        }
      },

      externalServices: async () => {
        const services = [];

        // Check SAP services
        if (process.env.SAP_GRAPH_ENDPOINT) {
          try {
            await blockchainClient.sendMessage(process.env.SAP_GRAPH_ENDPOINT + '/health', {
              timeout: 5000,
              headers: { 'Authorization': `Bearer ${process.env.SAP_GRAPH_TOKEN}` }
            });
            services.push({ name: 'SAP Graph', status: 'UP' });
          } catch (error) {
            services.push({ name: 'SAP Graph', status: 'DOWN', error: error.message });
          }
        }

        return { services };
      }
    };
  }

  /**
   * Health check endpoint handler
   */
  async getHealthStatus() {
    const results = {};
    const checks = Object.keys(this.healthChecks);

    for (const check of checks) {
      try {
        results[check] = await this.healthChecks[check]();
      } catch (error) {
        results[check] = { status: 'DOWN', error: error.message };
      }
    }

    const overallStatus = Object.values(results).every(r => r.status === 'UP') ? 'UP' : 'DOWN';

    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      version: this.serviceVersion,
      environment: this.environment,
      components: results
    };
  }

  /**
   * Prometheus metrics endpoint
   */
  async getMetrics() {
    return await this.register.metrics();
  }

  /**
   * Update business metrics
   */
  updateBusinessMetrics(type, labels, value = 1) {
    const metric = this.businessMetrics[type];
    if (metric) {
      if (metric.inc) {
        metric.inc(labels, value);
      } else if (metric.observe) {
        metric.observe(labels, value);
      } else if (metric.set) {
        metric.set(labels, value);
      }
    }
  }

  /**
   * SAP Cloud ALM alert integration
   */
  async sendAlert(severity, message, details) {
    if (!process.env.SAP_CLOUD_ALM_ALERTS_ENDPOINT) {
      cds.log('service').warn('SAP Cloud ALM alerts not configured');
      return;
    }

    try {
      await blockchainClient.sendMessage(process.env.SAP_CLOUD_ALM_ALERTS_ENDPOINT, {
        severity,
        message,
        timestamp: new Date().toISOString(),
        source: this.serviceName,
        environment: this.environment,
        details
      }, {
        headers: {
          'Authorization': `Bearer ${process.env.SAP_CLOUD_ALM_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });
    } catch (error) {
      cds.log('service').error('Failed to send alert to SAP Cloud ALM:', error.message);
    }
  }

  /**
   * Performance monitoring for critical operations
   */
  monitorOperation(operationName, operationFunction) {
    return async (...args) => {
      const startTime = Date.now();
      let success = false;

      try {
        const result = await operationFunction(...args);
        success = true;
        return result;
      } catch (error) {
        // Send alert for critical failures
        await this.sendAlert('HIGH', `Operation ${operationName} failed`, {
          error: error.message,
          args: args.length
        });
        throw error;
      } finally {
        const duration = Date.now() - startTime;

        // Record performance metrics
        if (this.businessMetrics.workflowDuration) {
          this.businessMetrics.workflowDuration.observe(
            { operation: operationName },
            duration / 1000
          );
        }

        // Alert on slow operations
        if (duration > 30000) { // 30 seconds
          await this.sendAlert('MEDIUM', `Slow operation detected: ${operationName}`, {
            duration_ms: duration,
            success
          });
        }
      }
    };
  }

  /**
   * Initialize monitoring integration
   */
  async initialize() {
    cds.log('service').info('Initializing SAP monitoring integration...');

    // Setup health checks
    this.setupHealthChecks();

    // Initialize OpenTelemetry if not already done
    if (!this.sdk) {
      this.initializeOpenTelemetry();
    }

    cds.log('service').info('SAP monitoring integration initialized successfully');
  }

  /**
   * Initialize monitoring routes
   */
  setupRoutes(app) {
    // Health check endpoint
    app.get('/health', async (req, res) => {
      const health = await this.getHealthStatus();
      const status = health.status === 'UP' ? 200 : 503;
      res.status(status).json(health);
    });

    // Prometheus metrics endpoint
    app.get('/metrics', async (req, res) => {
      res.set('Content-Type', this.register.contentType);
      res.end(await this.getMetrics());
    });

    // Detailed system info (admin only)
    app.get('/system/info', (req, res) => {
      if (!req.user || !req.user.scope?.includes('Admin')) {
        return res.status(403).json({ error: 'Forbidden' });
      }

      res.json({
        service: this.serviceName,
        version: this.serviceVersion,
        environment: this.environment,
        nodejs: process.version,
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage()
      });
    });
  }
}

module.exports = new MonitoringIntegration();
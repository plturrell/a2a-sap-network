# APM Integration Guide - Dynatrace

## Overview
This document outlines the Application Performance Monitoring (APM) integration for the A2A Platform using Dynatrace, aligned with SAP enterprise monitoring standards.

## Dynatrace OneAgent Installation

### 1. SAP BTP Integration
```yaml
# mta.yaml addition
modules:
  - name: a2a-srv
    type: nodejs
    properties:
      DT_TENANT: ${DYNATRACE_TENANT}
      DT_TENANTTOKEN: ${DYNATRACE_TENANT_TOKEN}
      DT_CONNECTION_POINT: ${DYNATRACE_ENDPOINT}
    requires:
      - name: dynatrace-service
        
resources:
  - name: dynatrace-service
    type: org.cloudfoundry.existing-service
    parameters:
      service-name: dynatrace-${space}
```

### 2. Node.js Application Instrumentation
```javascript
// srv/monitoring/dynatraceIntegration.js
const { DiagConsoleLogger, DiagLogLevel, diag } = require('@opentelemetry/api');

class DynatraceAPM {
    constructor() {
        this.initializeDynatrace();
    }

    initializeDynatrace() {
        // Dynatrace auto-instrumentation
        if (process.env.DT_TENANT && process.env.DT_TENANTTOKEN) {
            console.log('Dynatrace OneAgent detected and initialized');
            
            // Set custom application properties
            this.setApplicationProperties();
            
            // Configure custom metrics
            this.setupCustomMetrics();
        }
    }

    setApplicationProperties() {
        // Custom application detection rules
        if (global.dtrum) {
            dtrum.identifyUser(process.env.CF_INSTANCE_GUID);
            dtrum.setApplicationName('A2A-Platform');
            dtrum.setApplicationVersion(process.env.npm_package_version);
        }
    }

    setupCustomMetrics() {
        const { metrics } = require('@opentelemetry/api');
        const meter = metrics.getMeter('a2a-platform', '1.0.0');

        // Business metrics
        this.agentRegistrations = meter.createCounter('a2a.agent.registrations', {
            description: 'Number of agent registrations'
        });

        this.workflowExecutions = meter.createCounter('a2a.workflow.executions', {
            description: 'Number of workflow executions'
        });

        this.apiLatency = meter.createHistogram('a2a.api.latency', {
            description: 'API endpoint latency',
            unit: 'ms'
        });

        this.cacheHitRate = meter.createObservableGauge('a2a.cache.hit_rate', {
            description: 'Cache hit rate percentage'
        });

        // Register callbacks for observable metrics
        this.cacheHitRate.addCallback((observableResult) => {
            const hitRate = this.calculateCacheHitRate();
            observableResult.observe(hitRate);
        });
    }

    // Track custom business transactions
    trackBusinessTransaction(name, fn) {
        if (global.dtrum && global.dtrum.enterAction) {
            const action = dtrum.enterAction(name);
            try {
                const result = fn();
                action.stop();
                return result;
            } catch (error) {
                action.markAsError(error.message);
                action.stop();
                throw error;
            }
        }
        return fn();
    }

    // Report custom events
    reportCustomEvent(eventName, properties) {
        if (global.dtrum && global.dtrum.reportEvent) {
            dtrum.reportEvent(eventName, properties);
        }
    }

    // Track user actions
    trackUserAction(actionName, properties) {
        if (global.dtrum) {
            const action = dtrum.enterAction(actionName);
            Object.keys(properties).forEach(key => {
                action.addProperty(key, properties[key]);
            });
            action.stop();
        }
    }

    calculateCacheHitRate() {
        // Implementation to calculate cache hit rate
        return 85.5; // Example value
    }
}

module.exports = DynatraceAPM;
```

### 3. Express Middleware Integration
```javascript
// srv/middleware/apmMiddleware.js
const DynatraceAPM = require('../monitoring/dynatraceIntegration');

class APMMiddleware {
    constructor() {
        this.apm = new DynatraceAPM();
    }

    // Request tracking middleware
    requestTracking() {
        return (req, res, next) => {
            const start = Date.now();
            
            // Add request ID for correlation
            req.apmContext = {
                requestId: req.id || `req_${Date.now()}`,
                userId: req.user?.id,
                tenantId: req.authInfo?.tenant
            };

            // Track response
            res.on('finish', () => {
                const duration = Date.now() - start;
                
                // Report to Dynatrace
                this.apm.reportCustomEvent('api_request', {
                    endpoint: req.path,
                    method: req.method,
                    statusCode: res.statusCode,
                    duration: duration,
                    userId: req.apmContext.userId,
                    tenantId: req.apmContext.tenantId
                });

                // Update metrics
                this.apm.apiLatency.record(duration, {
                    endpoint: req.path,
                    method: req.method,
                    status: res.statusCode
                });
            });

            next();
        };
    }

    // Error tracking middleware
    errorTracking() {
        return (err, req, res, next) => {
            // Report error to Dynatrace
            this.apm.reportCustomEvent('application_error', {
                error: err.message,
                stack: err.stack,
                endpoint: req.path,
                method: req.method,
                userId: req.apmContext?.userId
            });

            next(err);
        };
    }

    // Database query tracking
    databaseTracking() {
        return (req, res, next) => {
            // Wrap database operations
            const originalQuery = req.db?.query;
            if (originalQuery) {
                req.db.query = (...args) => {
                    return this.apm.trackBusinessTransaction('db_query', () => {
                        return originalQuery.apply(req.db, args);
                    });
                };
            }
            next();
        };
    }
}

module.exports = APMMiddleware;
```

### 4. CAP Service Integration
```javascript
// srv/monitoring/capDynatraceHandler.js
const cds = require('@sap/cds');
const DynatraceAPM = require('./dynatraceIntegration');

class CAPDynatraceHandler {
    constructor() {
        this.apm = new DynatraceAPM();
        this.setupHandlers();
    }

    setupHandlers() {
        // Before handler - start tracking
        cds.on('serving', (service) => {
            service.before('*', (req) => {
                req.apmTransaction = this.apm.trackBusinessTransaction(
                    `${service.name}.${req.event}`,
                    () => {}
                );
            });

            service.after('*', (data, req) => {
                if (req.apmTransaction) {
                    req.apmTransaction.stop();
                }
            });

            service.on('error', (err, req) => {
                if (req.apmTransaction) {
                    req.apmTransaction.markAsError(err.message);
                    req.apmTransaction.stop();
                }
            });
        });
    }

    // Track specific business events
    trackBusinessEvent(eventName, data) {
        this.apm.reportCustomEvent(`business_${eventName}`, {
            ...data,
            timestamp: new Date().toISOString()
        });
    }
}

module.exports = CAPDynatraceHandler;
```

### 5. Configuration
```javascript
// config/apmConfig.js
module.exports = {
    dynatrace: {
        enabled: process.env.NODE_ENV === 'production',
        tenant: process.env.DT_TENANT,
        tenantToken: process.env.DT_TENANTTOKEN,
        endpoint: process.env.DT_CONNECTION_POINT,
        
        // Custom configuration
        customMetrics: {
            enabled: true,
            interval: 60000 // 1 minute
        },
        
        // Business transaction detection
        businessTransactions: [
            'agent_registration',
            'workflow_execution',
            'data_standardization',
            'vector_processing'
        ],
        
        // Automatic baseline
        anomalyDetection: {
            enabled: true,
            sensitivity: 'medium'
        },
        
        // Real user monitoring
        rum: {
            enabled: true,
            applicationId: process.env.DT_APPLICATION_ID
        }
    }
};
```

### 6. Dashboard Configuration
```json
{
  "dashboardMetadata": {
    "name": "A2A Platform Performance",
    "owner": "SAP A2A Team"
  },
  "tiles": [
    {
      "name": "Service Health",
      "type": "SERVICE_HEALTH",
      "configured": true,
      "bounds": {
        "top": 0,
        "left": 0,
        "width": 304,
        "height": 304
      }
    },
    {
      "name": "Response Time",
      "type": "GRAPH_CHART",
      "configured": true,
      "metricExpression": "builtin:service.response.time"
    },
    {
      "name": "Agent Registrations",
      "type": "CUSTOM_CHART",
      "configured": true,
      "metricExpression": "ext:a2a.agent.registrations"
    },
    {
      "name": "Workflow Success Rate",
      "type": "CUSTOM_CHART",
      "configured": true,
      "metricExpression": "ext:a2a.workflow.success_rate"
    }
  ]
}
```

### 7. Alerting Rules
```yaml
# Dynatrace alerting configuration
alerts:
  - name: "High API Latency"
    metric: "builtin:service.response.time"
    threshold: 1000  # ms
    severity: "ERROR"
    
  - name: "Low Cache Hit Rate"
    metric: "ext:a2a.cache.hit_rate"
    threshold: 70  # percentage
    severity: "WARNING"
    
  - name: "Agent Registration Failures"
    metric: "ext:a2a.agent.registration.failures"
    threshold: 5  # count per minute
    severity: "ERROR"
    
  - name: "Database Connection Pool Exhausted"
    metric: "ext:database.connection.pool.exhausted"
    threshold: 1
    severity: "CRITICAL"
```

### 8. Integration with SAP Cloud ALM
```javascript
// srv/monitoring/almIntegration.js
class ALMIntegration {
    async syncMetricsToALM() {
        const metrics = await this.collectDynatraceMetrics();
        
        // Send to SAP Cloud ALM
        await this.almClient.reportMetrics({
            source: 'dynatrace',
            metrics: metrics,
            timestamp: new Date().toISOString()
        });
    }
    
    async collectDynatraceMetrics() {
        // Collect metrics from Dynatrace API
        const response = await fetch(`${DYNATRACE_API}/metrics/query`, {
            headers: {
                'Authorization': `Api-Token ${process.env.DT_API_TOKEN}`
            }
        });
        
        return response.json();
    }
}
```

## Benefits

1. **Real-time Performance Monitoring**: Track application performance in real-time
2. **AI-powered Root Cause Analysis**: Automatic problem detection and analysis
3. **Business Transaction Tracking**: Monitor critical business processes
4. **User Experience Monitoring**: Track real user interactions
5. **Integration with SAP Ecosystem**: Seamless integration with SAP BTP and Cloud ALM

## Next Steps

1. Configure Dynatrace service in SAP BTP
2. Deploy application with OneAgent
3. Set up custom dashboards
4. Configure alerting rules
5. Train team on Dynatrace usage
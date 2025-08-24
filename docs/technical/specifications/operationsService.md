# Operations Service Documentation

## Overview

The Operations Service (`operations-service.js`) provides comprehensive monitoring, health checking, and operational management capabilities for the A2A Network application, following SAP's operational standards for production systems.

## Service Definition

**Service Path**: `/ops`  
**CDS Definition**: `srv/operations-service.cds`  
**Implementation**: `srv/operations-service.js`  
**Authorization**: Requires authenticated user

## Architecture

### Components
```
OperationsService
├── Monitoring Module
│   ├── Application Logging (SAP)
│   ├── Metrics Collection
│   ├── Performance Tracking
│   └── Event Streaming
├── Alert Management
│   ├── Alert Rules Engine
│   ├── Notification Service
│   └── Escalation Manager
├── Health Check System
│   ├── Component Health
│   ├── Dependency Checks
│   └── Synthetic Monitoring
└── Cloud ALM Integration
    ├── Metric Sync
    ├── Incident Management
    └── Configuration Tracking
```

## Business Logic

### 1. Health Monitoring

#### Get Health Status
```javascript
// GET /ops/getHealth
function getHealth()
```

**Business Logic**:
1. Aggregates health from all components
2. Calculates overall health score
3. Identifies active issues
4. Provides actionable insights

**Health Score Calculation**:
```javascript
HealthScore = 100 - Σ(ComponentPenalties)

ComponentPenalties:
- Database disconnected: -30
- Blockchain unavailable: -25
- High CPU (>80%): -20
- High Memory (>85%): -20
- High Response Time (>1s): -10
- Active Critical Alerts: -10 each
```

**Health States**:
- `healthy`: Score ≥ 80
- `degraded`: Score 60-79
- `unhealthy`: Score < 60

**Component Checks**:
```javascript
{
  backend: {
    checks: ["memory", "cpu", "eventLoop"],
    weight: 0.3
  },
  database: {
    checks: ["connectivity", "poolSize", "queryTime"],
    weight: 0.3
  },
  blockchain: {
    checks: ["web3Connection", "contractAccess", "gasPrice"],
    weight: 0.2
  },
  external: {
    checks: ["authService", "messagingQueue", "cache"],
    weight: 0.2
  }
}
```

### 2. Metrics Management

#### Get Metrics
```javascript
// POST /ops/getMetrics
function getMetrics(startTime, endTime, metricNames, tags)
```

**Business Logic**:
1. Queries time-series metrics
2. Applies filtering and aggregation
3. Handles metric resolution
4. Optimizes for large datasets

**Metric Categories**:

**System Metrics**:
- `cpu.utilization`: CPU usage percentage
- `memory.heapUsed`: Heap memory in MB
- `memory.heapTotal`: Total heap size
- `memory.rss`: Resident set size
- `eventLoop.lag`: Event loop delay in ms

**Application Metrics**:
- `http.request.count`: Request volume
- `http.request.duration`: Response times
- `http.error.count`: Error rates
- `http.status.{code}`: Status code distribution

**Business Metrics**:
- `agents.total`: Registered agents
- `agents.active`: Active agents
- `services.calls`: Service invocations
- `workflows.completed`: Workflow completions
- `messages.delivered`: Message delivery rate

**Metric Storage**:
```javascript
{
  retention: {
    raw: "1 hour",      // Full resolution
    "5min": "1 day",    // 5-minute aggregates
    hourly: "1 week",   // Hourly aggregates
    daily: "1 month"    // Daily aggregates
  }
}
```

### 3. Alert Management

#### Alert Lifecycle
```javascript
// Alert states: open → acknowledged → resolved
```

**Alert Creation**:
1. Metric threshold exceeded
2. Error pattern detected
3. Health check failure
4. External monitoring trigger

**Alert Severity Mapping**:
```javascript
const SeverityLevels = {
  critical: {
    priority: 1,
    sla: "15 minutes",
    escalation: "immediate",
    examples: ["database_down", "service_outage"]
  },
  high: {
    priority: 2,
    sla: "1 hour",
    escalation: "30 minutes",
    examples: ["high_error_rate", "performance_degradation"]
  },
  medium: {
    priority: 3,
    sla: "4 hours",
    escalation: "2 hours",
    examples: ["high_cpu", "slow_queries"]
  },
  low: {
    priority: 4,
    sla: "24 hours",
    escalation: "optional",
    examples: ["deprecated_api_usage", "config_drift"]
  }
};
```

#### Alert Rules Engine
```javascript
// POST /ops/createAlertRule
action createAlertRule(name, metricName, condition, threshold, severity)
```

**Rule Types**:
1. **Threshold Rules**: Simple metric comparison
2. **Rate Rules**: Change rate detection
3. **Pattern Rules**: Complex event patterns
4. **Composite Rules**: Multiple conditions

**Rule Examples**:
```javascript
// CPU Alert Rule
{
  name: "high-cpu-usage",
  metric: "cpu.utilization",
  condition: "gt",
  threshold: 80,
  duration: "5 minutes",
  severity: "high"
}

// Error Rate Rule
{
  name: "elevated-error-rate",
  metric: "http.error.rate",
  condition: "gt",
  threshold: 5, // 5%
  duration: "2 minutes",
  severity: "critical"
}

// Composite Rule
{
  name: "service-degradation",
  conditions: [
    { metric: "http.request.duration", condition: "gt", threshold: 1000 },
    { metric: "http.error.rate", condition: "gt", threshold: 2 }
  ],
  operator: "AND",
  severity: "high"
}
```

### 4. Log Management

#### Get Logs
```javascript
// POST /ops/getLogs
function getLogs(startTime, endTime, level, logger, correlationId, limit)
```

**Business Logic**:
1. Queries Application Logging Service
2. Applies security filtering (no PII)
3. Enriches with context
4. Supports correlation tracking

**Log Processing Pipeline**:
```
Application → Structured Logger → Filter → Enrichment → Storage → Query API
                                    ↓
                            Security Scrubbing
                                    ↓
                            Alert Detection
```

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARN`: Warning conditions
- `ERROR`: Error conditions
- `FATAL`: Critical failures

**Log Enrichment**:
```javascript
{
  timestamp: ISO8601,
  level: "ERROR",
  logger: "a2a.service",
  message: "Service call failed",
  correlationId: "uuid-v4",
  tenant: "customer-123",
  user: "user@example.com",
  context: {
    service: "DataProcessor",
    method: "processData",
    duration: 1523,
    error: "Timeout"
  }
}
```

### 5. Performance Tracing

#### Get Traces
```javascript
// POST /ops/getTraces
function getTraces(startTime, endTime, serviceName, operationName, minDuration)
```

**Business Logic**:
1. Collects distributed traces
2. Reconstructs call chains
3. Identifies bottlenecks
4. Provides flamegraph data

**Trace Structure**:
```javascript
{
  traceId: "4bf92f3577b34da2",
  spans: [
    {
      spanId: "a3e7a5c9f0b2d4e6",
      parentSpanId: null,
      operationName: "HTTP POST /api/v1/callService",
      serviceName: "a2a-service",
      startTime: 1634567890123,
      duration: 523,
      tags: {
        "http.method": "POST",
        "http.url": "/api/v1/callService",
        "http.status": 200
      }
    },
    {
      spanId: "b4f8a6d0e1c3f5a7",
      parentSpanId: "a3e7a5c9f0b2d4e6",
      operationName: "BlockchainService.routeCall",
      serviceName: "blockchain-service",
      startTime: 1634567890145,
      duration: 456
    }
  ]
}
```

### 6. Configuration Management

#### Update Configuration
```javascript
// POST /ops/updateConfiguration
action updateConfiguration(name, value)
```

**Configuration Categories**:
1. **Monitoring Settings**
   - `monitoring.interval`: Metric collection frequency
   - `monitoring.retention`: Data retention periods
   - `monitoring.sampling`: Trace sampling rate

2. **Alert Settings**
   - `alerts.enabled`: Global alert toggle
   - `alerts.channels`: Notification channels
   - `alerts.throttle`: Alert rate limiting

3. **Performance Settings**
   - `performance.timeout`: Request timeout
   - `performance.poolSize`: Connection pool size
   - `performance.cacheSize`: Cache limits

**Configuration Validation**:
```javascript
const ValidationRules = {
  "monitoring.interval": {
    type: "number",
    min: 1000,
    max: 60000,
    description: "Milliseconds between metric collections"
  },
  "alerts.throttle": {
    type: "object",
    schema: {
      window: "number",
      maxAlerts: "number"
    }
  }
};
```

### 7. Dashboard API

#### Get Dashboard
```javascript
// GET /ops/getDashboard
function getDashboard()
```

**Dashboard Data Structure**:
```javascript
{
  health: {
    status: "healthy",
    score: 85,
    components: [
      { name: "backend", status: "healthy" },
      { name: "database", status: "degraded" },
      { name: "blockchain", status: "healthy" }
    ],
    issues: ["Database connection pool exhausted"]
  },
  metrics: {
    cpu: 45.2,
    memory: 62.8,
    requests: 1523,
    errors: 12,
    responseTime: 234
  },
  alerts: [
    {
      id: "alert-123",
      name: "high-memory-usage",
      severity: "medium",
      status: "open",
      message: "Memory usage at 85%",
      timestamp: "2024-01-15T10:30:00Z"
    }
  ],
  recentLogs: [
    // Last 5 error/warning logs
  ]
}
```

## Integration Points

### 1. SAP Application Logging Service

**Configuration**:
```javascript
const loggingConfig = {
  service: "application-logs",
  plan: "lite",
  settings: {
    appLogs: {
      loggers: [
        { name: "a2a-network", level: "info" },
        { name: "blockchain", level: "debug" }
      ],
      retentionPeriod: 7,
      forwardToSiem: true
    }
  }
};
```

### 2. SAP Alert Notification Service

**Alert Channels**:
```javascript
{
  email: {
    recipients: ["ops-team@company.com"],
    template: "alert-email-template"
  },
  webhook: {
    url: "https://webhook.company.com/alerts",
    headers: { "X-API-Key": "***" }
  },
  sms: {
    numbers: ["+1234567890"],
    criticalOnly: true
  }
}
```

### 3. SAP Cloud ALM

**Integration Flow**:
```
Metrics → Collection → Aggregation → Cloud ALM Push
           ↓              ↓              ↓
        Local Store    Transform    API Gateway
```

**Sync Configuration**:
```javascript
{
  metrics: {
    interval: 60000,      // 1 minute
    batchSize: 1000,
    compression: true
  },
  health: {
    interval: 30000,      // 30 seconds
    includeComponents: true
  },
  configuration: {
    interval: 300000,     // 5 minutes
    trackChanges: true
  }
}
```

## Security Considerations

### Access Control
```javascript
// Role-based access
const Permissions = {
  "ops.read": ["User", "Admin"],
  "ops.write": ["Admin"],
  "ops.alerts.manage": ["Admin", "Operator"],
  "ops.config.update": ["Admin"]
};
```

### Data Privacy
1. PII scrubbing in logs
2. Encrypted storage for sensitive metrics
3. Audit trail for all operations
4. Data retention compliance

## Performance Optimization

### Caching Strategy
```javascript
const CacheConfig = {
  health: { ttl: 30, key: "ops:health" },
  metrics: { ttl: 60, key: "ops:metrics:{range}" },
  dashboard: { ttl: 10, key: "ops:dashboard" }
};
```

### Query Optimization
1. Time-based partitioning for logs
2. Metric pre-aggregation
3. Indexed alert queries
4. Pagination for large results

## Monitoring the Monitor

### Self-Monitoring
```javascript
{
  "ops.service.health": "Monitor service availability",
  "ops.query.latency": "API response times",
  "ops.storage.usage": "Metric storage consumption",
  "ops.alert.delivery": "Alert notification success"
}
```

## API Examples

### Health Check
```bash
curl -X GET https://api.a2a-network.com/ops/getHealth \
  -H "Authorization: Bearer $TOKEN"
```

### Query Metrics
```bash
curl -X POST https://api.a2a-network.com/ops/getMetrics \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "startTime": "2024-01-15T00:00:00Z",
    "endTime": "2024-01-15T23:59:59Z",
    "metricNames": ["cpu.utilization", "memory.heapUsed"],
    "tags": { "component": "backend" }
  }'
```

### Create Alert Rule
```bash
curl -X POST https://api.a2a-network.com/ops/createAlertRule \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "database-connection-pool",
    "metricName": "db.pool.available",
    "condition": "lt",
    "threshold": 5,
    "severity": "high",
    "description": "Database connection pool running low"
  }'
```

## Troubleshooting

### Common Issues

1. **Missing Metrics**
   - Check metric collection interval
   - Verify component instrumentation
   - Review retention policies

2. **Alert Storm**
   - Enable alert throttling
   - Review alert thresholds
   - Implement alert correlation

3. **Dashboard Slow**
   - Check cache configuration
   - Review query complexity
   - Enable metric pre-aggregation

## Best Practices

1. **Metric Naming**: Use hierarchical names (e.g., `service.method.metric`)
2. **Alert Design**: Start with high thresholds, tune based on patterns
3. **Log Levels**: Use appropriate levels, avoid debug in production
4. **Dashboard Design**: Focus on actionable metrics
5. **Retention**: Balance storage cost with debugging needs
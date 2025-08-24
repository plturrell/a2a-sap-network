# A2A Network - SAP Standard Monitoring & Operations

## Overview

The A2A Network application implements comprehensive monitoring and operations capabilities following SAP standards, including:

- **SAP Application Logging Service** integration
- **SAP Alert Notification Service** configuration
- **SAP Cloud ALM** integration
- **Performance monitoring** with SAP standards

## Architecture

### 1. Application Logging Service

The application integrates with SAP Application Logging Service for centralized log management:

```javascript
// Configuration in mta.yaml
- name: a2a-network-application-logging
  type: org.cloudfoundry.managed-service
  parameters:
    service: application-logs
    service-plan: lite
```

Features:
- Structured JSON logging
- Automatic log forwarding
- Correlation ID tracking
- Multi-tenant support
- 7-day retention period

### 2. Alert Notification Service

Real-time alerting for critical events:

```javascript
// Alert conditions configured:
- CPU usage > 80%
- Memory usage > 85%
- Response time > 1000ms
- Error rate > 5%
```

Notification channels:
- Email notifications
- Webhook integration
- SAP Alert Inbox

### 3. Cloud ALM Integration

The application registers with SAP Cloud ALM for:
- Application lifecycle monitoring
- Performance analytics
- Incident management
- Configuration tracking

### 4. Performance Monitoring

Built-in performance metrics collection:
- CPU and memory utilization
- HTTP request/response metrics
- Event loop lag monitoring
- Database connection pool metrics
- Blockchain service health

## Endpoints

### Health Check
```
GET /health
```
Returns comprehensive health status including component health.

### Metrics (Prometheus Format)
```
GET /metrics
```
Exports metrics in Prometheus format for Cloud ALM scraping.

### Operations Dashboard
```
GET /ops/getDashboard
```
Provides real-time monitoring dashboard data.

## Operations Service API

### Health Management
- `GET /ops/Health` - Get current health status
- `POST /ops/triggerHealthCheck` - Force health check

### Metrics
- `POST /ops/getMetrics` - Query metrics by time range
- `POST /ops/exportToCloudALM` - Export metrics to Cloud ALM

### Alerts
- `GET /ops/Alerts` - List active alerts
- `POST /ops/acknowledgeAlert` - Acknowledge an alert
- `POST /ops/resolveAlert` - Resolve an alert
- `POST /ops/createAlertRule` - Create custom alert rule

### Logs
- `POST /ops/getLogs` - Query application logs
- Supports filtering by:
  - Time range
  - Log level
  - Logger name
  - Correlation ID

### Traces
- `POST /ops/getTraces` - Query performance traces
- Supports filtering by:
  - Service name
  - Operation name
  - Minimum duration

## Configuration

### Environment Variables

```bash
# Application Logging
export LOG_LEVEL=info
export LOG_RETENTION_DAYS=7

# Alert Notification
export ALERT_EMAIL=ops-team@company.com
export ALERT_WEBHOOK_URL=https://webhook.company.com/alerts

# Cloud ALM
export CLOUD_ALM_ENDPOINT=https://alm.company.com
export CLOUD_ALM_LANDSCAPE=production
```

### Auto-scaling Configuration

The application includes auto-scaling rules:
```yaml
scaling_rules:
  - metric_type: cpu
    threshold: 80
    operator: ">"
    adjustment: "+1"
  - metric_type: memory
    threshold: 85
    operator: ">"
    adjustment: "+1"
```

## Monitoring Dashboard

The Operations Dashboard provides:
- Real-time system health visualization
- Performance metrics charts
- Active alerts management
- Recent logs viewer
- Component status tracking

Access the dashboard at: `/app/a2a-fiori/webapp/index.html#/operations`

## Best Practices

1. **Logging**
   - Use structured logging with correlation IDs
   - Include tenant and user context
   - Use appropriate log levels

2. **Metrics**
   - Record business metrics alongside technical metrics
   - Use consistent naming conventions
   - Include relevant tags/dimensions

3. **Alerts**
   - Define clear thresholds
   - Include remediation steps in alert messages
   - Test alert rules regularly

4. **Health Checks**
   - Include all critical components
   - Return detailed status information
   - Use standard HTTP status codes

## Troubleshooting

### Common Issues

1. **Logs not appearing in Application Logging Service**
   - Check service binding in Cloud Foundry
   - Verify log level configuration
   - Ensure correlation ID is present

2. **Alerts not firing**
   - Verify Alert Notification service binding
   - Check alert rule configuration
   - Review metric thresholds

3. **Cloud ALM connection issues**
   - Verify credentials and endpoint
   - Check network connectivity
   - Review authentication token

### Debug Mode

Enable debug logging:
```javascript
// Set in environment
export LOG_LEVEL=debug
export DEBUG=monitoring:*,cloud-alm:*
```

## Security Considerations

- All monitoring endpoints require authentication
- Operations endpoints require Admin scope
- Health and metrics endpoints are public for monitoring tools
- Sensitive data is filtered from logs

## Integration with CI/CD

The monitoring setup integrates with CI/CD pipelines:

1. **Health checks** during deployment
2. **Performance baselines** for regression testing
3. **Alert rule validation** in staging
4. **Log aggregation** across environments

## Support

For monitoring-related issues:
1. Check the operations dashboard
2. Review recent alerts and logs
3. Contact the platform team
4. Create an incident in Cloud ALM
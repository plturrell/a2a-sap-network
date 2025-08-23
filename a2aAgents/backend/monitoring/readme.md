# A2A Network Monitoring Stack

## Overview
This directory contains a complete monitoring stack for the A2A (Agent-to-Agent) network using Prometheus, Grafana, and AlertManager.

## Components

### Prometheus
- **Configuration**: `prometheus.yml`
- **Alert Rules**: `rules/a2a-alerts.yml`
- **Port**: 9090
- **Purpose**: Metrics collection from all A2A services

### Grafana
- **Dashboards**: `grafana/dashboards/`
  - `a2a-overview.json`: Network-wide overview dashboard
  - `a2a-agents.json`: Agent-specific metrics dashboard
- **Datasources**: `grafana/datasources/prometheus.yaml`
- **Port**: 3000
- **Credentials**: admin / a2a_admin_123

### AlertManager
- **Configuration**: `alertmanager.yml`
- **Port**: 9093
- **Purpose**: Alert routing and notification management

### Additional Exporters
- **Node Exporter**: System metrics (Port 9100)
- **cAdvisor**: Container metrics (Port 8080)  
- **Redis Exporter**: Redis metrics (Port 9121)

## Quick Start

1. **Start the monitoring stack**:
   ```bash
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access Grafana**:
   - URL: http://localhost:3000
   - Username: admin
   - Password: a2a_admin_123

3. **Access Prometheus**:
   - URL: http://localhost:9090

4. **Access AlertManager**:
   - URL: http://localhost:9093

## Agent Metrics Integration

Each A2A agent exposes Prometheus metrics on dedicated ports:
- Agent 0 (Data Product): 8001
- Agent 1 (Data Integration): 8002
- Agent 2 (Data Validation): 8003
- Agent 3 (ML Orchestration): 8004
- Agent 4 (Workflow Orchestration): 8005
- Agent 5 (Compliance Monitor): 8006

### Key Metrics
- `a2a_agent_tasks_completed_total`: Total completed tasks per agent
- `a2a_agent_tasks_failed_total`: Total failed tasks per agent
- `a2a_agent_processing_time_seconds`: Task processing time histogram
- `a2a_agent_queue_depth`: Current task queue depth
- `a2a_agent_skills_count`: Number of skills available per agent

## Alert Configuration

Alerts are configured in `rules/a2a-alerts.yml` and include:

### Critical Alerts
- Service unavailability
- Agent failures
- High error rates
- SLA violations

### Warning Alerts
- High response times
- Resource usage warnings
- Task queue backlogs

### Notification Channels
Configure email and Slack notifications in `alertmanager.yml`:

```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'ops-team@example.com'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
```

## Dashboard Highlights

### A2A Network Overview
- Service availability status
- Request rates across services
- 95th percentile response times
- Error rates by service
- Real-time service health table

### A2A Agents Dashboard
- Agent queue depths
- Task processing times
- Task completion and failure rates
- Success rate calculations
- Skills inventory by agent

## Maintenance

### Data Retention
- Prometheus: 30 days (configurable in docker-compose)
- Grafana: Persistent volumes for dashboards and settings

### Backup
Important files to backup:
- `prometheus.yml` - Scrape configuration
- `rules/a2a-alerts.yml` - Alert rules
- `grafana/dashboards/*.json` - Custom dashboards
- `alertmanager.yml` - Notification configuration

### Scaling
To add new agents or services:
1. Add scrape job to `prometheus.yml`
2. Expose metrics endpoint from your service
3. Update dashboard queries if needed
4. Add relevant alert rules

## Troubleshooting

### Common Issues

**Metrics not showing**:
- Check if agent metrics endpoints are accessible
- Verify Prometheus scrape targets are healthy
- Ensure proper network connectivity between containers

**Alerts not firing**:
- Check AlertManager configuration
- Verify SMTP/Slack webhook settings
- Review alert rule expressions in Prometheus UI

**Dashboard issues**:
- Verify Prometheus datasource connection
- Check metric names and label consistency
- Review dashboard variable configurations

### Debug Commands
```bash
# Check container status
docker-compose -f docker-compose.monitoring.yml ps

# View Prometheus config
docker exec a2a-prometheus cat /etc/prometheus/prometheus.yml

# Check AlertManager config
docker exec a2a-alertmanager cat /etc/alertmanager/alertmanager.yml

# View container logs
docker-compose -f docker-compose.monitoring.yml logs prometheus
docker-compose -f docker-compose.monitoring.yml logs grafana
```

## Security Considerations

1. **Change default passwords** in production
2. **Configure HTTPS** for external access
3. **Restrict network access** to monitoring ports
4. **Review alert notification channels** for sensitive data
5. **Enable authentication** for Prometheus and AlertManager in production

## Performance Tuning

### Prometheus
- Adjust `--storage.tsdb.retention.time` for data retention
- Configure `--storage.tsdb.retention.size` for disk space limits
- Tune scrape intervals based on monitoring needs

### Grafana
- Use query caching for frequently accessed dashboards
- Optimize dashboard refresh rates
- Consider using recording rules for complex queries

## Integration with A2A SDK

The monitoring stack integrates with the A2A SDK through:
1. **Automatic metrics exposure** via SDK's telemetry module
2. **OpenTelemetry traces** forwarded to Prometheus
3. **Agent lifecycle events** captured as metrics
4. **Workflow execution metrics** for end-to-end visibility
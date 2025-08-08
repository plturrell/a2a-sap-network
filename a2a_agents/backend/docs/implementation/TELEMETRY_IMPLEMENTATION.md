# OpenTelemetry Implementation for A2A Network

## Overview

The A2A Network now includes comprehensive distributed tracing using OpenTelemetry, providing end-to-end visibility across all agent interactions, workflows, and system operations.

## Architecture

### Components

1. **OpenTelemetry SDK**: Integrated into all agents and services
2. **OTLP Collector**: Central collection point for traces, metrics, and logs
3. **Jaeger**: Distributed tracing backend for trace visualization
4. **Trace Propagation**: W3C TraceContext for cross-service correlation

### Key Features

- **Automatic Instrumentation**: FastAPI, HTTPX, and Redis are automatically instrumented
- **Custom Spans**: Agent-specific operations are traced with custom spans
- **Context Propagation**: Trace context flows through all agent communications
- **Sampling**: Configurable sampling rates (default 10%)
- **Resource Attributes**: Rich metadata including agent IDs, workflow IDs, and task IDs

## Configuration

### Environment Variables

```bash
# Core Settings
OTEL_ENABLED=true
OTEL_SERVICE_NAME=a2a-agent-{id}
OTEL_ENVIRONMENT=production

# Collector Endpoint
OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317

# Sampling
OTEL_TRACES_SAMPLER=traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampling

# Resource Attributes
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=production,service.namespace=a2a-network
```

### Per-Agent Configuration

Each agent automatically initializes with:
- Unique service name: `a2a-agent-{agent_id}`
- Agent-specific resource attributes
- Automatic HTTP client instrumentation

## Instrumentation Points

### 1. Agent Message Processing
```python
@trace_async("process_message", kind=trace.SpanKind.SERVER)
async def process_message(self, message: A2AMessage, context_id: str):
    # Automatically traced with message metadata
```

### 2. Task Execution
```python
with trace_agent_task(agent_id, task_id, task_type):
    # Task execution logic
```

### 3. Blockchain Operations
```python
with trace_blockchain_operation("register_agent", contract_address):
    # Blockchain interaction
```

### 4. Data Standardization
```python
with trace_standardization(data_type, record_count):
    # Standardization logic
```

## Trace Attributes

### Standard Attributes
- `agent.id`: Unique agent identifier
- `agent.name`: Human-readable agent name
- `message.id`: Message identifier
- `task.id`: Task identifier
- `workflow.id`: Workflow identifier
- `context.id`: Context identifier

### HTTP Attributes
- `http.method`: HTTP method
- `http.url`: Full URL
- `http.status_code`: Response status
- `http.user_agent`: Client user agent

### Custom A2A Attributes
- `a2a.source_agent`: Source agent ID
- `a2a.workflow_id`: Active workflow
- `a2a.task_id`: Current task
- `standardization.data_type`: Type of data being standardized
- `blockchain.operation`: Blockchain operation type

## Deployment

### Start Services with Telemetry

```bash
# Start all services with OpenTelemetry enabled
./start_a2a_services_with_telemetry.sh
```

### Docker Compose

```bash
# Start telemetry infrastructure only
docker-compose -f docker-compose.telemetry.yml up -d
```

### Access Points

- **Jaeger UI**: http://localhost:16686
- **OTLP gRPC**: localhost:4317
- **OTLP HTTP**: localhost:4318
- **Collector Health**: http://localhost:13133/health
- **Collector Metrics**: http://localhost:8888/metrics

## Usage Examples

### View Traces in Jaeger

1. Open http://localhost:16686
2. Select service (e.g., `a2a-agent-0`)
3. Find traces by:
   - Operation name
   - Tags (agent.id, workflow.id, etc.)
   - Time range

### Trace Analysis

Common queries in Jaeger:
- All messages for a workflow: `workflow.id=<id>`
- Failed operations: `error=true`
- Slow requests: `duration>5s`
- Agent interactions: `agent.id=<id>`

### Performance Analysis

1. **Service Dependencies**: View service map showing agent interactions
2. **Latency Distribution**: Analyze p50, p95, p99 latencies
3. **Error Rates**: Track error rates by service and operation
4. **Throughput**: Monitor requests per second by agent

## Monitoring Best Practices

### 1. Critical Paths
Monitor key workflows:
- Agent registration
- Data product registration
- Standardization pipeline
- Trust verification

### 2. SLA Monitoring
Track SLA compliance:
- Message processing time < 5s
- Workflow completion time < 30s
- Error rate < 1%

### 3. Anomaly Detection
Watch for:
- Sudden latency spikes
- Increased error rates
- Unusual traffic patterns
- Resource exhaustion

## Troubleshooting

### Common Issues

1. **No traces appearing**
   - Check OTEL_ENABLED=true
   - Verify collector is running
   - Check network connectivity

2. **Missing spans**
   - Verify instrumentation is loaded
   - Check sampling rate
   - Review span filters

3. **High latency**
   - Check batch processor settings
   - Review collector resources
   - Optimize sampling rate

### Debug Mode

Enable debug logging:
```bash
OTEL_PYTHON_LOG_LEVEL=DEBUG
```

## Future Enhancements

1. **Metrics Integration**: Add Prometheus metrics export
2. **Log Correlation**: Correlate logs with trace IDs
3. **Custom Dashboards**: Grafana dashboards for A2A metrics
4. **Alerting**: Automated alerts for SLA violations
5. **Long-term Storage**: Tempo integration for trace archival
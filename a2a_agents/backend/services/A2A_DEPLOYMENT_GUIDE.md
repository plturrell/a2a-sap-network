# A2A Microservices Deployment Guide

## Overview
This guide explains how to deploy the A2A (Agent-to-Agent) microservices architecture.

## Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Agent 0        │────▶│  Agent 1        │────▶│  Agent 2        │────▶│  Agent 3        │
│  Data Product   │     │  Standardization│     │  AI Preparation │     │  Vector Process │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┴───────────────────────┴───────────────────────┘
                                            │
                                    ┌───────────────┐
                                    │ Agent Manager │
                                    │ (Orchestrator)│
                                    └───────────────┘
```

## Deployment Options

### 1. Local Development
```bash
# Build all services
docker-compose build

# Start all A2A agents
docker-compose up

# Or start specific agents
docker-compose up agent0-data-product agent1-standardization
```

### 2. Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy A2A stack
docker stack deploy -c docker-compose.yml a2a-network

# Scale specific agents
docker service scale a2a-network_agent1-standardization=3
```

### 3. Kubernetes
```bash
# Create namespace
kubectl create namespace a2a-network

# Apply configurations
kubectl apply -f k8s/

# Check deployments
kubectl -n a2a-network get pods
```

## Environment Configuration

### Required Environment Variables
```bash
# A2A Network Configuration
AGENT_MANAGER_URL=http://agent-manager:8007
TRUST_CONTRACT_ADDRESS=0x...

# External Services
ORD_REGISTRY_URL=http://ord-registry:8000/api/v1/ord
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key
HANA_HOST=your-hana-host
HANA_PORT=30015
HANA_USER=your-user
HANA_PASSWORD=your-password

# Monitoring
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
PROMETHEUS_PUSHGATEWAY=http://prometheus-pushgateway:9091
```

## A2A Protocol Compliance

### Each Agent Must:
1. Expose `/.well-known/agent.json` endpoint
2. Implement JSON-RPC 2.0 at `/a2a/agentX/v1/rpc`
3. Register with Agent Manager on startup
4. Implement trust contract verification
5. Handle A2A messages according to protocol v0.2.9

### Health Checks
All agents expose:
- `/health` - Basic health status
- `/ready` - A2A readiness check
- `/a2a/agentX/v1/status` - Detailed agent status

## Monitoring

### Prometheus Metrics
Each agent exposes metrics at `/metrics`:
- `a2a_messages_processed_total`
- `a2a_processing_duration_seconds`
- `a2a_errors_total`
- `a2a_active_tasks`

### OpenTelemetry Tracing
Distributed tracing across all A2A communications:
- Trace ID propagation via `X-A2A-Trace-ID` header
- Span creation for each agent processing step
- Context propagation through the pipeline

## Security

### A2A Trust Contracts
- Each agent has a unique identity in the trust contract
- Messages are signed and verified
- Agent capabilities are registered on-chain

### Network Security
- mTLS between agents (production)
- API key authentication for external access
- Network policies in Kubernetes

## Troubleshooting

### Common Issues

1. **Agent Registration Fails**
   - Check Agent Manager is running
   - Verify network connectivity
   - Check trust contract deployment

2. **Message Processing Errors**
   - Verify A2A message format
   - Check downstream agent availability
   - Review agent logs

3. **Performance Issues**
   - Scale individual agents as needed
   - Check resource limits
   - Monitor message queue depths

### Debugging Commands
```bash
# Check agent logs
docker-compose logs agent1-standardization

# Test agent endpoint
curl http://localhost:8001/.well-known/agent.json

# Check A2A network status
curl http://localhost:8007/a2a/network/status

# Trace message flow
curl -H "X-A2A-Trace-ID: test-123" http://localhost:8001/a2a/agent0/v1/rpc
```

## Production Checklist

- [ ] All agents have resource limits defined
- [ ] Persistent volumes configured for stateful agents
- [ ] Backup strategy for Data Manager
- [ ] Monitoring and alerting configured
- [ ] Trust contracts deployed and verified
- [ ] Network policies configured
- [ ] TLS certificates installed
- [ ] Log aggregation configured
- [ ] Disaster recovery plan tested
- [ ] Performance benchmarks established
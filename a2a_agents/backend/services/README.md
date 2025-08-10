# A2A Network Microservices

## Overview

This is the production-ready A2A (Agent-to-Agent) network implementation following microservices architecture. Each agent is a standalone service that communicates via the A2A protocol v0.2.9.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent 0   │────▶│   Agent 1   │────▶│   Agent 2   │────▶│   Agent 3   │
│ Data Product│     │Standardize  │     │AI Prepare   │     │Vector Store │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ↓                    ↓                    ↓                    ↓
   ┌───────────────────────────────────────────────────────────────────┐
   │                        Agent Manager                               │
   │                   (Service Discovery & Orchestration)              │
   └───────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Local Development

```bash
# Build all services
docker-compose build

# Start the A2A network
docker-compose up

# Run tests
cd tests
pytest test_a2a_pipeline.py
```

### Production Deployment (Kubernetes)

```bash
# Deploy to Kubernetes
kubectl apply -k k8s/base

# Check deployment status
kubectl -n a2a-network get pods

# View logs
kubectl -n a2a-network logs -f deployment/agent1-standardization
```

## Services

### Core Pipeline Agents

1. **Agent 0 - Data Product Registration** (Port 8001)
   - Ingests financial data products
   - Extracts Dublin Core metadata
   - Registers with ORD catalog

2. **Agent 1 - Data Standardization** (Port 8002)
   - Standardizes financial data to L4 hierarchy
   - Supports: accounts, books, locations, measures, products
   - Outputs standardized JSON/Parquet

3. **Agent 2 - AI Preparation** (Port 8003)
   - Semantic enrichment
   - Embedding generation
   - Relationship extraction

4. **Agent 3 - Vector Processing** (Port 8004)
   - Stores vectors in SAP HANA Cloud
   - Builds knowledge graphs
   - Enables similarity search

### Supporting Services

- **Agent Manager** (Port 8007) - Service discovery and orchestration
- **Data Manager** (Port 8005) - Storage and CRUD operations
- **Catalog Manager** (Port 8006) - ORD catalog interface

## A2A Protocol Endpoints

Each agent exposes:

- `GET /.well-known/agent.json` - Agent capabilities
- `POST /a2a/agentX/v1/rpc` - JSON-RPC 2.0 endpoint
- `GET /a2a/agentX/v1/stream` - Server-sent events
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics

## Configuration

### Environment Variables

```bash
# Required for all agents
AGENT_MANAGER_URL=http://agent-manager:8007
TRUST_CONTRACT_ADDRESS=0x...

# Agent-specific
ORD_REGISTRY_URL=http://ord-registry:8000/api/v1/ord
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key
HANA_HOST=your-hana-host
HANA_PORT=30015
HANA_USER=your-user
HANA_PASSWORD=your-password
```

## Development

### Adding a New Agent

1. Create directory: `services/agent_name/`
2. Add Dockerfile, requirements.txt, src/
3. Implement A2A protocol in src/agent.py
4. Add to docker-compose.yml
5. Create Kubernetes manifests

### Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/test_a2a_pipeline.py

# Load tests
locust -f tests/load/locustfile.py
```

## Monitoring

### Prometheus Metrics

- `a2a_messages_processed_total` - Total messages processed
- `a2a_processing_duration_seconds` - Processing time histogram
- `a2a_errors_total` - Error count
- `a2a_active_tasks` - Currently active tasks

### Grafana Dashboards

Access at http://localhost:3000
- A2A Network Overview
- Agent Performance
- Pipeline Flow

### Alerts

Configured alerts:
- Agent down
- High error rate
- Processing latency
- Pipeline stalled
- Memory usage

## CI/CD

GitHub Actions workflow:
1. Test common library
2. Build Docker images
3. Run integration tests
4. Deploy to dev (on develop branch)
5. Deploy to prod (on main branch)

## Troubleshooting

### Agent Not Starting

```bash
# Check logs
docker-compose logs agent1-standardization

# Verify network
docker network ls
docker network inspect services_a2a-network

# Test connectivity
docker exec -it services_agent0-data-product_1 ping agent1-standardization
```

### Message Not Processing

```bash
# Check agent status
curl http://localhost:8001/a2a/agent0/v1/status

# Verify registration
curl http://localhost:8007/a2a/network/status

# Trace message
curl -H "X-A2A-Trace-ID: debug-123" http://localhost:8001/a2a/agent0/v1/rpc
```

## Security

- Smart contract trust verification
- mTLS between agents (production)
- API key authentication
- Network policies in Kubernetes
- Secrets management via Kubernetes secrets

## Performance

Benchmarks (per agent):
- Throughput: 1000 msg/sec
- Latency p50: 50ms
- Latency p95: 200ms
- Memory: 512MB-2GB
- CPU: 0.5-2 cores

## License

Proprietary - A2A Network

## Support

- Documentation: docs/
- Issues: GitHub Issues
- Slack: #a2a-network
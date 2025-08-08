# A2A Microservice Architecture

## Overview
Each A2A agent is a standalone microservice that can be independently deployed, scaled, and maintained.

## Structure
```
services/
├── agent0_data_product/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py         # Entry point
│   │   ├── agent.py        # Agent implementation
│   │   └── router.py       # FastAPI routes
│   ├── config/
│   │   └── config.yaml
│   └── tests/
│       └── test_agent.py
├── agent1_standardization/
│   └── (same structure)
├── agent2_ai_preparation/
│   └── (same structure)
├── agent3_vector_processing/
│   └── (same structure)
├── agent_manager/
│   └── (same structure)
├── data_manager/
│   └── (same structure)
├── catalog_manager/
│   └── (same structure)
├── shared/
│   └── a2a_common/         # Shared library
│       ├── setup.py
│       ├── sdk/
│       ├── core/
│       ├── skills/
│       └── security/
└── docker-compose.yml
```

## Benefits of This Approach

### 1. **True Microservice Architecture**
- Each agent is independently deployable
- Can scale agents individually based on load
- Failure isolation - one agent crash doesn't affect others

### 2. **Container-Ready**
- Each agent has its own Dockerfile
- Clear dependencies per agent
- Easy to deploy to Kubernetes

### 3. **Development Benefits**
- Teams can work on agents independently
- Clear boundaries and interfaces
- Easier testing and debugging

### 4. **Shared Code Management**
- Common code in `shared/a2a_common`
- Can be pip-installed or copied during build
- Version controlled separately if needed

## Running the System

### Development Mode
```bash
# Run individual agent
cd services/agent0_data_product
python -m src.main

# Run all agents
cd services
docker-compose up
```

### Production Mode
```bash
# Build images
docker-compose build

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## Communication Patterns

### Service Discovery
- In Docker: Use service names (e.g., `http://agent1-standardization:8002`)
- In K8s: Use Kubernetes Services
- Can add service mesh (Istio/Linkerd) for advanced routing

### A2A Protocol
- Each agent exposes:
  - `/.well-known/agent.json` - Agent capabilities
  - `/a2a/agentX/v1/rpc` - JSON-RPC endpoint
  - `/a2a/agentX/v1/stream` - SSE for events
  - `/health` - Health check

## Migration Path

### From Current Structure
1. Copy agent SDK files to respective `src/agent.py`
2. Update imports to use shared library
3. Create Dockerfile for each agent
4. Test individually, then together

### Gradual Migration
- Can run some agents in containers, others standalone
- Use environment variables for service URLs
- Maintain backward compatibility
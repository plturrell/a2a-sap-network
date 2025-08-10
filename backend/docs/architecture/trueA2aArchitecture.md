# True A2A (Agent-to-Agent) Microservices Architecture

## Overview

This implementation provides **true Agent-to-Agent communication** with dynamic service discovery, eliminating all hardcoded URLs and enabling genuine microservices architecture.

## Architecture Components

### 1. **A2A Registry Service** (Port 8000)
- Central service registry for agent discovery
- Health monitoring of registered agents
- Workflow requirement matching
- RESTful API for agent registration and discovery

### 2. **Agent 0 - Data Product Registration** (Port 8002)
- Independent microservice
- Registers with A2A Registry on startup
- Discovers downstream agents dynamically
- No hardcoded agent URLs

### 3. **Agent 1 - Financial Standardization** (Port 8001)
- Independent microservice
- Self-registers with A2A Registry
- Discoverable by other agents
- Processes messages from dynamically discovered agents

## Key Features

### ✅ **Dynamic Service Discovery**
- Agents register themselves with the registry on startup
- Agents discover each other through the registry
- No hardcoded URLs between agents
- Supports skill-based and tag-based discovery

### ✅ **True Microservices**
- Each agent runs as a separate process on different ports
- Independent deployment and scaling
- Fault isolation between services
- Can be containerized with Docker

### ✅ **Health Monitoring**
- Registry monitors agent health status
- Automatic detection of unhealthy agents
- Only routes to healthy agents

### ✅ **Smart Contract Trust**
- Cryptographic message signing between agents
- Trust verification for secure communication
- Agent identity management

## Quick Start

### 1. Start All Services

```bash
# Start all A2A microservices
./start_a2a_services.sh
```

This will start:
- Registry Service on http://localhost:8000
- Agent 0 on http://localhost:8002
- Agent 1 on http://localhost:8001

### 2. Test True A2A Communication

```bash
# Run the true A2A test suite
python3 test_true_a2a.py
```

### 3. Stop All Services

```bash
# Stop all services
./stop_a2a_services.sh
```

## Docker Deployment

```bash
# Start with Docker Compose
docker-compose -f docker-compose.a2a.yml up

# Scale agents
docker-compose -f docker-compose.a2a.yml up --scale agent0=3 --scale agent1=2
```

## API Examples

### Discover Agents
```bash
# Find all healthy agents
curl http://localhost:8000/api/v1/a2a/agents/search?status=healthy

# Find agents by skill
curl http://localhost:8000/api/v1/a2a/agents/search?skills=standardization

# Find agents by tags
curl http://localhost:8000/api/v1/a2a/agents/search?tags=financial,dublin-core
```

### Register an Agent
```bash
curl -X POST http://localhost:8000/api/v1/a2a/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my_agent",
    "agent_card": {
      "name": "My Custom Agent",
      "url": "http://localhost:8003",
      "skills": [...],
      ...
    }
  }'
```

### Match Workflow Requirements
```bash
curl -X POST http://localhost:8000/api/v1/a2a/agents/match \
  -H "Content-Type: application/json" \
  -d '{
    "required_skills": ["dublin-core-extraction", "standardization"],
    "preferred_tags": ["financial"]
  }'
```

## Code Example: Dynamic Agent Discovery

```python
from app.a2a_registry.client import A2ARegistryClient

# Initialize registry client
registry = A2ARegistryClient()

# Discover agent by skill
agent = await registry.find_agent_by_skill("standardization")
if agent:
    print(f"Found agent: {agent['name']} at {agent['url']}")
    
# Send message to discovered agent
result = await registry.send_message_to_agent(
    agent_id=agent["agent_id"],
    message={
        "role": "user",
        "parts": [{"kind": "text", "text": "Process this data"}]
    }
)
```

## Environment Variables

### Registry Service
- `HANA_HOST`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD` - HANA database
- `SQLITE_DB_PATH`, `SQLITE_JOURNAL_MODE` - SQLite configuration

### Agent 0
- `AGENT0_PORT` - Port to run on (default: 8002)
- `A2A_REGISTRY_URL` - Registry URL (default: http://localhost:8000/api/v1/a2a)
- `ORD_REGISTRY_URL` - ORD Registry URL

### Agent 1
- `AGENT1_PORT` - Port to run on (default: 8001)
- `A2A_REGISTRY_URL` - Registry URL

## Monitoring

### Check Agent Health
```bash
# Individual agent health
curl http://localhost:8001/health
curl http://localhost:8002/health

# All agents via registry
curl http://localhost:8000/api/v1/a2a/agents/search | jq '.agents[] | {name, status, last_health_check}'
```

### View Registered Agents
```bash
# See all registered agents
curl http://localhost:8000/api/v1/a2a/agents/search | jq .
```

## Troubleshooting

### Agent Not Discoverable
1. Check agent is running: `curl http://localhost:800X/health`
2. Check registration: `curl http://localhost:8000/api/v1/a2a/agents/{agent_id}`
3. Check registry logs for registration errors

### Communication Failures
1. Verify both agents are healthy
2. Check network connectivity between services
3. Verify message format matches A2A protocol v0.2.9

### Registry Issues
1. Ensure registry service is running on port 8000
2. Check database connectivity
3. Verify CORS settings if running across domains

## Architecture Benefits

1. **Scalability**: Each agent can be scaled independently
2. **Resilience**: Agent failures don't affect other agents
3. **Flexibility**: New agents can be added without modifying existing ones
4. **Maintainability**: Clear service boundaries and contracts
5. **Discoverability**: Agents found dynamically, no configuration needed

## Comparison: Before vs After

### Before (Monolithic)
```python
# Hardcoded URLs
agent1_url = "http://localhost:8000/a2a/v1/messages"
response = await client.post(agent1_url, json=message)
```

### After (True A2A)
```python
# Dynamic discovery
agent = await registry.find_agent_by_skill("standardization")
response = await registry.send_message_to_agent(agent["agent_id"], message)
```

## Next Steps

1. Add more agents to the ecosystem
2. Implement agent capability negotiation
3. Add distributed tracing (OpenTelemetry)
4. Implement circuit breakers for resilience
5. Add agent versioning support

---

**This is TRUE A2A Architecture** - No hardcoded URLs, dynamic discovery, independent services! 🚀
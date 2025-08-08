# A2A Registry Specification
## Agent-to-Agent Registry Service

### Overview

**Service Name**: A2A Registry Service  
**Purpose**: Centralized registry for A2A agents with discovery, orchestration, and health monitoring capabilities  
**Integration**: Backend service for A2A agent ecosystems and workflow orchestration  
**Compliance**: A2A Protocol v0.2.9+ compliant  

---

## Service Architecture

### Core Capabilities

- **Agent Registration**: Accept and validate A2A agent cards
- **Agent Discovery**: Search and browse registered agents by capabilities
- **Health Monitoring**: Track agent status and availability
- **Capability Matching**: Find agents by skills and requirements
- **Workflow Orchestration**: Support for multi-agent pipelines
- **Version Management**: Track agent versions and compatibility

### Supported Agent Types

| Agent Type | Description | Examples |
|------------|-------------|----------|
| `data-processing` | Data transformation and processing | Standardization, cleaning, validation |
| `ai-ml` | AI and machine learning services | Vectorization, semantic analysis, inference |
| `storage` | Data storage and retrieval | Vector databases, data lakes, catalogs |
| `orchestration` | Workflow and pipeline management | Data product creation, pipeline coordination |
| `analytics` | Data analysis and insights | Reporting, visualization, monitoring |

---

## API Specification

### Agent Registration Endpoints

#### Register A2A Agent
```http
POST /api/v1/agents/register
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Financial Data Standardization Agent",
  "description": "Standardizes financial entities...",
  "url": "https://standardization-agent.example.com",
  "version": "1.0.0",
  "protocolVersion": "0.2.9",
  "capabilities": {...},
  "skills": [...]
}
```

**Response:**
```json
{
  "agent_id": "agent_12345",
  "status": "registered",
  "validation_results": {
    "valid": true,
    "warnings": [],
    "errors": []
  },
  "registered_at": "2024-01-01T12:00:00Z",
  "registry_url": "https://a2a-registry.example.com/agents/agent_12345",
  "health_check_url": "https://a2a-registry.example.com/agents/agent_12345/health"
}
```

#### Update Agent Registration
```http
PUT /api/v1/agents/register/{agent_id}
Content-Type: application/json
Authorization: Bearer {token}
```

#### Deregister Agent
```http
DELETE /api/v1/agents/register/{agent_id}
Authorization: Bearer {token}
```

### Agent Discovery Endpoints

#### Search Agents by Capabilities
```http
GET /api/v1/agents/search?skills={skills}&tags={tags}&type={type}
```

**Response:**
```json
{
  "results": [
    {
      "agent_id": "agent_12345",
      "name": "Financial Data Standardization Agent",
      "description": "Standardizes financial entities...",
      "url": "https://standardization-agent.example.com",
      "version": "1.0.0",
      "skills": ["location-standardization", "account-standardization"],
      "status": "healthy",
      "last_seen": "2024-01-01T12:00:00Z",
      "response_time_ms": 150
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 20
}
```

#### Get Agent Details
```http
GET /api/v1/agents/{agent_id}
```

#### Find Agents for Workflow
```http
POST /api/v1/agents/match
Content-Type: application/json

{
  "workflow_requirements": [
    {
      "stage": "standardization",
      "required_skills": ["location-standardization", "account-standardization"],
      "input_modes": ["text/csv", "application/json"],
      "output_modes": ["application/json"]
    },
    {
      "stage": "vectorization", 
      "required_skills": ["vector-embedding-generation"],
      "input_modes": ["application/json"],
      "output_modes": ["application/json"]
    }
  ]
}
```

### Health Monitoring Endpoints

#### Get Agent Health Status
```http
GET /api/v1/agents/{agent_id}/health
```

**Response:**
```json
{
  "agent_id": "agent_12345",
  "status": "healthy",
  "last_health_check": "2024-01-01T12:00:00Z",
  "response_time_ms": 150,
  "health_details": {
    "service_status": "running",
    "memory_usage": "45%",
    "cpu_usage": "23%",
    "active_tasks": 3,
    "error_rate": "0.1%"
  },
  "capabilities_status": {
    "all_skills_available": true,
    "degraded_skills": [],
    "unavailable_skills": []
  }
}
```

#### Get System Health Overview
```http
GET /api/v1/system/health
```

#### Get Agent Metrics
```http
GET /api/v1/agents/{agent_id}/metrics?period={period}
```

### Orchestration Support Endpoints

#### Create Workflow Plan
```http
POST /api/v1/orchestration/plan
Content-Type: application/json

{
  "workflow_name": "financial_data_pipeline",
  "stages": [
    {
      "name": "data_product_creation",
      "required_capabilities": ["cds-schema-generation", "ord-descriptor-creation"]
    },
    {
      "name": "standardization",
      "required_capabilities": ["location-standardization", "account-standardization"],
      "depends_on": ["data_product_creation"]
    },
    {
      "name": "vectorization",
      "required_capabilities": ["vector-embedding-generation"],
      "depends_on": ["standardization"]
    }
  ]
}
```

**Response:**
```json
{
  "workflow_id": "workflow_67890",
  "execution_plan": [
    {
      "stage": "data_product_creation",
      "agent": {
        "agent_id": "agent_00001",
        "name": "Data Product Registration Agent",
        "url": "https://data-product-agent.example.com"
      }
    },
    {
      "stage": "standardization", 
      "agent": {
        "agent_id": "agent_12345",
        "name": "Financial Data Standardization Agent",
        "url": "https://standardization-agent.example.com"
      }
    },
    {
      "stage": "vectorization",
      "agent": {
        "agent_id": "agent_54321",
        "name": "AI Data Readiness Agent", 
        "url": "https://ai-readiness-agent.example.com"
      }
    }
  ],
  "estimated_duration": "5-10 minutes",
  "total_agents": 3
}
```

#### Execute Workflow
```http
POST /api/v1/orchestration/execute/{workflow_id}
Content-Type: application/json

{
  "input_data": {...},
  "context_id": "workflow_context_123",
  "execution_mode": "sequential"
}
```

---

## Data Model

### Agent Registration Record

```json
{
  "agent_id": "string",
  "agent_card": {
    "name": "string",
    "description": "string", 
    "url": "string",
    "version": "string",
    "protocolVersion": "string",
    "capabilities": {},
    "skills": [],
    "defaultInputModes": [],
    "defaultOutputModes": []
  },
  "registration_metadata": {
    "registered_by": "string",
    "registered_at": "datetime",
    "last_updated": "datetime",
    "status": "active|inactive|deprecated|retired"
  },
  "health_status": {
    "current_status": "healthy|degraded|unhealthy|unreachable",
    "last_health_check": "datetime",
    "response_time_ms": "number",
    "uptime_percentage": "number",
    "error_rate_percentage": "number"
  },
  "usage_analytics": {
    "total_invocations": "number",
    "successful_invocations": "number",
    "failed_invocations": "number",
    "average_response_time": "number",
    "last_invocation": "datetime"
  },
  "compatibility": {
    "protocol_versions": [],
    "supported_input_modes": [],
    "supported_output_modes": [],
    "dependency_requirements": []
  }
}
```

### Workflow Execution Record

```json
{
  "workflow_id": "string",
  "workflow_name": "string",
  "execution_id": "string", 
  "status": "planned|running|completed|failed|cancelled",
  "execution_plan": [],
  "current_stage": "string",
  "started_at": "datetime",
  "completed_at": "datetime",
  "duration_ms": "number",
  "context_id": "string",
  "input_data": {},
  "output_data": {},
  "stage_results": [],
  "error_details": {}
}
```

---

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `A2A_REGISTRY_PORT` | Service port | No | `8080` |
| `A2A_REGISTRY_DB_URL` | Database connection | Yes | - |
| `A2A_PROTOCOL_VERSION` | Supported A2A version | No | `0.2.9` |
| `HEALTH_CHECK_INTERVAL` | Health check frequency | No | `30s` |
| `AGENT_TIMEOUT` | Agent response timeout | No | `30s` |
| `AUTH_PROVIDER_URL` | Authentication service | Yes | - |
| `ORCHESTRATION_ENABLED` | Enable workflow orchestration | No | `true` |

### Database Schema

#### Tables Required

- `agent_registrations`: Agent registration records
- `agent_health`: Health monitoring data
- `agent_metrics`: Performance and usage metrics
- `workflow_plans`: Workflow execution plans
- `workflow_executions`: Workflow execution history
- `capability_index`: Searchable capability index

---

## Validation Rules

### Agent Card Validation

1. **A2A Protocol Compliance**: Validate against A2A specification
2. **URL Accessibility**: Verify agent endpoint is reachable
3. **Skill Definitions**: Validate skill structure and metadata
4. **Version Compatibility**: Check A2A protocol version support
5. **Capability Consistency**: Ensure capabilities match declared skills

### Health Check Validation

- **Response Time**: Must respond within configured timeout
- **Status Consistency**: Health status matches actual capabilities
- **Skill Availability**: All declared skills are functional
- **Protocol Compliance**: Responses follow A2A protocol

---

## Integration Patterns

### Agent Self-Registration

```javascript
// Agent startup registration
const registrationData = {
  ...agentCard,
  health_endpoint: `${agentUrl}/health`,
  metrics_endpoint: `${agentUrl}/metrics`
};

const response = await fetch(`${A2A_REGISTRY_URL}/api/v1/agents/register`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify(registrationData)
});
```

### Agent Discovery

```javascript
// Find agents for specific capabilities
const searchResponse = await fetch(
  `${A2A_REGISTRY_URL}/api/v1/agents/search?skills=location-standardization&status=healthy`
);

const availableAgents = await searchResponse.json();
```

### Workflow Orchestration Integration

```javascript
// Create and execute workflow
const workflowPlan = await createWorkflowPlan(requirements);
const executionResult = await executeWorkflow(workflowPlan.workflow_id, inputData);
```

---

## Security & Governance

### Authentication & Authorization

- **JWT Bearer Tokens**: Standard authentication for agent registration
- **Agent Authentication**: Mutual TLS for agent-to-registry communication
- **Role-Based Access**: Different permissions for agents vs. orchestrators
- **API Rate Limiting**: Prevent abuse of discovery endpoints

### Data Governance

- **Agent Audit Trail**: Complete registration and activity logging
- **Health Monitoring**: Continuous agent health and performance tracking
- **Capability Verification**: Regular validation of declared capabilities
- **Version Management**: Track agent versions and compatibility

### Monitoring & Analytics

- **Registration Metrics**: Track agent registration and deregistration
- **Health Metrics**: Monitor agent availability and performance
- **Usage Analytics**: Track agent invocation patterns
- **Workflow Metrics**: Monitor orchestration success rates

---

## Health Check Integration

### Agent Health Check Protocol

Agents must implement health check endpoint:

```http
GET /health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "capabilities": {
    "all_available": true,
    "degraded": [],
    "unavailable": []
  },
  "resources": {
    "memory_usage": "45%",
    "cpu_usage": "23%",
    "disk_usage": "12%"
  },
  "dependencies": {
    "database": "healthy",
    "external_services": "healthy"
  }
}
```

### Registry Health Monitoring

- **Periodic Health Checks**: Automated health verification
- **Status Aggregation**: Overall system health dashboard
- **Alert Generation**: Notifications for agent failures
- **Performance Tracking**: Response time and availability metrics

---

## Orchestration Support

### Workflow Definition

```yaml
workflow:
  name: "financial_data_pipeline"
  description: "End-to-end financial data processing"
  stages:
    - name: "data_product_creation"
      agent_requirements:
        skills: ["cds-schema-generation", "ord-descriptor-creation"]
        input_modes: ["text/csv", "application/json"]
        output_modes: ["application/json"]
    - name: "standardization" 
      depends_on: ["data_product_creation"]
      agent_requirements:
        skills: ["location-standardization", "account-standardization"]
        input_modes: ["application/json"]
        output_modes: ["application/json"]
    - name: "vectorization"
      depends_on: ["standardization"]
      agent_requirements:
        skills: ["vector-embedding-generation"]
        input_modes: ["application/json"]
        output_modes: ["application/json"]
```

### Execution Engine

- **Stage Dependency Resolution**: Automatic workflow ordering
- **Agent Selection**: Best-fit agent selection based on capabilities
- **Failure Recovery**: Retry logic and fallback agent selection
- **Context Propagation**: Maintain workflow context across agents

---

## API Response Formats

### Success Response Template
```json
{
  "status": "success",
  "data": {},
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345",
    "version": "1.0.0"
  }
}
```

### Error Response Template
```json
{
  "status": "error",
  "error": {
    "code": "AGENT_UNREACHABLE",
    "message": "Agent health check failed",
    "details": {
      "agent_id": "agent_12345",
      "last_successful_check": "2024-01-01T11:30:00Z"
    }
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345"
  }
}
```
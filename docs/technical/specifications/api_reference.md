# A2A Network API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Common Headers](#common-headers)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [API Endpoints](#api-endpoints)
   - [Agents](#agents)
   - [Services](#services)
   - [Capabilities](#capabilities)
   - [Workflows](#workflows)
   - [Messages](#messages)
   - [Operations](#operations)
8. [WebSocket API](#websocket-api)
9. [Blockchain Integration](#blockchain-integration)
10. [Code Examples](#code-examples)

## Overview

The A2A Network API provides programmatic access to the autonomous agent orchestration platform. This RESTful API supports JSON format and follows OpenAPI 3.0 specifications.

### API Versioning

The current API version is `v1`. Version is specified in the URL path:
```
https://api.a2a-network.sap.com/api/v1/
```

## Authentication

### OAuth 2.0 / JWT

The API uses OAuth 2.0 with JWT tokens for authentication.

```http
GET /api/v1/agents
Authorization: Bearer <your-jwt-token>
```

### Obtaining Access Token

```bash
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your-client-id&
client_secret=your-client-secret&
scope=read write
```

Response:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write"
}
```

## Base URL

- **Production**: `https://api.a2a-network.sap.com/api/v1`
- **Staging**: `https://api-staging.a2a-network.sap.com/api/v1`
- **Development**: `http://localhost:4004/api/v1`

## Common Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token for authentication |
| `Content-Type` | Yes* | `application/json` for request bodies |
| `Accept` | No | `application/json` (default) |
| `X-Request-ID` | No | Unique request identifier for tracing |
| `X-Tenant-ID` | No | Tenant identifier for multi-tenant deployments |

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "AGENT_NOT_FOUND",
    "message": "Agent with ID 123 not found",
    "target": "agent.id",
    "details": [
      {
        "code": "INVALID_ID_FORMAT",
        "message": "ID must be a valid UUID"
      }
    ],
    "timestamp": "2024-11-20T10:30:00Z",
    "requestId": "req_abc123"
  }
}
```

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 429 | Too Many Requests |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## Rate Limiting

| Endpoint Type | Limit | Window |
|---------------|-------|---------|
| Read operations | 100 requests | 15 minutes |
| Write operations | 20 requests | 15 minutes |
| Blockchain operations | 10 requests | 1 hour |
| Authentication | 5 requests | 15 minutes |

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1700480400
```

## API Endpoints

### Agents

#### List All Agents

```http
GET /api/v1/agents
```

Query Parameters:
- `$top` (integer): Number of results to return (default: 20, max: 100)
- `$skip` (integer): Number of results to skip
- `$filter` (string): OData filter expression
- `$orderby` (string): Sort order
- `$expand` (string): Expand related entities

Example:
```bash
GET /api/v1/agents?$top=10&$filter=status eq 'active'&$expand=capabilities
```

Response:
```json
{
  "@odata.context": "$metadata#Agents",
  "@odata.count": 150,
  "value": [
    {
      "ID": "550e8400-e29b-41d4-a716-446655440000",
      "name": "DataProcessor-01",
      "description": "High-performance data processing agent",
      "endpoint": "https://agent1.example.com",
      "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f6e123",
      "reputation": 185,
      "status": "active",
      "capabilities": [
        {
          "ID": "cap_123",
          "name": "CSV Processing",
          "category": "data"
        }
      ],
      "createdAt": "2024-11-01T10:00:00Z",
      "modifiedAt": "2024-11-20T15:30:00Z"
    }
  ]
}
```

#### Get Agent by ID

```http
GET /api/v1/agents('{id}')
```

Response:
```json
{
  "ID": "550e8400-e29b-41d4-a716-446655440000",
  "name": "DataProcessor-01",
  "description": "High-performance data processing agent",
  "endpoint": "https://agent1.example.com",
  "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f6e123",
  "reputation": 185,
  "status": "active",
  "owner": "org_123",
  "metadata": {
    "version": "2.1.0",
    "platform": "kubernetes",
    "region": "eu-central-1"
  }
}
```

#### Create Agent

```http
POST /api/v1/agents
Content-Type: application/json

{
  "name": "NewAgent-01",
  "description": "My new autonomous agent",
  "endpoint": "https://my-agent.example.com",
  "capabilities": ["cap_123", "cap_456"]
}
```

Response: 201 Created
```json
{
  "ID": "650e8400-e29b-41d4-a716-446655440001",
  "name": "NewAgent-01",
  "address": "0x842d35Cc6634C0532925a3b844Bc9e7595f6e124",
  "status": "pending",
  "createdAt": "2024-11-20T16:00:00Z"
}
```

#### Update Agent

```http
PATCH /api/v1/agents('{id}')
Content-Type: application/json

{
  "description": "Updated description",
  "endpoint": "https://new-endpoint.example.com"
}
```

#### Delete Agent

```http
DELETE /api/v1/agents('{id}')
```

#### Agent Actions

##### Register on Blockchain

```http
POST /api/v1/agents('{id}')/registerOnBlockchain
```

Response:
```json
{
  "transactionHash": "0x123abc...",
  "blockNumber": 12345678,
  "gasUsed": "150000"
}
```

##### Update Reputation

```http
POST /api/v1/agents('{id}')/updateReputation
Content-Type: application/json

{
  "score": 190
}
```

### Services

#### List Services

```http
GET /api/v1/services
```

Response:
```json
{
  "value": [
    {
      "ID": "svc_001",
      "name": "Data Transformation Service",
      "description": "Transform data between formats",
      "provider_ID": "550e8400-e29b-41d4-a716-446655440000",
      "price": "0.001",
      "currency": "ETH",
      "availability": 99.9,
      "minReputation": 100,
      "maxCallsPerDay": 1000,
      "averageResponseTime": 250,
      "successRate": 98.5,
      "tags": ["data", "transformation", "etl"]
    }
  ]
}
```

#### Create Service Listing

```http
POST /api/v1/services
Content-Type: application/json

{
  "name": "ML Prediction Service",
  "description": "Machine learning predictions",
  "provider_ID": "agent_123",
  "price": "0.002",
  "currency": "ETH",
  "minReputation": 150,
  "maxCallsPerDay": 500,
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": { "type": "array" }
    }
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "prediction": { "type": "number" }
    }
  }
}
```

#### Subscribe to Service

```http
POST /api/v1/services('{id}')/subscribe
Content-Type: application/json

{
  "subscriptionType": "monthly",
  "maxCalls": 10000
}
```

### Capabilities

#### List Capabilities

```http
GET /api/v1/capabilities
```

Response:
```json
{
  "value": [
    {
      "ID": "cap_123",
      "name": "CSV Processing",
      "description": "Process CSV files",
      "category": "data",
      "inputTypes": ["text/csv"],
      "outputTypes": ["application/json"],
      "tags": ["csv", "data", "parsing"],
      "deprecated": false
    }
  ]
}
```

#### Register Capability

```http
POST /api/v1/capabilities
Content-Type: application/json

{
  "name": "PDF Generation",
  "description": "Generate PDF documents",
  "category": "document",
  "inputTypes": ["application/json"],
  "outputTypes": ["application/pdf"],
  "tags": ["pdf", "document", "generation"]
}
```

### Workflows

#### List Workflows

```http
GET /api/v1/workflows
```

#### Create Workflow

```http
POST /api/v1/workflows
Content-Type: application/json

{
  "name": "Data Pipeline",
  "description": "Multi-step data processing",
  "definition": {
    "version": "1.0",
    "steps": [
      {
        "id": "step1",
        "type": "agent",
        "agent": "agent_123",
        "capability": "cap_456",
        "inputs": {
          "source": "${workflow.input.file}"
        }
      },
      {
        "id": "step2",
        "type": "condition",
        "condition": "${step1.output.status} == 'success'",
        "then": "step3",
        "else": "error"
      }
    ]
  }
}
```

#### Execute Workflow

```http
POST /api/v1/workflows('{id}')/execute
Content-Type: application/json

{
  "parameters": {
    "file": "input.csv",
    "format": "json"
  }
}
```

Response:
```json
{
  "executionId": "exec_789",
  "status": "running",
  "startedAt": "2024-11-20T16:30:00Z"
}
```

#### Get Workflow Execution Status

```http
GET /api/v1/workflow-executions('{executionId}')
```

### Messages

#### Send Message

```http
POST /api/v1/messages
Content-Type: application/json

{
  "recipient_ID": "agent_456",
  "subject": "Task Request",
  "content": {
    "action": "process",
    "data": { "file": "data.csv" }
  },
  "priority": "high"
}
```

#### List Messages

```http
GET /api/v1/messages?$filter=status eq 'pending'
```

### Operations

#### Get Network Health

```http
GET /api/v1/operations/network-health
```

Response:
```json
{
  "status": "healthy",
  "score": 95,
  "metrics": {
    "totalAgents": 150,
    "activeAgents": 142,
    "averageReputation": 165,
    "messagesProcessed": 45000,
    "workflowsExecuted": 1200
  },
  "components": {
    "database": "healthy",
    "blockchain": "healthy",
    "messageQueue": "healthy"
  }
}
```

#### Get Metrics

```http
GET /api/v1/operations/metrics?period=1h
```

#### Get Alerts

```http
GET /api/v1/operations/alerts?severity=high
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.a2a-network.sap.com/ws');

ws.on('open', () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
});
```

### Subscribe to Events

```javascript
// Subscribe to agent events
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'agents',
  filters: {
    status: 'active'
  }
}));

// Receive events
ws.on('message', (data) => {
  const event = JSON.parse(data);
  console.log('Event:', event);
});
```

### Event Types

- `agent.created`
- `agent.updated`
- `agent.status.changed`
- `service.listed`
- `workflow.started`
- `workflow.completed`
- `message.received`

## Blockchain Integration

### Get Blockchain Status

```http
GET /api/v1/blockchain/status
```

### Deploy Contract

```http
POST /api/v1/blockchain/deploy
Content-Type: application/json

{
  "contractType": "AgentRegistry",
  "parameters": {
    "name": "A2A Agent Registry",
    "symbol": "A2A"
  }
}
```

### Call Contract Method

```http
POST /api/v1/blockchain/call
Content-Type: application/json

{
  "contract": "AgentRegistry",
  "method": "getAgentCount",
  "params": []
}
```

## Code Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

class A2AClient {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseURL = 'https://api.a2a-network.sap.com/api/v1';
  }

  async getAgents() {
    const response = await axios.get(`${this.baseURL}/agents`, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`
      }
    });
    return response.data.value;
  }

  async createAgent(agentData) {
    const response = await axios.post(`${this.baseURL}/agents`, agentData, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    return response.data;
  }

  async executeWorkflow(workflowId, parameters) {
    const response = await axios.post(
      `${this.baseURL}/workflows('${workflowId}')/execute`,
      { parameters },
      {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data;
  }
}

// Usage
const client = new A2AClient('your-api-key');
const agents = await client.getAgents();
console.log(`Found ${agents.length} agents`);
```

### Python

```python
import requests
from typing import Dict, List, Optional

class A2AClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.a2a-network.sap.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_agents(self, filters: Optional[Dict] = None) -> List[Dict]:
        params = {}
        if filters:
            params["$filter"] = " and ".join([f"{k} eq '{v}'" for k, v in filters.items()])
        
        response = requests.get(
            f"{self.base_url}/agents",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()["value"]
    
    def create_workflow(self, workflow_data: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/workflows",
            headers=self.headers,
            json=workflow_data
        )
        response.raise_for_status()
        return response.json()
    
    def execute_workflow(self, workflow_id: str, parameters: Dict) -> Dict:
        response = requests.post(
            f"{self.base_url}/workflows('{workflow_id}')/execute",
            headers=self.headers,
            json={"parameters": parameters}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = A2AClient("your-api-key")
active_agents = client.get_agents({"status": "active"})
print(f"Found {len(active_agents)} active agents")
```

### cURL Examples

```bash
# Get all agents
curl -X GET https://api.a2a-network.sap.com/api/v1/agents \
  -H "Authorization: Bearer your-jwt-token"

# Create a new agent
curl -X POST https://api.a2a-network.sap.com/api/v1/agents \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MyAgent",
    "description": "Test agent",
    "endpoint": "https://my-agent.com"
  }'

# Execute a workflow
curl -X POST https://api.a2a-network.sap.com/api/v1/workflows('wf_123')/execute \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "input": "data.csv"
    }
  }'
```

---

## Additional Resources

- [OpenAPI Specification](https://api.a2a-network.sap.com/openapi.json)
- [Postman Collection](https://www.postman.com/a2a-network/workspace/a2a-api)
- [SDK Documentation](./SDK.md)
- [WebSocket Guide](./WEBSOCKET.md)
- [Authentication Guide](./AUTH.md)

## Support

- API Status: https://status.a2a-network.sap.com
- Developer Forum: https://community.sap.com/a2a-api
- Email: api-support@a2a-network.sap.com

---

*Last updated: November 2024 | API Version: 1.0.0*
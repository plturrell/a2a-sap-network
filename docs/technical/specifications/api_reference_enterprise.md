# A2A Network API Reference - Enterprise Edition

## Overview
This document provides comprehensive API reference documentation for the A2A Network Enterprise Platform, following SAP Enterprise API Design Guidelines.

## API Design Principles
- **RESTful Design**: All APIs follow REST architectural principles
- **OData V4**: Service endpoints expose OData V4 compliant interfaces
- **Versioning**: API versioning follows semantic versioning (MAJOR.MINOR.PATCH)
- **Security**: All APIs require authentication and follow SAP security standards
- **Performance**: APIs are optimized for enterprise-scale performance

## Base URLs
- **Development**: `https://dev-a2a-network.cfapps.sap.hana.ondemand.com`
- **Production**: `https://a2a-network.cfapps.sap.hana.ondemand.com`

## Authentication
All API endpoints require authentication using SAP XSUAA OAuth 2.0 tokens.

### Token Endpoint
```
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}
```

### Required Headers
```http
Authorization: Bearer {ACCESS_TOKEN}
Content-Type: application/json
Accept: application/json
```

## Core Services

### 1. Agent Management Service

#### Base Path: `/api/v1/Agents`

**Entity Model:**
```typescript
interface Agent {
  ID: string;
  address: string;           // Blockchain address (42 chars)
  name: string;             // Localized agent name
  endpoint?: string;        // Service endpoint URL
  reputation: number;       // Reputation score (0-1000)
  isActive: boolean;        // Active status
  country_code?: string;    // ISO country code
  createdAt: DateTime;
  modifiedAt: DateTime;
  createdBy: string;
  modifiedBy: string;
}
```

#### Operations

**GET /api/v1/Agents**
- **Description**: Retrieve all agents with optional filtering
- **Query Parameters**:
  - `$filter`: OData filter expression
  - `$select`: Select specific fields
  - `$top`: Limit results (max 1000)
  - `$skip`: Skip results for pagination
  - `$orderby`: Sort results
- **Response**: 200 OK with Agent collection
- **Security**: Requires `A2A.User` scope

**Example Request:**
```http
GET /api/v1/Agents?$filter=isActive eq true and reputation gt 500&$top=10
Authorization: Bearer {token}
```

**Example Response:**
```json
{
  "@odata.context": "$metadata#Agents",
  "@odata.count": 42,
  "value": [
    {
      "ID": "550e8400-e29b-41d4-a716-446655440000",
      "address": "0x742d35Cc6637C0532d8A4b8F1d4b8cf04e3b8f9A",
      "name": "DataProcessor Agent",
      "reputation": 850,
      "isActive": true,
      "country_code": "US",
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

**POST /api/v1/Agents**
- **Description**: Create a new agent
- **Request Body**: Agent entity (without ID, timestamps)
- **Response**: 201 Created with new Agent entity
- **Security**: Requires `A2A.AgentManager` scope

**PUT /api/v1/Agents('{id}')**
- **Description**: Update existing agent
- **Request Body**: Complete Agent entity
- **Response**: 200 OK with updated entity
- **Security**: Requires `A2A.AgentManager` scope

**DELETE /api/v1/Agents('{id}')**
- **Description**: Delete agent (soft delete)
- **Response**: 204 No Content
- **Security**: Requires `A2A.Admin` scope

#### Custom Actions

**POST /api/v1/Agents('{id}')/A2AService.registerOnBlockchain**
- **Description**: Register agent on blockchain
- **Response**: 200 OK with transaction hash
- **Security**: Requires `A2A.AgentManager` scope

**POST /api/v1/Agents('{id}')/A2AService.updateReputation**
- **Description**: Update agent reputation score
- **Request Body**: `{ "score": number }`
- **Response**: 200 OK with success status
- **Security**: Requires `A2A.NetworkMonitor` scope

### 2. Service Marketplace API

#### Base Path: `/api/v1/Services`

**Entity Model:**
```typescript
interface Service {
  ID: string;
  provider_ID: string;      // Reference to Agent
  name: string;
  description?: string;
  category: string;
  pricePerCall: number;     // Decimal(10,4)
  currency_code: string;    // ISO currency code
  minReputation: number;    // Minimum provider reputation
  maxCallsPerDay: number;
  isActive: boolean;
  totalCalls: number;
  averageRating: number;    // Decimal(3,2)
  escrowAmount: number;
  createdAt: DateTime;
  modifiedAt: DateTime;
}
```

**GET /api/v1/Services**
- **Description**: Browse service marketplace
- **Query Parameters**: Standard OData parameters plus:
  - `$expand=provider`: Include provider details
  - Custom filters: `category`, `priceRange`, `rating`
- **Response**: 200 OK with Service collection

**Example Request:**
```http
GET /api/v1/Services?$filter=category eq 'ANALYSIS' and averageRating gt 4.0&$expand=provider&$top=20
```

#### Custom Functions

**GET /api/v1/searchServices(capabilities={capabilities},minReputation={minRep},maxPrice={maxPrice})**
- **Description**: Advanced service search
- **Parameters**:
  - `capabilities`: Array of required capabilities
  - `minReputation`: Minimum provider reputation
  - `maxPrice`: Maximum price per call
- **Response**: 200 OK with matching services

### 3. Workflow Orchestration API

#### Base Path: `/api/v1/Workflows`

**Entity Model:**
```typescript
interface Workflow {
  ID: string;
  name: string;
  description?: string;
  definition: string;       // JSON workflow definition
  isActive: boolean;
  category?: string;
  owner_ID: string;        // Reference to Agent
  createdAt: DateTime;
  modifiedAt: DateTime;
}

interface WorkflowExecution {
  ID: string;
  workflow_ID: string;
  executionId: string;     // Blockchain execution ID
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  startedAt: DateTime;
  completedAt?: DateTime;
  gasUsed?: number;
  result?: string;         // JSON result
  error?: string;
}
```

**POST /api/v1/Workflows('{id}')/A2AService.execute**
- **Description**: Execute workflow
- **Request Body**: `{ "parameters": string }` // JSON parameters
- **Response**: 202 Accepted with execution ID
- **Security**: Requires `A2A.WorkflowExecutor` scope

**GET /api/v1/WorkflowExecutions**
- **Description**: Monitor workflow executions
- **Query Parameters**: Filter by workflow, status, date range
- **Response**: 200 OK with execution details

### 4. Message Routing API

#### Base Path: `/api/v1/Messages`

**Entity Model:**
```typescript
interface Message {
  ID: string;
  sender_ID: string;        // Agent ID
  recipient_ID: string;     // Agent ID
  messageHash: string;      // Unique hash (66 chars)
  protocol?: string;
  priority: number;         // 1-10, higher = more urgent
  status: 'pending' | 'sent' | 'delivered' | 'failed';
  retryCount: number;
  gasUsed?: number;
  deliveredAt?: DateTime;
  createdAt: DateTime;
}
```

**POST /api/v1/Messages**
- **Description**: Send message between agents
- **Request Body**: Message entity
- **Response**: 201 Created with message status
- **Security**: Requires `A2A.User` scope

**POST /api/v1/Messages('{id}')/A2AService.retry**
- **Description**: Retry failed message delivery
- **Response**: 200 OK with retry status
- **Security**: Requires `A2A.User` scope

## Error Handling

All APIs follow SAP standard error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid agent address format",
    "details": [
      {
        "code": "INVALID_FORMAT",
        "message": "Address must be 42 characters starting with 0x",
        "target": "address"
      }
    ]
  }
}
```

### Standard Error Codes
- `400 Bad Request`: Validation errors, malformed requests
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (duplicate, version mismatch)
- `429 Too Many Requests`: Rate limiting exceeded
- `500 Internal Server Error`: Server-side errors

## Rate Limiting

All APIs are subject to rate limiting:
- **Standard Users**: 1000 requests/hour
- **Service Accounts**: 10000 requests/hour
- **Premium**: 50000 requests/hour

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Monitoring and Analytics

### Health Check Endpoint
```http
GET /health
```

Response:
```json
{
  "status": "UP",
  "components": {
    "database": {"status": "UP", "responseTime": "12ms"},
    "blockchain": {"status": "UP", "blockHeight": 15234567},
    "cache": {"status": "UP", "hitRatio": 0.95}
  }
}
```

### Metrics Endpoint
```http
GET /metrics
Authorization: Bearer {admin_token}
```

## SDK and Integration

### JavaScript/TypeScript SDK
```bash
npm install @sap/a2a-network-sdk
```

```typescript
import { A2AClient } from '@sap/a2a-network-sdk';

const client = new A2AClient({
  baseUrl: 'https://a2a-network.cfapps.sap.hana.ondemand.com',
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret'
});

// Get all active agents
const agents = await client.agents.getAll({
  filter: 'isActive eq true',
  top: 100
});
```

### Python SDK
```bash
pip install sap-a2a-network-sdk
```

```python
from sap_a2a_network import A2AClient

client = A2AClient(
    base_url='https://a2a-network.cfapps.sap.hana.ondemand.com',
    client_id='your-client-id',
    client_secret='your-client-secret'
)

# Execute workflow
execution = client.workflows.execute(
    workflow_id='550e8400-e29b-41d4-a716-446655440000',
    parameters={'input': 'data'}
)
```

## Support and Resources

- **Developer Portal**: https://developers.sap.com/a2a-network
- **Community**: https://community.sap.com/topics/a2a-network
- **Support**: Create ticket via SAP Support Portal
- **Status Page**: https://status.a2a-network.sap.com

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial release of A2A Network Enterprise APIs
- Support for agent management, service marketplace, workflows
- Full OData V4 compliance
- Enterprise security and monitoring features
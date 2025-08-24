# A2A Platform API Documentation
*Enhanced API Coverage for Enterprise Integration*

## Overview

The A2A (Agent-to-Agent) Platform provides a comprehensive REST API for managing intelligent data processing workflows through autonomous agents. This documentation covers all endpoints, authentication, error handling, and integration patterns.

## Base URLs

- **Development**: `http://localhost:8000`
- **Staging**: `https://staging-api.a2a-platform.com`
- **Production**: `https://api.a2a-platform.com`

## Authentication

### JWT Bearer Token Authentication

All API endpoints (except public health checks) require JWT bearer token authentication.

```http
Authorization: Bearer <jwt_access_token>
```

### Token Management

#### Get Access Token
```http
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## Agent Management API

### List All Agents
```http
GET /api/v1/agents
Authorization: Bearer <token>
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "agent0_data_product",
      "name": "Data Product Agent",
      "status": "active",
      "specialties": ["data_processing", "ord_registration", "dublin_core"],
      "url": "http://localhost:8002",
      "health_status": "healthy",
      "last_heartbeat": "2025-01-08T10:30:00Z",
      "performance_metrics": {
        "success_rate": 98.5,
        "avg_response_time": 245,
        "total_requests": 1540
      }
    }
  ]
}
```

### Get Agent Details
```http
GET /api/v1/agents/{agent_id}
Authorization: Bearer <token>
```

### Update Agent Configuration
```http
PUT /api/v1/agents/{agent_id}/config
Authorization: Bearer <token>
Content-Type: application/json

{
  "timeout_seconds": 30,
  "max_retries": 3,
  "specialties": ["data_processing", "enhanced_analytics"],
  "enabled": true
}
```

## Data Processing API

### Submit Data Processing Request
```http
POST /api/v1/process
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": {
    "source": "ord_registry",
    "format": "json",
    "content": { /* data payload */ }
  },
  "processing_options": {
    "agents": ["agent0", "agent1"],
    "priority": "high",
    "callback_url": "https://your-app.com/webhook"
  }
}
```

**Response:**
```json
{
  "request_id": "req_001_20250108_103045",
  "status": "accepted",
  "estimated_completion": "2025-01-08T10:35:00Z",
  "tracking_url": "/api/v1/requests/req_001_20250108_103045"
}
```

### Get Processing Status
```http
GET /api/v1/requests/{request_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "request_id": "req_001_20250108_103045",
  "status": "processing",
  "progress": {
    "completed_steps": 3,
    "total_steps": 7,
    "current_agent": "agent1_standardization",
    "percentage": 42.8
  },
  "results": {
    "agent0_data_product": {
      "status": "completed",
      "output": { /* processed data */ },
      "duration": 2340,
      "confidence": 0.95
    },
    "agent1_standardization": {
      "status": "in_progress",
      "started_at": "2025-01-08T10:32:15Z"
    }
  }
}
```

## Agent Communication API

### Send Inter-Agent Message
```http
POST /api/v1/agents/{source_agent}/messages
Authorization: Bearer <token>
Content-Type: application/json

{
  "target_agent": "agent1_standardization",
  "message_type": "help_request",
  "content": {
    "problem_type": "data_validation",
    "description": "Unable to validate financial instrument data",
    "context": {
      "data_format": "crd_xml",
      "error_code": "VALIDATION_001"
    }
  },
  "priority": "high",
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "message_id": "msg_a0_a1_20250108_103055",
  "status": "sent",
  "delivery_confirmation": true,
  "expected_response_time": 15
}
```

### Get Agent Help Response
```http
GET /api/v1/messages/{message_id}
Authorization: Bearer <token>
```

## Data Management API

### Store Data
```http
POST /api/v1/data
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": {
    "type": "financial_instrument",
    "source": "manual_entry",
    "content": { /* data object */ }
  },
  "storage_options": {
    "database": "hana",
    "fallback": "sqlite",
    "retention_days": 2555,
    "encryption": true
  }
}
```

### Retrieve Data
```http
GET /api/v1/data/{data_id}
Authorization: Bearer <token>
```

### Query Data
```http
POST /api/v1/data/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": {
    "type": "financial_instrument",
    "filters": {
      "created_date": {
        "gte": "2024-01-01",
        "lt": "2025-01-01"
      },
      "status": "active"
    },
    "sort": [{"field": "created_date", "order": "desc"}],
    "limit": 100
  }
}
```

## AI Enhancement API

### Request AI Analysis
```http
POST /api/v1/ai/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": { /* data to analyze */ },
  "analysis_type": "quality_assessment",
  "model_preferences": {
    "provider": "grok",
    "temperature": 0.2,
    "max_tokens": 2000
  },
  "context": {
    "domain": "financial_services",
    "use_case": "regulatory_compliance"
  }
}
```

**Response:**
```json
{
  "analysis_id": "ai_analysis_20250108_103100",
  "status": "completed",
  "results": {
    "quality_score": 0.87,
    "issues_found": [
      {
        "type": "missing_field",
        "field": "issuer_identifier",
        "severity": "medium",
        "recommendation": "Add ISIN or CUSIP identifier"
      }
    ],
    "enhancements": {
      "suggested_fields": ["maturity_date", "credit_rating"],
      "confidence": 0.92
    }
  },
  "model_info": {
    "provider": "grok-beta",
    "version": "1.0",
    "processing_time": 1.2
  }
}
```

## Monitoring & Analytics API

### Get System Health
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-08T10:31:30Z",
  "services": {
    "database": {
      "hana": "healthy",
      "sqlite": "healthy"
    },
    "agents": {
      "total": 9,
      "healthy": 9,
      "degraded": 0,
      "down": 0
    },
    "external_services": {
      "grok_api": "healthy",
      "sap_graph": "healthy",
      "blockchain": "healthy"
    }
  },
  "performance": {
    "avg_response_time": 127,
    "requests_per_minute": 245,
    "success_rate": 99.2
  }
}
```

### Get Performance Metrics
```http
GET /api/v1/metrics
Authorization: Bearer <token>
```

### Get Audit Logs
```http
GET /api/v1/audit
Authorization: Bearer <token>
Query Parameters:
  - start_date: ISO 8601 date
  - end_date: ISO 8601 date
  - event_type: string
  - user_id: string
  - limit: integer (max 1000)
```

## SAP Integration API

### Execute SAP Graph Query
```http
POST /api/v1/sap/graph/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "SELECT * FROM BusinessPartner WHERE CompanyCode = 'US01'",
  "parameters": {
    "CompanyCode": "US01"
  },
  "options": {
    "timeout": 30,
    "format": "json"
  }
}
```

### Update SAP HANA Data
```http
PUT /api/v1/sap/hana/tables/{table_name}
Authorization: Bearer <token>
Content-Type: application/json

{
  "data": [
    {
      "id": "record_001",
      "fields": { /* record data */ }
    }
  ],
  "options": {
    "upsert": true,
    "batch_size": 1000
  }
}
```

## Error Handling

### Standard Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "A2A_VALIDATION_ERROR",
    "message": "Validation failed for request parameters",
    "category": "validation",
    "severity": "medium",
    "timestamp": "2025-01-08T10:31:45Z",
    "details": {
      "field_errors": {
        "email": ["Invalid email format"],
        "password": ["Password too weak"]
      }
    },
    "request_id": "req_20250108_103145",
    "documentation_url": "https://docs.a2a-platform.com/errors/validation"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `A2A_VALIDATION_ERROR` | Request validation failed |
| 401 | `A2A_AUTHENTICATION_ERROR` | Authentication required |
| 401 | `A2A_TOKEN_EXPIRED` | Access token has expired |
| 403 | `A2A_AUTHORIZATION_ERROR` | Insufficient permissions |
| 404 | `A2A_RESOURCE_NOT_FOUND` | Requested resource not found |
| 409 | `A2A_RESOURCE_CONFLICT` | Resource conflict (duplicate) |
| 429 | `A2A_RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `A2A_INTERNAL_ERROR` | Internal server error |
| 502 | `A2A_EXTERNAL_SERVICE_ERROR` | External service unavailable |
| 504 | `A2A_TIMEOUT_ERROR` | Request timeout |

### Error Recovery Patterns

#### Retry Strategy for Transient Errors
```javascript
async function apiCallWithRetry(endpoint, options, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(endpoint, options);
      
      if (response.status === 429) {
        // Rate limited - wait before retry
        const retryAfter = response.headers.get('Retry-After') || 60;
        await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
        continue;
      }
      
      if (response.status >= 500) {
        // Server error - exponential backoff
        if (attempt < maxRetries) {
          const backoffTime = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, backoffTime));
          continue;
        }
      }
      
      return response;
    } catch (error) {
      if (attempt === maxRetries) throw error;
    }
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Standard Users**: 1000 requests/hour, 60 requests/minute
- **Premium Users**: 5000 requests/hour, 300 requests/minute
- **System Integrations**: 10000 requests/hour, 600 requests/minute

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641648000
Retry-After: 60
```

## Webhooks

### Webhook Registration
```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/a2a",
  "events": ["processing_completed", "agent_status_changed"],
  "secret": "webhook_secret_key",
  "active": true
}
```

### Webhook Payload Example
```json
{
  "event_type": "processing_completed",
  "timestamp": "2025-01-08T10:35:00Z",
  "request_id": "req_001_20250108_103045",
  "data": {
    "status": "completed",
    "results": { /* processing results */ },
    "duration": 4500,
    "agents_used": ["agent0", "agent1", "agent2"]
  },
  "signature": "sha256=5d41402abc4b2a76b9719d911017c592"
}
```

## SDK Examples

### Python SDK Usage
```python
from a2a_sdk import A2AClient

# Initialize client
client = A2AClient(
    base_url="https://api.a2a-platform.com",
    access_token="your_access_token"
)

# Process data
response = await client.process_data({
    "source": "ord_registry",
    "content": data_payload
}, agents=["agent0", "agent1"])

print(f"Request ID: {response.request_id}")

# Check status
status = await client.get_request_status(response.request_id)
print(f"Progress: {status.progress.percentage}%")
```

### Node.js SDK Usage
```javascript
const { A2AClient } = require('@a2a-platform/sdk');

const client = new A2AClient({
  baseUrl: 'https://api.a2a-platform.com',
  accessToken: 'your_access_token'
});

// Process data
const response = await client.processData({
  source: 'ord_registry',
  content: dataPayload
}, { agents: ['agent0', 'agent1'] });

console.log(`Request ID: ${response.requestId}`);
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Development**: `http://localhost:8000/docs`
- **Production**: `https://api.a2a-platform.com/docs`

## Support & Resources

- **Documentation**: https://docs.a2a-platform.com
- **API Status**: https://status.a2a-platform.com
- **Support**: support@a2a-platform.com
- **GitHub**: https://github.com/a2a-platform/api
- **Discord Community**: https://discord.gg/a2a-platform

---

*This documentation is automatically updated with each API release. Last updated: January 8, 2025*
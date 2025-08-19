# A2A Platform API Documentation

## Overview

The A2A (Agent-to-Agent) Platform provides a comprehensive REST API for managing AI agents, workflows, and data processing pipelines. Built on SAP Business Technology Platform with enterprise-grade security and multi-tenant support.

## Quick Start

### 1. Authentication

All API requests require OAuth2 Bearer token authentication via SAP XSUAA:

```bash
# Get access token
curl -X POST "https://your-tenant.authentication.sap.hana.ondemand.com/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"
```

### 2. Making API Calls

Include the Bearer token in all API requests:

```bash
curl -X GET "https://api.a2a-platform.com/v1/agents" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json"
```

### 3. Basic Examples

#### List All Agents
```bash
curl -X GET "https://api.a2a-platform.com/v1/agents?limit=10&status=active" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Create a New Agent
```bash
curl -X POST "https://api.a2a-platform.com/v1/agents" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Custom Data Processing Agent",
    "type": "data_product",
    "description": "Specialized agent for custom data processing workflows",
    "capabilities": [
      {
        "id": "data_validation",
        "name": "Data Validation",
        "type": "validation",
        "enabled": true
      }
    ]
  }'
```

#### Execute an Agent
```bash
curl -X POST "https://api.a2a-platform.com/v1/agents/550e8400-e29b-41d4-a716-446655440000/execute" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "inputData": "sample data",
      "processingMode": "standard"
    },
    "async": false,
    "timeout": 600
  }'
```

## API Reference

### Base URLs

- **Production**: `https://api.a2a-platform.com/v1`
- **Staging**: `https://staging-api.a2a-platform.com/v1`
- **Local Development**: `http://localhost:4004/api/v1`

### OpenAPI Specification

The complete API specification is available in OpenAPI 3.0 format:
- [OpenAPI YAML](./openapi.yaml)
- [Interactive Documentation](https://docs.a2a-platform.com/api)

## Authentication & Authorization

### OAuth2 Scopes

The API uses role-based access control with the following scopes:

| Scope | Description | Required For |
|-------|-------------|--------------|
| `a2a-agents.Admin` | Full administrative access | All admin operations |
| `a2a-agents.AgentOperator` | Operate agents and workflows | Agent/workflow management |
| `a2a-agents.DataViewer` | View data and reports | Read-only operations |
| `a2a-agents.Agent0.Execute` | Execute Data Product Agent | Agent 0 execution |
| `a2a-agents.Agent1.Execute` | Execute Standardization Agent | Agent 1 execution |
| `a2a-agents.Agent2.Execute` | Execute AI Preparation Agent | Agent 2 execution |
| `a2a-agents.Agent3.Execute` | Execute Vector Processing Agent | Agent 3 execution |
| `a2a-agents.Agent4.Execute` | Execute Calculation Agent | Agent 4 execution |
| `a2a-agents.Agent5.Execute` | Execute Quality Assurance Agent | Agent 5 execution |

### Multi-Tenant Support

The API supports multi-tenant architecture:
- Each tenant has isolated data and configurations
- Tenant context is automatically determined from the OAuth2 token
- Cross-tenant access is strictly prohibited

## Rate Limiting

API requests are rate-limited based on user tier:

| User Tier | Requests per Hour | Burst Limit |
|-----------|-------------------|-------------|
| Standard | 1,000 | 100 |
| Premium | 10,000 | 500 |
| Enterprise | Unlimited | 1,000 |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per hour
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets (Unix timestamp)

## Error Handling

### Standard Error Response

All errors follow a consistent format:

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error description",
  "details": {
    "field": "fieldName",
    "issue": "Specific validation issue"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "requestId": "req_123456789"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Request validation failed |
| 401 | `UNAUTHORIZED` | Authentication required |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `CONFLICT` | Resource conflict |
| 429 | `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| 500 | `INTERNAL_ERROR` | Internal server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## Pagination

List endpoints support cursor-based pagination:

### Request Parameters
- `limit`: Number of items per page (1-100, default: 20)
- `offset`: Number of items to skip (default: 0)

### Response Format
```json
{
  "data": [...],
  "total": 150,
  "limit": 20,
  "offset": 40,
  "hasMore": true
}
```

## Filtering and Sorting

### Filtering
Most list endpoints support filtering via query parameters:
```bash
GET /agents?status=active&type=data_product
```

### Sorting
Use the `sort` parameter with field names:
```bash
GET /agents?sort=createdAt&order=desc
```

## Webhooks

The platform supports webhooks for real-time notifications:

### Supported Events
- `agent.created`
- `agent.updated`
- `agent.deleted`
- `agent.executed`
- `workflow.started`
- `workflow.completed`
- `workflow.failed`

### Webhook Configuration
Configure webhooks via the API or platform UI:
```bash
curl -X POST "https://api.a2a-platform.com/v1/webhooks" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "url": "https://your-app.com/webhooks/a2a",
    "events": ["agent.executed", "workflow.completed"],
    "secret": "your-webhook-secret"
  }'
```

## SDKs and Libraries

### Official SDKs
- [JavaScript/Node.js SDK](https://github.com/a2a-platform/sdk-javascript)
- [Python SDK](https://github.com/a2a-platform/sdk-python)
- [Java SDK](https://github.com/a2a-platform/sdk-java)

### Community SDKs
- [Go SDK](https://github.com/community/a2a-go-sdk)
- [.NET SDK](https://github.com/community/a2a-dotnet-sdk)

## Testing

### Postman Collection
Import our Postman collection for easy API testing:
- [Download Collection](https://docs.a2a-platform.com/postman/collection.json)

### Test Environment
Use our sandbox environment for testing:
- Base URL: `https://sandbox-api.a2a-platform.com/v1`
- Test credentials provided upon request

## Support and Resources

### Documentation
- [Platform Documentation](https://docs.a2a-platform.com)
- [API Reference](https://docs.a2a-platform.com/api)
- [Tutorials](https://docs.a2a-platform.com/tutorials)

### Support Channels
- **Email**: support@a2a-platform.com
- **Community Forum**: https://community.a2a-platform.com
- **Status Page**: https://status.a2a-platform.com

### SLA and Uptime
- **Production**: 99.9% uptime SLA
- **Staging**: 99.5% uptime (best effort)
- **Maintenance Windows**: Sundays 02:00-04:00 UTC

## Changelog

### Version 1.0.0 (Current)
- Initial API release
- Agent management endpoints
- Workflow management endpoints
- Service marketplace endpoints
- Network analytics endpoints
- OAuth2 authentication
- Multi-tenant support
- Rate limiting
- Webhook support

### Upcoming Features
- GraphQL API support
- Real-time subscriptions via WebSocket
- Advanced analytics and reporting
- Batch operations
- API versioning improvements

---

For the most up-to-date information, please refer to our [online documentation](https://docs.a2a-platform.com).

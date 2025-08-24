# A2A Service Documentation

## Overview

The A2A Service (`a2a-service.js`) is the core business service of the A2A Network application, responsible for managing agent-to-agent interactions, service marketplace operations, and workflow orchestration.

## Service Definition

**Service Path**: `/api/v1`  
**CDS Definition**: `srv/a2a-service.cds`  
**Implementation**: `srv/a2a-service.js`

## Business Logic

### 1. Agent Management

#### Agent Registration
```javascript
// POST /api/v1/Agents
```

**Business Logic**:
- Validates agent information (name, endpoint, capabilities)
- Generates unique agent ID
- Creates blockchain wallet address
- Registers agent on blockchain via `BlockchainService`
- Initializes reputation score (default: 100)
- Sets up service discovery metadata

**Integration Points**:
- `BlockchainService.registerAgent()` - Blockchain registration
- `CapabilityMatcher` - Capability validation
- `NetworkStats` - Update network statistics

**Business Rules**:
- Agent names must be unique within the network
- Endpoint URLs must be valid and reachable
- Minimum one capability required
- Blockchain registration must succeed before DB commit

#### Agent Discovery
```javascript
// GET /api/v1/searchAgents
```

**Business Logic**:
- Searches agents by capabilities using semantic matching
- Filters by reputation threshold
- Ranks results by reputation and match score
- Caches results for performance

**Integration Points**:
- `CapabilityMatcher.findMatches()` - Semantic search
- Redis cache for performance optimization
- Blockchain for real-time reputation data

### 2. Service Marketplace

#### Service Listing
```javascript
// POST /api/v1/Services
```

**Business Logic**:
- Validates service definition and pricing
- Ensures agent owns the service
- Registers service capabilities
- Publishes to marketplace
- Sets up SLA monitoring

**Integration Points**:
- `BlockchainService.registerService()` - On-chain registration
- `PricingEngine` - Dynamic pricing validation
- `SLAMonitor` - Service level tracking

**Business Rules**:
- Services must have clear input/output schemas
- Pricing must be in network tokens
- SLA parameters are mandatory
- Agent must be active to list services

#### Service Discovery
```javascript
// POST /api/v1/discoverServices
```

**Business Logic**:
- Matches requirements to available services
- Considers agent reputation and service ratings
- Applies cost optimization if requested
- Returns ranked service options

**Algorithms**:
1. **Capability Matching**: Semantic similarity using embeddings
2. **Ranking Formula**: 
   ```
   Score = (0.4 × CapabilityMatch) + (0.3 × Reputation) + 
           (0.2 × SuccessRate) + (0.1 × PriceScore)
   ```
3. **Cost Optimization**: Multi-objective optimization considering price, quality, and latency

### 3. Message Routing

#### Send Message
```javascript
// POST /api/v1/sendMessage
```

**Business Logic**:
- Validates sender authorization
- Encrypts message content
- Routes through blockchain message router
- Handles delivery confirmation
- Updates message analytics

**Integration Points**:
- `BlockchainService.routeMessage()` - On-chain routing
- `EncryptionService` - E2E encryption
- `MessageQueue` - Async processing
- `AnalyticsEngine` - Usage tracking

**Message Flow**:
1. Validate sender and recipient
2. Encrypt message payload
3. Submit to blockchain router
4. Queue for delivery
5. Await confirmation
6. Update delivery status

### 4. Workflow Orchestration

#### Execute Workflow
```javascript
// POST /api/v1/Workflows/:ID/execute
```

**Business Logic**:
- Parses workflow definition (DAG)
- Resolves service dependencies
- Executes steps in parallel where possible
- Handles failures and retries
- Aggregates results

**Workflow Engine**:
```javascript
class WorkflowEngine {
  async execute(workflow, input) {
    // 1. Build execution plan
    const plan = this.buildExecutionPlan(workflow);
    
    // 2. Resolve service bindings
    const services = await this.resolveServices(plan);
    
    // 3. Execute with parallelization
    const results = await this.executeParallel(plan, services, input);
    
    // 4. Handle compensation on failure
    if (results.failed) {
      await this.compensate(results.completedSteps);
    }
    
    return results;
  }
}
```

**Integration Points**:
- Service discovery for dynamic binding
- Transaction manager for atomicity
- Event bus for step notifications
- Blockchain for execution proof

### 5. Reputation Management

#### Update Reputation
```javascript
// Internal function called after service execution
```

**Business Logic**:
- Calculates reputation delta based on:
  - Service execution success/failure
  - Response time vs SLA
  - User ratings
  - Dispute resolutions
- Updates on-chain reputation
- Triggers rewards/penalties

**Reputation Algorithm**:
```javascript
function calculateReputationDelta(execution) {
  let delta = 0;
  
  // Success/Failure impact
  delta += execution.success ? 10 : -20;
  
  // SLA compliance
  if (execution.responseTime < execution.sla.targetTime) {
    delta += 5;
  } else if (execution.responseTime > execution.sla.maxTime) {
    delta -= 10;
  }
  
  // User rating impact
  if (execution.rating) {
    delta += (execution.rating - 3) * 2;
  }
  
  // Damping factor based on current reputation
  delta *= (1 - currentReputation / 200);
  
  return Math.max(-50, Math.min(50, delta));
}
```

## Data Flow Patterns

### 1. Service Call Flow
```
Client → A2AService → ServiceDiscovery → BlockchainRouter → TargetAgent
                           ↓
                    CapabilityMatcher
                           ↓
                    ReputationSystem
```

### 2. Async Message Flow
```
Sender → A2AService → MessageQueue → BlockchainService → MessageRouter
                           ↓
                    EncryptionService
                           ↓
                    DeliveryTracker
```

## Error Handling

### Business Errors
- `AGENT_NOT_FOUND`: Agent doesn't exist or is inactive
- `SERVICE_UNAVAILABLE`: Service is offline or at capacity
- `INSUFFICIENT_BALANCE`: Not enough tokens for transaction
- `CAPABILITY_MISMATCH`: Requirements don't match capabilities
- `REPUTATION_THRESHOLD`: Agent reputation below minimum

### Technical Errors
- `BLOCKCHAIN_ERROR`: Blockchain transaction failed
- `TIMEOUT_ERROR`: Service call exceeded timeout
- `VALIDATION_ERROR`: Input validation failed
- `AUTHORIZATION_ERROR`: Insufficient permissions

## Performance Optimizations

### 1. Caching Strategy
- Agent profiles: 5 min TTL
- Service listings: 2 min TTL
- Capability matches: 10 min TTL
- Reputation scores: 30 sec TTL

### 2. Database Optimization
- Indexed fields: `agentId`, `capabilities`, `reputation`
- Materialized views for analytics
- Connection pooling with 50 max connections
- Query optimization for complex joins

### 3. Async Processing
- Message routing via queue
- Workflow execution in background
- Reputation updates batched
- Analytics aggregation scheduled

## Security Considerations

### Authentication
- All endpoints require XSUAA authentication
- Service-to-service uses technical users
- API keys for external integrations

### Authorization
- Role-based access control
- Resource-level permissions
- Tenant isolation

### Data Protection
- E2E encryption for messages
- PII handling compliance
- Audit trail for all operations

## Monitoring & Metrics

### Key Metrics
- `a2a.agents.total`: Total registered agents
- `a2a.services.calls`: Service call volume
- `a2a.services.latency`: Average response time
- `a2a.workflows.success_rate`: Workflow completion rate
- `a2a.messages.delivered`: Message delivery rate

### Health Checks
- Database connectivity
- Blockchain service availability
- Message queue health
- Cache connectivity

## Integration Guidelines

### External Integration
```javascript
// Example: External system integration
const a2aClient = new A2AClient({
  baseUrl: 'https://api.a2a-network.com',
  apiKey: process.env.A2A_API_KEY,
  timeout: 30000
});

// Discover and call service
const services = await a2aClient.discoverServices({
  capabilities: ['data-processing', 'ml-inference'],
  maxPrice: 100
});

const result = await a2aClient.callService(services[0].id, {
  input: data
});
```

### Event Subscriptions
```javascript
// Subscribe to agent events
a2aService.on('agent.registered', async (agent) => {
  // Handle new agent
});

a2aService.on('service.completed', async (execution) => {
  // Handle service completion
});
```

## Testing

### Unit Tests
- Business logic validation
- Error handling scenarios
- Edge cases coverage

### Integration Tests
- End-to-end workflows
- Blockchain integration
- External service mocking

### Performance Tests
- Load testing: 1000 req/sec target
- Stress testing: Resource limits
- Endurance testing: 24-hour runs

## Troubleshooting

### Common Issues

1. **Service Discovery Returns No Results**
   - Check capability definitions
   - Verify agents are active
   - Review reputation thresholds

2. **Workflow Execution Fails**
   - Check service availability
   - Verify input/output compatibility
   - Review compensation logic

3. **Message Delivery Delays**
   - Check queue backlog
   - Verify blockchain confirmations
   - Review network connectivity

### Debug Mode
```bash
# Enable debug logging
export DEBUG=a2a:*
export LOG_LEVEL=debug
```

## API Examples

### Register Agent
```bash
curl -X POST https://api.a2a-network.com/api/v1/Agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DataProcessor-01",
    "endpoint": "https://processor.example.com",
    "capabilities": ["data-cleaning", "etl", "validation"],
    "description": "High-performance data processing agent"
  }'
```

### Discover Services
```bash
curl -X POST https://api.a2a-network.com/api/v1/discoverServices \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": {
      "capabilities": ["ml-training"],
      "minReputation": 150,
      "maxPrice": 500
    }
  }'
```

## Changelog

### Version 1.0.0
- Initial release
- Core agent management
- Service marketplace
- Basic workflow engine

### Version 1.1.0 (Planned)
- Advanced capability matching
- Multi-currency support
- Enhanced workflow patterns
- Performance optimizations
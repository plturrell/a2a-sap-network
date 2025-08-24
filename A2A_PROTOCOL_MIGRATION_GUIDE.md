# A2A Protocol Migration Guide

## Overview

This guide documents the complete migration from HTTP/WebSocket-based communication to the A2A blockchain protocol for the entire platform.

## Migration Components

### 1. Registry Migration

#### Before: HTTP-based Registry
```javascript
// FastAPI router
@router.post("/agents/register")
async def register_agent(request: AgentRegistrationRequest):
    # HTTP endpoint for registration
```

#### After: Blockchain-based Registry
```python
# A2A Registry Handler
@a2a_handler("REGISTER_AGENT")
async def handle_register_agent(self, message: A2AMessage, context_id: str):
    # Blockchain message handler for registration
```

**Files Created:**
- `a2a_registry_handler.py` - Complete A2A-compliant registry implementation

### 2. WebSocket to Blockchain Events

#### Before: WebSocket Server
```javascript
const wsServer = new WebSocket.Server({ port: 4006 });
wsServer.on('connection', (ws) => {
    ws.on('message', (data) => { /* handle */ });
    ws.send(JSON.stringify(update));
});
```

#### After: Blockchain Event Stream
```javascript
const eventServer = new BlockchainEventServer({ port: 4006 });
eventServer.on('blockchain-connection', (client) => {
    client.on('event', (data) => { /* handle */ });
    client.publishEvent('update', data);
});
```

**Files Created:**
- `blockchain_event_stream.py` - Python blockchain event streaming service
- `blockchain-event-adapter.js` - JavaScript WebSocket compatibility layer
- `websocket_to_blockchain_migrator.js` - Automated migration utility

**Migration Results:**
- 27 files successfully migrated from WebSocket to blockchain events
- All real-time communication now uses blockchain event streaming

### 3. Security Enhancements

#### Base Class Migration
All agents now inherit from `SecureA2AAgent` which provides:
- Built-in authentication and authorization
- Rate limiting
- Input validation
- Encrypted communication
- Audit logging
- Security scanning

**Migration Results:**
- 56 agent files migrated to SecureA2AAgent
- All agents now have enterprise-grade security features

### 4. REST to A2A Handler Migration

#### Before: REST Router
```python
@router.get("/agent/{agent_id}/status")
async def get_agent_status(agent_id: str):
    # REST endpoint
```

#### After: A2A Handler
```python
@a2a_handler("GET_AGENT_STATUS")
async def handle_get_agent_status(self, message: A2AMessage):
    # A2A message handler
```

**Migration Results:**
- 10 router files converted to A2A handlers
- All inter-agent communication now uses blockchain

## Implementation Details

### Blockchain Event Types

1. **A2AMessage** - General agent-to-agent messages
2. **AgentEvent** - Agent-specific events (registration, status updates)
3. **MetricUpdate** - Performance and health metrics
4. **WorkflowEvent** - Workflow orchestration events

### Event Subscription Model

```javascript
// Subscribe to events
eventStream.subscribe(subscriberId, ['agent_event', 'metric_update'], async (event) => {
    // Handle blockchain event
    console.log('Received event:', event);
});

// Publish events
await eventStream.publishEvent('agent_event', {
    agentId: 'agent-123',
    eventType: 'status_update',
    data: { status: 'healthy' }
});
```

### Security Features

1. **Authentication**: JWT tokens with blockchain verification
2. **Rate Limiting**: Configurable per-endpoint limits
3. **Input Validation**: Protection against injection attacks
4. **Encryption**: Data encryption at rest and in transit
5. **Audit Trail**: Complete audit logging on blockchain

## Migration Checklist

### Phase 1: Core Infrastructure ✅
- [x] Create SecureA2AAgent base class
- [x] Implement blockchain event streaming
- [x] Create migration utilities

### Phase 2: Agent Migration ✅
- [x] Migrate all agents to SecureA2AAgent
- [x] Convert REST routers to A2A handlers
- [x] Remove HTTP fallback mechanisms

### Phase 3: Network Services ✅
- [x] Replace WebSockets with blockchain events
- [x] Migrate registry to blockchain
- [x] Update real-time data services

### Phase 4: Testing & Deployment 🔄
- [ ] Integration testing with blockchain
- [ ] Performance testing
- [ ] Security audit
- [ ] Production deployment

## Configuration

### Environment Variables
```bash
# Blockchain configuration
BLOCKCHAIN_URL=http://localhost:8545
A2A_CONTRACT_ADDRESS=0x...

# Security settings
JWT_SECRET_KEY=your-secret-key
ENABLE_RATE_LIMITING=true
ENABLE_AUDIT_LOGGING=true

# Event streaming
EVENT_STREAM_BATCH_SIZE=10
EVENT_STREAM_POLL_INTERVAL=1000
```

### Blockchain Contract Deployment
```bash
# Deploy A2A messaging contract
forge create --rpc-url http://localhost:8545 \
  --private-key $PRIVATE_KEY \
  src/A2AMessaging.sol:A2AMessaging

# Deploy event streaming contract  
forge create --rpc-url http://localhost:8545 \
  --private-key $PRIVATE_KEY \
  src/EventStream.sol:EventStream
```

## Troubleshooting

### Common Issues

1. **WebSocket clients not connecting**
   - Ensure blockchain event adapter is properly configured
   - Check that old WebSocket URLs are updated to blockchain URLs

2. **Missing events**
   - Verify blockchain connection
   - Check event filter configuration
   - Ensure proper event type mapping

3. **Authentication failures**
   - Verify JWT configuration
   - Check blockchain address verification
   - Ensure proper key management

### Debug Mode
```javascript
// Enable debug logging
process.env.A2A_DEBUG = 'true';
process.env.BLOCKCHAIN_DEBUG = 'true';
```

## Best Practices

1. **Always use A2A handlers** for inter-agent communication
2. **Never bypass blockchain** for agent messaging
3. **Implement proper error handling** for blockchain failures
4. **Use event batching** for high-frequency updates
5. **Monitor blockchain gas costs** and optimize accordingly

## Next Steps

1. **Deploy blockchain contracts** to test network
2. **Run integration tests** with all agents
3. **Performance tune** event streaming
4. **Security audit** the implementation
5. **Plan production rollout**

## Resources

- [A2A Protocol Specification](./docs/a2a-protocol-spec.md)
- [Blockchain Event Schema](./docs/blockchain-events.md)
- [Security Best Practices](./docs/security-guidelines.md)
- [Migration Scripts](./scripts/migration/)

## Support

For questions or issues:
- Check the [FAQ](./docs/faq.md)
- Review [troubleshooting guide](./docs/troubleshooting.md)
- Contact the A2A platform team
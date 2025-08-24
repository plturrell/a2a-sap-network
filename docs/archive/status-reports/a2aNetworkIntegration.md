# A2A Network Integration for Developer Portal

This document describes the integration between the A2A Developer Portal and the A2A Network blockchain infrastructure.

## Overview

The A2A Network Integration enables the Developer Portal to interact directly with the A2A Network smart contracts, providing:

- **Agent Registration**: Register agents developed in the portal directly on the blockchain
- **Real-time Updates**: Webhook notifications for network events
- **Agent Discovery**: Search and browse agents on the network
- **Messaging**: Send messages between agents through the network
- **Reputation Tracking**: View agent reputation and performance metrics
- **Token Operations**: Check balances and perform token operations
- **Governance**: Participate in network governance

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Developer Portal  │────▶│  A2A Network API     │────▶│   A2A Network       │
│   (SAP UI5/Fiori)  │     │  Integration Layer   │     │   Smart Contracts   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
         │                           │                              │
         │                           │                              │
         ▼                           ▼                              ▼
   Portal Frontend            REST API + WebSocket            Blockchain (EVM)
   - Agent Manager            - /api/a2a-network/*          - AgentRegistry
   - Message Center           - Real-time events            - MessageRouter
   - Network Dashboard        - Webhook delivery            - A2AToken
                                                            - Governance
```

## Features

### 1. Agent Management

**UI Location**: Developer Portal > A2A Network > Agents Tab

- **Register Agent**: Deploy agents from portal projects to the blockchain
- **View Agents**: Browse all registered agents with filtering and search
- **Agent Profile**: View detailed agent information including reputation
- **Update Status**: Activate/deactivate agents
- **Agent Discovery**: Search agents by skills and capabilities

### 2. Messaging System

**UI Location**: Developer Portal > A2A Network > Messages Tab

- **Send Messages**: Send messages to any agent on the network
- **Message History**: View conversation history
- **Message Types**: Support for text, data, command, and query messages
- **Real-time Updates**: WebSocket notifications for new messages

### 3. Webhook Management

**UI Location**: Developer Portal > A2A Network > Webhooks Tab

- **Event Subscriptions**: Subscribe to network events
- **Webhook Configuration**: Set up HTTP endpoints for notifications
- **Event Types**:
  - `agent_registered`: New agent registered
  - `agent_updated`: Agent information updated
  - `agent_status_changed`: Agent activated/deactivated
  - `message_sent`: Message sent to agent
  - `message_received`: Message received by agent
- **Filtering**: Apply filters to receive specific events

### 4. Network Analytics

**UI Location**: Developer Portal > A2A Network > Analytics Tab

- **Network Statistics**: Total agents, active agents, message volume
- **Reputation Leaderboard**: Top performing agents
- **Activity Metrics**: Agent performance over time
- **Network Health**: Connection status and blockchain metrics

## API Endpoints

### Connection Management
- `POST /api/a2a-network/connect` - Connect to A2A Network
- `POST /api/a2a-network/disconnect` - Disconnect from network
- `GET /api/a2a-network/status` - Get connection status

### Agent Operations
- `POST /api/a2a-network/agents/register` - Register new agent
- `GET /api/a2a-network/agents` - List agents (paginated)
- `GET /api/a2a-network/agents/{agent_id}` - Get agent details
- `GET /api/a2a-network/agents/{agent_id}/profile` - Get agent profile
- `PUT /api/a2a-network/agents/{agent_id}` - Update agent
- `PATCH /api/a2a-network/agents/{agent_id}/status` - Set agent status
- `POST /api/a2a-network/agents/search` - Search agents
- `GET /api/a2a-network/agents/owner/{address}` - Get agents by owner

### Messaging
- `POST /api/a2a-network/messages/send` - Send message
- `GET /api/a2a-network/messages/{agent_id}` - Get message history

### Webhooks
- `POST /api/a2a-network/webhooks/subscribe` - Create webhook subscription
- `GET /api/a2a-network/webhooks/subscriptions` - List subscriptions
- `DELETE /api/a2a-network/webhooks/subscriptions/{id}` - Delete subscription
- `PATCH /api/a2a-network/webhooks/subscriptions/{id}` - Update subscription
- `POST /api/a2a-network/webhooks/test/{id}` - Test webhook delivery

### Analytics & Governance
- `GET /api/a2a-network/reputation/{agent_id}` - Get agent reputation
- `GET /api/a2a-network/reputation/leaderboard` - Get reputation leaderboard
- `GET /api/a2a-network/analytics/network` - Get network analytics
- `GET /api/a2a-network/tokens/balance/{address}` - Get token balance
- `GET /api/a2a-network/governance/proposals` - Get governance proposals

## Configuration

### Network Connection

The portal can connect to different A2A Network environments:

```javascript
{
    "network": "mainnet",      // mainnet, testnet, or local
    "rpc_url": "https://...",  // Custom RPC endpoint (optional)
    "private_key": "0x...",    // Private key for transactions (optional)
    "websocket_url": "wss://..." // WebSocket endpoint (optional)
}
```

### Webhook Configuration

Webhook subscriptions require:

```javascript
{
    "event_type": "agent_registered",
    "webhook_url": "https://your-server.com/webhook",
    "filters": {
        // Optional filters
        "agent_owner": "0x...",
        "capabilities": ["messaging"]
    },
    "active": true
}
```

## Usage Guide

### 1. Connecting to A2A Network

1. Navigate to **A2A Network** in the portal sidebar
2. Click **Connect** button
3. Select network (mainnet/testnet/local)
4. Optionally provide custom RPC URL and private key
5. Click **Connect**

### 2. Registering an Agent

1. Ensure you have a project with at least one agent
2. Go to **A2A Network > Agents**
3. Click **Register Agent**
4. Fill in agent details:
   - Name and description
   - Endpoint URL
   - Select capabilities
5. Click **Register**
6. Confirm transaction (if using private key)

### 3. Setting Up Webhooks

1. Go to **A2A Network > Webhooks**
2. Click **Manage Webhooks**
3. Click **Add Webhook**
4. Select event type
5. Enter webhook URL
6. Click **Create**

### 4. Sending Messages

1. Go to **A2A Network > Agents**
2. Select an agent from the list
3. Click **Send Message** button
4. Choose message type and enter content
5. Click **Send**

## Security Considerations

1. **Private Keys**: Never commit private keys. Use environment variables
2. **Webhook URLs**: Validate webhook endpoints before sending sensitive data
3. **Rate Limiting**: API endpoints are rate-limited to prevent abuse
4. **Authentication**: Portal authentication required for all operations
5. **Network Selection**: Be careful when switching between mainnet/testnet

## Troubleshooting

### Connection Issues
- Verify RPC URL is accessible
- Check network selection matches your contracts
- Ensure private key has sufficient balance for gas

### Agent Registration Fails
- Check agent name is unique
- Verify endpoint URL is valid
- Ensure sufficient tokens for registration fee

### Webhooks Not Received
- Verify webhook URL is publicly accessible
- Check webhook subscription is active
- Review event filters

### Performance Issues
- Use pagination for large agent lists
- Implement caching for frequently accessed data
- Consider WebSocket for real-time updates

## Development

### Running Tests

```bash
# Test the integration
python test_a2a_network_integration.py

# Run portal with A2A Network integration
python launch_developer_portal.py
```

### Adding New Features

1. Add API endpoint in `api/a2a_network_integration.py`
2. Update UI controller in `static/controller/A2ANetworkManager.controller.js`
3. Update view in `static/view/A2ANetworkManager.view.xml`
4. Add translations in `static/i18n/i18n.properties`

## Future Enhancements

- [ ] Batch agent operations
- [ ] Advanced analytics dashboard
- [ ] Agent performance monitoring
- [ ] Automated testing framework
- [ ] GraphQL API support
- [ ] Mobile app integration
- [ ] Multi-chain support
- [ ] IPFS integration for agent metadata
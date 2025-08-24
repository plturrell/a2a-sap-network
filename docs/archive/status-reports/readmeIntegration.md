# A2A Developer Portal - Network Integration

## Overview

The A2A Developer Portal now includes full integration with the A2A Network blockchain infrastructure. This enables developers to:

- Register and manage agents directly on the blockchain
- Send messages through the decentralized A2A Network
- Track agent reputation and performance metrics
- Set up webhooks for real-time blockchain events
- Manage governance proposals and voting

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Developer Portal      │     │  A2A Network API     │     │  A2A Smart         │
│   (SAP UI5/Fiori)      │────▶│  Integration Layer   │────▶│  Contracts         │
│                        │     │  (FastAPI)           │     │  (Blockchain)      │
└─────────────────────────┘     └──────────────────────┘     └─────────────────────┘
         │                                │                            │
         │                                │                            │
         ▼                                ▼                            ▼
    WebSocket Updates              Python A2A SDK               Ethereum/Polygon
    Real-time Events              Contract Calls                 Decentralized
    Live Collaboration            Event Listening                 Permanent Storage
```

## Features

### 1. Agent Management
- **Register Agents**: Deploy agents to the blockchain with metadata and capabilities
- **Update Status**: Activate/deactivate agents on-chain
- **Search & Discovery**: Find agents by skills, reputation, or capabilities
- **Profile Management**: View detailed agent profiles with performance metrics

### 2. Messaging System
- **Send Messages**: Route messages through the A2A Network
- **Message History**: View on-chain message records
- **Real-time Updates**: WebSocket notifications for new messages

### 3. Reputation System
- **Score Tracking**: Monitor agent reputation scores
- **Leaderboards**: View top-performing agents
- **Performance Metrics**: Quality, speed, reliability, innovation scores

### 4. Webhook Integration
- **Event Subscriptions**: Subscribe to blockchain events
- **Custom Filters**: Filter events by type, agent, or criteria
- **HTTP Callbacks**: Receive notifications at your endpoints

### 5. Network Settings
- **Multi-Network Support**: Mainnet, Testnet (Sepolia), Local development
- **Wallet Integration**: Connect with private key (stored locally)
- **Custom RPC**: Configure custom blockchain endpoints

## API Endpoints

### Connection Management
- `POST /api/a2a-network/connect` - Connect to A2A Network
- `POST /api/a2a-network/disconnect` - Disconnect from network
- `GET /api/a2a-network/status` - Get connection status

### Agent Operations
- `POST /api/a2a-network/agents/register` - Register new agent
- `GET /api/a2a-network/agents` - List all agents
- `GET /api/a2a-network/agents/{agent_id}` - Get agent details
- `GET /api/a2a-network/agents/{agent_id}/profile` - Get agent profile
- `PUT /api/a2a-network/agents/{agent_id}` - Update agent
- `PATCH /api/a2a-network/agents/{agent_id}/status` - Set agent status
- `POST /api/a2a-network/agents/search` - Search agents

### Messaging
- `POST /api/a2a-network/messages/send` - Send message
- `GET /api/a2a-network/messages/{agent_id}` - Get message history

### Reputation
- `GET /api/a2a-network/reputation/{agent_id}` - Get reputation score
- `GET /api/a2a-network/reputation/leaderboard` - Get top agents

### Webhooks
- `POST /api/a2a-network/webhooks/subscribe` - Subscribe to events
- `GET /api/a2a-network/webhooks/subscriptions` - List subscriptions
- `DELETE /api/a2a-network/webhooks/subscriptions/{id}` - Unsubscribe

## Quick Start

### 1. Install Dependencies
```bash
cd /Users/apple/projects/a2a/a2a_agents/backend
pip install -r requirements.txt
```

### 2. Start the Portal
```bash
cd app/a2a/developer_portal
python deploy_portal_integration.py
```

### 3. Access the Portal
Open http://localhost:3001 in your browser

### 4. Configure Network Connection
1. Click the gear icon in the A2A Network Manager
2. Select your network (mainnet/testnet/local)
3. Enter your private key (optional, for write operations)
4. Click "Save & Connect"

### 5. Register Your First Agent
1. Click "Register Agent" in the A2A Network Manager
2. Fill in agent details:
   - Name: Your agent's display name
   - Description: What your agent does
   - Endpoint: Where your agent can be reached
   - Capabilities: What features your agent supports
3. Click "Register" to deploy to blockchain

## Security Considerations

1. **Private Keys**: Stored only in browser localStorage, never sent to server
2. **Transaction Signing**: All blockchain transactions signed client-side
3. **API Authentication**: Portal uses SAP BTP authentication in production
4. **Rate Limiting**: API endpoints are rate-limited to prevent abuse
5. **WebSocket Security**: Authenticated connections with automatic reconnection

## Development Tips

### Testing with Local Blockchain
```bash
# Start local Hardhat node
cd /Users/apple/projects/a2a/a2a_network
npx hardhat node

# Deploy contracts
npx hardhat run scripts/deploy.js --network localhost
```

### WebSocket Events
```javascript
// Subscribe to agent events in your app
portalClient.on('agent_registered', (data) => {
    console.log('New agent:', data);
});

portalClient.on('message_sent', (data) => {
    console.log('Message sent:', data);
});
```

### Webhook Example
```python
# Your webhook endpoint
@app.post("/webhook")
async def handle_webhook(data: dict):
    event_type = data["event_type"]
    if event_type == "agent_registered":
        # Handle new agent registration
        agent_id = data["data"]["agentId"]
        print(f"New agent registered: {agent_id}")
```

## Troubleshooting

### Connection Issues
- Ensure blockchain node is running
- Check RPC URL is correct
- Verify network selection matches your contracts

### Transaction Failures
- Check wallet has sufficient funds for gas
- Ensure private key has correct permissions
- Verify contract addresses are correct

### WebSocket Disconnections
- Portal automatically reconnects
- Check browser console for errors
- Ensure firewall allows WebSocket connections

## Next Steps

1. **Deploy Smart Contracts**: Deploy A2A contracts to your chosen network
2. **Create Agents**: Build and register your AI agents
3. **Set Up Automation**: Configure webhooks for automated workflows
4. **Monitor Performance**: Track agent reputation and metrics
5. **Participate in Governance**: Vote on network proposals

## Support

For issues or questions:
- Portal UI: Check browser console for errors
- API Issues: Review server logs
- Blockchain: Check transaction receipts
- Documentation: See A2A Network docs

Happy building with A2A Network! 🚀
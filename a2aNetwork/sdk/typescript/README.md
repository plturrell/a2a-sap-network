# A2A Network TypeScript SDK

A comprehensive TypeScript SDK for interacting with the A2A (Agent-to-Agent) Network, providing tools for agent management, secure messaging, blockchain integration, and decentralized governance.

## Features

- 🤖 **Agent Management**: Register, update, and manage AI agents on the network
- 💬 **Secure Messaging**: End-to-end encrypted communication between agents
- 🔗 **Blockchain Integration**: Smart contract interactions with Ethereum-compatible networks
- 🗳️ **Governance**: Participate in decentralized governance proposals and voting
- ⚡ **Scalability**: Batch transactions and optimized gas usage
- 📊 **Reputation System**: Track and manage agent reputation scores
- 🔐 **Security**: Built-in encryption, validation, and error handling
- 📈 **Analytics**: Performance metrics and monitoring

## Installation

```bash
npm install @a2a-network/typescript-sdk
```

## Quick Start

### 1. Environment Setup

Copy the example environment file:

```bash
cp .env.example .env
```

Update the `.env` file with your configuration:

```env
INFURA_PROJECT_ID=your_infura_project_id
DEFAULT_NETWORK=localhost
PRIVATE_KEY=your_private_key_for_testing
```

### 2. Initialize the Client

```typescript
import { A2AClient } from '@a2a-network/typescript-sdk';

const client = new A2AClient({
  network: 'localhost',
  rpcUrl: 'http://localhost:8545',
  privateKey: process.env.PRIVATE_KEY,
  websocketUrl: 'ws://localhost:8545'
});

await client.connect();
```

### 3. Register an Agent

```typescript
const agent = await client.agents.register({
  name: 'MyAIAgent',
  description: 'An intelligent agent for data processing',
  endpoint: 'https://my-agent.example.com',
  capabilities: {
    'data_processing': true,
    'natural_language': true,
    'image_analysis': false
  }
});

console.log('Agent registered:', agent.transactionHash);
```

### 4. Send a Message

```typescript
const message = await client.messages.send({
  recipientAddress: '0x742d35Cc6635C0532FED37cb0f8710e87e6f6E2F',
  content: 'Hello from my agent!',
  messageType: 'direct',
  encrypted: true
});

console.log('Message sent:', message.messageId);
```

## Configuration

### Network Configuration

The SDK supports multiple networks:

- `localhost` - Local development (default)
- `mainnet` - Ethereum Mainnet
- `sepolia` - Sepolia Testnet
- `polygon` - Polygon Mainnet

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFURA_PROJECT_ID` | Infura project ID for network access | - |
| `DEFAULT_NETWORK` | Default network to connect to | `localhost` |
| `PRIVATE_KEY` | Private key for signing transactions | - |
| `WEBSOCKET_URL` | WebSocket URL for real-time updates | - |
| `API_TIMEOUT` | Request timeout in milliseconds | `30000` |
| `RETRY_ATTEMPTS` | Number of retry attempts for failed requests | `3` |
| `LOG_LEVEL` | Logging level (debug, info, warn, error) | `info` |

## API Reference

### A2AClient

The main client class for all SDK interactions.

#### Constructor

```typescript
new A2AClient(config: A2AClientConfig)
```

#### Methods

- `connect()` - Connect to the A2A Network
- `disconnect()` - Disconnect from the network
- `getProvider()` - Get the Ethereum provider
- `getSigner()` - Get the Ethereum signer
- `getContract(name)` - Get a contract instance

### Agent Management

#### Register Agent

```typescript
await client.agents.register({
  name: string,
  description: string,
  endpoint: string,
  capabilities: AgentCapabilities,
  metadata?: string
})
```

#### Update Agent

```typescript
await client.agents.update(agentId, {
  name?: string,
  description?: string,
  capabilities?: Partial<AgentCapabilities>
})
```

#### Get Agent

```typescript
const agent = await client.agents.get(agentAddress);
```

### Messaging

#### Send Message

```typescript
await client.messages.send({
  recipientAddress: string,
  content: string | object,
  messageType?: MessageType,
  encrypted?: boolean,
  priority?: 'low' | 'normal' | 'high'
})
```

#### Get Messages

```typescript
const messages = await client.messages.getInbox({
  limit: 50,
  offset: 0,
  status: 'unread'
});
```

### Token Management

#### Get Balance

```typescript
const balance = await client.tokens.getBalance(address);
```

#### Transfer Tokens

```typescript
await client.tokens.transfer({
  to: recipientAddress,
  amount: ethers.parseEther('10'),
  memo: 'Payment for services'
});
```

### Governance

#### Create Proposal

```typescript
await client.governance.createProposal({
  description: 'Upgrade network parameters',
  data: encodedCallData
});
```

#### Vote on Proposal

```typescript
await client.governance.vote({
  proposalId: '1',
  support: true
});
```

## Development

### Setup

```bash
git clone https://github.com/a2a-network/typescript-sdk.git
cd typescript-sdk
npm install
```

### Build

```bash
npm run build
```

### Development Mode

```bash
npm run dev
```

### Testing

```bash
npm test
```

### Linting

```bash
npm run lint
npm run lint:fix
```

## Error Handling

The SDK provides comprehensive error handling with specific error codes:

```typescript
import { A2AError, ErrorCode } from '@a2a-network/typescript-sdk';

try {
  await client.agents.register(params);
} catch (error) {
  if (error instanceof A2AError) {
    switch (error.code) {
      case ErrorCode.INVALID_ADDRESS:
        console.log('Invalid Ethereum address provided');
        break;
      case ErrorCode.NETWORK_ERROR:
        console.log('Network connection failed');
        break;
      default:
        console.log('Unknown error:', error.message);
    }
  }
}
```

## Events

The SDK emits various events that you can listen to:

```typescript
client.on('connected', ({ network, chainId }) => {
  console.log(`Connected to ${network} (Chain ID: ${chainId})`);
});

client.on('messageReceived', (message) => {
  console.log('New message:', message);
});

client.on('agentUpdate', (agent) => {
  console.log('Agent updated:', agent);
});
```

## Security Considerations

- Never expose private keys in client-side code
- Use environment variables for sensitive configuration
- Validate all input parameters
- Use encrypted messaging for sensitive data
- Regularly update the SDK to get security patches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://docs.a2anetwork.io](https://docs.a2anetwork.io)
- Issues: [GitHub Issues](https://github.com/a2a-network/typescript-sdk/issues)
- Discord: [A2A Network Discord](https://discord.gg/a2anetwork)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and version history.
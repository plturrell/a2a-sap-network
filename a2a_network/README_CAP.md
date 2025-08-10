# A2A Network - SAP CAP Application

This is a 100% integrated SAP Cloud Application Programming Model (CAP) application that provides a comprehensive UI and API for the A2A (Agent-to-Agent) coordination network.

## Architecture

### 1. **Data Layer** (`/db`)
- Comprehensive CDS data models for all A2A components
- Entities for Agents, Services, Capabilities, Workflows, etc.
- Analytics views for performance monitoring

### 2. **Service Layer** (`/srv`)
- **A2AService**: Main OData v4 service with full CRUD operations
- **BlockchainService**: Web3 integration for smart contract interaction
- Real-time WebSocket support for live updates
- Event-driven architecture for blockchain synchronization

### 3. **UI Layer** (`/app`)
- SAP Fiori Elements application
- Responsive UI5 components
- Real-time dashboard with WebSocket updates
- Full integration with all network features

## Features

### Agent Management
- Register and manage agents on blockchain
- Track reputation and performance metrics
- View agent capabilities and services
- Real-time status updates

### Service Marketplace
- List and discover agent services
- Automated pricing and escrow
- Service ratings and reviews
- Transaction history

### Capability Registry
- Register agent capabilities
- Version management
- Dependency tracking
- Capability matching

### Workflow Engine
- Visual workflow designer
- Multi-agent workflow execution
- Step-by-step monitoring
- Gas optimization

### Analytics Dashboard
- Network health monitoring
- Performance metrics
- Reputation distribution
- Service usage statistics

## Installation

```bash
# Install dependencies
npm install

# Deploy to SQLite (development)
npm run deploy

# Start the application
npm start
```

## Configuration

### Blockchain Connection
Configure blockchain RPC endpoint in `package.json`:
```json
"blockchain": {
    "rpc": "http://localhost:8545"
}
```

### Smart Contract Addresses
Update contract addresses after deployment in `package.json`.

## API Endpoints

### OData v4 Service
- Base URL: `/api/v1/`
- Entities: Agents, Services, Capabilities, Workflows, etc.
- Functions: matchCapabilities, calculateReputation, getNetworkHealth
- Actions: registerOnBlockchain, syncBlockchain, deployContract

### Blockchain API
- `/blockchain/gas-price` - Current gas price
- `/blockchain/block/:number` - Block information
- `/blockchain/transaction/:hash` - Transaction details
- `/blockchain/contract/:name` - Contract information

### WebSocket Events
- `agent:registered` - New agent registration
- `service:created` - New service listing
- `reputation:updated` - Reputation changes
- `workflow:completed` - Workflow completion

## Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
cds build --production
```

### Deployment to SAP BTP
```bash
cf push
```

## Integration Points

1. **Smart Contracts**: Full integration with all A2A smart contracts
2. **SDK**: Uses A2A SDK for blockchain operations
3. **Agent Services**: Can communicate with external agent endpoints
4. **Cross-chain**: Supports multi-chain deployments
5. **Privacy**: Implements privacy-preserving features

## Security

- SAP XSUAA authentication integration
- Role-based access control
- Secure WebSocket connections
- Encrypted private channels

## Monitoring

- Health check endpoint: `/health`
- Prometheus metrics (when configured)
- SAP Cloud ALM integration
- Real-time performance tracking

## License

MIT
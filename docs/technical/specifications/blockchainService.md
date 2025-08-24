# Blockchain Service Documentation

## Overview

The Blockchain Service (`blockchain-service.js`) provides the Web3 integration layer for the A2A Network, managing all blockchain interactions including smart contract deployment, agent registration, message routing, and reputation management on the Ethereum blockchain.

## Service Definition

**Service Path**: `/odata/v4/blockchain`  
**CDS Definition**: `srv/blockchain-service.cds`  
**Implementation**: `srv/blockchain-service.js`

## Architecture

### Components
```
BlockchainService
├── Web3 Provider (Ethereum/Polygon/BSC)
├── Contract Manager
│   ├── AgentRegistry
│   ├── MessageRouter
│   ├── ServiceMarketplace
│   ├── CapabilityMatcher
│   └── ReputationSystem
├── Transaction Manager
├── Event Listener
└── Gas Price Oracle
```

## Business Logic

### 1. Blockchain Connection Management

#### Initialize Web3
```javascript
// Called on service startup
async init()
```

**Business Logic**:
- Detects network environment (local/testnet/mainnet)
- Initializes Web3 provider with appropriate RPC endpoint
- Sets up account management
- Configures gas price strategy
- Loads deployed contract addresses

**Configuration Priority**:
1. Environment variables (`ETH_RPC_URL`, `ETH_PRIVATE_KEY`)
2. VCAP services binding
3. Default local development settings

**Network Support**:
- Local: Hardhat/Ganache
- Testnet: Goerli, Mumbai (Polygon)
- Mainnet: Ethereum, Polygon, BSC

### 2. Smart Contract Management

#### Contract Loading
```javascript
async loadContracts()
```

**Business Logic**:
- Reads contract artifacts from deployment
- Creates Web3 contract instances
- Validates contract addresses
- Sets up event listeners
- Caches contract ABIs

**Contract Dependencies**:
```javascript
{
  "AgentRegistry": "0x5fbdb2315678afecb367f032d93f642f64180aa3",
  "MessageRouter": "0xe7f1725e7734ce288f8367e1bb143e90bb3f0512",
  "ServiceMarketplace": "0x9fe46736679d2d9a65f0992f2272de9f3c7fa6e0",
  "CapabilityMatcher": "0xcf7ed3acca5a467e9e704c703e8d87f634fb0fc9",
  "ReputationSystem": "0xdc64a140aa3e981100a9beca4e685f962f0cf6c9"
}
```

### 3. Agent Registry Operations

#### Register Agent
```javascript
async registerAgent(agentData)
```

**Business Logic**:
1. Validates agent doesn't already exist on-chain
2. Prepares transaction data
3. Estimates gas requirements
4. Submits transaction with retry logic
5. Waits for confirmation
6. Emits registration event

**Smart Contract Interface**:
```solidity
function registerAgent(
    string memory name,
    address endpoint,
    string[] memory capabilities,
    uint256 initialStake
) external returns (uint256 agentId)
```

**Gas Optimization**:
- Batch capability registration
- Use events for data storage
- Optimize string storage

#### Get Agent Details
```javascript
async getAgentDetails(agentId)
```

**Business Logic**:
- Queries on-chain agent data
- Fetches associated services
- Retrieves reputation score
- Aggregates capability information
- Caches for performance

### 4. Message Routing

#### Route Message
```javascript
async routeMessage(messageData)
```

**Business Logic**:
1. Encrypts message payload
2. Validates sender permissions
3. Checks recipient availability
4. Calculates routing path
5. Submits to MessageRouter contract
6. Tracks delivery status

**Message Flow**:
```
Sender → Encryption → Smart Contract → Event → Recipient
         ↓                                      ↑
    IPFS Storage ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

**Smart Contract Interface**:
```solidity
struct Message {
    address sender;
    address recipient;
    bytes32 contentHash;  // IPFS hash
    uint256 timestamp;
    MessageType msgType;
    uint256 fee;
}

function routeMessage(Message memory msg) external payable
```

### 5. Service Marketplace

#### List Service
```javascript
async listService(serviceData)
```

**Business Logic**:
1. Validates service definition
2. Calculates listing fee
3. Registers capabilities on-chain
4. Sets pricing parameters
5. Enables service discovery

**Pricing Models**:
- Fixed price per call
- Dynamic pricing based on demand
- Auction-based pricing
- Subscription model

**Smart Contract Storage**:
```solidity
struct Service {
    uint256 id;
    address provider;
    string name;
    string[] capabilities;
    uint256 pricePerCall;
    uint256 minReputation;
    bool active;
}
```

### 6. Reputation Management

#### Update Reputation
```javascript
async updateReputation(agentId, delta, reason)
```

**Business Logic**:
1. Validates update authorization
2. Calculates new reputation score
3. Applies bounds (0-200)
4. Records update reason
5. Triggers rewards/penalties

**Reputation Factors**:
```javascript
const ReputationFactors = {
  SERVICE_SUCCESS: +10,
  SERVICE_FAILURE: -20,
  SLA_VIOLATION: -15,
  POSITIVE_RATING: +5,
  NEGATIVE_RATING: -10,
  DISPUTE_WON: +15,
  DISPUTE_LOST: -25,
  UPTIME_BONUS: +2,
  INACTIVITY_PENALTY: -5
};
```

**Consensus Mechanism**:
- Multi-signature validation for major changes
- Time-locked updates
- Appeal process for disputes

### 7. Gas Management

#### Estimate Gas
```javascript
async estimateGas(transaction)
```

**Business Logic**:
1. Simulates transaction
2. Adds safety buffer (20%)
3. Checks gas price oracle
4. Applies network-specific adjustments
5. Returns cost estimate

**Gas Optimization Strategies**:
- Batch operations
- Off-chain computation
- Storage packing
- Event-based data retrieval

### 8. Event Monitoring

#### Event Listeners
```javascript
setupEventListeners()
```

**Monitored Events**:
- `AgentRegistered`
- `ServiceListed`
- `MessageRouted`
- `ReputationUpdated`
- `DisputeRaised`
- `PaymentProcessed`

**Event Processing**:
1. Real-time event streaming
2. Event deduplication
3. Database synchronization
4. Webhook notifications
5. Analytics updates

## Integration Points

### 1. External Blockchain Networks

#### Multi-Chain Support
```javascript
const NetworkConfig = {
  ethereum: {
    rpc: "https://mainnet.infura.io/v3/YOUR-KEY",
    chainId: 1,
    contracts: { /* mainnet addresses */ }
  },
  polygon: {
    rpc: "https://polygon-rpc.com",
    chainId: 137,
    contracts: { /* polygon addresses */ }
  },
  bsc: {
    rpc: "https://bsc-dataseed.binance.org",
    chainId: 56,
    contracts: { /* bsc addresses */ }
  }
};
```

### 2. IPFS Integration

#### Content Storage
```javascript
async storeOnIPFS(content)
```

**Usage**:
- Large message payloads
- Service definitions
- Workflow specifications
- Audit logs

### 3. Oracle Integration

#### External Data Feeds
- Gas price oracle (Chainlink)
- Token price feeds
- Network congestion data
- Cross-chain messaging

## Security Considerations

### 1. Private Key Management
```javascript
// NEVER hardcode private keys
const account = process.env.ETH_PRIVATE_KEY
  ? web3.eth.accounts.privateKeyToAccount(process.env.ETH_PRIVATE_KEY)
  : await getFromHSM(); // Hardware Security Module
```

### 2. Transaction Security
- Nonce management to prevent replay
- Gas price limits to prevent drainage
- Reentrancy guards in contracts
- Time locks for critical operations

### 3. Access Control
```javascript
// Role-based contract permissions
const Roles = {
  ADMIN: "0x00",
  OPERATOR: "0x01",
  AGENT: "0x02",
  USER: "0x03"
};
```

## Performance Optimization

### 1. Caching Strategy
```javascript
const CacheConfig = {
  agentData: { ttl: 300 },      // 5 minutes
  serviceList: { ttl: 120 },     // 2 minutes
  gasPrice: { ttl: 30 },         // 30 seconds
  contractABI: { ttl: 3600 }     // 1 hour
};
```

### 2. Batch Operations
```javascript
// Batch multiple operations in single transaction
async batchRegisterAgents(agents) {
  const batch = new web3.BatchRequest();
  agents.forEach(agent => {
    batch.add(contract.methods.registerAgent(...).send.request());
  });
  return batch.execute();
}
```

### 3. Event Filtering
```javascript
// Efficient event queries
const filter = {
  fromBlock: lastProcessedBlock,
  toBlock: 'latest',
  topics: [
    web3.utils.sha3('AgentRegistered(address,uint256)'),
    null, // any address
    agentId // specific agent
  ]
};
```

## Error Handling

### Blockchain-Specific Errors
```javascript
const BlockchainErrors = {
  INSUFFICIENT_FUNDS: "Wallet balance too low",
  GAS_TOO_LOW: "Gas price below network minimum",
  NONCE_CONFLICT: "Transaction nonce already used",
  CONTRACT_REVERT: "Smart contract execution failed",
  NETWORK_CONGESTION: "Network too congested",
  TIMEOUT: "Transaction not mined within timeout"
};
```

### Retry Logic
```javascript
async function withRetry(fn, options = {}) {
  const maxRetries = options.maxRetries || 3;
  const delay = options.delay || 1000;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      if (!isRetryable(error)) throw error;
      await sleep(delay * Math.pow(2, i)); // Exponential backoff
    }
  }
}
```

## Monitoring & Metrics

### Key Metrics
- `blockchain.transactions.total`: Total transactions sent
- `blockchain.transactions.failed`: Failed transaction count
- `blockchain.gas.average`: Average gas used
- `blockchain.latency`: Transaction confirmation time
- `blockchain.events.processed`: Events processed count

### Health Checks
```javascript
async checkHealth() {
  return {
    connected: await web3.eth.net.isListening(),
    blockNumber: await web3.eth.getBlockNumber(),
    gasPrice: await web3.eth.getGasPrice(),
    balance: await web3.eth.getBalance(account.address),
    contracts: await this.validateContracts()
  };
}
```

## Testing

### Unit Tests
```javascript
describe('BlockchainService', () => {
  let mockWeb3;
  
  beforeEach(() => {
    mockWeb3 = new MockWeb3();
    // Setup mock contracts
  });
  
  it('should register agent on blockchain', async () => {
    // Test implementation
  });
});
```

### Integration Tests
- Ganache local blockchain
- Forked mainnet testing
- Contract interaction tests
- Event emission verification

## Troubleshooting

### Common Issues

1. **Transaction Stuck**
   ```javascript
   // Speed up transaction
   await web3.eth.sendTransaction({
     nonce: originalNonce,
     gasPrice: originalGasPrice * 1.1,
     to: originalTo,
     value: 0
   });
   ```

2. **Gas Estimation Failed**
   - Check contract state requirements
   - Verify account balance
   - Review function parameters

3. **Event Not Received**
   - Check block range
   - Verify event topics
   - Review filter criteria

## API Examples

### Register Agent on Blockchain
```javascript
const result = await blockchainService.callFunction('/registerAgent', {
  method: 'POST',
  data: {
    name: "ML-Agent-01",
    capabilities: ["machine-learning", "data-analysis"],
    endpoint: "https://agent.example.com",
    stake: "1000000000000000000" // 1 ETH in wei
  }
});
```

### Query Blockchain State
```javascript
// Get agent details
const agent = await blockchainService.callFunction('/getAgentDetails', {
  method: 'GET',
  data: { agentId: 123 }
});

// Get network stats
const stats = await blockchainService.callFunction('/getNetworkStats', {
  method: 'GET'
});
```

## Smart Contract Addresses

### Mainnet (Ethereum)
```json
{
  "AgentRegistry": "0x...",
  "MessageRouter": "0x...",
  "ServiceMarketplace": "0x...",
  "ReputationSystem": "0x..."
}
```

### Testnet (Goerli)
```json
{
  "AgentRegistry": "0x...",
  "MessageRouter": "0x...",
  "ServiceMarketplace": "0x...",
  "ReputationSystem": "0x..."
}
```

## Migration Guide

### From v1 to v2
1. Update contract addresses
2. Migrate agent data
3. Update event listeners
4. Test thoroughly on testnet
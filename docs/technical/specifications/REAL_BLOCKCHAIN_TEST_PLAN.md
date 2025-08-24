# Real Blockchain Testing Plan for A2A Agents

## Current State
- ✅ Smart contracts exist (AgentServiceMarketplace, PerformanceReputationSystem, etc.)
- ✅ Python blockchain SDK exists (web3Client, agentIntegration, eventListener)
- ✅ BlockchainIntegrationMixin implemented
- ✅ Blockchain features now enabled by default (BLOCKCHAIN_ENABLED="true")
- ⚠️ Some agents have import issues preventing full initialization

## Testing Approach - Step by Step

### Phase 1: Environment Setup
1. **Deploy Smart Contracts to Test Network**
   - Use existing contracts in `/a2aNetwork/contracts/`
   - Deploy to local Anvil network
   - Get contract addresses

2. **Configure Environment**
   ```bash
   export BLOCKCHAIN_ENABLED=true
   export A2A_RPC_URL=http://localhost:8545
   export AGENT_REGISTRY_ADDRESS=<deployed_address>
   export MESSAGE_ROUTER_ADDRESS=<deployed_address>
   export TRUST_MANAGER_ADDRESS=<deployed_address>
   ```

3. **Set Agent Private Keys**
   - Use Anvil's test accounts
   - Assign one account per agent (16 total)

### Phase 2: Component Testing
1. **Test Blockchain SDK**
   - Test web3Client.py connection
   - Test contract interactions
   - Test event listening

2. **Test Individual Agents**
   - Start with agentManager (most complete)
   - Verify blockchain initialization
   - Test message sending/receiving

3. **Test Agent Discovery**
   - Register agents on blockchain
   - Query by capabilities
   - Verify trust levels

### Phase 3: Integration Testing
1. **Multi-Agent Communication**
   - Send messages between agents
   - Verify blockchain routing
   - Check event emissions

2. **Reputation System**
   - Update agent reputations
   - Test trust-based access
   - Verify reputation queries

3. **Service Marketplace**
   - List agent services
   - Test service discovery
   - Verify payment flows

### Phase 4: A2A Compliance Testing
1. **Protocol Compliance**
   - Verify message format
   - Test capability matching
   - Check trust verification

2. **Performance Testing**
   - Transaction throughput
   - Gas optimization
   - Latency measurements

3. **Error Scenarios**
   - Network failures
   - Invalid signatures
   - Insufficient reputation

## Next Steps

1. **Create deployment script** for smart contracts
2. **Create environment setup script** with all required variables
3. **Create minimal test** with just 2 agents communicating
4. **Gradually expand** to all 16 agents

## Key Files to Use
- Smart Contracts: `/a2aNetwork/contracts/`
- Python SDK: `/a2aNetwork/sdk/pythonSdk/blockchain/`
- Agent Integration: `/a2aAgents/backend/app/a2a/sdk/blockchainIntegration.py`
- Test Network: Anvil (already installed)

## Success Criteria
- [ ] All contracts deployed successfully
- [ ] At least 2 agents can communicate via blockchain
- [ ] Messages are verifiable on-chain
- [ ] Reputation system works
- [ ] All 16 agents eventually integrated
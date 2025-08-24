# Real A2A Messaging Testing Guide

This guide explains how to test the A2A network with **real blockchain messaging** - no mocks, no simulations.

## Prerequisites

1. **Node.js and npm** installed
2. **Python 3.8+** with required packages
3. **Hardhat** for local blockchain

## Setup Steps

### 1. Install Dependencies

```bash
# In a2aNetwork directory
cd /Users/apple/projects/a2a/a2aNetwork
npm install
```

### 2. Start the Real A2A System

Run the comprehensive startup script:

```bash
cd /Users/apple/projects/a2a/a2aAgents/backend/services/chatAgent
python start_real_a2a_system.py
```

This will:
- Start local blockchain (Hardhat)
- Deploy smart contracts
- Start AgentManager service
- Start DataManager service
- Register agents on blockchain
- Verify system health

### 3. Verify Blockchain Integration

In a new terminal, verify the blockchain is real:

```bash
python verify_blockchain_integration.py
```

This will:
- Test Web3 connection
- Verify smart contracts are deployed
- Check contract ABIs
- Test BlockchainIntegration class
- Confirm no mock components

### 4. Run Real A2A Messaging Tests

Once verified, run the comprehensive tests:

```bash
python test_real_a2a_messaging.py
```

This will test:
- Real blockchain agent registration
- A2A message sending through blockchain
- Skills matching with AI intelligence
- Message lifecycle tracking
- Cross-agent collaboration
- Network statistics from AgentManager
- Blockchain persistence

## What Makes This "Real"?

1. **No Mock Objects**: All components use actual implementations
2. **Real Blockchain**: Messages go through actual smart contracts
3. **Real AI Integration**: Skills matching uses actual AI reasoning
4. **Real Network Communication**: Agents communicate via blockchain events
5. **Real Persistence**: All data stored on blockchain and tracked

## Monitoring

Watch the logs to see:
- Blockchain transactions
- Agent registrations
- Message routing decisions
- Skills matching analysis
- Reputation calculations

## Troubleshooting

### Blockchain Not Connecting
- Ensure Hardhat is running: `npx hardhat node`
- Check RPC URL: `http://localhost:8545`

### Contracts Not Deployed
- Deploy manually: `npx hardhat run scripts/deploy.js --network localhost`
- Check deployment addresses in logs

### Agents Not Responding
- Verify services are running with `ps aux | grep python`
- Check service logs for errors
- Ensure all environment variables are set

## Environment Variables

The system requires these environment variables (set automatically by startup script):

```bash
export A2A_SERVICE_URL="http://localhost:8010"
export A2A_SERVICE_HOST="localhost"
export A2A_BASE_URL="http://localhost:8010"
export A2A_RPC_URL="http://localhost:8545"
export A2A_AGENT_REGISTRY_ADDRESS="0x5FbDB2315678afecb367f032d93F642f64180aa3"
export A2A_MESSAGE_ROUTER_ADDRESS="0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
export AI_ENABLED="true"
export BLOCKCHAIN_ENABLED="true"
export ENABLE_AGENT_MANAGER_TRACKING="true"
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Chat Agent    │────▶│   Blockchain    │────▶│  Agent Manager  │
│ (Skills Match)  │     │ (Smart Contract)│     │  (Tracking)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        │                        ▼                        │
        │               ┌─────────────────┐               │
        └──────────────▶│  Message Router │◀──────────────┘
                        │  (Event Based)  │
                        └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌─────────────────┐      ┌─────────────────┐
            │  Data Manager   │      │   Calc Agent    │
            │  (Storage)      │      │  (Computation)  │
            └─────────────────┘      └─────────────────┘
```

## Success Criteria

The tests are successful when:
1. ✅ All agents register on blockchain
2. ✅ Messages route through smart contracts
3. ✅ Skills matching selects appropriate agents
4. ✅ Message lifecycle is tracked
5. ✅ Reputation scores are calculated
6. ✅ No mock components are used
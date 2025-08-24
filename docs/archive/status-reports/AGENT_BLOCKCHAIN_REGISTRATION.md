# A2A Agent Blockchain Registration Guide

## Overview

All 16 agents in the A2A network automatically register with the blockchain when they start up. The registration process ensures:
- Proper conversion of capabilities to bytes32 format
- Correct endpoint configuration
- Blockchain transaction handling

## How Registration Works

### 1. **Automatic Registration on Startup**

When any agent starts, it automatically calls `register_with_network()`:

```python
# In each agent's main.py lifespan function:
await agent_instance.initialize()
await agent_instance.register_with_network()  # This handles blockchain registration
```

### 2. **Capability Collection**

The registration process collects capabilities from:
- Skills decorated with `@a2a_skill`
- The `blockchain_capabilities` array in agent configuration
- Removes duplicates automatically

### 3. **Endpoint Configuration**

Endpoints are determined in priority order:
1. `base_url` passed to agent constructor
2. `A2A_AGENT_URL` environment variable
3. `A2A_SERVICE_URL` environment variable

If none are set, registration fails with a clear error message.

### 4. **Bytes32 Conversion**

Capabilities are automatically converted to bytes32 format in the blockchain client:

```python
# In web3Client.py
capability_hashes = [
    self.web3.keccak(text=cap)[:32] for cap in capabilities
]
```

## Required Environment Variables

```bash
# Blockchain Configuration
export A2A_RPC_URL=http://localhost:8545
export A2A_PRIVATE_KEY=<your_private_key>
export BLOCKCHAIN_ENABLED=true

# Agent Endpoint (at least one required)
export A2A_AGENT_URL=http://localhost:8001
# OR
export A2A_SERVICE_URL=http://localhost:8001

# Optional Registry
export A2A_REGISTRY_URL=http://localhost:8000
```

## Verification

### Check Registration Status

```python
# In any agent
blockchain_stats = agent.get_blockchain_stats()
print(f"Registered: {blockchain_stats['registered']}")
print(f"Address: {blockchain_stats['address']}")
```

### Query Blockchain Directly

```bash
# Using cast (Foundry)
cast call $AGENT_REGISTRY_ADDRESS "getAgent(address)" $AGENT_ADDRESS --rpc-url http://localhost:8545
```

## Troubleshooting

### Error: "No endpoint configured"
**Solution**: Set `A2A_AGENT_URL` or `A2A_SERVICE_URL` environment variable

### Error: "No private key configured"
**Solution**: Set `A2A_PRIVATE_KEY` environment variable

### Error: "Failed to register agent"
**Check**:
- Blockchain is running (`anvil` or your blockchain node)
- RPC URL is correct
- Private key has funds for gas
- Contract addresses are correct

### Capabilities Not Showing
**Check**:
- Skills are decorated with `@a2a_skill`
- Agent has `blockchain_capabilities` array defined
- Registration transaction succeeded

## All 16 Agents

1. **Data Product Agent** - `data_product_agent`
2. **Data Standardization Agent** - `data_standardization_agent`  
3. **AI Preparation Agent** - `ai_preparation_agent`
4. **Vector Processing Agent** - `vector_processing_agent`
5. **Calculation Validation Agent** - `calc_validation_agent`
6. **QA Validation Agent** - `qa_validation_agent`
7. **Quality Control Agent** - `quality_control_agent`
8. **SQL Agent** - `sql_agent`
9. **Calculation Agent** - `calculation_agent`
10. **Reasoning Agent** - `reasoning_agent`
11. **Agent Manager** - `agent_manager`
12. **Data Manager Agent** - `data_manager_agent`
13. **Catalog Manager Agent** - `catalog_manager_agent`
14. **Agent Builder** - `agent_builder`
15. **Embedding Fine Tuner** - `embedding_fine_tuner`
16. **Chat Agent** - `chat_agent`

## Implementation Status

✅ **Completed**:
- Base agent class includes `BlockchainIntegrationMixin`
- `register_with_network()` method handles registration
- Capabilities are properly converted to bytes32
- Endpoints are validated before registration
- All agents inherit from `A2AAgentBase`

The registration is now part of the standard agent lifecycle, not a one-off script!
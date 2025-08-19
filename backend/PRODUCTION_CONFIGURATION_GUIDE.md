# A2A Production Configuration Guide

This guide outlines the required configuration for deploying the A2A system in production after removing all mock implementations and fallbacks.

## Critical Configuration Changes Made

### ❌ **Removed Mock/Fallback Implementations:**

1. **Mock Reasoning Fallbacks** - Removed internal reasoning fallbacks in ReasoningAgent
2. **Fake Queue Metrics** - Replaced random queue metrics with real Redis-based metrics
3. **Development Cryptographic Keys** - Removed hardcoded test keys
4. **Telemetry Stubs** - Removed fallback telemetry implementations
5. **Contract Configuration Fallbacks** - Removed fallback contract configurations
6. **Service Discovery Fallbacks** - Implemented real A2A network service discovery

### ✅ **Now Required for Production:**

## Environment Variables

### **Cryptographic Keys (REQUIRED)**
```bash
# Agent private keys - MUST be real, no defaults
export AGENT_PRIVATE_KEY="0x<your-real-private-key>"
```

### **A2A Network Registry (REQUIRED)**
```bash
# Service discovery registry URL
export A2A_REGISTRY_URL="https://registry.a2a-network.com"

# Contract addresses - all required, no fallbacks
export A2A_AGENT_REGISTRY_ADDRESS="0x<AgentRegistry-contract-address>"
export A2A_MESSAGE_ROUTER_ADDRESS="0x<MessageRouter-contract-address>"
export A2A_ORD_REGISTRY_ADDRESS="0x<ORDRegistry-contract-address>"

# Contract artifacts path
export A2A_ABI_PATH="/path/to/contract/artifacts"
```

### **Redis Configuration (REQUIRED)**
```bash
# Redis for real queue metrics and caching
export REDIS_URL="redis://localhost:6379"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD="<your-redis-password>"
```

### **Blockchain Network (REQUIRED)**
```bash
# Web3 provider URL
export WEB3_PROVIDER_URL="https://mainnet.infura.io/v3/<your-key>"
export BLOCKCHAIN_NETWORK="mainnet"  # or testnet
```

## Required Services

### **1. A2A Registry Service**
- Must be running and accessible at `A2A_REGISTRY_URL`
- Provides real agent discovery and registration
- **API Endpoints Required:**
  - `GET /api/v1/agents/search` - Service discovery
  - `POST /api/v1/agents/register` - Agent registration
  - `DELETE /api/v1/agents/{id}` - Agent deregistration

### **2. Redis Instance**
- Required for real queue metrics and caching
- Used for message queue management
- **Redis Keys Used:**
  - `message_queue:pending` - Pending messages count
  - `message_queue:failed` - Failed messages queue
  - `queue_stats:processed_today` - Daily processed count
  - `queue_stats:failed_today` - Daily failed count

### **3. Agent Services in A2A Network**
The system now requires real agents to be available:

#### **QA Validation Agents**
- Service type: `qa_validation`
- Required capabilities: `["question_analysis", "validation"]`
- Must expose `/health` endpoint
- Must support `generate_reasoning_chain` skill

#### **Data Manager Agents**
- Service type: `data_manager` 
- Required capabilities: `["data_retrieval", "search"]`
- Must expose `/health` endpoint
- Must support `search_knowledge_base` skill

#### **Reasoning Engine Agents**
- Service type: `reasoning_engine`
- Required capabilities: `["logical_reasoning", "inference"]`
- Must expose `/health` endpoint

#### **Answer Synthesis Agents**
- Service type: `answer_synthesis`
- Required capabilities: `["synthesis", "aggregation"]`
- Must expose `/health` endpoint

### **4. Telemetry System**
- OpenTelemetry must be properly configured
- No fallback stubs - will fail if not available
- **Required modules:**
  - `app.a2a.core.telemetry`
  - `app.a2a.config.telemetryConfig`

### **5. Smart Contracts**
- All contracts must be deployed and accessible
- Contract artifacts must be available at `A2A_ABI_PATH`
- No zero addresses or fallback configurations

## Configuration Validation

### **Pre-deployment Checklist:**

1. **✅ Environment Variables Set**
   ```bash
   # Check all required vars are set
   echo $AGENT_PRIVATE_KEY
   echo $A2A_REGISTRY_URL
   echo $A2A_AGENT_REGISTRY_ADDRESS
   echo $REDIS_URL
   ```

2. **✅ Services Accessible**
   ```bash
   # Test registry connectivity
   curl -f $A2A_REGISTRY_URL/api/v1/agents/search
   
   # Test Redis connectivity  
   redis-cli -u $REDIS_URL ping
   ```

3. **✅ Contract Artifacts Available**
   ```bash
   # Check contract artifacts exist
   ls $A2A_ABI_PATH/AgentRegistry.sol/AgentRegistry.json
   ls $A2A_ABI_PATH/MessageRouter.sol/MessageRouter.json
   ls $A2A_ABI_PATH/ORDRegistry.sol/ORDRegistry.json
   ```

4. **✅ Agent Network Populated**
   ```bash
   # Verify agents are registered
   curl "$A2A_REGISTRY_URL/api/v1/agents/search?service_type=qa_validation"
   curl "$A2A_REGISTRY_URL/api/v1/agents/search?service_type=data_manager"
   ```

## Error Handling Changes

### **Previous Behavior (Removed):**
- Fallback to internal reasoning when external agents unavailable
- Random metrics when Redis unavailable
- Default test keys when environment variable missing
- Silent telemetry disabling
- Fallback contract configurations

### **New Production Behavior:**
- **Hard failures** when required services unavailable
- **Detailed error logging** with actionable messages
- **No silent fallbacks** - all failures are explicit
- **Runtime validation** of all critical dependencies

## Deployment Steps

1. **Deploy Infrastructure:**
   ```bash
   # Deploy Redis
   docker run -d --name redis -p 6379:6379 redis:latest
   
   # Deploy A2A Registry Service
   # (Follow A2A Registry deployment guide)
   ```

2. **Deploy Smart Contracts:**
   ```bash
   # Deploy contracts using Foundry
   forge script script/Deploy.s.sol --rpc-url $WEB3_PROVIDER_URL --broadcast
   ```

3. **Configure Environment:**
   ```bash
   # Set all required environment variables
   source production.env
   ```

4. **Register Initial Agents:**
   ```bash
   # Register core agents in network
   python scripts/registerA2aAgents.js
   ```

5. **Start Services:**
   ```bash
   # Start with production configuration
   python -m app.a2a.agents.reasoningAgent.reasoningAgent
   ```

## Monitoring and Alerts

### **Critical Alerts to Configure:**

1. **Service Discovery Failures**
   - Alert when agent discovery fails
   - Monitor agent network health

2. **Queue System Issues**
   - Alert on Redis connectivity failures
   - Monitor queue depth and processing rates

3. **Contract Interaction Failures**
   - Alert on blockchain connectivity issues
   - Monitor transaction failures

4. **Agent Network Health**
   - Alert when required agent types unavailable
   - Monitor agent response times

## Troubleshooting

### **Common Production Issues:**

1. **"No QA agents available in A2A network"**
   - Check agent registration in registry
   - Verify agent health endpoints

2. **"AGENT_PRIVATE_KEY environment variable is required"**
   - Set real private key, no test keys allowed
   - Ensure key has sufficient permissions

3. **"Redis client not available for queue metrics"**
   - Verify Redis connectivity
   - Check Redis authentication

4. **"ABI not found for contract"**
   - Ensure contract artifacts are available
   - Check A2A_ABI_PATH configuration

5. **"Telemetry modules are required"**
   - Install OpenTelemetry dependencies
   - Configure telemetry system properly

## Security Considerations

1. **Private Key Management**
   - Use secure key management systems
   - Never commit private keys to code
   - Rotate keys regularly

2. **Service Authentication**
   - Implement proper authentication between services
   - Use TLS for all network communication

3. **Access Control**
   - Restrict access to Redis and registry services
   - Implement proper network segmentation

4. **Monitoring and Auditing**
   - Log all critical operations
   - Monitor for suspicious activity
   - Implement proper audit trails
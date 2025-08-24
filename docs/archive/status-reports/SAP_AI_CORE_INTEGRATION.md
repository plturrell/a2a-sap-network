# SAP AI Core SDK Integration for A2A Agents

## Overview

All 16 A2A agents have been integrated with the SAP AI Core SDK, providing enterprise-grade LLM capabilities with automatic failover.

## Architecture

### LLM Service Hierarchy

1. **Development Mode**: 
   - Primary: Grok4 (via X.AI)
   - Fallback: LNN (Local Neural Network)

2. **Production Mode**:
   - Primary: SAP AI Core with Claude Opus 4
   - Fallback: LNN (Local Neural Network)

3. **Local Mode**:
   - Primary: Grok4 (if available)
   - Fallback: LNN (always available)

## Integration Points

### 1. Core GrokClient (`app/a2a/core/grokClient.py`)
- Enhanced with SAP AI Core SDK integration
- Maintains backward compatibility
- Automatic service detection and failover

### 2. AIIntelligenceMixin (`app/a2a/sdk/aiIntelligenceMixin.py`)
- Updated to detect and log SAP AI Core availability
- Shows active failover chain on initialization

### 3. All Agent SDKs
- No changes required - integration is transparent
- All agents automatically benefit from the new capabilities

## Configuration

### Environment Variables

Copy `.env.sap_ai_core` to `.env` and configure:

```bash
# Set execution mode
AIQ_LLM_MODE=auto  # auto, development, or production

# For Development (Grok4)
GROK_API_KEY=your-grok-api-key
XAI_API_KEY=your-xai-api-key  # Alternative

# For Production (SAP AI Core)
SAP_AI_CORE_API_KEY=your-sap-ai-core-api-key
SAP_AI_CORE_URL=https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com
SAP_AI_CORE_RESOURCE_GROUP=default
SAP_AI_CORE_DEPLOYMENT_ID=claude-opus-4-deployment
```

### Auto Mode Detection

The system automatically detects the environment:
- **BTP Environment**: Uses Production mode (SAP AI Core)
- **Local Environment**: Uses Development mode (Grok4)

Detection indicators:
- `VCAP_SERVICES` environment variable
- `VCAP_APPLICATION` environment variable
- `CF_INSTANCE_GUID` environment variable
- `BTP_CLUSTER_ID` environment variable

## Usage

### No Code Changes Required

All agents continue to work as before:

```python
# In any agent
await self.initialize_ai_intelligence()

# Use AI reasoning
result = await self.grok_client.analyze(prompt)
```

### Check Active Service

```python
# Get system status
status = self.grok_client.get_system_status()
print(f"Active service: {status['sap_ai_core']['mode']}")
print(f"Failover chain: {status['failover_chain']}")
```

## Benefits

1. **Enterprise Ready**: SAP AI Core integration for production
2. **Development Flexibility**: Grok4 for local development
3. **Reliability**: Automatic failover to LNN
4. **No Vendor Lock-in**: Easy to switch between services
5. **Cost Optimization**: Use appropriate service per environment
6. **Performance**: Response caching and optimization

## Monitoring

### Check Service Status

```python
# In any agent
system_status = self.grok_client.get_system_status()

# Returns:
{
    "sap_ai_core": {
        "available": true,
        "mode": "production",
        "info": { ... }
    },
    "grok_api": {
        "available": true,
        "model": "grok-2-1212"
    },
    "lnn_fallback": {
        "available": true,
        "trained": true
    },
    "failover_chain": ["SAP AI Core (Claude Opus 4)", "LNN", "Rule-based"]
}
```

### Failover Testing

```python
# Test failover readiness
readiness = await self.grok_client.check_failover_readiness()

# Force failover test
test_result = await self.grok_client.force_failover_test()
```

## Troubleshooting

### SAP AI Core Not Available

1. Check environment variables are set correctly
2. Verify SAP AI Core credentials
3. Check network connectivity to SAP AI Core endpoint
4. Verify deployment is running in SAP AI Core

### Grok API Not Available

1. Check GROK_API_KEY or XAI_API_KEY is set
2. Verify API key is valid
3. Check network connectivity to api.x.ai

### LNN Fallback Issues

1. Check if LNN model is trained: `grok_client.get_lnn_info()`
2. Train if needed: `await grok_client.train_lnn()`
3. Verify PyTorch is installed for LNN support

## Migration Checklist

- [x] Core GrokClient updated with SAP AI Core SDK
- [x] AIIntelligenceMixin updated to detect services
- [x] Environment configuration file created
- [x] Backward compatibility maintained
- [x] All agents automatically integrated
- [x] Failover chain implemented
- [x] Documentation updated

## Next Steps

1. Set environment variables from `.env.sap_ai_core`
2. Restart all agent services
3. Monitor logs for service detection
4. Test failover scenarios
5. Deploy to BTP for production use
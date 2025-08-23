# A2A Trust System Configuration

## Overview

The A2A agent system uses a centralized trust manager that provides cryptographic security for inter-agent communication. The trust system can be configured via environment variables to support different deployment scenarios.

## Environment Variables

### Core Trust Configuration

```bash
# Enable/disable trust system (default: true)
export ENABLE_TRUST="true"

# Require trust verification (default: false) 
export REQUIRE_TRUST="false"

# Environment mode (affects trust behavior)
export ENVIRONMENT="production"  # or "development"

# A2A Network path for trust components
export A2A_NETWORK_PATH="/path/to/a2aNetwork"
```

### Agent-Specific Timeouts

```bash
# Catalog Manager communication timeout
export A2A_CATALOG_MANAGER_TIMEOUT="30.0"

# ORD Registry timeout
export A2A_ORD_REGISTRY_TIMEOUT="30.0"

# Data product processing timeout
export A2A_DATA_PRODUCT_TIMEOUT="30.0"

# QA validation timeout  
export A2A_QA_VALIDATION_TIMEOUT="30.0"

# Data manager timeout
export A2A_DATA_MANAGER_TIMEOUT="30.0"

# HTTP client timeout (for general HTTP operations)
export A2A_HTTP_CLIENT_TIMEOUT="30.0"

# Circuit breaker timeout
export A2A_CIRCUIT_BREAKER_TIMEOUT="30.0"

# Calculation processing timeout
export A2A_CALCULATION_TIMEOUT="30.0"
```

### Blockchain Integration

```bash
# Agents that prefer blockchain messaging over HTTP
export BLOCKCHAIN_PREFERRED_AGENTS="sql_agent,data_product_agent_0"

# Agent URLs for HTTP fallback
export DATA_PRODUCT_AGENT_URL="http://localhost:8001"
export STANDARDIZATION_AGENT_URL="http://localhost:8002"
export AI_PREPARATION_AGENT_URL="http://localhost:8003"
export VECTOR_PROCESSING_AGENT_URL="http://localhost:8004"
export CATALOG_MANAGER_URL="http://localhost:8005"
export SQL_AGENT_URL="http://localhost:8006"
```

## Trust System Behavior

### Development Mode (`ENVIRONMENT=development`)

- **ENABLE_TRUST=false**: All trust operations return success without verification
- **ENABLE_TRUST=true, REQUIRE_TRUST=false**: Trust operations attempted, failures are logged but not fatal
- **ENABLE_TRUST=true, REQUIRE_TRUST=true**: Trust operations required, failures cause errors

### Production Mode (`ENVIRONMENT=production`)

- **ENABLE_TRUST=false**: Trust disabled (not recommended for production)
- **ENABLE_TRUST=true, REQUIRE_TRUST=false**: Trust enabled, failures logged as warnings
- **ENABLE_TRUST=true, REQUIRE_TRUST=true**: Trust required, failures cause fatal errors

## Trust Functions

### Available Functions

1. **`sign_a2a_message(message, agent_id)`**
   - Signs outgoing A2A messages with agent's private key
   - Returns signed message with cryptographic signature

2. **`initialize_agent_trust(agent_id, *args, **kwargs)`**
   - Initializes trust identity for an agent
   - Sets up cryptographic keys and blockchain registration

3. **`verify_a2a_message(message, agent_id)`**
   - Verifies incoming A2A message signatures
   - Returns verification result with signer information

### Usage Examples

```python
from app.a2a.core.trustManager import trust_manager, sign_a2a_message, verify_a2a_message

# Check if trust system is available
if trust_manager.is_available:
    # Sign a message
    signed_msg = sign_a2a_message(message_data, "my_agent_id")
    
    # Verify a message
    verification = verify_a2a_message(incoming_msg, "my_agent_id")
    if verification["valid"]:
        # Process trusted message
        pass
```

## Migration from Legacy Trust

### What Changed

1. **Removed fallback functions**: No more mock/dev fallback implementations in agent files
2. **Centralized configuration**: All trust behavior controlled via environment variables
3. **Explicit error handling**: Clear distinction between development and production behavior
4. **Consistent imports**: All agents use `from app.a2a.core.trustManager import ...`

### Migration Steps

1. **Update imports** in agent files:
   ```python
   # OLD - with fallback functions
   try:
       from trustSystem.smartContractTrust import sign_a2a_message
   except ImportError:
       def sign_a2a_message(*args, **kwargs):
           return {"signature": "dev_fallback"}
   
   # NEW - centralized trust manager
   from app.a2a.core.trustManager import sign_a2a_message
   ```

2. **Configure environment variables** for your deployment

3. **Test trust behavior** in development and production modes

## Security Considerations

### Production Deployment

- **Always set `ENVIRONMENT=production`** in production
- **Use `REQUIRE_TRUST=true`** for high-security environments
- **Ensure A2A_NETWORK_PATH** points to valid trust system components
- **Monitor trust verification failures** in logs

### Development

- **Use `REQUIRE_TRUST=false`** for development flexibility
- **Set `ENABLE_TRUST=false`** for offline development
- **Review trust warnings** in development logs

## Troubleshooting

### Common Issues

1. **Trust system unavailable**
   ```
   TrustSystemError: Trust system is required but unavailable
   ```
   - Check A2A_NETWORK_PATH is correct
   - Verify trust system components are installed
   - Set REQUIRE_TRUST=false for development

2. **Message verification failures**
   ```
   Message verification failed, continuing without trust
   ```
   - Check agent trust initialization
   - Verify blockchain connectivity
   - Review message signing process

3. **Blockchain communication errors**
   ```
   Blockchain sending failed, falling back to HTTP
   ```
   - Check blockchain node connectivity
   - Verify agent blockchain registration
   - Review BLOCKCHAIN_PREFERRED_AGENTS configuration

### Debug Commands

```bash
# Test trust manager import
python3 -c "from app.a2a.core.trustManager import trust_manager; print(f'Trust available: {trust_manager.is_available}')"

# Check environment configuration
env | grep -E "(TRUST|A2A_)" | sort

# Verify agent trust initialization
python3 -c "from app.a2a.core.trustManager import initialize_agent_trust; print(initialize_agent_trust('test_agent'))"
```

## Best Practices

1. **Use environment-specific configurations**
2. **Monitor trust system health** in production
3. **Implement proper error handling** for trust failures
4. **Keep trust system components updated**
5. **Test trust behavior** during deployment
6. **Use blockchain messaging** for critical operations
7. **Configure appropriate timeouts** for network calls
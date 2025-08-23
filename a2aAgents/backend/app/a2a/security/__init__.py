"""
A2A Security Components
Delegated to a2aNetwork trust system
"""

# Import from a2aNetwork trust system
try:
    import sys
    import os
    from app.core.config import settings
    
    # Use configured path instead of hardcoded
    a2a_network_path = settings.A2A_NETWORK_PATH
    if os.path.exists(a2a_network_path):
        sys.path.insert(0, a2a_network_path)
    else:
        raise ImportError(f"A2A Network path not found: {a2a_network_path}")
    
    # Import individual modules and re-export functions
    from trustSystem import smartContractTrust
    from trustSystem import delegationContracts
    from trustSystem import sharedTrust
    
    # Re-export key functions for backward compatibility
    from trustSystem.smartContractTrust import (
        sign_a2a_message, verify_a2a_message, initialize_agent_trust, 
        get_trust_contract
    )
    from trustSystem.delegationContracts import (


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        get_delegation_contract, DelegationAction, can_agent_delegate, 
        record_delegation_usage, create_delegation_contract
    )
    
    print("✅ Using a2aNetwork trust system")
except ImportError as e:
    print(f"⚠️  a2aNetwork trust system not available: {e}")
    raise ImportError("Trust system not available from a2aNetwork")

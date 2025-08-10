"""
A2A Security Components
Delegated to a2aNetwork trust system
"""

# Import from a2aNetwork trust system
try:
    import sys
    sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")
    
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
        get_delegation_contract, DelegationAction, can_agent_delegate, 
        record_delegation_usage, create_delegation_contract
    )
    
    print("✅ Using a2aNetwork trust system")
except ImportError as e:
    print(f"⚠️  a2aNetwork trust system not available: {e}")
    raise ImportError("Trust system not available from a2aNetwork")

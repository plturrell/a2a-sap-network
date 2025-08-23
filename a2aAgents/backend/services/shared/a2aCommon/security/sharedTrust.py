"""
Shared Trust Contract Instance
Ensures all agents use the same trust contract
"""

import os
from .smartContractTrust import SmartContractTrust

# Environment variable to store contract ID
TRUST_CONTRACT_ID_ENV = "A2A_TRUST_CONTRACT_ID"

# Singleton trust contract instance
_shared_trust_contract = None


def get_shared_trust_contract():
    """Get or create the shared trust contract instance"""
    global _shared_trust_contract
    
    if _shared_trust_contract is None:
        # Check if we have an existing contract ID
        existing_contract_id = os.environ.get(TRUST_CONTRACT_ID_ENV)
        
        _shared_trust_contract = SmartContractTrust()
        
        if existing_contract_id:
            # Restore existing contract ID
            _shared_trust_contract.contract_id = existing_contract_id
        else:
            # Save new contract ID to environment
            os.environ[TRUST_CONTRACT_ID_ENV] = _shared_trust_contract.contract_id
    
    return _shared_trust_contract


def reset_shared_trust_contract():
    """Reset the shared trust contract (for testing)"""
    global _shared_trust_contract
    _shared_trust_contract = None
    if TRUST_CONTRACT_ID_ENV in os.environ:
        del os.environ[TRUST_CONTRACT_ID_ENV]
"""
A2A Security Module - Smart Contract Trust and Cryptographic Message Signing
"""

from .smartContractTrust import (
    SmartContractTrust,
    AgentIdentity,
    get_trust_contract,
    initialize_agent_trust,
    sign_a2a_message,
    verify_a2a_message
)

__all__ = [
    "SmartContractTrust",
    "AgentIdentity", 
    "get_trust_contract",
    "initialize_agent_trust",
    "sign_a2a_message",
    "verify_a2a_message"
]
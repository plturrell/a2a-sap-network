"""
A2A Blockchain Integration Package
Provides Web3 integration for finsight_cib agents to use a2a_network smart contracts
"""

from .web3_client import (
    A2ABlockchainClient,
    AgentIdentity,
    get_blockchain_client,
    initialize_blockchain_client
)
from .ord_blockchain_adapter import (
    ORDBlockchainAdapter,
    create_ord_blockchain_adapter
)

__all__ = [
    "A2ABlockchainClient",
    "AgentIdentity", 
    "get_blockchain_client",
    "initialize_blockchain_client",
    "ORDBlockchainAdapter",
    "create_ord_blockchain_adapter"
]
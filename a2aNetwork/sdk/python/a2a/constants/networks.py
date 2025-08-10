"""
A2A Network Constants - Networks

Network configuration constants.
"""

NETWORKS = {
    'mainnet': {
        'chain_id': 1,
        'rpc_urls': ['https://mainnet.infura.io/v3/'],
        'poa': False
    },
    'goerli': {
        'chain_id': 5,
        'rpc_urls': ['https://goerli.infura.io/v3/'],
        'poa': True
    },
    'sepolia': {
        'chain_id': 11155111,
        'rpc_urls': ['https://sepolia.infura.io/v3/'],
        'poa': False
    },
    'localhost': {
        'chain_id': 31337,
        'rpc_urls': ['http://localhost:8545'],
        'poa': False
    }
}

DEFAULT_NETWORK = 'localhost'

__all__ = ['NETWORKS', 'DEFAULT_NETWORK']
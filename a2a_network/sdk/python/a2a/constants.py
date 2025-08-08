"""
A2A SDK Constants
"""
from enum import Enum


class MessageType(Enum):
    """Message types for A2A communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ErrorCategory(Enum):
    """Error categories for debugging"""
    NETWORK = "network"
    BLOCKCHAIN = "blockchain"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    CONTRACT = "contract"
    AGENT = "agent"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'response_time': {
        'excellent': 0.1,
        'good': 0.5,
        'acceptable': 1.0,
        'poor': 5.0
    },
    'success_rate': {
        'excellent': 0.99,
        'good': 0.95,
        'acceptable': 0.90,
        'poor': 0.80
    },
    'throughput': {
        'excellent': 100,
        'good': 50,
        'acceptable': 10,
        'poor': 1
    }
}

# Default configuration values
DEFAULT_CONFIG = {
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit': 100,
    'batch_size': 10,
    'max_connections': 50,
    'heartbeat_interval': 30,
    'debug_trace_limit': 1000,
    'performance_sample_rate': 0.1
}

# Contract event signatures - Generated from contract ABIs
# Use: web3.keccak(text="EventName(type1,type2,...)").hex()
CONTRACT_EVENTS = {
    # AgentRegistry contract events
    'AgentRegistered': "0xd983b353843079ea72b0607ed788e429e4b01c5c7f6156f061b2989604f319ea",  # AgentRegistered(address,string,string[],uint256)
    'AgentUpdated': "0xbaef45cead810df032ca6f6dc22d0fa78f7bc08a3e2f3177cfe2f37f4b08071a",     # AgentUpdated(address,uint256)
    
    # MessageRouter contract events  
    'MessageSent': "0xd8331239dcfe177d02e949e3cfa5969f4a3c658a80554db93c64ab76f7acf9c5",      # MessageSent(bytes32,address,address,bytes,uint256)
    'MessageDelivered': "0x3556f710055b94163d2468428dd83a9be7548047bb5242c0fb714773ee86497b", # MessageDelivered(bytes32,address,bool)
    
    # Task management events
    'TaskCreated': "0x5a4db692818f1938ef550c82b67bf87d982f1b039bee7bd09502498ac9b80936",      # TaskCreated(bytes32,address,address,string,uint256)
    'TaskCompleted': "0xa9442c3bfbd3ea4b685ba0781c4cf38e46e10d27d6b11fea6b9f05eee402fbcc",    # TaskCompleted(bytes32,address,bool,bytes)
    
    # Reputation events
    'ReputationUpdated': "0x5b88a2e3fc1a53234357ab78c104df11c33ccfa79886793654565ad70b8afb6e" # ReputationUpdated(address,uint256,uint256)
}

# Error codes
ERROR_CODES = {
    'CONNECTION_FAILED': 1001,
    'INVALID_MESSAGE': 1002,
    'AGENT_NOT_FOUND': 1003,
    'AUTHENTICATION_FAILED': 1004,
    'RATE_LIMIT_EXCEEDED': 1005,
    'CONTRACT_ERROR': 1006,
    'TIMEOUT': 1007,
    'PERMISSION_DENIED': 1008,
    'INVALID_RESPONSE': 1009,
    'NETWORK_ERROR': 1010
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'update_interval': 5,  # seconds
    'chart_history': 100,  # data points
    'alert_thresholds': {
        'error_rate': 0.05,
        'response_time': 2.0,
        'cpu_usage': 0.8,
        'memory_usage': 0.85
    },
    'colors': {
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8',
        'primary': '#007bff'
    }
}

# Blockchain network configuration
NETWORK_CONFIG = {
    'mainnet': {
        'chain_id': 1,
        'rpc_url': 'https://mainnet.infura.io/v3/',
        'block_confirmations': 12
    },
    'testnet': {
        'chain_id': 5,
        'rpc_url': 'https://goerli.infura.io/v3/',
        'block_confirmations': 3
    },
    'local': {
        'chain_id': 1337,
        'rpc_url': 'http://localhost:8545',
        'block_confirmations': 1
    }
}
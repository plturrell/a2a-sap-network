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

# Contract event signatures
CONTRACT_EVENTS = {
    'AgentRegistered': '0x...',  # Event signatures would be filled in
    'MessageSent': '0x...',
    'MessageDelivered': '0x...',
    'TaskCreated': '0x...',
    'TaskCompleted': '0x...',
    'ReputationUpdated': '0x...'
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
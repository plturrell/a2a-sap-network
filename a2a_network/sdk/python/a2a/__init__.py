"""
A2A Network Python SDK

Official Python SDK for integrating with A2A Network infrastructure.
Provides easy-to-use interfaces for agent management, messaging, token operations,
governance participation, and network scalability features.

Example:
    >>> from a2a import A2AClient
    >>> client = A2AClient({
    ...     'network': 'mainnet',
    ...     'private_key': 'your_private_key'
    ... })
    >>> await client.connect()
    >>> agents = await client.agents.get_all_agents()
"""

from .client import A2AClient
from .services import (
    AgentManager,
    MessageManager,
    TokenManager,
    GovernanceManager,
    ScalabilityManager,
    ReputationManager
)
from .utils import (
    crypto,
    validation,
    formatting,
    errors
)
from .constants import (
    networks,
    contracts,
    config
)

__version__ = "1.0.0"
__author__ = "A2A Network Team"
__email__ = "dev@a2a.network"
__license__ = "MIT"

__all__ = [
    # Main client
    "A2AClient",
    
    # Service managers
    "AgentManager",
    "MessageManager", 
    "TokenManager",
    "GovernanceManager",
    "ScalabilityManager",
    "ReputationManager",
    
    # Utility modules
    "crypto",
    "validation", 
    "formatting",
    "errors",
    
    # Constants
    "networks",
    "contracts",
    "config",
    
    # Version info
    "__version__"
]

# Package metadata
__title__ = "a2a-network-sdk"
__description__ = "Official Python SDK for A2A Network"
__url__ = "https://github.com/a2a-network/sdk-python"
__version_info__ = tuple(int(i) for i in __version__.split('.'))

# Supported Python versions
__python_requires__ = ">=3.8"

# SDK configuration
DEFAULT_CONFIG = {
    'api_timeout': 30,
    'retry_attempts': 3,
    'auto_reconnect': True,
    'rate_limits': {
        'requests_per_second': 10,
        'requests_per_minute': 600,
        'requests_per_hour': 36000
    },
    'caching': {
        'enabled': True,
        'ttl': 300,  # 5 minutes
        'max_size': 1000
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

# Supported networks
SUPPORTED_NETWORKS = [
    'mainnet',
    'goerli', 
    'sepolia',
    'polygon',
    'polygon-mumbai',
    'bsc',
    'bsc-testnet',
    'arbitrum',
    'arbitrum-goerli',
    'optimism',
    'optimism-goerli',
    'localhost'
]

# Connection states
class ConnectionState:
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting' 
    CONNECTED = 'connected'
    ERROR = 'error'

# Message types
class MessageType:
    DIRECT = 'direct'
    BROADCAST = 'broadcast'
    REPLY = 'reply'
    SYSTEM = 'system'
    TASK_REQUEST = 'task_request'
    TASK_RESPONSE = 'task_response'
    NOTIFICATION = 'notification'
    FILE_TRANSFER = 'file_transfer'

# Message status
class MessageStatus:
    PENDING = 'pending'
    SENT = 'sent'
    DELIVERED = 'delivered'
    READ = 'read'
    FAILED = 'failed'
    DELETED = 'deleted'
    EXPIRED = 'expired'

# Agent status
class AgentStatus:
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    SUSPENDED = 'suspended'
    MAINTENANCE = 'maintenance'
    PENDING_APPROVAL = 'pending_approval'
    DEREGISTERED = 'deregistered'

# Task status
class TaskStatus:
    PENDING = 'pending'
    ACCEPTED = 'accepted'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    DISPUTED = 'disputed'
    FAILED = 'failed'

# Governance proposal status
class ProposalStatus:
    PENDING = 'pending'
    ACTIVE = 'active'
    CANCELLED = 'cancelled'
    DEFEATED = 'defeated'
    SUCCEEDED = 'succeeded'
    QUEUED = 'queued'
    EXPIRED = 'expired'
    EXECUTED = 'executed'

# Export commonly used classes and constants
__all__.extend([
    'DEFAULT_CONFIG',
    'SUPPORTED_NETWORKS', 
    'ConnectionState',
    'MessageType',
    'MessageStatus',
    'AgentStatus',
    'TaskStatus',
    'ProposalStatus'
])

# Module-level convenience functions
def create_client(config=None):
    """
    Create a new A2A client instance with optional configuration.
    
    Args:
        config (dict, optional): Client configuration options
        
    Returns:
        A2AClient: Configured client instance
        
    Example:
        >>> client = create_client({
        ...     'network': 'mainnet',
        ...     'private_key': 'your_private_key'
        ... })
    """
    return A2AClient(config or {})

async def quick_connect(network='mainnet', private_key=None, rpc_url=None):
    """
    Quick connection helper for common use cases.
    
    Args:
        network (str): Network to connect to
        private_key (str, optional): Private key for signing
        rpc_url (str, optional): Custom RPC URL
        
    Returns:
        A2AClient: Connected client instance
        
    Example:
        >>> client = await quick_connect('mainnet', 'your_private_key')
    """
    config = {'network': network}
    if private_key:
        config['private_key'] = private_key
    if rpc_url:
        config['rpc_url'] = rpc_url
        
    client = A2AClient(config)
    await client.connect()
    return client

# Add convenience functions to __all__
__all__.extend(['create_client', 'quick_connect'])
"""
A2A SDK Type Definitions
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime


class MessageStatus(Enum):
    """Message status enumeration"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


class MessageType(Enum):
    """Message type enumeration"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class TaskStatus(Enum):
    """Task status enumeration"""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Agent:
    """Agent data structure"""
    address: str
    name: str
    description: str
    capabilities: List[str]
    endpoint: str
    status: AgentStatus
    reputation: int
    last_active: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'address': self.address,
            'name': self.name,
            'description': self.description,
            'capabilities': self.capabilities,
            'endpoint': self.endpoint,
            'status': self.status.value,
            'reputation': self.reputation,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'metadata': self.metadata or {}
        }


@dataclass
class Message:
    """Message data structure"""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    payload: Dict[str, Any]
    status: MessageStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'gas_used': self.gas_used,
            'error': self.error
        }


@dataclass
class Task:
    """Task data structure"""
    id: str
    description: str
    requester: str
    assigned_agent: Optional[str]
    status: TaskStatus
    created_at: datetime
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'description': self.description,
            'requester': self.requester,
            'assigned_agent': self.assigned_agent,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result
        }


@dataclass
class BlockchainEvent:
    """Blockchain event data structure"""
    block_number: int
    transaction_hash: str
    contract_address: str
    event_name: str
    args: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'block_number': self.block_number,
            'transaction_hash': self.transaction_hash,
            'contract_address': self.contract_address,
            'event_name': self.event_name,
            'args': self.args,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceSnapshot:
    """Performance snapshot data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_latency: float
    active_connections: int
    request_count: int
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'active_connections': self.active_connections,
            'request_count': self.request_count,
            'error_count': self.error_count
        }


# Type aliases for common patterns
AgentAddress = str
MessageId = str
TaskId = str
TransactionHash = str
BlockNumber = int
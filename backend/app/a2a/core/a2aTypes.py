"""
Shared A2A types to avoid circular imports
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class MessagePart(BaseModel):
    kind: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    parts: List[MessagePart]
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    signature: Optional[str] = None

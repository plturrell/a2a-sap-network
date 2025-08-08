"""
A2A Agent SDK
Simplifies development of new agents in the A2A network
"""

from .agent_base import A2AAgentBase
from .client import A2AClient
from .decorators import a2a_handler, a2a_task, a2a_skill
from .types import (
    A2AMessage, MessagePart, MessageRole, TaskStatus,
    AgentCard, AgentCapability, SkillDefinition
)
from .utils import create_agent_id, validate_message, sign_message

__version__ = "1.0.0"
__all__ = [
    "A2AAgentBase",
    "A2AClient", 
    "a2a_handler",
    "a2a_task",
    "a2a_skill",
    "A2AMessage",
    "MessagePart", 
    "MessageRole",
    "TaskStatus",
    "AgentCard",
    "AgentCapability",
    "SkillDefinition",
    "create_agent_id",
    "validate_message",
    "sign_message"
]
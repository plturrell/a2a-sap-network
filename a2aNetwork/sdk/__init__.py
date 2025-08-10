"""
A2A Agent SDK
Simplifies development of new agents in the A2A network
"""

# Import types (these always work)
from .types import A2AMessage, MessagePart, MessageRole
from .types import (
    TaskStatus, AgentCard, AgentCapability, SkillDefinition
)

# Import utilities
from .utils import create_agent_id, validate_message, sign_message

# Conditional imports for components that need relative imports
try:
    from .agentBase import A2AAgentBase
except ImportError as e:
    print(f"Warning: A2AAgentBase not available: {e}")
    A2AAgentBase = None

try:
    from .client import A2AClient
except ImportError as e:
    print(f"Warning: A2AClient not available: {e}")
    A2AClient = None

try:
    from .decorators import a2a_handler, a2a_task, a2a_skill
except ImportError as e:
    print(f"Warning: SDK decorators not available: {e}")
    a2a_handler = a2a_task = a2a_skill = None

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
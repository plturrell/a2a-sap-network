"""
A2A Common Library
Shared components for Agent-to-Agent (A2A) protocol implementation
"""

__version__ = "0.2.9"

# Export main components for easy access
from .sdk import A2AAgentBase, a2a_handler, a2a_skill, a2a_task
from .core.a2a_types import A2AMessage, MessageRole, MessagePart
from .security.smart_contract_trust import initialize_agent_trust, verify_a2a_message, sign_a2a_message

__all__ = [
    "A2AAgentBase",
    "a2a_handler", 
    "a2a_skill",
    "a2a_task",
    "A2AMessage",
    "MessageRole",
    "MessagePart",
    "initialize_agent_trust",
    "verify_a2a_message",
    "sign_a2a_message",
]
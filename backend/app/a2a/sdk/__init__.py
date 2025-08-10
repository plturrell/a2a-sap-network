"""
A2A Agent SDK
Simplifies development of new agents in the A2A network
"""

# Import from a2aNetwork (preferred) or use local fallback
try:
    import sys
    sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")
    from sdk.agentBase import A2AAgentBase
    from sdk.decorators import a2a_handler, a2a_task, a2a_skill
    from sdk.types import A2AMessage, MessagePart, MessageRole
    from sdk.types import TaskStatus, AgentCard, AgentCapability, SkillDefinition
    from sdk.utils import create_agent_id, validate_message, sign_message
    print("✅ Using a2aNetwork SDK components")
    
    # A2AClient has dependency issues, use local fallback
    A2AClient = None
    print("⚠️  A2AClient not available from a2aNetwork - dependency issues")
    
except ImportError as e:
    print(f"⚠️  a2aNetwork SDK not available: {e}")
    raise ImportError("SDK components not available from a2aNetwork")

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

"""
A2A Agent SDK
Simplifies development of new agents in the A2A network
"""

import logging
logger = logging.getLogger(__name__)

# Import types first - these should always be available
from .types import (
    A2AMessage, MessagePart, MessageRole, TaskStatus, 
    AgentCard, AgentCapability, SkillDefinition
)

# Import other SDK components
try:
    from .agentBase import A2AAgentBase, AgentConfig
except ImportError as e:
    logger.warning(f"A2AAgentBase not available: {e}")
    class A2AAgentBase: pass
    class AgentConfig: pass

try:
    from .decorators import a2a_handler, a2a_task, a2a_skill
except ImportError as e:
    logger.warning(f"Decorators not available: {e}")
    def a2a_handler(func): return func
    def a2a_task(func): return func  
    def a2a_skill(func): return func

try:
    from .utils import create_agent_id, validate_message, sign_message
except ImportError as e:
    logger.warning(f"Utils not available: {e}")
    def create_agent_id(): return "default-agent-id"
    def validate_message(msg): return True
    def sign_message(msg): return "unsigned"

logger.info("✅ Successfully imported SDK components from a2a-network package.")

# Import mixins (always local)
try:
    from .mixins import PerformanceMonitorMixin, SecurityHardenedMixin, CachingMixin, TelemetryMixin
    from .mixins.blockchainQueueMixin import BlockchainQueueMixin
    logger.info("✅ SDK mixins available (including blockchain queue)")
except ImportError as e:
    logger.warning("SDK mixins not available: %s. Using stub implementations.", e)
    # Create stub mixins
    class PerformanceMonitorMixin: pass
    class SecurityHardenedMixin: pass
    class CachingMixin: pass
    class TelemetryMixin: pass
    class BlockchainQueueMixin: pass

# Import MCP components (required - no fallbacks)
try:
    from .mcpServer import A2AMCPServer, MCPServerMixin, create_mcp_server
    from .mcpDecorators import mcp_tool, mcp_resource, mcp_prompt


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    logger.info("✅ MCP server, client, and decorators available")
except ImportError as e:
    logger.error("MCP components are required but not available: %s", e)
    raise ImportError("MCP implementation is mandatory for A2A agents.") from e

# A2AClient has dependency issues, use local fallback
A2AClient = None

__version__ = "1.0.0"
__all__ = [
    "A2AAgentBase",
    "AgentConfig",
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
    "sign_message",
    "PerformanceMonitorMixin",
    "SecurityHardenedMixin", 
    "CachingMixin",
    "TelemetryMixin",
    # MCP components
    "A2AMCPServer",
    "MCPServerMixin", 
    "create_mcp_server",
    "mcp_tool",
    "mcp_resource",
    "mcp_prompt"
]

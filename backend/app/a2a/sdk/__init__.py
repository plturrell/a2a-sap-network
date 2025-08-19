"""
A2A Agent SDK
Simplifies development of new agents in the A2A network
"""

import logging
logger = logging.getLogger(__name__)

# Imports from local SDK components
try:
    from .agentBase import A2AAgentBase
    from .decorators import a2a_handler, a2a_task, a2a_skill  
    from .types import A2AMessage, MessagePart, MessageRole, TaskStatus, AgentCard, AgentCapability, SkillDefinition
    from .utils import create_agent_id, validate_message, sign_message
except ImportError as e:
    logger.warning(f"Some SDK components not available: {e}")
    # Create minimal stubs to prevent import failures
    class A2AAgentBase: pass
    def a2a_handler(func): return func
    def a2a_task(func): return func  
    def a2a_skill(func): return func

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
    logger.info("✅ MCP server, client, and decorators available")
except ImportError as e:
    logger.error("MCP components are required but not available: %s", e)
    raise ImportError("MCP implementation is mandatory for A2A agents.") from e

# A2AClient has dependency issues, use local fallback
A2AClient = None

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

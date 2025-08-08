"""
A2A (Agent-to-Agent) Network Package

This package provides a comprehensive framework for building and managing
agent-to-agent communication networks in enterprise environments.

Core Components:
- agents: Specialized data processing agents (Agent 0-5)
- core: Core infrastructure and utilities
- sdk: Software Development Kit for building custom agents
- security: Trust and security management
- skills: Reusable agent capabilities
"""

__version__ = "0.2.9"
__author__ = "A2A Development Team"
__email__ = "dev@a2a-network.com"

# Core imports for easy access
from .core.a2a_types import (
    AgentType,
    MessageType,
    ProcessingStatus,
    TaskPriority,
)

# Agent imports
from .agents.agent0_data_product.active.data_product_agent_sdk import (
    DataProductRegistrationAgentSDK,
)
from .agents.agent1_standardization.active.data_standardization_agent_sdk import (
    DataStandardizationAgentSDK,
)
from .agents.agent2_ai_preparation.active.ai_preparation_agent_sdk import (
    AiPreparationAgentSDK,
)
from .agents.agent3_vector_processing.active.vector_processing_agent_sdk import (
    VectorProcessingAgentSDK,
)

# SDK imports
from .sdk.agent_base import BaseAgent
from .sdk.client import A2AClient
from .sdk.decorators import agent_endpoint, require_auth

# Utils
from .core.response_mapper import ResponseMapper
from .core.workflow_context import WorkflowContext

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Types
    "AgentType",
    "MessageType",
    "ProcessingStatus", 
    "TaskPriority",
    
    # Agents
    "DataProductRegistrationAgentSDK",
    "DataStandardizationAgentSDK",
    "AiPreparationAgentSDK",
    "VectorProcessingAgentSDK",
    
    # SDK
    "BaseAgent",
    "A2AClient",
    "agent_endpoint",
    "require_auth",
    
    # Utils
    "ResponseMapper",
    "WorkflowContext",
]
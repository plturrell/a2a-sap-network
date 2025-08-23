"""
SDK Import Handler
REQUIRES A2A SDK - NO FALLBACK IMPLEMENTATIONS
"""

import logging
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class SDKImportHandler:
    """Handle SDK imports - REQUIRES A2A SDK (NO FALLBACKS)"""
    
    @staticmethod
    def import_sdk_types():
        """Import SDK types - MUST have real A2A SDK available"""
        from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus, AgentCard
        logger.info("Successfully imported SDK types")
        return {
            "A2AMessage": A2AMessage,
            "MessagePart": MessagePart,
            "MessageRole": MessageRole,
            "TaskStatus": TaskStatus,
            "AgentCard": AgentCard,
            "using_fallback": False
        }
    
    @staticmethod
    def import_reasoning_skills():
        """Import reasoning skills - REQUIRES real skills (NO FALLBACKS)"""
        from .reasoningSkills import (
            MultiAgentReasoningSkills, ReasoningOrchestrationSkills,
            HierarchicalReasoningSkills, SwarmReasoningSkills
        )
        from .enhancedReasoningSkills import EnhancedReasoningSkills
        
        logger.info("Successfully imported reasoning skills")
        return {
            "MultiAgentReasoningSkills": MultiAgentReasoningSkills,
            "ReasoningOrchestrationSkills": ReasoningOrchestrationSkills,
            "HierarchicalReasoningSkills": HierarchicalReasoningSkills,
            "SwarmReasoningSkills": SwarmReasoningSkills,
            "EnhancedReasoningSkills": EnhancedReasoningSkills,
            "using_fallback": False
        }
    
    @staticmethod
    def import_mcp_coordination():
        """Import MCP coordination - REQUIRES real MCP coordination (NO FALLBACKS)"""
        from app.a2a.sdk.mcpSkillCoordination import MCPSkillCoordinationMixin, skill_depends_on, skill_provides, MCPSkillClientMixin
        
        logger.info("Successfully imported MCP coordination")
        return {
            "MCPSkillCoordinationMixin": MCPSkillCoordinationMixin,
            "MCPSkillClientMixin": MCPSkillClientMixin,
            "skill_depends_on": skill_depends_on,
            "skill_provides": skill_provides,
            "using_fallback": False
        }
    
    @staticmethod
    def check_imports_status() -> Dict[str, bool]:
        """Check status of all imports"""
        sdk_types = SDKImportHandler.import_sdk_types()
        reasoning_skills = SDKImportHandler.import_reasoning_skills()
        mcp_coordination = SDKImportHandler.import_mcp_coordination()
        
        return {
            "sdk_types_available": not sdk_types["using_fallback"],
            "reasoning_skills_available": not reasoning_skills["using_fallback"],
            "mcp_coordination_available": not mcp_coordination["using_fallback"],
            "all_imports_successful": not any([
                sdk_types["using_fallback"],
                reasoning_skills["using_fallback"],
                mcp_coordination["using_fallback"]
            ])
        }


# Helper function for easy import
def safe_import_sdk():
    """Safely import all SDK components with fallbacks"""
    handler = SDKImportHandler()
    
    sdk_types = handler.import_sdk_types()
    reasoning_skills = handler.import_reasoning_skills()
    mcp_coordination = handler.import_mcp_coordination()
    
    # Log import status
    status = handler.check_imports_status()
    if status["all_imports_successful"]:
        logger.info("All SDK imports successful")
    else:
        logger.warning(f"Some imports using fallbacks: {status}")
    
    # Return all imports
    imports = {}
    imports.update(sdk_types)
    imports.update(reasoning_skills)
    imports.update(mcp_coordination)
    
    return imports
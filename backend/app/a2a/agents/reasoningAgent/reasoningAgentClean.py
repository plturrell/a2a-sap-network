"""
Reasoning Agent - A2A Agent for Advanced Reasoning
This is a pure A2A agent that discovers and uses MCP skills for reasoning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

# Import A2A SDK components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, 
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.mixins import (
    PerformanceMonitorMixin, SecurityHardenedMixin, TelemetryMixin
)

logger = logging.getLogger(__name__)


class ReasoningArchitecture(Enum):
    """Available reasoning architectures"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    BLACKBOARD = "blackboard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SWARM = "swarm"
    DEBATE = "debate"


class ReasoningAgent(A2AAgentBase, PerformanceMonitorMixin, SecurityHardenedMixin, TelemetryMixin):
    """
    Pure A2A agent for reasoning tasks
    Uses MCP protocol to discover and call reasoning skills
    """
    
    def __init__(self, name: str = "ReasoningAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        
        
        # Available MCP skills (discovered via MCP)
        self.available_skills = {}
        
        # Agent metadata
        self.agent_type = "reasoning"
        self.capabilities = {
            "multi_architecture": True,
            "hypothesis_generation": True,
            "debate_orchestration": True,
            "chain_analysis": True
        }
        
        logger.info(f"Initialized {name} as pure A2A agent with MCP client")
    
    async def initialize(self):
        """Initialize agent and discover MCP skills"""
        await super().initialize()
        
        # Discover available MCP skills
        await self._discover_mcp_skills()
        
        logger.info(f"Agent initialized with {len(self.available_skills)} discovered skills")
    
    async def _discover_mcp_skills(self):
        """Discover available MCP skills via protocol"""
        try:
            # In a real implementation, this would use MCP discovery protocol
            # For now, we'll check for known skill endpoints
            skill_endpoints = [
                "advanced_reasoning",
                "hypothesis_generation",
                "debate_orchestration",
                "reasoning_chain_analysis"
            ]
            
            for skill_name in skill_endpoints:
                # Check if skill is available via MCP
                skill_info = await self.mcp_client.discover_skill(skill_name)
                if skill_info:
                    self.available_skills[skill_name] = skill_info
                    logger.info(f"Discovered MCP skill: {skill_name}")
        
        except Exception as e:
            logger.error(f"Error discovering MCP skills: {e}")
    
    @a2a_handler(role=MessageRole.REASONING)
    async def handle_reasoning_request(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Handle reasoning requests via A2A protocol
        Delegates to appropriate MCP skills
        """
        try:
            content = message.content
            question = content.get("question", "")
            architecture = content.get("architecture", ReasoningArchitecture.HIERARCHICAL.value)
            context = content.get("context", {})
            
            # Call appropriate MCP skill
            if "advanced_reasoning" in self.available_skills:
                result = await self.mcp_client.call_skill(
                    "advanced_reasoning",
                    question=question,
                    reasoning_architecture=architecture,
                    context=context
                )
                
                return {
                    "success": True,
                    "result": result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Advanced reasoning skill not available",
                    "agent_id": self.agent_id
                }
                
        except Exception as e:
            logger.error(f"Error handling reasoning request: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    @a2a_handler(role=MessageRole.HYPOTHESIS)
    async def handle_hypothesis_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle hypothesis generation requests"""
        try:
            content = message.content
            problem = content.get("problem", "")
            domain = content.get("domain")
            
            if "hypothesis_generation" in self.available_skills:
                result = await self.mcp_client.call_skill(
                    "hypothesis_generation",
                    problem=problem,
                    domain=domain
                )
                
                return {
                    "success": True,
                    "result": result,
                    "agent_id": self.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": "Hypothesis generation skill not available",
                    "agent_id": self.agent_id
                }
                
        except Exception as e:
            logger.error(f"Error handling hypothesis request: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    @a2a_handler(role=MessageRole.DEBATE)
    async def handle_debate_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle debate orchestration requests"""
        try:
            content = message.content
            topic = content.get("topic", "")
            max_rounds = content.get("max_rounds", 5)
            
            if "debate_orchestration" in self.available_skills:
                result = await self.mcp_client.call_skill(
                    "debate_orchestration",
                    topic=topic,
                    max_rounds=max_rounds
                )
                
                return {
                    "success": True,
                    "result": result,
                    "agent_id": self.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": "Debate orchestration skill not available",
                    "agent_id": self.agent_id
                }
                
        except Exception as e:
            logger.error(f"Error handling debate request: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    @a2a_handler(role=MessageRole.ANALYSIS)
    async def handle_analysis_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle reasoning chain analysis requests"""
        try:
            content = message.content
            reasoning_steps = content.get("reasoning_steps", [])
            
            if "reasoning_chain_analysis" in self.available_skills:
                result = await self.mcp_client.call_skill(
                    "reasoning_chain_analysis",
                    reasoning_steps=reasoning_steps
                )
                
                return {
                    "success": True,
                    "result": result,
                    "agent_id": self.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": "Chain analysis skill not available",
                    "agent_id": self.agent_id
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": "active",
            "discovered_skills": list(self.available_skills.keys()),
            "capabilities": self.capabilities,
            "uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info(f"Shutting down {self.name}")
        await super().shutdown()


# Factory function
def create_reasoning_agent(name: str = "ReasoningAgent", **kwargs) -> ReasoningAgent:
    """Create a reasoning agent instance"""
    return ReasoningAgent(name=name, **kwargs)
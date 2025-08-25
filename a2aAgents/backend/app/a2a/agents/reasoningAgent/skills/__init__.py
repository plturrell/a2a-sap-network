"""
MCP Skills for Reasoning Agent
These skills are exposed via MCP protocol and discovered/used by the A2A reasoning agent
"""

from .advancedReasoning import advanced_reasoning
from .hypothesisGeneration import hypothesis_generation
from .debateOrchestration import debate_orchestration
from .reasoningChainAnalysis import reasoning_chain_analysis

__all__ = [
    "advanced_reasoning",
    "hypothesis_generation",
    "debate_orchestration",
    "reasoning_chain_analysis"
]

# MCP skill registry for discovery
MCP_SKILLS = {
    "advanced_reasoning": {
        "function": advanced_reasoning,
        "category": "reasoning",
        "complexity": "high"
    },
    "hypothesis_generation": {
        "function": hypothesis_generation,
        "category": "analysis",
        "complexity": "medium"
    },
    "debate_orchestration": {
        "function": debate_orchestration,
        "category": "collaboration",
        "complexity": "high"
    },
    "reasoning_chain_analysis": {
        "function": reasoning_chain_analysis,
        "category": "validation",
        "complexity": "medium"
    }
}
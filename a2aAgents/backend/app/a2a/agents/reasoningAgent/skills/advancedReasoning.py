"""
Advanced Reasoning MCP Skill
Provides advanced multi-agent reasoning capabilities via MCP protocol
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.a2a.sdk.mcpDecorators import mcp_tool

logger = logging.getLogger(__name__)


@mcp_tool(
    name="advanced_reasoning",
    description="Perform advanced multi-agent reasoning on complex questions using multiple architectures"
)
async def advanced_reasoning(
    question: str,
    reasoning_architecture: str = "hierarchical",
    context: Optional[Dict[str, Any]] = None,
    enable_debate: bool = True,
    max_debate_rounds: int = 3,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Advanced reasoning with multi-agent orchestration

    Args:
        question: Complex question to reason about
        reasoning_architecture: Architecture (hierarchical, peer_to_peer, blackboard, chain_of_thought, swarm, debate)
        context: Additional context for reasoning
        enable_debate: Enable multi-agent debate
        max_debate_rounds: Maximum debate rounds
        confidence_threshold: Minimum confidence threshold
    """
    try:
        # Import architecture implementations
        from ..peerToPeerArchitecture import create_peer_to_peer_coordinator
        from ..chainOfThoughtArchitecture import create_chain_of_thought_reasoner
        from ..swarmIntelligenceArchitecture import create_swarm_intelligence_coordinator
        from ..debateArchitecture import create_debate_coordinator
        from ..blackboardArchitecture import create_blackboard_system
        from ..reasoningAgent import HierarchicalReasoner

        # Select architecture
        result = None
        if reasoning_architecture == "peer_to_peer":
            coordinator = create_peer_to_peer_coordinator()
            result = await coordinator.reason(question, context)
        elif reasoning_architecture == "chain_of_thought":
            reasoner = create_chain_of_thought_reasoner()
            result = await reasoner.reason(question, context)
        elif reasoning_architecture == "swarm":
            coordinator = create_swarm_intelligence_coordinator()
            result = await coordinator.reason(question, context)
        elif reasoning_architecture == "debate":
            coordinator = create_debate_coordinator()
            result = await coordinator.reason(question, context, rounds=max_debate_rounds)
        elif reasoning_architecture == "blackboard":
            system = create_blackboard_system()
            result = await system.reason(question, context)
        else:  # hierarchical default
            reasoner = HierarchicalReasoner()
            result = await reasoner.reason(question, context, confidence_threshold)

        # Format response for MCP
        return {
            "success": True,
            "question": question,
            "answer": result.get("answer", "Unable to determine answer"),
            "confidence": result.get("confidence", 0.0),
            "architecture_used": reasoning_architecture,
            "reasoning_time": result.get("execution_time", 0.0),
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "threshold_met": result.get("confidence", 0) >= confidence_threshold
            }
        }
    except Exception as e:
        logger.error(f"Advanced reasoning error: {e}")
        return {
            "success": False,
            "error": str(e),
            "question": question
        }

"""
Debate Orchestration MCP Skill
Orchestrates multi-agent debates for complex reasoning
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.a2a.sdk.mcpDecorators import mcp_tool

logger = logging.getLogger(__name__)


@mcp_tool(
    name="debate_orchestration",
    description="Orchestrate multi-agent debate on a topic with structured arguments"
)
async def debate_orchestration(
    topic: str,
    positions: List[str] = None,
    num_agents: int = 3,
    max_rounds: int = 5,
    consensus_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Orchestrate a structured debate between multiple agents

    Args:
        topic: Topic to debate
        positions: Initial positions (if None, agents develop their own)
        num_agents: Number of debating agents
        max_rounds: Maximum debate rounds
        consensus_threshold: Threshold for consensus
    """
    try:
        # Import debate architecture
        from ..debateArchitecture import create_debate_coordinator

        # Create debate coordinator
        coordinator = create_debate_coordinator()

        # Run debate
        result = await coordinator.reason(
            question=topic,
            context={"positions": positions} if positions else None,
            rounds=max_rounds
        )

        # Extract debate transcript
        transcript = await coordinator.get_debate_transcript()

        # Analyze debate dynamics
        dynamics = await coordinator.analyze_dynamics()

        return {
            "success": True,
            "topic": topic,
            "consensus_reached": result.get("consensus_reached", False),
            "final_position": result.get("answer", "No consensus reached"),
            "confidence": result.get("confidence", 0.0),
            "rounds_completed": result.get("rounds_completed", 0),
            "key_arguments": result.get("key_arguments", []),
            "debate_summary": result.get("debate_summary", {}),
            "transcript": transcript,
            "dynamics_analysis": dynamics,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "consensus_threshold": consensus_threshold,
                "threshold_met": result.get("confidence", 0) >= consensus_threshold
            }
        }

    except Exception as e:
        logger.error(f"Debate orchestration error: {e}")
        return {
            "success": False,
            "error": str(e),
            "topic": topic
        }

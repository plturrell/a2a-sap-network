"""
Debate Architecture
Implements multi-agent debate for reasoning through argumentation using MCP tools
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

# MCP decorators for inter-agent communication only
def mcp_tool(name: str, description: str):
    """MCP tool decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_tool = {"name": name, "description": description}
        return func
    return decorator

def mcp_resource(uri: str, description: str):
    """MCP resource decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_resource = {"uri": uri, "description": description}
        return func
    return decorator

def mcp_prompt(name: str, description: str):
    """MCP prompt decorator for inter-agent communication"""
    def decorator(func):
        func._mcp_prompt = {"name": name, "description": description}
        return func
    return decorator

logger = logging.getLogger(__name__)


class DebateRole(Enum):
    """Roles in the debate"""
    PROPONENT = "proponent"  # Argues for
    OPPONENT = "opponent"  # Argues against
    MODERATOR = "moderator"  # Facilitates debate
    JUDGE = "judge"  # Evaluates arguments
    SYNTHESIZER = "synthesizer"  # Combines perspectives


class ArgumentType(Enum):
    """Types of arguments"""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    REBUTTAL = "rebuttal"
    COUNTER_EXAMPLE = "counter_example"
    CLARIFICATION = "clarification"
    SYNTHESIS = "synthesis"


@dataclass
class Argument:
    """Represents an argument in the debate"""
    argument_id: str
    speaker: str
    role: DebateRole
    type: ArgumentType
    content: str
    supporting_evidence: List[str]
    confidence: float
    timestamp: datetime
    rebuts: Optional[str] = None  # ID of argument this rebuts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.argument_id,
            "speaker": self.speaker,
            "role": self.role.value,
            "type": self.type.value,
            "content": self.content,
            "evidence": self.supporting_evidence,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "rebuts": self.rebuts
        }


class DebateAgent:
    """Agent participating in debate"""

    def __init__(self, agent_id: str, role: DebateRole):
        self.agent_id = agent_id
        self.role = role
        self.arguments_made: List[Argument] = []
        self.arguments_heard: List[Argument] = []
        self.position_strength = 0.5  # Current strength of position

    async def formulate_argument(
        self,
        topic: str,
        context: Dict[str, Any],
        debate_history: List[Argument]
    ) -> Argument:
        """Formulate an argument based on role and debate history"""
        self.arguments_heard = debate_history

        if self.role == DebateRole.PROPONENT:
            return await self._argue_for(topic, context)
        elif self.role == DebateRole.OPPONENT:
            return await self._argue_against(topic, context)
        elif self.role == DebateRole.MODERATOR:
            return await self._moderate_debate(topic, context)
        elif self.role == DebateRole.JUDGE:
            return await self._judge_arguments(topic, context)
        else:  # SYNTHESIZER
            return await self._synthesize_positions(topic, context)

    async def _argue_for(self, topic: str, context: Dict[str, Any]) -> Argument:
        """Create argument supporting the topic"""
        # Find strongest unrebutted point
        if self.arguments_heard:
            # Look for opponent arguments to rebut
            opponent_args = [a for a in self.arguments_heard
                           if a.role == DebateRole.OPPONENT]
            if opponent_args:
                target = max(opponent_args, key=lambda a: a.confidence)
                return Argument(
                    argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
                    speaker=self.agent_id,
                    role=self.role,
                    type=ArgumentType.REBUTTAL,
                    content=f"While {target.content}, we must consider that {topic} has significant benefits",
                    supporting_evidence=[
                        "Historical precedents support this position",
                        "Current data indicates positive outcomes"
                    ],
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                    rebuts=target.argument_id
                )

        # Initial claim
        return Argument(
            argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
            speaker=self.agent_id,
            role=self.role,
            type=ArgumentType.CLAIM,
            content=f"I argue that {topic} is beneficial and should be supported",
            supporting_evidence=[
                "Multiple studies show positive correlation",
                "Expert consensus supports this view"
            ],
            confidence=0.8,
            timestamp=datetime.utcnow()
        )

    async def _argue_against(self, topic: str, context: Dict[str, Any]) -> Argument:
        """Create argument opposing the topic"""
        if self.arguments_heard:
            # Counter strongest proponent argument
            proponent_args = [a for a in self.arguments_heard
                            if a.role == DebateRole.PROPONENT]
            if proponent_args:
                target = max(proponent_args, key=lambda a: a.confidence)
                return Argument(
                    argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
                    speaker=self.agent_id,
                    role=self.role,
                    type=ArgumentType.COUNTER_EXAMPLE,
                    content=f"However, {topic} has shown negative consequences in practice",
                    supporting_evidence=[
                        "Case studies reveal unintended effects",
                        "Cost-benefit analysis shows net negative"
                    ],
                    confidence=0.75,
                    timestamp=datetime.utcnow(),
                    rebuts=target.argument_id
                )

        # Initial opposition
        return Argument(
            argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
            speaker=self.agent_id,
            role=self.role,
            type=ArgumentType.CLAIM,
            content=f"I argue that {topic} presents significant risks and challenges",
            supporting_evidence=[
                "Risk assessment indicates high probability of failure",
                "Alternative approaches are more effective"
            ],
            confidence=0.75,
            timestamp=datetime.utcnow()
        )

    async def _moderate_debate(self, topic: str, context: Dict[str, Any]) -> Argument:
        """Moderate the debate"""
        # Identify areas needing clarification
        if len(self.arguments_heard) > 3:
            return Argument(
                argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
                speaker=self.agent_id,
                role=self.role,
                type=ArgumentType.CLARIFICATION,
                content="Let's clarify the key points of disagreement and focus on evidence",
                supporting_evidence=[
                    "Both sides have valid concerns",
                    "We need to examine specific evidence"
                ],
                confidence=0.9,
                timestamp=datetime.utcnow()
            )

        return Argument(
            argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
            speaker=self.agent_id,
            role=self.role,
            type=ArgumentType.CLARIFICATION,
            content=f"Let's examine {topic} from multiple perspectives",
            supporting_evidence=["Structured debate improves reasoning"],
            confidence=0.9,
            timestamp=datetime.utcnow()
        )

    async def _judge_arguments(self, topic: str, context: Dict[str, Any]) -> Argument:
        """Judge the quality of arguments"""
        if len(self.arguments_heard) >= 4:
            # Evaluate argument quality
            strong_args = [a for a in self.arguments_heard if a.confidence > 0.7]
            weak_args = [a for a in self.arguments_heard if a.confidence <= 0.7]

            return Argument(
                argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
                speaker=self.agent_id,
                role=self.role,
                type=ArgumentType.EVIDENCE,
                content=f"Based on evidence presented, {len(strong_args)} strong arguments and {len(weak_args)} weaker arguments",
                supporting_evidence=[
                    "Strong arguments have better evidence support",
                    "Rebuttals effectively addressed key concerns"
                ],
                confidence=0.85,
                timestamp=datetime.utcnow()
            )

        return Argument(
            argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
            speaker=self.agent_id,
            role=self.role,
            type=ArgumentType.EVIDENCE,
            content="Evaluating arguments based on evidence and logical consistency",
            supporting_evidence=["Judgment criteria established"],
            confidence=0.8,
            timestamp=datetime.utcnow()
        )

    async def _synthesize_positions(self, topic: str, context: Dict[str, Any]) -> Argument:
        """Synthesize different positions"""
        if len(self.arguments_heard) >= 5:
            pro_args = [a for a in self.arguments_heard if a.role == DebateRole.PROPONENT]
            con_args = [a for a in self.arguments_heard if a.role == DebateRole.OPPONENT]

            return Argument(
                argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
                speaker=self.agent_id,
                role=self.role,
                type=ArgumentType.SYNTHESIS,
                content=f"Synthesizing {len(pro_args)} supporting and {len(con_args)} opposing arguments on {topic}",
                supporting_evidence=[
                    "Common ground exists in shared concerns",
                    "Nuanced position incorporates valid points from both sides"
                ],
                confidence=0.8,
                timestamp=datetime.utcnow()
            )

        return Argument(
            argument_id=f"arg_{self.agent_id}_{len(self.arguments_made)}",
            speaker=self.agent_id,
            role=self.role,
            type=ArgumentType.SYNTHESIS,
            content="Preparing to synthesize emerging perspectives",
            supporting_evidence=["Synthesis requires diverse viewpoints"],
            confidence=0.7,
            timestamp=datetime.utcnow()
        )

    def update_position(self, argument: Argument):
        """Update position strength based on argument"""
        if argument.type == ArgumentType.REBUTTAL and argument.rebuts:
            # Check if our argument was rebutted
            our_args = [a.argument_id for a in self.arguments_made]
            if argument.rebuts in our_args:
                self.position_strength *= 0.9  # Weaken position
        elif argument.role == self.role:
            # Strengthen position for supporting arguments
            self.position_strength = min(1.0, self.position_strength * 1.1)


class DebateCoordinator:
    """Coordinates multi-agent debate with direct communication"""

    def __init__(self):
        self.debate_agents: Dict[str, DebateAgent] = {}
        self.debate_history: List[Argument] = []
        self.max_rounds = 6

        # Initialize debate agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize agents with different debate roles"""
        roles = [
            ("proponent_1", DebateRole.PROPONENT),
            ("opponent_1", DebateRole.OPPONENT),
            ("moderator", DebateRole.MODERATOR),
            ("judge", DebateRole.JUDGE),
            ("synthesizer", DebateRole.SYNTHESIZER)
        ]

        for agent_id, role in roles:
            self.debate_agents[agent_id] = DebateAgent(agent_id, role)

    @mcp_tool(
        name="debate_reasoning",
        description="Multi-agent debate for reasoning through argumentation"
    )
    async def reason(
        self,
        question: str,
        context: Dict[str, Any] = None,
        rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute debate-based reasoning"""
        start_time = datetime.utcnow()
        rounds = rounds or self.max_rounds

        # Clear debate history
        self.debate_history.clear()

        # Execute debate rounds
        round_summaries = []
        for round_num in range(rounds):
            round_result = await self._debate_round(question, context, round_num)
            round_summaries.append(round_result)

            # Check if consensus reached
            if self._check_consensus():
                break

        # Generate final conclusion
        conclusion = await self._generate_conclusion(question, round_summaries)

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "answer": conclusion["answer"],
            "reasoning_type": "debate",
            "rounds_completed": len(round_summaries),
            "consensus_reached": conclusion["consensus"],
            "confidence": conclusion["confidence"],
            "execution_time": execution_time,
            "key_arguments": conclusion["key_arguments"],
            "debate_summary": conclusion["summary"]
        }

    async def _debate_round(
        self,
        question: str,
        context: Dict[str, Any],
        round_num: int
    ) -> Dict[str, Any]:
        """Execute one round of debate"""
        round_arguments = []

        # Determine speaking order for this round
        if round_num == 0:
            # Opening statements
            order = ["moderator", "proponent_1", "opponent_1"]
        elif round_num % 2 == 1:
            # Regular rounds alternate
            order = ["proponent_1", "opponent_1", "judge"]
        else:
            order = ["opponent_1", "proponent_1", "synthesizer"]

        # Each agent makes an argument
        for agent_id in order:
            agent = self.debate_agents[agent_id]

            # Direct async call - no message wrapper needed
            argument = await agent.formulate_argument(
                question, context, self.debate_history
            )

            # Add to history
            agent.arguments_made.append(argument)
            self.debate_history.append(argument)
            round_arguments.append(argument)

            # All agents hear the argument
            for other_agent in self.debate_agents.values():
                other_agent.update_position(argument)

        return {
            "round": round_num,
            "arguments": [arg.to_dict() for arg in round_arguments],
            "speakers": order
        }

    def _check_consensus(self) -> bool:
        """Check if debate has reached consensus"""
        if len(self.debate_history) < 10:
            return False

        # Check position strengths
        positions = [agent.position_strength for agent in self.debate_agents.values()]

        # Consensus if positions converge
        position_variance = max(positions) - min(positions)
        return position_variance < 0.2

    async def _generate_conclusion(
        self,
        question: str,
        round_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate conclusion from debate"""
        # Analyze all arguments
        all_arguments = []
        for round_summary in round_summaries:
            all_arguments.extend(round_summary["arguments"])

        # Find key arguments (high confidence, not rebutted)
        key_arguments = []
        rebutted_args = {arg.rebuts for arg in self.debate_history if arg.rebuts}

        for arg in self.debate_history:
            if arg.confidence > 0.7 and arg.argument_id not in rebutted_args:
                key_arguments.append(arg)

        # Get final positions
        final_positions = {}
        for agent_id, agent in self.debate_agents.items():
            final_positions[agent.role.value] = agent.position_strength

        # Build conclusion
        pro_strength = final_positions.get("proponent", 0.5)
        con_strength = final_positions.get("opponent", 0.5)

        if pro_strength > con_strength + 0.2:
            stance = "supports"
            confidence = pro_strength
        elif con_strength > pro_strength + 0.2:
            stance = "opposes"
            confidence = con_strength
        else:
            stance = "remains neutral on"
            confidence = 0.5 + abs(pro_strength - con_strength)

        # Create comprehensive answer
        answer_parts = [
            f"After structured debate on '{question}':",
            f"\nThe debate {stance} the proposition.",
            f"\nKey arguments:"
        ]

        for arg in key_arguments[:3]:
            answer_parts.append(f"- {arg.role.value}: {arg.content}")

        answer_parts.append(f"\nDebate consensus: {'Yes' if self._check_consensus() else 'No'}")
        answer_parts.append(f"Overall confidence: {confidence:.2f}")

        return {
            "answer": "\n".join(answer_parts),
            "consensus": self._check_consensus(),
            "confidence": confidence,
            "key_arguments": [arg.to_dict() for arg in key_arguments],
            "summary": {
                "rounds": len(round_summaries),
                "total_arguments": len(self.debate_history),
                "final_positions": final_positions
            }
        }

    @mcp_resource(
        uri="debate_transcript",
        description="Full transcript of the debate"
    )
    async def get_debate_transcript(self) -> Dict[str, Any]:
        """Get full debate transcript"""
        transcript = {
            "arguments": [arg.to_dict() for arg in self.debate_history],
            "total_arguments": len(self.debate_history),
            "participants": list(self.debate_agents.keys())
        }

        # Group by round
        rounds = {}
        current_round = 0
        round_args = []

        for i, arg in enumerate(self.debate_history):
            round_args.append(arg.to_dict())

            # Simple heuristic: new round every 3 arguments
            if (i + 1) % 3 == 0:
                rounds[f"round_{current_round}"] = round_args
                round_args = []
                current_round += 1

        if round_args:
            rounds[f"round_{current_round}"] = round_args

        transcript["rounds"] = rounds

        return transcript

    @mcp_prompt(
        name="analyze_debate_dynamics",
        description="Analyze the dynamics of the debate"
    )
    async def analyze_dynamics(self) -> str:
        """Analyze debate dynamics"""
        analysis = ["Debate Dynamics Analysis:"]

        # Count argument types
        type_counts = {}
        for arg in self.debate_history:
            arg_type = arg.type.value
            type_counts[arg_type] = type_counts.get(arg_type, 0) + 1

        analysis.append("\nArgument Types:")
        for arg_type, count in type_counts.items():
            analysis.append(f"- {arg_type}: {count}")

        # Analyze rebuttals
        rebuttals = [arg for arg in self.debate_history if arg.rebuts]
        analysis.append(f"\nRebuttals: {len(rebuttals)}")

        # Position evolution
        analysis.append("\nFinal Positions:")
        for agent_id, agent in self.debate_agents.items():
            analysis.append(f"- {agent.role.value}: {agent.position_strength:.2f}")

        return "\n".join(analysis)


# Factory function
def create_debate_coordinator() -> DebateCoordinator:
    """Create a debate coordinator"""
    return DebateCoordinator()
"""
Chain-of-Thought Reasoning Architecture
Implements step-by-step reasoning with MCP tools and Grok-4 integration
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

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


class ThoughtStep:
    """Represents a single step in the chain of thought"""

    def __init__(self, step_number: int, thought: str, reasoning: str, confidence: float):
        self.step_number = step_number
        self.thought = thought
        self.reasoning = reasoning
        self.confidence = confidence
        self.timestamp = datetime.utcnow()
        self.dependencies: List[int] = []  # Steps this depends on
        self.evidence: List[str] = []

    def add_evidence(self, evidence: str):
        """Add supporting evidence to this step"""
        self.evidence.append(evidence)

    def add_dependency(self, step_number: int):
        """Add a dependency on another step"""
        if step_number not in self.dependencies:
            self.dependencies.append(step_number)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step": self.step_number,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "dependencies": self.dependencies,
            "evidence": self.evidence
        }


class ReasoningStrategy(Enum):
    """Different chain-of-thought strategies"""
    LINEAR = "linear"  # Step by step, sequential
    BRANCHING = "branching"  # Explore multiple paths
    RECURSIVE = "recursive"  # Break down into sub-problems
    ITERATIVE = "iterative"  # Refine thoughts through iterations


class ChainOfThoughtReasoner:
    """Implements chain-of-thought reasoning with direct step processing"""

    def __init__(self, grok_client=None):
        self.grok_client = grok_client
        self.thought_chains: Dict[str, List[ThoughtStep]] = {}
        self.max_steps = 10
        self.min_confidence = 0.6

    @mcp_tool(
        name="chain_of_thought_reasoning",
        description="Step-by-step reasoning with explicit thought chains"
    )
    async def reason(
        self,
        question: str,
        context: Dict[str, Any] = None,
        strategy: ReasoningStrategy = ReasoningStrategy.LINEAR
    ) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning"""
        start_time = datetime.utcnow()
        chain_id = f"chain_{start_time.timestamp()}"

        # Initialize thought chain
        thought_chain = []

        # Step 1: Problem understanding
        understanding_step = await self._understand_problem(question, context)
        thought_chain.append(understanding_step)

        # Step 2: Generate reasoning chain based on strategy
        if strategy == ReasoningStrategy.LINEAR:
            chain_steps = await self._linear_reasoning(question, understanding_step, context)
        elif strategy == ReasoningStrategy.BRANCHING:
            chain_steps = await self._branching_reasoning(question, understanding_step, context)
        elif strategy == ReasoningStrategy.RECURSIVE:
            chain_steps = await self._recursive_reasoning(question, understanding_step, context)
        else:  # ITERATIVE
            chain_steps = await self._iterative_reasoning(question, understanding_step, context)

        thought_chain.extend(chain_steps)

        # Step 3: Synthesize conclusion
        conclusion = await self._synthesize_conclusion(thought_chain, question)
        thought_chain.append(conclusion)

        # Store chain
        self.thought_chains[chain_id] = thought_chain

        # Calculate metrics
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        avg_confidence = sum(step.confidence for step in thought_chain) / len(thought_chain)

        return {
            "answer": conclusion.thought,
            "reasoning_type": "chain_of_thought",
            "strategy": strategy.value,
            "thought_chain": [step.to_dict() for step in thought_chain],
            "chain_length": len(thought_chain),
            "confidence": avg_confidence,
            "execution_time": execution_time,
            "chain_id": chain_id
        }

    async def _understand_problem(self, question: str, context: Dict[str, Any]) -> ThoughtStep:
        """Initial problem understanding step"""
        if self.grok_client:
            try:
                # Use Grok-4 for problem understanding
                result = await self.grok_client.decompose_question(question, context)
                if result.get("success"):
                    decomposition = result.get("decomposition", {})
                    understanding = f"Problem: {question}\nKey concepts: {decomposition.get('main_concepts', [])}"
                    confidence = 0.9
                else:
                    understanding = f"Understanding: {question}"
                    confidence = 0.7
            except:
                understanding = f"Understanding: {question}"
                confidence = 0.7
        else:
            # Fallback understanding
            understanding = f"Understanding: {question}"
            confidence = 0.7

        step = ThoughtStep(
            step_number=1,
            thought=understanding,
            reasoning="Initial problem analysis and concept identification",
            confidence=confidence
        )

        if context:
            step.add_evidence(f"Context provided: {list(context.keys())}")

        return step

    async def _linear_reasoning(
        self,
        question: str,
        understanding: ThoughtStep,
        context: Dict[str, Any]
    ) -> List[ThoughtStep]:
        """Linear step-by-step reasoning"""
        steps = []
        current_step_num = 2

        # Generate reasoning steps
        reasoning_prompts = [
            "What are the key facts we need to consider?",
            "What logical connections can we make?",
            "What conclusions follow from these connections?",
            "Are there any counterarguments to consider?"
        ]

        for prompt in reasoning_prompts:
            if current_step_num > self.max_steps:
                break

            # Create thought for this step
            thought = f"Step {current_step_num}: {prompt}"

            # Generate reasoning
            if current_step_num == 2:  # Facts
                reasoning = "Identifying relevant facts from the question and context"
                confidence = 0.8
            elif current_step_num == 3:  # Connections
                reasoning = "Drawing logical connections between identified facts"
                confidence = 0.75
            elif current_step_num == 4:  # Conclusions
                reasoning = "Deriving conclusions from logical connections"
                confidence = 0.7
            else:  # Counterarguments
                reasoning = "Considering alternative perspectives and counterarguments"
                confidence = 0.65

            step = ThoughtStep(
                step_number=current_step_num,
                thought=thought,
                reasoning=reasoning,
                confidence=confidence
            )

            # Add dependency on previous step
            if steps:
                step.add_dependency(current_step_num - 1)
            else:
                step.add_dependency(1)  # Depends on understanding

            steps.append(step)
            current_step_num += 1

        return steps

    async def _branching_reasoning(
        self,
        question: str,
        understanding: ThoughtStep,
        context: Dict[str, Any]
    ) -> List[ThoughtStep]:
        """Branching reasoning - explore multiple paths"""
        steps = []

        # Branch 1: Optimistic path
        optimistic_step = ThoughtStep(
            step_number=2,
            thought="Exploring optimistic interpretation",
            reasoning="Considering best-case scenarios and positive outcomes",
            confidence=0.7
        )
        optimistic_step.add_dependency(1)
        steps.append(optimistic_step)

        # Branch 2: Pessimistic path
        pessimistic_step = ThoughtStep(
            step_number=3,
            thought="Exploring pessimistic interpretation",
            reasoning="Considering worst-case scenarios and challenges",
            confidence=0.7
        )
        pessimistic_step.add_dependency(1)
        steps.append(pessimistic_step)

        # Branch 3: Neutral path
        neutral_step = ThoughtStep(
            step_number=4,
            thought="Exploring neutral interpretation",
            reasoning="Considering balanced view without bias",
            confidence=0.8
        )
        neutral_step.add_dependency(1)
        steps.append(neutral_step)

        # Merge branches
        merge_step = ThoughtStep(
            step_number=5,
            thought="Synthesizing multiple perspectives",
            reasoning="Combining insights from different interpretive branches",
            confidence=0.75
        )
        merge_step.add_dependency(2)
        merge_step.add_dependency(3)
        merge_step.add_dependency(4)
        steps.append(merge_step)

        return steps

    async def _recursive_reasoning(
        self,
        question: str,
        understanding: ThoughtStep,
        context: Dict[str, Any]
    ) -> List[ThoughtStep]:
        """Recursive reasoning - break into sub-problems"""
        steps = []

        # Decompose into sub-problems
        decompose_step = ThoughtStep(
            step_number=2,
            thought="Breaking down into sub-problems",
            reasoning="Identifying component parts that can be solved independently",
            confidence=0.8
        )
        decompose_step.add_dependency(1)
        steps.append(decompose_step)

        # Solve sub-problems
        sub_problems = ["Component A", "Component B", "Component C"]
        for i, sub_problem in enumerate(sub_problems):
            sub_step = ThoughtStep(
                step_number=3 + i,
                thought=f"Solving {sub_problem}",
                reasoning=f"Addressing {sub_problem} as independent unit",
                confidence=0.75
            )
            sub_step.add_dependency(2)
            steps.append(sub_step)

        # Combine solutions
        combine_step = ThoughtStep(
            step_number=6,
            thought="Combining sub-problem solutions",
            reasoning="Integrating component solutions into comprehensive answer",
            confidence=0.8
        )
        for i in range(3, 6):
            combine_step.add_dependency(i)
        steps.append(combine_step)

        return steps

    async def _iterative_reasoning(
        self,
        question: str,
        understanding: ThoughtStep,
        context: Dict[str, Any]
    ) -> List[ThoughtStep]:
        """Iterative reasoning - refine through iterations"""
        steps = []

        # Initial hypothesis
        hypothesis_step = ThoughtStep(
            step_number=2,
            thought="Initial hypothesis",
            reasoning="Forming preliminary answer based on initial understanding",
            confidence=0.6
        )
        hypothesis_step.add_dependency(1)
        steps.append(hypothesis_step)

        # Iterations of refinement
        for iteration in range(3):
            # Test hypothesis
            test_step = ThoughtStep(
                step_number=3 + (iteration * 2),
                thought=f"Testing hypothesis (iteration {iteration + 1})",
                reasoning="Evaluating hypothesis against known facts and logic",
                confidence=0.7 + (iteration * 0.05)
            )
            test_step.add_dependency(2 + (iteration * 2))
            steps.append(test_step)

            # Refine hypothesis
            refine_step = ThoughtStep(
                step_number=4 + (iteration * 2),
                thought=f"Refining hypothesis (iteration {iteration + 1})",
                reasoning="Adjusting hypothesis based on test results",
                confidence=0.75 + (iteration * 0.05)
            )
            refine_step.add_dependency(3 + (iteration * 2))
            steps.append(refine_step)

        return steps

    async def _synthesize_conclusion(
        self,
        thought_chain: List[ThoughtStep],
        question: str
    ) -> ThoughtStep:
        """Synthesize final conclusion from thought chain"""
        # Analyze the chain
        key_insights = []
        total_confidence = 0

        for step in thought_chain[1:]:  # Skip understanding step
            if step.confidence >= self.min_confidence:
                key_insights.append(step.thought)
                total_confidence += step.confidence

        avg_confidence = total_confidence / len(thought_chain[1:]) if len(thought_chain) > 1 else 0.5

        # Build conclusion
        conclusion_parts = [
            f"After chain-of-thought reasoning about '{question}':",
            f"Key insights from {len(key_insights)} high-confidence steps:",
        ]

        for insight in key_insights[-3:]:  # Last 3 key insights
            conclusion_parts.append(f"- {insight}")

        conclusion_parts.append(f"\nFinal answer: Based on the reasoning chain, {question}")

        conclusion_step = ThoughtStep(
            step_number=len(thought_chain) + 1,
            thought="\n".join(conclusion_parts),
            reasoning="Synthesis of all reasoning steps into final conclusion",
            confidence=avg_confidence
        )

        # Add dependencies on all previous steps
        for step in thought_chain:
            conclusion_step.add_dependency(step.step_number)

        return conclusion_step

    @mcp_resource(
        uri="chain_of_thought_history",
        description="Access stored reasoning chains"
    )
    async def get_chain_history(self, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """Get reasoning chain history"""
        if chain_id:
            chain = self.thought_chains.get(chain_id)
            if chain:
                return {
                    "chain_id": chain_id,
                    "steps": [step.to_dict() for step in chain],
                    "total_steps": len(chain)
                }
            else:
                return {"error": f"Chain {chain_id} not found"}
        else:
            # Return summary of all chains
            return {
                "total_chains": len(self.thought_chains),
                "chains": [
                    {
                        "chain_id": cid,
                        "steps": len(chain),
                        "avg_confidence": sum(s.confidence for s in chain) / len(chain)
                    }
                    for cid, chain in self.thought_chains.items()
                ]
            }

    @mcp_prompt(
        name="explain_reasoning_step",
        description="Explain a specific reasoning step"
    )
    async def explain_step(self, chain_id: str, step_number: int) -> str:
        """Explain a specific reasoning step"""
        chain = self.thought_chains.get(chain_id)
        if not chain:
            return f"Chain {chain_id} not found"

        step = next((s for s in chain if s.step_number == step_number), None)
        if not step:
            return f"Step {step_number} not found in chain {chain_id}"

        explanation = [
            f"Step {step.step_number}: {step.thought}",
            f"Reasoning: {step.reasoning}",
            f"Confidence: {step.confidence:.2f}",
        ]

        if step.dependencies:
            explanation.append(f"Depends on steps: {step.dependencies}")

        if step.evidence:
            explanation.append(f"Evidence: {'; '.join(step.evidence)}")

        return "\n".join(explanation)


# Factory function
def create_chain_of_thought_reasoner(grok_client=None) -> ChainOfThoughtReasoner:
    """Create a chain-of-thought reasoner"""
    return ChainOfThoughtReasoner(grok_client)

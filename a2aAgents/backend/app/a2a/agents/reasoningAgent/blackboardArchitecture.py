"""
Blackboard Architecture Implementation with Grok-4 Integration
A collaborative problem-solving architecture where multiple knowledge sources
work together on a shared blackboard workspace
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json

try:
    from .grokReasoning import GrokReasoning
except ImportError:
    # For direct imports
    from grokReasoning import GrokReasoning


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

logger = logging.getLogger(__name__)


class KnowledgeSourceType(Enum):
    """Types of knowledge sources in the blackboard system"""
    PATTERN_RECOGNITION = "pattern_recognition"
    LOGICAL_REASONING = "logical_reasoning"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    CAUSAL_ANALYSIS = "causal_analysis"
    ANALOGY_DETECTION = "analogy_detection"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"


class BlackboardState:
    """Represents the shared blackboard workspace"""

    def __init__(self):
        self.problem: str = ""
        self.facts: List[Dict[str, Any]] = []
        self.hypotheses: List[Dict[str, Any]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.conclusions: List[Dict[str, Any]] = []
        self.patterns: List[Dict[str, Any]] = []
        self.causal_chains: List[Dict[str, Any]] = []
        self.analogies: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []
        self.contributions: List[Dict[str, Any]] = []
        self.iteration: int = 0
        self.confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert blackboard state to dictionary"""
        return {
            "problem": self.problem,
            "facts": self.facts,
            "hypotheses": self.hypotheses,
            "evidence": self.evidence,
            "conclusions": self.conclusions,
            "patterns": self.patterns,
            "causal_chains": self.causal_chains,
            "analogies": self.analogies,
            "constraints": self.constraints,
            "contributions": self.contributions,
            "iteration": self.iteration,
            "confidence": self.confidence
        }


class KnowledgeSource:
    """Base class for knowledge sources"""

    def __init__(self, source_type: KnowledgeSourceType, grok_client: GrokReasoning):
        self.source_type = source_type
        self.grok = grok_client
        self.name = source_type.value

    async def can_contribute(self, blackboard: BlackboardState) -> Tuple[bool, float]:
        """Check if this knowledge source can contribute to current state"""
        # To be overridden by specific knowledge sources
        return False, 0.0

    async def contribute(self, blackboard: BlackboardState) -> Dict[str, Any]:
        """Make contribution to blackboard"""
        # To be overridden by specific knowledge sources
        return {"contributed": False}


class PatternRecognitionSource(KnowledgeSource):
    """Knowledge source for pattern recognition using Grok-4"""

    def __init__(self, grok_client: GrokReasoning):
        super().__init__(KnowledgeSourceType.PATTERN_RECOGNITION, grok_client)

    async def can_contribute(self, blackboard: BlackboardState) -> Tuple[bool, float]:
        """Check if we can find patterns in current state"""
        # Can contribute if we have facts or evidence but limited patterns
        has_data = len(blackboard.facts) > 0 or len(blackboard.evidence) > 0
        needs_patterns = len(blackboard.patterns) < 3

        if has_data and needs_patterns:
            priority = 0.8 if blackboard.iteration < 3 else 0.6
            return True, priority
        return False, 0.0

    async def contribute(self, blackboard: BlackboardState) -> Dict[str, Any]:
        """Find patterns using Grok-4"""
        try:
            # Combine facts and evidence for pattern analysis
            text_elements = []
            for fact in blackboard.facts:
                text_elements.append(f"Fact: {fact.get('content', '')}")
            for evidence in blackboard.evidence:
                text_elements.append(f"Evidence: {evidence.get('content', '')}")

            combined_text = "\n".join(text_elements)

            # Use Grok-4 for pattern analysis
            result = await self.grok.analyze_patterns(
                combined_text,
                existing_patterns=blackboard.patterns
            )

            if result.get('success'):
                patterns = result.get('patterns', {})

                # Extract different types of patterns
                new_patterns = []

                # Semantic patterns
                if 'semantic_patterns' in patterns:
                    for pattern in patterns['semantic_patterns']:
                        new_patterns.append({
                            'type': 'semantic',
                            'pattern': pattern,
                            'source': self.name,
                            'confidence': 0.85,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Logical relationships
                if 'logical_relationships' in patterns:
                    for rel in patterns['logical_relationships']:
                        new_patterns.append({
                            'type': 'logical',
                            'pattern': rel,
                            'source': self.name,
                            'confidence': 0.9,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Key insights
                if 'key_insights' in patterns:
                    for insight in patterns['key_insights']:
                        new_patterns.append({
                            'type': 'insight',
                            'pattern': insight,
                            'source': self.name,
                            'confidence': 0.8,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Add to blackboard
                blackboard.patterns.extend(new_patterns)

                contribution = {
                    'source': self.name,
                    'action': 'added_patterns',
                    'count': len(new_patterns),
                    'timestamp': datetime.utcnow().isoformat()
                }
                blackboard.contributions.append(contribution)

                return {
                    'contributed': True,
                    'patterns_added': len(new_patterns),
                    'contribution': contribution
                }

        except Exception as e:
            logger.error(f"Pattern recognition error: {e}")

        return {'contributed': False}


class LogicalReasoningSource(KnowledgeSource):
    """Knowledge source for logical reasoning using Grok-4"""

    def __init__(self, grok_client: GrokReasoning):
        super().__init__(KnowledgeSourceType.LOGICAL_REASONING, grok_client)

    async def can_contribute(self, blackboard: BlackboardState) -> Tuple[bool, float]:
        """Check if we can apply logical reasoning"""
        # Can contribute if we have facts and patterns but need conclusions
        has_basis = len(blackboard.facts) > 1 and len(blackboard.patterns) > 0
        needs_conclusions = len(blackboard.conclusions) < 2

        if has_basis and needs_conclusions:
            priority = 0.9 if blackboard.iteration > 2 else 0.7
            return True, priority
        return False, 0.0

    async def contribute(self, blackboard: BlackboardState) -> Dict[str, Any]:
        """Apply logical reasoning to derive conclusions"""
        try:
            # Prepare logical components for Grok-4
            components = {
                "problem": blackboard.problem,
                "facts": [f.get('content', '') for f in blackboard.facts],
                "patterns": [p.get('pattern', '') for p in blackboard.patterns],
                "hypotheses": [h.get('content', '') for h in blackboard.hypotheses]
            }

            # Use Grok-4 to decompose and reason
            result = await self.grok.decompose_question(
                blackboard.problem,
                context={
                    "task": "logical_reasoning",
                    "components": components
                }
            )

            if result.get('success'):
                decomposition = result.get('decomposition', {})

                # Extract logical conclusions
                new_conclusions = []

                if 'logical_inferences' in decomposition:
                    for inference in decomposition['logical_inferences']:
                        new_conclusions.append({
                            'type': 'logical_inference',
                            'content': inference,
                            'source': self.name,
                            'confidence': 0.85,
                            'reasoning': 'deductive',
                            'timestamp': datetime.utcnow().isoformat()
                        })

                if 'conclusions' in decomposition:
                    for conclusion in decomposition['conclusions']:
                        new_conclusions.append({
                            'type': 'derived_conclusion',
                            'content': conclusion,
                            'source': self.name,
                            'confidence': 0.8,
                            'reasoning': 'inductive',
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Add to blackboard
                blackboard.conclusions.extend(new_conclusions)

                # Generate hypotheses if needed
                if len(blackboard.hypotheses) < 3 and 'potential_hypotheses' in decomposition:
                    for hyp in decomposition['potential_hypotheses']:
                        blackboard.hypotheses.append({
                            'content': hyp,
                            'source': self.name,
                            'confidence': 0.7,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                contribution = {
                    'source': self.name,
                    'action': 'logical_reasoning',
                    'conclusions_added': len(new_conclusions),
                    'hypotheses_added': len(decomposition.get('potential_hypotheses', [])),
                    'timestamp': datetime.utcnow().isoformat()
                }
                blackboard.contributions.append(contribution)

                return {
                    'contributed': True,
                    'conclusions_added': len(new_conclusions),
                    'contribution': contribution
                }

        except Exception as e:
            logger.error(f"Logical reasoning error: {e}")

        return {'contributed': False}


class EvidenceEvaluationSource(KnowledgeSource):
    """Knowledge source for evaluating evidence using Grok-4"""

    def __init__(self, grok_client: GrokReasoning):
        super().__init__(KnowledgeSourceType.EVIDENCE_EVALUATION, grok_client)

    async def can_contribute(self, blackboard: BlackboardState) -> Tuple[bool, float]:
        """Check if we can evaluate evidence"""
        # Can contribute if we have hypotheses that need evidence evaluation
        has_hypotheses = len(blackboard.hypotheses) > 0
        needs_evaluation = any(h.get('evidence_score', 0) == 0 for h in blackboard.hypotheses)

        if has_hypotheses and needs_evaluation:
            priority = 0.85
            return True, priority
        return False, 0.0

    async def contribute(self, blackboard: BlackboardState) -> Dict[str, Any]:
        """Evaluate evidence for hypotheses"""
        try:
            evaluations_made = 0

            for hypothesis in blackboard.hypotheses:
                if hypothesis.get('evidence_score', 0) == 0:
                    # Prepare evidence for evaluation
                    evidence_text = "\n".join([
                        f"Evidence: {e.get('content', '')}"
                        for e in blackboard.evidence
                    ])

                    # Use Grok-4 to analyze evidence support
                    combined = f"Hypothesis: {hypothesis['content']}\n\n{evidence_text}"
                    result = await self.grok.analyze_patterns(
                        combined,
                        existing_patterns=[{
                            'type': 'evidence_evaluation',
                            'focus': 'support_strength'
                        }]
                    )

                    if result.get('success'):
                        patterns = result.get('patterns', {})

                        # Calculate evidence score
                        evidence_score = 0.5  # Base score

                        if 'support_indicators' in patterns:
                            evidence_score += len(patterns['support_indicators']) * 0.1

                        if 'contradiction_indicators' in patterns:
                            evidence_score -= len(patterns['contradiction_indicators']) * 0.15

                        evidence_score = max(0.1, min(1.0, evidence_score))

                        # Update hypothesis
                        hypothesis['evidence_score'] = evidence_score
                        hypothesis['evidence_evaluation'] = {
                            'evaluated_by': self.name,
                            'support': patterns.get('support_indicators', []),
                            'contradictions': patterns.get('contradiction_indicators', []),
                            'timestamp': datetime.utcnow().isoformat()
                        }

                        evaluations_made += 1

                        # Add supporting evidence as new evidence if strong
                        if evidence_score > 0.7:
                            blackboard.evidence.append({
                                'content': f"Strong evidence supports: {hypothesis['content']}",
                                'source': self.name,
                                'confidence': evidence_score,
                                'type': 'derived_evidence',
                                'timestamp': datetime.utcnow().isoformat()
                            })

            if evaluations_made > 0:
                contribution = {
                    'source': self.name,
                    'action': 'evidence_evaluation',
                    'evaluations_made': evaluations_made,
                    'timestamp': datetime.utcnow().isoformat()
                }
                blackboard.contributions.append(contribution)

                return {
                    'contributed': True,
                    'evaluations_made': evaluations_made,
                    'contribution': contribution
                }

        except Exception as e:
            logger.error(f"Evidence evaluation error: {e}")

        return {'contributed': False}


class CausalAnalysisSource(KnowledgeSource):
    """Knowledge source for causal analysis using Grok-4"""

    def __init__(self, grok_client: GrokReasoning):
        super().__init__(KnowledgeSourceType.CAUSAL_ANALYSIS, grok_client)

    async def can_contribute(self, blackboard: BlackboardState) -> Tuple[bool, float]:
        """Check if we can perform causal analysis"""
        # Can contribute if we have patterns but limited causal chains
        has_patterns = len(blackboard.patterns) > 1
        needs_causal = len(blackboard.causal_chains) < 2

        if has_patterns and needs_causal:
            priority = 0.75
            return True, priority
        return False, 0.0

    async def contribute(self, blackboard: BlackboardState) -> Dict[str, Any]:
        """Analyze causal relationships"""
        try:
            # Combine problem context with patterns
            context_text = f"Problem: {blackboard.problem}\n\n"
            context_text += "Patterns:\n" + "\n".join([
                f"- {p.get('pattern', '')}" for p in blackboard.patterns
            ])

            # Use Grok-4 for causal analysis
            result = await self.grok.analyze_patterns(
                context_text,
                existing_patterns=[{
                    'type': 'causal_analysis',
                    'focus': 'cause_effect_relationships'
                }]
            )

            if result.get('success'):
                patterns = result.get('patterns', {})
                new_chains = []

                if 'causal_relationships' in patterns:
                    for rel in patterns['causal_relationships']:
                        new_chains.append({
                            'cause': rel.get('cause', ''),
                            'effect': rel.get('effect', ''),
                            'strength': rel.get('strength', 0.7),
                            'type': rel.get('type', 'direct'),
                            'source': self.name,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Add to blackboard
                blackboard.causal_chains.extend(new_chains)

                contribution = {
                    'source': self.name,
                    'action': 'causal_analysis',
                    'chains_added': len(new_chains),
                    'timestamp': datetime.utcnow().isoformat()
                }
                blackboard.contributions.append(contribution)

                return {
                    'contributed': True,
                    'chains_added': len(new_chains),
                    'contribution': contribution
                }

        except Exception as e:
            logger.error(f"Causal analysis error: {e}")

        return {'contributed': False}


class BlackboardController:
    """Controls the blackboard reasoning process"""

    def __init__(self):
        self.grok = GrokReasoning()
        self.knowledge_sources: List[KnowledgeSource] = []
        self.blackboard = BlackboardState()
        self.max_iterations = 10

        # Initialize knowledge sources
        self._initialize_knowledge_sources()

    def _initialize_knowledge_sources(self):
        """Initialize all knowledge sources"""
        self.knowledge_sources = [
            PatternRecognitionSource(self.grok),
            LogicalReasoningSource(self.grok),
            EvidenceEvaluationSource(self.grok),
            CausalAnalysisSource(self.grok)
        ]

    async def reason(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main blackboard reasoning process"""
        try:
            # Initialize blackboard with problem
            self.blackboard.problem = question

            # Use Grok-4 to decompose question into initial facts
            decomp_result = await self.grok.decompose_question(question, context)

            if decomp_result.get('success'):
                decomposition = decomp_result.get('decomposition', {})

                # Extract initial facts
                if 'facts' in decomposition:
                    for fact in decomposition['facts']:
                        self.blackboard.facts.append({
                            'content': fact,
                            'source': 'initial_decomposition',
                            'confidence': 0.9,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Extract initial hypotheses
                if 'hypotheses' in decomposition:
                    for hyp in decomposition['hypotheses']:
                        self.blackboard.hypotheses.append({
                            'content': hyp,
                            'source': 'initial_decomposition',
                            'confidence': 0.7,
                            'evidence_score': 0,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                # Also check for other common keys
                if 'main_concepts' in decomposition:
                    for concept in decomposition['main_concepts']:
                        self.blackboard.facts.append({
                            'content': f"Key concept: {concept}",
                            'source': 'initial_decomposition',
                            'confidence': 0.8,
                            'timestamp': datetime.utcnow().isoformat()
                        })

                if 'sub_questions' in decomposition:
                    for sub_q in decomposition['sub_questions']:
                        self.blackboard.facts.append({
                            'content': f"Sub-question: {sub_q}",
                            'source': 'initial_decomposition',
                            'confidence': 0.85,
                            'timestamp': datetime.utcnow().isoformat()
                        })

            # If no facts were extracted, add the question as initial fact
            if not self.blackboard.facts:
                self.blackboard.facts.append({
                    'content': f"Question to analyze: {question}",
                    'source': 'initial_setup',
                    'confidence': 1.0,
                    'timestamp': datetime.utcnow().isoformat()
                })

            # Main reasoning loop
            for iteration in range(self.max_iterations):
                self.blackboard.iteration = iteration

                # Check termination conditions
                if self._should_terminate():
                    break

                # Get contributions from knowledge sources
                contributed = await self._execute_iteration()

                if not contributed:
                    # No knowledge source could contribute
                    break

                # Update overall confidence
                self._update_confidence()

            # Synthesize final answer
            answer = await self._synthesize_solution()

            return {
                'answer': answer,
                'confidence': self.blackboard.confidence,
                'reasoning_architecture': 'blackboard',
                'iterations': self.blackboard.iteration + 1,
                'blackboard_state': self.blackboard.to_dict(),
                'enhanced': True
            }

        except Exception as e:
            logger.error(f"Blackboard reasoning error: {e}")
            return {
                'answer': f"Blackboard reasoning failed: {str(e)}",
                'confidence': 0.0,
                'reasoning_architecture': 'blackboard',
                'enhanced': False
            }

    async def _execute_iteration(self) -> bool:
        """Execute one iteration of blackboard reasoning"""
        # Evaluate which knowledge sources can contribute
        candidates = []

        for source in self.knowledge_sources:
            can_contribute, priority = await source.can_contribute(self.blackboard)
            if can_contribute:
                candidates.append((source, priority))

        if not candidates:
            return False

        # Sort by priority and select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_source = candidates[0][0]

        # Let selected source contribute
        result = await selected_source.contribute(self.blackboard)

        return result.get('contributed', False)

    def _should_terminate(self) -> bool:
        """Check if reasoning should terminate"""
        # Terminate if we have high confidence conclusions
        if self.blackboard.conclusions:
            max_confidence = max(c.get('confidence', 0) for c in self.blackboard.conclusions)
            if max_confidence > 0.9:
                return True

        # Terminate if we have strong evidence for hypotheses
        if self.blackboard.hypotheses:
            max_evidence = max(h.get('evidence_score', 0) for h in self.blackboard.hypotheses)
            if max_evidence > 0.85:
                return True

        # Terminate if confidence is high enough
        if self.blackboard.confidence > 0.85:
            return True

        return False

    def _update_confidence(self):
        """Update overall blackboard confidence"""
        scores = []

        # Factor in conclusions
        if self.blackboard.conclusions:
            conclusion_conf = max(c.get('confidence', 0) for c in self.blackboard.conclusions)
            scores.append(conclusion_conf)

        # Factor in hypothesis evidence
        if self.blackboard.hypotheses:
            hyp_scores = [h.get('evidence_score', 0) for h in self.blackboard.hypotheses if h.get('evidence_score', 0) > 0]
            if hyp_scores:
                scores.append(max(hyp_scores))

        # Factor in pattern quality
        if self.blackboard.patterns:
            pattern_conf = sum(p.get('confidence', 0) for p in self.blackboard.patterns) / len(self.blackboard.patterns)
            scores.append(pattern_conf * 0.8)  # Weight patterns less

        # Calculate overall confidence
        if scores:
            self.blackboard.confidence = sum(scores) / len(scores)
        else:
            self.blackboard.confidence = 0.3  # Base confidence

    async def _synthesize_solution(self) -> str:
        """Synthesize final answer from blackboard state"""
        try:
            # Prepare all components for synthesis
            sub_answers = []

            # Add conclusions
            for conclusion in self.blackboard.conclusions:
                sub_answers.append({
                    'content': conclusion['content'],
                    'type': 'conclusion',
                    'confidence': conclusion.get('confidence', 0.7),
                    'source': conclusion.get('source', 'unknown')
                })

            # Add high-scoring hypotheses
            for hypothesis in self.blackboard.hypotheses:
                if hypothesis.get('evidence_score', 0) > 0.7:
                    sub_answers.append({
                        'content': hypothesis['content'],
                        'type': 'hypothesis',
                        'confidence': hypothesis['evidence_score'],
                        'evidence': hypothesis.get('evidence_evaluation', {})
                    })

            # Add key patterns
            for pattern in self.blackboard.patterns[:3]:  # Top 3 patterns
                sub_answers.append({
                    'content': str(pattern.get('pattern', '')),
                    'type': 'pattern',
                    'confidence': pattern.get('confidence', 0.7)
                })

            # Use Grok-4 to synthesize
            result = await self.grok.synthesize_answer(sub_answers, self.blackboard.problem)

            if result.get('success'):
                return result.get('synthesis')
            else:
                # Fallback synthesis
                if self.blackboard.conclusions:
                    best_conclusion = max(self.blackboard.conclusions, key=lambda c: c.get('confidence', 0))
                    return best_conclusion['content']
                elif self.blackboard.hypotheses:
                    best_hypothesis = max(self.blackboard.hypotheses, key=lambda h: h.get('evidence_score', 0))
                    return f"Based on evidence: {best_hypothesis['content']}"
                else:
                    return "Unable to reach a conclusion with available information"

        except Exception as e:
            logger.error(f"Solution synthesis error: {e}")
            return "Error synthesizing solution"


# Integration function
async def blackboard_reasoning(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main entry point for blackboard reasoning"""
    controller = BlackboardController()
    return await controller.reason(question, context)

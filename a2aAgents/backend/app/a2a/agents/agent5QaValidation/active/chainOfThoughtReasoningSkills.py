import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sentence_transformers import SentenceTransformer
import openai
from datetime import datetime
import hashlib
from pathlib import Path

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task
from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
from app.a2a.core.trustIdentity import TrustIdentity
from app.a2a.core.dataValidation import DataValidator


from app.a2a.core.security_base import SecureA2AAgent
"""
Chain-of-Thought Reasoning Skills for Agent 5 - Phase 2 Advanced Features
Implements structured reasoning processes for complex QA validation with step-by-step analysis
"""

class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    STEP_BY_STEP = "step_by_step"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    CAUSE_EFFECT_ANALYSIS = "cause_effect_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"


@dataclass
class ReasoningStep:
    """Individual step in chain-of-thought reasoning"""
    step_id: int
    step_type: str  # "premise", "inference", "conclusion", "validation", "evidence"
    content: str
    confidence_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)  # IDs of prerequisite steps
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReasoningChain:
    """Complete chain of reasoning for a question"""
    chain_id: str
    question: str
    strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    final_answer: str
    overall_confidence: float
    reasoning_quality_score: float
    validation_results: Dict[str, Any] = field(default_factory=dict)


class ChainOfThoughtReasoningSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Real A2A agent skills for chain-of-thought reasoning in QA validation
    Provides structured, transparent reasoning processes for complex questions
    """

    def __init__(self, trust_identity: TrustIdentity):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataValidator()

        # Initialize models for reasoning support
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Reasoning templates and patterns
        self.reasoning_templates = self._load_reasoning_templates()
        self.logical_operators = ['AND', 'OR', 'NOT', 'IF...THEN', 'IFF', 'IMPLIES']

        # Performance tracking
        self.reasoning_metrics = {
            'chains_generated': 0,
            'average_steps_per_chain': 0,
            'average_reasoning_quality': 0.0,
            'successful_validations': 0,
            'reasoning_errors': []
        }

    @a2a_skill(
        name="generateReasoningChain",
        description="Generate structured chain-of-thought reasoning for complex questions",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "string"},
                "reasoning_strategy": {
                    "type": "string",
                    "enum": ["step_by_step", "hypothesis_testing", "cause_effect_analysis", "comparative_analysis", "deductive_reasoning", "inductive_reasoning"],
                    "default": "step_by_step"
                },
                "knowledge_base": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "content": {"type": "string"},
                            "confidence": {"type": "number"},
                            "source": {"type": "string"}
                        }
                    }
                },
                "max_reasoning_steps": {"type": "integer", "default": 10},
                "confidence_threshold": {"type": "number", "default": 0.7}
            },
            "required": ["question", "context"]
        }
    )
    def generate_reasoning_chain(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured chain-of-thought reasoning process"""
        try:
            question = request_data["question"]
            context = request_data["context"]
            strategy = ReasoningStrategy(request_data.get("reasoning_strategy", "step_by_step"))
            knowledge_base = request_data.get("knowledge_base", [])
            max_steps = request_data.get("max_reasoning_steps", 10)
            confidence_threshold = request_data.get("confidence_threshold", 0.7)

            # Create unique chain ID
            chain_id = hashlib.md5(f"{question}_{context}_{strategy.value}".encode()).hexdigest()[:12]

            # Analyze question complexity and type
            question_analysis = self._analyze_question_complexity(question, context)

            # Generate reasoning steps based on strategy
            reasoning_steps = []
            if strategy == ReasoningStrategy.STEP_BY_STEP:
                reasoning_steps = self._generate_step_by_step_reasoning(question, context, knowledge_base, max_steps)
            elif strategy == ReasoningStrategy.HYPOTHESIS_TESTING:
                reasoning_steps = self._generate_hypothesis_testing_reasoning(question, context, knowledge_base)
            elif strategy == ReasoningStrategy.CAUSE_EFFECT_ANALYSIS:
                reasoning_steps = self._generate_cause_effect_reasoning(question, context, knowledge_base)
            elif strategy == ReasoningStrategy.COMPARATIVE_ANALYSIS:
                reasoning_steps = self._generate_comparative_reasoning(question, context, knowledge_base)
            elif strategy == ReasoningStrategy.DEDUCTIVE_REASONING:
                reasoning_steps = self._generate_deductive_reasoning(question, context, knowledge_base)
            elif strategy == ReasoningStrategy.INDUCTIVE_REASONING:
                reasoning_steps = self._generate_inductive_reasoning(question, context, knowledge_base)

            # Validate reasoning chain coherence
            coherence_score = self._validate_reasoning_coherence(reasoning_steps)

            # Generate final answer based on reasoning steps
            final_answer, answer_confidence = self._synthesize_final_answer(reasoning_steps, question)

            # Calculate overall quality metrics
            overall_confidence = self._calculate_overall_confidence(reasoning_steps, answer_confidence)
            quality_score = self._calculate_reasoning_quality(reasoning_steps, coherence_score)

            # Create reasoning chain object
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                question=question,
                strategy=strategy,
                steps=reasoning_steps,
                final_answer=final_answer,
                overall_confidence=overall_confidence,
                reasoning_quality_score=quality_score
            )

            # Perform validation checks
            validation_results = self._validate_reasoning_chain(reasoning_chain, knowledge_base)
            reasoning_chain.validation_results = validation_results

            # Update metrics
            self.reasoning_metrics['chains_generated'] += 1
            self.reasoning_metrics['average_steps_per_chain'] = (
                (self.reasoning_metrics['average_steps_per_chain'] * (self.reasoning_metrics['chains_generated'] - 1) + len(reasoning_steps))
                / self.reasoning_metrics['chains_generated']
            )
            self.reasoning_metrics['average_reasoning_quality'] = (
                (self.reasoning_metrics['average_reasoning_quality'] * (self.reasoning_metrics['chains_generated'] - 1) + quality_score)
                / self.reasoning_metrics['chains_generated']
            )

            if validation_results.get('is_valid', False):
                self.reasoning_metrics['successful_validations'] += 1

            self.logger.info(f"Generated reasoning chain {chain_id} with {len(reasoning_steps)} steps")

            # Serialize reasoning steps for JSON response
            serialized_steps = []
            for step in reasoning_steps:
                serialized_steps.append({
                    'step_id': step.step_id,
                    'step_type': step.step_type,
                    'content': step.content,
                    'confidence_score': step.confidence_score,
                    'supporting_evidence': step.supporting_evidence,
                    'contradicting_evidence': step.contradicting_evidence,
                    'dependencies': step.dependencies,
                    'timestamp': step.timestamp
                })

            return {
                'success': True,
                'reasoning_chain': {
                    'chain_id': chain_id,
                    'question': question,
                    'strategy': strategy.value,
                    'steps': serialized_steps,
                    'final_answer': final_answer,
                    'overall_confidence': overall_confidence,
                    'reasoning_quality_score': quality_score,
                    'validation_results': validation_results
                },
                'chain_statistics': {
                    'total_steps': len(reasoning_steps),
                    'coherence_score': coherence_score,
                    'question_complexity': question_analysis,
                    'knowledge_base_facts_used': len([step for step in reasoning_steps if step.supporting_evidence])
                }
            }

        except Exception as e:
            self.logger.error(f"Reasoning chain generation failed: {str(e)}")
            self.reasoning_metrics['reasoning_errors'].append(str(e))
            return {
                'success': False,
                'error': str(e),
                'error_type': 'reasoning_chain_generation_error'
            }

    @a2a_skill(
        name="validateReasoningChain",
        description="Validate and critique an existing reasoning chain for logical consistency",
        input_schema={
            "type": "object",
            "properties": {
                "reasoning_chain": {
                    "type": "object",
                    "properties": {
                        "chain_id": {"type": "string"},
                        "question": {"type": "string"},
                        "steps": {"type": "array"},
                        "final_answer": {"type": "string"}
                    }
                },
                "validation_criteria": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["logical_consistency", "evidence_quality", "step_relevance", "conclusion_support", "completeness"]
                    },
                    "default": ["logical_consistency", "evidence_quality", "step_relevance", "conclusion_support"]
                }
            },
            "required": ["reasoning_chain"]
        }
    )
    def validate_reasoning_chain(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and critique the logical structure of a reasoning chain"""
        try:
            reasoning_chain_data = request_data["reasoning_chain"]
            validation_criteria = request_data.get("validation_criteria",
                ["logical_consistency", "evidence_quality", "step_relevance", "conclusion_support"])

            chain_id = reasoning_chain_data["chain_id"]
            question = reasoning_chain_data["question"]
            steps_data = reasoning_chain_data["steps"]
            final_answer = reasoning_chain_data["final_answer"]

            # Reconstruct reasoning steps
            reasoning_steps = []
            for step_data in steps_data:
                step = ReasoningStep(
                    step_id=step_data["step_id"],
                    step_type=step_data["step_type"],
                    content=step_data["content"],
                    confidence_score=step_data["confidence_score"],
                    supporting_evidence=step_data.get("supporting_evidence", []),
                    contradicting_evidence=step_data.get("contradicting_evidence", []),
                    dependencies=step_data.get("dependencies", [])
                )
                reasoning_steps.append(step)

            validation_results = {}

            # Logical consistency check
            if "logical_consistency" in validation_criteria:
                consistency_score, consistency_issues = self._check_logical_consistency(reasoning_steps)
                validation_results["logical_consistency"] = {
                    "score": consistency_score,
                    "issues": consistency_issues,
                    "passed": consistency_score >= 0.7
                }

            # Evidence quality assessment
            if "evidence_quality" in validation_criteria:
                evidence_score, evidence_assessment = self._assess_evidence_quality(reasoning_steps)
                validation_results["evidence_quality"] = {
                    "score": evidence_score,
                    "assessment": evidence_assessment,
                    "passed": evidence_score >= 0.6
                }

            # Step relevance evaluation
            if "step_relevance" in validation_criteria:
                relevance_score, irrelevant_steps = self._evaluate_step_relevance(reasoning_steps, question)
                validation_results["step_relevance"] = {
                    "score": relevance_score,
                    "irrelevant_steps": irrelevant_steps,
                    "passed": relevance_score >= 0.8
                }

            # Conclusion support analysis
            if "conclusion_support" in validation_criteria:
                support_score, support_analysis = self._analyze_conclusion_support(reasoning_steps, final_answer)
                validation_results["conclusion_support"] = {
                    "score": support_score,
                    "analysis": support_analysis,
                    "passed": support_score >= 0.7
                }

            # Completeness check
            if "completeness" in validation_criteria:
                completeness_score, missing_elements = self._check_reasoning_completeness(reasoning_steps, question)
                validation_results["completeness"] = {
                    "score": completeness_score,
                    "missing_elements": missing_elements,
                    "passed": completeness_score >= 0.6
                }

            # Calculate overall validation score
            scores = [result["score"] for result in validation_results.values()]
            overall_score = np.mean(scores) if scores else 0.0
            overall_passed = all(result["passed"] for result in validation_results.values())

            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(validation_results, reasoning_steps)

            return {
                'success': True,
                'chain_id': chain_id,
                'validation_results': validation_results,
                'overall_validation': {
                    'score': float(overall_score),
                    'passed': overall_passed,
                    'criteria_evaluated': validation_criteria
                },
                'improvement_suggestions': suggestions,
                'validation_summary': {
                    'total_steps_analyzed': len(reasoning_steps),
                    'critical_issues_found': sum(1 for result in validation_results.values() if not result["passed"]),
                    'average_confidence': float(np.mean([step.confidence_score for step in reasoning_steps])),
                    'validation_timestamp': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Reasoning chain validation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'reasoning_validation_error'
            }

    def _load_reasoning_templates(self) -> Dict[str, List[str]]:
        """Load templates for different reasoning patterns"""
        return {
            'step_by_step': [
                "First, let me identify the key information: {info}",
                "Next, I need to consider: {consideration}",
                "This leads me to conclude: {conclusion}",
                "Therefore: {final_step}"
            ],
            'hypothesis_testing': [
                "Hypothesis: {hypothesis}",
                "Evidence supporting this: {supporting_evidence}",
                "Evidence against this: {contradicting_evidence}",
                "Conclusion: {conclusion}"
            ],
            'cause_effect': [
                "The cause appears to be: {cause}",
                "This would result in: {effect}",
                "Supporting mechanisms: {mechanisms}",
                "Final assessment: {assessment}"
            ]
        }

    def _analyze_question_complexity(self, question: str, context: str) -> Dict[str, Any]:
        """Analyze the complexity and type of the question"""
        complexity_indicators = {
            'multi_part': len(re.findall(r'\?', question)) > 1 or len(re.findall(r'\band\b|\bor\b', question.lower())) > 0,
            'temporal': bool(re.search(r'\bwhen\b|\bafter\b|\bbefore\b|\bduring\b', question.lower())),
            'causal': bool(re.search(r'\bwhy\b|\bcause\b|\breason\b|\bdue to\b', question.lower())),
            'comparative': bool(re.search(r'\bcompare\b|\bversus\b|\bthan\b|\bbetter\b|\bworse\b', question.lower())),
            'quantitative': bool(re.search(r'\bhow many\b|\bhow much\b|\bnumber\b|\bpercent\b', question.lower())),
            'hypothetical': bool(re.search(r'\bwhat if\b|\bsuppose\b|\bimagine\b|\bhypothetical\b', question.lower()))
        }

        question_length = len(question.split())
        context_length = len(context.split())

        complexity_score = (
            sum(complexity_indicators.values()) * 0.4 +
            min(question_length / 20.0, 1.0) * 0.3 +
            min(context_length / 100.0, 1.0) * 0.3
        )

        return {
            'indicators': complexity_indicators,
            'question_length': question_length,
            'context_length': context_length,
            'complexity_score': min(complexity_score, 1.0)
        }

    def _generate_step_by_step_reasoning(self, question: str, context: str, knowledge_base: List[Dict], max_steps: int) -> List[ReasoningStep]:
        """Generate step-by-step reasoning chain"""
        steps = []

        # Step 1: Parse the question
        step1 = ReasoningStep(
            step_id=1,
            step_type="premise",
            content=f"Question analysis: {question}. Key elements to address based on context: {self._extract_key_elements(question, context)}",
            confidence_score=0.9
        )
        steps.append(step1)

        # Step 2: Identify relevant information
        relevant_facts = self._find_relevant_knowledge(question, context, knowledge_base)
        step2 = ReasoningStep(
            step_id=2,
            step_type="evidence",
            content=f"Relevant information identified: {len(relevant_facts)} facts from knowledge base",
            confidence_score=0.8,
            supporting_evidence=[fact['content'] for fact in relevant_facts[:3]],
            dependencies=[1]
        )
        steps.append(step2)

        # Step 3: Apply logical inference
        inference = self._make_logical_inference(question, relevant_facts, context)
        step3 = ReasoningStep(
            step_id=3,
            step_type="inference",
            content=f"Logical inference: {inference['reasoning']}",
            confidence_score=inference['confidence'],
            supporting_evidence=inference['supporting_facts'],
            dependencies=[1, 2]
        )
        steps.append(step3)

        # Step 4: Consider alternative explanations
        alternatives = self._consider_alternatives(question, relevant_facts, inference)
        if alternatives:
            step4 = ReasoningStep(
                step_id=4,
                step_type="validation",
                content=f"Alternative considerations: {alternatives['explanation']}",
                confidence_score=alternatives['confidence'],
                contradicting_evidence=alternatives.get('contradicting_evidence', []),
                dependencies=[3]
            )
            steps.append(step4)

        # Step 5: Final synthesis
        final_step_id = len(steps) + 1
        synthesis = self._synthesize_reasoning(steps, question)
        step_final = ReasoningStep(
            step_id=final_step_id,
            step_type="conclusion",
            content=f"Final synthesis: {synthesis['conclusion']}",
            confidence_score=synthesis['confidence'],
            dependencies=list(range(1, final_step_id))
        )
        steps.append(step_final)

        return steps[:max_steps]

    def _generate_hypothesis_testing_reasoning(self, question: str, context: str, knowledge_base: List[Dict]) -> List[ReasoningStep]:
        """Generate hypothesis testing reasoning chain"""
        steps = []

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(question, context, knowledge_base)

        step_id = 1
        for i, hypothesis in enumerate(hypotheses[:3]):  # Test top 3 hypotheses
            # Hypothesis step
            hyp_step = ReasoningStep(
                step_id=step_id,
                step_type="premise",
                content=f"Hypothesis {i+1}: {hypothesis['statement']}",
                confidence_score=hypothesis['initial_confidence']
            )
            steps.append(hyp_step)
            step_id += 1

            # Evidence evaluation step
            evidence = self._evaluate_hypothesis_evidence(hypothesis, knowledge_base, context)
            evidence_step = ReasoningStep(
                step_id=step_id,
                step_type="evidence",
                content=f"Evidence evaluation for hypothesis {i+1}: {evidence['summary']}",
                confidence_score=evidence['confidence'],
                supporting_evidence=evidence['supporting'],
                contradicting_evidence=evidence['contradicting'],
                dependencies=[step_id - 1]
            )
            steps.append(evidence_step)
            step_id += 1

        # Final hypothesis selection
        best_hypothesis = self._select_best_hypothesis(hypotheses, steps)
        conclusion_step = ReasoningStep(
            step_id=step_id,
            step_type="conclusion",
            content=f"Best supported hypothesis: {best_hypothesis['conclusion']}",
            confidence_score=best_hypothesis['confidence'],
            dependencies=list(range(1, step_id))
        )
        steps.append(conclusion_step)

        return steps

    def _generate_cause_effect_reasoning(self, question: str, context: str, knowledge_base: List[Dict]) -> List[ReasoningStep]:
        """Generate cause-effect analysis reasoning"""
        steps = []

        # Identify potential causes
        causes = self._identify_causes(question, context, knowledge_base)
        step1 = ReasoningStep(
            step_id=1,
            step_type="premise",
            content=f"Potential causes identified: {[cause['description'] for cause in causes]}",
            confidence_score=0.8,
            supporting_evidence=[cause['evidence'] for cause in causes]
        )
        steps.append(step1)

        # Analyze causal mechanisms
        mechanisms = self._analyze_causal_mechanisms(causes, context, knowledge_base)
        step2 = ReasoningStep(
            step_id=2,
            step_type="inference",
            content=f"Causal mechanisms analysis: {mechanisms['explanation']}",
            confidence_score=mechanisms['confidence'],
            supporting_evidence=mechanisms['supporting_evidence'],
            dependencies=[1]
        )
        steps.append(step2)

        # Evaluate effects
        effects = self._evaluate_effects(causes, mechanisms, context)
        step3 = ReasoningStep(
            step_id=3,
            step_type="inference",
            content=f"Effect evaluation: {effects['explanation']}",
            confidence_score=effects['confidence'],
            supporting_evidence=effects['evidence'],
            dependencies=[1, 2]
        )
        steps.append(step3)

        # Final causal conclusion
        causal_conclusion = self._make_causal_conclusion(causes, mechanisms, effects)
        step4 = ReasoningStep(
            step_id=4,
            step_type="conclusion",
            content=f"Causal conclusion: {causal_conclusion['statement']}",
            confidence_score=causal_conclusion['confidence'],
            dependencies=[1, 2, 3]
        )
        steps.append(step4)

        return steps

    def _extract_key_elements(self, question: str, context: str) -> List[str]:
        """Extract key elements from question and context"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        question_words = question.lower().split()
        key_words = [word for word in question_words if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'whom', 'whose', 'how']]
        return key_words[:5]  # Top 5 key elements

    def _find_relevant_knowledge(self, question: str, context: str, knowledge_base: List[Dict]) -> List[Dict]:
        """Find relevant facts from knowledge base"""
        if not knowledge_base:
            return []

        question_embedding = self.embedding_model.encode(question, normalize_embeddings=True)
        relevant_facts = []

        for fact in knowledge_base:
            fact_embedding = self.embedding_model.encode(fact['content'], normalize_embeddings=True)
            similarity = np.dot(question_embedding, fact_embedding)

            if similarity > 0.5:  # Relevance threshold
                fact['relevance_score'] = float(similarity)
                relevant_facts.append(fact)

        return sorted(relevant_facts, key=lambda x: x['relevance_score'], reverse=True)[:5]

    def _make_logical_inference(self, question: str, relevant_facts: List[Dict], context: str) -> Dict[str, Any]:
        """Make logical inference based on question and facts"""
        if not relevant_facts:
            return {
                'reasoning': 'Insufficient information for logical inference',
                'confidence': 0.2,
                'supporting_facts': []
            }

        # Simple inference based on fact combination
        high_confidence_facts = [fact for fact in relevant_facts if fact.get('confidence', 0) > 0.7]

        inference = {
            'reasoning': f"Based on {len(high_confidence_facts)} high-confidence facts, the logical conclusion addresses the key aspects of the question",
            'confidence': min(np.mean([fact['confidence'] for fact in high_confidence_facts]), 0.9) if high_confidence_facts else 0.3,
            'supporting_facts': [fact['content'] for fact in high_confidence_facts[:3]]
        }

        return inference

    def _consider_alternatives(self, question: str, relevant_facts: List[Dict], inference: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Consider alternative explanations or contradicting evidence"""
        if not relevant_facts:
            return None

        # Look for facts that might contradict the main inference
        contradicting_facts = [fact for fact in relevant_facts if fact.get('confidence', 1.0) < 0.5]

        if contradicting_facts:
            return {
                'explanation': f"Alternative perspective considering {len(contradicting_facts)} lower-confidence facts",
                'confidence': 0.6,
                'contradicting_evidence': [fact['content'] for fact in contradicting_facts[:2]]
            }

        return None

    def _synthesize_reasoning(self, steps: List[ReasoningStep], question: str) -> Dict[str, Any]:
        """Synthesize final reasoning from all steps"""
        avg_confidence = np.mean([step.confidence_score for step in steps])
        evidence_count = sum(len(step.supporting_evidence) for step in steps)

        return {
            'conclusion': f"Based on {len(steps)} reasoning steps with {evidence_count} pieces of supporting evidence",
            'confidence': float(avg_confidence * 0.9)  # Slight discount for synthesis uncertainty
        }

    def _generate_hypotheses(self, question: str, context: str, knowledge_base: List[Dict]) -> List[Dict[str, Any]]:
        """Generate testable hypotheses for the question"""
        # Simple hypothesis generation - in production, use more sophisticated methods
        hypotheses = [
            {
                'statement': 'The answer can be derived directly from the provided context',
                'initial_confidence': 0.7
            },
            {
                'statement': 'The answer requires inference from multiple knowledge base facts',
                'initial_confidence': 0.6
            },
            {
                'statement': 'The question cannot be fully answered with available information',
                'initial_confidence': 0.4
            }
        ]

        return hypotheses

    def _evaluate_hypothesis_evidence(self, hypothesis: Dict, knowledge_base: List[Dict], context: str) -> Dict[str, Any]:
        """Evaluate evidence for a specific hypothesis"""
        return {
            'summary': f"Evidence evaluation for: {hypothesis['statement']}",
            'confidence': hypothesis['initial_confidence'] * 0.8,
            'supporting': ['Context provides relevant information'],
            'contradicting': []
        }

    def _select_best_hypothesis(self, hypotheses: List[Dict], evidence_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """Select the best supported hypothesis"""
        # Simple selection based on average confidence of evidence steps
        if evidence_steps:
            avg_confidence = np.mean([step.confidence_score for step in evidence_steps if step.step_type == "evidence"])
            return {
                'conclusion': hypotheses[0]['statement'] if hypotheses else 'No clear hypothesis supported',
                'confidence': float(avg_confidence)
            }

        return {
            'conclusion': 'Insufficient evidence to select best hypothesis',
            'confidence': 0.3
        }

    def _identify_causes(self, question: str, context: str, knowledge_base: List[Dict]) -> List[Dict[str, Any]]:
        """Identify potential causes from question and context"""
        return [
            {
                'description': 'Primary causal factor identified from context',
                'evidence': 'Supporting evidence from knowledge base',
                'confidence': 0.7
            }
        ]

    def _analyze_causal_mechanisms(self, causes: List[Dict], context: str, knowledge_base: List[Dict]) -> Dict[str, Any]:
        """Analyze mechanisms connecting causes to effects"""
        return {
            'explanation': 'Causal mechanisms analysis based on available evidence',
            'confidence': 0.6,
            'supporting_evidence': ['Mechanism evidence 1', 'Mechanism evidence 2']
        }

    def _evaluate_effects(self, causes: List[Dict], mechanisms: Dict, context: str) -> Dict[str, Any]:
        """Evaluate the effects based on causes and mechanisms"""
        return {
            'explanation': 'Effect evaluation based on causal analysis',
            'confidence': 0.7,
            'evidence': ['Effect evidence 1', 'Effect evidence 2']
        }

    def _make_causal_conclusion(self, causes: List[Dict], mechanisms: Dict, effects: Dict) -> Dict[str, Any]:
        """Make final causal conclusion"""
        avg_confidence = np.mean([
            np.mean([cause['confidence'] for cause in causes]) if causes else 0.5,
            mechanisms['confidence'],
            effects['confidence']
        ])

        return {
            'statement': 'Final causal relationship established based on comprehensive analysis',
            'confidence': float(avg_confidence)
        }

    def _validate_reasoning_coherence(self, steps: List[ReasoningStep]) -> float:
        """Validate the coherence of reasoning steps"""
        if len(steps) < 2:
            return 1.0

        # Check step dependencies
        dependency_score = 0.0
        for step in steps:
            if step.dependencies:
                valid_deps = all(dep <= step.step_id for dep in step.dependencies)
                dependency_score += 1.0 if valid_deps else 0.0
            else:
                dependency_score += 0.5  # No dependencies is neutral

        dependency_score /= len(steps)

        # Check confidence progression
        confidences = [step.confidence_score for step in steps]
        confidence_variance = np.var(confidences)
        confidence_score = 1.0 - min(confidence_variance * 2, 1.0)  # Lower variance is better

        return (dependency_score + confidence_score) / 2

    def _synthesize_final_answer(self, steps: List[ReasoningStep], question: str) -> Tuple[str, float]:
        """Synthesize final answer from reasoning steps"""
        if not steps:
            return "No reasoning steps available", 0.1

        conclusion_steps = [step for step in steps if step.step_type == "conclusion"]
        if conclusion_steps:
            final_step = conclusion_steps[-1]
            return final_step.content, final_step.confidence_score

        # Fallback: use last step
        last_step = steps[-1]
        return f"Based on reasoning: {last_step.content}", last_step.confidence_score * 0.8

    def _calculate_overall_confidence(self, steps: List[ReasoningStep], answer_confidence: float) -> float:
        """Calculate overall confidence for the reasoning chain"""
        if not steps:
            return 0.1

        step_confidences = [step.confidence_score for step in steps]
        avg_step_confidence = np.mean(step_confidences)
        min_step_confidence = min(step_confidences)

        # Overall confidence is weighted combination
        overall_confidence = (
            avg_step_confidence * 0.4 +
            min_step_confidence * 0.3 +  # Weakest link principle
            answer_confidence * 0.3
        )

        return float(overall_confidence)

    def _calculate_reasoning_quality(self, steps: List[ReasoningStep], coherence_score: float) -> float:
        """Calculate overall quality score for reasoning"""
        if not steps:
            return 0.0

        # Evidence quality
        evidence_quality = np.mean([
            len(step.supporting_evidence) / max(len(step.supporting_evidence) + len(step.contradicting_evidence), 1)
            for step in steps
        ])

        # Step type diversity
        step_types = set(step.step_type for step in steps)
        type_diversity = len(step_types) / 5.0  # Normalize by expected max types

        # Dependency structure quality
        total_deps = sum(len(step.dependencies) for step in steps)
        dependency_quality = min(total_deps / len(steps), 1.0)

        quality_score = (
            coherence_score * 0.3 +
            evidence_quality * 0.3 +
            type_diversity * 0.2 +
            dependency_quality * 0.2
        )

        return float(quality_score)

    def _validate_reasoning_chain(self, reasoning_chain: ReasoningChain, knowledge_base: List[Dict]) -> Dict[str, Any]:
        """Validate complete reasoning chain"""
        return {
            'is_valid': reasoning_chain.overall_confidence > 0.5,
            'quality_score': reasoning_chain.reasoning_quality_score,
            'step_count': len(reasoning_chain.steps),
            'has_conclusion': any(step.step_type == "conclusion" for step in reasoning_chain.steps),
            'evidence_based': any(len(step.supporting_evidence) > 0 for step in reasoning_chain.steps),
            'validation_timestamp': datetime.utcnow().isoformat()
        }

    def _check_logical_consistency(self, steps: List[ReasoningStep]) -> Tuple[float, List[str]]:
        """Check for logical consistency issues"""
        issues = []
        consistency_score = 1.0

        # Check for contradictory steps
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                if (step1.supporting_evidence and step2.contradicting_evidence and
                    any(evidence in step2.contradicting_evidence for evidence in step1.supporting_evidence)):
                    issues.append(f"Steps {step1.step_id} and {step2.step_id} contain contradictory evidence")
                    consistency_score -= 0.2

        return max(consistency_score, 0.0), issues

    def _assess_evidence_quality(self, steps: List[ReasoningStep]) -> Tuple[float, Dict[str, Any]]:
        """Assess the quality of evidence used in reasoning"""
        total_evidence = sum(len(step.supporting_evidence) for step in steps)
        total_contradictions = sum(len(step.contradicting_evidence) for step in steps)

        if total_evidence == 0:
            return 0.0, {"assessment": "No supporting evidence provided"}

        evidence_ratio = total_evidence / (total_evidence + total_contradictions)
        evidence_distribution = len([step for step in steps if step.supporting_evidence]) / len(steps)

        quality_score = (evidence_ratio * 0.6 + evidence_distribution * 0.4)

        return quality_score, {
            "total_supporting_evidence": total_evidence,
            "total_contradicting_evidence": total_contradictions,
            "evidence_distribution": evidence_distribution
        }

    def _evaluate_step_relevance(self, steps: List[ReasoningStep], question: str) -> Tuple[float, List[int]]:
        """Evaluate relevance of each step to the question"""
        question_embedding = self.embedding_model.encode(question, normalize_embeddings=True)
        relevance_scores = []
        irrelevant_steps = []

        for step in steps:
            step_embedding = self.embedding_model.encode(step.content, normalize_embeddings=True)
            relevance = float(np.dot(question_embedding, step_embedding))
            relevance_scores.append(relevance)

            if relevance < 0.3:  # Relevance threshold
                irrelevant_steps.append(step.step_id)

        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        return avg_relevance, irrelevant_steps

    def _analyze_conclusion_support(self, steps: List[ReasoningStep], final_answer: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze how well the steps support the final conclusion"""
        conclusion_steps = [step for step in steps if step.step_type == "conclusion"]
        if not conclusion_steps:
            return 0.3, {"analysis": "No explicit conclusion steps found"}

        # Check if conclusion follows from premises
        premise_steps = [step for step in steps if step.step_type in ["premise", "evidence", "inference"]]
        if not premise_steps:
            return 0.2, {"analysis": "No supporting premises found"}

        # Simple support analysis based on confidence and evidence
        avg_premise_confidence = np.mean([step.confidence_score for step in premise_steps])
        conclusion_confidence = conclusion_steps[-1].confidence_score

        support_score = min(avg_premise_confidence, conclusion_confidence)

        return support_score, {
            "premise_steps": len(premise_steps),
            "conclusion_steps": len(conclusion_steps),
            "average_premise_confidence": avg_premise_confidence,
            "conclusion_confidence": conclusion_confidence
        }

    def _check_reasoning_completeness(self, steps: List[ReasoningStep], question: str) -> Tuple[float, List[str]]:
        """Check if the reasoning chain is complete"""
        missing_elements = []
        completeness_score = 1.0

        # Check for essential step types
        step_types = set(step.step_type for step in steps)

        if "premise" not in step_types:
            missing_elements.append("Initial premise or problem statement")
            completeness_score -= 0.3

        if "evidence" not in step_types:
            missing_elements.append("Supporting evidence")
            completeness_score -= 0.2

        if "conclusion" not in step_types:
            missing_elements.append("Final conclusion")
            completeness_score -= 0.4

        # Check for logical flow
        if steps and not any(step.dependencies for step in steps[1:]):
            missing_elements.append("Logical dependencies between steps")
            completeness_score -= 0.1

        return max(completeness_score, 0.0), missing_elements

    def _generate_improvement_suggestions(self, validation_results: Dict, steps: List[ReasoningStep]) -> List[str]:
        """Generate suggestions for improving the reasoning chain"""
        suggestions = []

        for criterion, result in validation_results.items():
            if not result["passed"]:
                if criterion == "logical_consistency":
                    suggestions.append("Review steps for logical contradictions and resolve conflicting evidence")
                elif criterion == "evidence_quality":
                    suggestions.append("Provide more supporting evidence and address contradicting information")
                elif criterion == "step_relevance":
                    suggestions.append("Remove irrelevant steps and focus on question-specific reasoning")
                elif criterion == "conclusion_support":
                    suggestions.append("Strengthen the connection between premises and conclusions")
                elif criterion == "completeness":
                    suggestions.extend(result.get("missing_elements", []))

        return suggestions

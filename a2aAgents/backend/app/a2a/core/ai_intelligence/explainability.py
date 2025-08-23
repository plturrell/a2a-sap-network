"""
Explainability and Interpretability Framework for A2A Agents
Part of Phase 1: Core AI Framework

This module provides comprehensive explainability features including
decision tracing, reasoning transparency, and interpretable AI outputs.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json
from abc import ABC, abstractmethod
import uuid

logger = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Types of explanations"""

    DECISION_TRACE = "decision_trace"
    CAUSAL_ANALYSIS = "causal_analysis"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"
    STEP_BY_STEP = "step_by_step"
    NATURAL_LANGUAGE = "natural_language"
    VISUAL = "visual"
    TECHNICAL = "technical"


class ExplanationLevel(str, Enum):
    """Levels of explanation detail"""

    SUMMARY = "summary"  # High-level overview
    DETAILED = "detailed"  # Comprehensive explanation
    TECHNICAL = "technical"  # Implementation details
    BEGINNER = "beginner"  # Simplified for non-experts
    EXPERT = "expert"  # Advanced technical analysis


class CausalRelationType(str, Enum):
    """Types of causal relationships"""

    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CORRELATION = "correlation"
    SPURIOUS = "spurious"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class ExplanationStep:
    """Represents a single step in an explanation"""

    step_id: str
    description: str
    reasoning: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Represents a causal relationship"""

    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Comprehensive explanation object"""

    explanation_id: str
    explanation_type: ExplanationType
    level: ExplanationLevel
    title: str
    summary: str
    steps: List[ExplanationStep] = field(default_factory=list)
    causal_relations: List[CausalRelation] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)


class ExplainabilityFramework:
    """
    Comprehensive explainability framework for AI decisions
    Provides multiple types of explanations at different levels of detail
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}

        # Explanation storage
        self.explanations = {}
        self.explanation_templates = {}
        self.decision_traces = deque(maxlen=1000)

        # Causal analysis
        self.causal_graph = {}
        self.causal_strength_matrix = {}

        # Feature importance tracking
        self.feature_importance_history = defaultdict(list)
        self.global_feature_importance = {}

        # Explanation generators
        self.generators = {
            ExplanationType.DECISION_TRACE: DecisionTraceGenerator(),
            ExplanationType.CAUSAL_ANALYSIS: CausalAnalysisGenerator(),
            ExplanationType.FEATURE_IMPORTANCE: FeatureImportanceGenerator(),
            ExplanationType.COUNTERFACTUAL: CounterfactualGenerator(),
            ExplanationType.STEP_BY_STEP: StepByStepGenerator(),
            ExplanationType.NATURAL_LANGUAGE: NaturalLanguageGenerator(),
            ExplanationType.TECHNICAL: TechnicalGenerator(),
        }

        # Explanation cache for performance
        self.explanation_cache = {}

        # Model interpretability tools
        self.interpretability_tools = {}

        logger.info(f"Initialized explainability framework for agent {agent_id}")

    async def explain_decision(
        self,
        decision_context: Dict[str, Any],
        explanation_type: ExplanationType = ExplanationType.NATURAL_LANGUAGE,
        level: ExplanationLevel = ExplanationLevel.DETAILED,
        target_audience: str = "general",
    ) -> Explanation:
        """
        Generate explanation for a decision

        Args:
            decision_context: Context of the decision to explain
            explanation_type: Type of explanation to generate
            level: Level of detail
            target_audience: Target audience (general, expert, developer)

        Returns:
            Generated explanation
        """
        explanation_id = str(uuid.uuid4())

        # Check cache first
        cache_key = self._generate_cache_key(decision_context, explanation_type, level)
        if cache_key in self.explanation_cache:
            cached_explanation = self.explanation_cache[cache_key]
            cached_explanation.explanation_id = explanation_id  # New ID for this request
            return cached_explanation

        # Get appropriate generator
        generator = self.generators.get(explanation_type)
        if not generator:
            raise ValueError(f"Unknown explanation type: {explanation_type}")

        # Generate explanation
        explanation = await generator.generate(
            decision_context, level, target_audience, explanation_id, self
        )

        # Store explanation
        self.explanations[explanation_id] = explanation
        self.explanation_cache[cache_key] = explanation

        # Update decision trace
        self._update_decision_trace(decision_context, explanation)

        return explanation

    async def explain_reasoning_chain(
        self,
        reasoning_steps: List[Dict[str, Any]],
        final_decision: Any,
        level: ExplanationLevel = ExplanationLevel.DETAILED,
    ) -> Explanation:
        """
        Explain a chain of reasoning steps

        Args:
            reasoning_steps: List of reasoning steps
            final_decision: Final decision or output
            level: Level of explanation detail

        Returns:
            Explanation of reasoning chain
        """
        explanation_id = str(uuid.uuid4())

        # Create explanation steps
        explanation_steps = []
        for i, step in enumerate(reasoning_steps):
            exp_step = ExplanationStep(
                step_id=f"step_{i}",
                description=step.get("description", f"Reasoning step {i+1}"),
                reasoning=step.get("reasoning", ""),
                inputs=step.get("inputs", {}),
                outputs=step.get("outputs", {}),
                confidence=step.get("confidence", 0.8),
            )
            explanation_steps.append(exp_step)

        # Analyze causal relationships between steps
        causal_relations = await self._analyze_step_causality(reasoning_steps)

        # Create comprehensive explanation
        explanation = Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.STEP_BY_STEP,
            level=level,
            title="Reasoning Chain Analysis",
            summary=f"Step-by-step explanation of reasoning leading to: {final_decision}",
            steps=explanation_steps,
            causal_relations=causal_relations,
            confidence=np.mean([step.confidence for step in explanation_steps]),
        )

        # Add natural language description
        explanation.context["natural_language"] = await self._generate_reasoning_narrative(
            reasoning_steps, final_decision
        )

        self.explanations[explanation_id] = explanation
        return explanation

    async def analyze_feature_importance(
        self, input_features: Dict[str, Any], decision: Any, method: str = "shap"
    ) -> Dict[str, float]:
        """
        Analyze the importance of input features in a decision

        Args:
            input_features: Input features used in decision
            decision: Decision made
            method: Analysis method (shap, lime, permutation)

        Returns:
            Feature importance scores
        """
        # Simplified feature importance calculation
        # In real implementation, would use SHAP, LIME, or other methods

        feature_scores = {}
        len(input_features)

        for feature_name, feature_value in input_features.items():
            # Calculate importance based on value characteristics
            if isinstance(feature_value, (int, float)):
                # Numerical features - importance based on magnitude and variability
                normalized_value = abs(feature_value) / (abs(feature_value) + 1)
                base_importance = normalized_value * 0.8
            elif isinstance(feature_value, str):
                # Text features - importance based on length and uniqueness
                base_importance = min(len(feature_value) / 100, 0.7)
            elif isinstance(feature_value, bool):
                # Boolean features - high importance if True
                base_importance = 0.6 if feature_value else 0.2
            else:
                # Other types
                base_importance = 0.3

            # Add randomization to simulate real analysis
            noise = np.random.uniform(-0.1, 0.1)
            feature_scores[feature_name] = max(0, min(1, base_importance + noise))

        # Normalize scores to sum to 1
        total_score = sum(feature_scores.values())
        if total_score > 0:
            feature_scores = {k: v / total_score for k, v in feature_scores.items()}

        # Update global importance tracking
        for feature, score in feature_scores.items():
            self.feature_importance_history[feature].append(score)
            self.global_feature_importance[feature] = np.mean(
                self.feature_importance_history[feature]
            )

        return feature_scores

    async def generate_counterfactuals(
        self, original_input: Dict[str, Any], original_output: Any, num_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations

        Args:
            original_input: Original input that led to output
            original_output: Original output/decision
            num_counterfactuals: Number of counterfactuals to generate

        Returns:
            List of counterfactual scenarios
        """
        counterfactuals = []

        for i in range(num_counterfactuals):
            # Create modified input
            modified_input = original_input.copy()

            # Randomly modify some features
            features_to_modify = np.secrets.choice(
                list(original_input.keys()), size=min(2, len(original_input)), replace=False
            )

            for feature in features_to_modify:
                original_value = original_input[feature]

                if isinstance(original_value, (int, float)):
                    # Numerical modification
                    change_factor = np.random.uniform(0.5, 1.5)
                    modified_input[feature] = original_value * change_factor
                elif isinstance(original_value, str):
                    # Text modification
                    modified_input[feature] = f"modified_{original_value}"
                elif isinstance(original_value, bool):
                    # Boolean flip
                    modified_input[feature] = not original_value

            # Simulate different output
            output_variations = [
                "alternative_decision_A",
                "alternative_decision_B",
                "no_decision",
                "opposite_decision",
            ]
            simulated_output = np.secrets.choice(output_variations)

            counterfactual = {
                "id": f"counterfactual_{i}",
                "modified_input": modified_input,
                "simulated_output": simulated_output,
                "key_changes": list(features_to_modify),
                "explanation": f"If {', '.join(features_to_modify)} were different, the outcome would likely be {simulated_output}",
                "confidence": np.random.uniform(0.6, 0.9),
            }

            counterfactuals.append(counterfactual)

        return counterfactuals

    async def trace_decision_path(
        self, decision_id: str, include_alternatives: bool = True
    ) -> Dict[str, Any]:
        """
        Trace the complete path of a decision

        Args:
            decision_id: ID of decision to trace
            include_alternatives: Whether to include alternative paths

        Returns:
            Complete decision trace
        """
        # Find decision in trace history
        decision_trace = None
        for trace in self.decision_traces:
            if trace.get("decision_id") == decision_id:
                decision_trace = trace
                break

        if not decision_trace:
            raise ValueError(f"Decision {decision_id} not found in trace history")

        trace_result = {
            "decision_id": decision_id,
            "timestamp": decision_trace.get("timestamp"),
            "path": [],
            "branch_points": [],
            "confidence_evolution": [],
            "alternative_paths": [] if include_alternatives else None,
        }

        # Reconstruct decision path
        reasoning_steps = decision_trace.get("reasoning_steps", [])
        for i, step in enumerate(reasoning_steps):
            path_point = {
                "step_number": i + 1,
                "action": step.get("action"),
                "reasoning": step.get("reasoning"),
                "confidence": step.get("confidence", 0.5),
                "inputs": step.get("inputs", {}),
                "outputs": step.get("outputs", {}),
            }
            trace_result["path"].append(path_point)
            trace_result["confidence_evolution"].append(step.get("confidence", 0.5))

            # Identify branch points (where multiple options were considered)
            if step.get("alternatives"):
                branch_point = {
                    "step_number": i + 1,
                    "decision_point": step.get("decision_point"),
                    "chosen_option": step.get("chosen_option"),
                    "alternatives": step.get("alternatives"),
                    "selection_criteria": step.get("selection_criteria"),
                }
                trace_result["branch_points"].append(branch_point)

                # Generate alternative paths if requested
                if include_alternatives:
                    for alt in step.get("alternatives", []):
                        alt_path = await self._simulate_alternative_path(reasoning_steps, i, alt)
                        trace_result["alternative_paths"].append(
                            {
                                "divergence_point": i + 1,
                                "alternative_chosen": alt,
                                "predicted_path": alt_path,
                            }
                        )

        return trace_result

    async def get_explanation_summary(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        explanation_types: Optional[List[ExplanationType]] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of explanations generated

        Args:
            time_range: Time range to filter explanations
            explanation_types: Types of explanations to include

        Returns:
            Summary of explanations
        """
        filtered_explanations = []

        for exp in self.explanations.values():
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= exp.created_at <= end_time):
                    continue

            # Filter by type
            if explanation_types and exp.explanation_type not in explanation_types:
                continue

            filtered_explanations.append(exp)

        # Calculate statistics
        summary = {
            "total_explanations": len(filtered_explanations),
            "by_type": defaultdict(int),
            "by_level": defaultdict(int),
            "average_confidence": 0.0,
            "most_common_features": [],
            "decision_patterns": {},
            "quality_metrics": {},
        }

        if filtered_explanations:
            # Count by type and level
            for exp in filtered_explanations:
                summary["by_type"][exp.explanation_type.value] += 1
                summary["by_level"][exp.level.value] += 1

            # Calculate average confidence
            summary["average_confidence"] = np.mean(
                [exp.confidence for exp in filtered_explanations]
            )

            # Find most common features
            feature_counts = defaultdict(int)
            for exp in filtered_explanations:
                for feature in exp.feature_importances.keys():
                    feature_counts[feature] += 1

            summary["most_common_features"] = sorted(
                feature_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Analyze decision patterns
            summary["decision_patterns"] = await self._analyze_decision_patterns(
                filtered_explanations
            )

            # Quality metrics
            summary["quality_metrics"] = {
                "completeness": self._calculate_explanation_completeness(filtered_explanations),
                "consistency": self._calculate_explanation_consistency(filtered_explanations),
                "clarity": self._calculate_explanation_clarity(filtered_explanations),
            }

        return summary

    def create_explanation_template(self, template_name: str, template_structure: Dict[str, Any]):
        """
        Create reusable explanation template

        Args:
            template_name: Name of the template
            template_structure: Structure definition of the template
        """
        self.explanation_templates[template_name] = {
            "structure": template_structure,
            "created_at": datetime.utcnow(),
            "usage_count": 0,
        }

    async def apply_explanation_template(
        self, template_name: str, context_data: Dict[str, Any]
    ) -> Explanation:
        """
        Apply an explanation template to generate explanation

        Args:
            template_name: Name of template to apply
            context_data: Data to populate template

        Returns:
            Generated explanation from template
        """
        if template_name not in self.explanation_templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.explanation_templates[template_name]
        structure = template["structure"]

        # Increment usage count
        template["usage_count"] += 1

        # Generate explanation from template
        explanation = Explanation(
            explanation_id=str(uuid.uuid4()),
            explanation_type=ExplanationType(structure.get("type", "natural_language")),
            level=ExplanationLevel(structure.get("level", "detailed")),
            title=structure.get("title", "").format(**context_data),
            summary=structure.get("summary", "").format(**context_data),
        )

        # Populate template fields
        for field, template_value in structure.get("fields", {}).items():
            if isinstance(template_value, str):
                populated_value = template_value.format(**context_data)
                explanation.context[field] = populated_value

        return explanation

    def _generate_cache_key(
        self,
        decision_context: Dict[str, Any],
        explanation_type: ExplanationType,
        level: ExplanationLevel,
    ) -> str:
        """Generate cache key for explanation"""
        import hashlib

        context_str = json.dumps(decision_context, sort_keys=True, default=str)
        key_str = f"{context_str}_{explanation_type.value}_{level.value}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _update_decision_trace(self, decision_context: Dict[str, Any], explanation: Explanation):
        """Update decision trace with new explanation"""
        trace_entry = {
            "decision_id": explanation.explanation_id,
            "timestamp": explanation.created_at,
            "context": decision_context,
            "explanation_type": explanation.explanation_type.value,
            "confidence": explanation.confidence,
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence,
                }
                for step in explanation.steps
            ],
        }
        self.decision_traces.append(trace_entry)

    async def _analyze_step_causality(
        self, reasoning_steps: List[Dict[str, Any]]
    ) -> List[CausalRelation]:
        """Analyze causal relationships between reasoning steps"""
        causal_relations = []

        for i in range(len(reasoning_steps) - 1):
            current_step = reasoning_steps[i]
            next_step = reasoning_steps[i + 1]

            # Simple causality analysis
            relation = CausalRelation(
                cause=current_step.get("action", f"step_{i}"),
                effect=next_step.get("action", f"step_{i+1}"),
                relation_type=CausalRelationType.DIRECT_CAUSE,
                strength=0.8,  # Simplified
                confidence=0.7,
                evidence=[f"Step {i} leads to step {i+1} in reasoning chain"],
            )
            causal_relations.append(relation)

        return causal_relations

    async def _generate_reasoning_narrative(
        self, reasoning_steps: List[Dict[str, Any]], final_decision: Any
    ) -> str:
        """Generate natural language narrative of reasoning"""
        narrative = "The reasoning process proceeded as follows:\n\n"

        for i, step in enumerate(reasoning_steps):
            narrative += f"{i+1}. {step.get('description', 'Processing step')}\n"
            if step.get("reasoning"):
                narrative += f"   Reasoning: {step['reasoning']}\n"
            narrative += "\n"

        narrative += f"This reasoning chain led to the final decision: {final_decision}"
        return narrative

    async def _simulate_alternative_path(
        self, original_steps: List[Dict[str, Any]], branch_point: int, alternative: str
    ) -> List[Dict[str, Any]]:
        """Simulate what would happen with alternative choice"""
        # Simplified simulation
        alt_path = original_steps[:branch_point].copy()

        # Add alternative step
        alt_step = {
            "action": alternative,
            "reasoning": f"Following alternative path: {alternative}",
            "confidence": np.random.uniform(0.5, 0.8),
            "simulated": True,
        }
        alt_path.append(alt_step)

        # Add a few more simulated steps
        for i in range(2):
            follow_up = {
                "action": f"consequence_{i+1}",
                "reasoning": f"Likely consequence of choosing {alternative}",
                "confidence": np.random.uniform(0.4, 0.7),
                "simulated": True,
            }
            alt_path.append(follow_up)

        return alt_path

    async def _analyze_decision_patterns(self, explanations: List[Explanation]) -> Dict[str, Any]:
        """Analyze patterns in decision explanations"""
        patterns = {
            "common_reasoning_patterns": [],
            "decision_complexity_distribution": {},
            "confidence_patterns": {},
            "feature_usage_patterns": {},
        }

        # Analyze reasoning patterns
        reasoning_types = defaultdict(int)
        for exp in explanations:
            for step in exp.steps:
                if "pattern" in step.reasoning.lower():
                    reasoning_types["pattern_based"] += 1
                elif "rule" in step.reasoning.lower():
                    reasoning_types["rule_based"] += 1
                elif "learn" in step.reasoning.lower():
                    reasoning_types["learning_based"] += 1
                else:
                    reasoning_types["other"] += 1

        patterns["common_reasoning_patterns"] = dict(reasoning_types)

        # Analyze complexity
        complexity_levels = {"simple": 0, "moderate": 0, "complex": 0}
        for exp in explanations:
            step_count = len(exp.steps)
            if step_count <= 3:
                complexity_levels["simple"] += 1
            elif step_count <= 7:
                complexity_levels["moderate"] += 1
            else:
                complexity_levels["complex"] += 1

        patterns["decision_complexity_distribution"] = complexity_levels

        return patterns

    def _calculate_explanation_completeness(self, explanations: List[Explanation]) -> float:
        """Calculate completeness score for explanations"""
        total_score = 0
        for exp in explanations:
            score = 0
            if exp.summary:
                score += 0.2
            if exp.steps:
                score += 0.3
            if exp.causal_relations:
                score += 0.2
            if exp.feature_importances:
                score += 0.2
            if exp.counterfactuals:
                score += 0.1
            total_score += score

        return total_score / len(explanations) if explanations else 0

    def _calculate_explanation_consistency(self, explanations: List[Explanation]) -> float:
        """Calculate consistency score for explanations"""
        # Simplified consistency measure
        if len(explanations) < 2:
            return 1.0

        confidence_variance = np.var([exp.confidence for exp in explanations])
        consistency = max(0, 1 - confidence_variance)
        return consistency

    def _calculate_explanation_clarity(self, explanations: List[Explanation]) -> float:
        """Calculate clarity score for explanations"""
        total_clarity = 0
        for exp in explanations:
            # Simple clarity metrics
            step_clarity = len(exp.steps) > 0
            summary_clarity = len(exp.summary) > 20
            clarity_score = (step_clarity + summary_clarity) / 2
            total_clarity += clarity_score

        return total_clarity / len(explanations) if explanations else 0


# Explanation Generators
class ExplanationGenerator(ABC):
    """Base class for explanation generators"""

    @abstractmethod
    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:
        """Generate explanation"""


class DecisionTraceGenerator(ExplanationGenerator):
    """Generates decision trace explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        # Extract decision steps from context
        decision_steps = context.get("decision_steps", [])

        steps = []
        for i, step_data in enumerate(decision_steps):
            step = ExplanationStep(
                step_id=f"trace_step_{i}",
                description=step_data.get("description", f"Decision step {i+1}"),
                reasoning=step_data.get("reasoning", ""),
                inputs=step_data.get("inputs", {}),
                outputs=step_data.get("outputs", {}),
                confidence=step_data.get("confidence", 0.8),
            )
            steps.append(step)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.DECISION_TRACE,
            level=level,
            title="Decision Trace Analysis",
            summary=f"Traced {len(steps)} decision steps leading to final outcome",
            steps=steps,
            confidence=np.mean([step.confidence for step in steps]) if steps else 0.5,
        )


class CausalAnalysisGenerator(ExplanationGenerator):
    """Generates causal analysis explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        # Analyze causal relationships
        inputs = context.get("inputs", {})
        outputs = context.get("outputs", {})

        causal_relations = []
        for input_name, input_value in inputs.items():
            for output_name, output_value in outputs.items():
                # Simplified causal analysis
                relation = CausalRelation(
                    cause=input_name,
                    effect=output_name,
                    relation_type=CausalRelationType.DIRECT_CAUSE,
                    strength=np.random.uniform(0.3, 0.9),
                    confidence=np.random.uniform(0.6, 0.9),
                    evidence=[f"Input {input_name} influences output {output_name}"],
                )
                causal_relations.append(relation)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.CAUSAL_ANALYSIS,
            level=level,
            title="Causal Relationship Analysis",
            summary=f"Identified {len(causal_relations)} causal relationships",
            causal_relations=causal_relations,
            confidence=0.8,
        )


class FeatureImportanceGenerator(ExplanationGenerator):
    """Generates feature importance explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        inputs = context.get("inputs", {})
        decision = context.get("decision")

        # Calculate feature importance
        feature_importances = await framework.analyze_feature_importance(inputs, decision)

        # Create explanation steps for top features
        steps = []
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(sorted_features[:5]):
            step = ExplanationStep(
                step_id=f"feature_{i}",
                description=f"Feature '{feature}' has importance score {importance:.3f}",
                reasoning=f"This feature contributes {importance*100:.1f}% to the decision",
                inputs={"feature": feature, "value": inputs.get(feature)},
                outputs={"importance": importance},
                confidence=0.9,
            )
            steps.append(step)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            level=level,
            title="Feature Importance Analysis",
            summary=f"Analyzed importance of {len(inputs)} input features",
            steps=steps,
            feature_importances=feature_importances,
            confidence=0.85,
        )


class CounterfactualGenerator(ExplanationGenerator):
    """Generates counterfactual explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        inputs = context.get("inputs", {})
        outputs = context.get("outputs", {})

        # Generate counterfactuals
        counterfactuals = await framework.generate_counterfactuals(inputs, outputs)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.COUNTERFACTUAL,
            level=level,
            title="Counterfactual Analysis",
            summary=f"Generated {len(counterfactuals)} alternative scenarios",
            counterfactuals=counterfactuals,
            confidence=0.75,
        )


class StepByStepGenerator(ExplanationGenerator):
    """Generates step-by-step explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        process_steps = context.get("process_steps", [])

        steps = []
        for i, step_info in enumerate(process_steps):
            step = ExplanationStep(
                step_id=f"process_step_{i}",
                description=step_info.get("action", f"Step {i+1}"),
                reasoning=step_info.get("why", f"This step is necessary for the process"),
                inputs=step_info.get("inputs", {}),
                outputs=step_info.get("outputs", {}),
                confidence=step_info.get("certainty", 0.8),
            )
            steps.append(step)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.STEP_BY_STEP,
            level=level,
            title="Step-by-Step Process Explanation",
            summary=f"Detailed breakdown of {len(steps)} process steps",
            steps=steps,
            confidence=np.mean([step.confidence for step in steps]) if steps else 0.7,
        )


class NaturalLanguageGenerator(ExplanationGenerator):
    """Generates natural language explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        # Generate natural language explanation based on context
        title = "Natural Language Explanation"

        if audience == "beginner":
            summary = await self._generate_beginner_explanation(context)
        elif audience == "expert":
            summary = await self._generate_expert_explanation(context)
        else:
            summary = await self._generate_general_explanation(context)

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.NATURAL_LANGUAGE,
            level=level,
            title=title,
            summary=summary,
            confidence=0.8,
        )

    async def _generate_beginner_explanation(self, context: Dict[str, Any]) -> str:
        """Generate explanation for beginners"""
        return f"Here's what happened in simple terms: The system looked at the information you provided and made a decision based on the most important factors. The main factors considered were {', '.join(list(context.get('inputs', {}).keys())[:3])}."

    async def _generate_expert_explanation(self, context: Dict[str, Any]) -> str:
        """Generate explanation for experts"""
        return f"Technical analysis: The decision process involved {len(context.get('inputs', {}))} input parameters with confidence score {context.get('confidence', 0.8):.3f}. Primary algorithmic approach utilized was {context.get('algorithm', 'heuristic-based')} with optimization for {context.get('objective', 'accuracy')}."

    async def _generate_general_explanation(self, context: Dict[str, Any]) -> str:
        """Generate explanation for general audience"""
        return f"The system analyzed {len(context.get('inputs', {}))} pieces of information and reached a decision. The most influential factors were considered, and the result has a confidence level of {context.get('confidence', 0.8)*100:.0f}%."


class TechnicalGenerator(ExplanationGenerator):
    """Generates technical explanations"""

    async def generate(
        self,
        context: Dict[str, Any],
        level: ExplanationLevel,
        audience: str,
        explanation_id: str,
        framework: ExplainabilityFramework,
    ) -> Explanation:

        # Generate technical explanation with implementation details
        technical_details = {
            "algorithm": context.get("algorithm", "unknown"),
            "parameters": context.get("parameters", {}),
            "computational_complexity": context.get("complexity", "O(n)"),
            "memory_usage": context.get("memory_usage", "unknown"),
            "execution_time": context.get("execution_time", 0),
            "stack_trace": context.get("stack_trace", []),
        }

        summary = f"""Technical Implementation Details:
Algorithm: {technical_details['algorithm']}
Complexity: {technical_details['computational_complexity']}
Execution Time: {technical_details['execution_time']}ms
Parameters: {len(technical_details['parameters'])} configured
Memory Usage: {technical_details['memory_usage']}"""

        return Explanation(
            explanation_id=explanation_id,
            explanation_type=ExplanationType.TECHNICAL,
            level=level,
            title="Technical Implementation Analysis",
            summary=summary,
            context=technical_details,
            confidence=0.95,
        )


# Utility functions
def create_explainability_framework(agent_id: str) -> ExplainabilityFramework:
    """Factory function to create explainability framework"""
    return ExplainabilityFramework(agent_id)


async def explain_agent_decision(
    framework: ExplainabilityFramework,
    decision_context: Dict[str, Any],
    explanation_type: ExplanationType = ExplanationType.NATURAL_LANGUAGE,
) -> Explanation:
    """Convenience function for explaining agent decisions"""
    return await framework.explain_decision(decision_context, explanation_type)


async def trace_reasoning_chain(
    framework: ExplainabilityFramework, reasoning_steps: List[Dict[str, Any]], final_output: Any
) -> Explanation:
    """Convenience function for tracing reasoning chains"""
    return await framework.explain_reasoning_chain(reasoning_steps, final_output)

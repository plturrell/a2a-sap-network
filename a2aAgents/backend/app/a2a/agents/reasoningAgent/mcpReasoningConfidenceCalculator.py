"""
MCP-enabled Reasoning Confidence Calculator
Exposes confidence calculation methods as MCP tools for cross-agent usage
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...sdk.mcpSkillCoordination import skill_provides, skill_depends_on

class ConfidenceFactors(Enum):
    """Factors that influence confidence calculations"""
    EVIDENCE_QUALITY = "evidence_quality"
    LOGICAL_CONSISTENCY = "logical_consistency"
    SEMANTIC_ALIGNMENT = "semantic_alignment"
    HISTORICAL_SUCCESS = "historical_success"
    COMPLEXITY_PENALTY = "complexity_penalty"
    VALIDATION_SCORE = "validation_score"

@dataclass
class ConfidenceMetrics:
    """Metrics for confidence calculation"""
    base_confidence: float
    evidence_factor: float
    consistency_factor: float
    complexity_factor: float
    historical_factor: float
    final_confidence: float
    factor_breakdown: Dict[str, float]
    explanation: str

class MCPReasoningConfidenceCalculator:
    """MCP-enabled confidence calculator for reasoning with exposed tools"""

    def __init__(self):
        self.factor_weights = {
            ConfidenceFactors.EVIDENCE_QUALITY: 0.25,
            ConfidenceFactors.LOGICAL_CONSISTENCY: 0.20,
            ConfidenceFactors.SEMANTIC_ALIGNMENT: 0.20,
            ConfidenceFactors.HISTORICAL_SUCCESS: 0.15,
            ConfidenceFactors.COMPLEXITY_PENALTY: 0.10,
            ConfidenceFactors.VALIDATION_SCORE: 0.10
        }

        # Confidence bounds
        self.min_confidence = 0.05  # Never return 0
        self.max_confidence = 0.95  # Never claim 100% certainty

    @mcp_tool(
        name="calculate_reasoning_confidence",
        description="Calculate confidence score for reasoning with detailed breakdown",
        input_schema={
            "type": "object",
            "properties": {
                "reasoning_context": {
                    "type": "object",
                    "description": "Context containing evidence, reasoning chain, question/answer",
                    "properties": {
                        "evidence": {"type": "array"},
                        "reasoning_chain": {"type": "array"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "historical_data": {"type": "object"},
                        "validation_results": {"type": "object"}
                    }
                },
                "custom_weights": {
                    "type": "object",
                    "description": "Custom factor weights (optional)",
                    "additionalProperties": {"type": "number"}
                },
                "include_explanation": {
                    "type": "boolean",
                    "description": "Include detailed explanation",
                    "default": True
                }
            },
            "required": ["reasoning_context"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "factor_breakdown": {"type": "object"},
                "explanation": {"type": "string"},
                "recommendations": {"type": "array"}
            }
        }
    )
    @skill_provides("confidence_calculation", "reasoning_analysis")
    async def calculate_reasoning_confidence_mcp(self,
                                           reasoning_context: Dict[str, Any],
                                           custom_weights: Optional[Dict[str, float]] = None,
                                           include_explanation: bool = True) -> Dict[str, Any]:
        """MCP-exposed confidence calculation with configurable weights"""

        # Use custom weights if provided
        weights = custom_weights or self.factor_weights

        factors = {}
        explanations = []

        # Evidence quality factor
        evidence_quality = self._calculate_evidence_quality(
            reasoning_context.get("evidence", [])
        )
        factors[ConfidenceFactors.EVIDENCE_QUALITY] = evidence_quality
        explanations.append(f"Evidence quality: {evidence_quality:.2f}")

        # Logical consistency factor
        logical_consistency = self._calculate_logical_consistency(
            reasoning_context.get("reasoning_chain", [])
        )
        factors[ConfidenceFactors.LOGICAL_CONSISTENCY] = logical_consistency
        explanations.append(f"Logical consistency: {logical_consistency:.2f}")

        # Semantic alignment factor
        semantic_alignment = self._calculate_semantic_alignment(
            reasoning_context.get("question", ""),
            reasoning_context.get("answer", "")
        )
        factors[ConfidenceFactors.SEMANTIC_ALIGNMENT] = semantic_alignment
        explanations.append(f"Semantic alignment: {semantic_alignment:.2f}")

        # Historical success factor
        historical_success = self._calculate_historical_success(
            reasoning_context.get("historical_data", {})
        )
        factors[ConfidenceFactors.HISTORICAL_SUCCESS] = historical_success
        explanations.append(f"Historical success: {historical_success:.2f}")

        # Complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(
            reasoning_context
        )
        factors[ConfidenceFactors.COMPLEXITY_PENALTY] = complexity_penalty
        explanations.append(f"Complexity score: {complexity_penalty:.2f}")

        # Validation score factor
        validation_score = self._calculate_validation_score(
            reasoning_context.get("validation_results", {})
        )
        factors[ConfidenceFactors.VALIDATION_SCORE] = validation_score
        explanations.append(f"Validation score: {validation_score:.2f}")

        # Calculate weighted confidence
        weighted_confidence = 0.0
        factor_breakdown = {}

        for factor, value in factors.items():
            weight = weights.get(factor, self.factor_weights.get(factor, 0.1))
            contribution = value * weight
            weighted_confidence += contribution
            factor_breakdown[factor.value] = {
                "value": value,
                "weight": weight,
                "contribution": contribution
            }

        # Apply bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, weighted_confidence))

        # Generate recommendations
        recommendations = self._generate_recommendations(factors, final_confidence)

        result = {
            "confidence": final_confidence,
            "factor_breakdown": factor_breakdown,
            "recommendations": recommendations
        }

        if include_explanation:
            result["explanation"] = " | ".join(explanations) + f" | Final confidence: {final_confidence:.2f}"

        return result

    @mcp_tool(
        name="calculate_evidence_quality",
        description="Calculate quality score for provided evidence",
        input_schema={
            "type": "object",
            "properties": {
                "evidence": {
                    "type": "array",
                    "description": "List of evidence items with source_type"
                },
                "return_details": {
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["evidence"]
        }
    )
    @skill_provides("evidence_analysis")
    async def calculate_evidence_quality_mcp(self,
                                       evidence: List[Any],
                                       return_details: bool = False) -> Dict[str, Any]:
        """MCP tool for evidence quality calculation"""
        quality_score = self._calculate_evidence_quality(evidence)

        result = {"quality_score": quality_score}

        if return_details:
            # Analyze evidence composition
            evidence_types = {}
            for item in evidence:
                if isinstance(item, dict):
                    source_type = item.get("source_type", "unknown")
                    evidence_types[source_type] = evidence_types.get(source_type, 0) + 1

            result["evidence_composition"] = evidence_types
            result["evidence_count"] = len(evidence)
            result["has_academic_sources"] = "academic" in evidence_types
            result["has_verified_sources"] = "verified" in evidence_types

        return result

    @mcp_tool(
        name="calculate_semantic_alignment",
        description="Calculate semantic alignment between question and answer",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "analyze_keywords": {"type": "boolean", "default": False}
            },
            "required": ["question", "answer"]
        }
    )
    @skill_provides("semantic_analysis")
    async def calculate_semantic_alignment_mcp(self,
                                         question: str,
                                         answer: str,
                                         analyze_keywords: bool = False) -> Dict[str, Any]:
        """MCP tool for semantic alignment calculation"""
        alignment_score = self._calculate_semantic_alignment(question, answer)

        result = {"alignment_score": alignment_score}

        if analyze_keywords:
            # Provide keyword analysis
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}

            question_words -= common_words
            answer_words -= common_words

            result["keyword_overlap"] = list(question_words.intersection(answer_words))
            result["unique_question_keywords"] = list(question_words - answer_words)
            result["unique_answer_keywords"] = list(answer_words - question_words)
            result["question_type"] = self._identify_question_type(question)

        return result

    @mcp_tool(
        name="adjust_confidence_for_uncertainty",
        description="Adjust confidence score based on uncertainty factors",
        input_schema={
            "type": "object",
            "properties": {
                "base_confidence": {"type": "number"},
                "uncertainty_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of uncertainty factors"
                }
            },
            "required": ["base_confidence", "uncertainty_factors"]
        }
    )
    async def adjust_confidence_for_uncertainty_mcp(self,
                                              base_confidence: float,
                                              uncertainty_factors: List[str]) -> Dict[str, Any]:
        """MCP tool for uncertainty adjustment"""
        adjusted_confidence = self.adjust_confidence_for_uncertainty(
            base_confidence, uncertainty_factors
        )

        # Calculate total penalty
        penalty = 1.0
        uncertainty_penalties = {
            "missing_evidence": 0.85,
            "contradictory_evidence": 0.7,
            "weak_inference": 0.8,
            "unvalidated_source": 0.9,
            "complex_reasoning": 0.9,
            "limited_data": 0.85
        }

        factor_impacts = {}
        for factor in uncertainty_factors:
            factor_penalty = uncertainty_penalties.get(factor, 0.95)
            penalty *= factor_penalty
            factor_impacts[factor] = 1.0 - factor_penalty

        return {
            "original_confidence": base_confidence,
            "adjusted_confidence": adjusted_confidence,
            "total_penalty": 1.0 - penalty,
            "factor_impacts": factor_impacts
        }

    @mcp_resource(
        uri="confidence://calculator/status",
        name="Confidence Calculator Status",
        description="Current status and configuration of confidence calculator",
        mime_type="application/json"
    )
    async def get_calculator_status(self) -> Dict[str, Any]:
        """Get current calculator configuration and status"""
        return {
            "factor_weights": {k.value: v for k, v in self.factor_weights.items()},
            "confidence_bounds": {
                "min": self.min_confidence,
                "max": self.max_confidence
            },
            "available_factors": [f.value for f in ConfidenceFactors],
            "uncertainty_factors": [
                "missing_evidence",
                "contradictory_evidence",
                "weak_inference",
                "unvalidated_source",
                "complex_reasoning",
                "limited_data"
            ]
        }

    @mcp_prompt(
        name="confidence_analysis",
        description="Interactive confidence analysis assistant",
        arguments=[
            {
                "name": "context",
                "description": "Reasoning context to analyze",
                "required": True
            },
            {
                "name": "focus_area",
                "description": "Specific area to focus on (evidence, logic, alignment)",
                "required": False
            }
        ]
    )
    async def confidence_analysis_prompt(self, context: Dict[str, Any],
                                   focus_area: Optional[str] = None) -> str:
        """Interactive prompt for confidence analysis"""
        analysis = await self.calculate_reasoning_confidence_mcp(
            context, include_explanation=True
        )

        prompt = f"""Based on the confidence analysis:

Overall Confidence: {analysis['confidence']:.2%}

Factor Breakdown:
"""
        for factor, details in analysis['factor_breakdown'].items():
            prompt += f"- {factor}: {details['value']:.2f} (weight: {details['weight']:.2f})\n"

        if focus_area:
            prompt += f"\nFocusing on {focus_area}:\n"
            if focus_area == "evidence":
                evidence_analysis = await self.calculate_evidence_quality_mcp(
                    context.get("evidence", []), return_details=True
                )
                prompt += f"Evidence composition: {evidence_analysis['evidence_composition']}\n"
            elif focus_area == "alignment":
                alignment_analysis = await self.calculate_semantic_alignment_mcp(
                    context.get("question", ""),
                    context.get("answer", ""),
                    analyze_keywords=True
                )
                prompt += f"Keyword overlap: {alignment_analysis['keyword_overlap']}\n"

        prompt += f"\nRecommendations:\n"
        for rec in analysis['recommendations']:
            prompt += f"- {rec}\n"

        return prompt

    # Keep existing calculation methods as internal helpers
    def _calculate_evidence_quality(self, evidence: List[Any]) -> float:
        """Calculate evidence quality factor"""
        if not evidence:
            return 0.2

        quality_score = 0.0

        for item in evidence:
            if isinstance(item, dict):
                if item.get("source_type") == "academic":
                    quality_score += 0.9
                elif item.get("source_type") == "verified":
                    quality_score += 0.8
                elif item.get("source_type") == "empirical":
                    quality_score += 0.7
                else:
                    quality_score += 0.4
            else:
                quality_score += 0.3

        normalized_score = quality_score / len(evidence)

        if len(evidence) > 3:
            normalized_score = min(1.0, normalized_score * 1.2)

        return normalized_score

    def _calculate_logical_consistency(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """Calculate logical consistency factor"""
        if not reasoning_chain:
            return 0.3

        consistency_score = 1.0

        for i, step in enumerate(reasoning_chain):
            if i > 0:
                prev_step = reasoning_chain[i-1]

                if step.get("confidence", 0) < prev_step.get("confidence", 0) * 0.5:
                    consistency_score *= 0.8

                if step.get("conclusion") == prev_step.get("premise"):
                    consistency_score *= 0.6

        inference_rules_used = [step.get("inference_rule") for step in reasoning_chain
                               if step.get("inference_rule")]

        if inference_rules_used:
            valid_rules = ["modus_ponens", "modus_tollens", "hypothetical_syllogism"]
            valid_rule_ratio = sum(1 for rule in inference_rules_used if rule in valid_rules) / len(inference_rules_used)
            consistency_score *= (0.8 + 0.2 * valid_rule_ratio)

        return max(0.1, consistency_score)

    def _calculate_semantic_alignment(self, question: str, answer: str) -> float:
        """Calculate semantic alignment between question and answer"""
        if not question or not answer:
            return 0.3

        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        question_words -= common_words
        answer_words -= common_words

        if not question_words:
            return 0.5

        overlap = len(question_words.intersection(answer_words))
        alignment_score = overlap / len(question_words)

        key_words = {"what", "how", "why", "when", "where", "who"}
        question_type_words = question_words.intersection(key_words)

        if question_type_words:
            if "what" in question_type_words and any(word in answer_words for word in ["is", "are", "means"]):
                alignment_score *= 1.2
            elif "how" in question_type_words and any(word in answer_words for word in ["by", "through", "using"]):
                alignment_score *= 1.2
            elif "why" in question_type_words and any(word in answer_words for word in ["because", "reason", "cause"]):
                alignment_score *= 1.3

        return min(1.0, alignment_score)

    def _calculate_historical_success(self, historical_data: Dict[str, Any]) -> float:
        """Calculate historical success factor"""
        if not historical_data:
            return 0.5

        total_attempts = historical_data.get("total_attempts", 0)
        successful_attempts = historical_data.get("successful_attempts", 0)

        if total_attempts == 0:
            return 0.5

        success_rate = successful_attempts / total_attempts

        if total_attempts < 5:
            confidence_multiplier = 0.6 + (total_attempts * 0.08)
        elif total_attempts < 20:
            confidence_multiplier = 0.9
        else:
            confidence_multiplier = 1.0

        recent_results = historical_data.get("recent_results", [])
        if len(recent_results) >= 3:
            recent_success_rate = sum(recent_results[-3:]) / 3
            success_rate = (success_rate * 0.7) + (recent_success_rate * 0.3)

        return success_rate * confidence_multiplier

    def _calculate_complexity_penalty(self, reasoning_context: Dict[str, Any]) -> float:
        """Calculate complexity penalty (inverted - higher is better)"""
        penalty_score = 1.0

        chain_length = len(reasoning_context.get("reasoning_chain", []))
        if chain_length > 10:
            penalty_score *= 0.9
        elif chain_length > 15:
            penalty_score *= 0.7
        elif chain_length < 2:
            penalty_score *= 0.8

        branch_factor = reasoning_context.get("branch_factor", 1)
        if branch_factor > 5:
            penalty_score *= 0.8

        circular_deps = reasoning_context.get("circular_dependencies", 0)
        penalty_score *= (1.0 - (circular_deps * 0.1))

        return max(0.3, penalty_score)

    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate validation score factor"""
        if not validation_results:
            return 0.5

        logical_consistency = validation_results.get("logical_consistency", 0.5)
        evidence_support = validation_results.get("evidence_support", 0.5)
        completeness = validation_results.get("completeness", 0.5)

        validation_score = (
            logical_consistency * 0.4 +
            evidence_support * 0.3 +
            completeness * 0.3
        )

        validator_confidence = validation_results.get("validator_confidence", 0.7)

        return validation_score * validator_confidence

    def _generate_recommendations(self, factors: Dict[ConfidenceFactors, float],
                                confidence: float) -> List[str]:
        """Generate recommendations based on factor analysis"""
        recommendations = []

        # Check each factor for improvement opportunities
        if factors[ConfidenceFactors.EVIDENCE_QUALITY] < 0.5:
            recommendations.append("Gather more high-quality evidence from academic or verified sources")

        if factors[ConfidenceFactors.LOGICAL_CONSISTENCY] < 0.6:
            recommendations.append("Strengthen logical connections between reasoning steps")

        if factors[ConfidenceFactors.SEMANTIC_ALIGNMENT] < 0.5:
            recommendations.append("Ensure answer directly addresses the question")

        if factors[ConfidenceFactors.COMPLEXITY_PENALTY] < 0.7:
            recommendations.append("Simplify reasoning chain to reduce complexity")

        if confidence < 0.4:
            recommendations.append("Consider alternative reasoning approaches")

        return recommendations

    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question"""
        question_lower = question.lower()

        if question_lower.startswith("what"):
            return "definition"
        elif question_lower.startswith("how"):
            return "process"
        elif question_lower.startswith("why"):
            return "explanation"
        elif question_lower.startswith("when"):
            return "temporal"
        elif question_lower.startswith("where"):
            return "location"
        elif question_lower.startswith("who"):
            return "identity"
        else:
            return "other"

    def calculate_fallback_confidence(self, scenario: str) -> float:
        """Calculate appropriate fallback confidence for different scenarios"""
        fallback_scores = {
            "no_historical_data": 0.3,
            "single_agent_fallback": 0.35,
            "internal_analysis": 0.4,
            "simplified_implementation": 0.45,
            "partial_evidence": 0.5,
            "consensus_fallback": 0.55,
            "validated_fallback": 0.6
        }

        return fallback_scores.get(scenario, 0.4)

    def adjust_confidence_for_uncertainty(self, base_confidence: float,
                                        uncertainty_factors: List[str]) -> float:
        """Adjust confidence based on uncertainty factors"""
        if not uncertainty_factors:
            return base_confidence

        uncertainty_penalties = {
            "missing_evidence": 0.85,
            "contradictory_evidence": 0.7,
            "weak_inference": 0.8,
            "unvalidated_source": 0.9,
            "complex_reasoning": 0.9,
            "limited_data": 0.85
        }

        adjusted_confidence = base_confidence

        for factor in uncertainty_factors:
            penalty = uncertainty_penalties.get(factor, 0.95)
            adjusted_confidence *= penalty

        return max(self.min_confidence, adjusted_confidence)


# Singleton instance
mcp_confidence_calculator = MCPReasoningConfidenceCalculator()
"""
Reasoning Confidence Calculator
Dynamic confidence calculation replacing hardcoded values
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

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

class DynamicConfidenceCalculator:
    """Calculate confidence scores dynamically based on multiple factors"""

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

    def calculate_reasoning_confidence(self,
                                     reasoning_context: Dict[str, Any]) -> float:
        """Calculate dynamic confidence score for reasoning"""

        factors = {}

        # Evidence quality factor
        factors[ConfidenceFactors.EVIDENCE_QUALITY] = self._calculate_evidence_quality(
            reasoning_context.get("evidence", [])
        )

        # Logical consistency factor
        factors[ConfidenceFactors.LOGICAL_CONSISTENCY] = self._calculate_logical_consistency(
            reasoning_context.get("reasoning_chain", [])
        )

        # Semantic alignment factor
        factors[ConfidenceFactors.SEMANTIC_ALIGNMENT] = self._calculate_semantic_alignment(
            reasoning_context.get("question", ""),
            reasoning_context.get("answer", "")
        )

        # Historical success factor
        factors[ConfidenceFactors.HISTORICAL_SUCCESS] = self._calculate_historical_success(
            reasoning_context.get("historical_data", {})
        )

        # Complexity penalty
        factors[ConfidenceFactors.COMPLEXITY_PENALTY] = self._calculate_complexity_penalty(
            reasoning_context
        )

        # Validation score factor
        factors[ConfidenceFactors.VALIDATION_SCORE] = self._calculate_validation_score(
            reasoning_context.get("validation_results", {})
        )

        # Calculate weighted confidence
        weighted_confidence = 0.0
        for factor, value in factors.items():
            weight = self.factor_weights.get(factor, 0.1)
            weighted_confidence += value * weight

        # Apply bounds
        final_confidence = max(self.min_confidence, min(self.max_confidence, weighted_confidence))

        return final_confidence

    def _calculate_evidence_quality(self, evidence: List[Any]) -> float:
        """Calculate evidence quality factor"""

        if not evidence:
            return 0.2  # Low confidence without evidence

        quality_score = 0.0

        for item in evidence:
            if isinstance(item, dict):
                # Check for high-quality evidence markers
                if item.get("source_type") == "academic":
                    quality_score += 0.9
                elif item.get("source_type") == "verified":
                    quality_score += 0.8
                elif item.get("source_type") == "empirical":
                    quality_score += 0.7
                else:
                    quality_score += 0.4
            else:
                # Basic evidence
                quality_score += 0.3

        # Normalize by evidence count
        normalized_score = quality_score / len(evidence)

        # Bonus for multiple corroborating evidence
        if len(evidence) > 3:
            normalized_score = min(1.0, normalized_score * 1.2)

        return normalized_score

    def _calculate_logical_consistency(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """Calculate logical consistency factor"""

        if not reasoning_chain:
            return 0.3  # Low confidence without reasoning

        consistency_score = 1.0

        # Check for logical flaws
        for i, step in enumerate(reasoning_chain):
            if i > 0:
                prev_step = reasoning_chain[i-1]

                # Check connection strength
                if step.get("confidence", 0) < prev_step.get("confidence", 0) * 0.5:
                    consistency_score *= 0.8  # Penalty for weak connections

                # Check for circular reasoning
                if step.get("conclusion") == prev_step.get("premise"):
                    consistency_score *= 0.6  # Penalty for circularity

        # Bonus for clear inference rules
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

        # Simple word overlap metric
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        question_words -= common_words
        answer_words -= common_words

        if not question_words:
            return 0.5

        # Calculate alignment
        overlap = len(question_words.intersection(answer_words))
        alignment_score = overlap / len(question_words)

        # Boost for key question words appearing in answer
        key_words = {"what", "how", "why", "when", "where", "who"}
        question_type_words = question_words.intersection(key_words)

        if question_type_words:
            # Check if answer addresses the question type
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
            return 0.5  # Neutral without history

        # Extract success metrics
        total_attempts = historical_data.get("total_attempts", 0)
        successful_attempts = historical_data.get("successful_attempts", 0)

        if total_attempts == 0:
            return 0.5

        # Basic success rate
        success_rate = successful_attempts / total_attempts

        # Adjust based on sample size
        if total_attempts < 5:
            # Low confidence in small samples
            confidence_multiplier = 0.6 + (total_attempts * 0.08)
        elif total_attempts < 20:
            # Medium confidence
            confidence_multiplier = 0.9
        else:
            # High confidence in large samples
            confidence_multiplier = 1.0

        # Consider recent trend
        recent_results = historical_data.get("recent_results", [])
        if len(recent_results) >= 3:
            recent_success_rate = sum(recent_results[-3:]) / 3
            # Weight recent performance more heavily
            success_rate = (success_rate * 0.7) + (recent_success_rate * 0.3)

        return success_rate * confidence_multiplier

    def _calculate_complexity_penalty(self, reasoning_context: Dict[str, Any]) -> float:
        """Calculate complexity penalty (inverted - higher is better)"""

        penalty_score = 1.0

        # Penalize overly complex reasoning chains
        chain_length = len(reasoning_context.get("reasoning_chain", []))
        if chain_length > 10:
            penalty_score *= 0.9
        elif chain_length > 15:
            penalty_score *= 0.7
        elif chain_length < 2:
            penalty_score *= 0.8  # Too simple might be insufficient

        # Penalize excessive branching
        branch_factor = reasoning_context.get("branch_factor", 1)
        if branch_factor > 5:
            penalty_score *= 0.8

        # Penalize circular dependencies
        circular_deps = reasoning_context.get("circular_dependencies", 0)
        penalty_score *= (1.0 - (circular_deps * 0.1))

        return max(0.3, penalty_score)

    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate validation score factor"""

        if not validation_results:
            return 0.5  # Neutral without validation

        # Extract validation metrics
        logical_consistency = validation_results.get("logical_consistency", 0.5)
        evidence_support = validation_results.get("evidence_support", 0.5)
        completeness = validation_results.get("completeness", 0.5)

        # Weighted average of validation metrics
        validation_score = (
            logical_consistency * 0.4 +
            evidence_support * 0.3 +
            completeness * 0.3
        )

        # Apply validation confidence
        validator_confidence = validation_results.get("validator_confidence", 0.7)

        return validation_score * validator_confidence

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

    def calculate_pattern_confidence(self, pattern_metrics: Dict[str, Any]) -> float:
        """Calculate confidence for pattern analysis"""

        pattern_count = pattern_metrics.get("pattern_count", 0)
        pattern_diversity = pattern_metrics.get("pattern_diversity", 0)
        semantic_coherence = pattern_metrics.get("semantic_coherence", 0.5)

        if pattern_count == 0:
            return 0.2

        # Base score on pattern quality
        base_score = min(pattern_count / 10, 1.0) * 0.5

        # Diversity bonus
        diversity_bonus = min(pattern_diversity / 5, 1.0) * 0.3

        # Coherence factor
        coherence_factor = semantic_coherence * 0.2

        return base_score + diversity_bonus + coherence_factor

    def calculate_similarity_confidence(self, similarity_score: float,
                                      sample_size: int) -> float:
        """Calculate confidence for similarity-based operations"""

        if sample_size == 0:
            return 0.1

        # Base confidence on similarity
        base_confidence = similarity_score

        # Adjust for sample size
        if sample_size < 3:
            size_multiplier = 0.6
        elif sample_size < 10:
            size_multiplier = 0.8
        else:
            size_multiplier = 0.95

        # Non-linear scaling for very high similarity
        if similarity_score > 0.9:
            base_confidence = 0.9 + (similarity_score - 0.9) * 0.5

        return base_confidence * size_multiplier

    def adjust_confidence_for_uncertainty(self, base_confidence: float,
                                        uncertainty_factors: List[str]) -> float:
        """Adjust confidence based on uncertainty factors"""

        if not uncertainty_factors:
            return base_confidence

        # Define uncertainty penalties
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
confidence_calculator = DynamicConfidenceCalculator()

# Helper functions for easy access
def calculate_reasoning_confidence(context: Dict[str, Any]) -> float:
    """Calculate dynamic reasoning confidence"""
    return confidence_calculator.calculate_reasoning_confidence(context)

def calculate_fallback_confidence(scenario: str) -> float:
    """Get appropriate fallback confidence"""
    return confidence_calculator.calculate_fallback_confidence(scenario)

def calculate_pattern_confidence(metrics: Dict[str, Any]) -> float:
    """Calculate pattern analysis confidence"""
    return confidence_calculator.calculate_pattern_confidence(metrics)

def calculate_similarity_confidence(similarity: float, samples: int) -> float:
    """Calculate similarity-based confidence"""
    return confidence_calculator.calculate_similarity_confidence(similarity, samples)

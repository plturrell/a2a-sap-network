"""
Reasoning Validation Framework
Implements comprehensive validation for reasoning processes and outputs
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    COMPREHENSIVE = "comprehensive"

class ValidationCriteria(Enum):
    LOGICAL_CONSISTENCY = "logical_consistency"
    EVIDENCE_SUPPORT = "evidence_support"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    CONFIDENCE_CALIBRATION = "confidence_calibration"

@dataclass
class ValidationResult:
    criterion: ValidationCriteria
    score: float
    confidence: float
    issues_found: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningValidationReport:
    overall_score: float
    validation_level: ValidationLevel
    criteria_results: List[ValidationResult]
    critical_issues: List[str]
    improvement_recommendations: List[str]
    validated_at: datetime
    validator_confidence: float

class LogicalConsistencyValidator:
    """Validates logical consistency of reasoning chains"""
    
    def __init__(self):
        self.consistency_rules = {
            "no_contradictions": self._check_contradictions,
            "premise_conclusion_alignment": self._check_premise_conclusion_alignment,
            "inference_validity": self._check_inference_validity,
            "temporal_consistency": self._check_temporal_consistency
        }
    
    async def validate(self, reasoning_chain: List[Dict[str, Any]], 
                      question: str) -> ValidationResult:
        """Validate logical consistency"""
        
        issues = []
        suggestions = []
        evidence = {}
        scores = []
        
        # Run all consistency checks
        for rule_name, rule_func in self.consistency_rules.items():
            try:
                rule_result = await rule_func(reasoning_chain, question)
                scores.append(rule_result["score"])
                
                if rule_result["issues"]:
                    issues.extend(rule_result["issues"])
                if rule_result["suggestions"]:
                    suggestions.extend(rule_result["suggestions"])
                
                evidence[rule_name] = rule_result["evidence"]
                
            except Exception as e:
                logger.error(f"Consistency rule {rule_name} failed: {e}")
                scores.append(0.5)  # Neutral score on failure
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return ValidationResult(
            criterion=ValidationCriteria.LOGICAL_CONSISTENCY,
            score=overall_score,
            confidence=0.8,
            issues_found=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    async def _check_contradictions(self, reasoning_chain: List[Dict[str, Any]], 
                                  question: str) -> Dict[str, Any]:
        """Check for logical contradictions"""
        
        issues = []
        evidence = {"contradictions_found": []}
        
        # Extract statements from reasoning chain
        statements = []
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                statements.append(inference.premise)
                statements.append(inference.conclusion)
        
        # Look for contradictions
        contradictions = []
        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i+1:]:
                if self._are_contradictory(stmt1, stmt2):
                    contradictions.append((stmt1, stmt2))
                    issues.append(f"Contradiction found: '{stmt1}' vs '{stmt2}'")
        
        evidence["contradictions_found"] = contradictions
        score = max(0.0, 1.0 - (len(contradictions) * 0.3))
        
        suggestions = []
        if contradictions:
            suggestions.append("Resolve contradictory statements by clarifying context or correcting logic")
            suggestions.append("Review premise-conclusion relationships for consistency")
        
        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "evidence": evidence
        }
    
    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Simple contradiction detection"""
        
        # Look for explicit negations
        if "not " in statement1.lower() and "not " not in statement2.lower():
            base1 = statement1.lower().replace("not ", "").strip()
            if base1 in statement2.lower():
                return True
        
        if "not " in statement2.lower() and "not " not in statement1.lower():
            base2 = statement2.lower().replace("not ", "").strip()
            if base2 in statement1.lower():
                return True
        
        # Look for opposite terms
        opposites = [
            ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
            ("positive", "negative"), ("true", "false"), ("yes", "no")
        ]
        
        stmt1_lower = statement1.lower()
        stmt2_lower = statement2.lower()
        
        for term1, term2 in opposites:
            if term1 in stmt1_lower and term2 in stmt2_lower:
                return True
            if term2 in stmt1_lower and term1 in stmt2_lower:
                return True
        
        return False
    
    async def _check_premise_conclusion_alignment(self, reasoning_chain: List[Dict[str, Any]], 
                                                question: str) -> Dict[str, Any]:
        """Check if conclusions follow from premises"""
        
        issues = []
        evidence = {"alignment_scores": []}
        alignment_scores = []
        
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                premise = inference.premise
                conclusion = inference.conclusion
                
                # Simple alignment check
                alignment_score = self._calculate_alignment_score(premise, conclusion)
                alignment_scores.append(alignment_score)
                evidence["alignment_scores"].append({
                    "premise": premise,
                    "conclusion": conclusion,
                    "alignment_score": alignment_score
                })
                
                if alignment_score < 0.5:
                    issues.append(f"Weak premise-conclusion alignment: '{premise}' → '{conclusion}'")
        
        overall_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 1.0
        
        suggestions = []
        if overall_score < 0.7:
            suggestions.append("Strengthen logical connections between premises and conclusions")
            suggestions.append("Provide additional justification for inferential steps")
        
        return {
            "score": overall_score,
            "issues": issues,
            "suggestions": suggestions,
            "evidence": evidence
        }
    
    def _calculate_alignment_score(self, premise: str, conclusion: str) -> float:
        """Calculate how well conclusion aligns with premise"""
        
        # Simple word overlap metric
        premise_words = set(premise.lower().split())
        conclusion_words = set(conclusion.lower().split())
        
        if not premise_words or not conclusion_words:
            return 0.5
        
        overlap = len(premise_words.intersection(conclusion_words))
        total_unique = len(premise_words.union(conclusion_words))
        
        # Base score on overlap
        overlap_score = overlap / total_unique if total_unique > 0 else 0.0
        
        # Boost for logical connectors
        logical_connectors = ["therefore", "thus", "hence", "consequently", "so", "because"]
        if any(connector in conclusion.lower() for connector in logical_connectors):
            overlap_score += 0.2
        
        return min(overlap_score, 1.0)


class EvidenceSupportValidator:
    """Validates evidence support for reasoning"""
    
    async def validate(self, reasoning_chain: List[Dict[str, Any]], 
                      context: Dict[str, Any]) -> ValidationResult:
        """Validate evidence support"""
        
        issues = []
        suggestions = []
        evidence = {}
        
        # Check evidence quality
        evidence_quality = await self._assess_evidence_quality(reasoning_chain, context)
        
        # Check evidence coverage
        coverage_score = await self._assess_evidence_coverage(reasoning_chain, context)
        
        # Check source reliability
        reliability_score = await self._assess_source_reliability(reasoning_chain, context)
        
        overall_score = (evidence_quality * 0.4 + coverage_score * 0.3 + reliability_score * 0.3)
        
        if evidence_quality < 0.6:
            issues.append("Insufficient evidence quality to support conclusions")
            suggestions.append("Provide more specific and detailed evidence")
        
        if coverage_score < 0.5:
            issues.append("Evidence does not adequately cover all reasoning steps")
            suggestions.append("Ensure each reasoning step is supported by relevant evidence")
        
        evidence = {
            "evidence_quality": evidence_quality,
            "coverage_score": coverage_score,
            "reliability_score": reliability_score
        }
        
        return ValidationResult(
            criterion=ValidationCriteria.EVIDENCE_SUPPORT,
            score=overall_score,
            confidence=0.7,
            issues_found=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    async def _assess_evidence_quality(self, reasoning_chain: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> float:
        """Assess quality of supporting evidence"""
        
        quality_scores = []
        
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                supporting_evidence = inference.supporting_evidence
                
                if not supporting_evidence:
                    quality_scores.append(0.2)  # Low score for no evidence
                else:
                    # Score based on evidence specificity and relevance
                    evidence_score = 0.0
                    for evidence_item in supporting_evidence:
                        if len(evidence_item) > 20:  # Detailed evidence
                            evidence_score += 0.3
                        if any(word in evidence_item.lower() for word in ["data", "study", "research", "analysis"]):
                            evidence_score += 0.2
                        if "http" in evidence_item or "www" in evidence_item:  # Has sources
                            evidence_score += 0.2
                    
                    quality_scores.append(min(evidence_score, 1.0))
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    
    async def _assess_evidence_coverage(self, reasoning_chain: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> float:
        """Assess how well evidence covers the reasoning"""
        
        total_steps = len(reasoning_chain)
        supported_steps = 0
        
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                if inference.supporting_evidence:
                    supported_steps += 1
        
        return supported_steps / total_steps if total_steps > 0 else 1.0
    
    async def _assess_source_reliability(self, reasoning_chain: List[Dict[str, Any]], 
                                       context: Dict[str, Any]) -> float:
        """Assess reliability of evidence sources"""
        
        # Simple heuristic-based assessment
        reliability_indicators = [
            "research", "study", "analysis", "peer-reviewed", "academic",
            "official", "government", "verified", "documented"
        ]
        
        source_scores = []
        
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                for evidence_item in inference.supporting_evidence:
                    reliability_score = 0.5  # Base score
                    
                    for indicator in reliability_indicators:
                        if indicator in evidence_item.lower():
                            reliability_score += 0.1
                    
                    source_scores.append(min(reliability_score, 1.0))
        
        return sum(source_scores) / len(source_scores) if source_scores else 0.5


class CompletenessValidator:
    """Validates completeness of reasoning"""
    
    async def validate(self, reasoning_chain: List[Dict[str, Any]], 
                      question: str, patterns: Dict[str, Any]) -> ValidationResult:
        """Validate reasoning completeness"""
        
        issues = []
        suggestions = []
        evidence = {}
        
        # Check question coverage
        question_coverage = await self._assess_question_coverage(reasoning_chain, question)
        
        # Check reasoning depth
        depth_score = await self._assess_reasoning_depth(reasoning_chain)
        
        # Check pattern completeness
        pattern_completeness = await self._assess_pattern_completeness(reasoning_chain, patterns)
        
        overall_score = (question_coverage * 0.4 + depth_score * 0.3 + pattern_completeness * 0.3)
        
        if question_coverage < 0.6:
            issues.append("Reasoning does not fully address the original question")
            suggestions.append("Ensure all aspects of the question are covered")
        
        if depth_score < 0.5:
            issues.append("Reasoning lacks sufficient depth")
            suggestions.append("Provide deeper analysis and more detailed reasoning steps")
        
        evidence = {
            "question_coverage": question_coverage,
            "depth_score": depth_score,
            "pattern_completeness": pattern_completeness
        }
        
        return ValidationResult(
            criterion=ValidationCriteria.COMPLETENESS,
            score=overall_score,
            confidence=0.75,
            issues_found=issues,
            suggestions=suggestions,
            evidence=evidence
        )
    
    async def _assess_question_coverage(self, reasoning_chain: List[Dict[str, Any]], 
                                      question: str) -> float:
        """Assess how well reasoning covers the question"""
        
        question_words = set(question.lower().split())
        reasoning_words = set()
        
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                reasoning_words.update(inference.premise.lower().split())
                reasoning_words.update(inference.conclusion.lower().split())
        
        if not question_words:
            return 1.0
        
        coverage = len(question_words.intersection(reasoning_words)) / len(question_words)
        return min(coverage * 1.5, 1.0)  # Boost coverage score
    
    async def _assess_reasoning_depth(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """Assess depth of reasoning"""
        
        depth_indicators = {
            "chain_length": len(reasoning_chain),
            "inference_complexity": 0,
            "branching_factor": 0
        }
        
        # Assess inference complexity
        for step in reasoning_chain:
            if "inference" in step:
                inference = step["inference"]
                complexity = len(inference.premise.split()) + len(inference.conclusion.split())
                depth_indicators["inference_complexity"] += complexity
        
        # Normalize scores
        length_score = min(depth_indicators["chain_length"] / 5.0, 1.0)
        complexity_score = min(depth_indicators["inference_complexity"] / 100.0, 1.0)
        
        return (length_score * 0.6 + complexity_score * 0.4)


class ReasoningValidationFramework:
    """Main validation framework orchestrator"""
    
    def __init__(self):
        self.validators = {
            ValidationCriteria.LOGICAL_CONSISTENCY: LogicalConsistencyValidator(),
            ValidationCriteria.EVIDENCE_SUPPORT: EvidenceSupportValidator(),
            ValidationCriteria.COMPLETENESS: CompletenessValidator()
        }
        
        self.validation_thresholds = {
            ValidationLevel.BASIC: 0.5,
            ValidationLevel.INTERMEDIATE: 0.7,
            ValidationLevel.COMPREHENSIVE: 0.8
        }
    
    async def validate_reasoning(self, reasoning_chain: List[Dict[str, Any]], 
                               question: str, context: Dict[str, Any] = None,
                               patterns: Dict[str, Any] = None,
                               validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> ReasoningValidationReport:
        """Comprehensive reasoning validation"""
        
        logger.info(f"🔍 Starting {validation_level.value} reasoning validation")
        
        if context is None:
            context = {}
        if patterns is None:
            patterns = {}
        
        validation_results = []
        critical_issues = []
        
        # Run validators based on level
        validators_to_run = self._select_validators(validation_level)
        
        for criterion in validators_to_run:
            if criterion in self.validators:
                try:
                    if criterion == ValidationCriteria.LOGICAL_CONSISTENCY:
                        result = await self.validators[criterion].validate(reasoning_chain, question)
                    elif criterion == ValidationCriteria.EVIDENCE_SUPPORT:
                        result = await self.validators[criterion].validate(reasoning_chain, context)
                    elif criterion == ValidationCriteria.COMPLETENESS:
                        result = await self.validators[criterion].validate(reasoning_chain, question, patterns)
                    else:
                        continue
                    
                    validation_results.append(result)
                    
                    # Collect critical issues
                    threshold = self.validation_thresholds[validation_level]
                    if result.score < threshold:
                        critical_issues.extend(result.issues_found)
                    
                except Exception as e:
                    logger.error(f"Validation failed for {criterion.value}: {e}")
        
        # Calculate overall scores
        overall_score = sum(r.score for r in validation_results) / len(validation_results) if validation_results else 0.0
        validator_confidence = sum(r.confidence for r in validation_results) / len(validation_results) if validation_results else 0.0
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_recommendations(validation_results, validation_level)
        
        return ReasoningValidationReport(
            overall_score=overall_score,
            validation_level=validation_level,
            criteria_results=validation_results,
            critical_issues=list(set(critical_issues)),  # Remove duplicates
            improvement_recommendations=improvement_recommendations,
            validated_at=datetime.utcnow(),
            validator_confidence=validator_confidence
        )
    
    def _select_validators(self, validation_level: ValidationLevel) -> List[ValidationCriteria]:
        """Select validators based on validation level"""
        
        if validation_level == ValidationLevel.BASIC:
            return [ValidationCriteria.LOGICAL_CONSISTENCY]
        elif validation_level == ValidationLevel.INTERMEDIATE:
            return [
                ValidationCriteria.LOGICAL_CONSISTENCY,
                ValidationCriteria.EVIDENCE_SUPPORT
            ]
        else:  # COMPREHENSIVE
            return [
                ValidationCriteria.LOGICAL_CONSISTENCY,
                ValidationCriteria.EVIDENCE_SUPPORT,
                ValidationCriteria.COMPLETENESS
            ]
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                validation_level: ValidationLevel) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Collect all suggestions
        all_suggestions = []
        for result in validation_results:
            all_suggestions.extend(result.suggestions)
        
        # Remove duplicates and prioritize
        unique_suggestions = list(set(all_suggestions))
        
        # Add level-specific recommendations
        if validation_level == ValidationLevel.COMPREHENSIVE:
            avg_score = sum(r.score for r in validation_results) / len(validation_results)
            if avg_score < 0.8:
                recommendations.append("Consider using multiple reasoning approaches for cross-validation")
                recommendations.append("Implement iterative refinement of reasoning chains")
        
        # Combine all recommendations
        recommendations.extend(unique_suggestions[:5])  # Limit to top 5
        
        return recommendations
    
    async def validate_and_improve(self, reasoning_chain: List[Dict[str, Any]], 
                                 question: str, context: Dict[str, Any] = None,
                                 max_iterations: int = 3) -> Tuple[List[Dict[str, Any]], ReasoningValidationReport]:
        """Validate reasoning and iteratively improve it"""
        
        current_chain = reasoning_chain.copy()
        
        for iteration in range(max_iterations):
            # Validate current chain
            validation_report = await self.validate_reasoning(
                current_chain, question, context, validation_level=ValidationLevel.INTERMEDIATE
            )
            
            # If validation passes, return
            if validation_report.overall_score >= 0.7:
                logger.info(f"✅ Reasoning validation passed after {iteration + 1} iterations")
                return current_chain, validation_report
            
            # Attempt to improve based on validation feedback
            improved_chain = await self._attempt_improvement(current_chain, validation_report)
            
            if improved_chain:
                current_chain = improved_chain
            else:
                logger.warning("Could not improve reasoning chain further")
                break
        
        return current_chain, validation_report
    
    async def _attempt_improvement(self, reasoning_chain: List[Dict[str, Any]], 
                                 validation_report: ReasoningValidationReport) -> Optional[List[Dict[str, Any]]]:
        """Attempt to improve reasoning based on validation feedback"""
        
        # Simple improvement heuristics
        improved_chain = reasoning_chain.copy()
        
        # If logical consistency issues, try to fix contradictions
        consistency_result = next((r for r in validation_report.criteria_results 
                                 if r.criterion == ValidationCriteria.LOGICAL_CONSISTENCY), None)
        
        if consistency_result and consistency_result.score < 0.6:
            # Remove steps with contradictions (simplified approach)
            contradictions = consistency_result.evidence.get("contradictions_found", [])
            if contradictions:
                # Filter out problematic steps (very simplified)
                improved_chain = [step for step in improved_chain 
                                if not any(contradiction[0] in str(step) for contradiction in contradictions)]
        
        return improved_chain if improved_chain != reasoning_chain else None
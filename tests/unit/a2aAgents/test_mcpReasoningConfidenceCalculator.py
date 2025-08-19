"""
Tests for MCP-enabled Reasoning Confidence Calculator
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from .mcpReasoningConfidenceCalculator import MCPReasoningConfidenceCalculator, ConfidenceFactors

class TestMCPReasoningConfidenceCalculator:
    """Test suite for MCP reasoning confidence calculator"""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return MCPReasoningConfidenceCalculator()
    
    @pytest.fixture
    def sample_reasoning_context(self):
        """Sample reasoning context for testing"""
        return {
            "evidence": [
                {"source_type": "academic", "content": "Study shows..."},
                {"source_type": "verified", "content": "Data confirms..."},
                {"source_type": "empirical", "content": "Experiments indicate..."}
            ],
            "reasoning_chain": [
                {"premise": "A", "conclusion": "B", "confidence": 0.8, "inference_rule": "modus_ponens"},
                {"premise": "B", "conclusion": "C", "confidence": 0.7, "inference_rule": "hypothetical_syllogism"}
            ],
            "question": "What causes climate change?",
            "answer": "Climate change is caused by greenhouse gas emissions",
            "historical_data": {
                "total_attempts": 10,
                "successful_attempts": 8,
                "recent_results": [1, 1, 0]
            },
            "validation_results": {
                "logical_consistency": 0.85,
                "evidence_support": 0.9,
                "completeness": 0.75,
                "validator_confidence": 0.8
            }
        }
    
    @pytest.mark.asyncio
    async def test_calculate_reasoning_confidence_mcp(self, calculator, sample_reasoning_context):
        """Test MCP confidence calculation"""
        result = await calculator.calculate_reasoning_confidence_mcp(
            sample_reasoning_context,
            include_explanation=True
        )
        
        # Check structure
        assert "confidence" in result
        assert "factor_breakdown" in result
        assert "explanation" in result
        assert "recommendations" in result
        
        # Check confidence bounds
        assert 0.05 <= result["confidence"] <= 0.95
        
        # Check factor breakdown
        breakdown = result["factor_breakdown"]
        assert len(breakdown) == 6  # All factors should be present
        
        for factor in ConfidenceFactors:
            assert factor.value in breakdown
            factor_data = breakdown[factor.value]
            assert "value" in factor_data
            assert "weight" in factor_data
            assert "contribution" in factor_data
            assert 0 <= factor_data["value"] <= 1
    
    @pytest.mark.asyncio
    async def test_custom_weights(self, calculator, sample_reasoning_context):
        """Test custom weight configuration"""
        custom_weights = {
            ConfidenceFactors.EVIDENCE_QUALITY: 0.5,
            ConfidenceFactors.LOGICAL_CONSISTENCY: 0.3,
            ConfidenceFactors.SEMANTIC_ALIGNMENT: 0.1,
            ConfidenceFactors.HISTORICAL_SUCCESS: 0.05,
            ConfidenceFactors.COMPLEXITY_PENALTY: 0.03,
            ConfidenceFactors.VALIDATION_SCORE: 0.02
        }
        
        result = await calculator.calculate_reasoning_confidence_mcp(
            sample_reasoning_context,
            custom_weights=custom_weights
        )
        
        # Verify weights were applied
        breakdown = result["factor_breakdown"]
        for factor, weight in custom_weights.items():
            assert breakdown[factor.value]["weight"] == weight
    
    @pytest.mark.asyncio
    async def test_calculate_evidence_quality_mcp(self, calculator):
        """Test MCP evidence quality calculation"""
        evidence = [
            {"source_type": "academic"},
            {"source_type": "verified"},
            {"source_type": "empirical"},
            {"source_type": "unknown"}
        ]
        
        result = await calculator.calculate_evidence_quality_mcp(
            evidence,
            return_details=True
        )
        
        assert "quality_score" in result
        assert "evidence_composition" in result
        assert "evidence_count" in result
        assert "has_academic_sources" in result
        assert "has_verified_sources" in result
        
        assert result["evidence_count"] == 4
        assert result["has_academic_sources"] is True
        assert result["has_verified_sources"] is True
    
    @pytest.mark.asyncio
    async def test_calculate_semantic_alignment_mcp(self, calculator):
        """Test MCP semantic alignment calculation"""
        question = "What is machine learning?"
        answer = "Machine learning is a type of artificial intelligence that enables computers to learn"
        
        result = await calculator.calculate_semantic_alignment_mcp(
            question,
            answer,
            analyze_keywords=True
        )
        
        assert "alignment_score" in result
        assert 0 <= result["alignment_score"] <= 1
        
        # With keyword analysis
        assert "keyword_overlap" in result
        assert "unique_question_keywords" in result
        assert "unique_answer_keywords" in result
        assert "question_type" in result
        assert result["question_type"] == "definition"
    
    @pytest.mark.asyncio
    async def test_adjust_confidence_for_uncertainty_mcp(self, calculator):
        """Test MCP uncertainty adjustment"""
        base_confidence = 0.8
        uncertainty_factors = ["missing_evidence", "weak_inference", "limited_data"]
        
        result = await calculator.adjust_confidence_for_uncertainty_mcp(
            base_confidence,
            uncertainty_factors
        )
        
        assert "original_confidence" in result
        assert "adjusted_confidence" in result
        assert "total_penalty" in result
        assert "factor_impacts" in result
        
        assert result["original_confidence"] == base_confidence
        assert result["adjusted_confidence"] < base_confidence
        assert len(result["factor_impacts"]) == len(uncertainty_factors)
    
    @pytest.mark.asyncio
    async def test_get_calculator_status(self, calculator):
        """Test MCP resource - calculator status"""
        status = await calculator.get_calculator_status()
        
        assert "factor_weights" in status
        assert "confidence_bounds" in status
        assert "available_factors" in status
        assert "uncertainty_factors" in status
        
        assert len(status["factor_weights"]) == 6
        assert status["confidence_bounds"]["min"] == 0.05
        assert status["confidence_bounds"]["max"] == 0.95
    
    @pytest.mark.asyncio
    async def test_confidence_analysis_prompt(self, calculator, sample_reasoning_context):
        """Test MCP prompt - confidence analysis"""
        prompt = await calculator.confidence_analysis_prompt(
            sample_reasoning_context,
            focus_area="evidence"
        )
        
        assert isinstance(prompt, str)
        assert "Overall Confidence:" in prompt
        assert "Factor Breakdown:" in prompt
        assert "Focusing on evidence:" in prompt
        assert "Recommendations:" in prompt
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, calculator):
        """Test edge cases"""
        # Empty context
        empty_context = {
            "evidence": [],
            "reasoning_chain": [],
            "question": "",
            "answer": ""
        }
        
        result = await calculator.calculate_reasoning_confidence_mcp(empty_context)
        assert result["confidence"] >= 0.05  # Should return minimum confidence
        
        # Missing fields
        partial_context = {
            "evidence": [{"source_type": "academic"}]
        }
        
        result = await calculator.calculate_reasoning_confidence_mcp(partial_context)
        assert "confidence" in result
        assert "factor_breakdown" in result
    
    def test_fallback_confidence(self, calculator):
        """Test fallback confidence calculations"""
        scenarios = [
            "no_historical_data",
            "single_agent_fallback",
            "internal_analysis",
            "unknown_scenario"
        ]
        
        for scenario in scenarios:
            confidence = calculator.calculate_fallback_confidence(scenario)
            assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_mcp_tool_metadata(self, calculator):
        """Test MCP tool metadata is properly set"""
        # Check main tool
        calc_method = calculator.calculate_reasoning_confidence_mcp
        assert hasattr(calc_method, '_mcp_tool')
        
        tool_meta = calc_method._mcp_tool
        assert tool_meta['name'] == "calculate_reasoning_confidence"
        assert 'description' in tool_meta
        assert 'input_schema' in tool_meta
        assert 'output_schema' in tool_meta
        
        # Check skill provides
        assert hasattr(calc_method, '_skill_provides')
        provides = calc_method._skill_provides
        assert "confidence_calculation" in provides
        assert "reasoning_analysis" in provides


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
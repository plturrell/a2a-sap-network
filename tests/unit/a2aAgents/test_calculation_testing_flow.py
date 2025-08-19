"""
Integration tests for the complete calculation testing flow
Tests CalcTesting <-> CalculationAgent <-> Data_Manager integration
"""

import asyncio
import pytest
import json
import os
import sys
from unittest.mock import Mock, AsyncMock, patch

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')

# Import components to test
from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
from app.a2a.agents.calculationAgent.active.calculationAgentSdk import CalculationAgentSDK
from app.a2a.agents.dataManager.active.dataManagerAgentSdk import DataManagerAgentSDK
from app.a2a.agents.agent4CalcValidation.active.calcTestingIntegrationSkills import (
    CalcTestingIntegrationSkills, TestQuestion, CalculationResult, EvaluationScore
)
from app.a2a.core.grokClient import GrokClient


class TestCalculationTestingFlow:
    """Integration tests for calculation testing flow"""
    
    @pytest.fixture
    async def calc_validation_agent(self):
        """Create CalcValidation agent for testing"""
        agent = CalcValidationAgentSDK(
            base_url="http://localhost:8004",
            enable_monitoring=False
        )
        # Mock the data manager and catalog manager calls
        agent._call_data_manager = AsyncMock(return_value={"success": True})
        agent._call_catalog_manager = AsyncMock(return_value={"success": True})
        
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.fixture
    async def calculation_agent(self):
        """Create Calculation agent for testing"""
        agent = CalculationAgentSDK(
            base_url="http://localhost:8007",
            enable_monitoring=False
        )
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.fixture
    def grok_client(self):
        """Create GrokClient for testing"""
        # Use local analysis mode for testing
        client = GrokClient()
        client.use_local_analysis = True
        return client
    
    @pytest.mark.asyncio
    async def test_grok_client_local_analysis(self, grok_client):
        """Test GrokClient local analysis functionality"""
        prompt = """
        Evaluate the following calculation result:
        
        Question: Calculate the derivative of x^2 + 3x + 5
        Provided Answer: 2x + 3
        Methodology: Power rule differentiation
        Steps: [{"description": "Apply power rule", "result": "2x + 3"}]
        """
        
        result = await grok_client.analyze(prompt)
        
        # Should return valid JSON
        data = json.loads(result)
        assert "accuracy_score" in data
        assert "methodology_score" in data
        assert "explanation_score" in data
        assert "overall_score" in data
        assert "feedback" in data
        assert "passed" in data
        
        # Scores should be reasonable for a correct answer
        assert data["accuracy_score"] >= 60
        assert data["overall_score"] >= 50
    
    @pytest.mark.asyncio
    async def test_grok_client_evaluate_calculation(self, grok_client):
        """Test GrokClient specialized evaluation method"""
        result = await grok_client.evaluate_calculation(
            question="What is the derivative of x^2?",
            answer="2x",
            methodology="Power rule",
            steps=[{"step": 1, "description": "Apply power rule", "result": "2x"}],
            expected_answer="2x"
        )
        
        assert isinstance(result, dict)
        assert "accuracy_score" in result
        assert "passed" in result
        assert result["accuracy_score"] >= 70  # Should be high for correct answer
    
    @pytest.mark.asyncio
    async def test_calculation_testing_integration_skills_dispatch(self, calc_validation_agent):
        """Test dispatching test questions to calculation agent"""
        integration_skills = CalcTestingIntegrationSkills(calc_validation_agent)
        
        # Mock the calculation agent response
        mock_response = {
            "success": True,
            "data": {
                "result": {
                    "answer": "2x + 3",
                    "methodology": "Power rule differentiation",
                    "steps": [{"description": "Apply power rule", "result": "2x + 3"}],
                    "confidence": 0.95,
                    "computation_time": 0.1
                }
            }
        }
        
        integration_skills._call_calculation_agent = AsyncMock(return_value=mock_response)
        
        test_question = TestQuestion(
            question="Calculate the derivative of x^2 + 3x + 5",
            category="mathematical",
            difficulty="easy"
        )
        
        result = await integration_skills.dispatch_test_question(test_question)
        
        assert result["status"] == "success"
        assert "calculation_result" in result
        assert result["question_id"] == test_question.question_id
    
    @pytest.mark.asyncio
    async def test_calculation_testing_integration_skills_evaluation(self, calc_validation_agent):
        """Test evaluation of calculation results"""
        integration_skills = CalcTestingIntegrationSkills(calc_validation_agent)
        
        test_question = TestQuestion(
            question="Calculate the derivative of x^2 + 3x + 5",
            category="mathematical",
            difficulty="easy"
        )
        
        calc_result = CalculationResult(
            answer="2x + 3",
            methodology="Power rule differentiation",
            steps=[{"description": "Apply power rule", "result": "2x + 3"}],
            confidence=0.95,
            computation_time=0.1
        )
        
        evaluation = await integration_skills.evaluate_calculation_answer(
            test_question,
            calc_result,
            expected_answer="2x + 3"
        )
        
        assert isinstance(evaluation, EvaluationScore)
        assert evaluation.question_id == test_question.question_id
        assert evaluation.overall_score > 0
        assert isinstance(evaluation.passed, bool)
    
    @pytest.mark.asyncio
    async def test_calculation_agent_intelligent_dispatch(self, calculation_agent):
        """Test CalculationAgent intelligent dispatch skill"""
        result = await calculation_agent.intelligent_dispatch(
            request="Calculate the derivative of x^2 + 3x + 5",
            metadata={"test_mode": True}
        )
        
        assert result["status"] == "success"
        assert "result" in result
        assert "agent_id" in result
        assert result["agent_id"] == calculation_agent.agent_id
    
    @pytest.mark.asyncio
    async def test_calc_validation_agent_dispatch_skill(self, calc_validation_agent):
        """Test CalcValidation agent dispatch skill"""
        # Mock the testing integration to avoid actual network calls
        calc_validation_agent.testing_integration._call_calculation_agent = AsyncMock(
            return_value={
                "success": True,
                "data": {
                    "result": {
                        "answer": "2x + 3",
                        "methodology": "Power rule",
                        "steps": [],
                        "confidence": 0.9
                    }
                }
            }
        )
        
        result = await calc_validation_agent.dispatch_calculation_test(
            question="Calculate the derivative of x^2 + 3x + 5",
            category="mathematical",
            difficulty="easy"
        )
        
        assert result["status"] == "success"
        assert "calculation_result" in result
    
    @pytest.mark.asyncio
    async def test_calc_validation_agent_scoreboard(self, calc_validation_agent):
        """Test CalcValidation agent scoreboard functionality"""
        # Add some test data to the scoreboard
        calc_validation_agent.testing_integration.scoreboard.total_questions = 5
        calc_validation_agent.testing_integration.scoreboard.correct_answers = 4
        calc_validation_agent.testing_integration.scoreboard.accuracy_rate = 80.0
        
        result = await calc_validation_agent.get_calculation_test_scoreboard()
        
        assert "summary" in result
        assert result["summary"]["total_questions"] == 5
        assert result["summary"]["accuracy_rate"] == "80.00%"
    
    @pytest.mark.asyncio
    async def test_enhanced_calculation_skills_integration(self, calculation_agent):
        """Test enhanced calculation skills integration"""
        # Test that enhanced skills are properly integrated
        assert hasattr(calculation_agent, 'enhanced_skills')
        assert calculation_agent.enhanced_skills is not None
        
        # Test calculation with explanation
        result = await calculation_agent.enhanced_skills.calculate_with_explanation(
            "x^2 + 3x + 5"
        )
        
        assert "answer" in result
        assert "methodology" in result
        assert "steps" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_a2a_message_flow_simulation(self, calc_validation_agent, calculation_agent):
        """Test simulated A2A message flow between agents"""
        # This test simulates the message flow without actual network calls
        
        # 1. CalcValidation creates test question
        test_question = TestQuestion(
            question="Calculate the integral of 2x",
            category="mathematical", 
            difficulty="medium"
        )
        
        # 2. Mock calculation agent response
        mock_calc_response = {
            "status": "success",
            "result": {
                "answer": "x^2 + C",
                "methodology": "Basic integration rule",
                "steps": [
                    {"description": "Apply integration rule", "result": "x^2 + C"}
                ],
                "confidence": 0.9,
                "computation_time": 0.2
            }
        }
        
        # 3. Mock data manager storage
        calc_validation_agent._call_data_manager = AsyncMock(return_value={"success": True})
        
        # 4. Test the flow
        calc_validation_agent.testing_integration._call_calculation_agent = AsyncMock(
            return_value=mock_calc_response
        )
        
        # Dispatch question
        dispatch_result = await calc_validation_agent.testing_integration.dispatch_test_question(test_question)
        assert dispatch_result["status"] == "success"
        
        # Evaluate result
        calc_result = CalculationResult(**dispatch_result["calculation_result"])
        evaluation = await calc_validation_agent.testing_integration.evaluate_calculation_answer(
            test_question, calc_result, "x^2 + C"
        )
        
        assert evaluation.overall_score > 0
        assert isinstance(evaluation.passed, bool)
        
        # Get updated scoreboard
        scoreboard = await calc_validation_agent.testing_integration.get_scoreboard_report()
        assert "summary" in scoreboard
    
    @pytest.mark.asyncio
    async def test_trust_system_integration(self, calc_validation_agent):
        """Test that trust system components are properly integrated"""
        # Check that trust system is initialized
        assert hasattr(calc_validation_agent, 'trust_identity')
        assert hasattr(calc_validation_agent, 'trusted_agents')
        
        # Check that essential agents are pre-trusted
        expected_trusted = {
            "agent_manager",
            "data_product_agent_0", 
            "data_standardization_agent_1",
            "ai_preparation_agent_2",
            "vector_processing_agent_3"
        }
        
        # Should have some overlap with expected trusted agents
        assert len(calc_validation_agent.trusted_agents.intersection(expected_trusted)) > 0
    
    def test_integration_skills_initialization(self, calc_validation_agent):
        """Test that integration skills are properly initialized"""
        assert hasattr(calc_validation_agent, 'testing_integration')
        assert isinstance(calc_validation_agent.testing_integration, CalcTestingIntegrationSkills)
        assert calc_validation_agent.testing_integration.agent == calc_validation_agent


@pytest.mark.asyncio 
async def test_src_aiq_agent_import():
    """Test that the enhanced agent in src/aiq can be imported"""
    try:
        sys.path.insert(0, '/Users/apple/projects/a2a/src')
        from aiq.agent.calculation_agent import A2ACalculationAgent, create_calculation_agent
        
        # Test that we can create an instance
        agent = A2ACalculationAgent(enable_aiq=False)  # Disable AIQ for testing
        assert agent.agent_id == "enhanced_calculation_agent"
        assert agent.name == "Enhanced A2A Calculation Agent"
        assert agent.version == "2.0.0"
        
    except ImportError as e:
        pytest.skip(f"Could not import enhanced agent: {e}")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_src_aiq_agent_import())
    print("All imports successful!")
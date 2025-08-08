# Integration test for AI Decision Logger with Financial Standardization Agent
# Tests real integration without false claims

import asyncio
import json
import pytest
import tempfile
from unittest.mock import AsyncMock, patch

from src.a2a.core.ai_decision_logger import DecisionType, OutcomeStatus
from src.a2a.core.a2a_types import A2AMessage, MessagePart, MessageRole


class TestAIDecisionIntegration:
    """Test AI Decision Logger integration with actual agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_logger(self):
        """Test that agent initializes with AI Decision Logger"""
        # Import here to avoid circular imports
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent instance
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            
            # Verify AI Decision Logger was initialized
            assert hasattr(agent, 'ai_decision_logger')
            assert agent.ai_decision_logger is not None
            assert agent.ai_decision_logger.agent_id == agent.agent_id
            
            # Verify logger is registered globally (agent uses database registry)
            from src.a2a.core.ai_decision_logger_database import get_global_database_decision_registry
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_advisor_request_logging(self, mock_create_advisor):
        """Test that advisor requests are logged"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_advisor.process_a2a_help_message = AsyncMock(
            return_value={
                "answer": "Check data integrity checksums and verify source data",
                "confidence": 0.85,
                "sources": ["documentation"]
            }
        )
        mock_create_advisor.return_value = mock_advisor
        
        # Create agent
        agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
        
        # Create advisor request message
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="text", 
                    text="How do I handle data integrity issues?"
                ),
                MessagePart(
                    kind="data", 
                    data={
                        "advisor_request": True,
                        "question": "How do I handle data integrity issues?"
                    }
                )
            ]
        )
        
        # Process the advisor request
        response = await agent._handle_advisor_request(message, "test_context")
        
        # Verify response structure
        assert response["message_type"] == "advisor_response"
        assert "decision_id" in response
        assert response["agent_id"] == agent.agent_id
        
        # Verify decision was logged
        decision_id = response["decision_id"]
        assert decision_id in agent.ai_decision_logger.decisions
        assert decision_id in agent.ai_decision_logger.outcomes
        
        # Check decision details
        decision = agent.ai_decision_logger.decisions[decision_id]
        assert decision.decision_type == DecisionType.ADVISOR_GUIDANCE
        assert decision.question == "How do I handle data integrity issues?"
        assert decision.agent_id == agent.agent_id
        
        # Check outcome
        outcome = agent.ai_decision_logger.outcomes[decision_id]
        assert outcome.outcome_status == OutcomeStatus.SUCCESS
        assert outcome.success_metrics["has_answer"] is True
        assert outcome.success_metrics["response_time"] > 0
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_help_seeking_logging(self, mock_create_advisor):
        """Test that help-seeking is logged"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_create_advisor.return_value = mock_advisor
        
        # Create agent
        agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
        
        # Mock the seek_help_for_error method to simulate help response
        agent.seek_help_for_error = AsyncMock(
            return_value={
                "advisor_response": {
                    "answer": "Try reconnecting to the database and verify credentials",
                    "confidence": 0.7
                },
                "agent_id": "helper_agent"
            }
        )
        
        # Simulate error handling with help seeking
        test_error = Exception("Database connection timeout")
        await agent._handle_error_with_help_seeking(test_error, "task_123", "context_123")
        
        # Verify help-seeking decision was logged
        help_decisions = [
            d for d in agent.ai_decision_logger.decisions.values()
            if d.decision_type == DecisionType.HELP_REQUEST
        ]
        
        assert len(help_decisions) >= 1
        help_decision = help_decisions[-1]  # Get the most recent
        
        assert "Database connection timeout" in help_decision.question
        assert help_decision.context["error_type"] == "Exception"
        assert help_decision.context["processing_stage"] == "standardization_execution"
        
        # Verify outcome was logged
        help_decision_id = help_decision.decision_id
        assert help_decision_id in agent.ai_decision_logger.outcomes
        
        outcome = agent.ai_decision_logger.outcomes[help_decision_id]
        assert outcome.outcome_status == OutcomeStatus.SUCCESS
        assert outcome.success_metrics["help_received"] is True
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_analytics_endpoints_work(self, mock_create_advisor):
        """Test that analytics endpoints return data"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_create_advisor.return_value = mock_advisor
        
        # Create agent
        agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
        
        # Add some test decisions
        await agent.ai_decision_logger.log_decision(
            DecisionType.ADVISOR_GUIDANCE,
            "Test question 1",
            {"answer": "Test answer 1"},
            {"test": True}
        )
        
        decision_id = await agent.ai_decision_logger.log_decision(
            DecisionType.HELP_REQUEST,
            "Test question 2", 
            {"answer": "Test answer 2"}
        )
        
        await agent.ai_decision_logger.log_outcome(
            decision_id,
            OutcomeStatus.SUCCESS,
            {"test_metric": 1.0}
        )
        
        # Test analytics
        analytics = agent.ai_decision_logger.get_decision_analytics()
        
        assert analytics["summary"]["total_decisions"] == 2
        assert analytics["summary"]["successful_outcomes"] == 1
        assert DecisionType.ADVISOR_GUIDANCE.value in analytics["by_type"]
        assert DecisionType.HELP_REQUEST.value in analytics["by_type"]
        
        # Test history
        history = agent.ai_decision_logger.get_decision_history(limit=5)
        assert len(history) == 2
        assert history[0]["question"] == "Test question 2"  # Most recent first
        
        # Test recommendations
        recommendations = await agent.ai_decision_logger.get_recommendations(
            DecisionType.ADVISOR_GUIDANCE,
            {"test": True}
        )
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_persistence_works(self, mock_create_advisor):
        """Test that decision data persists correctly"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        from src.a2a.core.ai_decision_logger import AIDecisionLogger
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_create_advisor.return_value = mock_advisor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent with specific storage path
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.storage_path = temp_dir
            
            # Add test decision
            decision_id = await agent.ai_decision_logger.log_decision(
                DecisionType.ADVISOR_GUIDANCE,
                "Persistent test question",
                {"answer": "Persistent test answer"}
            )
            
            await agent.ai_decision_logger.log_outcome(
                decision_id,
                OutcomeStatus.SUCCESS
            )
            
            # Force persistence
            await agent.ai_decision_logger._persist_data()
            
            # Create new logger with same storage path
            new_logger = AIDecisionLogger(
                agent_id=agent.agent_id,
                storage_path=temp_dir
            )
            
            await new_logger.load_historical_data()
            
            # Verify data was loaded
            assert len(new_logger.decisions) == 1
            assert len(new_logger.outcomes) == 1
            assert decision_id in new_logger.decisions
            assert new_logger.decisions[decision_id].question == "Persistent test question"
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_pattern_learning_works(self, mock_create_advisor):
        """Test that pattern learning works with real data"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_create_advisor.return_value = mock_advisor
        
        # Create agent
        agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
        
        # Create multiple decisions with patterns
        for i in range(6):  # Above learning threshold
            decision_id = await agent.ai_decision_logger.log_decision(
                DecisionType.ADVISOR_GUIDANCE,
                f"Question about data integrity {i}",
                {"answer": f"Answer {i}", "confidence": 0.8 if i < 4 else 0.3},
                {"complexity": "high" if i < 4 else "low"}
            )
            
            # First 4 succeed, last 2 fail
            await agent.ai_decision_logger.log_outcome(
                decision_id,
                OutcomeStatus.SUCCESS if i < 4 else OutcomeStatus.FAILURE
            )
        
        # Trigger pattern analysis
        await agent.ai_decision_logger._analyze_patterns()
        
        # Verify patterns were learned
        assert len(agent.ai_decision_logger.learned_patterns) > 0
        
        # Test recommendations based on patterns
        recommendations = await agent.ai_decision_logger.get_recommendations(
            DecisionType.ADVISOR_GUIDANCE,
            {"complexity": "high"}
        )
        
        assert isinstance(recommendations, list)
        # Should have some recommendations since we have patterns
        # (exact content depends on pattern learning logic)
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_insights_report_generation(self, mock_create_advisor):
        """Test that insights report generates correctly"""
        from app.a2a.agents.data_standardization_agent import FinancialStandardizationAgent
        
        # Mock AI advisor
        mock_advisor = AsyncMock()
        mock_create_advisor.return_value = mock_advisor
        
        # Create agent
        agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
        
        # Add comprehensive test data
        for decision_type in [DecisionType.ADVISOR_GUIDANCE, DecisionType.HELP_REQUEST]:
            for i in range(3):
                decision_id = await agent.ai_decision_logger.log_decision(
                    decision_type,
                    f"{decision_type.value} question {i}",
                    {"answer": f"Answer {i}"},
                    {"test_context": i}
                )
                
                await agent.ai_decision_logger.log_outcome(
                    decision_id,
                    OutcomeStatus.SUCCESS if i % 2 == 0 else OutcomeStatus.FAILURE
                )
        
        # Generate insights report
        report = await agent.ai_decision_logger.export_insights_report()
        
        # Verify report structure
        assert report["agent_id"] == agent.agent_id
        assert "report_timestamp" in report
        assert "summary" in report
        assert "performance_by_type" in report
        assert "success_analysis" in report
        assert "recommendations" in report
        
        # Verify summary data
        assert report["summary"]["total_decisions"] == 6
        assert report["summary"]["successful_outcomes"] == 3
        assert report["summary"]["failed_outcomes"] == 3
        assert report["summary"]["overall_success_rate"] == 0.5
        
        # Verify recommendations exist for each decision type
        assert DecisionType.ADVISOR_GUIDANCE.value in report["recommendations"]
        assert DecisionType.HELP_REQUEST.value in report["recommendations"]


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
# Integration test for Database AI Decision Logger with Data Standardization Agent
# Tests real integration with database persistence

import asyncio
import json
import pytest
import tempfile
from unittest.mock import AsyncMock, patch

from app.a2a.core.aiDecisionLogger import DecisionType, OutcomeStatus
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole


class TestDatabaseAIDecisionIntegration:
    """Test Database AI Decision Logger integration with actual agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_database_logger(self):
        """Test that agent initializes with Database AI Decision Logger"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient'):
            # Import here to avoid circular imports
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Create agent instance
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            
            # Verify Database AI Decision Logger was initialized
            assert hasattr(agent, 'ai_decision_logger')
            assert agent.ai_decision_logger is not None
            assert agent.ai_decision_logger.agent_id == agent.agent_id
            
            # Verify it's the database version
            from app.a2a.core.aiDecisionLoggerDatabase import AIDecisionDatabaseLogger
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            
            # Verify data manager URL was set
            expected_url = "http://localhost:8001/data-manager"
            assert agent.ai_decision_logger.data_manager_url == expected_url
            
            # Verify logger is registered globally
            from app.a2a.core.aiDecisionLoggerDatabase import get_global_database_decision_registry
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_advisor_request_logging_to_database(self, mock_create_advisor):
        """Test that advisor requests are logged to database"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            # Mock successful database responses
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "overall_status": "SUCCESS",
                "primary_result": {"status": "SUCCESS"}
            }
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
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
            agent.ai_decision_logger.http_client = mock_client
            
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
            
            # Verify database calls were made
            assert mock_client.post.call_count >= 2  # Decision log + outcome log
            
            # Check decision logging call
            decision_calls = [
                call for call in mock_client.post.call_args_list
                if "ai_decisions" in str(call)
            ]
            assert len(decision_calls) >= 1
            
            decision_call = decision_calls[0]
            message_data = decision_call[1]["json"]
            
            # Find data part with decision
            data_part = None
            for part in message_data["parts"]:
                if (part["kind"] == "data" and 
                    part["data"].get("path") == "ai_decisions"):
                    data_part = part["data"]
                    break
            
            assert data_part is not None
            assert data_part["operation"] == "CREATE"
            assert data_part["data"]["agent_id"] == agent.agent_id
            assert data_part["data"]["decision_type"] == "advisor_guidance"
            assert "data integrity issues" in data_part["data"]["question"]
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_help_seeking_logging_to_database(self, mock_create_advisor):
        """Test that help-seeking is logged to database"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "overall_status": "SUCCESS",
                "primary_result": {"status": "SUCCESS"}
            }
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.http_client = mock_client
            
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
            
            # Verify database calls were made for help-seeking
            help_request_calls = [
                call for call in mock_client.post.call_args_list
                if "help_request" in str(call) or "HELP_REQUEST" in str(call)
            ]
            
            # Should have calls for logging the help request decision and outcome
            assert len(help_request_calls) >= 1
            
            # Check one of the help request calls
            if help_request_calls:
                call = help_request_calls[0]
                message_data = call[1]["json"]
                
                # Find help request data
                for part in message_data["parts"]:
                    if (part["kind"] == "data" and 
                        "ai_decisions" in part["data"].get("path", "")):
                        decision_data = part["data"]["data"]
                        if decision_data.get("decision_type") == "help_request":
                            assert "Database connection timeout" in decision_data["question"]
                            assert decision_data["agent_id"] == agent.agent_id
                            break
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_analytics_endpoints_with_database(self, mock_create_advisor):
        """Test that analytics endpoints work with database backend"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            # Mock analytics response from database
            mock_client = AsyncMock()
            mock_analytics_response = AsyncMock()
            mock_analytics_response.status_code = 200
            mock_analytics_response.json.return_value = {
                "overall_status": "SUCCESS",
                "data": [
                    {
                        "agent_id": "financial_data_standardization_agent",
                        "decision_type": "advisor_guidance",
                        "total_decisions": 10,
                        "successful_outcomes": 8,
                        "decision_date": "2024-01-01"
                    }
                ]
            }
            mock_client.post.return_value = mock_analytics_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.http_client = mock_client
            
            # Test analytics
            analytics = await agent.ai_decision_logger.get_decision_analytics()
            
            assert analytics["summary"]["total_decisions"] == 10
            assert analytics["summary"]["successful_outcomes"] == 8
            assert analytics["summary"]["overall_success_rate"] == 0.8
            
            # Verify database query was made
            mock_client.post.assert_called()
            call_args = mock_client.post.call_args
            message_data = call_args[1]["json"]
            
            # Should query analytics view
            data_part = None
            for part in message_data["parts"]:
                if part["kind"] == "data" and "query" in part["data"]:
                    data_part = part["data"]
                    break
            
            assert data_part is not None
            assert data_part["operation"] == "READ"
            assert data_part["query"]["table"] == "ai_global_analytics"
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_recommendations_from_database_patterns(self, mock_create_advisor):
        """Test getting recommendations from database patterns"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            # Mock patterns response from database
            mock_client = AsyncMock()
            mock_patterns_response = AsyncMock()
            mock_patterns_response.status_code = 200
            mock_patterns_response.json.return_value = {
                "overall_status": "SUCCESS",
                "data": [
                    {
                        "pattern_id": "pattern_1",
                        "pattern_type": "advisor_guidance",
                        "confidence": 0.9,
                        "recommendations": json.dumps([
                            "Use high confidence thresholds for financial data",
                            "Validate responses with multiple data sources",
                            "Consider data lineage in standardization decisions"
                        ])
                    }
                ]
            }
            mock_client.post.return_value = mock_patterns_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.http_client = mock_client
            
            # Get recommendations
            recommendations = await agent.ai_decision_logger.get_recommendations(
                DecisionType.ADVISOR_GUIDANCE,
                {"complexity": "high", "domain": "finance"}
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) == 3
            assert "Use high confidence thresholds for financial data" in recommendations
            assert "Validate responses with multiple data sources" in recommendations
            assert "Consider data lineage in standardization decisions" in recommendations
            
            # Verify database query was made to patterns table
            call_args = mock_client.post.call_args
            message_data = call_args[1]["json"]
            
            data_part = None
            for part in message_data["parts"]:
                if part["kind"] == "data" and "query" in part["data"]:
                    data_part = part["data"]
                    break
            
            assert data_part["query"]["table"] == "ai_learned_patterns"
            assert data_part["query"]["where"]["agent_id"] == agent.agent_id
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_cross_agent_insights(self, mock_create_advisor):
        """Test getting cross-agent insights from global database registry"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "overall_status": "SUCCESS",
                "data": [{"total_decisions": 50, "successful_outcomes": 42}]
            }
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            from app.a2a.core.aiDecisionLoggerDatabase import get_global_database_decision_registry
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create multiple agents to simulate cross-agent scenario
            agent1 = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent1.ai_decision_logger.http_client = mock_client
            
            # Create another mock agent in registry
            registry = get_global_database_decision_registry()
            mock_agent2_logger = AsyncMock()
            mock_agent2_logger.get_decision_analytics = AsyncMock(
                return_value={
                    "summary": {"total_decisions": 25, "successful_outcomes": 20}
                }
            )
            registry.register_agent("agent2", mock_agent2_logger)
            
            # Get cross-agent insights
            insights = await registry.get_global_insights()
            
            assert insights["total_agents"] >= 2
            assert "agent_performance" in insights
            assert agent1.agent_id in insights["agent_performance"]
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_database_failure_fallback(self, mock_create_advisor):
        """Test fallback to cache when database is unavailable"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            # Mock database failure
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.text = "Database unavailable"
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.http_client = mock_client
            
            # Try to log decision - should fall back to cache
            decision_id = await agent.ai_decision_logger.log_decision(
                decision_type=DecisionType.ADVISOR_GUIDANCE,
                question="Test with database failure",
                ai_response={"answer": "Test response"}
            )
            
            # Decision should still be recorded in cache
            assert decision_id
            assert decision_id in agent.ai_decision_logger._decision_cache
            
            # Analytics should work from cache
            analytics = await agent.ai_decision_logger.get_decision_analytics()
            assert "summary" in analytics
            assert analytics["summary"]["total_decisions"] >= 1
    
    @pytest.mark.asyncio
    @patch('app.a2a.advisors.agent_ai_advisor.create_agent_advisor')
    async def test_insights_report_generation_from_database(self, mock_create_advisor):
        """Test that insights report generates from database data"""
        
        with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
            # Mock comprehensive database responses
            responses = [
                # Analytics response
                {
                    "overall_status": "SUCCESS",
                    "data": [{"total_decisions": 15, "successful_outcomes": 12}]
                },
                # Pattern effectiveness response  
                {
                    "overall_status": "SUCCESS",
                    "data": [
                        {
                            "pattern_type": "high_confidence_financial_decisions",
                            "description": "High confidence financial decisions have better outcomes",
                            "confidence": 0.88,
                            "evidence_count": 25,
                            "actual_success_rate": 0.92,
                            "recommendations": json.dumps([
                                "Maintain confidence above 0.8 for financial data",
                                "Use multiple validation sources",
                                "Consider regulatory compliance factors"
                            ])
                        }
                    ]
                }
            ]
            
            mock_client = AsyncMock()
            mock_client.post.side_effect = [
                AsyncMock(status_code=200, json=AsyncMock(return_value=responses[0])),
                AsyncMock(status_code=200, json=AsyncMock(return_value=responses[1]))
            ]
            mock_client_class.return_value = mock_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Mock AI advisor
            mock_advisor = AsyncMock()
            mock_create_advisor.return_value = mock_advisor
            
            # Create agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            agent.ai_decision_logger.http_client = mock_client
            
            # Generate insights report
            report = await agent.ai_decision_logger.export_insights_report()
            
            # Verify report structure
            assert report["agent_id"] == agent.agent_id
            assert "report_timestamp" in report
            assert "summary" in report
            assert "learned_insights" in report
            assert "data_quality" in report
            
            # Check learned insights from database
            assert len(report["learned_insights"]) == 1
            insight = report["learned_insights"][0]
            assert insight["pattern"] == "high_confidence_financial_decisions"
            assert insight["confidence"] == 0.88
            assert insight["effectiveness"] == 0.92
            assert len(insight["recommendations"]) == 3
            assert "Maintain confidence above 0.8 for financial data" in insight["recommendations"]
            
            # Check data quality metrics indicate database backing
            assert report["data_quality"]["decisions_stored"] == 15
            assert "cache_hit_rate" in report["data_quality"]


class TestDatabaseSchemaAndMigration:
    """Test database schema initialization and data migration"""
    
    @pytest.mark.asyncio
    async def test_schema_initialization_message_format(self):
        """Test that schema initialization creates proper A2A message"""
        
        with patch('app.a2a.core.ai_decision_database_integration.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"overall_status": "SUCCESS"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            from app.a2a.core.ai_decision_database_integration import initialize_ai_decision_database_schema
            
            success = await initialize_ai_decision_database_schema("http://localhost:8000/data-manager")
            
            assert success is True
            
            # Verify message structure
            call_args = mock_client.post.call_args
            message_data = call_args[1]["json"]
            
            assert message_data["role"] == "system"
            assert len(message_data["parts"]) >= 1
            
            # Find SQL execution part
            sql_part = None
            for part in message_data["parts"]:
                if part["kind"] == "data" and "sql" in part["data"]:
                    sql_part = part["data"]
                    break
            
            assert sql_part is not None
            assert sql_part["operation"] == "EXECUTE_SQL"
            assert sql_part["storage_type"] == "HANA"
            assert "CREATE TABLE ai_decisions" in sql_part["sql"]
            assert "CREATE TABLE ai_decision_outcomes" in sql_part["sql"]
            assert "CREATE TABLE ai_learned_patterns" in sql_part["sql"]
    
    @pytest.mark.asyncio
    async def test_json_to_database_migration(self):
        """Test migration from JSON files to database"""
        
        import tempfile
        import json as json_lib
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSON data
            decisions_data = {
                "decision_1": {
                    "decision_type": "advisor_guidance",
                    "question": "How to standardize financial data?",
                    "ai_response": {"answer": "Use standard formats"},
                    "context": {"domain": "finance"},
                    "confidence_score": 0.85,
                    "response_time": 1.2
                },
                "decision_2": {
                    "decision_type": "help_request", 
                    "question": "Need help with data validation",
                    "ai_response": {"response": "Check data integrity"},
                    "context": {"urgent": True},
                    "confidence_score": 0.7,
                    "response_time": 0.8
                }
            }
            
            outcomes_data = {
                "decision_1": {
                    "outcome_status": "success",
                    "success_metrics": {"data_standardized": True},
                    "failure_reason": None,
                    "feedback": "Worked well"
                },
                "decision_2": {
                    "outcome_status": "partial_success",
                    "success_metrics": {"validation_improved": True},
                    "failure_reason": None,
                    "feedback": "Helped but not complete"
                }
            }
            
            # Write JSON files
            with open(f"{temp_dir}/decisions.json", "w") as f:
                json_lib.dump(decisions_data, f)
            
            with open(f"{temp_dir}/outcomes.json", "w") as f:
                json_lib.dump(outcomes_data, f)
            
            # Mock database logger for migration
            with patch('app.a2a.core.ai_decision_database_integration.AIDecisionDatabaseLogger') as mock_logger_class:
                mock_logger = AsyncMock()
                mock_logger.log_decision = AsyncMock(side_effect=["migrated_1", "migrated_2"])
                mock_logger.log_outcome = AsyncMock(return_value=True)
                mock_logger.shutdown = AsyncMock()
                mock_logger_class.return_value = mock_logger
                
                from app.a2a.core.ai_decision_database_integration import migrate_json_data_to_database
                
                # Run migration
                result = await migrate_json_data_to_database(
                    temp_dir,
                    "http://localhost:8000/data-manager",
                    "test_agent"
                )
                
                # Verify migration results
                assert result["decisions_migrated"] == 2
                assert result["outcomes_migrated"] == 2
                assert len(result["errors"]) == 0
                
                # Verify database logger calls
                assert mock_logger.log_decision.call_count == 2
                assert mock_logger.log_outcome.call_count == 2
                
                # Verify decision data was passed correctly
                decision_calls = mock_logger.log_decision.call_args_list
                
                # Check first decision
                first_call = decision_calls[0]
                assert first_call[1]["decision_type"].value == "advisor_guidance"
                assert first_call[1]["question"] == "How to standardize financial data?"
                assert first_call[1]["confidence_score"] == 0.85
                
                # Check second decision
                second_call = decision_calls[1]
                assert second_call[1]["decision_type"].value == "help_request"
                assert second_call[1]["question"] == "Need help with data validation"
                assert second_call[1]["confidence_score"] == 0.7


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
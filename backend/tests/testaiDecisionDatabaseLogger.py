# Tests for AI Decision Database Logger
# Comprehensive test suite for AI decision database logging capabilities

import json
import pytest
import pytest_asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from app.a2a.core.aiDecisionLoggerDatabase import (
    AIDecisionDatabaseLogger
)

try:
    from app.a2a.core.aiDecisionDatabaseIntegration import (
        DatabaseAIDecisionIntegration,
        AIDatabaseDecisionIntegrationMixin
    )
except ImportError:
    DatabaseAIDecisionIntegration = None
    AIDatabaseDecisionIntegrationMixin = None

# Import types and enums from the regular logger for compatibility
from app.a2a.core.aiDecisionLogger import (
    DecisionType, 
    OutcomeStatus
)
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole


# -------------------------
# Fixtures (module level)
# -------------------------

@pytest.fixture
def mock_data_manager_response():
    """Mock Data Manager successful response"""
    return {
        "overall_status": "SUCCESS",
        "primary_result": {
            "status": "SUCCESS",
            "location": {
                "storage_type": "hana",
                "database": "HANA",
                "schema": "PUBLIC",
                "table": "ai_decisions",
                "row_count": 1
            }
        },
        "data": [
            {
                "decision_id": "test_id",
                "agent_id": "test_agent",
                "decision_type": "advisor_guidance",
                "timestamp": "2024-01-01T12:00:00",
                "question": "Test question",
                "context": "{}",
                "ai_response": "{}",
                "confidence_score": 0.8,
                "response_time": 0.5
            }
        ]
    }

@pytest.fixture
def mock_http_client(mock_data_manager_response):
    """Mock HTTP client for Data Manager communication"""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_data_manager_response
    mock_client.post.return_value = mock_response
    return mock_client

@pytest_asyncio.fixture
async def database_logger(mock_http_client):
    """Create database logger with mocked HTTP client"""
    with patch('app.a2a.core.ai_decision_logger_database.httpx.AsyncClient') as mock_client_class:
        mock_client_class.return_value = mock_http_client
        logger = AIDecisionDatabaseLogger(
            agent_id="test_agent",
            data_manager_url="http://localhost:8000/data-manager",
            memory_size=100,
            learning_threshold=3
        )
        # Replace the HTTP client with our mock
        logger.http_client = mock_http_client

        # Don't start background tasks in tests to avoid interference with mocks
        # Tests can call specific methods directly
        yield logger
        await logger.shutdown()


class TestAIDecisionDatabaseLogger:
    """Test cases for Database-backed AI Decision Logger"""

    @pytest.mark.asyncio
    async def test_log_decision_to_database(self, database_logger, mock_http_client):
        """Test logging decision to database via Data Manager"""
        
        decision_id = await database_logger.log_decision(
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            question="How do I process financial data?",
            ai_response={"answer": "Use standardized formats"},
            context={"user": "test", "domain": "finance"},
            confidence_score=0.8,
            response_time=0.5
        )
        
        # Verify decision was logged
        assert decision_id
        assert decision_id in database_logger._decision_cache
        
        # Verify Data Manager was called
        mock_http_client.post.assert_called()
        call_args = mock_http_client.post.call_args
        assert "data-manager/process" in call_args[0][0]  # URL is first positional arg
        
        # Verify message structure
        message_data = call_args[1]["json"]
        assert message_data["role"] == "agent"
        assert len(message_data["parts"]) >= 2
        
        # Find data part
        data_part = None
        for part in message_data["parts"]:
            if part["kind"] == "data" and "operation" in part["data"]:
                data_part = part["data"]
                break
        
        assert data_part is not None
        assert data_part["operation"] == "CREATE"
        assert data_part["path"] == "ai_decisions"
        assert data_part["data"]["decision_id"] == decision_id
        assert data_part["data"]["agent_id"] == "test_agent"
        assert data_part["data"]["decision_type"] == "advisor_guidance"
    
    @pytest.mark.asyncio
    async def test_log_outcome_to_database(self, database_logger, mock_http_client): 
        """Test logging outcome to database"""
        
        # First log a decision
        decision_id = await database_logger.log_decision(
            decision_type=DecisionType.HELP_REQUEST,
            question="Need help with data integrity",
            ai_response={"advisor_response": {"answer": "Check checksums"}}
        )
        
        # Reset mock to track outcome call
        mock_http_client.reset_mock()
        
        # Log outcome
        success = await database_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=OutcomeStatus.SUCCESS,
            success_metrics={"problem_solved": True, "time_to_resolve": 30.0},
            feedback="Very helpful advice"
        )
        
        assert success
        assert decision_id in database_logger._outcome_cache
        
        # Verify Data Manager was called for outcome
        mock_http_client.post.assert_called()
        call_args = mock_http_client.post.call_args
        message_data = call_args[1]["json"]
        
        # Find outcome data part
        data_part = None
        for part in message_data["parts"]:
            if part["kind"] == "data" and "operation" in part["data"]:
                data_part = part["data"]
                break
        
        assert data_part["operation"] == "CREATE"
        assert data_part["path"] == "ai_decision_outcomes"
        assert data_part["data"]["decision_id"] == decision_id
        assert data_part["data"]["outcome_status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_recommendations_from_database(self, database_logger, mock_http_client): 
        """Test getting recommendations from database patterns"""
        
        # Mock patterns response
        patterns_response = {
            "overall_status": "SUCCESS",
            "data": [
                {
                    "pattern_id": "pattern_1",
                    "pattern_type": "advisor_guidance",
                    "confidence": 0.9,
                    "recommendations": json.dumps([
                        "Use high confidence thresholds",
                        "Validate responses with multiple sources"
                    ])
                }
            ]
        }
        
        # Create a comprehensive mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = patterns_response
        mock_http_client.post.return_value = mock_response
        
        recommendations = await database_logger.get_recommendations(
            DecisionType.ADVISOR_GUIDANCE,
            {"complexity": "high"}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 2
        assert "Use high confidence thresholds" in recommendations
        assert "Validate responses with multiple sources" in recommendations
        
        # Verify query was made to patterns table
        call_args = mock_http_client.post.call_args
        message_data = call_args[1]["json"]
        
        data_part = None
        for part in message_data["parts"]:
            if part["kind"] == "data" and "query" in part["data"]:
                data_part = part["data"]
                break
        
        assert data_part["operation"] == "READ"
        assert data_part["query"]["table"] == "ai_learned_patterns"
    
    @pytest.mark.asyncio
    async def test_get_analytics_from_database(self, database_logger, mock_http_client): 
        """Test getting analytics from database"""
        
        # Mock analytics response
        analytics_response = {
            "overall_status": "SUCCESS",
            "data": [
                {
                    "agent_id": "test_agent",
                    "decision_type": "advisor_guidance",
                    "total_decisions": 10,
                    "successful_outcomes": 8,
                    "decision_date": "2024-01-01"
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = analytics_response
        mock_http_client.post.return_value = mock_response
        
        analytics = await database_logger.get_decision_analytics()
        
        assert "summary" in analytics
        assert analytics["summary"]["total_decisions"] == 10
        assert analytics["summary"]["successful_outcomes"] == 8
        assert analytics["summary"]["overall_success_rate"] == 0.8
        
        # Verify query was made to analytics view
        call_args = mock_http_client.post.call_args
        message_data = call_args[1]["json"]
        
        data_part = None
        for part in message_data["parts"]:
            if part["kind"] == "data" and "query" in part["data"]:
                data_part = part["data"]
                break
        
        assert data_part["query"]["table"] == "ai_global_analytics"
    
    @pytest.mark.asyncio
    async def test_decision_history_from_database(self, database_logger, mock_http_client): 
        """Test getting decision history from database"""
        
        # Mock history response
        history_response = {
            "overall_status": "SUCCESS",
            "data": [
                {
                    "decision_id": "dec_1",
                    "decision_type": "advisor_guidance",
                    "question": "Test question 1",
                    "confidence_score": 0.8,
                    "outcome_status": "success",
                    "timestamp": "2024-01-01T12:00:00",
                    "success_score": 1.0
                },
                {
                    "decision_id": "dec_2",
                    "decision_type": "help_request",
                    "question": "Test question 2",
                    "confidence_score": 0.6,
                    "outcome_status": "failure",
                    "timestamp": "2024-01-01T11:00:00",
                    "success_score": 0.0
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = history_response
        mock_http_client.post.return_value = mock_response
        
        history = await database_logger.get_decision_history(
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            limit=5
        )
        
        assert len(history) == 2
        assert history[0]["decision_id"] == "dec_1"
        assert history[0]["type"] == "advisor_guidance"
        assert history[0]["success_score"] == 1.0
        
        # Verify query was made with correct filters
        call_args = mock_http_client.post.call_args
        message_data = call_args[1]["json"]
        
        data_part = None
        for part in message_data["parts"]:
            if part["kind"] == "data" and "query" in part["data"]:
                data_part = part["data"]
                break
        
        assert data_part["query"]["table"] == "ai_decision_performance"
        assert data_part["query"]["where"]["agent_id"] == "test_agent"
        assert data_part["query"]["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_export_insights_report(self, database_logger, mock_http_client): 
        """Test exporting comprehensive insights report"""
        
        # Mock multiple responses for different queries
        responses = [
            # Analytics response
            {
                "overall_status": "SUCCESS",
                "data": [{"agent_id": "test_agent", "total_decisions": 5, "successful_outcomes": 4}]
            },
            # Pattern effectiveness response
            {
                "overall_status": "SUCCESS",
                "data": [
                    {
                        "pattern_type": "high_confidence_success",
                        "description": "High confidence decisions succeed more",
                        "confidence": 0.85,
                        "evidence_count": 10,
                        "actual_success_rate": 0.9,
                        "recommendations": json.dumps(["Maintain high confidence", "Validate results"])
                    }
                ]
            }
        ]
        
        # Create proper mock responses
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = responses[0]
        
        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = responses[1]
        
        mock_http_client.post.side_effect = [mock_response_1, mock_response_2]
        
        report = await database_logger.export_insights_report()
        
        assert report["agent_id"] == "test_agent"
        assert "report_timestamp" in report
        assert "summary" in report
        assert "learned_insights" in report
        assert "data_quality" in report
        
        # Check learned insights
        assert len(report["learned_insights"]) == 1
        insight = report["learned_insights"][0]
        assert insight["pattern"] == "high_confidence_success"
        assert insight["confidence"] == 0.85
        assert insight["effectiveness"] == 0.9
        assert len(insight["recommendations"]) == 2
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_and_storage(self, database_logger, mock_http_client): 
        """Test pattern analysis and storage to database"""
        
        # Mock history for pattern analysis
        history_response = {
            "overall_status": "SUCCESS",
            "data": [
                {
                    "decision_type": "advisor_guidance",
                    "confidence_score": 0.8,
                    "success_score": 1.0
                } for _ in range(5)  # 5 successful high-confidence decisions
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = history_response
        mock_http_client.post.return_value = mock_response
        
        # Trigger pattern analysis
        await database_logger._analyze_patterns()
        
        # Should have made calls to read history and store patterns
        assert mock_http_client.post.call_count >= 2
        
        # Check that pattern storage was attempted
        calls = mock_http_client.post.call_args_list
        pattern_store_call = None
        
        for call in calls:
            message_data = call[1]["json"]
            for part in message_data["parts"]:
                if (part["kind"] == "data" and 
                    part["data"].get("operation") == "CREATE" and
                    part["data"].get("path") == "ai_learned_patterns"):
                    pattern_store_call = part["data"]
                    break
        
        if pattern_store_call:
            assert pattern_store_call["data"]["agent_id"] == "test_agent"
            assert "high_success" in pattern_store_call["data"]["pattern_type"]
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, database_logger):
        """Test in-memory cache functionality"""
        
        # Add decisions to cache
        decision_id = "test_cache_decision"
        database_logger._decision_cache[decision_id] = MagicMock()
        database_logger._cache_timestamps[f"decision_{decision_id}"] = time.time() - 400  # Old timestamp
        
        # Add outcome to cache
        database_logger._outcome_cache[decision_id] = MagicMock()
        database_logger._cache_timestamps[f"outcome_{decision_id}"] = time.time() - 400  # Old timestamp
        
        # Trigger cache cleanup
        await database_logger._cleanup_cache()
        
        # Old entries should be removed
        assert decision_id not in database_logger._decision_cache
        assert decision_id not in database_logger._outcome_cache
        assert f"decision_{decision_id}" not in database_logger._cache_timestamps
        assert f"outcome_{decision_id}" not in database_logger._cache_timestamps
    
    @pytest.mark.asyncio
    async def test_database_communication_failure(self, database_logger, mock_http_client): 
        """Test handling of database communication failures"""
        
        # Mock failed response
        mock_http_client.post.return_value.status_code = 500
        mock_http_client.post.return_value.text = "Internal Server Error"
        
        # Should fall back to cache-only operation
        decision_id = await database_logger.log_decision(
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            question="Test with failure",
            ai_response={"answer": "Test"}
        )
        
        # Decision should still be in cache
        assert decision_id in database_logger._decision_cache
        
        # Analytics should fall back to memory
        analytics = await database_logger.get_decision_analytics()
        assert "summary" in analytics  # Should return memory-based analytics


class TestAIDecisionProtocols:
    """Test A2A message protocols for AI decision operations"""
    
    def test_log_decision_request_creation(self):
        """Test creating log decision request"""
        request = LogDecisionRequest.create(
            agent_id="test_agent",
            decision_id="test_decision",
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            question="Test question",
            ai_response={"answer": "Test answer"},
            context={"test": True},
            confidence_score=0.8,
            response_time=0.5
        )
        
        assert request.agent_id == "test_agent"
        assert request.decision_data["decision_id"] == "test_decision"
        assert request.decision_data["decision_type"] == "advisor_guidance"
        assert request.decision_data["question"] == "Test question"
        assert request.decision_data["confidence_score"] == 0.8
    
    def test_log_outcome_request_creation(self):
        """Test creating log outcome request"""
        request = LogOutcomeRequest.create(
            agent_id="test_agent",
            decision_id="test_decision",
            outcome_status=OutcomeStatus.SUCCESS,
            success_metrics={"solved": True},
            feedback="Great result"
        )
        
        assert request.agent_id == "test_agent"
        assert request.outcome_data["decision_id"] == "test_decision"
        assert request.outcome_data["outcome_status"] == "success"
        assert request.outcome_data["success_metrics"]["solved"] is True
        assert request.outcome_data["feedback"] == "Great result"
    
    def test_create_decision_log_message(self):
        """Test creating A2A message for decision logging"""
        message = create_decision_log_message(
            agent_id="test_agent",
            decision_id="test_decision",
            decision_type=DecisionType.HELP_REQUEST,
            question="Need help",
            ai_response={"response": "Here to help"},
            context={"urgent": True},
            confidence_score=0.7,
            response_time=1.2
        )
        
        assert isinstance(message, A2AMessage)
        assert message.role == MessageRole.AGENT
        assert len(message.parts) >= 2
        
        # Find data part with operation
        data_part = None
        for part in message.parts:
            if part.kind == "data" and "operation" in part.data:
                data_part = part.data
                break
        
        assert data_part is not None
        assert data_part["operation"] == "CREATE"
        assert data_part["path"] == "ai_decisions"
        assert data_part["data"]["decision_id"] == "test_decision"
    
    def test_create_outcome_log_message(self):
        """Test creating A2A message for outcome logging"""
        message = create_outcome_log_message(
            agent_id="test_agent",
            decision_id="test_decision",
            outcome_status=OutcomeStatus.PARTIAL_SUCCESS,
            success_metrics={"partially_solved": True},
            failure_reason=None
        )
        
        assert isinstance(message, A2AMessage)
        assert message.role == MessageRole.AGENT
        
        # Find outcome data part
        data_part = None
        for part in message.parts:
            if part.kind == "data" and "operation" in part.data:
                data_part = part.data
                break
        
        assert data_part["operation"] == "CREATE"
        assert data_part["path"] == "ai_decision_outcomes"
        assert data_part["data"]["outcome_status"] == "partial_success"


class TestDatabaseIntegrationMixin:
    """Test database integration mixin"""
    
    class TestAgent(AIDatabaseDecisionIntegrationMixin):
        """Test agent with database integration"""
        
        def __init__(self):
            self.agent_id = "test_agent"
            self.base_url = "http://localhost:8000/agents/test"
            super().__init__()
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent with mocked database logger"""
        with patch('app.a2a.core.ai_decision_database_integration.AIDecisionDatabaseLogger') as mock_logger_class:
            mock_logger = AsyncMock()
            mock_logger_class.return_value = mock_logger
            
            agent = self.TestAgent()
            agent.ai_decision_logger = mock_logger
            
            return agent
    
    @pytest.mark.asyncio
    async def test_enhanced_advisor_request_handler(self, test_agent):
        """Test enhanced advisor request handler with database logging"""
        
        # Mock AI advisor
        test_agent.ai_advisor = AsyncMock()
        test_agent.ai_advisor.process_a2a_help_message = AsyncMock(
            return_value={"answer": "Helpful response", "confidence": 0.8}
        )
        
        # Mock database logger methods
        test_agent.ai_decision_logger.log_decision = AsyncMock(return_value="decision_123")
        test_agent.ai_decision_logger.log_outcome = AsyncMock(return_value=True)
        test_agent.ai_decision_logger.get_recommendations = AsyncMock(return_value=["Recommendation 1"])
        test_agent.ai_decision_logger._decision_cache = {"decision_123": MagicMock()}
        
        # Create test message
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(kind="text", text="How do I solve this problem?"),
                MessagePart(kind="data", data={"advisor_request": True})
            ]
        )
        
        # Process message
        response = await test_agent._enhanced_handle_advisor_request(message, "test_context")
        
        # Verify response structure
        assert response["message_type"] == "advisor_response"
        assert "decision_metadata" in response
        assert response["decision_metadata"]["database_backed"] is True
        assert response["decision_metadata"]["learning_active"] is True
        
        # Verify database calls were made
        test_agent.ai_decision_logger.log_decision.assert_called_once()
        test_agent.ai_decision_logger.log_outcome.assert_called_once()
        test_agent.ai_decision_logger.get_recommendations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_planning_logging(self, test_agent):
        """Test task planning decision logging"""
        
        test_agent.ai_decision_logger.log_decision = AsyncMock(return_value="task_decision_123")
        test_agent.ai_decision_logger.log_outcome = AsyncMock(return_value=True)
        
        # Log task planning decision
        decision_id = await test_agent._log_task_planning_decision(
            "Process financial data",
            {"task_id": "task_123", "estimated_duration": 120.0}
        )
        
        assert decision_id == "task_decision_123"
        
        # Verify decision was logged with correct type
        call_args = test_agent.ai_decision_logger.log_decision.call_args
        assert call_args[1]["decision_type"] == DecisionType.TASK_PLANNING
        assert "Process financial data" in call_args[1]["question"]
        
        # Update task outcome
        await test_agent._update_task_outcome(decision_id, True, 115.0)
        
        # Verify outcome was updated
        outcome_call_args = test_agent.ai_decision_logger.log_outcome.call_args_list[-1]
        assert outcome_call_args[1]["decision_id"] == decision_id
        assert outcome_call_args[1]["outcome_status"] == OutcomeStatus.SUCCESS
        assert outcome_call_args[1]["success_metrics"]["task_completed"] is True
    
    @pytest.mark.asyncio
    async def test_get_cross_agent_insights(self, test_agent):
        """Test getting cross-agent insights"""
        
        with patch('app.a2a.core.ai_decision_database_integration.get_global_database_decision_registry') as mock_registry:
            mock_registry.return_value.get_global_insights = AsyncMock(
                return_value={
                    "total_agents": 3,
                    "total_decisions": 150,
                    "global_success_rate": 0.85
                }
            )
            
            insights = await test_agent.get_cross_agent_insights()
            
            assert insights["total_agents"] == 3
            assert insights["total_decisions"] == 150
            assert insights["global_success_rate"] == 0.85


class TestUtilityFunctions:
    """Test utility functions for database AI decision logging"""
    
    @pytest.mark.asyncio
    async def test_initialize_schema(self):
        """Test database schema initialization"""
        
        with patch('app.a2a.core.ai_decision_database_integration.httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"overall_status": "SUCCESS"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            success = await initialize_ai_decision_database_schema("http://localhost:8000/data-manager")
            
            assert success is True
            mock_client.post.assert_called_once()
            
            # Verify SQL schema was sent
            call_args = mock_client.post.call_args
            message_data = call_args[1]["json"]
            
            data_part = None
            for part in message_data["parts"]:
                if part["kind"] == "data" and "sql" in part["data"]:
                    data_part = part["data"]
                    break
            
            assert data_part is not None
            assert data_part["operation"] == "EXECUTE_SQL" 
            assert "CREATE TABLE ai_decisions" in data_part["sql"]
    
    @pytest.mark.asyncio
    async def test_migration_from_json(self):
        """Test migration from JSON files to database"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock JSON files
            decisions_data = {
                "dec_1": {
                    "decision_type": "advisor_guidance",
                    "question": "Test question",
                    "ai_response": {"answer": "Test answer"},
                    "context": {"test": True},
                    "confidence_score": 0.8,
                    "response_time": 0.5
                }
            }
            
            outcomes_data = {
                "dec_1": {
                    "outcome_status": "success",
                    "success_metrics": {"solved": True},
                    "failure_reason": None,
                    "feedback": "Good result"
                }
            }
            
            import json as json_lib
            with open(f"{temp_dir}/decisions.json", "w") as f:
                json_lib.dump(decisions_data, f)
            
            with open(f"{temp_dir}/outcomes.json", "w") as f:
                json_lib.dump(outcomes_data, f)
            
            # Mock database logger
            with patch('app.a2a.core.ai_decision_database_integration.AIDecisionDatabaseLogger') as mock_logger_class:
                mock_logger = AsyncMock()
                mock_logger.log_decision = AsyncMock(return_value="migrated_dec_1")
                mock_logger.log_outcome = AsyncMock(return_value=True)
                mock_logger.shutdown = AsyncMock()
                mock_logger_class.return_value = mock_logger
                
                # Run migration
                result = await migrate_json_data_to_database(
                    temp_dir,
                    "http://localhost:8000/data-manager",
                    "test_agent"
                )
                
                assert result["decisions_migrated"] == 1
                assert result["outcomes_migrated"] == 1
                assert len(result["errors"]) == 0
                
                # Verify database logger was used
                mock_logger.log_decision.assert_called_once()
                mock_logger.log_outcome.assert_called_once()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
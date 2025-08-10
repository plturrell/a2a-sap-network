# Integration test for Database AI Decision Logger across all agents
# Tests that all agents have been successfully upgraded to use database-backed logging

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from app.a2a.core.aiDecisionLoggerDatabase import AIDecisionDatabaseLogger, get_global_database_decision_registry


class TestAllAgentsDatabaseIntegration:
    """Test that all agents have been upgraded to use Database AI Decision Logger"""
    
    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for Data Manager communication"""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "overall_status": "SUCCESS",
            "primary_result": {"status": "SUCCESS"}
        }
        mock_client.post.return_value = mock_response
        return mock_client
    
    @pytest.mark.asyncio
    async def test_data_standardization_agent_database_logger(self, mock_http_client):
        """Test Data Standardization Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == agent.agent_id
            assert "data-manager" in agent.ai_decision_logger.data_manager_url
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_data_manager_agent_database_logger(self, mock_http_client):
        """Test Data Manager Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataManagerAgent import DataManagerAgent
            
            agent = DataManagerAgent(
                base_url="http://localhost:8000", 
                ord_registry_url="http://localhost:8001"
            )
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == "data_manager_agent"
            # Data Manager uses self-reference for URL
            assert agent.ai_decision_logger.data_manager_url == agent.base_url
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_data_product_agent_database_logger(self, mock_http_client):
        """Test Data Product Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataProductAgent import DataProductRegistrationAgent
            
            agent = DataProductRegistrationAgent(
                base_url="http://localhost:8002",
                ord_registry_url="http://localhost:8001"
            )
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == "data_product_agent_0"
            assert agent.ai_decision_logger.learning_threshold == 8  # Custom threshold for data product decisions
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_catalog_manager_agent_database_logger(self, mock_http_client):
        """Test Catalog Manager Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.catalogManagerAgent import CatalogManagerAgent
            
            agent = CatalogManagerAgent(
                base_url="http://localhost:8003",
                ord_registry_url="http://localhost:8001"
            )
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == "catalog_manager_agent"
            assert agent.ai_decision_logger.learning_threshold == 7  # Custom threshold for catalog decisions
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_agent_manager_agent_database_logger(self, mock_http_client):
        """Test Agent Manager Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.agentManagerAgent import AgentManagerAgent
            
            agent = AgentManagerAgent(base_url="http://localhost:8004")
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == "agent_manager"
            assert agent.ai_decision_logger.memory_size == 1500  # Higher memory for ecosystem management
            assert agent.ai_decision_logger.learning_threshold == 12  # Higher threshold for complexity
            assert agent.ai_decision_logger.cache_ttl == 600  # Longer cache for management decisions
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_vector_processing_agent_database_logger(self, mock_http_client):
        """Test Vector Processing Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.vectorProcessingAgent import VectorProcessingAgent
            
            agent = VectorProcessingAgent(
                base_url="http://localhost:8005",
                ord_registry_url="http://localhost:8001"
            )
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == "vector_processing_agent"
            assert agent.ai_decision_logger.memory_size == 800  # Moderate memory for vector processing
            assert agent.ai_decision_logger.learning_threshold == 6  # Lower threshold for vector operations
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_ai_preparation_agent_database_logger(self, mock_http_client):
        """Test AI Preparation Agent has database logger"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.aiPreparationAgent import AIPreparationAgent
            
            agent = AIPreparationAgent(
                base_url="http://localhost:8006",
                ord_registry_url="http://localhost:8001"
            )
            
            # Verify database logger integration
            assert hasattr(agent, 'ai_decision_logger')
            assert isinstance(agent.ai_decision_logger, AIDecisionDatabaseLogger)
            assert agent.ai_decision_logger.agent_id == agent.agent_id
            assert agent.ai_decision_logger.memory_size == 600  # Moderate memory for AI preparation
            assert agent.ai_decision_logger.learning_threshold == 5  # Lower threshold for AI-focused learning
            
            # Verify global registration
            registry = get_global_database_decision_registry()
            assert agent.agent_id in registry.agent_loggers
    
    @pytest.mark.asyncio
    async def test_global_registry_cross_agent_integration(self, mock_http_client):
        """Test that all agents are registered in global registry and can share insights"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            # Create multiple agents
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            from app.a2a.agents.dataManagerAgent import DataManagerAgent
            from app.a2a.agents.catalogManagerAgent import CatalogManagerAgent
            
            agents = [
                FinancialStandardizationAgent(base_url="http://localhost:8001"),
                DataManagerAgent(base_url="http://localhost:8000", ord_registry_url="http://localhost:8001"),
                CatalogManagerAgent(base_url="http://localhost:8003", ord_registry_url="http://localhost:8001")
            ]
            
            # Verify all agents are in global registry
            registry = get_global_database_decision_registry()
            
            for agent in agents:
                assert agent.agent_id in registry.agent_loggers
                assert isinstance(registry.agent_loggers[agent.agent_id], AIDecisionDatabaseLogger)
            
            # Test cross-agent insights
            assert len(registry.agent_loggers) >= 3
            
            # Mock analytics responses for insights
            analytics_responses = []
            for agent in agents:
                agent.ai_decision_logger.get_decision_analytics = AsyncMock(
                    return_value={
                        "summary": {"total_decisions": 10, "successful_outcomes": 8}
                    }
                )
            
            # Get global insights
            insights = await registry.get_global_insights()
            
            assert insights["total_agents"] >= 3
            assert "agent_performance" in insights
            assert len(insights["agent_performance"]) >= 3
    
    @pytest.mark.asyncio 
    async def test_agent_specific_configurations(self, mock_http_client):
        """Test that agents have appropriate configurations for their roles"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            # Test different agent configurations
            test_cases = [
                {
                    "agent_class": "FinancialStandardizationAgent",
                    "module": "app.a2a.agents.dataStandardizationAgent",
                    "init_args": {"base_url": "http://localhost:8001"},
                    "expected_memory": 1000,
                    "expected_threshold": 5,
                    "expected_cache_ttl": 300
                },
                {
                    "agent_class": "DataManagerAgent", 
                    "module": "app.a2a.agents.dataManagerAgent",
                    "init_args": {"base_url": "http://localhost:8000", "ord_registry_url": "http://localhost:8001"},
                    "expected_memory": 1000,
                    "expected_threshold": 10,
                    "expected_cache_ttl": 300
                },
                {
                    "agent_class": "AgentManagerAgent",
                    "module": "app.a2a.agents.agentManagerAgent", 
                    "init_args": {"base_url": "http://localhost:8004"},
                    "expected_memory": 1500,  # Higher for ecosystem management
                    "expected_threshold": 12,  # Higher threshold for complexity
                    "expected_cache_ttl": 600   # Longer cache for management decisions
                },
                {
                    "agent_class": "VectorProcessingAgent",
                    "module": "app.a2a.agents.vectorProcessingAgent",
                    "init_args": {"base_url": "http://localhost:8005", "ord_registry_url": "http://localhost:8001"},
                    "expected_memory": 800,    # Moderate for vector processing
                    "expected_threshold": 6,   # Lower threshold for vector operations
                    "expected_cache_ttl": 300
                }
            ]
            
            for test_case in test_cases:
                # Dynamically import and create agent
                module = __import__(test_case["module"], fromlist=[test_case["agent_class"]])
                agent_class = getattr(module, test_case["agent_class"])
                agent = agent_class(**test_case["init_args"])
                
                # Verify configuration
                logger = agent.ai_decision_logger
                assert logger.memory_size == test_case["expected_memory"]
                assert logger.learning_threshold == test_case["expected_threshold"]
                assert logger.cache_ttl == test_case["expected_cache_ttl"]
                
                # Verify data manager URL is properly constructed
                assert "data-manager" in logger.data_manager_url
    
    @pytest.mark.asyncio
    async def test_all_agents_decision_logging_capability(self, mock_http_client):
        """Test that all agents can log decisions to database"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            from app.a2a.core.aiDecisionLogger import DecisionType, OutcomeStatus
            
            # Test with one representative agent
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            
            # Test decision logging
            decision_id = await agent.ai_decision_logger.log_decision(
                decision_type=DecisionType.ADVISOR_GUIDANCE,
                question="Test cross-agent decision logging",
                ai_response={"answer": "Database logging works across all agents"},
                context={"test": "cross_agent_integration"}
            )
            
            assert decision_id
            
            # Test outcome logging
            success = await agent.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.SUCCESS,
                success_metrics={"integration_test": True}
            )
            
            assert success
            
            # Verify Data Manager was called
            mock_http_client.post.assert_called()
            
            # Verify message structure contains database operations
            calls = mock_http_client.post.call_args_list
            assert len(calls) >= 2  # Decision log + Outcome log
            
            # Check decision call
            decision_call = calls[0]
            message_data = decision_call[1]["json"]
            
            # Find data part with decision
            found_decision_operation = False
            for part in message_data["parts"]:
                if (part["kind"] == "data" and 
                    part["data"].get("operation") == "CREATE" and
                    part["data"].get("path") == "ai_decisions"):
                    found_decision_operation = True
                    break
            
            assert found_decision_operation, "Decision logging operation not found in Data Manager call"
    
    @pytest.mark.asyncio
    async def test_agent_registry_cleanup(self, mock_http_client):
        """Test registry cleanup and agent lifecycle management"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            
            # Create agent and verify registration
            agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            registry = get_global_database_decision_registry()
            
            assert agent.agent_id in registry.agent_loggers
            
            # Test logger shutdown
            await agent.ai_decision_logger.shutdown()
            
            # Verify background tasks are cancelled
            assert agent.ai_decision_logger._analysis_task.cancelled()
            assert agent.ai_decision_logger._cache_cleanup_task.cancelled()


class TestAgentSpecificFeatures:
    """Test agent-specific AI decision logging features"""
    
    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client with specific responses"""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "overall_status": "SUCCESS",
            "data": [
                {
                    "pattern_type": "data_standardization_high_confidence",
                    "recommendations": '["Use ISO standards", "Validate with checksums"]'
                }
            ]
        }
        mock_client.post.return_value = mock_response
        return mock_client
    
    @pytest.mark.asyncio
    async def test_data_manager_self_reference_logging(self, mock_http_client):
        """Test that Data Manager can log its own decisions"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataManagerAgent import DataManagerAgent
            from app.a2a.core.aiDecisionLogger import DecisionType
            
            agent = DataManagerAgent(
                base_url="http://localhost:8000",
                ord_registry_url="http://localhost:8001"
            )
            
            # Test that Data Manager can log decisions about its own operations
            decision_id = await agent.ai_decision_logger.log_decision(
                decision_type=DecisionType.QUALITY_ASSESSMENT,
                question="Should I store this data in HANA or SQLite?",
                ai_response={"recommendation": "Use HANA for structured data"},
                context={"data_type": "structured", "size": "large"}
            )
            
            assert decision_id
            # Data Manager should use self-reference URL
            assert agent.ai_decision_logger.data_manager_url == agent.base_url
    
    @pytest.mark.asyncio
    async def test_cross_agent_pattern_sharing(self, mock_http_client):
        """Test that patterns learned by one agent can benefit others"""
        
        with patch('app.a2a.core.aiDecisionLoggerDatabase.httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value = mock_http_client
            
            from app.a2a.agents.dataStandardizationAgent import FinancialStandardizationAgent
            from app.a2a.agents.catalogManagerAgent import CatalogManagerAgent
            from app.a2a.core.aiDecisionLogger import DecisionType
            
            # Create two different agents
            standardization_agent = FinancialStandardizationAgent(base_url="http://localhost:8001")
            catalog_agent = CatalogManagerAgent(
                base_url="http://localhost:8003", 
                ord_registry_url="http://localhost:8001"
            )
            
            # Both agents should be able to get recommendations from shared patterns
            std_recommendations = await standardization_agent.ai_decision_logger.get_recommendations(
                DecisionType.ADVISOR_GUIDANCE,
                {"domain": "finance"}
            )
            
            catalog_recommendations = await catalog_agent.ai_decision_logger.get_recommendations(
                DecisionType.ADVISOR_GUIDANCE,
                {"domain": "finance"}
            )
            
            # Both should get some recommendations (from mocked response)
            assert isinstance(std_recommendations, list)
            assert isinstance(catalog_recommendations, list)
            
            # Verify both agents call the same database patterns table
            calls = mock_http_client.post.call_args_list
            pattern_calls = [
                call for call in calls 
                if "ai_learned_patterns" in str(call)
            ]
            
            assert len(pattern_calls) >= 2, "Both agents should query patterns table"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
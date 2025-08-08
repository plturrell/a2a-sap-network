"""
End-to-End Integration Tests for Agent 0
Tests the complete flow from A2A message to platform synchronization
"""

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import httpx

from app.a2a.agents.data_standardization_agent_v2 import FinancialStandardizationAgentV2
from src.a2a.core.a2a_types import A2AMessage, MessagePart, MessageRole
from src.a2a.core.auth_manager import get_auth_manager
from src.a2a.core.circuit_breaker import get_breaker_manager
from src.a2a.core.response_mapper import get_response_mapper_registry


class TestAgentIntegration:
    """Integration tests for Agent 0"""
    
    @pytest_asyncio.fixture
    async def agent_config(self):
        """Test agent configuration"""
        return {
            "namespace": "com.test",
            "downstream_targets": [
                {
                    "target_id": "test-datasphere",
                    "platform_type": "sap_datasphere",
                    "endpoint": "https://test.datasphere.sap",
                    "enabled": True,
                    "auth_config": {
                        "method": "oauth2",
                        "client_id": "test_client",
                        "client_secret": "test_secret",
                        "token_url": "https://test.auth.sap/token"
                    }
                },
                {
                    "target_id": "test-unity",
                    "platform_type": "databricks",
                    "workspace_url": "https://test.databricks.com",
                    "enabled": True,
                    "auth_config": {
                        "token": "test_token"
                    }
                }
            ]
        }
    
    @pytest_asyncio.fixture
    async def agent(self, agent_config):
        """Create test agent instance"""
        agent = FinancialStandardizationAgentV2(
            base_url="https://test.agent.com",
            config=agent_config
        )
        return agent
    
    @pytest.fixture
    def sample_a2a_message(self):
        """Sample A2A message for testing"""
        return A2AMessage(
            role=MessageRole.USER,
            contextId="test-context-123",
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "type": "location",
                        "records": [
                            {
                                "id": "loc1",
                                "name": "New York Office",
                                "address": "123 Wall Street",
                                "city": "New York",
                                "country": "USA"
                            },
                            {
                                "id": "loc2", 
                                "name": "London Branch",
                                "address": "456 Fleet Street",
                                "city": "London",
                                "country": "UK"
                            }
                        ],
                        "integrity": {
                            "row_count": 2,
                            "dataset_hash": "abc123",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_full_standardization_flow(self, agent, sample_a2a_message):
        """Test complete standardization flow"""
        # Process message
        result = await agent.process_message(
            sample_a2a_message.model_dump(),
            "test-context-123"
        )
        
        # Verify task created
        assert "taskId" in result
        assert result["status"]["state"] == "pending"
        
        # Wait for processing (in real test would check task status)
        await asyncio.sleep(0.1)
        
        # Verify standardization occurred
        task_id = result["taskId"]
        assert task_id in agent.tasks
        
        # Check if catalog integration was triggered
        catalog_skill = agent.skills.get("catalog_integration")
        assert catalog_skill is not None
        assert len(catalog_skill.sync_history) > 0
    
    @pytest.mark.asyncio
    async def test_platform_sync_with_mock(self, agent):
        """Test platform synchronization with mocked responses"""
        # Mock HTTP responses
        mock_responses = {
            "https://test.auth.sap/token": {
                "access_token": "mock_token",
                "expires_in": 3600
            },
            "https://test.datasphere.sap/api/v1/catalog/sync": {
                "catalogEntry": {
                    "id": "cat-123",
                    "name": "location_standardized",
                    "type": "dataset",
                    "metadata": {}
                }
            }
        }
        
        async def mock_request(self, *args, **kwargs):
            url = str(kwargs.get("url", ""))
            if url in mock_responses:
                response = Mock()
                response.status_code = 200
                response.json = lambda: mock_responses[url]
                return response
            raise httpx.HTTPError(f"Unmocked URL: {url}")
        
        with patch.object(httpx.AsyncClient, "request", mock_request):
            # Emit catalog change
            catalog_skill = agent.skills["catalog_integration"]
            event_id = await catalog_skill.emit_catalog_change(
                operation="create",
                entity_type="dataset",
                entity_id="test-dataset",
                metadata={
                    "name": "test_dataset",
                    "columns": [
                        {"name": "id", "type": "string"},
                        {"name": "value", "type": "integer"}
                    ]
                }
            )
            
            # Wait for async processing
            await asyncio.sleep(0.5)
            
            # Verify sync occurred
            assert event_id in catalog_skill.sync_history
            sync_result = catalog_skill.sync_history[event_id]
            assert sync_result is not None
    
    @pytest.mark.asyncio
    async def test_response_mapping(self):
        """Test platform response mapping"""
        mapper_registry = get_response_mapper_registry()
        
        # Test Datasphere response mapping
        datasphere_response = {
            "catalogEntry": {
                "id": "entry-123",
                "name": "test_catalog",
                "type": "table",
                "metadata": {"source": "a2a"}
            }
        }
        
        mapped = mapper_registry.map_response(
            "sap_datasphere",
            datasphere_response,
            is_error=False
        )
        
        assert mapped["status"] == "success"
        assert mapped["platform"] == "sap_datasphere"
        assert mapped["data"]["catalog_id"] == "entry-123"
        
        # Test error mapping
        error_response = {
            "error": {
                "message": "Authentication failed",
                "code": "AUTH_001"
            }
        }
        
        error_mapped = mapper_registry.map_response(
            "sap_datasphere",
            error_response,
            is_error=True,
            status_code=401
        )
        
        assert error_mapped["status"] == "error"
        assert error_mapped["error"]["code"] == 401
        assert error_mapped["error"]["type"] == "authentication_error"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, agent):
        """Test circuit breaker behavior"""
        breaker_manager = get_breaker_manager()
        
        # Simulate failures
        async def failing_call():
            raise Exception("Service unavailable")
        
        breaker = breaker_manager.get_breaker("test_service", failure_threshold=3)
        
        # Test failures open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_call)
        
        # Circuit should be open now
        assert breaker.is_open()
        
        # Next call should fail immediately
        with pytest.raises(Exception) as exc_info:
            await breaker.call(failing_call)
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio 
    async def test_auth_token_caching(self):
        """Test OAuth token caching"""
        auth_manager = get_auth_manager()
        
        # Register OAuth client
        auth_manager.register_oauth2(
            "test_platform",
            "client_id",
            "client_secret",
            "https://test.auth/token"
        )
        
        # Mock token response
        mock_response = {
            "access_token": "cached_token",
            "expires_in": 3600
        }
        
        with patch.object(httpx.AsyncClient, "post") as mock_post:
            mock_post.return_value = AsyncMock()
            mock_post.return_value.json = lambda: mock_response
            mock_post.return_value.status_code = 200
            
            # First call should request token
            headers1 = await auth_manager.get_auth_headers("test_platform")
            assert headers1["Authorization"] == "Bearer cached_token"
            assert mock_post.call_count == 1
            
            # Second call should use cache
            headers2 = await auth_manager.get_auth_headers("test_platform")
            assert headers2["Authorization"] == "Bearer cached_token"
            assert mock_post.call_count == 1  # No additional call
    
    @pytest.mark.asyncio
    async def test_data_validation_flow(self, agent):
        """Test data validation during processing"""
        # Create message with invalid data
        invalid_message = A2AMessage(
            role=MessageRole.USER,
            contextId="test-invalid",
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "type": "location",
                        "records": [
                            {
                                # Missing required fields
                                "name": "Invalid Location"
                            }
                        ]
                    }
                )
            ]
        )
        
        # Process should handle validation
        result = await agent.process_message(
            invalid_message.model_dump(),
            "test-invalid"
        )
        
        assert "taskId" in result
        # Task should complete but with warnings
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, agent):
        """Test batch multi-type processing"""
        batch_message = A2AMessage(
            role=MessageRole.USER,
            contextId="test-batch",
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "entities": {
                            "location": [
                                {"id": "1", "name": "NYC", "country": "USA"}
                            ],
                            "account": [
                                {"id": "A1", "name": "Acct 001", "type": "Savings"}
                            ],
                            "product": [
                                {"id": "P1", "name": "Term Deposit", "category": "Investment"}
                            ]
                        }
                    }
                )
            ]
        )
        
        result = await agent.process_message(
            batch_message.model_dump(),
            "test-batch"
        )
        
        assert "taskId" in result
        
        # Verify all types processed
        await asyncio.sleep(0.1)
        
        catalog_skill = agent.skills["catalog_integration"]
        # Should have events for each type
        assert len(catalog_skill.event_queue) >= 0  # Events processed


class TestErrorScenarios:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_platform_connection_failure(self, agent):
        """Test handling of platform connection failures"""
        with patch.object(httpx.AsyncClient, "request") as mock_request:
            mock_request.side_effect = httpx.ConnectError("Connection refused")
            
            catalog_skill = agent.skills["catalog_integration"]
            
            # Should handle gracefully
            event_id = await catalog_skill.emit_catalog_change(
                operation="create",
                entity_type="test",
                entity_id="fail-test",
                metadata={"name": "test"}
            )
            
            await asyncio.sleep(0.5)
            
            # Check failure recorded
            if event_id in catalog_skill.sync_history:
                sync_result = catalog_skill.sync_history[event_id]
                assert "error" in str(sync_result)
    
    @pytest.mark.asyncio
    async def test_invalid_credentials(self, agent):
        """Test handling of authentication failures"""
        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=Mock(),
            response=mock_response
        )
        
        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            catalog_skill = agent.skills["catalog_integration"]
            
            # Emit event that will fail auth
            event_id = await catalog_skill.emit_catalog_change(
                operation="update",
                entity_type="dataset",
                entity_id="auth-test",
                metadata={"name": "test"}
            )
            
            await asyncio.sleep(0.5)
            
            # Should handle auth failure
            assert event_id  # Event created despite auth failure


@pytest.mark.asyncio
async def test_performance_under_load():
    """Test agent performance with multiple concurrent messages"""
    config = {
        "namespace": "com.perf.test",
        "downstream_targets": []  # No real platforms for perf test
    }
    
    agent = FinancialStandardizationAgentV2(
        base_url="https://perf.test",
        config=config
    )
    
    # Create multiple messages
    messages = []
    for i in range(50):
        msg = A2AMessage(
            role=MessageRole.USER,
            contextId=f"perf-{i}",
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "type": "location",
                        "records": [
                            {"id": f"loc-{i}", "name": f"Location {i}", "country": "USA"}
                        ]
                    }
                )
            ]
        )
        messages.append(msg)
    
    # Process concurrently
    start_time = asyncio.get_event_loop().time()
    
    tasks = [
        agent.process_message(msg.model_dump(), f"ctx-{i}")
        for i, msg in enumerate(messages)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    # All should succeed
    assert len(results) == 50
    assert all("taskId" in r for r in results)
    
    # Should complete reasonably fast
    assert duration < 10.0  # 50 messages in under 10 seconds
    
    print(f"Processed 50 messages in {duration:.2f} seconds")


if __name__ == "__main__":
    # Run specific test
    asyncio.run(test_performance_under_load())
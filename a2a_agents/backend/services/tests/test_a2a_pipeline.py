"""
A2A Pipeline Integration Tests
Tests the complete Agent-to-Agent communication flow
"""

import pytest
import httpx
import asyncio
import json
from datetime import datetime
import time


class TestA2APipeline:
    """Test the complete A2A pipeline from Agent 0 to Agent 3"""
    
    @pytest.fixture
    def base_urls(self):
        """A2A agent URLs"""
        return {
            "agent0": "http://localhost:8001",
            "agent1": "http://localhost:8002",
            "agent2": "http://localhost:8003",
            "agent3": "http://localhost:8004",
            "agent_manager": "http://localhost:8007"
        }
    
    @pytest.fixture
    async def http_client(self):
        """Async HTTP client"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            yield client
    
    async def wait_for_agent(self, client, url, max_retries=30):
        """Wait for agent to be ready"""
        for i in range(max_retries):
            try:
                response = await client.get(f"{url}/ready")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ready") and data.get("registered"):
                        return True
            except:
                pass
            await asyncio.sleep(1)
        return False
    
    @pytest.mark.asyncio
    async def test_all_agents_healthy(self, http_client, base_urls):
        """Test that all A2A agents are healthy"""
        for agent_name, url in base_urls.items():
            response = await http_client.get(f"{url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "agent_id" in data
            assert data.get("a2a_protocol") == "0.2.9" or agent_name == "agent_manager"
    
    @pytest.mark.asyncio
    async def test_agent_cards(self, http_client, base_urls):
        """Test A2A agent card endpoints"""
        for agent_name, url in base_urls.items():
            if agent_name == "agent_manager":
                continue  # Agent manager might have different structure
            
            response = await http_client.get(f"{url}/.well-known/agent.json")
            assert response.status_code == 200
            
            card = response.json()
            assert "agent_id" in card
            assert "name" in card
            assert "protocol_version" in card
            assert card["protocol_version"] == "0.2.9"
            assert "capabilities" in card
            assert "endpoints" in card
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, http_client, base_urls):
        """Test that all agents are registered with Agent Manager"""
        # Wait for Agent Manager to be ready
        assert await self.wait_for_agent(http_client, base_urls["agent_manager"])
        
        # Check network status
        response = await http_client.get(f"{base_urls['agent_manager']}/a2a/network/status")
        assert response.status_code == 200
        
        network_status = response.json()
        assert "registered_agents" in network_status
        
        # Verify all agents are registered
        registered_ids = [agent["agent_id"] for agent in network_status["registered_agents"]]
        expected_agents = [
            "data_product_agent_0",
            "data_standardization_agent_1",
            "ai_preparation_agent_2",
            "vector_processing_agent_3"
        ]
        
        for agent_id in expected_agents:
            assert agent_id in registered_ids
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self, http_client, base_urls):
        """Test complete A2A data flow through all agents"""
        # Sample financial data
        test_data = {
            "data_to_standardize": {
                "account": [
                    {
                        "account_number": "ACC001",
                        "account_name": "Test Account",
                        "currency": "USD",
                        "balance": 10000.00
                    }
                ],
                "book": [
                    {
                        "book_id": "BOOK001",
                        "book_name": "Trading Book A",
                        "book_type": "TRADING"
                    }
                ]
            },
            "context_id": f"test-{int(time.time())}"
        }
        
        # Step 1: Submit to Agent 0
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "process_data_product",
            "params": test_data,
            "id": 1
        }
        
        response = await http_client.post(
            f"{base_urls['agent0']}/a2a/agent0/v1/rpc",
            json=rpc_request
        )
        assert response.status_code == 200
        
        result = response.json()
        assert "result" in result
        assert "task_id" in result["result"]
        
        agent0_task_id = result["result"]["task_id"]
        
        # Step 2: Wait for Agent 0 to complete and trigger Agent 1
        await asyncio.sleep(2)
        
        # Check Agent 1 received the data
        status_request = {
            "jsonrpc": "2.0",
            "method": "get_status",
            "params": {"context_id": test_data["context_id"]},
            "id": 2
        }
        
        response = await http_client.post(
            f"{base_urls['agent1']}/a2a/agent1/v1/rpc",
            json=status_request
        )
        
        # Verify data is being processed
        assert response.status_code == 200
        
        # Step 3: Check final status through Agent Manager
        await asyncio.sleep(5)  # Allow time for processing
        
        workflow_response = await http_client.get(
            f"{base_urls['agent_manager']}/a2a/workflows/{test_data['context_id']}"
        )
        
        if workflow_response.status_code == 200:
            workflow_status = workflow_response.json()
            assert workflow_status["status"] in ["completed", "processing"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, http_client, base_urls):
        """Test A2A error handling"""
        # Send invalid data
        invalid_request = {
            "jsonrpc": "2.0",
            "method": "invalid_method",
            "params": {},
            "id": 99
        }
        
        response = await http_client.post(
            f"{base_urls['agent0']}/a2a/agent0/v1/rpc",
            json=invalid_request
        )
        
        assert response.status_code in [404, 400]
        result = response.json()
        assert "error" in result
        assert result["error"]["code"] == -32601  # Method not found
    
    @pytest.mark.asyncio
    async def test_stream_endpoints(self, http_client, base_urls):
        """Test A2A SSE stream endpoints"""
        # Test that stream endpoints exist and return proper headers
        for agent_name in ["agent0", "agent1", "agent2", "agent3"]:
            url = base_urls[agent_name]
            
            # Make a HEAD request to check endpoint exists
            response = await http_client.request(
                "GET",
                f"{url}/a2a/{agent_name}/v1/stream",
                headers={"Accept": "text/event-stream"}
            )
            
            # Stream should return 200 and correct content type
            assert response.status_code == 200
            assert response.headers.get("content-type") == "text/event-stream"
            
            # Close the stream
            response.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
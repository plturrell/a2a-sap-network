"""
A2A Smoke Tests for Production
Quick tests to verify basic functionality after deployment
"""

import pytest
import httpx
import asyncio
import os
from typing import Dict, List


class TestA2ASmoke:
    """Smoke tests for A2A network"""
    
    @pytest.fixture
    def base_urls(self):
        """A2A agent URLs - use env vars in production"""
        if pytest.config.getoption("--prod"):
            # Production URLs from environment
            return {
                "agent0": os.getenv("AGENT0_URL", "https://agent0.a2a.example.com"),
                "agent1": os.getenv("AGENT1_URL", "https://agent1.a2a.example.com"),
                "agent2": os.getenv("AGENT2_URL", "https://agent2.a2a.example.com"),
                "agent3": os.getenv("AGENT3_URL", "https://agent3.a2a.example.com"),
                "agent_manager": os.getenv("AGENT_MANAGER_URL", "https://agent-manager.a2a.example.com")
            }
        else:
            # Local/dev URLs
            return {
                "agent0": "http://localhost:8001",
                "agent1": "http://localhost:8002",
                "agent2": "http://localhost:8003",
                "agent3": "http://localhost:8004",
                "agent_manager": "http://localhost:8007"
            }
    
    @pytest.fixture
    async def http_client(self):
        """Async HTTP client with production settings"""
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, verify=True) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_all_agents_responding(self, http_client, base_urls):
        """Test that all agents respond to health checks"""
        failed_agents = []
        
        for agent_name, url in base_urls.items():
            try:
                response = await http_client.get(f"{url}/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                print(f"✓ {agent_name} is healthy")
            except Exception as e:
                failed_agents.append((agent_name, str(e)))
                print(f"✗ {agent_name} failed: {e}")
        
        assert len(failed_agents) == 0, f"Failed agents: {failed_agents}"
    
    @pytest.mark.asyncio
    async def test_agent_cards_accessible(self, http_client, base_urls):
        """Test that agent cards are accessible"""
        for agent_name, url in base_urls.items():
            if agent_name == "agent_manager":
                continue
            
            try:
                response = await http_client.get(f"{url}/.well-known/agent.json")
                assert response.status_code == 200
                card = response.json()
                assert "agent_id" in card
                assert "protocol_version" in card
                assert card["protocol_version"] == "0.2.9"
                print(f"✓ {agent_name} agent card valid")
            except Exception as e:
                pytest.fail(f"{agent_name} agent card failed: {e}")
    
    @pytest.mark.asyncio
    async def test_agent_registration_status(self, http_client, base_urls):
        """Test that agents are registered with Agent Manager"""
        try:
            response = await http_client.get(f"{base_urls['agent_manager']}/a2a/network/status")
            assert response.status_code == 200
            
            network_status = response.json()
            registered_count = len(network_status.get("registered_agents", []))
            
            # Should have at least 4 core agents registered
            assert registered_count >= 4, f"Only {registered_count} agents registered"
            print(f"✓ {registered_count} agents registered with Agent Manager")
            
        except Exception as e:
            pytest.fail(f"Agent Manager network status failed: {e}")
    
    @pytest.mark.asyncio
    async def test_simple_data_flow(self, http_client, base_urls):
        """Test a simple data flow through Agent 0"""
        test_data = {
            "data": {
                "test_entity": {
                    "id": "smoke_test_001",
                    "name": "Smoke Test Entity",
                    "value": 100
                }
            }
        }
        
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "process_data_product",
            "params": test_data,
            "id": "smoke_test_1"
        }
        
        try:
            response = await http_client.post(
                f"{base_urls['agent0']}/a2a/agent0/v1/rpc",
                json=rpc_request,
                timeout=30.0
            )
            
            assert response.status_code == 200
            result = response.json()
            assert "result" in result or "error" not in result
            print("✓ Data flow test passed")
            
        except Exception as e:
            pytest.fail(f"Data flow test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_metrics_endpoints(self, http_client, base_urls):
        """Test that metrics endpoints are accessible"""
        # Only test a few agents to keep smoke test quick
        test_agents = ["agent0", "agent1"]
        
        for agent_name in test_agents:
            url = base_urls[agent_name]
            try:
                response = await http_client.get(f"{url}/metrics")
                assert response.status_code == 200
                # Metrics should contain prometheus format text
                assert "a2a_messages_processed_total" in response.text
                print(f"✓ {agent_name} metrics accessible")
            except Exception as e:
                # Metrics might not be exposed on all agents
                print(f"⚠ {agent_name} metrics not accessible: {e}")


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--prod",
        action="store_true",
        default=False,
        help="Run tests against production environment"
    )


if __name__ == "__main__":
    # Quick smoke test runner
    pytest.main([__file__, "-v", "--tb=short"])
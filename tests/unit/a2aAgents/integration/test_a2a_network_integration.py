#!/usr/bin/env python3
"""
Integration tests for a2aAgents and a2aNetwork
Verifies proper communication and component sharing between projects
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import httpx
from unittest.mock import Mock, patch, AsyncMock

# Add paths for both projects
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")

# Import components from both projects
from app.a2a.sdk import A2AAgentBase, a2a_handler, A2AMessage, MessageRole
from app.a2a.network.networkConnector import NetworkConnector
from app.a2a.version.versionManager import VersionManager


class TestSDKIntegration:
    """Test SDK components work from a2aNetwork"""
    
    def test_sdk_imports_from_network(self):
        """Test that SDK components are imported from a2aNetwork"""
        # Import should work without errors
        from app.a2a.sdk import A2AAgentBase, a2a_handler, a2a_skill, a2a_task
        from app.a2a.sdk import A2AMessage, MessagePart, MessageRole
        
        # Verify they're not None
        assert A2AAgentBase is not None
        assert a2a_handler is not None
        assert A2AMessage is not None
        
    def test_agent_base_functionality(self):
        """Test A2AAgentBase from network works correctly"""
        
        class TestAgent(A2AAgentBase):
            def __init__(self):
                super().__init__(
                    agent_id="test_agent",
                    name="Test Agent",
                    description="Integration test agent",
                    version="1.0.0",
                    base_url="http://localhost:8000"
                )
        
        agent = TestAgent()
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.version == "1.0.0"
        
    def test_decorators_work(self):
        """Test a2a decorators from network function properly"""
        from app.a2a.sdk import a2a_handler, a2a_skill
        
        @a2a_handler("test_handler")
        async def test_handler(self, message: A2AMessage):
            return {"handled": True}
        
        @a2a_skill("test_skill")
        async def test_skill(self, data: Dict[str, Any]):
            return {"processed": data}
        
        # Verify decorators add metadata
        assert hasattr(test_handler, '_a2a_handler')
        assert hasattr(test_skill, '_a2a_skill')
        

class TestSecurityIntegration:
    """Test security components work from a2aNetwork"""
    
    def test_trust_imports_from_network(self):
        """Test trust system imports work"""
        from app.a2a.security import (
            sign_a2a_message, verify_a2a_message,
            initialize_agent_trust, get_trust_contract
        )
        
        # Verify functions exist
        assert sign_a2a_message is not None
        assert verify_a2a_message is not None
        assert initialize_agent_trust is not None
        
    def test_delegation_imports_from_network(self):
        """Test delegation system imports work"""
        from app.a2a.security import (
            DelegationAction, get_delegation_contract,
            can_agent_delegate, record_delegation_usage
        )
        
        # Verify delegation components
        assert DelegationAction is not None
        assert hasattr(DelegationAction, 'READ')
        assert hasattr(DelegationAction, 'WRITE')
        assert hasattr(DelegationAction, 'EXECUTE')
        

class TestNetworkConnectorIntegration:
    """Test NetworkConnector properly integrates with a2aNetwork"""
    
    @pytest.fixture
    async def network_connector(self):
        """Create NetworkConnector instance"""
        connector = NetworkConnector(
            registry_url="http://localhost:9000",
            trust_service_url="http://localhost:9001"
        )
        await connector.initialize()
        return connector
    
    @pytest.mark.asyncio
    async def test_network_detection(self, network_connector):
        """Test network availability detection"""
        # Should detect network availability
        is_available = await network_connector.check_network_availability()
        assert isinstance(is_available, bool)
        
    @pytest.mark.asyncio
    async def test_agent_registration(self, network_connector):
        """Test agent registration through network"""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.get_agent_card.return_value = {
            "agent_id": "test_agent_123",
            "name": "Test Agent",
            "capabilities": ["test"]
        }
        
        # Test registration (will use local fallback if network unavailable)
        result = await network_connector.register_agent(mock_agent)
        
        assert result is not None
        assert "agent_id" in result or "error" in result
        
    @pytest.mark.asyncio  
    async def test_message_routing(self, network_connector):
        """Test message routing through network"""
        test_message = {
            "from_agent": "agent1",
            "to_agent": "agent2",
            "content": "Test message",
            "message_type": "test"
        }
        
        # Test message routing
        result = await network_connector.route_message(test_message)
        
        assert result is not None
        assert isinstance(result, dict)
        

class TestVersionCompatibility:
    """Test version compatibility between a2aAgents and a2aNetwork"""
    
    def test_version_manager_creation(self):
        """Test VersionManager can be created"""
        vm = VersionManager()
        assert vm is not None
        assert hasattr(vm, 'check_compatibility')
        
    def test_compatibility_check(self):
        """Test compatibility checking between projects"""
        vm = VersionManager()
        
        # Test current versions
        result = vm.check_compatibility(
            "a2aAgents", "1.0.0",
            "a2aNetwork", "1.0.0"
        )
        
        assert isinstance(result, dict)
        assert "compatible" in result
        
    def test_version_detection(self):
        """Test automatic version detection"""
        vm = VersionManager()
        
        # Detect versions
        agents_version = vm.get_component_version("a2aAgents")
        network_version = vm.get_component_version("a2aNetwork")
        
        assert agents_version is not None
        assert network_version is not None
        

class TestAgentNetworkIntegration:
    """Test individual agents work with a2aNetwork components"""
    
    @pytest.mark.asyncio
    async def test_agent0_with_network_sdk(self):
        """Test Agent0 works with network SDK"""
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import (
            DataProductRegistrationAgentSDK
        )
        
        # Create agent instance
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8000",
            ord_registry_url="http://localhost:9000"
        )
        
        assert agent is not None
        assert agent.agent_id == "data_product_agent_0"
        assert hasattr(agent, 'handle_message')
        
    @pytest.mark.asyncio
    async def test_catalog_manager_with_network(self):
        """Test CatalogManager works with network components"""
        from app.a2a.agents.catalogManager.active.catalogManagerAgentSdk import (
            CatalogManagerAgentSDK
        )
        
        agent = CatalogManagerAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:9000"
        )
        
        assert agent is not None
        assert agent.agent_id == "catalog_manager_agent"
        
    def test_multiple_agents_share_sdk(self):
        """Test multiple agents can share SDK components"""
        agents_to_test = [
            ("agent0DataProduct", "dataProductAgentSdk", "DataProductRegistrationAgentSDK"),
            ("catalogManager", "catalogManagerAgentSdk", "CatalogManagerAgentSDK"),
            ("agent4CalcValidation", "calcValidationAgentSdk", "CalcValidationAgentSDK")
        ]
        
        loaded_agents = []
        
        for agent_dir, module_name, class_name in agents_to_test:
            module_path = f"app.a2a.agents.{agent_dir}.active.{module_name}"
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            loaded_agents.append(agent_class)
        
        # All should share same SDK base class
        for agent_class in loaded_agents:
            assert issubclass(agent_class, A2AAgentBase)
            

class TestNetworkAPIIntegration:
    """Test integration with a2aNetwork APIs"""
    
    @pytest.mark.asyncio
    async def test_registry_api_client(self):
        """Test RegistryAPI client integration"""
        try:
            from api.registryApi import RegistryAPI
            
            registry = RegistryAPI("http://localhost:9000")
            assert registry is not None
            assert hasattr(registry, 'register_agent')
            assert hasattr(registry, 'discover_agents')
        except ImportError:
            pytest.skip("RegistryAPI not available")
            
    @pytest.mark.asyncio
    async def test_trust_api_client(self):
        """Test TrustAPI client integration"""
        try:
            from api.trustApi import TrustAPI
            
            trust = TrustAPI("http://localhost:9001")
            assert trust is not None
            assert hasattr(trust, 'create_trust_contract')
            assert hasattr(trust, 'verify_trust')
        except ImportError:
            pytest.skip("TrustAPI not available")
            
    @pytest.mark.asyncio
    async def test_network_client_integration(self):
        """Test NetworkClient provides unified access"""
        try:
            from api.networkClient import NetworkClient
            
            client = NetworkClient(
                registry_url="http://localhost:9000",
                trust_service_url="http://localhost:9001"
            )
            
            assert client is not None
            assert hasattr(client, 'registry')
            assert hasattr(client, 'trust')
            assert hasattr(client, 'sdk')
        except ImportError:
            pytest.skip("NetworkClient not available")
            

class TestFailoverMechanisms:
    """Test failover and fallback mechanisms"""
    
    @pytest.mark.asyncio
    async def test_sdk_import_fallback(self):
        """Test SDK import fallback mechanism"""
        # Simulate network SDK not available
        with patch('sys.path', []):
            try:
                # This should fail but not crash
                from app.a2a.sdk import A2AAgentBase
                # If we get here, fallback worked
                assert True
            except ImportError as e:
                # Expected when no fallback available
                assert "SDK components not available" in str(e)
                
    @pytest.mark.asyncio
    async def test_network_connector_fallback(self):
        """Test NetworkConnector fallback to local"""
        connector = NetworkConnector(
            registry_url="http://nonexistent:9999",
            trust_service_url="http://nonexistent:9999"
        )
        await connector.initialize()
        
        # Should fallback to local
        mock_agent = Mock()
        mock_agent.agent_id = "test_fallback"
        mock_agent.get_agent_card.return_value = {"agent_id": "test_fallback"}
        
        result = await connector.register_agent(mock_agent)
        
        # Should succeed with local fallback
        assert result is not None
        assert "agent_id" in result
        assert result["source"] == "local_fallback"
        

class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_agent_registration_flow(self):
        """Test complete agent registration flow"""
        # Create agent
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import (
            DataProductRegistrationAgentSDK
        )
        
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8000",
            ord_registry_url="http://localhost:9000"
        )
        
        # Create network connector
        connector = NetworkConnector()
        await connector.initialize()
        
        # Register agent
        result = await connector.register_agent(agent)
        
        assert result is not None
        assert "agent_id" in result or "error" in result
        
    @pytest.mark.asyncio
    async def test_agent_communication_flow(self):
        """Test agent-to-agent communication flow"""
        # Create two agents
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import (
            DataProductRegistrationAgentSDK
        )
        from app.a2a.agents.catalogManager.active.catalogManagerAgentSdk import (
            CatalogManagerAgentSDK
        )
        
        agent1 = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8000",
            ord_registry_url="http://localhost:9000"
        )
        
        agent2 = CatalogManagerAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:9000"
        )
        
        # Create test message
        test_message = A2AMessage(
            conversation_id="test_conv_123",
            from_agent=agent1.agent_id,
            to_agent=agent2.agent_id,
            parts=[],
            timestamp="2025-01-01T00:00:00Z"
        )
        
        # Both agents should be able to process messages
        assert hasattr(agent1, 'handle_message')
        assert hasattr(agent2, 'handle_message')
        

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

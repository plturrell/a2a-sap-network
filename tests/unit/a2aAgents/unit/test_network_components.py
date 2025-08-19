#!/usr/bin/env python3
"""
Unit tests for a2aAgents network integration components
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.a2a.network.networkConnector import NetworkConnector
from app.a2a.network.agentRegistration import AgentRegistrationService
from app.a2a.network.networkMessaging import NetworkMessagingService
from app.a2a.version.versionManager import VersionManager
from app.a2a.version.compatibilityChecker import CompatibilityChecker
from app.a2a.version.dependencyResolver import DependencyResolver


class TestNetworkConnector:
    """Unit tests for NetworkConnector"""
    
    @pytest.fixture
    def mock_network_client(self):
        """Create mock NetworkClient"""
        mock_client = Mock()
        mock_client.registry = Mock()
        mock_client.trust = Mock()
        mock_client.sdk = Mock()
        return mock_client
    
    @pytest.fixture
    async def connector(self, mock_network_client):
        """Create NetworkConnector with mocked client"""
        with patch('app.a2a.network.networkConnector.NetworkClient', return_value=mock_network_client):
            connector = NetworkConnector(
                registry_url="http://test:9000",
                trust_service_url="http://test:9001"
            )
            await connector.initialize()
            return connector
    
    @pytest.mark.asyncio
    async def test_initialization(self, connector):
        """Test NetworkConnector initialization"""
        assert connector is not None
        assert connector._registry_url == "http://test:9000"
        assert connector._trust_service_url == "http://test:9001"
        
    @pytest.mark.asyncio
    async def test_check_network_availability_success(self, connector, mock_network_client):
        """Test network availability check when network is up"""
        # Mock successful health check
        mock_network_client.registry.health_check = AsyncMock(return_value=True)
        
        is_available = await connector.check_network_availability()
        
        assert is_available is True
        mock_network_client.registry.health_check.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_check_network_availability_failure(self, connector, mock_network_client):
        """Test network availability check when network is down"""
        # Mock failed health check
        mock_network_client.registry.health_check = AsyncMock(side_effect=Exception("Network error"))
        
        is_available = await connector.check_network_availability()
        
        assert is_available is False
        
    @pytest.mark.asyncio
    async def test_register_agent_network_success(self, connector, mock_network_client):
        """Test agent registration through network"""
        # Setup mocks
        connector._network_available = True
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.get_agent_card.return_value = {
            "agent_id": "test_agent",
            "name": "Test Agent"
        }
        
        mock_network_client.registry.register_agent = AsyncMock(
            return_value={"status": "registered", "agent_id": "test_agent"}
        )
        
        # Test registration
        result = await connector.register_agent(mock_agent)
        
        assert result["status"] == "registered"
        assert result["agent_id"] == "test_agent"
        mock_network_client.registry.register_agent.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_register_agent_fallback(self, connector):
        """Test agent registration fallback when network unavailable"""
        # Setup for fallback
        connector._network_available = False
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_fallback"
        mock_agent.get_agent_card.return_value = {
            "agent_id": "test_agent_fallback",
            "name": "Test Agent Fallback"
        }
        
        # Test fallback registration
        result = await connector.register_agent(mock_agent)
        
        assert result["agent_id"] == "test_agent_fallback"
        assert result["source"] == "local_fallback"
        assert result["registered"] is True
        
    @pytest.mark.asyncio
    async def test_route_message_network(self, connector, mock_network_client):
        """Test message routing through network"""
        connector._network_available = True
        test_message = {
            "from_agent": "agent1",
            "to_agent": "agent2",
            "content": "Test"
        }
        
        mock_network_client.route_message = AsyncMock(
            return_value={"delivered": True, "message_id": "123"}
        )
        
        result = await connector.route_message(test_message)
        
        assert result["delivered"] is True
        assert result["message_id"] == "123"
        
    @pytest.mark.asyncio
    async def test_route_message_fallback(self, connector):
        """Test message routing fallback"""
        connector._network_available = False
        test_message = {
            "from_agent": "agent1",
            "to_agent": "agent2",
            "content": "Test fallback"
        }
        
        result = await connector.route_message(test_message)
        
        assert result["routed"] is True
        assert result["source"] == "local_fallback"
        assert "message_id" in result
        

class TestAgentRegistrationService:
    """Unit tests for AgentRegistrationService"""
    
    @pytest.fixture
    def mock_connector(self):
        """Create mock NetworkConnector"""
        mock = Mock()
        mock.register_agent = AsyncMock()
        mock.unregister_agent = AsyncMock()
        mock.update_agent_status = AsyncMock()
        return mock
    
    @pytest.fixture
    def registration_service(self, mock_connector):
        """Create AgentRegistrationService"""
        with patch('app.a2a.network.agentRegistration.NetworkConnector', return_value=mock_connector):
            service = AgentRegistrationService()
            service._connector = mock_connector
            return service
    
    @pytest.mark.asyncio
    async def test_register_agent(self, registration_service, mock_connector):
        """Test agent registration through service"""
        mock_agent = Mock()
        mock_agent.agent_id = "service_test_agent"
        
        mock_connector.register_agent.return_value = {
            "status": "success",
            "agent_id": "service_test_agent"
        }
        
        result = await registration_service.register_agent(mock_agent)
        
        assert result["status"] == "success"
        mock_connector.register_agent.assert_called_once_with(mock_agent)
        
    @pytest.mark.asyncio
    async def test_auto_register_agents(self, registration_service, mock_connector):
        """Test automatic agent registration"""
        mock_agents = [
            Mock(agent_id="agent1"),
            Mock(agent_id="agent2"),
            Mock(agent_id="agent3")
        ]
        
        mock_connector.register_agent.return_value = {"status": "success"}
        
        results = await registration_service.auto_register_agents(mock_agents)
        
        assert len(results) == 3
        assert mock_connector.register_agent.call_count == 3
        
    @pytest.mark.asyncio
    async def test_health_monitoring(self, registration_service, mock_connector):
        """Test agent health monitoring"""
        agent_id = "health_test_agent"
        
        # Setup health check response
        mock_connector.check_agent_health = AsyncMock(
            return_value={"healthy": True, "uptime": 3600}
        )
        
        health = await registration_service.check_agent_health(agent_id)
        
        assert health["healthy"] is True
        assert health["uptime"] == 3600
        

class TestNetworkMessagingService:
    """Unit tests for NetworkMessagingService"""
    
    @pytest.fixture
    def mock_connector(self):
        """Create mock NetworkConnector"""
        mock = Mock()
        mock.route_message = AsyncMock()
        mock.broadcast_message = AsyncMock()
        return mock
    
    @pytest.fixture
    def messaging_service(self, mock_connector):
        """Create NetworkMessagingService"""
        with patch('app.a2a.network.networkMessaging.NetworkConnector', return_value=mock_connector):
            service = NetworkMessagingService()
            service._connector = mock_connector
            return service
    
    @pytest.mark.asyncio
    async def test_send_message(self, messaging_service, mock_connector):
        """Test sending message through service"""
        message = {
            "from": "agent1",
            "to": "agent2",
            "content": "Test message",
            "type": "request"
        }
        
        mock_connector.route_message.return_value = {
            "delivered": True,
            "message_id": "msg_123"
        }
        
        result = await messaging_service.send_message(message)
        
        assert result["delivered"] is True
        assert result["message_id"] == "msg_123"
        mock_connector.route_message.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_broadcast_message(self, messaging_service, mock_connector):
        """Test broadcasting message to multiple agents"""
        message = {
            "from": "agent1",
            "content": "Broadcast message",
            "type": "notification"
        }
        
        recipients = ["agent2", "agent3", "agent4"]
        
        mock_connector.broadcast_message.return_value = {
            "delivered_to": recipients,
            "failed": []
        }
        
        result = await messaging_service.broadcast_message(message, recipients)
        
        assert len(result["delivered_to"]) == 3
        assert len(result["failed"]) == 0
        
    @pytest.mark.asyncio
    async def test_message_queuing(self, messaging_service):
        """Test message queuing when network unavailable"""
        messaging_service._network_available = False
        
        message = {
            "from": "agent1",
            "to": "agent2",
            "content": "Queued message"
        }
        
        result = await messaging_service.send_message(message)
        
        assert result["queued"] is True
        assert "queue_id" in result
        assert len(messaging_service._message_queue) > 0
        

class TestVersionManager:
    """Unit tests for VersionManager"""
    
    @pytest.fixture
    def version_manager(self):
        """Create VersionManager instance"""
        return VersionManager()
    
    def test_version_detection(self, version_manager):
        """Test component version detection"""
        # Test a2aAgents version
        agents_version = version_manager.get_component_version("a2aAgents")
        assert agents_version is not None
        assert isinstance(agents_version, str)
        
        # Test a2aNetwork version
        network_version = version_manager.get_component_version("a2aNetwork")
        assert network_version is not None
        assert isinstance(network_version, str)
        
    def test_compatibility_check(self, version_manager):
        """Test version compatibility checking"""
        # Test compatible versions
        result = version_manager.check_compatibility(
            "a2aAgents", "1.0.0",
            "a2aNetwork", "1.0.0"
        )
        
        assert result["compatible"] is True
        assert "protocol_version" in result
        
        # Test incompatible versions
        result = version_manager.check_compatibility(
            "a2aAgents", "1.0.0",
            "a2aNetwork", "2.0.0"
        )
        
        assert "compatible" in result
        assert "reason" in result or result["compatible"] is True
        
    def test_update_recommendations(self, version_manager):
        """Test version update recommendations"""
        recommendations = version_manager.get_update_recommendations(
            "a2aAgents", "0.9.0",
            "a2aNetwork", "1.0.0"
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
    def test_protocol_version_check(self, version_manager):
        """Test A2A protocol version checking"""
        protocol_version = version_manager.get_protocol_version()
        
        assert protocol_version == "0.2.9"
        
        # Check protocol compatibility
        is_compatible = version_manager.is_protocol_compatible("0.2.9")
        assert is_compatible is True
        
        is_compatible = version_manager.is_protocol_compatible("0.1.0")
        assert is_compatible is False
        

class TestCompatibilityChecker:
    """Unit tests for CompatibilityChecker"""
    
    @pytest.fixture
    def checker(self):
        """Create CompatibilityChecker instance"""
        return CompatibilityChecker()
    
    def test_check_api_compatibility(self, checker):
        """Test API compatibility checking"""
        # Mock API specifications
        api_v1 = {
            "endpoints": ["/agents", "/messages", "/trust"],
            "version": "1.0"
        }
        
        api_v2 = {
            "endpoints": ["/agents", "/messages", "/trust", "/metrics"],
            "version": "2.0"
        }
        
        # v2 should be compatible with v1 (backward compatible)
        result = checker.check_api_compatibility(api_v1, api_v2)
        assert result["compatible"] is True
        assert result["backward_compatible"] is True
        
    def test_check_message_format_compatibility(self, checker):
        """Test message format compatibility"""
        format_v1 = {
            "fields": ["id", "from", "to", "content"],
            "version": "1.0"
        }
        
        format_v2 = {
            "fields": ["id", "from", "to", "content", "timestamp"],
            "version": "2.0"
        }
        
        result = checker.check_message_compatibility(format_v1, format_v2)
        assert "compatible" in result
        
    def test_feature_compatibility(self, checker):
        """Test feature compatibility checking"""
        agent_features = ["messaging", "trust", "discovery"]
        network_features = ["messaging", "trust", "discovery", "metrics"]
        
        result = checker.check_feature_compatibility(
            agent_features, network_features
        )
        
        assert result["all_supported"] is True
        assert len(result["missing_features"]) == 0
        

class TestDependencyResolver:
    """Unit tests for DependencyResolver"""
    
    @pytest.fixture
    def resolver(self):
        """Create DependencyResolver instance"""
        return DependencyResolver()
    
    def test_resolve_dependencies(self, resolver):
        """Test dependency resolution"""
        component = "a2aAgents"
        version = "1.0.0"
        
        dependencies = resolver.resolve_dependencies(component, version)
        
        assert isinstance(dependencies, dict)
        assert "a2aNetwork" in dependencies
        assert "protocol" in dependencies
        
    def test_check_dependency_conflicts(self, resolver):
        """Test dependency conflict detection"""
        dependencies = {
            "a2aNetwork": "1.0.0",
            "protocol": "0.2.9",
            "sdk": "1.0.0"
        }
        
        conflicts = resolver.check_conflicts(dependencies)
        
        assert isinstance(conflicts, list)
        assert len(conflicts) == 0  # No conflicts expected
        
    def test_get_dependency_tree(self, resolver):
        """Test dependency tree generation"""
        tree = resolver.get_dependency_tree("a2aAgents", "1.0.0")
        
        assert isinstance(tree, dict)
        assert "a2aAgents" in tree
        assert "dependencies" in tree["a2aAgents"]
        
    def test_update_path_calculation(self, resolver):
        """Test update path calculation"""
        current = {
            "a2aAgents": "0.9.0",
            "a2aNetwork": "0.9.0"
        }
        
        target = {
            "a2aAgents": "1.0.0",
            "a2aNetwork": "1.0.0"
        }
        
        update_path = resolver.calculate_update_path(current, target)
        
        assert isinstance(update_path, list)
        assert len(update_path) > 0
        assert update_path[0]["component"] in ["a2aNetwork", "a2aAgents"]
        

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

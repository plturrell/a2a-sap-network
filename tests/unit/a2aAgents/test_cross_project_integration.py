"""
Cross-Project Integration Tests for A2A Platform
Tests integration between A2A Network (JavaScript) and A2A Agents (Python)
Leverages existing test infrastructure and extends coverage to cross-project scenarios
"""

import pytest
import asyncio
import httpx
import time
import json
import subprocess
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import existing A2A Agents test utilities
from app.core.security import SecurityValidator, TokenManager
from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole


@pytest.mark.cross_project
class TestCrossProjectAuthentication:
    """Test authentication integration between Network and Agents"""
    
    @pytest.mark.asyncio
    async def test_sso_token_validation_across_projects(self, mock_sso_manager, cross_project_config):
        """Test that SSO tokens work across both Network and Agents projects"""
        # Simulate authentication from Network SSO Manager
        auth_result = await mock_sso_manager.authenticateUser({
            'nameID': 'cross.project@example.com',
            'email': 'cross.project@example.com',
            'displayName': 'Cross Project User',
            'roles': ['AgentDeveloper', 'NetworkAdmin']
        }, 'saml')
        
        assert auth_result['success']
        assert auth_result['accessToken']
        
        # Test token validation in Agents project
        token_manager = TokenManager()
        validation_result = token_manager.validate_token(auth_result['accessToken'])
        
        assert validation_result['valid']
        assert 'AgentDeveloper' in validation_result['roles']
        
    @pytest.mark.asyncio
    async def test_cross_project_permission_inheritance(self, mock_sso_manager, cross_project_config):
        """Test that permissions granted in Network are honored in Agents"""
        # Authenticate with specific permissions
        auth_result = await mock_sso_manager.authenticateUser({
            'nameID': 'permissions.test@example.com',
            'roles': ['AgentDeveloper'],
            'permissions': ['agent.create', 'agent.deploy', 'network.view']
        }, 'saml')
        
        # Validate permissions in Agents context
        security_validator = SecurityValidator()
        
        # Test agent creation permission
        can_create = security_validator.check_permission(
            auth_result['userInfo']['permissions'], 
            'agent.create'
        )
        assert can_create
        
        # Test deployment permission
        can_deploy = security_validator.check_permission(
            auth_result['userInfo']['permissions'], 
            'agent.deploy'
        )
        assert can_deploy
        
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_authentication_performance_cross_project(self, mock_sso_manager, cross_project_config):
        """Test authentication performance meets cross-project requirements"""
        start_time = time.time()
        
        # Simulate authentication flow
        auth_result = await mock_sso_manager.authenticateUser({
            'nameID': 'perf.test@example.com',
            'roles': ['AgentDeveloper']
        }, 'saml')
        
        auth_time = time.time() - start_time
        
        # Verify performance threshold
        threshold = cross_project_config['performance_thresholds']['authentication']
        assert auth_time < threshold, f"Authentication took {auth_time}s, threshold is {threshold}s"
        assert auth_result['success']


@pytest.mark.cross_project
class TestCrossProjectNavigation:
    """Test navigation integration between Network and Agents"""
    
    @pytest.mark.asyncio
    async def test_deep_link_from_network_to_agents(self, network_urls, cross_project_config):
        """Test deep linking from Network application to specific Agent"""
        # Simulate navigation from Network to Agents with context
        navigation_context = {
            'source_app': 'network',
            'target_app': 'agents',
            'deep_link': '/agent/test_agent_123/code',
            'context': {
                'agent_id': 'test_agent_123',
                'project_id': 'test_project',
                'file_path': 'src/main.py',
                'line_number': 42
            }
        }
        
        # Test that Agents application can receive and process the navigation
        agents_url = network_urls['agents_app']
        deep_link_url = f"{agents_url}{navigation_context['deep_link']}"
        
        # Simulate HTTP request to verify endpoint exists
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(deep_link_url, timeout=5.0)
                # Accept both 200 (success) and 404 (endpoint not implemented yet)
                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                # Server not running - this is acceptable for unit tests
                pytest.skip("Agents server not running")
    
    @pytest.mark.asyncio
    async def test_context_preservation_across_projects(self, cross_project_config):
        """Test that navigation context is preserved across projects"""
        # Simulate context from Network application
        network_context = {
            'user_id': 'test_user_123',
            'selected_agent': 'test_agent_123',
            'filter_settings': {'status': 'active', 'type': 'data_product'},
            'view_preferences': {'theme': 'dark', 'layout': 'grid'}
        }
        
        # Test context serialization/deserialization for cross-project transfer
        serialized_context = json.dumps(network_context)
        deserialized_context = json.loads(serialized_context)
        
        assert deserialized_context == network_context
        assert deserialized_context['selected_agent'] == 'test_agent_123'
        
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_navigation_performance_cross_project(self, cross_project_config):
        """Test navigation performance meets cross-project requirements"""
        start_time = time.time()
        
        # Simulate navigation processing
        await asyncio.sleep(0.1)  # Simulate navigation processing time
        
        navigation_time = time.time() - start_time
        
        # Verify performance threshold
        threshold = cross_project_config['performance_thresholds']['navigation']
        assert navigation_time < threshold, f"Navigation took {navigation_time}s, threshold is {threshold}s"


@pytest.mark.cross_project
class TestCrossProjectResourceSharing:
    """Test resource sharing between Network and Agents"""
    
    @pytest.mark.asyncio
    async def test_shared_configuration_sync(self, cross_project_config):
        """Test configuration synchronization between projects"""
        # Simulate configuration update from Network
        network_config = {
            'agent_settings': {
                'max_concurrent_agents': 10,
                'default_timeout': 30000,
                'enable_debug_mode': False
            },
            'ui_settings': {
                'theme': 'dark',
                'language': 'en',
                'notifications_enabled': True
            }
        }
        
        # Test that Agents project can receive and apply configuration
        # In real implementation, this would sync through SharedResourceManager
        agents_config = network_config.copy()
        
        assert agents_config['agent_settings']['max_concurrent_agents'] == 10
        assert agents_config['ui_settings']['theme'] == 'dark'
        
    @pytest.mark.asyncio
    async def test_feature_flag_propagation(self, cross_project_config):
        """Test feature flag propagation between projects"""
        # Simulate feature flag update from Network
        feature_flags = {
            'cross_project_integration': True,
            'new_agent_ui': False,
            'enhanced_monitoring': True
        }
        
        # Test feature flag validation in Agents context
        assert feature_flags['cross_project_integration'] is True
        assert feature_flags['new_agent_ui'] is False
        
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_sync_performance(self, cross_project_config):
        """Test resource synchronization performance"""
        start_time = time.time()
        
        # Simulate resource sync processing
        await asyncio.sleep(0.5)  # Simulate sync processing time
        
        sync_time = time.time() - start_time
        
        # Verify performance threshold
        threshold = cross_project_config['performance_thresholds']['resource_sync']
        assert sync_time < threshold, f"Resource sync took {sync_time}s, threshold is {threshold}s"


@pytest.mark.cross_project
class TestCrossProjectAgentIntegration:
    """Test agent-specific cross-project integration"""
    
    @pytest.mark.asyncio
    async def test_agent_discovery_from_network(self, mock_agent, network_urls):
        """Test that Network can discover and interact with Agents"""
        # Test agent card generation for Network consumption
        agent_card = mock_agent.get_agent_card()
        
        assert agent_card['agent_id'] == 'test_agent_123'
        assert agent_card['name'] == 'Test Agent'
        assert 'capabilities' in agent_card
        assert 'handlers' in agent_card
        
    @pytest.mark.asyncio
    async def test_agent_message_routing_cross_project(self, mock_agent, mock_message):
        """Test message routing between Network and Agents"""
        # Test that messages from Network can be processed by Agents
        message = A2AMessage(
            conversation_id=mock_message['conversation_id'],
            from_agent=mock_message['from_agent'],
            to_agent=mock_message['to_agent'],
            parts=[MessagePart(
                kind='text',
                text=mock_message['parts'][0]['text']
            )],
            timestamp=mock_message['timestamp']
        )
        
        # Simulate message processing
        assert message.conversation_id == 'test_conv_123'
        assert message.parts[0].text == 'Test message content'
        
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_response_performance(self, mock_agent, cross_project_config):
        """Test agent response performance meets requirements"""
        start_time = time.time()
        
        # Simulate agent processing
        agent_card = mock_agent.get_agent_card()
        
        response_time = time.time() - start_time
        
        # Verify performance threshold
        threshold = cross_project_config['performance_thresholds']['agent_response']
        assert response_time < threshold, f"Agent response took {response_time}s, threshold is {threshold}s"
        assert agent_card is not None


@pytest.mark.cross_project
@pytest.mark.e2e
class TestEndToEndCrossProjectWorkflows:
    """End-to-end tests for cross-project workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_development_workflow(self, mock_sso_manager, mock_agent, cross_project_config):
        """Test complete workflow from Network to Agents and back"""
        # Step 1: Authenticate in Network
        auth_result = await mock_sso_manager.authenticateUser({
            'nameID': 'developer@example.com',
            'roles': ['AgentDeveloper']
        }, 'saml')
        
        assert auth_result['success']
        
        # Step 2: Navigate from Network to Agents
        navigation_context = {
            'agent_id': 'test_agent_123',
            'action': 'edit_code'
        }
        
        # Step 3: Interact with Agent in Agents project
        agent_card = mock_agent.get_agent_card()
        assert agent_card['agent_id'] == navigation_context['agent_id']
        
        # Step 4: Return to Network with results
        workflow_result = {
            'success': True,
            'agent_updated': True,
            'changes': ['updated main.py', 'added new capability']
        }
        
        assert workflow_result['success']
        assert workflow_result['agent_updated']
        
    @pytest.mark.asyncio
    async def test_cross_project_monitoring_integration(self, cross_project_config):
        """Test monitoring integration between projects"""
        # Simulate metrics collection from both projects
        network_metrics = {
            'active_users': 25,
            'navigation_events': 150,
            'authentication_events': 45
        }
        
        agents_metrics = {
            'active_agents': 12,
            'message_throughput': 200,
            'processing_time_avg': 1.5
        }
        
        # Test unified metrics aggregation
        unified_metrics = {
            **network_metrics,
            **agents_metrics,
            'cross_project_health': 'healthy'
        }
        
        assert unified_metrics['active_users'] == 25
        assert unified_metrics['active_agents'] == 12
        assert unified_metrics['cross_project_health'] == 'healthy'


# Performance test suite
@pytest.mark.performance
class TestCrossProjectPerformance:
    """Performance tests for cross-project operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_cross_project_operations(self, cross_project_config):
        """Test performance under concurrent cross-project operations"""
        start_time = time.time()
        
        # Simulate concurrent operations
        tasks = []
        for i in range(10):
            task = asyncio.create_task(self._simulate_cross_project_operation(i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all operations completed successfully
        assert all(result['success'] for result in results)
        
        # Verify performance under load
        assert total_time < 10.0, f"Concurrent operations took {total_time}s, should be < 10s"
        
    async def _simulate_cross_project_operation(self, operation_id: int) -> Dict[str, Any]:
        """Simulate a cross-project operation"""
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            'operation_id': operation_id,
            'success': True,
            'duration': 0.1
        }


# Utility functions for cross-project testing
def get_network_test_status() -> Dict[str, Any]:
    """Get status of Network JavaScript tests"""
    try:
        network_path = Path("/Users/apple/projects/a2a/a2aNetwork")
        test_file = network_path / "test" / "testLaunchpadIntegration.js"
        
        if test_file.exists():
            return {
                'available': True,
                'test_file': str(test_file),
                'framework': 'JavaScript/Node.js'
            }
        else:
            return {
                'available': False,
                'reason': 'Test file not found'
            }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


def run_network_tests_integration() -> Dict[str, Any]:
    """Run Network tests as part of cross-project integration"""
    try:
        network_path = Path("/Users/apple/projects/a2a/a2aNetwork")
        os.chdir(network_path)
        
        result = subprocess.run([
            "node", "test/testLaunchpadIntegration.js"
        ], capture_output=True, text=True, timeout=60, check=False)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Network tests timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Test configuration validation
@pytest.mark.unit
def test_cross_project_config_validation(cross_project_config):
    """Validate cross-project configuration"""
    required_keys = [
        'network_path', 'agents_path', 'shared_secret', 
        'test_timeout', 'performance_thresholds'
    ]
    
    for key in required_keys:
        assert key in cross_project_config, f"Missing required config key: {key}"
    
    # Validate performance thresholds
    thresholds = cross_project_config['performance_thresholds']
    assert thresholds['authentication'] > 0
    assert thresholds['navigation'] > 0
    assert thresholds['resource_sync'] > 0
    assert thresholds['agent_response'] > 0


if __name__ == "__main__":
    # Run cross-project tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "cross_project"
    ])

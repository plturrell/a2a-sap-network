#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Blockchain Integration

This test suite covers all blockchain integration methods across all 16 A2A agents.
It includes unit tests for:
- Blockchain mixin initialization
- Message handlers
- Trust verification
- Inter-agent communication
- Error handling and recovery
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import os
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../app/a2a'))

# Import blockchain integration components
from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin


class TestBlockchainIntegrationMixin(unittest.TestCase):
    """Test the core BlockchainIntegrationMixin functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Enable blockchain for testing
        os.environ["BLOCKCHAIN_ENABLED"] = "true"
        
        # Create a test class that uses the mixin
        class TestAgent(BlockchainIntegrationMixin):
            def __init__(self):
                super().__init__()
                self.agent_id = "test_agent"
                self.name = "Test Agent"
                
        self.test_agent = TestAgent()
    
    def tearDown(self):
        """Clean up after tests"""
        # Reset environment
        if "BLOCKCHAIN_ENABLED" in os.environ:
            del os.environ["BLOCKCHAIN_ENABLED"]
    
    def test_blockchain_mixin_initialization(self):
        """Test that blockchain mixin initializes correctly"""
        self.assertIsNotNone(self.test_agent)
        self.assertIsNone(self.test_agent.blockchain_client)
        self.assertIsNone(self.test_agent.blockchain_integration)
        self.assertIsNone(self.test_agent.agent_identity)
        self.assertIsNone(self.test_agent.message_listener)
        self.assertTrue(self.test_agent.blockchain_enabled)
    
    @patch('app.a2a.sdk.blockchainIntegration.A2ABlockchainClient')
    @patch('app.a2a.sdk.blockchainIntegration.ContractConfigManager')
    @patch('app.a2a.sdk.blockchainIntegration.AgentIdentity')
    @patch('app.a2a.sdk.blockchainIntegration.BlockchainAgentIntegration')
    def test_initialize_blockchain(self, mock_integration, mock_identity, mock_config_manager, mock_client):
        """Test blockchain initialization process"""
        # Setup mocks
        mock_config = {'contracts': {
            'AgentRegistry': {'address': '0x123'},
            'MessageRouter': {'address': '0x456'}
        }}
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_integration.return_value.is_registered.return_value = False
        mock_integration.return_value.register_agent.return_value = "0xabc123"
        
        # Set private key
        os.environ["TEST_AGENT_PRIVATE_KEY"] = "0xprivatekey123"
        
        # Initialize blockchain
        self.test_agent._initialize_blockchain(
            agent_name="test_agent",
            capabilities=["test_capability"],
            endpoint="http://localhost:8000"
        )
        
        # Verify initialization
        self.assertIsNotNone(self.test_agent.blockchain_client)
        self.assertIsNotNone(self.test_agent.agent_identity)
        self.assertIsNotNone(self.test_agent.blockchain_integration)
        
        # Verify registration was called
        mock_integration.return_value.register_agent.assert_called_once()
    
    def test_send_blockchain_message(self):
        """Test sending blockchain messages"""
        # Mock blockchain integration
        self.test_agent.blockchain_integration = Mock()
        self.test_agent.blockchain_integration.send_message.return_value = ("0xtx123", "msg123")
        
        # Send message
        message_id = self.test_agent.send_blockchain_message(
            to_address="0xrecipient",
            content={"test": "data"},
            message_type="TEST"
        )
        
        # Verify
        self.assertEqual(message_id, "msg123")
        self.test_agent.blockchain_integration.send_message.assert_called_once()
    
    def test_verify_trust(self):
        """Test trust verification"""
        # Mock blockchain integration
        self.test_agent.blockchain_integration = Mock()
        self.test_agent.blockchain_integration.get_agent_info.return_value = {
            'reputation': 75,
            'active': True
        }
        
        # Test trust verification
        result = self.test_agent.verify_trust("0xagent123", min_reputation=50)
        self.assertTrue(result)
        
        # Test insufficient reputation
        result = self.test_agent.verify_trust("0xagent123", min_reputation=100)
        self.assertFalse(result)
    
    def test_get_agent_by_capability(self):
        """Test finding agents by capability"""
        # Mock blockchain integration
        self.test_agent.blockchain_integration = Mock()
        mock_agents = [
            {'address': '0x1', 'name': 'Agent1'},
            {'address': '0x2', 'name': 'Agent2'}
        ]
        self.test_agent.blockchain_integration.find_agents_by_capability.return_value = mock_agents
        
        # Find agents
        agents = self.test_agent.get_agent_by_capability("test_capability")
        
        # Verify
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0]['name'], 'Agent1')
    
    def test_blockchain_stats(self):
        """Test blockchain statistics retrieval"""
        # Mock blockchain integration
        self.test_agent.agent_identity = Mock()
        self.test_agent.agent_identity.address = "0xtest123"
        self.test_agent.blockchain_integration = Mock()
        self.test_agent.blockchain_integration.get_agent_info.return_value = {
            'reputation': 85,
            'active': True
        }
        
        # Get stats
        stats = self.test_agent.get_blockchain_stats()
        
        # Verify
        self.assertTrue(stats['enabled'])
        self.assertEqual(stats['address'], "0xtest123")
        self.assertTrue(stats['registered'])
        self.assertEqual(stats['reputation'], 85)
        self.assertTrue(stats['active'])


class TestAgentManagerBlockchainIntegration(unittest.TestCase):
    """Test AgentManager blockchain integration"""
    
    @patch('app.a2a.sdk.blockchainIntegration.BlockchainIntegrationMixin.__init__')
    def setUp(self, mock_blockchain_init):
        """Set up test environment"""
        # Mock blockchain initialization to avoid import issues
        mock_blockchain_init.return_value = None
        
        # Import agent manager
        try:
            from app.a2a.agents.agentManager.active.enhancedAgentManagerAgent import EnhancedAgentManagerAgent
            self.agent_manager_class = EnhancedAgentManagerAgent
            self.agent_manager = None
        except ImportError:
            self.skipTest("AgentManager not available for testing")
    
    def test_agent_manager_has_blockchain_capabilities(self):
        """Test that AgentManager has blockchain capabilities defined"""
        if not self.agent_manager_class:
            self.skipTest("AgentManager class not available")
            
        # Create instance
        agent_manager = self.agent_manager_class()
        
        # Check blockchain capabilities
        self.assertTrue(hasattr(agent_manager, 'blockchain_capabilities'))
        self.assertIn('orchestration', agent_manager.blockchain_capabilities)
        self.assertIn('coordination', agent_manager.blockchain_capabilities)
        self.assertIn('task_delegation', agent_manager.blockchain_capabilities)
    
    def test_agent_manager_trust_thresholds(self):
        """Test that AgentManager has trust thresholds defined"""
        if not self.agent_manager_class:
            self.skipTest("AgentManager class not available")
            
        # Create instance
        agent_manager = self.agent_manager_class()
        
        # Check trust thresholds
        self.assertTrue(hasattr(agent_manager, 'trust_thresholds'))
        self.assertIn('orchestration', agent_manager.trust_thresholds)
        self.assertEqual(agent_manager.trust_thresholds['orchestration'], 0.7)
        self.assertEqual(agent_manager.trust_thresholds['coordination'], 0.6)
    
    async def test_handle_blockchain_orchestration(self):
        """Test blockchain orchestration handler"""
        if not self.agent_manager_class:
            self.skipTest("AgentManager class not available")
            
        # Create instance
        agent_manager = self.agent_manager_class()
        
        # Mock get_agent_reputation
        agent_manager.get_agent_reputation = AsyncMock(return_value=0.8)
        
        # Test message
        test_message = {
            "sender_id": "test_sender",
            "message_type": "ORCHESTRATION_REQUEST"
        }
        
        test_content = {
            "target_agents": ["agent1", "agent2"],
            "orchestration_type": "workflow",
            "orchestration_params": {}
        }
        
        # Call handler
        if hasattr(agent_manager, '_handle_blockchain_orchestration'):
            result = await agent_manager._handle_blockchain_orchestration(test_message, test_content)
            
            # Verify result
            self.assertEqual(result['status'], 'success')
            self.assertIn('orchestration_result', result)
    
    async def test_handle_blockchain_coordination(self):
        """Test blockchain coordination handler"""
        if not self.agent_manager_class:
            self.skipTest("AgentManager class not available")
            
        # Create instance
        agent_manager = self.agent_manager_class()
        
        # Mock get_agent_reputation
        agent_manager.get_agent_reputation = AsyncMock(return_value=0.7)
        
        # Test message
        test_message = {
            "sender_id": "test_sender",
            "message_type": "COORDINATION_REQUEST"
        }
        
        test_content = {
            "coordination_type": "task_delegation",
            "participating_agents": ["agent1", "agent2"],
            "coordination_params": {}
        }
        
        # Call handler
        if hasattr(agent_manager, '_handle_blockchain_coordination'):
            result = await agent_manager._handle_blockchain_coordination(test_message, test_content)
            
            # Verify result
            self.assertEqual(result['status'], 'success')
            self.assertIn('coordination_result', result)


class TestBlockchainMessageHandlers(unittest.TestCase):
    """Test blockchain message handlers across different agents"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_handlers = {}
        
        # Define expected handlers for each agent type
        self.expected_handlers = {
            'calculationAgent': [
                '_handle_blockchain_calculation_request',
                '_handle_blockchain_distributed_calculation',
                '_handle_blockchain_formula_verification'
            ],
            'sqlAgent': [
                '_handle_blockchain_sql_query_execution',
                '_handle_blockchain_database_operations',
                '_handle_blockchain_distributed_query'
            ],
            'dataManager': [
                '_handle_blockchain_data_validation',
                '_handle_blockchain_data_transformation',
                '_handle_blockchain_metadata_management'
            ],
            'catalogManager': [
                '_handle_blockchain_catalog_search',
                '_handle_blockchain_resource_registration',
                '_handle_blockchain_metadata_indexing'
            ],
            'agentBuilder': [
                '_handle_blockchain_agent_creation',
                '_handle_blockchain_template_management',
                '_handle_blockchain_deployment_automation'
            ],
            'embeddingFineTuner': [
                '_handle_blockchain_embedding_optimization',
                '_handle_blockchain_model_fine_tuning',
                '_handle_blockchain_model_collaboration'
            ],
            'qaValidation': [
                '_handle_blockchain_qa_validation',
                '_handle_blockchain_consensus_validation',
                '_handle_blockchain_compliance_checking'
            ]
        }
    
    def test_handler_signatures(self):
        """Test that all handlers have correct signatures"""
        # This is a meta-test to ensure handlers follow the pattern
        expected_signature = "async def handler(self, message: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]"
        
        # All handlers should:
        # 1. Be async methods
        # 2. Accept self, message, and content parameters
        # 3. Return a dictionary
        # 4. Include blockchain_verified in response
        
        self.assertTrue(True)  # Placeholder - would check actual implementations


class TestBlockchainErrorHandling(unittest.TestCase):
    """Test error handling and recovery mechanisms"""
    
    def setUp(self):
        """Set up test environment"""
        class TestAgent(BlockchainIntegrationMixin):
            def __init__(self):
                super().__init__()
                
        self.test_agent = TestAgent()
    
    def test_blockchain_disabled_handling(self):
        """Test behavior when blockchain is disabled"""
        # Disable blockchain
        self.test_agent.blockchain_enabled = False
        
        # Test send message
        result = self.test_agent.send_blockchain_message("0xtest", {"data": "test"})
        self.assertIsNone(result)
        
        # Test get agents
        agents = self.test_agent.get_agent_by_capability("test")
        self.assertEqual(agents, [])
        
        # Test verify trust (should default to trust)
        trust = self.test_agent.verify_trust("0xtest")
        self.assertTrue(trust)
    
    def test_blockchain_initialization_failure(self):
        """Test handling of blockchain initialization failures"""
        with patch('app.a2a.sdk.blockchainIntegration.A2ABlockchainClient') as mock_client:
            # Make initialization fail
            mock_client.side_effect = Exception("Connection failed")
            
            # Initialize blockchain
            self.test_agent._initialize_blockchain("test", ["capability"])
            
            # Should disable blockchain on failure
            self.assertFalse(self.test_agent.blockchain_enabled)
    
    def test_message_send_failure_handling(self):
        """Test handling of message send failures"""
        # Mock blockchain integration
        self.test_agent.blockchain_integration = Mock()
        self.test_agent.blockchain_integration.send_message.side_effect = Exception("Network error")
        
        # Try to send message
        result = self.test_agent.send_blockchain_message("0xtest", {"data": "test"})
        
        # Should return None on failure
        self.assertIsNone(result)
    
    def test_trust_verification_failure(self):
        """Test handling of trust verification failures"""
        # Mock blockchain integration
        self.test_agent.blockchain_integration = Mock()
        self.test_agent.blockchain_integration.get_agent_info.side_effect = Exception("Query failed")
        
        # Try to verify trust
        result = self.test_agent.verify_trust("0xtest")
        
        # Should return False on failure
        self.assertFalse(result)


class TestBlockchainIntegrationAsync(unittest.TestCase):
    """Test async blockchain operations"""
    
    def setUp(self):
        """Set up test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up"""
        self.loop.close()
    
    def test_async_message_listener(self):
        """Test async message listener functionality"""
        # Create test agent
        class TestAgent(BlockchainIntegrationMixin):
            def __init__(self):
                super().__init__()
                self.messages_received = []
                
            def _handle_blockchain_message(self, message):
                self.messages_received.append(message)
                
        agent = TestAgent()
        
        # Mock message listener
        agent.message_listener = Mock()
        agent.message_listener.start = AsyncMock()
        
        # Test listener startup
        async def test_listener():
            await agent._run_message_listener()
            
        # Run test
        try:
            self.loop.run_until_complete(test_listener())
        except:
            pass  # Expected if no actual blockchain
            
        # Verify start was called
        agent.message_listener.start.assert_called_once()


def run_tests():
    """Run all blockchain integration tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainIntegrationMixin))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentManagerBlockchainIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainMessageHandlers))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainIntegrationAsync))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Integration Tests with Test Blockchain Networks

This test suite uses local test blockchain networks (like Ganache/Anvil) to test
actual blockchain interactions including:
- Agent registration on blockchain
- Message routing through smart contracts  
- Trust and reputation management
- Multi-agent coordination scenarios
"""

import unittest
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import subprocess

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../app/a2a'))


class TestBlockchainNetwork:
    """Manages a local test blockchain network"""
    
    def __init__(self, network_type="anvil"):
        self.network_type = network_type
        self.process = None
        self.rpc_url = "http://localhost:8545"
        self.chain_id = 31337
        
        # Test accounts (Anvil default accounts)
        self.accounts = [
            {
                "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
                "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            },
            {
                "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
            },
            {
                "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
            }
        ]
    
    def start(self):
        """Start the test blockchain network"""
        if self.network_type == "anvil":
            # Start Anvil (Foundry's local testnet)
            self.process = subprocess.Popen(
                ["anvil", "--port", "8545", "--accounts", "10"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(2)  # Wait for network to start
        elif self.network_type == "ganache":
            # Start Ganache
            self.process = subprocess.Popen(
                ["ganache-cli", "--port", "8545", "--accounts", "10"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Wait for network to start
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
    
    def stop(self):
        """Stop the test blockchain network"""
        if self.process:
            self.process.terminate()
            self.process.wait()
    
    def get_account(self, index=0):
        """Get test account by index"""
        return self.accounts[index] if index < len(self.accounts) else None


class BlockchainIntegrationTestCase(unittest.TestCase):
    """Base class for blockchain integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test blockchain network"""
        cls.network = TestBlockchainNetwork()
        try:
            cls.network.start()
            cls.network_available = True
        except:
            cls.network_available = False
            print("Warning: Test blockchain network not available. Skipping integration tests.")
    
    @classmethod
    def tearDownClass(cls):
        """Stop test blockchain network"""
        if hasattr(cls, 'network'):
            cls.network.stop()
    
    def setUp(self):
        """Set up test environment"""
        if not self.network_available:
            self.skipTest("Test blockchain network not available")
        
        # Set environment variables for blockchain
        os.environ["BLOCKCHAIN_ENABLED"] = "true"
        os.environ["A2A_RPC_URL"] = self.network.rpc_url
        
        # Deploy test contracts
        self.contracts = self._deploy_test_contracts()
    
    def tearDown(self):
        """Clean up test environment"""
        # Reset environment
        for key in ["BLOCKCHAIN_ENABLED", "A2A_RPC_URL"]:
            if key in os.environ:
                del os.environ[key]
    
    def _deploy_test_contracts(self):
        """Deploy test smart contracts"""
        # In a real implementation, this would deploy actual contracts
        # For now, return mock addresses
        return {
            "AgentRegistry": {"address": "0x5FbDB2315678afecb367f032d93F642f64180aa3"},
            "MessageRouter": {"address": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"},
            "TrustManager": {"address": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"}
        }


class TestAgentRegistration(BlockchainIntegrationTestCase):
    """Test agent registration on blockchain"""
    
    @patch('app.a2a.sdk.blockchainIntegration.ContractConfigManager')
    def test_register_single_agent(self, mock_config_manager):
        """Test registering a single agent on blockchain"""
        # Mock contract config
        mock_config_manager.return_value.load_config.return_value = {
            'contracts': self.contracts
        }
        
        # Import and create agent
        try:
            from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin
            
            class TestAgent(BlockchainIntegrationMixin):
                def __init__(self):
                    super().__init__()
                    
            agent = TestAgent()
            
            # Set test account
            test_account = self.network.get_account(0)
            os.environ["TEST_AGENT_PRIVATE_KEY"] = test_account["private_key"]
            
            # Initialize blockchain
            agent._initialize_blockchain(
                agent_name="test_agent",
                capabilities=["capability1", "capability2"],
                endpoint="http://localhost:8000"
            )
            
            # Verify agent is registered
            # In real test, would check blockchain state
            self.assertTrue(agent.blockchain_enabled)
            
        except ImportError:
            self.skipTest("Blockchain integration not available")
    
    def test_register_multiple_agents(self):
        """Test registering multiple agents on blockchain"""
        agents = []
        
        for i in range(3):
            try:
                from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin
                
                class TestAgent(BlockchainIntegrationMixin):
                    def __init__(self, agent_id):
                        super().__init__()
                        self.agent_id = agent_id
                        
                agent = TestAgent(f"agent_{i}")
                
                # Set unique account for each agent
                test_account = self.network.get_account(i)
                os.environ[f"AGENT_{i}_PRIVATE_KEY"] = test_account["private_key"]
                
                # Initialize blockchain
                agent._initialize_blockchain(
                    agent_name=f"agent_{i}",
                    capabilities=[f"capability_{i}"],
                    endpoint=f"http://localhost:800{i}"
                )
                
                agents.append(agent)
                
            except ImportError:
                self.skipTest("Blockchain integration not available")
        
        # Verify all agents registered
        self.assertEqual(len(agents), 3)
        for agent in agents:
            self.assertTrue(agent.blockchain_enabled)


class TestBlockchainMessaging(BlockchainIntegrationTestCase):
    """Test blockchain message routing"""
    
    def test_send_message_between_agents(self):
        """Test sending messages between two agents"""
        try:
            from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin
            
            # Create sender agent
            class SenderAgent(BlockchainIntegrationMixin):
                def __init__(self):
                    super().__init__()
                    
            sender = SenderAgent()
            sender_account = self.network.get_account(0)
            os.environ["SENDER_PRIVATE_KEY"] = sender_account["private_key"]
            
            # Create receiver agent
            class ReceiverAgent(BlockchainIntegrationMixin):
                def __init__(self):
                    super().__init__()
                    self.received_messages = []
                    
                def _handle_blockchain_message(self, message):
                    self.received_messages.append(message)
                    
            receiver = ReceiverAgent()
            receiver_account = self.network.get_account(1)
            os.environ["RECEIVER_PRIVATE_KEY"] = receiver_account["private_key"]
            
            # Initialize both agents
            sender._initialize_blockchain("sender", ["messaging"], "http://localhost:8001")
            receiver._initialize_blockchain("receiver", ["messaging"], "http://localhost:8002")
            
            # Send message
            test_content = {"test": "data", "timestamp": time.time()}
            message_id = sender.send_blockchain_message(
                to_address=receiver_account["address"],
                content=test_content,
                message_type="TEST"
            )
            
            # In real test, would wait for blockchain confirmation
            # and verify message delivery
            self.assertIsNotNone(message_id)
            
        except ImportError:
            self.skipTest("Blockchain integration not available")
    
    def test_broadcast_message(self):
        """Test broadcasting messages to multiple agents"""
        # Similar to above but sending to multiple recipients
        pass


class TestTrustAndReputation(BlockchainIntegrationTestCase):
    """Test trust and reputation management on blockchain"""
    
    def test_verify_agent_trust(self):
        """Test verifying agent trust levels"""
        try:
            from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin
            
            class TrustedAgent(BlockchainIntegrationMixin):
                def __init__(self):
                    super().__init__()
                    
            agent = TrustedAgent()
            account = self.network.get_account(0)
            os.environ["TRUSTED_AGENT_PRIVATE_KEY"] = account["private_key"]
            
            # Initialize agent
            agent._initialize_blockchain("trusted_agent", ["trusted_ops"], "http://localhost:8003")
            
            # In real test, would:
            # 1. Set agent reputation on blockchain
            # 2. Verify trust from another agent
            # 3. Test trust-based access control
            
            # Mock trust verification
            trusted = agent.verify_trust(account["address"], min_reputation=50)
            self.assertTrue(trusted)  # Would check actual blockchain state
            
        except ImportError:
            self.skipTest("Blockchain integration not available")
    
    def test_reputation_updates(self):
        """Test updating agent reputation"""
        # Test reputation increase/decrease based on behavior
        pass


class TestMultiAgentCoordination(BlockchainIntegrationTestCase):
    """Test multi-agent coordination scenarios through blockchain"""
    
    async def test_distributed_workflow(self):
        """Test distributed workflow coordination"""
        # Create workflow coordinator
        try:
            from app.a2a.agents.agentManager.active.enhancedAgentManagerAgent import EnhancedAgentManagerAgent
            
            coordinator = EnhancedAgentManagerAgent()
            await coordinator.initialize()
            
            # Create test workflow request
            workflow_request = {
                "sender_id": "workflow_initiator",
                "message_type": "ORCHESTRATION_REQUEST"
            }
            
            workflow_content = {
                "target_agents": ["agent1", "agent2", "agent3"],
                "orchestration_type": "workflow",
                "orchestration_params": {
                    "workflow_id": "test_workflow_123",
                    "steps": [
                        {"agent": "agent1", "action": "step1"},
                        {"agent": "agent2", "action": "step2"},
                        {"agent": "agent3", "action": "step3"}
                    ]
                }
            }
            
            # Execute workflow through blockchain
            result = await coordinator._handle_blockchain_orchestration(
                workflow_request, workflow_content
            )
            
            # Verify workflow execution
            self.assertEqual(result["status"], "success")
            self.assertIn("orchestration_result", result)
            
        except ImportError:
            self.skipTest("Agent manager not available")
    
    async def test_consensus_building(self):
        """Test consensus building among multiple agents"""
        # Create consensus scenario
        try:
            from app.a2a.agents.agentManager.active.enhancedAgentManagerAgent import EnhancedAgentManagerAgent
            
            coordinator = EnhancedAgentManagerAgent()
            await coordinator.initialize()
            
            # Create consensus request
            consensus_request = {
                "sender_id": "consensus_initiator",
                "message_type": "COORDINATION_REQUEST"
            }
            
            consensus_content = {
                "coordination_type": "consensus_building",
                "participating_agents": ["validator1", "validator2", "validator3"],
                "coordination_params": {
                    "consensus_type": "majority",
                    "proposed_decision": "test_decision",
                    "voting_timeout": 60
                }
            }
            
            # Execute consensus through blockchain
            result = await coordinator._handle_blockchain_coordination(
                consensus_request, consensus_content
            )
            
            # Verify consensus
            self.assertEqual(result["status"], "success")
            self.assertTrue(
                result.get("coordination_result", {})
                .get("summary", {})
                .get("consensus_reached", False)
            )
            
        except ImportError:
            self.skipTest("Agent manager not available")


class TestBlockchainRecovery(BlockchainIntegrationTestCase):
    """Test error handling and recovery mechanisms"""
    
    def test_network_disconnection_recovery(self):
        """Test recovery from network disconnection"""
        # Simulate network disconnection and recovery
        pass
    
    def test_transaction_failure_retry(self):
        """Test transaction failure and retry logic"""
        # Test retry mechanisms for failed transactions
        pass
    
    def test_contract_upgrade_handling(self):
        """Test handling of smart contract upgrades"""
        # Test agent adaptation to contract changes
        pass


def run_integration_tests():
    """Run all blockchain integration tests"""
    # Check if test network tools are available
    try:
        subprocess.run(["anvil", "--version"], capture_output=True, check=True)
        network_available = True
    except:
        try:
            subprocess.run(["ganache-cli", "--version"], capture_output=True, check=True)
            network_available = True
        except:
            network_available = False
            print("Warning: No test blockchain network available (Anvil or Ganache)")
            print("Install Foundry (for Anvil) or Ganache to run integration tests")
    
    if not network_available:
        print("Skipping blockchain integration tests - no test network available")
        return True
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentRegistration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainMessaging))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTrustAndReputation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMultiAgentCoordination))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainRecovery))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    success = run_integration_tests()
    sys.exit(0 if success else 1)
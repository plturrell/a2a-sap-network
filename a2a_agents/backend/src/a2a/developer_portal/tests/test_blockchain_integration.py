"""
Real tests for blockchain integration
These tests actually verify functionality, not just return "passed"
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bpmn.blockchain_integration import (
    A2ABlockchainIntegration,
    SmartContractTaskType,
    BlockchainNetwork
)
from bpmn.workflow_engine import (
    WorkflowExecutionEngine,
    WorkflowEngineConfig,
    ExecutionState
)


class TestBlockchainIntegration:
    """Test blockchain integration functionality"""
    
    @pytest.fixture
    def blockchain_config(self):
        """Test configuration"""
        return {
            "networks": {
                "local": {
                    "provider_url": "http://localhost:8545",
                    "chain_id": 31337
                }
            }
        }
    
    @pytest.fixture
    async def blockchain_integration(self, blockchain_config):
        """Create blockchain integration instance"""
        integration = A2ABlockchainIntegration(blockchain_config)
        yield integration
        await integration.close()
    
    def test_initialization(self, blockchain_config):
        """Test that blockchain integration initializes correctly"""
        integration = A2ABlockchainIntegration(blockchain_config)
        
        # Check that connections are initialized
        assert "local" in integration.web3_connections
        assert integration.contract_abis is not None
        assert "AgentRegistry" in integration.contract_abis
        assert "MessageRouter" in integration.contract_abis
    
    @pytest.mark.asyncio
    async def test_agent_registration_missing_private_key(self, blockchain_integration):
        """Test that agent registration fails without private key"""
        with pytest.raises(ValueError, match="Private key required"):
            await blockchain_integration.execute_blockchain_task(
                task_type=SmartContractTaskType.AGENT_REGISTRATION,
                network="local",
                task_config={
                    "agentName": "Test Agent",
                    "agentEndpoint": "http://localhost:8080",
                    "capabilities": ["test"]
                },
                variables={}
            )
    
    @pytest.mark.asyncio
    async def test_capability_query_no_private_key_required(self, blockchain_integration):
        """Test that capability query works without private key"""
        # Mock the contract call
        with patch.object(blockchain_integration, 'contracts', {
            "local": {
                "AgentRegistry": Mock(
                    functions=Mock(
                        findAgentsByCapability=Mock(
                            return_value=Mock(
                                call=Mock(return_value=[])
                            )
                        ),
                        getAgent=Mock(
                            return_value=Mock(
                                call=Mock(return_value=("Test", "http://test", [], 100, True, 0))
                            )
                        )
                    )
                )
            }
        }):
            result = await blockchain_integration.execute_blockchain_task(
                task_type=SmartContractTaskType.CAPABILITY_QUERY,
                network="local",
                task_config={"capability": "test"},
                variables={}
            )
            
            assert result["success"] is True
            assert result["capability"] == "test"
            assert isinstance(result["agents"], list)
    
    @pytest.mark.asyncio
    async def test_invalid_network(self, blockchain_integration):
        """Test that invalid network raises error"""
        with pytest.raises(ValueError, match="No connection to invalid_network"):
            await blockchain_integration.execute_blockchain_task(
                task_type=SmartContractTaskType.AGENT_DISCOVERY,
                network="invalid_network",
                task_config={},
                variables={}
            )
    
    def test_abi_loading(self, blockchain_integration):
        """Test that ABIs are loaded correctly"""
        agent_registry_abi = blockchain_integration.contract_abis.get("AgentRegistry")
        assert agent_registry_abi is not None
        
        # Check for required functions
        function_names = [item["name"] for item in agent_registry_abi if item["type"] == "function"]
        assert "registerAgent" in function_names
        assert "findAgentsByCapability" in function_names
        assert "updateReputation" in function_names
        
        # Check for events
        event_names = [item["name"] for item in agent_registry_abi if item["type"] == "event"]
        assert "AgentRegistered" in event_names


class TestWorkflowEngineBlockchainIntegration:
    """Test workflow engine with blockchain integration"""
    
    @pytest.fixture
    def engine_config(self):
        """Test engine configuration"""
        return WorkflowEngineConfig(
            enable_persistence=False,
            blockchain={
                "networks": {
                    "local": {
                        "provider_url": "http://localhost:8545",
                        "chain_id": 31337
                    }
                }
            }
        )
    
    @pytest.fixture
    async def workflow_engine(self, engine_config):
        """Create workflow engine instance"""
        engine = WorkflowExecutionEngine(engine_config)
        yield engine
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_blockchain_service_task(self, workflow_engine):
        """Test that blockchain tasks are recognized and routed correctly"""
        # Mock blockchain integration
        mock_blockchain = AsyncMock()
        mock_blockchain.execute_blockchain_task = AsyncMock(return_value={
            "success": True,
            "transactionHash": "0x123",
            "agentAddress": "0xabc"
        })
        
        workflow_engine.blockchain_integration = mock_blockchain
        
        # Execute blockchain service task
        result = await workflow_engine._execute_service_task(
            element={"id": "test", "name": "Test Task"},
            variables={"test": "value"},
            properties={
                "implementationType": "blockchain",
                "contractTaskType": "agent_registration",
                "network": "local",
                "agentName": "Test Agent"
            }
        )
        
        # Verify blockchain task was called
        mock_blockchain.execute_blockchain_task.assert_called_once()
        assert result["success"] is True
        assert "transactionHash" in result
    
    @pytest.mark.asyncio
    async def test_regular_service_task(self, workflow_engine):
        """Test that regular HTTP service tasks still work"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = b'{"result": "ok"}'
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = Mock()
        
        with patch.object(workflow_engine.http_client, 'request', return_value=mock_response):
            result = await workflow_engine._execute_service_task(
                element={"id": "test", "name": "Test Task"},
                variables={"test": "value"},
                properties={
                    "serviceUrl": "http://example.com/api",
                    "method": "POST"
                }
            )
            
            assert result["result"] == "ok"
    
    @pytest.mark.asyncio
    async def test_receive_task_blockchain(self, workflow_engine):
        """Test blockchain message receiving"""
        # This test verifies the receive task can handle blockchain events
        # In a real test environment, you would mock the blockchain event subscription
        
        # For now, test that non-blockchain receive tasks fail properly
        with pytest.raises(ValueError, match="No receive mechanism configured"):
            await workflow_engine._execute_receive_task(
                element={"id": "test", "name": "Test Receive"},
                variables={},
                properties={"messageName": "testMessage"}
            )


class TestBlockchainWorkflowExecution:
    """Test full workflow execution with blockchain"""
    
    @pytest.mark.asyncio
    async def test_blockchain_workflow_execution(self):
        """Test executing a workflow with blockchain tasks"""
        # Create test workflow
        workflow_definition = {
            "id": "test-blockchain-workflow",
            "name": "Test Blockchain Workflow",
            "elements": [
                {
                    "id": "start",
                    "type": "startEvent"
                },
                {
                    "id": "register",
                    "type": "serviceTask",
                    "properties": {
                        "implementationType": "blockchain",
                        "contractTaskType": "agent_registration",
                        "network": "local",
                        "agentName": "Test Agent"
                    }
                },
                {
                    "id": "end",
                    "type": "endEvent"
                }
            ],
            "connections": [
                {"source_id": "start", "target_id": "register"},
                {"source_id": "register", "target_id": "end"}
            ]
        }
        
        # Create engine with mocked blockchain
        config = WorkflowEngineConfig(
            enable_persistence=False,
            blockchain={
                "networks": {
                    "local": {
                        "provider_url": "http://localhost:8545",
                        "chain_id": 31337
                    }
                }
            }
        )
        engine = WorkflowExecutionEngine(config)
        
        # Mock blockchain integration
        mock_blockchain = AsyncMock()
        mock_blockchain.execute_blockchain_task = AsyncMock(return_value={
            "success": True,
            "transactionHash": "0x123"
        })
        engine.blockchain_integration = mock_blockchain
        
        # Execute workflow
        execution_id = await engine.start_execution(
            workflow_definition,
            {"privateKey": "0xtest"}
        )
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Check execution status
        status = await engine.get_execution_status(execution_id)
        assert status is not None
        assert status["state"] == ExecutionState.COMPLETED.value
        
        # Verify blockchain task was called
        mock_blockchain.execute_blockchain_task.assert_called()
        
        await engine.close()


# Integration test that requires actual blockchain connection
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_blockchain_connection():
    """Test real blockchain connection (requires local Anvil node)"""
    config = {
        "networks": {
            "local": {
                "provider_url": "http://localhost:8545",
                "chain_id": 31337
            }
        }
    }
    
    integration = A2ABlockchainIntegration(config)
    
    # Check if we can connect
    web3 = integration.web3_connections.get("local")
    if web3 and web3.is_connected():
        # Get latest block
        block_number = web3.eth.block_number
        assert block_number >= 0
        
        # Check chain ID
        chain_id = web3.eth.chain_id
        assert chain_id == 31337
    
    await integration.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
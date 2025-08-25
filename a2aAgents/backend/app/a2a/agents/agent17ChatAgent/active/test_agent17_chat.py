"""
Test suite for Agent 17 Chat Agent
Tests A2A protocol compliance and blockchain-only communication
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from agent17ChatAgentSdk import Agent17ChatAgent, create_agent17_chat_agent
from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole


class TestAgent17ChatAgent:
    """Test suite for Agent 17 Chat Agent"""
    
    @pytest.fixture
    async def agent(self):
        """Create test agent instance"""
        blockchain_config = {
            "private_key": "test_private_key",
            "contract_address": "0xtest",
            "rpc_url": "http://localhost:8545"
        }
        
        agent = create_agent17_chat_agent(
            base_url="http://localhost:8017",
            blockchain_config=blockchain_config
        )
        
        # Mock blockchain client
        agent.blockchain_client = AsyncMock()
        agent.blockchain_client.register_agent = AsyncMock(return_value="0xtxhash")
        agent.blockchain_client.get_registered_agents = AsyncMock(return_value=[])
        agent.blockchain_client.send_message = AsyncMock(return_value="0xmsghash")
        agent.blockchain_client.get_messages_for_agent = AsyncMock(return_value=[])
        
        yield agent
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        await agent.initialize()
        
        # Verify blockchain registration was called
        agent.blockchain_client.register_agent.assert_called_once()
        
        # Verify agent properties
        assert agent.AGENT_ID == "agent17_chat"
        assert agent.AGENT_NAME == "A2A Chat Interface Agent"
        assert agent.AGENT_VERSION == "1.0.0"
        assert "conversational_interface" in agent.config.allowed_operations
    
    @pytest.mark.asyncio
    async def test_intent_analysis(self, agent):
        """Test intent analysis functionality"""
        # Test various prompts
        test_cases = [
            ("I need to standardize some financial data", ["agent1_standardization"]),
            ("Can you help me calculate some values?", ["agent10_calculator", "agent4_calc_validation"]),
            ("I want to search for similar documents", ["agent3_vector_processing"]),
            ("Build me a new agent", ["agent7_builder"]),
            ("Show me the data catalog", ["agent11_catalog"]),
        ]
        
        for prompt, expected_agents in test_cases:
            result = await agent._analyze_intent(prompt)
            
            assert result["intent_type"] in ["analytical", "creative", "search", "general"]
            assert len(result["recommended_agents"]) > 0
            assert any(agent in result["recommended_agents"] for agent in expected_agents)
            assert 0 <= result["confidence"] <= 1
            assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_blockchain_routing(self, agent):
        """Test routing via blockchain only"""
        prompt = "Process this data"
        target_agents = ["agent0_data_product", "agent1_standardization"]
        conversation_id = "test_conv_123"
        
        # Route messages
        results = await agent._route_via_blockchain(prompt, target_agents, conversation_id)
        
        # Verify blockchain messages sent
        assert len(results) == len(target_agents)
        assert agent.blockchain_client.send_message.call_count == len(target_agents)
        
        for result in results:
            assert result["status"] == "sent"
            assert "tx_hash" in result
            assert result["tx_hash"] == "0xmsghash"
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_parallel(self, agent):
        """Test parallel multi-agent coordination"""
        query = "Analyze and process this financial data"
        target_agents = ["agent0_data_product", "agent1_standardization", "agent2_ai_preparation"]
        
        result = await agent._coordinate_agents(
            query, target_agents, "parallel", "test_context"
        )
        
        assert result["coordination_type"] == "parallel"
        assert len(result["results"]) == len(target_agents)
        assert agent.blockchain_client.send_message.call_count == len(target_agents)
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_sequential(self, agent):
        """Test sequential multi-agent coordination"""
        query = "Step by step data processing"
        target_agents = ["agent0_data_product", "agent1_standardization"]
        
        result = await agent._coordinate_agents(
            query, target_agents, "sequential", "test_context"
        )
        
        assert result["coordination_type"] == "sequential"
        assert len(result["results"]) == len(target_agents)
        # In sequential mode, each agent gets previous results
        assert agent.blockchain_client.send_message.call_count == len(target_agents)
    
    @pytest.mark.asyncio
    async def test_conversation_tracking(self, agent):
        """Test conversation tracking functionality"""
        conversation_id = "test_conv_456"
        user_id = "test_user"
        
        # Track conversation
        await agent._track_conversation(conversation_id, user_id)
        
        assert conversation_id in agent.active_conversations
        assert agent.active_conversations[conversation_id]["user_id"] == user_id
        assert agent.active_conversations[conversation_id]["message_count"] == 1
        
        # Track another message
        await agent._track_conversation(conversation_id, user_id)
        assert agent.active_conversations[conversation_id]["message_count"] == 2
    
    @pytest.mark.asyncio
    async def test_secure_message_handler(self, agent):
        """Test secure message handler compliance"""
        await agent.initialize()
        
        # Test message creation would go here if needed
        # Currently just testing handler existence and decoration
        
        # Mock the handler
        handler = agent._handlers.get("chat_message")
        assert handler is not None
        
        # Handler should be decorated with security
        assert hasattr(handler, "__wrapped__")
    
    @pytest.mark.asyncio
    async def test_blockchain_message_listener(self, agent):
        """Test blockchain message listener"""
        # Mock blockchain messages
        mock_messages = [
            {
                "from_agent": "agent0_data_product",
                "message_id": "msg_123",
                "data": {
                    "operation": "response",
                    "response": {"status": "success", "data": "processed"}
                }
            }
        ]
        
        agent.blockchain_client.get_messages_for_agent.return_value = mock_messages
        
        # Process one cycle of the listener
        await agent._blockchain_message_listener()
        
        # Should have retrieved messages
        agent.blockchain_client.get_messages_for_agent.assert_called_with(agent.AGENT_ID)
        
        # Stats should be updated
        assert agent.stats["blockchain_messages_received"] > 0
    
    @pytest.mark.asyncio
    async def test_no_http_communication(self, agent):
        """Test that agent does not use HTTP for agent communication"""
        # Ensure no HTTP client attributes
        assert not hasattr(agent, 'http_client')
        assert not hasattr(agent, 'session')
        assert not hasattr(agent, 'websocket')
        
        # Verify blockchain-only communication
        await agent._route_via_blockchain("test", ["agent0"], "ctx")
        
        # Only blockchain client should be used
        agent.blockchain_client.send_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_agent_statistics(self, agent):
        """Test agent statistics tracking"""
        # Perform some operations
        await agent._track_conversation("conv1", "user1")
        await agent._route_via_blockchain("test", ["agent0"], "ctx")
        
        stats = await agent.get_statistics()
        
        assert "statistics" in stats
        assert stats["statistics"]["blockchain_messages_sent"] > 0
        assert stats["active_conversations"] == 1
        assert stats["agent_id"] == agent.AGENT_ID
    
    @pytest.mark.asyncio
    async def test_security_compliance(self, agent):
        """Test security compliance with SecureA2AAgent"""
        # Verify security configuration
        assert agent.config.enable_authentication
        assert agent.config.enable_rate_limiting
        assert agent.config.enable_input_validation
        
        # Verify allowed operations
        assert "chat_message" in agent.config.allowed_operations
        assert "analyze_intent" in agent.config.allowed_operations
        assert "multi_agent_query" in agent.config.allowed_operations
    
    @pytest.mark.asyncio
    async def test_mcp_tools(self, agent):
        """Test MCP tool functionality"""
        # Test conversation analysis tool
        conversation_id = "test_conv"
        await agent._track_conversation(conversation_id, "user1")
        
        result = await agent.analyze_conversation(conversation_id)
        
        assert "conversation_id" in result
        assert result["conversation_id"] == conversation_id
        assert "message_count" in result
        assert "user_id" in result


class TestAgent17Integration:
    """Integration tests for Agent 17"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_message_flow(self):
        """Test complete message flow through Agent 17"""
        # This would test against a real blockchain test network
        # Marked as integration test to skip in unit test runs
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_agent_orchestration(self):
        """Test orchestrating multiple agents through chat interface"""
        # This would test real multi-agent coordination
        # Marked as integration test to skip in unit test runs
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
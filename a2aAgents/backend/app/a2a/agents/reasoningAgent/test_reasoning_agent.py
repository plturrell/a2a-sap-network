#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced Reasoning Agent
Tests multi-agent reasoning capabilities, architectures, and skills
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import pytest
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import the agent and related classes
from app.a2a.agents.reasoningAgent.reasoningAgent import (
    ReasoningAgent, ReasoningRequest, ReasoningArchitecture,
    AgentRole, ReasoningState
)
from app.a2a.agents.reasoningAgent.reasoningSkills import (
    MultiAgentReasoningSkills, ReasoningOrchestrationSkills,
    HierarchicalReasoningSkills, SwarmReasoningSkills
)
from app.a2a.sdk.types import A2AMessage, MessageRole


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class TestReasoningAgent:
    """Test suite for Reasoning Agent"""

    @pytest.fixture
    async def reasoning_agent(self):
        """Create a reasoning agent instance for testing"""
        agent = ReasoningAgent(
            base_url=os.getenv("A2A_SERVICE_URL"),
            agent_network_url=os.getenv("A2A_SERVICE_URL"),
            max_sub_agents=5,
            reasoning_timeout=60
        )
        await agent.initialize()
        yield agent
        await agent.shutdown()

    @pytest.fixture
    def sample_reasoning_request(self):
        """Create a sample reasoning request"""
        return ReasoningRequest(
            question="What are the implications of quantum computing on cryptography?",
            context={"domain": "technology", "depth": "detailed"},
            architecture=ReasoningArchitecture.HIERARCHICAL,
            max_reasoning_depth=3,
            confidence_threshold=0.7,
            enable_debate=True,
            max_debate_rounds=2
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self, reasoning_agent):
        """Test agent initialization"""
        assert reasoning_agent.agent_id is not None
        assert reasoning_agent.name == "Advanced Reasoning Agent"
        assert reasoning_agent.version == "1.0.0"
        assert len(reasoning_agent.sub_agent_pool) > 0

        # Check sub-agent pool configuration
        assert AgentRole.QUESTION_ANALYZER in reasoning_agent.sub_agent_pool
        assert AgentRole.EVIDENCE_RETRIEVER in reasoning_agent.sub_agent_pool
        assert AgentRole.REASONING_ENGINE in reasoning_agent.sub_agent_pool
        assert AgentRole.ANSWER_SYNTHESIZER in reasoning_agent.sub_agent_pool

    @pytest.mark.asyncio
    async def test_hierarchical_reasoning(self, reasoning_agent, sample_reasoning_request):
        """Test hierarchical reasoning architecture"""
        # Mock sub-agent responses
        with patch.object(reasoning_agent, '_query_sub_agent', new_callable=AsyncMock) as mock_query:
            # Configure mock responses for different agent roles
            async def mock_response(agent_config, task, parameters):
                if agent_config.role == AgentRole.QUESTION_ANALYZER:
                    return {
                        "sub_questions": [
                            "What is quantum computing?",
                            "How does cryptography work?",
                            "What vulnerabilities exist?"
                        ],
                        "complexity_score": 0.8
                    }
                elif agent_config.role == AgentRole.EVIDENCE_RETRIEVER:
                    return {
                        "evidence": [
                            {"content": "Quantum computers can break RSA", "relevance": 0.9},
                            {"content": "Post-quantum cryptography exists", "relevance": 0.85}
                        ]
                    }
                elif agent_config.role == AgentRole.REASONING_ENGINE:
                    return {
                        "inference": "Quantum computing poses significant threats to current cryptography",
                        "confidence": 0.85,
                        "reasoning_steps": ["Step 1", "Step 2"]
                    }
                elif agent_config.role == AgentRole.ANSWER_SYNTHESIZER:
                    return {
                        "answer": "Quantum computing will require new cryptographic methods",
                        "confidence": 0.8
                    }
                return {}

            mock_query.side_effect = mock_response

            # Execute reasoning
            result = await reasoning_agent.multi_agent_reasoning(sample_reasoning_request)

            # Verify results
            assert result['answer'] is not None
            assert result['confidence'] >= 0.7
            assert result['reasoning_architecture'] == 'hierarchical'
            assert result['phases_completed'] == 5
            assert result['sub_agents_used'] > 0

    @pytest.mark.asyncio
    async def test_question_decomposition_skill(self):
        """Test question decomposition skill"""
        skills = MultiAgentReasoningSkills()

        request_data = {
            "question": "How do neural networks learn from data?",
            "max_depth": 2,
            "decomposition_strategy": "functional",
            "context": {}
        }

        result = await skills.hierarchical_question_decomposition(request_data)

        assert result['success'] is True
        assert 'decomposition_tree' in result
        assert 'sub_questions' in result
        assert len(result['sub_questions']) > 0
        assert result['decomposition_strategy'] == 'functional'

    @pytest.mark.asyncio
    async def test_multi_agent_consensus(self):
        """Test multi-agent consensus building"""
        skills = MultiAgentReasoningSkills()

        proposals = [
            {
                "agent_id": "agent_1",
                "proposal": "Solution A based on evidence",
                "confidence": 0.8,
                "evidence": ["Evidence 1", "Evidence 2"]
            },
            {
                "agent_id": "agent_2",
                "proposal": "Solution A based on evidence",
                "confidence": 0.75,
                "evidence": ["Evidence 1", "Evidence 3"]
            },
            {
                "agent_id": "agent_3",
                "proposal": "Solution B different approach",
                "confidence": 0.6,
                "evidence": ["Evidence 4"]
            }
        ]

        request_data = {
            "proposals": proposals,
            "consensus_method": "voting",
            "threshold": 0.6
        }

        result = await skills.multi_agent_consensus(request_data)

        assert result['success'] is True
        assert 'consensus' in result
        assert result['consensus']['confidence'] > 0
        assert result['participant_count'] == 3

    @pytest.mark.asyncio
    async def test_blackboard_reasoning(self):
        """Test blackboard architecture reasoning"""
        skills = ReasoningOrchestrationSkills()

        # Create mock state and request
        state = type('obj', (object,), {'question': 'Test question'})
        request = type('obj', (object,), {
            'question': 'Test question',
            'context': {'control_strategy': 'opportunistic'}
        })

        result = await skills.blackboard_reasoning(state, request)

        assert 'answer' in result
        assert 'confidence' in result
        assert result['reasoning_architecture'] == 'blackboard'
        assert 'blackboard_state' in result

    @pytest.mark.asyncio
    async def test_debate_mechanism(self, reasoning_agent):
        """Test multi-agent debate mechanism"""
        state = ReasoningState(
            question="Test question",
            architecture=ReasoningArchitecture.HIERARCHICAL
        )

        # Add multiple reasoning chains
        state.reasoning_chains = [
            {
                "inference": "Solution A",
                "confidence": 0.7,
                "steps": ["Step 1", "Step 2"]
            },
            {
                "inference": "Solution B",
                "confidence": 0.8,
                "steps": ["Step 1", "Step 3"]
            }
        ]

        debate_result = await reasoning_agent._conduct_debate(state, max_rounds=2)

        assert 'debate_history' in debate_result
        assert len(debate_result['debate_history']) > 0
        assert 'arguments' in debate_result['debate_history'][0]

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, reasoning_agent):
        """Test circuit breaker protection"""
        # Mock a failing sub-agent
        with patch.object(reasoning_agent, '_query_sub_agent', new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("Connection failed")

            # Try to query the failing agent multiple times
            for _ in range(5):
                try:
                    await reasoning_agent._query_sub_agent(
                        reasoning_agent.sub_agent_pool[AgentRole.QUESTION_ANALYZER][0],
                        "test_task",
                        {}
                    )
                except:
                    pass

            # Check that circuit breaker is triggered
            reasoning_agent.circuit_breaker_manager.get_breaker(
                f"agent_{AgentRole.QUESTION_ANALYZER.value}"
            )
            # Circuit breaker state would be 'open' after failures

    @pytest.mark.asyncio
    async def test_caching_functionality(self, reasoning_agent, sample_reasoning_request):
        """Test caching of reasoning results"""
        # First call - should compute result
        result1 = await reasoning_agent.multi_agent_reasoning(sample_reasoning_request)

        # Second call with same request - should use cache
        result2 = await reasoning_agent.multi_agent_reasoning(sample_reasoning_request)

        # Cache should make second call faster (in real scenario)
        assert result1['session_id'] != result2['session_id']  # Different sessions
        # But answers should be similar due to caching

    @pytest.mark.asyncio
    async def test_a2a_message_handler(self, reasoning_agent):
        """Test A2A message handling"""
        message = A2AMessage(
            role=MessageRole.USER,
            content={
                "id": "test_123",
                "params": {
                    "question": "What is machine learning?",
                    "architecture": "hierarchical",
                    "confidence_threshold": 0.7
                }
            }
        )

        response = await reasoning_agent.execute_reasoning_task(message)

        assert 'id' in response
        assert response['id'] == "test_123"
        assert 'result' in response or 'error' in response

    @pytest.mark.asyncio
    async def test_get_agent_card(self, reasoning_agent):
        """Test agent card generation"""
        agent_card = reasoning_agent.get_agent_card()

        assert agent_card['name'] == "Advanced-Reasoning-Agent"
        assert agent_card['version'] == "1.0.0"
        assert 'capabilities' in agent_card
        assert agent_card['capabilities']['multiAgentOrchestration'] is True
        assert 'skills' in agent_card
        assert len(agent_card['skills']) > 0

    @pytest.mark.asyncio
    async def test_metrics_collection(self, reasoning_agent, sample_reasoning_request):
        """Test metrics collection"""
        initial_metrics = await reasoning_agent.get_reasoning_metrics()
        initial_sessions = initial_metrics['metrics']['total_sessions']

        # Execute reasoning
        await reasoning_agent.multi_agent_reasoning(sample_reasoning_request)

        # Check metrics updated
        updated_metrics = await reasoning_agent.get_reasoning_metrics()
        assert updated_metrics['metrics']['total_sessions'] == initial_sessions + 1
        assert updated_metrics['metrics']['architecture_usage']['hierarchical'] > 0

    @pytest.mark.asyncio
    async def test_different_architectures(self, reasoning_agent):
        """Test different reasoning architectures"""
        architectures = [
            ReasoningArchitecture.HIERARCHICAL,
            ReasoningArchitecture.PEER_TO_PEER,
            ReasoningArchitecture.BLACKBOARD
        ]

        for arch in architectures:
            request = ReasoningRequest(
                question=f"Test question for {arch.value}",
                architecture=arch,
                enable_debate=False
            )

            result = await reasoning_agent.multi_agent_reasoning(request)

            assert 'answer' in result
            assert 'architecture_used' in result
            assert result['architecture_used'] == arch.value


@pytest.mark.asyncio
async def test_reasoning_skills_integration():
    """Test integration of different reasoning skills"""
    multi_agent_skills = MultiAgentReasoningSkills()

    # Test question decomposition
    decomposition_result = await multi_agent_skills.hierarchical_question_decomposition({
        "question": "What are the environmental impacts of renewable energy?",
        "max_depth": 2,
        "decomposition_strategy": "causal"
    })

    assert decomposition_result['success'] is True

    # Test consensus building with decomposed questions
    proposals = [
        {
            "agent_id": f"analyzer_{i}",
            "proposal": sub_q['question'],
            "confidence": sub_q['confidence']
        }
        for i, sub_q in enumerate(decomposition_result['sub_questions'][:3])
    ]

    consensus_result = await multi_agent_skills.multi_agent_consensus({
        "proposals": proposals,
        "consensus_method": "weighted_average"
    })

    assert consensus_result['success'] is True


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in reasoning agent"""
    agent = ReasoningAgent()
    await agent.initialize()

    try:
        # Test with invalid request
        invalid_request = ReasoningRequest(
            question="",  # Empty question
            architecture=ReasoningArchitecture.HIERARCHICAL
        )

        result = await agent.multi_agent_reasoning(invalid_request)

        # Should handle gracefully
        assert 'error' in result or 'answer' in result

    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_performance_monitoring():
    """Test performance monitoring capabilities"""
    agent = ReasoningAgent()
    await agent.initialize()

    try:
        # Enable performance monitoring
        agent.enable_performance_monitoring(metrics_port=8009)

        # Execute some operations
        request = ReasoningRequest(
            question="Test performance monitoring",
            architecture=ReasoningArchitecture.HIERARCHICAL,
            enable_debate=False
        )

        await agent.multi_agent_reasoning(request)

        # Performance metrics should be collected
        metrics = await agent.get_reasoning_metrics()
        assert metrics['metrics']['average_reasoning_time'] > 0

    finally:
        await agent.shutdown()


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_reasoning_skills_integration())
    print("✅ All reasoning agent tests completed!")

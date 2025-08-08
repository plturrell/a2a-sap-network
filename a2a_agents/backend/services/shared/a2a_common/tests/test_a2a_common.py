"""
Tests for A2A Common Library
"""

import pytest
from a2a_common import A2AAgentBase, A2AMessage, MessageRole
from a2a_common.sdk.utils import create_success_response, create_error_response


class TestA2ACommon:
    """Test A2A common library components"""
    
    def test_create_success_response(self):
        """Test success response creation"""
        response = create_success_response({"data": "test"})
        assert response["status"] == "success"
        assert response["data"]["data"] == "test"
    
    def test_create_error_response(self):
        """Test error response creation"""
        response = create_error_response(404, "Not found")
        assert response["status"] == "error"
        assert response["error"]["code"] == 404
        assert response["error"]["message"] == "Not found"
    
    def test_message_roles(self):
        """Test message role enum"""
        assert MessageRole.USER == "user"
        assert MessageRole.AGENT == "agent"
        assert MessageRole.SYSTEM == "system"
    
    @pytest.mark.asyncio
    async def test_agent_base_initialization(self):
        """Test A2AAgentBase initialization"""
        agent = A2AAgentBase(
            agent_id="test_agent",
            name="Test Agent",
            description="Test Description",
            version="1.0.0",
            base_url="http://localhost:8000"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.version == "1.0.0"
        assert len(agent.handlers) == 0
        assert len(agent.skills) == 0
        assert len(agent.tasks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
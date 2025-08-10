"""
Pytest configuration for a2aAgents tests
"""

import pytest
import sys
import os
from pathlib import Path
import asyncio
from unittest.mock import Mock

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")

# Configure asyncio for tests
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()
    agent.agent_id = "test_agent_123"
    agent.name = "Test Agent"
    agent.version = "1.0.0"
    agent.base_url = "http://localhost:8000"
    agent.get_agent_card.return_value = {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "version": agent.version,
        "capabilities": ["test"],
        "handlers": ["test_handler"]
    }
    return agent


@pytest.fixture
def mock_message():
    """Create a mock A2A message"""
    return {
        "conversation_id": "test_conv_123",
        "from_agent": "agent1",
        "to_agent": "agent2",
        "parts": [
            {
                "kind": "text",
                "text": "Test message content"
            }
        ],
        "timestamp": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def test_environment():
    """Set up test environment variables"""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["A2A_ENV"] = "test"
    os.environ["A2A_AGENTS_PATH"] = str(project_root)
    os.environ["A2A_NETWORK_PATH"] = "/Users/apple/projects/a2a/a2aNetwork"
    os.environ["A2A_PROTOCOL_VERSION"] = "0.2.9"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def network_urls():
    """Provide test network URLs"""
    return {
        "registry": "http://localhost:9000",
        "trust": "http://localhost:9001",
        "sdk": "http://localhost:9002"
    }


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )

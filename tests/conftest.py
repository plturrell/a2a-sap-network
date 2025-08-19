"""
Main pytest configuration for A2A test suite
Handles test discovery, fixtures, and import paths
"""

import sys
import os
from pathlib import Path
import pytest
import asyncio
from unittest.mock import Mock

# Add project paths to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "a2aAgents"))
sys.path.insert(0, str(PROJECT_ROOT / "a2aAgents" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "a2aNetwork"))

# Configure pytest plugins
pytest_plugins = ["pytest_asyncio"]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests requiring external services")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Tests that take more than 5 seconds")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "blockchain: Blockchain-related tests")
    config.addinivalue_line("markers", "agent: Agent-specific tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add markers based on test path
        test_path = str(item.fspath)
        
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in test_path:
            item.add_marker(pytest.mark.e2e)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        elif "/security/" in test_path:
            item.add_marker(pytest.mark.security)
        
        # Add specific markers based on content
        if "blockchain" in test_path.lower():
            item.add_marker(pytest.mark.blockchain)
        if "agent" in test_path.lower():
            item.add_marker(pytest.mark.agent)
        if "network" in test_path.lower() or "hana" in test_path.lower():
            item.add_marker(pytest.mark.network)


# Global fixtures
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
    agent.capabilities = ["test", "mock"]
    agent.get_agent_card.return_value = {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "version": agent.version,
        "capabilities": agent.capabilities,
        "handlers": ["test_handler"]
    }
    return agent


@pytest.fixture
def mock_a2a_message():
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
def test_environment(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("A2A_ENV", "test")
    monkeypatch.setenv("A2A_AGENTS_PATH", str(PROJECT_ROOT / "a2aAgents"))
    monkeypatch.setenv("A2A_NETWORK_PATH", str(PROJECT_ROOT / "a2aNetwork"))
    monkeypatch.setenv("A2A_PROTOCOL_VERSION", "0.2.9")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    yield
    

@pytest.fixture
def network_urls():
    """Provide test network URLs"""
    return {
        "registry": "http://localhost:9000",
        "trust": "http://localhost:9001",
        "sdk": "http://localhost:9002"
    }


@pytest.fixture
def test_db_config():
    """Database configuration for tests"""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "a2a_test",
        "user": "test_user",
        "password": "test_password"
    }


@pytest.fixture
def blockchain_config():
    """Blockchain configuration for tests"""
    return {
        "network": "localhost",
        "port": 8545,
        "chain_id": 31337,
        "gas_limit": 8000000,
        "gas_price": 20000000000
    }


# Skip markers for conditional test execution
def pytest_runtest_setup(item):
    """Skip tests based on markers and conditions"""
    # Skip integration tests unless explicitly requested
    if "integration" in item.keywords and not item.config.getoption("-m"):
        pytest.skip("Integration test - use '-m integration' to run")
    
    # Skip slow tests in quick mode
    if "slow" in item.keywords and item.config.getoption("--quick"):
        pytest.skip("Slow test - skipped in quick mode")
    
    # Skip network tests if offline
    if "network" in item.keywords and os.environ.get("OFFLINE_MODE"):
        pytest.skip("Network test - skipped in offline mode")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only quick tests"
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run all tests including slow and integration"
    )
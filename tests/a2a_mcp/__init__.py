"""
A2A Test Suite MCP Toolset
Model Context Protocol integration for comprehensive test management
"""

__version__ = "1.0.0"
__author__ = "A2A Development Team"
__description__ = "MCP toolset for A2A enterprise test suite management and orchestration"

from .server.test_mcp_server import server
from .tools.test_executor import TestExecutor, TestSuite, TestResult, TestStatus
from .agents.test_orchestrator import TestOrchestrator, TestWorkflow, TestPriority

__all__ = [
    "server",
    "TestExecutor", 
    "TestSuite",
    "TestResult",
    "TestStatus",
    "TestOrchestrator",
    "TestWorkflow", 
    "TestPriority"
]
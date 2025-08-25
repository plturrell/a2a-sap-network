"""
Test suite for Glean Agent
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sqlite3
import tempfile
import shutil

from gleanAgentSdk import (
    GleanAgent, AnalysisType, IssueType, IssueSeverity,
    CodeIssue, AnalysisResult
)
from app.a2a.sdk.types import A2AMessage, MessageRole, MessagePart


@pytest.fixture
async def glean_agent():
    """Create a Glean Agent instance for testing"""
    agent = GleanAgent()
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files"""
    temp_dir = tempfile.mkdtemp()

    # Create sample Python file
    python_file = Path(temp_dir) / "sample.py"
    python_file.write_text("""
def calculate_sum(a, b):
    # This function adds two numbers
    return a + b

def unused_function():
    pass

# Missing docstring
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
""")

    # Create sample JavaScript file
    js_file = Path(temp_dir) / "sample.js"
    js_file.write_text("""
function add(a, b) {
    return a + b;
}

// Unused variable
const unusedVar = 42;

// Missing semicolon
const result = add(1, 2)

// Complex nested function
function complexFunction(arr) {
    return arr.map(x => {
        if (x > 0) {
            return x * 2;
        } else {
            return 0;
        }
    }).filter(x => x > 5);
}
""")

    # Create test file
    test_file = Path(temp_dir) / "test_sample.py"
    test_file.write_text("""
import pytest
from sample import calculate_sum

def test_calculate_sum():
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(-1, 1) == 0

def test_negative_numbers():
    assert calculate_sum(-5, -3) == -8
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestGleanAgent:
    """Test cases for Glean Agent"""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, glean_agent):
        """Test agent is properly initialized"""
        assert glean_agent.agent_id.startswith("glean_agent")
        assert glean_agent.name == "Glean Code Analysis Agent"
        assert glean_agent.version == "1.0.0"
        assert glean_agent.db_path.exists()

    @pytest.mark.asyncio
    async def test_analyze_code_comprehensive(self, glean_agent, temp_project_dir):
        """Test comprehensive code analysis"""
        # Mock the Glean service response
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "dependency_graph": {"nodes": [], "edges": []},
                "refactoring_suggestions": []
            }
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            # Mock subprocess calls for linters
            with patch('subprocess.run') as mock_run:
                # Mock pylint output
                pylint_output = json.dumps([{
                    "type": "warning",
                    "line": 8,
                    "column": 0,
                    "message": "Missing function docstring",
                    "symbol": "missing-docstring",
                    "message-id": "C0111"
                }])

                # Mock different tool outputs
                mock_run.side_effect = [
                    # Tool availability checks
                    Mock(returncode=0),  # pylint available
                    Mock(returncode=0),  # flake8 available
                    Mock(returncode=0),  # eslint available
                    # Actual linting runs
                    Mock(stdout=pylint_output, stderr="", returncode=0),  # pylint
                    Mock(stdout="sample.py:9:1: C901 'complex_function' is too complex (4)\n", stderr="", returncode=0),  # flake8
                    Mock(stdout='[{"filePath":"sample.js","messages":[{"line":9,"column":25,"severity":2,"message":"Missing semicolon","ruleId":"semi"}]}]', stderr="", returncode=0),  # eslint
                    # Test execution
                    Mock(stdout="", stderr="", returncode=0),  # pytest
                    # Security scan
                    Mock(stdout='{"results": []}', stderr="", returncode=0)  # bandit
                ]

                result = await glean_agent.analyze_code_comprehensive(
                    directory=str(temp_project_dir),
                    analysis_types=[AnalysisType.FULL],
                    file_patterns=["*.py", "*.js"]
                )

        assert result["analysis_id"]
        assert result["directory"] == str(temp_project_dir)
        assert "analyses" in result
        assert "summary" in result
        assert result["summary"]["files_analyzed"] >= 2

    @pytest.mark.asyncio
    async def test_lint_analysis(self, glean_agent, temp_project_dir):
        """Test linting analysis specifically"""
        with patch('subprocess.run') as mock_run:
            # Mock tool availability and outputs
            mock_run.side_effect = [
                Mock(returncode=0),  # pylint available
                Mock(returncode=0),  # flake8 available
                Mock(stdout='[]', stderr="", returncode=0),  # pylint
                Mock(stdout="", stderr="", returncode=0),  # flake8
            ]

            result = await glean_agent._perform_lint_analysis(
                str(temp_project_dir),
                ["*.py"]
            )

        assert "files_analyzed" in result
        assert "total_issues" in result
        assert "issues_by_severity" in result
        assert "issues_by_type" in result

    @pytest.mark.asyncio
    async def test_security_analysis(self, glean_agent, temp_project_dir):
        """Test security analysis"""
        with patch('subprocess.run') as mock_run:
            bandit_output = json.dumps({
                "results": [{
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "filename": "sample.py",
                    "line_number": 10,
                    "test_name": "hardcoded_password",
                    "issue_text": "Possible hardcoded password"
                }]
            })

            mock_run.return_value = Mock(stdout=bandit_output, stderr="", returncode=0)

            result = await glean_agent._perform_security_analysis(str(temp_project_dir))

        assert result["total_issues"] == 1
        assert result["critical_issues"] == 1
        assert len(result["issues"]) == 1

    @pytest.mark.asyncio
    async def test_message_handler(self, glean_agent):
        """Test A2A message handling"""
        message = A2AMessage(
            messageId="test-123",
            role=MessageRole.USER,
            parts=[MessagePart(
                kind="data",
                data={
                    "directory": ".",
                    "analysis_types": ["lint"],
                    "file_patterns": ["*.py"]
                }
            )]
        )

        with patch.object(glean_agent, 'analyze_code_comprehensive', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {"analysis_id": "test", "summary": {}}

            result = await glean_agent.handle_analyze_code(message, "context-123")

            assert result["success"] is True
            mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_analysis_history(self, glean_agent):
        """Test retrieving analysis history"""
        # First, store a dummy analysis
        test_results = {
            "analysis_id": "test-123",
            "directory": "/test",
            "timestamp": "2024-01-20T10:00:00",
            "summary": {"files_analyzed": 10, "total_issues": 5},
            "duration": 10.5
        }

        await glean_agent._store_analysis_results("test-123", test_results)

        # Retrieve history
        history = await glean_agent.get_analysis_history(limit=5)

        assert len(history) >= 1
        assert history[0]["id"] == "test-123"
        assert history[0]["directory"] == "/test"

    @pytest.mark.asyncio
    async def test_mcp_tools(self, glean_agent):
        """Test MCP tool registration"""
        tools = glean_agent.list_mcp_tools()
        tool_names = [tool["name"] for tool in tools]

        assert "glean_analyze_dependencies" in tool_names
        assert "glean_run_linters" in tool_names

    @pytest.mark.asyncio
    async def test_issue_mapping(self, glean_agent):
        """Test issue type and severity mapping"""
        # Test pylint mapping
        assert glean_agent._map_pylint_type("error") == IssueType.SYNTAX_ERROR
        assert glean_agent._map_pylint_severity("warning") == IssueSeverity.MEDIUM

        # Test flake8 mapping
        assert glean_agent._map_flake8_type("E501") == IssueType.SYNTAX_ERROR
        assert glean_agent._map_flake8_severity("W292") == IssueSeverity.MEDIUM

        # Test eslint mapping
        assert glean_agent._map_eslint_type(2) == IssueType.SYNTAX_ERROR
        assert glean_agent._map_eslint_severity(1) == IssueSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, glean_agent):
        """Test quality score calculation"""
        analyses = {
            "lint": {
                "total_issues": 10,
                "files_analyzed": 5,
                "issues_by_severity": {
                    "critical": 1,
                    "high": 2,
                    "medium": 3,
                    "low": 4
                }
            },
            "test": {
                "tests_run": 20,
                "tests_passed": 19,
                "tests_failed": 1
            }
        }

        summary = glean_agent._calculate_summary_metrics(analyses)

        assert summary["total_issues"] == 10
        assert summary["critical_issues"] == 3  # critical + high
        assert summary["files_analyzed"] == 5
        assert summary["test_pass_rate"] == 0.95
        assert 0 <= summary["quality_score"] <= 100


@pytest.mark.asyncio
async def test_glean_agent_a2a_compliance():
    """Test A2A protocol compliance"""
    agent = GleanAgent()

    # Test agent card generation
    agent_card = agent.get_agent_card()
    assert agent_card.name == "Glean Code Analysis Agent"
    assert agent_card.protocolVersion == "0.2.9"
    assert "code_analysis" in agent_card.capabilities

    # Test skill registration
    skills = agent.list_skills()
    skill_names = [skill["name"] for skill in skills]
    assert "analyze_code_comprehensive" in skill_names
    assert "get_analysis_history" in skill_names

    # Test handler registration
    assert "analyze_code" in agent.handlers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

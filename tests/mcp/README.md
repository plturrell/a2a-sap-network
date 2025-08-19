# A2A Test Suite MCP Toolset

A comprehensive Model Context Protocol (MCP) toolset for managing and executing the A2A enterprise test suite with intelligent orchestration and automation.

## Overview

This MCP toolset transforms the organized A2A test structure into a powerful, agent-driven test management system that provides:

- **Intelligent Test Orchestration** - AI-powered test execution planning and optimization
- **Multi-Agent Architecture** - Specialized agents for different test types
- **Comprehensive Test Management** - Full lifecycle test execution and reporting
- **Enterprise Integration** - SAP-compliant test workflows and reporting

## Architecture

```
tests/mcp/
├── server/                    # MCP server implementation
│   └── test_mcp_server.py    # Main MCP server with tools and resources
├── tools/                     # Test execution tools
│   └── test_executor.py      # Advanced test execution engine
├── agents/                    # Intelligent test agents
│   └── test_orchestrator.py  # Workflow orchestration and management
└── config/                    # Configuration and documentation
    ├── mcp_config.json       # MCP server and client configuration
    └── README.md             # This documentation
```

## Features

### 🤖 **Intelligent Test Agents**

- **Unit Test Agent** - Handles isolated component testing (Python/JavaScript)
- **Integration Agent** - Manages component interaction testing
- **E2E Agent** - Executes end-to-end UI and workflow tests
- **Contract Agent** - Specialized for smart contract testing (Solidity/Forge)
- **Performance Agent** - Load, stress, and benchmark testing
- **Security Agent** - Security validation and compliance testing

### 🛠️ **MCP Tools**

#### Core Test Management
- `run_tests` - Execute test suites with advanced options
- `discover_tests` - Intelligent test discovery and categorization
- `validate_test_setup` - Environment and dependency validation
- `manage_test_data` - Test data lifecycle management

#### Analysis & Reporting
- `analyze_test_results` - Comprehensive result analysis
- `get_test_coverage` - Coverage reporting and analysis

### 📊 **Resources**

- `test://unit` - Unit test resources and metadata
- `test://integration` - Integration test resources
- `test://e2e` - End-to-end test resources
- `test://performance` - Performance test resources
- `test://security` - Security test resources
- `test://contracts` - Smart contract test resources
- `test://reports` - Test execution reports and coverage

### 🔄 **Workflow Management**

Pre-configured workflows for common scenarios:

- **Quick Validation** - Fast CI/CD validation (unit tests only)
- **Comprehensive Testing** - Full test suite execution
- **Security Audit** - Security-focused test execution
- **Performance Baseline** - Performance and load testing
- **Pre-Deployment** - Complete pre-deployment validation

## Installation & Setup

### 1. Install Dependencies

```bash
# Install MCP SDK
pip install mcp

# Install test dependencies
pip install pytest pytest-cov pytest-xdist
npm install jest cypress

# Install Solidity testing (if using contracts)
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

### 2. Configure MCP Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "a2a-test-suite": {
      "command": "python",
      "args": ["-m", "tests.mcp.server.test_mcp_server"],
      "cwd": "/path/to/a2a/project",
      "env": {
        "PYTHONPATH": "/path/to/a2a/project",
        "A2A_TEST_ROOT": "/path/to/a2a/project/tests"
      }
    }
  }
}
```

### 3. Start MCP Server

```bash
# From project root
cd /Users/apple/projects/a2a
python -m tests.mcp.server.test_mcp_server
```

## Usage Examples

### Basic Test Execution

```python
# Execute unit tests for a2aAgents module
await mcp_client.call_tool("run_tests", {
    "test_type": "unit",
    "module": "a2aAgents",
    "coverage": True,
    "verbose": True
})

# Run comprehensive test suite
await mcp_client.call_tool("run_tests", {
    "test_type": "all",
    "module": "all",
    "coverage": True
})
```

### Workflow Orchestration

```python
# Create and execute a custom workflow
orchestrator = TestOrchestrator(test_root)

# Create workflow
workflow_id = await orchestrator.create_workflow(
    name="Feature Validation",
    test_type="integration",
    module="a2aNetwork",
    priority=TestPriority.HIGH,
    coverage=True,
    parallel=True
)

# Execute workflow
result = await orchestrator.execute_workflow(workflow_id)

# Get detailed status
status = orchestrator.get_workflow_status(workflow_id)
```

### Test Discovery and Analysis

```python
# Discover all available tests
tests = await mcp_client.call_tool("discover_tests", {
    "test_type": "all",
    "include_disabled": False
})

# Validate test environment
validation = await mcp_client.call_tool("validate_test_setup", {
    "check_dependencies": True,
    "check_config": True
})

# Get coverage report
coverage = await mcp_client.call_tool("get_test_coverage", {
    "module": "a2aAgents",
    "format": "html"
})
```

### Resource Access

```python
# Get unit test information
unit_tests = await mcp_client.read_resource("test://unit")

# Get test reports
reports = await mcp_client.read_resource("test://reports")

# Get contract test details
contracts = await mcp_client.read_resource("test://contracts")
```

## Configuration

### Agent Configuration

Customize agent behavior in `mcp_config.json`:

```json
{
  "agent_config": {
    "unit_agent": {
      "max_concurrent": 3,
      "timeout": 300,
      "retry_count": 2
    },
    "performance_agent": {
      "max_concurrent": 1,
      "timeout": 1200,
      "retry_count": 0
    }
  }
}
```

### Workflow Templates

Define custom workflows:

```json
{
  "workflows": {
    "my_custom_workflow": {
      "description": "Custom validation workflow",
      "test_types": ["unit", "integration"],
      "modules": ["a2aAgents"],
      "parallel": true,
      "timeout": 600,
      "coverage": true,
      "priority": "high"
    }
  }
}
```

### Optimization Settings

Configure performance optimization:

```json
{
  "optimization": {
    "parallel_execution": true,
    "test_selection": {
      "enabled": true,
      "impact_analysis": true
    },
    "caching": {
      "enabled": true,
      "ttl_hours": 24
    }
  }
}
```

## Integration with Development Workflow

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: A2A Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install mcp pytest
          npm install
      
      - name: Run MCP Test Suite
        run: |
          python -m tests.mcp.server.test_mcp_server --workflow quick_validation
```

### IDE Integration

Configure your IDE to use MCP tools:

```json
// VS Code settings.json
{
  "mcp.servers": {
    "a2a-test-suite": {
      "command": "python",
      "args": ["-m", "tests.mcp.server.test_mcp_server"]
    }
  }
}
```

## Monitoring and Reporting

### Performance Metrics

The orchestrator tracks and optimizes:

- Test execution duration trends
- Agent utilization and load balancing
- Failure pattern analysis
- Coverage impact assessment
- Resource usage optimization

### Reports

Available report formats:

- **JSON** - Machine-readable test results
- **HTML** - Interactive web reports
- **JUnit XML** - CI/CD integration
- **Text** - Console-friendly summaries

### Notifications

Configure notifications for:

- Test failures
- Coverage thresholds
- Performance regressions
- Agent status changes

## Best Practices

### Test Organization

1. **Use appropriate test types** - Unit for isolation, integration for interactions
2. **Leverage parallel execution** - For independent test suites
3. **Monitor coverage trends** - Maintain high coverage thresholds
4. **Regular performance baselines** - Track performance regressions

### Agent Management

1. **Balance agent loads** - Avoid overloading specialized agents
2. **Monitor agent health** - Track success rates and timeouts
3. **Optimize workflows** - Use orchestrator recommendations
4. **Scale based on demand** - Adjust concurrent limits as needed

### Maintenance

1. **Regular cleanup** - Remove old test artifacts
2. **Update dependencies** - Keep test frameworks current
3. **Review metrics** - Analyze performance trends
4. **Optimize configuration** - Tune based on usage patterns

## Troubleshooting

### Common Issues

1. **MCP Server won't start**
   - Check Python path and dependencies
   - Verify test root directory exists
   - Review environment variables

2. **Tests failing unexpectedly**
   - Validate test environment setup
   - Check dependency versions
   - Review test data and fixtures

3. **Poor performance**
   - Review parallel execution settings
   - Check agent load balancing
   - Optimize test selection

4. **Coverage issues**
   - Verify coverage configuration
   - Check source code paths
   - Review exclusion patterns

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m tests.mcp.server.test_mcp_server
```

### Health Checks

Regular validation commands:

```python
# Validate complete setup
await mcp_client.call_tool("validate_test_setup", {
    "check_dependencies": True,
    "check_config": True
})

# Check agent status
orchestrator.get_agent_status()

# Review performance metrics
orchestrator.get_performance_metrics()
```

## Contributing

When adding new test types or capabilities:

1. **Update agent capabilities** - Add to appropriate agent configs
2. **Extend MCP tools** - Add new tools for specific functionality
3. **Document workflows** - Create template workflows
4. **Test thoroughly** - Validate with existing test suites

## License

MIT License - See project root for details.

---

**Enterprise Ready** | **SAP Compliant** | **AI-Powered** | **Production Tested**
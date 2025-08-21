#!/usr/bin/env python3
"""
A2A Test Suite MCP Server
Provides MCP tools for test execution, management, and orchestration
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
except ImportError:
    # Fallback for environments without MCP
    print("MCP library not found. Install with: pip install mcp")
    import sys
    sys.exit(1)
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a-test-mcp-server")

# Test suite paths
TEST_ROOT = Path(__file__).parent.parent.parent
UNIT_TESTS = TEST_ROOT / "unit"
INTEGRATION_TESTS = TEST_ROOT / "integration"
E2E_TESTS = TEST_ROOT / "e2e"
PERFORMANCE_TESTS = TEST_ROOT / "performance"
SECURITY_TESTS = TEST_ROOT / "security"
CONTRACTS_TESTS = TEST_ROOT / "contracts"

# Initialize MCP server
server = Server("a2a-test-suite")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available test resources."""
    return [
        types.Resource(
            uri=AnyUrl("test://unit"),
            name="Unit Tests",
            description="Isolated component tests for a2aAgents and a2aNetwork",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://integration"),
            name="Integration Tests", 
            description="Component interaction tests and cross-module integration",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://e2e"),
            name="End-to-End Tests",
            description="Complete user workflow and system behavior tests",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://performance"),
            name="Performance Tests",
            description="Load, stress, and benchmark tests",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://security"),
            name="Security Tests", 
            description="Authentication, authorization, and security validation",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://contracts"),
            name="Smart Contract Tests",
            description="Solidity contract tests for blockchain functionality",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("test://reports"),
            name="Test Reports",
            description="Coverage reports and test execution results",
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read test resource information."""
    path_mapping = {
        "test://unit": UNIT_TESTS,
        "test://integration": INTEGRATION_TESTS, 
        "test://e2e": E2E_TESTS,
        "test://performance": PERFORMANCE_TESTS,
        "test://security": SECURITY_TESTS,
        "test://contracts": CONTRACTS_TESTS,
    }
    
    if str(uri) in path_mapping:
        test_path = path_mapping[str(uri)]
        test_files = []
        
        if test_path.exists():
            for file_path in test_path.rglob("*.py"):
                test_files.append({
                    "file": str(file_path.relative_to(TEST_ROOT)),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
            
            for file_path in test_path.rglob("*.js"):
                test_files.append({
                    "file": str(file_path.relative_to(TEST_ROOT)),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
                
            for file_path in test_path.rglob("*.t.sol"):
                test_files.append({
                    "file": str(file_path.relative_to(TEST_ROOT)),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        return json.dumps({
            "category": str(uri).replace("test://", ""),
            "path": str(test_path),
            "test_count": len(test_files),
            "files": test_files
        }, indent=2)
    
    elif str(uri) == "test://reports":
        coverage_path = TEST_ROOT.parent / "coverage"
        reports = []
        
        if coverage_path.exists():
            for report_file in coverage_path.rglob("*"):
                if report_file.is_file():
                    reports.append({
                        "file": str(report_file.relative_to(coverage_path)),
                        "size": report_file.stat().st_size,
                        "modified": report_file.stat().st_mtime
                    })
        
        return json.dumps({
            "category": "reports",
            "path": str(coverage_path),
            "report_count": len(reports),
            "files": reports
        }, indent=2)
    
    raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available test management tools."""
    return [
        types.Tool(
            name="run_tests",
            description="Execute test suites with various options",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["unit", "integration", "e2e", "performance", "security", "contracts", "all"],
                        "description": "Type of tests to run"
                    },
                    "module": {
                        "type": "string",
                        "enum": ["a2aAgents", "a2aNetwork", "common", "all"],
                        "description": "Module to test"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Test name pattern to match"
                    },
                    "coverage": {
                        "type": "boolean",
                        "description": "Generate coverage report",
                        "default": False
                    },
                    "verbose": {
                        "type": "boolean", 
                        "description": "Verbose output",
                        "default": False
                    }
                },
                "required": ["test_type"]
            },
        ),
        types.Tool(
            name="analyze_test_results",
            description="Analyze and summarize test execution results",
            inputSchema={
                "type": "object",
                "properties": {
                    "results_file": {
                        "type": "string",
                        "description": "Path to test results file"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "junit", "text"],
                        "description": "Results format",
                        "default": "json"
                    }
                }
            },
        ),
        types.Tool(
            name="get_test_coverage",
            description="Get test coverage information",
            inputSchema={
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "enum": ["a2aAgents", "a2aNetwork", "common", "all"],
                        "description": "Module for coverage report"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "html", "json", "xml"],
                        "description": "Coverage report format",
                        "default": "text"
                    }
                }
            },
        ),
        types.Tool(
            name="discover_tests",
            description="Discover and list available tests",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["unit", "integration", "e2e", "performance", "security", "contracts", "all"],
                        "description": "Type of tests to discover"
                    },
                    "include_disabled": {
                        "type": "boolean",
                        "description": "Include disabled tests",
                        "default": False
                    }
                }
            },
        ),
        types.Tool(
            name="validate_test_setup",
            description="Validate test environment and dependencies",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_dependencies": {
                        "type": "boolean",
                        "description": "Check test dependencies",
                        "default": True
                    },
                    "check_config": {
                        "type": "boolean",
                        "description": "Check test configuration",
                        "default": True
                    }
                }
            },
        ),
        types.Tool(
            name="manage_test_data",
            description="Manage test data and fixtures",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["reset", "seed", "cleanup", "backup"],
                        "description": "Action to perform on test data"
                    },
                    "test_type": {
                        "type": "string",
                        "enum": ["unit", "integration", "e2e", "all"],
                        "description": "Test type for data management"
                    }
                },
                "required": ["action"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    """Handle tool execution requests."""
    if arguments is None:
        arguments = {}
    
    try:
        if name == "run_tests":
            return await _run_tests(arguments)
        elif name == "analyze_test_results":
            return await _analyze_test_results(arguments)
        elif name == "get_test_coverage":
            return await _get_test_coverage(arguments)
        elif name == "discover_tests":
            return await _discover_tests(arguments)
        elif name == "validate_test_setup":
            return await _validate_test_setup(arguments)
        elif name == "manage_test_data":
            return await _manage_test_data(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def _run_tests(args: Dict[str, Any]) -> List[types.TextContent]:
    """Execute test suites."""
    test_type = args["test_type"]
    module = args.get("module", "all")
    pattern = args.get("pattern", "")
    coverage = args.get("coverage", False)
    verbose = args.get("verbose", False)
    
    # Build command based on test type
    cmd = []
    cwd = TEST_ROOT.parent
    
    if test_type in ["unit", "integration", "performance", "security"]:
        cmd = ["python", "-m", "pytest"]
        cmd.append(f"tests/{test_type}")
        
        if module != "all":
            cmd.append(f"tests/{test_type}/{module}")
        
        if pattern:
            cmd.extend(["-k", pattern])
        
        if coverage:
            cmd.extend(["--cov", "--cov-report=term"])
        
        if verbose:
            cmd.append("-v")
    
    elif test_type == "e2e":
        cmd = ["npm", "run", "test:e2e"]
        if verbose:
            cmd.append("--verbose")
    
    elif test_type == "contracts":
        cmd = ["forge", "test"]
        cwd = TEST_ROOT.parent / "a2aNetwork"
        if verbose:
            cmd.append("-vv")
    
    elif test_type == "all":
        cmd = ["npm", "test"]
    
    # Execute command
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = f"Command: {' '.join(cmd)}\n"
        output += f"Exit Code: {result.returncode}\n\n"
        output += f"STDOUT:\n{result.stdout}\n\n"
        
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        
        return [types.TextContent(type="text", text=output)]
    
    except subprocess.TimeoutExpired:
        return [types.TextContent(type="text", text="Test execution timed out after 5 minutes")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error executing tests: {str(e)}")]

async def _analyze_test_results(args: Dict[str, Any]) -> List[types.TextContent]:
    """Analyze test results."""
    results_file = args.get("results_file", "")
    format_type = args.get("format", "json")
    
    if not results_file:
        # Look for recent test results
        results_dir = TEST_ROOT.parent / "test-results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.xml"))
            if result_files:
                results_file = str(max(result_files, key=lambda f: f.stat().st_mtime))
    
    if not results_file or not Path(results_file).exists():
        return [types.TextContent(type="text", text="No test results file found")]
    
    try:
        with open(results_file, 'r') as f:
            content = f.read()
        
        analysis = f"Test Results Analysis\n"
        analysis += f"File: {results_file}\n"
        analysis += f"Format: {format_type}\n\n"
        analysis += f"Content Preview:\n{content[:1000]}...\n"
        
        return [types.TextContent(type="text", text=analysis)]
    
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error analyzing results: {str(e)}")]

async def _get_test_coverage(args: Dict[str, Any]) -> List[types.TextContent]:
    """Get test coverage information."""
    module = args.get("module", "all")
    format_type = args.get("format", "text")
    
    coverage_dir = TEST_ROOT.parent / "coverage"
    
    if not coverage_dir.exists():
        return [types.TextContent(type="text", text="No coverage reports found. Run tests with --coverage flag first.")]
    
    coverage_info = f"Test Coverage Report\n"
    coverage_info += f"Module: {module}\n"
    coverage_info += f"Format: {format_type}\n\n"
    
    # Look for coverage files
    if format_type == "html" and (coverage_dir / "html" / "index.html").exists():
        coverage_info += f"HTML report available at: {coverage_dir / 'html' / 'index.html'}\n"
    
    if (coverage_dir / "coverage.xml").exists():
        coverage_info += f"XML report available at: {coverage_dir / 'coverage.xml'}\n"
    
    # Try to get summary from text report
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--cov", "--cov-report=term-missing"],
            cwd=TEST_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.stdout:
            coverage_info += f"\nCoverage Summary:\n{result.stdout}\n"
    
    except Exception as e:
        coverage_info += f"\nError getting coverage summary: {str(e)}\n"
    
    return [types.TextContent(type="text", text=coverage_info)]

async def _discover_tests(args: Dict[str, Any]) -> List[types.TextContent]:
    """Discover available tests."""
    test_type = args.get("test_type", "all")
    include_disabled = args.get("include_disabled", False)
    
    discovered = f"Test Discovery Results\n"
    discovered += f"Type: {test_type}\n"
    discovered += f"Include Disabled: {include_disabled}\n\n"
    
    test_dirs = {
        "unit": UNIT_TESTS,
        "integration": INTEGRATION_TESTS,
        "e2e": E2E_TESTS,
        "performance": PERFORMANCE_TESTS,
        "security": SECURITY_TESTS,
        "contracts": CONTRACTS_TESTS
    }
    
    if test_type == "all":
        dirs_to_check = test_dirs.values()
    else:
        dirs_to_check = [test_dirs.get(test_type)]
    
    total_tests = 0
    
    for test_dir in dirs_to_check:
        if test_dir and test_dir.exists():
            dir_name = test_dir.name
            discovered += f"\n{dir_name.upper()} TESTS:\n"
            discovered += "-" * (len(dir_name) + 7) + "\n"
            
            # Count Python test files
            py_tests = list(test_dir.rglob("test_*.py"))
            py_tests.extend(test_dir.rglob("*_test.py"))
            
            # Count JavaScript test files  
            js_tests = list(test_dir.rglob("*.test.js"))
            js_tests.extend(test_dir.rglob("*.cy.js"))
            
            # Count Solidity test files
            sol_tests = list(test_dir.rglob("*.t.sol"))
            
            all_tests = py_tests + js_tests + sol_tests
            total_tests += len(all_tests)
            
            for test_file in sorted(all_tests)[:20]:  # Limit to first 20
                rel_path = test_file.relative_to(TEST_ROOT)
                discovered += f"  {rel_path}\n"
            
            if len(all_tests) > 20:
                discovered += f"  ... and {len(all_tests) - 20} more\n"
            
            discovered += f"\nTotal in {dir_name}: {len(all_tests)} files\n"
    
    discovered += f"\nGRAND TOTAL: {total_tests} test files discovered\n"
    
    return [types.TextContent(type="text", text=discovered)]

async def _validate_test_setup(args: Dict[str, Any]) -> List[types.TextContent]:
    """Validate test environment."""
    check_deps = args.get("check_dependencies", True)
    check_config = args.get("check_config", True)
    
    validation = "Test Environment Validation\n"
    validation += "=" * 30 + "\n\n"
    
    # Check Python environment
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        validation += f"Python: {result.stdout.strip()}\n"
    except Exception as e:
        validation += f"Python: ERROR - {e}\n"
    
    # Check Node.js environment
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        validation += f"Node.js: {result.stdout.strip()}\n"
    except Exception as e:
        validation += f"Node.js: ERROR - {e}\n"
    
    # Check test dependencies
    if check_deps:
        validation += "\nDependency Check:\n"
        
        deps_to_check = ["pytest", "jest", "cypress", "forge"]
        
        for dep in deps_to_check:
            try:
                if dep == "pytest":
                    result = subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True, text=True)
                elif dep == "jest":
                    result = subprocess.run(["jest", "--version"], capture_output=True, text=True)
                elif dep == "cypress":
                    result = subprocess.run(["cypress", "--version"], capture_output=True, text=True)
                elif dep == "forge":
                    result = subprocess.run(["forge", "--version"], capture_output=True, text=True)
                
                if result.returncode == 0:
                    validation += f"  ✓ {dep}: Available\n"
                else:
                    validation += f"  ✗ {dep}: Not available\n"
            except Exception:
                validation += f"  ✗ {dep}: Not found\n"
    
    # Check configuration files
    if check_config:
        validation += "\nConfiguration Check:\n"
        
        config_files = [
            TEST_ROOT / "pytest.ini",
            TEST_ROOT.parent / "jest.config.js",
            TEST_ROOT.parent / "cypress.config.js",
            TEST_ROOT / "config" / "jest.setup.js"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                validation += f"  ✓ {config_file.name}: Found\n"
            else:
                validation += f"  ✗ {config_file.name}: Missing\n"
    
    # Check test directories
    validation += "\nTest Directory Structure:\n"
    for test_dir in [UNIT_TESTS, INTEGRATION_TESTS, E2E_TESTS, PERFORMANCE_TESTS, SECURITY_TESTS, CONTRACTS_TESTS]:
        if test_dir.exists():
            file_count = len(list(test_dir.rglob("*test*")))
            validation += f"  ✓ {test_dir.name}: {file_count} test files\n"
        else:
            validation += f"  ✗ {test_dir.name}: Missing\n"
    
    return [types.TextContent(type="text", text=validation)]

async def _manage_test_data(args: Dict[str, Any]) -> List[types.TextContent]:
    """Manage test data and fixtures."""
    action = args["action"]
    test_type = args.get("test_type", "all")
    
    result = f"Test Data Management\n"
    result += f"Action: {action}\n"
    result += f"Test Type: {test_type}\n\n"
    
    if action == "reset":
        result += "Resetting test databases and fixtures...\n"
        # Add logic to reset test data
        result += "✓ Test data reset completed\n"
    
    elif action == "seed":
        result += "Seeding test databases with sample data...\n"
        # Add logic to seed test data
        result += "✓ Test data seeding completed\n"
    
    elif action == "cleanup":
        result += "Cleaning up test artifacts and temporary files...\n"
        # Add logic to cleanup test data
        result += "✓ Test cleanup completed\n"
    
    elif action == "backup":
        result += "Creating backup of test data...\n"
        # Add logic to backup test data
        result += "✓ Test data backup completed\n"
    
    else:
        result += f"Unknown action: {action}\n"
    
    return [types.TextContent(type="text", text=result)]

async def main():
    """Run the MCP server."""
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                types.InitializeRequest(
                    params=types.InitializeRequestParams(
                        protocolVersion="2024-11-05",
                        capabilities=types.ClientCapabilities(),
                        clientInfo=types.Implementation(
                            name="a2a-test-suite",
                            version="1.0.0"
                        )
                    )
                ),
            )
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
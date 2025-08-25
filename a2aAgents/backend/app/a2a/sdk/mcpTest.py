"""
Simple test to validate MCP implementation
"""

import asyncio
import json
from typing import Dict, Any

from .mcpServer import A2AMCPServer
from .mcpTypes import MCPRequest, MCPError
from .mcpDecorators import mcp_tool, mcp_resource


class TestMCPImplementation:
    """Test class to validate MCP functionality"""

    def __init__(self):
        self.mcp_server = A2AMCPServer("test-agent", "Test Agent", "1.0.0")
        self.test_data = {"test": "data", "value": 42}

    @mcp_tool(
        name="test_tool",
        description="Test tool for validation",
        input_schema={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"}
            },
            "required": ["input"]
        }
    )
    async def test_tool(self, input: str) -> Dict[str, Any]:
        """Test MCP tool"""
        return {"result": f"Processed: {input}", "success": True}

    @mcp_resource(
        uri="test://data",
        name="Test Data",
        description="Test resource for validation",
        mime_type="application/json"
    )
    async def test_resource(self) -> Dict[str, Any]:
        """Test MCP resource"""
        return self.test_data

    def register_components(self):
        """Register test components with MCP server"""
        # Register tool
        tool_metadata = getattr(self.test_tool, '_mcp_tool', None)
        if tool_metadata:
            self.mcp_server.register_tool(
                name=tool_metadata['name'],
                description=tool_metadata['description'],
                handler=self.test_tool,
                input_schema=tool_metadata['input_schema']
            )

        # Register resource
        resource_metadata = getattr(self.test_resource, '_mcp_resource', None)
        if resource_metadata:
            self.mcp_server.register_resource(
                uri=resource_metadata['uri'],
                name=resource_metadata['name'],
                description=resource_metadata['description'],
                content_provider=self.test_resource,
                mime_type=resource_metadata['mime_type']
            )

    async def run_tests(self) -> Dict[str, Any]:
        """Run MCP validation tests"""
        results = {"tests": [], "success": True}

        # Register components
        self.register_components()

        try:
            # Test 1: Initialize
            init_request = MCPRequest(
                id="test-1",
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            )

            init_response = await self.mcp_server.handle_request(init_request)
            results["tests"].append({
                "name": "initialize",
                "success": init_response.error is None,
                "error": str(init_response.error) if init_response.error else None
            })

            # Test 2: List tools
            list_tools_request = MCPRequest(
                id="test-2",
                method="tools/list"
            )

            list_response = await self.mcp_server.handle_request(list_tools_request)
            results["tests"].append({
                "name": "list_tools",
                "success": list_response.error is None and len(list_response.result.get("tools", [])) > 0,
                "error": str(list_response.error) if list_response.error else None,
                "tools_count": len(list_response.result.get("tools", [])) if list_response.result else 0
            })

            # Test 3: Call tool
            call_tool_request = MCPRequest(
                id="test-3",
                method="tools/call",
                params={
                    "name": "test_tool",
                    "arguments": {"input": "test_value"}
                }
            )

            call_response = await self.mcp_server.handle_request(call_tool_request)
            results["tests"].append({
                "name": "call_tool",
                "success": call_response.error is None,
                "error": str(call_response.error) if call_response.error else None,
                "result": call_response.result if call_response.result else None
            })

            # Test 4: List resources
            list_resources_request = MCPRequest(
                id="test-4",
                method="resources/list"
            )

            resources_response = await self.mcp_server.handle_request(list_resources_request)
            results["tests"].append({
                "name": "list_resources",
                "success": resources_response.error is None and len(resources_response.result.get("resources", [])) > 0,
                "error": str(resources_response.error) if resources_response.error else None,
                "resources_count": len(resources_response.result.get("resources", [])) if resources_response.result else 0
            })

            # Test 5: Read resource
            read_resource_request = MCPRequest(
                id="test-5",
                method="resources/read",
                params={"uri": "test://data"}
            )

            read_response = await self.mcp_server.handle_request(read_resource_request)
            results["tests"].append({
                "name": "read_resource",
                "success": read_response.error is None,
                "error": str(read_response.error) if read_response.error else None,
                "content": read_response.result if read_response.result else None
            })

            # Test 6: Invalid method
            invalid_request = MCPRequest(
                id="test-6",
                method="invalid/method"
            )

            invalid_response = await self.mcp_server.handle_request(invalid_request)
            results["tests"].append({
                "name": "invalid_method",
                "success": invalid_response.error is not None and invalid_response.error.code == -32601,
                "error": str(invalid_response.error) if invalid_response.error else None
            })

        except Exception as e:
            results["success"] = False
            results["exception"] = str(e)

        # Check overall success
        results["success"] = all(test["success"] for test in results["tests"])
        results["passed"] = sum(1 for test in results["tests"] if test["success"])
        results["total"] = len(results["tests"])

        return results


async def run_mcp_tests():
    """Run MCP implementation tests"""
    test_impl = TestMCPImplementation()
    results = await test_impl.run_tests()

    print("=== MCP Implementation Test Results ===")
    print(f"Overall Success: {results['success']}")
    print(f"Tests Passed: {results['passed']}/{results['total']}")

    for test in results["tests"]:
        status = "✓" if test["success"] else "✗"
        print(f"{status} {test['name']}: {test.get('error', 'OK')}")

    if not results["success"]:
        print(f"Exception: {results.get('exception', 'N/A')}")

    return results


if __name__ == "__main__":
    asyncio.run(run_mcp_tests())
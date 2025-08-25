"""
MCP Helper Mixin - Extracted from A2AAgentBase to reduce God Object complexity
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


class MCPHelperMixin:
    """Mixin providing MCP-related helper methods"""

    def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        if hasattr(self, 'mcp_server') and hasattr(self.mcp_server, 'tools'):
            return [asdict(tool) for tool in self.mcp_server.tools.values()]
        return []

    def list_mcp_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources"""
        if hasattr(self, 'mcp_server') and hasattr(self.mcp_server, 'resources'):
            return [asdict(resource) for resource in self.mcp_server.resources.values()]
        return []

    def get_mcp_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive MCP capabilities"""
        return {
            "tools": self.list_mcp_tools(),
            "resources": self.list_mcp_resources(),
            "total_tools": len(self.list_mcp_tools()),
            "total_resources": len(self.list_mcp_resources())
        }

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool with arguments"""
        if not hasattr(self, 'mcp_server'):
            return {"error": "MCP server not available"}

        try:
            from .mcpTypes import MCPRequest
            request = MCPRequest(
                jsonrpc="2.0",
                id="tool_call",
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            response = await self.process_mcp_request(request)
            return response.result if response.result else {"error": response.error}
        except Exception as e:
            logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def get_mcp_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Get an MCP resource by URI"""
        if not hasattr(self, 'mcp_server'):
            return {"error": "MCP server not available"}

        try:
            from .mcpTypes import MCPRequest
            request = MCPRequest(
                jsonrpc="2.0",
                id="resource_get",
                method="resources/read",
                params={"uri": resource_uri}
            )
            response = await self.process_mcp_request(request)
            return response.result if response.result else {"error": response.error}
        except Exception as e:
            logger.error(f"Failed to get MCP resource {resource_uri}: {e}")
            return {"error": str(e)}
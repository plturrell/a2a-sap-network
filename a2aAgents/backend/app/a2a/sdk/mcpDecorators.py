"""
MCP decorators for A2A agents
"""

from typing import Dict, Any, Callable, List
import functools
import logging
import asyncio

logger = logging.getLogger(__name__)


def mcp_tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any] = None,
    output_schema: Dict[str, Any] = None
):
    """
    Decorator to mark a method as an MCP tool

    Args:
        name: Tool name
        description: Tool description
        input_schema: JSON schema for input validation
        output_schema: JSON schema for output validation
    """
    def decorator(func: Callable) -> Callable:
        func._mcp_tool = {
            "name": name,
            "description": description,
            "input_schema": input_schema or {"type": "object", "properties": {}},
            "output_schema": output_schema
        }

        # Preserve the MCP metadata on the original function
        # Don't wrap the function - just add metadata
        return func

    return decorator


def mcp_resource(
    uri: str,
    name: str,
    description: str,
    mime_type: str = "text/plain"
):
    """
    Decorator to mark a method as an MCP resource provider

    Args:
        uri: Resource URI
        name: Resource name
        description: Resource description
        mime_type: MIME type of the resource
    """
    def decorator(func: Callable) -> Callable:
        func._mcp_resource = {
            "uri": uri,
            "name": name,
            "description": description,
            "mime_type": mime_type
        }

        # Preserve the MCP metadata on the original function
        # Don't wrap the function - just add metadata
        return func

    return decorator


def mcp_prompt(
    name: str,
    description: str,
    arguments: List[Dict[str, Any]] = None
):
    """
    Decorator to mark a method as an MCP prompt

    Args:
        name: Prompt name
        description: Prompt description
        arguments: List of prompt arguments
    """
    def decorator(func: Callable) -> Callable:
        func._mcp_prompt = {
            "name": name,
            "description": description,
            "arguments": arguments or []
        }

        # Preserve the MCP metadata on the original function
        # Don't wrap the function - just add metadata
        return func

    return decorator


def get_mcp_tool_metadata(method):
    """Extract MCP tool metadata from method"""
    return getattr(method, '_mcp_tool', None)


def get_mcp_resource_metadata(method):
    """Extract MCP resource metadata from method"""
    return getattr(method, '_mcp_resource', None)


def get_mcp_prompt_metadata(method):
    """Extract MCP prompt metadata from method"""
    return getattr(method, '_mcp_prompt', None)

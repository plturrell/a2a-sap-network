"""
Real MCP Extension for Intra-Agent Skill Communication
Extends the Model Context Protocol (MCP) JSON-RPC 2.0 specification
for skill-to-skill communication within a single agent
"""

import json
import logging
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """MCP protocol error"""

    # JSON-RPC 2.0 error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific error codes
    RESOURCE_NOT_FOUND = -32001
    TOOL_NOT_FOUND = -32002
    PROMPT_NOT_FOUND = -32003
    SUBSCRIPTION_ERROR = -32004

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class MCPIntraAgentMessageType(Enum):
    """MCP extension message types for intra-agent communication"""
    # Standard MCP methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

    # Intra-agent extension methods
    SKILLS_LIST = "skills/list"
    SKILLS_CALL = "skills/call"
    SKILLS_SUBSCRIBE = "skills/subscribe"
    SKILLS_UNSUBSCRIBE = "skills/unsubscribe"
    SKILLS_NOTIFICATION = "skills/notification"


@dataclass
class MCPRequest:
    """MCP JSON-RPC 2.0 request message"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC 2.0 response message"""
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPNotification:
    """MCP JSON-RPC 2.0 notification message"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPSkillCapability:
    """MCP skill capability descriptor"""
    name: str
    description: str
    tools: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    prompts: List[Dict[str, Any]]
    subscriptions: List[str]  # Event types this skill subscribes to


class MCPIntraAgentServer:
    """MCP server for intra-agent skill communication"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.skills: Dict[str, MCPSkillCapability] = {}
        self.skill_handlers: Dict[str, Callable] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # event_type -> [skill_names]
        self.request_id_counter = 0
        self.message_history: List[Dict[str, Any]] = []

    def register_skill(self, skill_name: str, capability: MCPSkillCapability, handler: Callable):
        """Register a skill with MCP server"""
        self.skills[skill_name] = capability
        self.skill_handlers[skill_name] = handler

        # Register subscriptions
        for event_type in capability.subscriptions:
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []
            self.subscriptions[event_type].append(skill_name)

        logger.info(f"MCP skill registered: {skill_name}")

    def _get_next_request_id(self) -> int:
        """Get next request ID"""
        self.request_id_counter += 1
        return self.request_id_counter

    async def handle_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request"""
        try:
            # Log request
            self.message_history.append({
                "type": "request",
                "method": request.method,
                "id": request.id,
                "timestamp": datetime.utcnow().isoformat(),
                "params": request.params
            })

            if request.method == MCPIntraAgentMessageType.SKILLS_LIST.value:
                result = await self._handle_skills_list(request.params or {})
            elif request.method == MCPIntraAgentMessageType.SKILLS_CALL.value:
                result = await self._handle_skills_call(request.params or {})
            elif request.method == MCPIntraAgentMessageType.TOOLS_LIST.value:
                result = await self._handle_tools_list(request.params or {})
            elif request.method == MCPIntraAgentMessageType.TOOLS_CALL.value:
                result = await self._handle_tools_call(request.params or {})
            elif request.method == MCPIntraAgentMessageType.RESOURCES_LIST.value:
                result = await self._handle_resources_list(request.params or {})
            elif request.method == MCPIntraAgentMessageType.RESOURCES_READ.value:
                result = await self._handle_resources_read(request.params or {})
            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {request.method}"
                    }
                )

            response = MCPResponse(id=request.id, result=result)

            # Log response
            self.message_history.append({
                "type": "response",
                "id": request.id,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            })

            return response

        except Exception as e:
            logger.error(f"MCP request handling error: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            )

    async def _handle_skills_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skills/list request"""
        skills_list = []
        for skill_name, capability in self.skills.items():
            skills_list.append({
                "name": skill_name,
                "description": capability.description,
                "tools_count": len(capability.tools),
                "resources_count": len(capability.resources),
                "prompts_count": len(capability.prompts),
                "subscriptions": capability.subscriptions
            })

        return {"skills": skills_list}

    async def _handle_skills_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle skills/call request"""
        skill_name = params.get("skill")
        method = params.get("method")
        arguments = params.get("arguments", {})

        if skill_name not in self.skill_handlers:
            raise Exception(f"Skill not found: {skill_name}")

        handler = self.skill_handlers[skill_name]

        # Create MCP request for the skill
        skill_request = MCPRequest(
            id=self._get_next_request_id(),
            method=method,
            params=arguments
        )

        # Call skill handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(skill_request)
        else:
            result = handler(skill_request)

        return {"result": result, "skill": skill_name, "method": method}

    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        all_tools = []
        for skill_name, capability in self.skills.items():
            for tool in capability.tools:
                tool_entry = tool.copy()
                tool_entry["skill"] = skill_name
                all_tools.append(tool_entry)

        return {"tools": all_tools}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Find which skill has this tool
        target_skill = None
        for skill_name, capability in self.skills.items():
            for tool in capability.tools:
                if tool["name"] == tool_name:
                    target_skill = skill_name
                    break
            if target_skill:
                break

        if not target_skill:
            raise Exception(f"Tool not found: {tool_name}")

        # Call the skill with tool execution request
        return await self._handle_skills_call({
            "skill": target_skill,
            "method": "tools/call",
            "arguments": {"name": tool_name, "arguments": arguments}
        })

    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request"""
        all_resources = []
        for skill_name, capability in self.skills.items():
            for resource in capability.resources:
                resource_entry = resource.copy()
                resource_entry["skill"] = skill_name
                all_resources.append(resource_entry)

        return {"resources": all_resources}

    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        resource_uri = params.get("uri")

        # Find which skill has this resource
        target_skill = None
        for skill_name, capability in self.skills.items():
            for resource in capability.resources:
                if resource["uri"] == resource_uri:
                    target_skill = skill_name
                    break
            if target_skill:
                break

        if not target_skill:
            raise Exception(f"Resource not found: {resource_uri}")

        # Call the skill with resource read request
        return await self._handle_skills_call({
            "skill": target_skill,
            "method": "resources/read",
            "arguments": {"uri": resource_uri}
        })

    async def send_notification(self, event_type: str, data: Dict[str, Any]):
        """Send notification to subscribed skills"""
        if event_type in self.subscriptions:
            notification = MCPNotification(
                method=MCPIntraAgentMessageType.SKILLS_NOTIFICATION.value,
                params={
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Send to all subscribed skills
            for skill_name in self.subscriptions[event_type]:
                if skill_name in self.skill_handlers:
                    try:
                        handler = self.skill_handlers[skill_name]
                        if asyncio.iscoroutinefunction(handler):
                            await handler(notification)
                        else:
                            handler(notification)
                    except Exception as e:
                        logger.error(f"Error sending notification to {skill_name}: {e}")

    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get MCP message history"""
        return self.message_history.copy()


class MCPIntraAgentClient:
    """MCP client for making requests to other skills"""

    def __init__(self, skill_name: str, mcp_server: MCPIntraAgentServer):
        self.skill_name = skill_name
        self.mcp_server = mcp_server
        self.request_id_counter = 0

    def _get_next_request_id(self) -> int:
        """Get next request ID"""
        self.request_id_counter += 1
        return self.request_id_counter

    async def list_skills(self) -> Dict[str, Any]:
        """List available skills"""
        request = MCPRequest(
            id=self._get_next_request_id(),
            method=MCPIntraAgentMessageType.SKILLS_LIST.value
        )

        response = await self.mcp_server.handle_mcp_request(request)
        if response.error:
            raise Exception(f"MCP error: {response.error}")

        return response.result

    async def call_skill(self, skill_name: str, method: str, arguments: Dict[str, Any] = None) -> Any:
        """Call another skill via MCP"""
        request = MCPRequest(
            id=self._get_next_request_id(),
            method=MCPIntraAgentMessageType.SKILLS_CALL.value,
            params={
                "skill": skill_name,
                "method": method,
                "arguments": arguments or {}
            }
        )

        response = await self.mcp_server.handle_mcp_request(request)
        if response.error:
            raise Exception(f"MCP error calling {skill_name}.{method}: {response.error}")

        return response.result["result"]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool via MCP"""
        request = MCPRequest(
            id=self._get_next_request_id(),
            method=MCPIntraAgentMessageType.TOOLS_CALL.value,
            params={
                "name": tool_name,
                "arguments": arguments or {}
            }
        )

        response = await self.mcp_server.handle_mcp_request(request)
        if response.error:
            raise Exception(f"MCP error calling tool {tool_name}: {response.error}")

        return response.result

    async def read_resource(self, resource_uri: str) -> Any:
        """Read a resource via MCP"""
        request = MCPRequest(
            id=self._get_next_request_id(),
            method=MCPIntraAgentMessageType.RESOURCES_READ.value,
            params={"uri": resource_uri}
        )

        response = await self.mcp_server.handle_mcp_request(request)
        if response.error:
            raise Exception(f"MCP error reading resource {resource_uri}: {response.error}")

        return response.result

    async def subscribe_to_events(self, event_types: List[str]):
        """Subscribe to event notifications"""
        # This would be handled during skill registration
        pass


class MCPSkillBase:
    """Base class for MCP-enabled skills"""

    def __init__(self, skill_name: str, description: str, mcp_server: MCPIntraAgentServer):
        self.skill_name = skill_name
        self.description = description
        self.mcp_server = mcp_server
        self.mcp_client = MCPIntraAgentClient(skill_name, mcp_server)

        # Skill capabilities
        self.tools = []
        self.resources = []
        self.prompts = []
        self.subscriptions = []

        # Register with MCP server
        self._register_skill()

    def _register_skill(self):
        """Register this skill with the MCP server"""
        capability = MCPSkillCapability(
            name=self.skill_name,
            description=self.description,
            tools=self.tools,
            resources=self.resources,
            prompts=self.prompts,
            subscriptions=self.subscriptions
        )

        self.mcp_server.register_skill(self.skill_name, capability, self.handle_mcp_request)

    async def handle_mcp_request(self, request: Union[MCPRequest, MCPNotification]) -> Any:
        """Handle incoming MCP request or notification"""
        if isinstance(request, MCPNotification):
            return await self.handle_notification(request)
        else:
            return await self.handle_request(request)

    async def handle_request(self, request: MCPRequest) -> Any:
        """Handle MCP request - override in subclasses"""
        if request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            return await self.call_tool(tool_name, arguments)
        elif request.method == "resources/read":
            resource_uri = request.params.get("uri")
            return await self.read_resource(resource_uri)
        else:
            # Try to find a method on this skill
            method_name = request.method.replace("/", "_")
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if asyncio.iscoroutinefunction(method):
                    return await method(request.params or {})
                else:
                    return method(request.params or {})
            else:
                raise Exception(f"Unknown method: {request.method}")

    async def handle_notification(self, notification: MCPNotification) -> None:
        """Handle MCP notification - override in subclasses"""
        pass

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool - override in subclasses"""
        raise Exception(f"Tool not implemented: {tool_name}")

    async def read_resource(self, resource_uri: str) -> Any:
        """Read a resource - override in subclasses"""
        raise Exception(f"Resource not implemented: {resource_uri}")

    def add_tool(self, name: str, description: str, input_schema: Dict[str, Any]):
        """Add a tool to this skill"""
        self.tools.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema
        })

    def add_resource(self, uri: str, name: str, description: str, mime_type: str = "application/json"):
        """Add a resource to this skill"""
        self.resources.append({
            "uri": uri,
            "name": name,
            "description": description,
            "mimeType": mime_type
        })

    def add_subscription(self, event_type: str):
        """Add event subscription"""
        if event_type not in self.subscriptions:
            self.subscriptions.append(event_type)

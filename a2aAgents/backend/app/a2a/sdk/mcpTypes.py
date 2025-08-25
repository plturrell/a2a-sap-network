"""
MCP (Model Context Protocol) type definitions for A2A agents
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class MCPRole(str, Enum):
    """MCP message roles"""
    CLIENT = "client"
    SERVER = "server"


class MCPMessageType(str, Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class MCPErrorCodes:
    """Standard MCP error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR_START = -32099
    SERVER_ERROR_END = -32000

    # Custom A2A error codes
    AUTHENTICATION_FAILED = -32001
    AUTHORIZATION_FAILED = -32002
    RATE_LIMITED = -32003
    SERVICE_UNAVAILABLE = -32004


class MCPError(BaseModel):
    """MCP error definition"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPRequest(BaseModel):
    """MCP JSON-RPC 2.0 request"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None
    _meta: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC 2.0 response"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    _meta: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_result_or_error(self):
        """Validate that either result or error is present, but not both"""
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")
        return self


class MCPNotification(BaseModel):
    """MCP JSON-RPC 2.0 notification"""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    _meta: Optional[Dict[str, Any]] = None


class MCPToolDefinition(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any] = Field(default_factory=dict)
    outputSchema: Optional[Dict[str, Any]] = None


class MCPResourceDefinition(BaseModel):
    """MCP resource definition"""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None


class MCPPromptDefinition(BaseModel):
    """MCP prompt definition"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


class MCPToolCall(BaseModel):
    """MCP tool execution request"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class TextContent(BaseModel):
    """MCP text content"""
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """MCP image content"""
    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mimeType: str


class AudioContent(BaseModel):
    """MCP audio content"""
    type: Literal["audio"] = "audio"
    data: str  # base64 encoded
    mimeType: str


class EmbeddedResource(BaseModel):
    """MCP embedded resource content"""
    type: Literal["resource"] = "resource"
    resource: Dict[str, Any]


MCPContent = Union[TextContent, ImageContent, AudioContent, EmbeddedResource]


class MCPToolResult(BaseModel):
    """MCP tool execution result"""
    content: List[MCPContent]
    isError: bool = False
    _meta: Optional[Dict[str, Any]] = None


class MCPResourceContent(BaseModel):
    """MCP resource content"""
    uri: str
    mimeType: str
    text: Optional[str] = None
    blob: Optional[str] = None
    _meta: Optional[Dict[str, Any]] = None


class MCPCapabilities(BaseModel):
    """MCP server capabilities"""
    experimental: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, bool]] = None
    resources: Optional[Dict[str, bool]] = None
    tools: Optional[Dict[str, bool]] = None
    sampling: Optional[Dict[str, Any]] = None
    roots: Optional[Dict[str, bool]] = None


class MCPInitializeRequest(BaseModel):
    """MCP initialize request"""
    protocolVersion: str = "2025-06-18"
    capabilities: MCPCapabilities
    clientInfo: Dict[str, str]
    _meta: Optional[Dict[str, Any]] = None


class MCPInitializeResponse(BaseModel):
    """MCP initialize response"""
    protocolVersion: str = "2025-06-18"
    capabilities: MCPCapabilities
    serverInfo: Dict[str, str]
    instructions: Optional[str] = None
    _meta: Optional[Dict[str, Any]] = None


class MCPListToolsRequest(BaseModel):
    """MCP list tools request"""
    cursor: Optional[str] = None


class MCPListToolsResponse(BaseModel):
    """MCP list tools response"""
    tools: List[MCPToolDefinition]
    nextCursor: Optional[str] = None


class MCPListResourcesRequest(BaseModel):
    """MCP list resources request"""
    cursor: Optional[str] = None


class MCPListResourcesResponse(BaseModel):
    """MCP list resources response"""
    resources: List[MCPResourceDefinition]
    nextCursor: Optional[str] = None


class MCPReadResourceRequest(BaseModel):
    """MCP read resource request"""
    uri: str


class MCPReadResourceResponse(BaseModel):
    """MCP read resource response"""
    contents: List[MCPResourceContent]


class MCPCallToolRequest(BaseModel):
    """MCP call tool request"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPCallToolResponse(BaseModel):
    """MCP call tool response"""
    content: List[MCPContent]
    isError: bool = False
    _meta: Optional[Dict[str, Any]] = None


class MCPListPromptsRequest(BaseModel):
    """MCP list prompts request"""
    cursor: Optional[str] = None


class MCPListPromptsResponse(BaseModel):
    """MCP list prompts response"""
    prompts: List[MCPPromptDefinition]
    nextCursor: Optional[str] = None


class MCPPingRequest(BaseModel):
    """MCP ping request"""
    pass


class MCPPingResult(BaseModel):
    """MCP ping result"""
    pass


class MCPCompleteRequest(BaseModel):
    """MCP complete request"""
    ref: Dict[str, Any]
    argument: Dict[str, Any]


class MCPCompleteResult(BaseModel):
    """MCP complete result"""
    completion: Dict[str, Any]


class MCPCreateMessageRequest(BaseModel):
    """MCP create message request"""
    messages: List[Dict[str, Any]]
    maxTokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    modelPreferences: Optional[Dict[str, Any]] = None
    stopSequences: Optional[List[str]] = None
    systemPrompt: Optional[str] = None
    temperature: Optional[float] = None
    includeContext: Optional[str] = None


class MCPCreateMessageResult(BaseModel):
    """MCP create message result"""
    content: MCPContent
    model: str
    role: str
    stopReason: Optional[str] = None


class MCPGetPromptRequest(BaseModel):
    """MCP get prompt request"""
    name: str
    arguments: Optional[Dict[str, Any]] = None


class MCPGetPromptResult(BaseModel):
    """MCP get prompt result"""
    description: Optional[str] = None
    messages: List[Dict[str, Any]]


class MCPCancelledNotification(BaseModel):
    """MCP cancelled notification"""
    requestId: Union[str, int]
    reason: Optional[str] = None


class MCPProgressNotification(BaseModel):
    """MCP progress notification"""
    progressToken: Union[str, int]
    progress: int
    total: Optional[int] = None


class MCPResourceUpdatedNotification(BaseModel):
    """MCP resource updated notification"""
    uri: str


class MCPToolListChangedNotification(BaseModel):
    """MCP tool list changed notification"""
    pass


class MCPPromptListChangedNotification(BaseModel):
    """MCP prompt list changed notification"""
    pass


class MCPResourceListChangedNotification(BaseModel):
    """MCP resource list changed notification"""
    pass


class MCPInitializedNotification(BaseModel):
    """MCP initialized notification"""
    pass
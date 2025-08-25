"""
Type definitions for A2A Agent SDK
"""

import os
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class MessagePart(BaseModel):
    """Part of an A2A message"""
    kind: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    """A2A Protocol message"""
    messageId: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    role: MessageRole
    parts: List[MessagePart]
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    signature: Optional[str] = None


class AgentCapability(BaseModel):
    """Agent capability definition"""
    name: str
    description: str
    enabled: bool = True
    version: str = "1.0.0"
    parameters: Optional[Dict[str, Any]] = None


class SkillDefinition(BaseModel):
    """A2A skill definition"""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    capabilities: List[str] = []
    domain: str = "general"
    method_name: str
    version: str = "1.0.0"
    tags: List[str] = []


class AgentCard(BaseModel):
    """A2A Agent Card (/.well-known/agent.json)"""
    name: str
    description: str
    url: str
    version: str
    protocolVersion: str = "0.2.9"
    provider: Dict[str, str]
    capabilities: Dict[str, bool]
    skills: List[str] = []
    endpoints: Dict[str, str] = {}
    metadata: Optional[Dict[str, Any]] = None


class TaskDefinition(BaseModel):
    """Task definition"""
    id: str
    type: str
    status: TaskStatus
    data: Dict[str, Any]
    created_at: str
    updated_at: str
    result: Optional[Any] = None
    error: Optional[str] = None
    timeout: int = 300
    retry_attempts: int = 3


class AgentConfig(BaseModel):
    """Agent configuration"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    base_url: str = os.getenv("A2A_SERVICE_URL")
    port: int = 8000

    # Telemetry settings
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None

    # Registry settings
    registry_url: Optional[str] = None
    auto_register: bool = True

    # Performance settings
    max_concurrent_tasks: int = 10
    task_timeout: int = 300
    message_queue_size: int = 100

    # Security settings
    enable_auth: bool = False
    api_key: Optional[str] = None
    trust_contract_address: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check status"""
    status: str  # healthy, degraded, unhealthy
    agent_id: str
    name: str
    version: str
    timestamp: str
    details: Dict[str, Any] = {}
    active_tasks: int = 0
    total_tasks: int = 0
    skills_count: int = 0
    handlers_count: int = 0
    uptime_seconds: float = 0.0


class SkillExecutionRequest(BaseModel):
    """Request to execute a skill"""
    skill_name: str
    input_data: Dict[str, Any]
    context_id: Optional[str] = None
    timeout: Optional[int] = None


class SkillExecutionResponse(BaseModel):
    """Response from skill execution"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    skill: str
    timestamp: str
    execution_time_ms: Optional[float] = None


class MessageHandlerRequest(BaseModel):
    """Request to handle a message"""
    message: A2AMessage
    context_id: str
    priority: str = "medium"
    timeout: Optional[int] = None


class MessageHandlerResponse(BaseModel):
    """Response from message handler"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str
    processing_time_ms: Optional[float] = None


class AgentRegistrationRequest(BaseModel):
    """Request to register agent with registry"""
    agent_card: AgentCard
    endpoint: str
    health_check_url: str
    trust_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class AgentRegistrationResponse(BaseModel):
    """Response from agent registration"""
    success: bool
    agent_id: str
    registration_id: str
    trust_contract_address: Optional[str] = None
    assigned_skills: List[str] = []
    network_endpoints: Dict[str, str] = {}


class NetworkDiscoveryRequest(BaseModel):
    """Request to discover agents in network"""
    required_skills: List[str] = []
    required_capabilities: List[str] = []
    minimum_trust_score: float = 0.0
    max_results: int = 10


class NetworkDiscoveryResponse(BaseModel):
    """Response from network discovery"""
    agents: List[Dict[str, Any]]
    total_found: int
    search_time_ms: float
    filters_applied: Dict[str, Any]

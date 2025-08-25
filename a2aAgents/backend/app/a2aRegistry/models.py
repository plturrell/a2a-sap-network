from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


class AgentType(str, Enum):
    """Agent type enumeration"""
    DATA_PROCESSING = "data-processing"
    AI_ML = "ai-ml"
    STORAGE = "storage"
    ORCHESTRATION = "orchestration"
    ANALYTICS = "analytics"


class WorkflowExecutionMode(str, Enum):
    """Workflow execution mode"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Core A2A Models
class AgentSkill(BaseModel):
    """A2A Agent Skill Definition"""
    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., description="Detailed skill description")
    tags: List[str] = Field(default=[], description="Skill categorization tags")
    inputModes: List[str] = Field(default=[], description="Supported input modes")
    outputModes: List[str] = Field(default=[], description="Supported output modes")
    examples: Optional[List[str]] = Field(None, description="Usage examples")
    specifications: Optional[Dict[str, Any]] = Field(None, description="Technical specifications")


class AgentCapabilities(BaseModel):
    """A2A Agent Capabilities"""
    streaming: Optional[bool] = Field(False, description="Supports streaming data")
    pushNotifications: Optional[bool] = Field(False, description="Supports push notifications")
    stateTransitionHistory: Optional[bool] = Field(False, description="Tracks state transitions")
    batchProcessing: Optional[bool] = Field(False, description="Supports batch processing")
    metadataExtraction: Optional[bool] = Field(False, description="Supports metadata extraction")
    dublinCoreCompliance: Optional[bool] = Field(False, description="Dublin Core compliant")


class AgentProvider(BaseModel):
    """Agent provider information"""
    organization: str = Field(..., description="Provider organization")
    url: Optional[HttpUrl] = Field(None, description="Provider URL")
    contact: Optional[str] = Field(None, description="Contact information")


class AgentCard(BaseModel):
    """A2A Agent Card - Core agent metadata following A2A v0.2.9"""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: HttpUrl = Field(..., description="Agent endpoint URL")
    version: str = Field(..., description="Agent version")
    protocolVersion: str = Field(..., description="A2A protocol version", pattern=r"^\d+\.\d+\.\d+$")
    provider: Optional[AgentProvider] = Field(None, description="Provider information")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities, description="Agent capabilities")
    skills: List[AgentSkill] = Field(..., description="Agent skills")
    defaultInputModes: List[str] = Field(default=[], description="Default input modes")
    defaultOutputModes: List[str] = Field(default=[], description="Default output modes")
    tags: Optional[List[str]] = Field(None, description="Agent tags")
    healthEndpoint: Optional[HttpUrl] = Field(None, description="Health check endpoint")
    metricsEndpoint: Optional[HttpUrl] = Field(None, description="Metrics endpoint")
    securitySchemes: Optional[Dict[str, Any]] = Field(None, description="Security configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ConnectivityResult(BaseModel):
    """Agent connectivity test result"""
    reachable: bool = Field(..., description="Agent is reachable")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    error_message: Optional[str] = Field(None, description="Error message if unreachable")


class ValidationResult(BaseModel):
    """Agent card validation result"""
    valid: bool = Field(..., description="Validation passed")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    errors: List[str] = Field(default=[], description="Validation errors")
    protocol_compliance: bool = Field(..., description="A2A protocol compliant")
    connectivity_check: bool = Field(..., description="Connectivity check passed")


class RegistrationMetadata(BaseModel):
    """Agent registration metadata"""
    registered_by: str = Field(..., description="User who registered the agent")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Registration status")


class AgentHealthStatus(BaseModel):
    """Agent health status"""
    current_status: HealthStatus = Field(..., description="Current health status")
    last_health_check: datetime = Field(..., description="Last health check timestamp")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    uptime_percentage: float = Field(..., description="Uptime percentage")
    error_rate_percentage: float = Field(..., description="Error rate percentage")


class AgentUsageAnalytics(BaseModel):
    """Agent usage analytics"""
    total_invocations: int = Field(default=0, description="Total invocations")
    successful_invocations: int = Field(default=0, description="Successful invocations")
    failed_invocations: int = Field(default=0, description="Failed invocations")
    average_response_time: float = Field(default=0.0, description="Average response time")
    last_invocation: Optional[datetime] = Field(None, description="Last invocation timestamp")


class AgentCompatibility(BaseModel):
    """Agent compatibility information"""
    protocol_versions: List[str] = Field(default=[], description="Supported protocol versions")
    supported_input_modes: List[str] = Field(default=[], description="Supported input modes")
    supported_output_modes: List[str] = Field(default=[], description="Supported output modes")
    dependency_requirements: List[str] = Field(default=[], description="Dependency requirements")


# Registration Models
class AgentRegistrationRecord(BaseModel):
    """Complete agent registration record"""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_card: AgentCard = Field(..., description="Agent card")
    registration_metadata: RegistrationMetadata = Field(..., description="Registration metadata")
    health_status: Optional[AgentHealthStatus] = Field(None, description="Health status")
    usage_analytics: AgentUsageAnalytics = Field(default_factory=AgentUsageAnalytics, description="Usage analytics")
    compatibility: AgentCompatibility = Field(default_factory=AgentCompatibility, description="Compatibility info")


class AgentRegistrationRequest(BaseModel):
    """Agent registration request"""
    agent_card: AgentCard = Field(..., description="Agent card")
    registered_by: str = Field(..., description="User registering the agent")
    tags: Optional[List[str]] = Field(None, description="Additional tags")
    labels: Optional[Dict[str, str]] = Field(None, description="Additional labels")


class AgentRegistrationResponse(BaseModel):
    """Agent registration response"""
    agent_id: str = Field(..., description="Generated agent ID")
    status: str = Field(..., description="Registration status")
    validation_results: ValidationResult = Field(..., description="Validation results")
    registered_at: datetime = Field(..., description="Registration timestamp")
    registry_url: str = Field(..., description="Registry URL for this agent")
    health_check_url: str = Field(..., description="Health check URL")


class AgentUpdateResponse(BaseModel):
    """Agent update response"""
    agent_id: str = Field(..., description="Agent ID")
    version: str = Field(..., description="Updated version")
    updated_at: datetime = Field(..., description="Update timestamp")
    validation_results: ValidationResult = Field(..., description="Validation results")


# Discovery Models
class AgentSearchRequest(BaseModel):
    """Agent search request"""
    skills: Optional[List[str]] = Field(None, description="Required skills")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    agent_type: Optional[AgentType] = Field(None, description="Agent type filter")
    status: Optional[HealthStatus] = Field(None, description="Health status filter")
    inputModes: Optional[List[str]] = Field(None, description="Required input modes")
    outputModes: Optional[List[str]] = Field(None, description="Required output modes")
    page: int = Field(default=1, ge=1, description="Page number")
    pageSize: int = Field(default=20, ge=1, le=100, description="Page size")


class AgentSearchResult(BaseModel):
    """Agent search result item"""
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    url: HttpUrl = Field(..., description="Agent URL")
    version: str = Field(..., description="Agent version")
    skills: List[str] = Field(..., description="Agent skills")
    status: HealthStatus = Field(..., description="Health status")
    last_seen: datetime = Field(..., description="Last seen timestamp")
    response_time_ms: float = Field(..., description="Response time")
    tags: List[str] = Field(default=[], description="Agent tags")
    trust_score: Optional[float] = Field(default=None, description="Trust score (0.0-1.0)")
    trust_level: Optional[str] = Field(default=None, description="Trust level (untrusted, low, medium, high, verified)")


class AgentSearchResponse(BaseModel):
    """Agent search response"""
    results: List[AgentSearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total result count")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class AgentDetails(BaseModel):
    """Detailed agent information"""
    agent_id: str = Field(..., description="Agent ID")
    agent_card: AgentCard = Field(..., description="Agent card")
    health_status: AgentHealthStatus = Field(..., description="Health status")
    usage_analytics: AgentUsageAnalytics = Field(..., description="Usage analytics")
    compatibility: AgentCompatibility = Field(..., description="Compatibility information")


# Workflow Models
class WorkflowStageRequirement(BaseModel):
    """Workflow stage requirement"""
    stage: str = Field(..., description="Stage name")
    required_skills: List[str] = Field(..., description="Required skills")
    input_modes: List[str] = Field(..., description="Required input modes")
    output_modes: List[str] = Field(..., description="Required output modes")
    depends_on: Optional[List[str]] = Field(None, description="Dependencies")


class WorkflowMatchRequest(BaseModel):
    """Workflow matching request"""
    workflow_requirements: List[WorkflowStageRequirement] = Field(..., description="Workflow requirements")


class WorkflowStageMatch(BaseModel):
    """Workflow stage matching result"""
    stage: str = Field(..., description="Stage name")
    agents: List[AgentSearchResult] = Field(..., description="Matching agents")


class WorkflowMatchResponse(BaseModel):
    """Workflow matching response"""
    workflow_id: str = Field(..., description="Generated workflow ID")
    matching_agents: List[WorkflowStageMatch] = Field(..., description="Stage matches")
    total_stages: int = Field(..., description="Total stages")
    coverage_percentage: float = Field(..., description="Coverage percentage")


class WorkflowPlanStage(BaseModel):
    """Workflow plan stage"""
    name: str = Field(..., description="Stage name")
    required_capabilities: List[str] = Field(..., description="Required capabilities")
    depends_on: Optional[List[str]] = Field(None, description="Dependencies")


class WorkflowPlanRequest(BaseModel):
    """Workflow plan request"""
    workflow_name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    stages: List[WorkflowPlanStage] = Field(..., description="Workflow stages")


class WorkflowExecutionPlan(BaseModel):
    """Workflow execution plan"""
    stage: str = Field(..., description="Stage name")
    agent: Dict[str, Any] = Field(..., description="Selected agent")


class WorkflowPlanResponse(BaseModel):
    """Workflow plan response"""
    workflow_id: str = Field(..., description="Workflow ID")
    execution_plan: List[WorkflowExecutionPlan] = Field(..., description="Execution plan")
    estimated_duration: str = Field(..., description="Estimated duration")
    total_agents: int = Field(..., description="Total agents")


class WorkflowExecutionRequest(BaseModel):
    """Workflow execution request"""
    input_data: Dict[str, Any] = Field(..., description="Input data")
    context_id: Optional[str] = Field(None, description="Context ID")
    execution_mode: WorkflowExecutionMode = Field(default=WorkflowExecutionMode.SEQUENTIAL, description="Execution mode")


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response"""
    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    status: WorkflowStatus = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Start timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")


class WorkflowExecutionStatus(BaseModel):
    """Workflow execution status"""
    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    status: WorkflowStatus = Field(..., description="Current status")
    current_stage: Optional[str] = Field(None, description="Current stage")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    stage_results: List[Dict[str, Any]] = Field(default=[], description="Stage results")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details")


# Health Monitoring Models
class AgentHealthDetails(BaseModel):
    """Detailed agent health information"""
    service_status: str = Field(..., description="Service status")
    memory_usage: str = Field(..., description="Memory usage")
    cpu_usage: str = Field(..., description="CPU usage")
    active_tasks: int = Field(..., description="Active tasks")
    error_rate: str = Field(..., description="Error rate")


class AgentCapabilitiesStatus(BaseModel):
    """Agent capabilities status"""
    all_skills_available: bool = Field(..., description="All skills available")
    degraded_skills: List[str] = Field(default=[], description="Degraded skills")
    unavailable_skills: List[str] = Field(default=[], description="Unavailable skills")


class AgentHealthResponse(BaseModel):
    """Agent health check response"""
    agent_id: str = Field(..., description="Agent ID")
    status: HealthStatus = Field(..., description="Health status")
    last_health_check: datetime = Field(..., description="Last check timestamp")
    response_time_ms: float = Field(..., description="Response time")
    health_details: AgentHealthDetails = Field(..., description="Detailed health info")
    capabilities_status: AgentCapabilitiesStatus = Field(..., description="Capabilities status")


class SystemHealthMetrics(BaseModel):
    """System health metrics"""
    registry_uptime: str = Field(..., description="Registry uptime")
    avg_agent_response_time: float = Field(..., description="Average agent response time")
    total_registrations_today: int = Field(..., description="Registrations today")


class SystemHealthResponse(BaseModel):
    """System health response"""
    status: HealthStatus = Field(..., description="System status")
    total_agents: int = Field(..., description="Total agents")
    healthy_agents: int = Field(..., description="Healthy agents")
    unhealthy_agents: int = Field(..., description="Unhealthy agents")
    last_health_sweep: datetime = Field(..., description="Last health sweep")
    system_metrics: SystemHealthMetrics = Field(..., description="System metrics")


class AgentMetricsResponse(BaseModel):
    """Agent metrics response"""
    agent_id: str = Field(..., description="Agent ID")
    period: str = Field(..., description="Metrics period")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")


# Service Health Models
class ServiceHealthResponse(BaseModel):
    """Service health response"""
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(..., description="Service statuses")
    metrics: Dict[str, Any] = Field(..., description="Health metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Error Models
class A2AError(BaseModel):
    """A2A Registry error"""
    status: str = Field(default="error", description="Response status")
    error: Dict[str, Any] = Field(..., description="Error details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Error metadata")


class ValidationError(A2AError):
    """Validation error for invalid data or constraints"""

    def __init__(self, field: str, message: str, value: Any = None):
        """
        Initialize validation error

        Args:
            field: The field that failed validation
            message: Descriptive error message
            value: The invalid value (optional)
        """
        error_detail = {
            "type": "validation_error",
            "field": field,
            "message": message
        }
        if value is not None:
            error_detail["invalid_value"] = str(value)

        super().__init__(
            status="validation_error",
            error=error_detail,
            metadata={"error_code": "VALIDATION_FAILED"}
        )


class ConflictError(A2AError):
    """Conflict error for resource conflicts"""

    def __init__(self, resource_type: str, resource_id: str, conflict_reason: str):
        """
        Initialize conflict error

        Args:
            resource_type: Type of resource (e.g., 'agent', 'workflow')
            resource_id: ID of the conflicting resource
            conflict_reason: Reason for the conflict
        """
        super().__init__(
            status="conflict",
            error={
                "type": "conflict_error",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "reason": conflict_reason
            },
            metadata={"error_code": "RESOURCE_CONFLICT"}
        )


class NotFoundError(A2AError):
    """Not found error for missing resources"""

    def __init__(self, resource_type: str, resource_id: str, search_criteria: Optional[Dict[str, Any]] = None):
        """
        Initialize not found error

        Args:
            resource_type: Type of resource not found
            resource_id: ID of the missing resource
            search_criteria: Search criteria used (optional)
        """
        error_detail = {
            "type": "not_found_error",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "message": f"{resource_type} with ID '{resource_id}' not found"
        }

        metadata = {"error_code": "RESOURCE_NOT_FOUND"}
        if search_criteria:
            metadata["search_criteria"] = search_criteria

        super().__init__(
            status="not_found",
            error=error_detail,
            metadata=metadata
        )
"""
A2A Trust System Models
Trust relationship management models for A2A agents
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrustLevel(str, Enum):
    """Trust commitment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    """Agent registration status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEREGISTERED = "deregistered"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    CREATED = "created"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Core Trust Models

class TrustCommitment(BaseModel):
    """Agent trust commitment"""
    level: TrustLevel = Field(..., description="Trust commitment level")
    reputation_weight: float = Field(default=1.0, description="Reputation weight factor")
    performance_bond: bool = Field(default=False, description="Performance guarantee flag")
    commitment_date: datetime = Field(default_factory=datetime.utcnow)


class TrustScore(BaseModel):
    """Agent trust score"""
    agent_id: str = Field(..., description="Agent trust ID")
    total_interactions: int = Field(default=0)
    successful_interactions: int = Field(default=0)
    trust_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    skill_ratings: Dict[str, float] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions


class InteractionRecord(BaseModel):
    """Trust interaction record between agents"""
    provider_id: str = Field(..., description="Provider agent ID")
    consumer_id: str = Field(..., description="Consumer agent ID")
    rating: int = Field(..., ge=1, le=5, description="Interaction rating (1-5)")
    skill_used: str = Field(..., description="Skill/capability used")
    response_time: Optional[int] = Field(None, description="Response time in milliseconds")
    error_occurred: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SLATerms(BaseModel):
    """Service Level Agreement terms"""
    response_time_max: int = Field(..., description="Maximum response time in milliseconds")
    availability_min: float = Field(..., ge=0.0, le=100.0, description="Minimum availability percentage")
    error_rate_max: float = Field(..., ge=0.0, le=100.0, description="Maximum error rate percentage")


class TrustWorkflow(BaseModel):
    """Trust-managed workflow"""
    workflow_id: str = Field(..., description="Workflow ID")
    initiator: str = Field(..., description="Workflow initiator")
    total_trust_required: float = Field(..., description="Total trust score required")
    current_stage: int = Field(default=0)
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completion_timestamp: Optional[datetime] = None


# Request/Response Models

class TrustAgentRegistrationRequest(BaseModel):
    """Request to register agent with trust system"""
    agent_card: Dict[str, Any] = Field(..., description="A2A Agent Card")
    commitment_level: TrustLevel = Field(..., description="Trust commitment level")


class TrustAgentRegistrationResponse(BaseModel):
    """Response from trust agent registration"""
    success: bool = Field(..., description="Registration success")
    registry_agent_id: str = Field(..., description="A2A Registry agent ID")
    trust_agent_id: str = Field(..., description="Trust system agent ID")
    commitment_level: TrustLevel = Field(..., description="Commitment level")
    initial_trust_score: float = Field(default=0.0, description="Initial trust score")


class TrustScoreResponse(BaseModel):
    """Trust score response"""
    agent_id: str = Field(..., description="Agent ID")
    overall_trust_score: float = Field(..., ge=0.0, le=5.0)
    trust_metrics: TrustScore = Field(..., description="Trust metrics")
    registry_metrics: Dict[str, float] = Field(..., description="A2A Registry metrics")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class TrustWorkflowRequest(BaseModel):
    """Request to create trust-managed workflow"""
    workflow_definition: Dict[str, Any] = Field(..., description="A2A workflow definition")
    trust_requirements: Dict[str, float] = Field(..., description="Trust requirements per stage")


class TrustWorkflowResponse(BaseModel):
    """Response from trust workflow creation"""
    workflow_id: str = Field(..., description="A2A Registry workflow ID")
    trust_workflow_id: str = Field(..., description="Trust system workflow ID")
    trust_requirements: Dict[str, float] = Field(..., description="Trust requirements")


class SLACreationRequest(BaseModel):
    """Request to create SLA"""
    provider_id: str = Field(..., description="Provider agent ID")
    consumer_id: str = Field(..., description="Consumer agent ID")
    terms: SLATerms = Field(..., description="SLA terms")
    validity_hours: int = Field(default=24, description="Validity in hours")


# System Models

class SystemHealth(BaseModel):
    """Trust system health"""
    status: str = Field(..., description="Overall system status")
    total_registered_agents: int = Field(default=0)
    active_workflows: int = Field(default=0)
    total_trust_interactions: int = Field(default=0)


class TrustMetrics(BaseModel):
    """Trust system metrics"""
    period: str = Field(..., description="Metrics period")
    agent_registrations: int = Field(default=0)
    trust_updates: int = Field(default=0)
    workflows_created: int = Field(default=0)
    workflows_completed: int = Field(default=0)
    average_trust_score: float = Field(default=0.0)
    workflow_success_rate: float = Field(default=0.0)
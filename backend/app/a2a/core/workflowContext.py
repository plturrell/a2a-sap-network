"""
A2A Workflow Context Management
Compliant with A2A Protocol v0.2.9
Provides workflow tracking, data instance management, and context propagation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import hashlib
from enum import Enum


class WorkflowStage(str, Enum):
    """Workflow stage states"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DataArtifact(BaseModel):
    """A2A-compliant data artifact with provenance"""

    artifact_id: str = Field(..., description="Unique artifact identifier")
    artifact_type: str = Field(..., description="Type of artifact (raw, processed, etc)")
    location: str = Field(..., description="Storage location of artifact")
    checksum: str = Field(..., description="SHA256 checksum of artifact")
    created_by: str = Field(..., description="Agent ID that created artifact")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    parent_artifacts: List[str] = Field(default=[], description="Parent artifact IDs")

    @field_serializer("created_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


class WorkflowContext(BaseModel):
    """A2A-compliant workflow context for inter-agent communication"""

    # Core workflow identifiers
    workflow_id: str = Field(..., description="Unique workflow instance ID")
    workflow_plan_id: str = Field(..., description="Workflow plan/template ID")
    context_id: str = Field(..., description="Context ID for message correlation")

    # Workflow metadata
    workflow_name: str = Field(..., description="Human-readable workflow name")
    initiated_by: str = Field(..., description="Agent or user that initiated workflow")
    initiated_at: datetime = Field(default_factory=datetime.utcnow)

    # Trust and SLA tracking
    trust_contract_id: Optional[str] = Field(None, description="Associated trust contract ID")
    sla_id: Optional[str] = Field(None, description="Associated SLA ID")
    required_trust_level: float = Field(default=0.0, description="Required trust level")

    # Stage tracking
    current_stage: str = Field(..., description="Current workflow stage")
    stage_history: List[Dict[str, Any]] = Field(default=[], description="Stage execution history")

    # Data artifact tracking
    artifacts: Dict[str, DataArtifact] = Field(default={}, description="Data artifacts by ID")
    current_artifact_id: Optional[str] = Field(None, description="Current artifact being processed")

    # Execution metadata
    execution_metadata: Dict[str, Any] = Field(
        default={}, description="Execution-specific metadata"
    )

    @field_serializer("initiated_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()

    model_config = ConfigDict()


class WorkflowContextManager:
    """Manages workflow context and data artifacts for A2A agents"""

    def __init__(self):
        self.active_contexts: Dict[str, WorkflowContext] = {}

    def create_workflow_context(
        self,
        workflow_plan_id: str,
        workflow_name: str,
        initiated_by: str,
        trust_contract_id: Optional[str] = None,
        sla_id: Optional[str] = None,
        required_trust_level: float = 0.0,
        initial_stage: str = "data_ingestion",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowContext:
        """Create a new workflow context"""
        import uuid

        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        context_id = f"ctx_{uuid.uuid4().hex[:12]}"

        context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_plan_id=workflow_plan_id,
            context_id=context_id,
            workflow_name=workflow_name,
            initiated_by=initiated_by,
            trust_contract_id=trust_contract_id,
            sla_id=sla_id,
            required_trust_level=required_trust_level,
            current_stage=initial_stage,
            execution_metadata=metadata or {},
        )

        self.active_contexts[workflow_id] = context
        return context

    def create_data_artifact(
        self,
        workflow_id: str,
        artifact_type: str,
        location: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_artifact_ids: Optional[List[str]] = None,
    ) -> DataArtifact:
        """Create a new data artifact with provenance tracking"""
        if workflow_id not in self.active_contexts:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Generate artifact ID based on content and context
        artifact_data = f"{workflow_id}:{artifact_type}:{location}:{datetime.utcnow().isoformat()}"
        artifact_id = f"art_{hashlib.sha256(artifact_data.encode()).hexdigest()[:12]}"

        # Calculate checksum of the artifact (simplified for now)
        checksum = hashlib.sha256(f"{location}:{artifact_type}".encode()).hexdigest()

        artifact = DataArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            location=location,
            checksum=checksum,
            created_by=created_by,
            metadata=metadata or {},
            parent_artifacts=parent_artifact_ids or [],
        )

        # Add to workflow context
        context = self.active_contexts[workflow_id]
        context.artifacts[artifact_id] = artifact
        context.current_artifact_id = artifact_id

        return artifact

    def update_stage(
        self,
        workflow_id: str,
        new_stage: str,
        stage_result: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> WorkflowContext:
        """Update workflow stage and record history"""
        if workflow_id not in self.active_contexts:
            raise ValueError(f"Workflow {workflow_id} not found")

        context = self.active_contexts[workflow_id]

        # Record stage transition
        stage_record = {
            "stage": context.current_stage,
            "completed_at": datetime.utcnow().isoformat(),
            "next_stage": new_stage,
            "result": stage_result,
            "processed_by": agent_id,
            "artifacts_produced": [
                aid
                for aid in context.artifacts.keys()
                if context.artifacts[aid].created_by == agent_id
            ],
        }

        context.stage_history.append(stage_record)
        context.current_stage = new_stage

        return context

    def get_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get workflow context by ID"""
        return self.active_contexts.get(workflow_id)

    def get_context_by_context_id(self, context_id: str) -> Optional[WorkflowContext]:
        """Get workflow context by context ID"""
        for context in self.active_contexts.values():
            if context.context_id == context_id:
                return context
        return None

    def serialize_for_message(self, workflow_id: str) -> Dict[str, Any]:
        """Serialize workflow context for A2A message passing"""
        context = self.get_context(workflow_id)
        if not context:
            return {}

        return {
            "workflow_id": context.workflow_id,
            "context_id": context.context_id,
            "workflow_plan_id": context.workflow_plan_id,
            "current_stage": context.current_stage,
            "trust_contract_id": context.trust_contract_id,
            "sla_id": context.sla_id,
            "current_artifact_id": context.current_artifact_id,
            "artifacts": {
                aid: {
                    "artifact_id": art.artifact_id,
                    "type": art.artifact_type,
                    "location": art.location,
                    "checksum": art.checksum,
                }
                for aid, art in context.artifacts.items()
            },
        }

    def get_data_lineage(self, artifact_id: str, workflow_id: str) -> List[DataArtifact]:
        """Get full data lineage for an artifact"""
        context = self.get_context(workflow_id)
        if not context or artifact_id not in context.artifacts:
            return []

        lineage = []
        visited = set()

        def trace_lineage(aid: str):
            if aid in visited:
                return
            visited.add(aid)

            if aid in context.artifacts:
                artifact = context.artifacts[aid]
                lineage.append(artifact)

                # Trace parent artifacts
                for parent_id in artifact.parent_artifacts:
                    trace_lineage(parent_id)

        trace_lineage(artifact_id)
        return lineage[::-1]  # Return in chronological order


# Global workflow context manager instance
workflowContextManager = WorkflowContextManager()

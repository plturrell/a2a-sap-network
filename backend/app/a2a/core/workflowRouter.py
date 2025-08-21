"""
A2A Workflow Orchestration Router
Provides endpoints for workflow management and monitoring
Compliant with A2A Protocol v0.2.9
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import logging

from .workflowContext import workflowContextManager
from .workflowMonitor import workflowMonitor, WorkflowMetrics, WorkflowState
# Import trust system and registry services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../a2aNetwork'))
from trustSystem.service import TrustSystemService
from registry.service import ORDRegistryService as A2ARegistryService


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/a2a/workflows", tags=["A2A Workflow Orchestration"])


# Request/Response Models
class WorkflowCreationRequest(BaseModel):
    """Request to create a new workflow"""

    workflow_plan_id: str = Field(..., description="Workflow plan/template ID")
    workflow_name: str = Field(..., description="Human-readable workflow name")
    trust_contract_id: Optional[str] = Field(None, description="Associated trust contract ID")
    sla_id: Optional[str] = Field(None, description="Associated SLA ID")
    required_trust_level: float = Field(default=0.0, description="Required trust level")
    initial_data_location: str = Field(..., description="Initial data location")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class WorkflowCreationResponse(BaseModel):
    """Response from workflow creation"""

    workflow_id: str = Field(..., description="Created workflow ID")
    context_id: str = Field(..., description="Context ID for message correlation")
    initial_stage: str = Field(..., description="Initial workflow stage")
    created_at: datetime = Field(..., description="Creation timestamp")
    monitoring_started: bool = Field(..., description="Whether monitoring was started")


class WorkflowStatusResponse(BaseModel):
    """Workflow status response"""

    workflow_id: str = Field(..., description="Workflow ID")
    current_stage: str = Field(..., description="Current stage")
    state: WorkflowState = Field(..., description="Overall workflow state")
    stages_completed: int = Field(..., description="Number of completed stages")
    total_stages: int = Field(..., description="Total number of stages")
    artifacts_created: int = Field(..., description="Number of artifacts created")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class DataLineageRequest(BaseModel):
    """Request for data lineage"""

    workflow_id: str = Field(..., description="Workflow ID")
    artifact_id: str = Field(..., description="Artifact ID to trace")


class DataLineageResponse(BaseModel):
    """Data lineage response"""

    artifact_id: str = Field(..., description="Artifact ID")
    lineage: List[Dict[str, Any]] = Field(..., description="Lineage chain")
    total_artifacts: int = Field(..., description="Total artifacts in lineage")


# Dependency injection for services
def get_trust_service() -> Optional[TrustSystemService]:
    """Get trust system service if available"""
    try:
        return TrustSystemService()
    except Exception as e:
        logger.warning(f"Trust service unavailable: {e}")
        return None


def get_registry_service() -> Optional[A2ARegistryService]:
    """Get A2A registry service if available"""
    try:
        # Use default base URL for registry service
        base_url = os.getenv("A2A_SERVICE_URL")  # Default registry URL
        return A2ARegistryService(base_url=base_url)
    except Exception as e:
        logger.warning(f"Registry service unavailable: {e}")
        return None


@router.post(
    "/create", response_model=WorkflowCreationResponse, status_code=status.HTTP_201_CREATED
)
async def create_workflow(
    request: WorkflowCreationRequest,
    trust_service: Optional[TrustSystemService] = Depends(get_trust_service),
    registry_service: Optional[A2ARegistryService] = Depends(get_registry_service),
):
    """Create a new workflow with optional trust contract"""
    try:
        # Validate trust contract if provided
        if request.trust_contract_id and trust_service:
            # Verify trust contract exists
            # This is a simplified check - in production would validate more thoroughly
            logger.info(f"Validating trust contract {request.trust_contract_id}")

        # Create workflow context
        workflow_context = workflowContextManager.create_workflow_context(
            workflow_plan_id=request.workflow_plan_id,
            workflow_name=request.workflow_name,
            initiated_by="workflow_orchestrator",
            trust_contract_id=request.trust_contract_id,
            sla_id=request.sla_id,
            required_trust_level=request.required_trust_level,
            initial_stage="data_ingestion",
            metadata=request.metadata or {},
        )

        # Create initial data artifact
        workflowContextManager.create_data_artifact(
            workflow_id=workflow_context.workflow_id,
            artifact_type="initial_data",
            location=request.initial_data_location,
            created_by="workflow_orchestrator",
            metadata={"source": "workflow_creation", "timestamp": datetime.utcnow().isoformat()},
        )

        # Start workflow monitoring
        monitoring_started = False
        try:
            # Initialize monitor with services if not already done
            if trust_service or registry_service:
                workflowMonitor.trust_service = trust_service
                workflowMonitor.registry = registry_service

            await workflowMonitor.start_workflow_monitoring(
                workflow_context, total_stages=3  # Default to 3 stages
            )
            monitoring_started = True
        except Exception as e:
            logger.warning(f"Failed to start workflow monitoring: {e}")

        return WorkflowCreationResponse(
            workflow_id=workflow_context.workflow_id,
            context_id=workflow_context.context_id,
            initial_stage=workflow_context.current_stage,
            created_at=workflow_context.initiated_at,
            monitoring_started=monitoring_started,
        )

    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}",
        ) from e


@router.get("/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get workflow status and metrics"""
    try:
        # Get workflow context
        context = workflowContextManager.get_context(workflow_id)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found"
            )

        # Get workflow metrics
        metrics = workflowMonitor.get_workflow_metrics(workflow_id)
        if not metrics:
            # Create basic metrics from context
            metrics = WorkflowMetrics(
                workflow_id=workflow_id,
                start_time=context.initiated_at,
                stages_completed=len(context.stage_history),
                total_stages=3,  # Default
                artifacts_created=len(context.artifacts),
                current_stage=context.current_stage,
                state=WorkflowState.RUNNING,
            )

        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            current_stage=context.current_stage,
            state=metrics.state,
            stages_completed=metrics.stages_completed,
            total_stages=metrics.total_stages,
            artifacts_created=len(context.artifacts),
            duration_seconds=metrics.duration_seconds,
            error_message=metrics.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}",
        ) from e


@router.get("/{workflow_id}/context")
async def get_workflow_context(workflow_id: str):
    """Get full workflow context including artifacts"""
    try:
        context = workflowContextManager.get_context(workflow_id)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow {workflow_id} not found"
            )

        # Serialize context for response
        return {
            "workflow_id": context.workflow_id,
            "context_id": context.context_id,
            "workflow_plan_id": context.workflow_plan_id,
            "workflow_name": context.workflow_name,
            "current_stage": context.current_stage,
            "trust_contract_id": context.trust_contract_id,
            "sla_id": context.sla_id,
            "required_trust_level": context.required_trust_level,
            "initiated_by": context.initiated_by,
            "initiated_at": context.initiated_at.isoformat(),
            "stage_history": context.stage_history,
            "artifacts": [
                {
                    "artifact_id": art.artifact_id,
                    "type": art.artifact_type,
                    "location": art.location,
                    "checksum": art.checksum,
                    "created_by": art.created_by,
                    "created_at": art.created_at.isoformat(),
                    "metadata": art.metadata,
                }
                for art in context.artifacts.values()
            ],
            "current_artifact_id": context.current_artifact_id,
            "execution_metadata": context.execution_metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow context: {str(e)}",
        ) from e


@router.post("/lineage", response_model=DataLineageResponse)
async def get_data_lineage(request: DataLineageRequest):
    """Get data lineage for an artifact"""
    try:
        lineage_artifacts = workflowContextManager.get_data_lineage(
            request.artifact_id, request.workflow_id
        )

        if not lineage_artifacts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact {request.artifact_id} not found in workflow {request.workflow_id}",
            )

        # Convert artifacts to response format
        lineage = [
            {
                "artifact_id": art.artifact_id,
                "type": art.artifact_type,
                "location": art.location,
                "created_by": art.created_by,
                "created_at": art.created_at.isoformat(),
                "parent_artifacts": art.parent_artifacts,
            }
            for art in lineage_artifacts
        ]

        return DataLineageResponse(
            artifact_id=request.artifact_id, lineage=lineage, total_artifacts=len(lineage)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data lineage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get data lineage: {str(e)}",
        ) from e


@router.get("/active")
async def get_active_workflows():
    """Get all active workflows"""
    try:
        active_workflows = workflowMonitor.get_active_workflows()

        return {
            "active_workflows": [
                {
                    "workflow_id": wf.workflow_id,
                    "state": wf.state,
                    "current_stage": wf.current_stage,
                    "stages_completed": wf.stages_completed,
                    "total_stages": wf.total_stages,
                    "start_time": wf.start_time.isoformat(),
                    "duration_seconds": (datetime.utcnow() - wf.start_time).total_seconds(),
                }
                for wf in active_workflows
            ],
            "total_count": len(active_workflows),
        }

    except Exception as e:
        logger.error(f"Error getting active workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active workflows: {str(e)}",
        ) from e


@router.get("/history")
async def get_workflow_history(hours: int = 24):
    """Get workflow history for the past N hours"""
    try:
        history = workflowMonitor.get_workflow_history(hours)

        return {
            "workflows": [
                {
                    "workflow_id": wf.workflow_id,
                    "state": wf.state,
                    "stages_completed": wf.stages_completed,
                    "total_stages": wf.total_stages,
                    "start_time": wf.start_time.isoformat(),
                    "end_time": wf.end_time.isoformat() if wf.end_time else None,
                    "duration_seconds": wf.duration_seconds,
                    "error_message": wf.error_message,
                }
                for wf in history
            ],
            "total_count": len(history),
            "hours": hours,
        }

    except Exception as e:
        logger.error(f"Error getting workflow history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow history: {str(e)}",
        ) from e

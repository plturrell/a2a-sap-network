"""
A2A Workflow Monitor
Tracks and monitors active workflows across agents
Compliant with A2A Protocol v0.2.9
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
from enum import Enum

from .workflowContext import WorkflowContext, workflowContextManager

logger = logging.getLogger(__name__)


class WorkflowState(str, Enum):
    """Overall workflow state"""

    INITIATING = "initiating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class WorkflowMetrics(BaseModel):
    """Workflow execution metrics"""

    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stages_completed: int = 0
    total_stages: int = 0
    artifacts_created: int = 0
    current_stage: str = ""
    state: WorkflowState = WorkflowState.INITIATING
    error_message: Optional[str] = None


class WorkflowMonitor:
    """Monitors and tracks workflow execution across A2A agents"""

    def __init__(self, registry_service=None, trust_service=None):
        self.registry = registry_service
        self.trust_service = trust_service
        self.active_workflows: Dict[str, WorkflowMetrics] = {}
        self.workflow_timeouts: Dict[str, float] = {}  # Workflow ID -> timeout in seconds
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def start_workflow_monitoring(
        self,
        workflow_context: WorkflowContext,
        total_stages: int,
        timeout_seconds: float = 3600,  # 1 hour default
    ) -> WorkflowMetrics:
        """Start monitoring a workflow"""
        metrics = WorkflowMetrics(
            workflow_id=workflow_context.workflow_id,
            start_time=workflow_context.initiated_at,
            total_stages=total_stages,
            current_stage=workflow_context.current_stage,
            state=WorkflowState.RUNNING,
        )

        self.active_workflows[workflow_context.workflow_id] = metrics
        self.workflow_timeouts[workflow_context.workflow_id] = timeout_seconds

        # Start async monitoring task
        self._monitoring_tasks[workflow_context.workflow_id] = asyncio.create_task(
            self._monitor_workflow(workflow_context.workflow_id)
        )

        logger.info(f"Started monitoring workflow {workflow_context.workflow_id}")
        return metrics

    async def _monitor_workflow(self, workflow_id: str):
        """Monitor a workflow for timeout and state changes"""
        timeout = self.workflow_timeouts.get(workflow_id, 3600)
        start_time = datetime.utcnow()

        try:
            while workflow_id in self.active_workflows:
                metrics = self.active_workflows[workflow_id]

                # Check timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > timeout:
                    await self.mark_workflow_timeout(workflow_id)
                    break

                # Update metrics from context
                context = workflowContextManager.get_context(workflow_id)
                if context:
                    metrics.current_stage = context.current_stage
                    metrics.stages_completed = len(context.stage_history)
                    metrics.artifacts_created = len(context.artifacts)

                # Check if workflow is complete
                if metrics.stages_completed >= metrics.total_stages:
                    await self.mark_workflow_complete(workflow_id)
                    break

                # Sleep before next check
                await asyncio.sleep(5)  # Check every 5 seconds

        except Exception as e:
            logger.error(f"Error monitoring workflow {workflow_id}: {e}")
            await self.mark_workflow_failed(workflow_id, str(e))

    async def update_workflow_stage(self, workflow_id: str, new_stage: str, agent_id: str):
        """Update workflow stage progress"""
        if workflow_id not in self.active_workflows:
            logger.warning(f"Workflow {workflow_id} not being monitored")
            return

        metrics = self.active_workflows[workflow_id]
        metrics.current_stage = new_stage

        # Update context
        context = workflowContextManager.get_context(workflow_id)
        if context:
            workflowContextManager.update_stage(workflow_id, new_stage, agent_id=agent_id)

        # Check trust requirements if trust service available
        if self.trust_service and context and context.trust_contract_id:
            trust_score = await self._check_agent_trust(agent_id)
            if trust_score < context.required_trust_level:
                logger.warning(
                    f"Agent {agent_id} trust score {trust_score} below required "
                    f"{context.required_trust_level} for workflow {workflow_id}"
                )

    async def _check_agent_trust(self, agent_id: str) -> float:
        """Check agent trust score"""
        if not self.trust_service:
            return 0.0

        try:
            trust_response = await self.trust_service.get_agent_trust_score(agent_id)
            return trust_response.overall_trust_score
        except Exception as e:
            logger.error(f"Error checking trust for agent {agent_id}: {e}")
            return 0.0

    async def mark_workflow_complete(self, workflow_id: str):
        """Mark workflow as completed"""
        if workflow_id not in self.active_workflows:
            return

        metrics = self.active_workflows[workflow_id]
        metrics.state = WorkflowState.COMPLETED
        metrics.end_time = datetime.utcnow()
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()

        # Cancel monitoring task
        if workflow_id in self._monitoring_tasks:
            self._monitoring_tasks[workflow_id].cancel()
            del self._monitoring_tasks[workflow_id]

        logger.info(f"Workflow {workflow_id} completed in {metrics.duration_seconds}s")

    async def mark_workflow_failed(self, workflow_id: str, error_message: str):
        """Mark workflow as failed"""
        if workflow_id not in self.active_workflows:
            return

        metrics = self.active_workflows[workflow_id]
        metrics.state = WorkflowState.FAILED
        metrics.end_time = datetime.utcnow()
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.error_message = error_message

        # Cancel monitoring task
        if workflow_id in self._monitoring_tasks:
            self._monitoring_tasks[workflow_id].cancel()
            del self._monitoring_tasks[workflow_id]

        logger.error(f"Workflow {workflow_id} failed: {error_message}")

    async def mark_workflow_timeout(self, workflow_id: str):
        """Mark workflow as timed out"""
        if workflow_id not in self.active_workflows:
            return

        metrics = self.active_workflows[workflow_id]
        metrics.state = WorkflowState.TIMEOUT
        metrics.end_time = datetime.utcnow()
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.error_message = (
            f"Workflow timed out after {self.workflow_timeouts.get(workflow_id, 0)}s"
        )

        logger.warning(f"Workflow {workflow_id} timed out")

    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow"""
        return self.active_workflows.get(workflow_id)

    def get_active_workflows(self) -> List[WorkflowMetrics]:
        """Get all active workflows"""
        return [
            m
            for m in self.active_workflows.values()
            if m.state in [WorkflowState.INITIATING, WorkflowState.RUNNING]
        ]

    def get_workflow_history(self, hours: int = 24) -> List[WorkflowMetrics]:
        """Get workflow history for the past N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.active_workflows.values() if m.start_time >= cutoff]

    async def cleanup_old_workflows(self, days: int = 7):
        """Clean up old workflow data"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        to_remove = []

        for wf_id, metrics in self.active_workflows.items():
            if metrics.start_time < cutoff and metrics.state in [
                WorkflowState.COMPLETED,
                WorkflowState.FAILED,
                WorkflowState.TIMEOUT,
            ]:
                to_remove.append(wf_id)

        for wf_id in to_remove:
            del self.active_workflows[wf_id]
            if wf_id in self.workflow_timeouts:
                del self.workflow_timeouts[wf_id]

        logger.info(f"Cleaned up {len(to_remove)} old workflows")


# Global workflow monitor instance
workflowMonitor = WorkflowMonitor()

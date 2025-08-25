"""
Goal Management API Endpoints
Provides REST API endpoints for goal management dashboard integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import traceback
from enum import Enum

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from .orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
from .comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK

logger = logging.getLogger(__name__)

# Error handling classes
class GoalManagementError(Exception):
    """Base exception for goal management errors"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class GoalNotFoundError(GoalManagementError):
    """Exception for when goals are not found"""
    def __init__(self, agent_id: str):
        super().__init__(f"Goals not found for agent: {agent_id}", status_code=404)

class GoalValidationError(GoalManagementError):
    """Exception for goal validation errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=400, details=details)

class GoalConflictError(GoalManagementError):
    """Exception for goal conflict errors"""
    def __init__(self, message: str, conflicting_goals: List[str]):
        super().__init__(message, status_code=409, details={"conflicting_goals": conflicting_goals})

class GoalDependencyError(GoalManagementError):
    """Exception for goal dependency errors"""
    def __init__(self, message: str, missing_dependencies: List[str]):
        super().__init__(message, status_code=400, details={"missing_dependencies": missing_dependencies})

# Initialize router
router = APIRouter(prefix="/api/v1/goals", tags=["Goal Management"])

# Global orchestrator handler instance
_orchestrator_handler: Optional[OrchestratorAgentA2AHandler] = None

async def get_orchestrator_handler() -> OrchestratorAgentA2AHandler:
    """Get or initialize orchestrator handler with proper error handling"""
    global _orchestrator_handler
    try:
        if _orchestrator_handler is None:
            sdk = ComprehensiveOrchestratorAgentSDK()
            _orchestrator_handler = OrchestratorAgentA2AHandler(sdk)
            await _orchestrator_handler.start()
        return _orchestrator_handler
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator handler: {e}\n{traceback.format_exc()}")
        raise GoalManagementError(
            "Failed to initialize goal management system",
            status_code=503,
            details={"error": str(e), "type": type(e).__name__}
        )

@router.get("/agents/{agent_id}")
async def get_agent_goals(
    agent_id: str,
    include_progress: bool = True,
    include_history: bool = False,
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Get goals for a specific agent with enhanced error handling"""
    try:
        # Validate agent_id
        if not agent_id or not agent_id.strip():
            raise GoalValidationError("Agent ID cannot be empty")

        # Create A2A message
        message_data = {
            "operation": "get_agent_goals",
            "data": {
                "agent_id": agent_id.strip(),
                "include_progress": include_progress,
                "include_history": include_history
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        elif result.get("status") == "not_found":
            raise GoalNotFoundError(agent_id)
        else:
            error_details = result.get("error_details", {})
            raise GoalManagementError(
                result.get("message", "Failed to retrieve goals"),
                status_code=result.get("status_code", 500),
                details=error_details
            )

    except GoalManagementError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting agent goals: {e}\n{traceback.format_exc()}")
        raise GoalManagementError(
            "An unexpected error occurred while retrieving goals",
            details={"error": str(e), "agent_id": agent_id}
        )

@router.get("/")
async def get_all_agent_goals(
    include_progress: bool = True,
    include_history: bool = False,
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Get goals for all agents"""
    try:
        message_data = {
            "operation": "get_agent_goals",
            "data": {
                "include_progress": include_progress,
                "include_history": include_history
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to get goals"))

    except Exception as e:
        logger.error(f"Failed to get all agent goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}")
async def set_agent_goals(
    agent_id: str,
    goals_data: Dict[str, Any],
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Set goals for a specific agent with validation and conflict detection"""
    try:
        # Validate input
        if not agent_id or not agent_id.strip():
            raise GoalValidationError("Agent ID cannot be empty")

        if not goals_data:
            raise GoalValidationError("Goals data cannot be empty")

        # Validate goal structure
        required_fields = ["goal_id", "specific", "measurable", "achievable", "relevant", "time_bound"]
        missing_fields = [field for field in required_fields if field not in goals_data]
        if missing_fields:
            raise GoalValidationError(
                "Missing required goal fields",
                details={"missing_fields": missing_fields}
            )

        # Check for dependencies if specified
        if "dependencies" in goals_data:
            await validate_goal_dependencies(agent_id, goals_data["dependencies"], handler)

        # Check for conflicts with existing goals
        await check_goal_conflicts(agent_id, goals_data, handler)

        message_data = {
            "operation": "set_agent_goals",
            "data": {
                "agent_id": agent_id.strip(),
                "goals": goals_data
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"], status_code=201)
        elif result.get("status") == "conflict":
            raise GoalConflictError(
                result.get("message", "Goal conflicts detected"),
                result.get("conflicting_goals", [])
            )
        elif result.get("status") == "validation_error":
            raise GoalValidationError(
                result.get("message", "Goal validation failed"),
                details=result.get("validation_errors", {})
            )
        else:
            raise GoalManagementError(
                result.get("message", "Failed to set goals"),
                status_code=result.get("status_code", 400),
                details=result.get("error_details", {})
            )

    except GoalManagementError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error setting agent goals: {e}\n{traceback.format_exc()}")
        raise GoalManagementError(
            "An unexpected error occurred while setting goals",
            details={"error": str(e), "agent_id": agent_id}
        )

@router.put("/agents/{agent_id}/progress")
async def update_agent_progress(
    agent_id: str,
    progress_data: Dict[str, Any],
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Update progress for a specific agent with validation"""
    try:
        # Validate input
        if not agent_id or not agent_id.strip():
            raise GoalValidationError("Agent ID cannot be empty")

        if not progress_data:
            raise GoalValidationError("Progress data cannot be empty")

        # Validate progress values are in valid range
        if "overall_progress" in progress_data:
            overall = progress_data["overall_progress"]
            if not isinstance(overall, (int, float)) or overall < 0 or overall > 100:
                raise GoalValidationError(
                    "Overall progress must be between 0 and 100",
                    details={"invalid_value": overall}
                )

        # Validate objective progress if provided
        if "objective_progress" in progress_data:
            for obj_name, value in progress_data["objective_progress"].items():
                if not isinstance(value, (int, float)) or value < 0 or value > 100:
                    raise GoalValidationError(
                        f"Objective progress for '{obj_name}' must be between 0 and 100",
                        details={"objective": obj_name, "invalid_value": value}
                    )

        message_data = {
            "operation": "track_goal_progress",
            "data": {
                "agent_id": agent_id.strip(),
                "progress": progress_data
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        elif result.get("status") == "not_found":
            raise GoalNotFoundError(agent_id)
        else:
            raise GoalManagementError(
                result.get("message", "Failed to update progress"),
                status_code=result.get("status_code", 400),
                details=result.get("error_details", {})
            )

    except GoalManagementError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating agent progress: {e}\n{traceback.format_exc()}")
        raise GoalManagementError(
            "An unexpected error occurred while updating progress",
            details={"error": str(e), "agent_id": agent_id}
        )

# Valid goal statuses
class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@router.put("/agents/{agent_id}/status")
async def update_goal_status(
    agent_id: str,
    status_data: Dict[str, str],
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Update goal status for a specific agent with validation"""
    try:
        # Validate input
        if not agent_id or not agent_id.strip():
            raise GoalValidationError("Agent ID cannot be empty")

        if not status_data or "status" not in status_data:
            raise GoalValidationError("Status is required")

        # Validate status value
        new_status = status_data.get("status")
        try:
            GoalStatus(new_status)
        except ValueError:
            raise GoalValidationError(
                f"Invalid status value: {new_status}",
                details={"valid_statuses": [s.value for s in GoalStatus]}
            )

        # Require reason for certain status changes
        if new_status in [GoalStatus.CANCELLED, GoalStatus.FAILED] and not status_data.get("reason"):
            raise GoalValidationError(
                f"Reason is required when setting status to {new_status}"
            )

        message_data = {
            "operation": "update_goal_status",
            "data": {
                "agent_id": agent_id.strip(),
                "status": new_status,
                "reason": status_data.get("reason", ""),
                "updated_by": status_data.get("updated_by", "api_client")
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        elif result.get("status") == "not_found":
            raise GoalNotFoundError(agent_id)
        elif result.get("status") == "invalid_transition":
            raise GoalValidationError(
                result.get("message", "Invalid status transition"),
                details=result.get("transition_details", {})
            )
        else:
            raise GoalManagementError(
                result.get("message", "Failed to update status"),
                status_code=result.get("status_code", 400),
                details=result.get("error_details", {})
            )

    except GoalManagementError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating goal status: {e}\n{traceback.format_exc()}")
        raise GoalManagementError(
            "An unexpected error occurred while updating goal status",
            details={"error": str(e), "agent_id": agent_id}
        )

@router.get("/analytics")
async def get_system_analytics(
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Get system-wide goal analytics"""
    try:
        message_data = {
            "operation": "get_goal_analytics",
            "data": {}
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to get analytics"))

    except Exception as e:
        logger.error(f"Failed to get system analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/{agent_id}")
async def get_agent_analytics(
    agent_id: str,
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Get analytics for a specific agent"""
    try:
        message_data = {
            "operation": "get_goal_analytics",
            "data": {
                "agent_id": agent_id
            }
        }

        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            return JSONResponse(content=result["data"])
        else:
            raise HTTPException(status_code=404, detail=result.get("message", "Analytics not found"))

    except Exception as e:
        logger.error(f"Failed to get agent analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "goal-management-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Background task for automated progress updates
@router.post("/agents/{agent_id}/auto-update")
async def enable_auto_progress_updates(
    agent_id: str,
    background_tasks: BackgroundTasks,
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Enable automated progress updates for an agent"""
    try:
        # Add background task for automated updates
        background_tasks.add_task(automated_progress_monitor, agent_id, handler)

        return {
            "status": "success",
            "message": f"Automated progress monitoring enabled for {agent_id}",
            "agent_id": agent_id
        }

    except Exception as e:
        logger.error(f"Failed to enable auto-updates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def automated_progress_monitor(agent_id: str, handler: OrchestratorAgentA2AHandler):
    """Background task for automated progress monitoring with error recovery"""
    retry_count = 0
    max_retries = 3
    base_delay = 300  # 5 minutes

    try:
        logger.info(f"Starting automated progress monitoring for {agent_id}")

        while True:
            try:
                # Check agent performance metrics and update progress
                await update_progress_from_metrics(agent_id, handler)

                # Reset retry count on success
                retry_count = 0

                # Wait before next update
                await asyncio.sleep(base_delay)

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Error in progress monitoring for {agent_id} (attempt {retry_count}/{max_retries}): {e}"
                )

                if retry_count >= max_retries:
                    logger.error(
                        f"Max retries reached for progress monitoring of {agent_id}. Stopping monitor."
                    )
                    break

                # Exponential backoff
                wait_time = base_delay * (2 ** retry_count)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    except asyncio.CancelledError:
        logger.info(f"Progress monitoring cancelled for {agent_id}")
        raise
    except Exception as e:
        logger.error(f"Fatal error in automated progress monitoring for {agent_id}: {e}\n{traceback.format_exc()}")

async def update_progress_from_metrics(agent_id: str, handler: OrchestratorAgentA2AHandler):
    """Update goal progress based on real agent metrics with enhanced error handling"""
    try:
        # Get current agent metrics from real monitoring system
        metrics = await get_agent_performance_metrics(agent_id)

        if not metrics:
            logger.debug(f"No metrics available for {agent_id}, skipping progress update")
            return

        # Calculate progress based on metrics with safe defaults
        try:
            # Response time progress calculation with bounds checking
            response_time = metrics.get("avg_response_time", 5000)
            response_time_progress = max(0, min(100, 100 - (response_time / 50)))

            progress_update = {
                "overall_progress": min(100.0, metrics.get("success_rate", 0.0)),
                "objective_progress": {
                    "data_registration": metrics.get("registration_success_rate", 0.0),
                    "validation_accuracy": metrics.get("validation_accuracy", 0.0),
                    "response_time": response_time_progress,
                    "compliance_tracking": metrics.get("compliance_score", 0.0),
                    "quality_assessment": metrics.get("quality_score", 0.0),
                    "catalog_management": metrics.get("catalog_completeness", 0.0)
                },
                "metrics_timestamp": datetime.utcnow().isoformat()
            }
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating progress for {agent_id}: {e}")
            return

        # Add new milestones based on metrics
        milestones = []
        milestone_checks = [
            (metrics.get("success_rate", 0) > 95, "High success rate achieved (>95%)"),
            (metrics.get("avg_response_time", 5000) < 2000, "Fast response time achieved (<2s)"),
            (metrics.get("uptime", 0) > 99.9, "High availability achieved (>99.9%)"),
            (metrics.get("error_rate", 100) < 5, "Low error rate achieved (<5%)"),
            (metrics.get("validation_accuracy", 0) > 98, "High validation accuracy achieved (>98%)")
        ]

        for condition, milestone in milestone_checks:
            if condition:
                milestones.append(milestone)

        if milestones:
            progress_update["milestones_achieved"] = milestones

        # Update progress through A2A message
        message_data = {
            "operation": "track_goal_progress",
            "data": {
                "agent_id": agent_id,
                "progress": progress_update,
                "source": "automated_monitor"
            }
        }

        message = A2AMessage(
            sender_id="automated_monitor",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data=message_data
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)

        if result.get("status") == "success":
            logger.info(
                f"Automated progress update successful for {agent_id}: "
                f"overall={progress_update['overall_progress']:.1f}%"
            )
        else:
            logger.warning(
                f"Automated progress update failed for {agent_id}: "
                f"status={result.get('status')}, message={result.get('message')}"
            )

    except Exception as e:
        logger.error(f"Failed to update progress from metrics for {agent_id}: {e}\n{traceback.format_exc()}")

async def get_agent_performance_metrics(agent_id: str) -> Optional[Dict[str, float]]:
    """Get real-time performance metrics for an agent with proper error handling"""
    try:
        # Connect to real monitoring system - Prometheus, agent health endpoints
        from ....core.monitoring import get_agent_metrics_client

        metrics_client = get_agent_metrics_client()
        if not metrics_client:
            logger.warning(f"No metrics client available for {agent_id}")
            return None

        # Get real metrics from monitoring system with timeout
        try:
            raw_metrics = await asyncio.wait_for(
                metrics_client.get_agent_metrics(agent_id),
                timeout=5.0  # 5 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting metrics for agent {agent_id}")
            return None

        if not raw_metrics:
            logger.warning(f"No metrics available for agent {agent_id}")
            return None

        # Transform raw metrics to goal-relevant format with validation
        metrics = {}
        metric_mappings = {
            "success_rate": (0.0, 100.0),
            "registration_success_rate": (0.0, 100.0),
            "validation_accuracy": (0.0, 100.0),
            "avg_response_time": (0, float('inf')),
            "compliance_score": (0.0, 100.0),
            "quality_score": (0.0, 100.0),
            "catalog_completeness": (0.0, 100.0),
            "uptime": (0.0, 100.0),
            "error_rate": (0.0, 100.0)
        }

        for metric_name, (min_val, max_val) in metric_mappings.items():
            value = raw_metrics.get(metric_name, min_val)
            # Validate and clamp values
            try:
                value = float(value)
                value = max(min_val, min(max_val, value))
                metrics[metric_name] = value
            except (TypeError, ValueError):
                logger.warning(f"Invalid metric value for {metric_name}: {value}")
                metrics[metric_name] = min_val

        return metrics

    except ImportError:
        logger.error(f"Monitoring module not available for agent {agent_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to get performance metrics for {agent_id}: {e}\n{traceback.format_exc()}")
        return None

# Helper functions for validation and conflict detection
async def validate_goal_dependencies(agent_id: str, dependencies: List[str], handler: OrchestratorAgentA2AHandler):
    """Validate that all goal dependencies exist and are achievable"""
    if not dependencies:
        return

    missing_deps = []
    for dep_goal_id in dependencies:
        # Check if dependency goal exists
        message = A2AMessage(
            sender_id="api_client",
            recipient_id="orchestrator_agent",
            parts=[MessagePart(
                role=MessageRole.USER,
                data={
                    "operation": "check_goal_exists",
                    "data": {"goal_id": dep_goal_id}
                }
            )],
            timestamp=datetime.utcnow()
        )

        result = await handler.process_a2a_message(message)
        if not result.get("exists", False):
            missing_deps.append(dep_goal_id)

    if missing_deps:
        raise GoalDependencyError(
            "Cannot set goal with missing dependencies",
            missing_dependencies=missing_deps
        )

async def check_goal_conflicts(agent_id: str, new_goal: Dict[str, Any], handler: OrchestratorAgentA2AHandler):
    """Check for conflicts with existing goals"""
    # Get existing goals
    message = A2AMessage(
        sender_id="api_client",
        recipient_id="orchestrator_agent",
        parts=[MessagePart(
            role=MessageRole.USER,
            data={
                "operation": "check_goal_conflicts",
                "data": {
                    "agent_id": agent_id,
                    "new_goal": new_goal
                }
            }
        )],
        timestamp=datetime.utcnow()
    )

    result = await handler.process_a2a_message(message)
    if result.get("has_conflicts", False):
        conflicting_goals = result.get("conflicting_goals", [])
        raise GoalConflictError(
            f"New goal conflicts with existing goals for agent {agent_id}",
            conflicting_goals=conflicting_goals
        )

# Global error handler for all goal management endpoints
@router.exception_handler(GoalManagementError)
async def goal_management_error_handler(request, exc: GoalManagementError):
    """Handle goal management specific errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

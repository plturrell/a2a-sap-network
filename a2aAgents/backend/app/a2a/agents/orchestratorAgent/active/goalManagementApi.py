"""
Goal Management API Endpoints
Provides REST API endpoints for goal management dashboard integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from .orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
from .comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/goals", tags=["Goal Management"])

# Global orchestrator handler instance
_orchestrator_handler: Optional[OrchestratorAgentA2AHandler] = None

async def get_orchestrator_handler() -> OrchestratorAgentA2AHandler:
    """Get or initialize orchestrator handler"""
    global _orchestrator_handler
    if _orchestrator_handler is None:
        sdk = ComprehensiveOrchestratorAgentSDK()
        _orchestrator_handler = OrchestratorAgentA2AHandler(sdk)
        await _orchestrator_handler.start()
    return _orchestrator_handler

@router.get("/agents/{agent_id}")
async def get_agent_goals(
    agent_id: str,
    include_progress: bool = True,
    include_history: bool = False,
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Get goals for a specific agent"""
    try:
        # Create A2A message
        message_data = {
            "operation": "get_agent_goals",
            "data": {
                "agent_id": agent_id,
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
            raise HTTPException(status_code=404, detail=result.get("message", "Goals not found"))
            
    except Exception as e:
        logger.error(f"Failed to get agent goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Set goals for a specific agent"""
    try:
        message_data = {
            "operation": "set_agent_goals",
            "data": {
                "agent_id": agent_id,
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
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to set goals"))
            
    except Exception as e:
        logger.error(f"Failed to set agent goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/agents/{agent_id}/progress")
async def update_agent_progress(
    agent_id: str,
    progress_data: Dict[str, Any],
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Update progress for a specific agent"""
    try:
        message_data = {
            "operation": "track_goal_progress",
            "data": {
                "agent_id": agent_id,
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
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to update progress"))
            
    except Exception as e:
        logger.error(f"Failed to update agent progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/agents/{agent_id}/status")
async def update_goal_status(
    agent_id: str,
    status_data: Dict[str, str],
    handler: OrchestratorAgentA2AHandler = Depends(get_orchestrator_handler)
):
    """Update goal status for a specific agent"""
    try:
        message_data = {
            "operation": "update_goal_status",
            "data": {
                "agent_id": agent_id,
                "status": status_data.get("status"),
                "reason": status_data.get("reason", "")
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
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to update status"))
            
    except Exception as e:
        logger.error(f"Failed to update goal status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Background task for automated progress monitoring"""
    try:
        logger.info(f"Starting automated progress monitoring for {agent_id}")
        
        while True:
            # Check agent performance metrics and update progress
            await update_progress_from_metrics(agent_id, handler)
            
            # Wait 5 minutes before next update
            await asyncio.sleep(300)
            
    except Exception as e:
        logger.error(f"Automated progress monitoring failed for {agent_id}: {e}")

async def update_progress_from_metrics(agent_id: str, handler: OrchestratorAgentA2AHandler):
    """Update goal progress based on real agent metrics"""
    try:
        # Get current agent metrics from real monitoring system
        metrics = await get_agent_performance_metrics(agent_id)
        
        if not metrics:
            return
        
        # Calculate progress based on metrics
        progress_update = {
            "overall_progress": min(100.0, metrics.get("success_rate", 0.0)),
            "objective_progress": {
                "data_registration": metrics.get("registration_success_rate", 0.0),
                "validation_accuracy": metrics.get("validation_accuracy", 0.0),
                "response_time": max(0, 100 - (metrics.get("avg_response_time", 5000) / 50)),
                "compliance_tracking": metrics.get("compliance_score", 0.0),
                "quality_assessment": metrics.get("quality_score", 0.0),
                "catalog_management": metrics.get("catalog_completeness", 0.0)
            }
        }
        
        # Add new milestones based on metrics
        milestones = []
        if metrics.get("success_rate", 0) > 95:
            milestones.append("High success rate achieved (>95%)")
        if metrics.get("avg_response_time", 5000) < 2000:
            milestones.append("Fast response time achieved (<2s)")
        if metrics.get("uptime", 0) > 99.9:
            milestones.append("High availability achieved (>99.9%)")
        
        if milestones:
            progress_update["milestones_achieved"] = milestones
        
        # Update progress through A2A message
        message_data = {
            "operation": "track_goal_progress",
            "data": {
                "agent_id": agent_id,
                "progress": progress_update
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
            logger.info(f"Automated progress update successful for {agent_id}")
        else:
            logger.warning(f"Automated progress update failed for {agent_id}: {result}")
            
    except Exception as e:
        logger.error(f"Failed to update progress from metrics for {agent_id}: {e}")

async def get_agent_performance_metrics(agent_id: str) -> Optional[Dict[str, float]]:
    """Get real-time performance metrics for an agent"""
    try:
        # Connect to real monitoring system - Prometheus, agent health endpoints
        from ....core.monitoring import get_agent_metrics_client
        
        metrics_client = get_agent_metrics_client()
        if not metrics_client:
            logger.warning(f"No metrics client available for {agent_id}")
            return None
        
        # Get real metrics from monitoring system
        raw_metrics = await metrics_client.get_agent_metrics(agent_id)
        if not raw_metrics:
            logger.warning(f"No metrics available for agent {agent_id}")
            return None
        
        # Transform raw metrics to goal-relevant format
        return {
            "success_rate": raw_metrics.get("success_rate", 0.0),
            "registration_success_rate": raw_metrics.get("registration_success_rate", 0.0),
            "validation_accuracy": raw_metrics.get("validation_accuracy", 0.0),
            "avg_response_time": raw_metrics.get("avg_response_time", 5000),
            "compliance_score": raw_metrics.get("compliance_score", 0.0),
            "quality_score": raw_metrics.get("quality_score", 0.0),
            "catalog_completeness": raw_metrics.get("catalog_completeness", 0.0),
            "uptime": raw_metrics.get("uptime", 0.0),
            "error_rate": raw_metrics.get("error_rate", 100.0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics for {agent_id}: {e}")
        return None

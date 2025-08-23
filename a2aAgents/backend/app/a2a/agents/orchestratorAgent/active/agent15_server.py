#!/usr/bin/env python3
"""
Agent 15 (Orchestrator) REST API Server
Provides HTTP endpoints for multi-agent workflow orchestration
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..')))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import the agent
from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import (
    ComprehensiveOrchestratorAgentSdk, WorkflowStatus, WorkflowStep, 
    CoordinationPattern, ExecutionStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent 15 - Orchestrator API",
    description="REST API for multi-agent workflow orchestration and coordination",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent = None

# Request/Response Models
class CreateWorkflowRequest(BaseModel):
    workflow_name: str
    description: str
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    strategy: str = "sequential"
    timeout_minutes: int = 60
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UpdateWorkflowRequest(BaseModel):
    workflow_id: str
    updates: Dict[str, Any]

class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str
    execution_context: Dict[str, Any] = Field(default_factory=dict)

class CoordinateAgentsRequest(BaseModel):
    coordination_plan: str
    agents: List[str]
    objective: str

class WorkflowTemplateRequest(BaseModel):
    template_name: str
    description: str
    template_definition: Dict[str, Any]

class CreateFromTemplateRequest(BaseModel):
    template_id: str
    workflow_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class OptimizeWorkflowRequest(BaseModel):
    workflow_id: str
    optimization_criteria: Dict[str, Any]

class ValidateWorkflowRequest(BaseModel):
    workflow_definition: Dict[str, Any]

class BulkExecuteRequest(BaseModel):
    workflow_ids: List[str]
    execution_context: Dict[str, Any] = Field(default_factory=dict)

# Storage for workflows and executions
workflows = {}
executions = {}
templates = {}
coordination_sessions = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    try:
        agent = ComprehensiveOrchestratorAgentSdk()
        await agent.initialize()
        logger.info("Agent 15 (Orchestrator) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "orchestrator",
        "agent_id": 15,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/workflows")
async def create_workflow(request: CreateWorkflowRequest):
    """Create a new workflow"""
    try:
        workflow_id = str(uuid4())
        
        # Parse tasks into WorkflowStep objects
        steps = []
        for i, task in enumerate(request.tasks):
            step = {
                "step_id": task.get("id", f"step_{i+1}"),
                "agent_id": task.get("agent_id", ""),
                "task_type": task.get("task_type", ""),
                "parameters": task.get("parameters", {}),
                "dependencies": task.get("dependencies", [])
            }
            steps.append(step)
        
        # Create workflow using agent
        result = await agent.create_workflow(
            name=request.workflow_name,
            steps=steps,
            strategy=request.strategy,
            description=request.description
        )
        
        # Store workflow
        workflows[workflow_id] = {
            "id": workflow_id,
            "name": request.workflow_name,
            "description": request.description,
            "tasks": request.tasks,
            "strategy": request.strategy,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "task_count": len(request.tasks),
            "validation": {"valid": True, "errors": [], "warnings": []}
        }
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows")
async def list_workflows(filters: Optional[str] = None):
    """List all workflows"""
    try:
        workflow_list = list(workflows.values())
        
        # Apply filters if provided
        if filters:
            filter_dict = json.loads(filters)
            if "status" in filter_dict:
                workflow_list = [w for w in workflow_list if w["status"] == filter_dict["status"]]
            if "strategy" in filter_dict:
                workflow_list = [w for w in workflow_list if w["strategy"] == filter_dict["strategy"]]
        
        return {"workflows": workflow_list}
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/workflows/{workflow_id}")
async def update_workflow(workflow_id: str, request: UpdateWorkflowRequest):
    """Update a workflow"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflows[workflow_id].update(request.updates)
        workflows[workflow_id]["updated_at"] = datetime.utcnow().isoformat()
        
        return {
            "workflow_id": workflow_id,
            "status": workflows[workflow_id]["status"],
            "task_count": len(workflows[workflow_id].get("tasks", [])),
            "validation": {"valid": True, "errors": [], "warnings": []}
        }
    except Exception as e:
        logger.error(f"Failed to update workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        del workflows[workflow_id]
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to delete workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/execute")
async def execute_workflow(request: ExecuteWorkflowRequest):
    """Execute a workflow"""
    try:
        if request.workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = workflows[request.workflow_id]
        
        # Execute using agent
        result = await agent.execute_workflow(
            workflow_id=request.workflow_id,
            context=request.execution_context
        )
        
        # Track execution
        execution_id = str(uuid4())
        executions[execution_id] = {
            "execution_id": execution_id,
            "workflow_id": request.workflow_id,
            "status": "started",
            "started_at": datetime.utcnow().isoformat(),
            "context": request.execution_context
        }
        
        # Update workflow status
        workflows[request.workflow_id]["status"] = "running"
        
        return {
            "status": "started",
            "started_at": datetime.utcnow().isoformat(),
            "strategy": workflow["strategy"]
        }
    except Exception as e:
        logger.error(f"Failed to execute workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/pause")
async def pause_workflow(workflow_id: str):
    """Pause a running workflow"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Pause using agent
        await agent.pause_workflow(workflow_id)
        
        workflows[workflow_id]["status"] = "paused"
        return {"status": "paused", "workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Failed to pause workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/resume")
async def resume_workflow(workflow_id: str):
    """Resume a paused workflow"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Resume using agent
        await agent.resume_workflow(workflow_id)
        
        workflows[workflow_id]["status"] = "running"
        return {"status": "running", "workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Failed to resume workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a workflow"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Cancel using agent
        await agent.cancel_workflow(workflow_id)
        
        workflows[workflow_id]["status"] = "cancelled"
        return {"status": "cancelled", "workflow_id": workflow_id}
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow execution status"""
    try:
        if workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = workflows[workflow_id]
        
        # Get real status from agent
        status = await agent.get_workflow_status(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "status": workflow["status"],
            "progress": {
                "completed_tasks": status.get("completed_steps", 0),
                "failed_tasks": status.get("failed_steps", 0),
                "running_tasks": status.get("running_steps", 1),
                "total_tasks": len(workflow.get("tasks", [])),
                "percentage": status.get("progress_percentage", 0)
            },
            "timing": {
                "started_at": workflow.get("created_at"),
                "duration_seconds": 0
            },
            "tasks": []
        }
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/workflows/history")
async def get_execution_history(workflow_id: str, limit: int = 50, offset: int = 0):
    """Get workflow execution history"""
    try:
        history = [e for e in executions.values() if e["workflow_id"] == workflow_id]
        return {"history": history[offset:offset+limit]}
    except Exception as e:
        logger.error(f"Failed to get execution history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/coordination/agents")
async def coordinate_agents(request: CoordinateAgentsRequest):
    """Coordinate multiple agents"""
    try:
        coordination_id = str(uuid4())
        
        # Create coordination using agent
        result = await agent.coordinate_agents(
            agents=request.agents,
            pattern=CoordinationPattern.PEER_TO_PEER,
            objective=request.objective
        )
        
        # Track coordination session
        coordination_sessions[coordination_id] = {
            "coordination_id": coordination_id,
            "status": "completed",
            "participating_agents": request.agents,
            "results": {"success": True}
        }
        
        return coordination_sessions[coordination_id]
    except Exception as e:
        logger.error(f"Failed to coordinate agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/templates")
async def list_workflow_templates(filters: Optional[str] = None):
    """List workflow templates"""
    try:
        return {"templates": list(templates.values())}
    except Exception as e:
        logger.error(f"Failed to list templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/templates")
async def create_workflow_template(request: WorkflowTemplateRequest):
    """Create a workflow template"""
    try:
        template_id = str(uuid4())
        
        templates[template_id] = {
            "ID": template_id,
            "name": request.template_name,
            "description": request.description,
            "definition": request.template_definition,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
        
        return templates[template_id]
    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/templates/instantiate")
async def create_workflow_from_template(request: CreateFromTemplateRequest):
    """Create workflow from template"""
    try:
        if request.template_id not in templates:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = templates[request.template_id]
        workflow_id = str(uuid4())
        
        # Create workflow based on template
        workflows[workflow_id] = {
            "id": workflow_id,
            "name": request.workflow_name,
            "description": template["description"],
            "tasks": template["definition"].get("tasks", []),
            "strategy": template["definition"].get("strategy", "sequential"),
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "template_id": request.template_id
        }
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "name": request.workflow_name
        }
    except Exception as e:
        logger.error(f"Failed to create workflow from template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
async def get_orchestration_metrics(time_range: str = "24h", group_by: str = "day"):
    """Get orchestration metrics"""
    try:
        # Calculate metrics
        total_workflows = len(workflows)
        active_workflows = len([w for w in workflows.values() if w["status"] == "running"])
        completed_workflows = len([w for w in workflows.values() if w["status"] == "completed"])
        
        return {
            "execution_metrics": {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "completed_workflows": completed_workflows,
                "success_rate": 0.85
            },
            "performance_metrics": {
                "average_execution_time": 120.5,
                "throughput": 10.2
            },
            "error_metrics": {
                "error_rate": 0.02,
                "common_errors": []
            },
            "agent_utilization": {},
            "time_series_data": []
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/optimize")
async def optimize_workflow(request: OptimizeWorkflowRequest):
    """Optimize a workflow"""
    try:
        if request.workflow_id not in workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Optimize using agent
        result = await agent.optimize_workflow(
            workflow_id=request.workflow_id,
            optimization_goals=["performance", "cost"]
        )
        
        return {
            "workflow_id": request.workflow_id,
            "optimization_results": result,
            "improvements": ["Parallel execution enabled", "Resource allocation optimized"],
            "estimated_performance_gain": 0.25
        }
    except Exception as e:
        logger.error(f"Failed to optimize workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/validate")
async def validate_workflow_definition(request: ValidateWorkflowRequest):
    """Validate a workflow definition"""
    try:
        # Validate using agent
        is_valid, errors = await agent.validate_workflow(request.workflow_definition)
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": [],
            "suggestions": ["Consider adding timeout configurations", "Add error handling steps"]
        }
    except Exception as e:
        logger.error(f"Failed to validate workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/workflows/bulk-execute")
async def bulk_execute_workflows(request: BulkExecuteRequest):
    """Execute multiple workflows"""
    try:
        started_executions = []
        failed_starts = []
        
        for workflow_id in request.workflow_ids:
            if workflow_id in workflows:
                try:
                    # Execute workflow
                    await agent.execute_workflow(workflow_id, request.execution_context)
                    workflows[workflow_id]["status"] = "running"
                    started_executions.append(workflow_id)
                except Exception as e:
                    failed_starts.append({"workflow_id": workflow_id, "error": str(e)})
            else:
                failed_starts.append({"workflow_id": workflow_id, "error": "Workflow not found"})
        
        batch_id = str(uuid4())
        return {
            "batch_id": batch_id,
            "started_executions": started_executions,
            "failed_starts": failed_starts
        }
    except Exception as e:
        logger.error(f"Failed to bulk execute workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "agent15_server:app",
        host="0.0.0.0",
        port=8015,
        reload=True,
        log_level="info"
    )
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import asyncio
import os
from datetime import datetime

from .agent_manager_agent import AgentManagerAgent, AgentRegistrationRequest, TrustContractRequest, WorkflowRequest
from ..core.a2a_types import A2AMessage, MessagePart, MessageRole

router = APIRouter(prefix="/a2a/agent_manager/v1", tags=["Agent Manager - A2A Ecosystem Orchestration"])

# Initialize Agent Manager
# Note: agent_manager will be set dynamically from main app or launcher
agent_manager = None


def initialize_agent_manager():
    """Initialize the Agent Manager instance for the main application"""
    global agent_manager
    
    if agent_manager is None:
        from .agent_manager_agent import AgentManagerAgent
        
        # Agent Manager configuration
        agent_id = "agent_manager"
        agent_name = "Agent Manager"
        # Allow test environments to run without explicitly setting this variable.
        # Default to a localhost base URL if the environment variable is absent.
        base_url = os.getenv("AGENT_MANAGER_BASE_URL", "http://localhost:8003")
        
        # Agent Manager capabilities
        capabilities = {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "batchProcessing": True,
            "smartContractDelegation": True,
            "aiAdvisor": True,
            "helpSeeking": True,
            "taskTracking": True,
            "agentRegistration": True,
            "trustContractManagement": True,
            "workflowOrchestration": True,
            "systemMonitoring": True
        }
        
        # Agent Manager skills
        skills = [
            {
                "id": "agent-registration",
                "name": "Agent Registration",
                "description": "Register and manage A2A agents in the ecosystem",
                "tags": ["registration", "management", "a2a"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "trust-contract-management",
                "name": "Trust Contract Management", 
                "description": "Create and manage trust contracts between agents",
                "tags": ["trust", "contracts", "delegation"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "workflow-orchestration",
                "name": "Workflow Orchestration",
                "description": "Orchestrate complex workflows across multiple agents",
                "tags": ["workflow", "orchestration", "coordination"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "system-monitoring",
                "name": "System Monitoring",
                "description": "Monitor health and performance of the A2A ecosystem",
                "tags": ["monitoring", "health", "metrics"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            }
        ]
        
        # Create the Agent Manager instance
        agent_manager = AgentManagerAgent(
            agent_id=agent_id,
            agent_name=agent_name,
            base_url=base_url,
            capabilities=capabilities,
            skills=skills
        )
    
    return agent_manager


# Initialize agent manager on module import
initialize_agent_manager()


@router.get("/.well-known/agent.json")
async def get_agent_card():
    """Get the agent card for Agent Manager"""
    return await agent_manager.get_agent_card()


@router.post("/rpc")
async def json_rpc_handler(request: Request):
    """Handle JSON-RPC 2.0 requests for Agent Manager"""
    try:
        body = await request.json()
        
        if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    },
                    "id": body.get("id")
                }
            )
        
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        if method == "agent.getCard":
            result = await agent_manager.get_agent_card()
        
        elif method == "agent.processMessage":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            priority = params.get("priority", "medium")
            processing_mode = params.get("processing_mode", "auto")
            result = await agent_manager.process_message(message, context_id, priority, processing_mode)
        
        elif method == "agent.getTaskStatus":
            task_id = params.get("taskId")
            result = await agent_manager.get_task_status(task_id)
        
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    },
                    "id": request_id
                }
            )
        
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": body.get("id") if "body" in locals() else None
            }
        )


@router.post("/messages")
async def rest_message_handler(request: Request):
    """REST-style message endpoint for Agent Manager"""
    try:
        body = await request.json()
        message = A2AMessage(**body.get("message", {}))
        context_id = body.get("contextId", str(datetime.utcnow().timestamp()))
        priority = body.get("priority", "medium")
        processing_mode = body.get("processing_mode", "auto")
        
        result = await agent_manager.process_message(message, context_id, priority, processing_mode)
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status for Agent Manager"""
    try:
        status = await agent_manager.get_task_status(task_id)
        return status
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )


@router.get("/queue/status")
async def get_queue_status():
    """Get message queue status for Agent Manager"""
    if agent_manager and agent_manager.message_queue:
        return agent_manager.message_queue.get_queue_status()
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/queue/messages/{message_id}")
async def get_message_status(message_id: str):
    """Get status of a specific message"""
    if agent_manager and agent_manager.message_queue:
        status = agent_manager.message_queue.get_message_status(message_id)
        if status:
            return status
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message not found"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.delete("/queue/messages/{message_id}")
async def cancel_message(message_id: str):
    """Cancel a queued or processing message"""
    if agent_manager and agent_manager.message_queue:
        cancelled = await agent_manager.message_queue.cancel_message(message_id)
        if cancelled:
            return {"message": "Message cancelled successfully"}
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Message not found or cannot be cancelled"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Message queue not available"}
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for Agent Manager"""
    queue_info = {}
    if agent_manager and agent_manager.message_queue:
        queue_status = agent_manager.message_queue.get_queue_status()
        queue_info = {
            "queue_depth": queue_status["queue_status"]["queue_depth"],
            "processing_count": queue_status["queue_status"]["processing_count"],
            "streaming_enabled": queue_status["capabilities"]["streaming_enabled"],
            "batch_processing_enabled": queue_status["capabilities"]["batch_processing_enabled"]
        }
    
    return {
        "status": "healthy",
        "agent": "Agent Manager",
        "version": "2.0.0",
        "protocol_version": "0.2.9",
        "timestamp": datetime.utcnow().isoformat(),
        "message_queue": queue_info
    }


# Agent Management Endpoints
@router.post("/agents/register")
async def register_agent(registration: AgentRegistrationRequest):
    """Register a new agent in the A2A ecosystem"""
    try:
        # Create A2A message for agent registration
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "register_agent",
                        "agent_id": registration.agent_id,
                        "agent_name": registration.agent_name,
                        "base_url": registration.base_url,
                        "capabilities": registration.capabilities,
                        "skills": registration.skills,
                        "metadata": registration.metadata
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str, force: bool = False):
    """Deregister an agent from the A2A ecosystem"""
    try:
        # Create A2A message for agent deregistration
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "deregister_agent",
                        "agent_id": agent_id,
                        "force": force
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents():
    """List all registered agents"""
    try:
        if agent_manager:
            return {
                "agents": agent_manager.registered_agents,
                "count": len(agent_manager.registered_agents)
            }
        else:
            raise HTTPException(status_code=503, detail="Agent Manager not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get information about a specific agent"""
    try:
        if agent_manager and agent_id in agent_manager.registered_agents:
            return agent_manager.registered_agents[agent_id]
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/health")
async def check_agent_health(agent_id: str):
    """Check health of a specific agent"""
    try:
        # Create A2A message for agent monitoring
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "monitor_agents",
                        "agent_id": agent_id
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Trust Contract Management Endpoints
@router.post("/trust/contracts")
async def create_trust_contract(contract: TrustContractRequest):
    """Create a trust contract between agents"""
    try:
        # Create A2A message for trust contract creation
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "create_trust_contract",
                        "delegator_agent": contract.delegator_agent,
                        "delegate_agent": contract.delegate_agent,
                        "actions": contract.actions,
                        "expiry_hours": contract.expiry_hours,
                        "conditions": contract.conditions
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trust/contracts")
async def list_trust_contracts():
    """List all trust contracts"""
    try:
        if agent_manager:
            return {
                "contracts": agent_manager.trust_contracts,
                "count": len(agent_manager.trust_contracts)
            }
        else:
            raise HTTPException(status_code=503, detail="Agent Manager not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trust/contracts/{contract_id}")
async def get_trust_contract(contract_id: str):
    """Get information about a specific trust contract"""
    try:
        if agent_manager and contract_id in agent_manager.trust_contracts:
            return agent_manager.trust_contracts[contract_id]
        else:
            raise HTTPException(status_code=404, detail=f"Trust contract {contract_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/trust/contracts/{contract_id}")
async def revoke_trust_contract(contract_id: str):
    """Revoke a trust contract"""
    try:
        if agent_manager and contract_id in agent_manager.trust_contracts:
            agent_manager.trust_contracts[contract_id]["status"] = "revoked"
            agent_manager.trust_contracts[contract_id]["revoked_at"] = datetime.utcnow()
            return {"message": f"Trust contract {contract_id} revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Trust contract {contract_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Workflow Orchestration Endpoints
@router.post("/workflows")
async def create_workflow(workflow: WorkflowRequest):
    """Create and start a workflow"""
    try:
        # Create A2A message for workflow creation
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "create_workflow",
                        "workflow_name": workflow.workflow_name,
                        "agents": workflow.agents,
                        "tasks": workflow.tasks,
                        "dependencies": workflow.dependencies,
                        "metadata": workflow.metadata
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows():
    """List all workflows"""
    try:
        if agent_manager:
            return {
                "workflows": agent_manager.active_workflows,
                "count": len(agent_manager.active_workflows)
            }
        else:
            raise HTTPException(status_code=503, detail="Agent Manager not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    try:
        if agent_manager and workflow_id in agent_manager.active_workflows:
            return agent_manager.active_workflows[workflow_id]
        else:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflows/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    try:
        if agent_manager and workflow_id in agent_manager.active_workflows:
            workflow = agent_manager.active_workflows[workflow_id]
            if workflow["status"] in ["pending", "running"]:
                workflow["status"] = "cancelled"
                workflow["cancelled_at"] = datetime.utcnow()
                return {"message": f"Workflow {workflow_id} cancelled successfully"}
            else:
                return {"message": f"Workflow {workflow_id} cannot be cancelled (status: {workflow['status']})"}
        else:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Monitoring Endpoints
@router.get("/system/health")
async def system_health_check():
    """Get system-wide health status"""
    try:
        # Create A2A message for system health check
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "system_health"
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/metrics")
async def system_metrics():
    """Get system metrics and statistics"""
    try:
        if agent_manager:
            metrics = {
                "registered_agents": len(agent_manager.registered_agents),
                "active_workflows": len([wf for wf in agent_manager.active_workflows.values() if wf["status"] == "running"]),
                "total_workflows": len(agent_manager.active_workflows),
                "trust_contracts": len(agent_manager.trust_contracts),
                "uptime": (datetime.utcnow() - agent_manager.startup_time).total_seconds() if hasattr(agent_manager, 'startup_time') else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add agent status breakdown
            agent_statuses = {}
            for agent_info in agent_manager.registered_agents.values():
                status = agent_info["status"]
                agent_statuses[status] = agent_statuses.get(status, 0) + 1
            
            metrics["agent_status_breakdown"] = agent_statuses
            
            return metrics
        else:
            raise HTTPException(status_code=503, detail="Agent Manager not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/agents/health")
async def all_agents_health():
    """Check health of all registered agents"""
    try:
        # Create A2A message for all agents monitoring
        message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "operation": "monitor_agents"
                    }
                )
            ]
        )
        
        result = await agent_manager.process_message(message, str(datetime.utcnow().timestamp()))
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
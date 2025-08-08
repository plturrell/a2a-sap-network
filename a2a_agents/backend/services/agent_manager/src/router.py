"""
A2A Protocol Router for Agent Manager
Implements network management endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import json
from datetime import datetime
from typing import Dict, Any

def create_a2a_router(agent):
    """Create A2A compliant router for Agent Manager"""
    router = APIRouter(tags=["A2A Agent Manager"])
    
    @router.get("/.well-known/agent.json")
    async def get_agent_card():
        """
        A2A Agent Card endpoint
        Returns agent manager capabilities
        """
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "protocol_version": "0.2.9",
            "capabilities": {
                "handlers": [
                    {
                        "name": handler.name,
                        "description": handler.description
                    } for handler in agent.handlers.values()
                ],
                "skills": [
                    {
                        "name": skill.name,
                        "description": skill.description
                    } for skill in agent.skills.values()
                ],
                "orchestration": True,
                "service_discovery": True,
                "health_monitoring": True,
                "workflow_management": True,
                "trust_verification": True
            },
            "endpoints": {
                "rpc": "/rpc",
                "network_status": "/a2a/network/status",
                "agents": "/a2a/agents",
                "workflows": "/a2a/workflows"
            }
        }
    
    @router.post("/rpc")
    async def json_rpc_handler(request: Request):
        """
        JSON-RPC 2.0 endpoint for agent management
        """
        try:
            body = await request.json()
            
            # Validate JSON-RPC 2.0 format
            if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request - Not JSON-RPC 2.0"
                        },
                        "id": body.get("id")
                    }
                )
            
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")
            
            # Route to appropriate handler
            if method == "register_agent":
                message = agent.create_message(params)
                result = await agent.handle_agent_registration(message, str(agent.generate_context_id()))
            
            elif method == "discover_agents":
                message = agent.create_message(params)
                result = await agent.handle_agent_discovery(message, str(agent.generate_context_id()))
            
            elif method == "start_workflow":
                message = agent.create_message(params)
                result = await agent.handle_workflow_start(message, str(agent.generate_context_id()))
            
            elif method == "get_network_status":
                result = await agent.get_network_status()
            
            else:
                return JSONResponse(
                    status_code=404,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        },
                        "id": request_id
                    }
                )
            
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
            
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    },
                    "id": None
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": body.get("id") if 'body' in locals() else None
                }
            )
    
    @router.get("/a2a/network/status")
    async def get_network_status():
        """Get current A2A network status"""
        status = await agent.get_network_status()
        return {
            "network": "a2a",
            "protocol_version": "0.2.9",
            "manager_status": "active" if agent.is_ready else "initializing",
            "registered_agents": [
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "status": a.status.value,
                    "base_url": a.base_url,
                    "last_heartbeat": a.last_heartbeat.isoformat()
                }
                for a in agent.registered_agents.values()
            ],
            "statistics": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/a2a/agents")
    async def list_agents(status: str = None, capability: str = None):
        """List registered agents with optional filters"""
        agents = []
        
        for agent_info in agent.registered_agents.values():
            # Apply filters
            if status and agent_info.status.value != status:
                continue
            if capability and capability not in agent_info.capabilities:
                continue
            
            agents.append({
                "agent_id": agent_info.agent_id,
                "name": agent_info.name,
                "base_url": agent_info.base_url,
                "status": agent_info.status.value,
                "capabilities": agent_info.capabilities,
                "last_heartbeat": agent_info.last_heartbeat.isoformat()
            })
        
        return {
            "agents": agents,
            "count": len(agents),
            "filters": {
                "status": status,
                "capability": capability
            }
        }
    
    @router.get("/a2a/agents/{agent_id}")
    async def get_agent_details(agent_id: str):
        """Get detailed information about a specific agent"""
        agent_info = agent.registered_agents.get(agent_id)
        
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return {
            "agent_id": agent_info.agent_id,
            "name": agent_info.name,
            "base_url": agent_info.base_url,
            "status": agent_info.status.value,
            "capabilities": agent_info.capabilities,
            "last_heartbeat": agent_info.last_heartbeat.isoformat(),
            "health_check_failures": agent_info.health_check_failures,
            "metadata": agent_info.metadata
        }
    
    @router.get("/a2a/workflows")
    async def list_workflows(status: str = None):
        """List workflows with optional status filter"""
        workflows = []
        
        for workflow in agent.active_workflows.values():
            if status and workflow.status.value != status:
                continue
            
            workflows.append({
                "workflow_id": workflow.workflow_id,
                "context_id": workflow.context_id,
                "status": workflow.status.value,
                "agents_involved": workflow.agents_involved,
                "current_agent": workflow.current_agent,
                "started_at": workflow.started_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "error": workflow.error
            })
        
        return {
            "workflows": workflows,
            "count": len(workflows),
            "filter": {"status": status} if status else None
        }
    
    @router.get("/a2a/workflows/{workflow_id}")
    async def get_workflow_details(workflow_id: str):
        """Get detailed information about a specific workflow"""
        workflow = agent.active_workflows.get(workflow_id)
        
        if not workflow:
            # Try to load from Redis
            data = await agent.redis_client.get(f"workflow:{workflow_id}")
            if data:
                workflow_data = json.loads(data)
                return workflow_data
            else:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "workflow_id": workflow.workflow_id,
            "context_id": workflow.context_id,
            "status": workflow.status.value,
            "agents_involved": workflow.agents_involved,
            "current_agent": workflow.current_agent,
            "started_at": workflow.started_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "error": workflow.error,
            "metadata": workflow.metadata
        }
    
    @router.post("/a2a/agents/{agent_id}/deregister")
    async def deregister_agent(agent_id: str):
        """Deregister an agent from the network"""
        if agent_id not in agent.registered_agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Remove from memory and Redis
        del agent.registered_agents[agent_id]
        await agent.redis_client.delete(f"agent:{agent_id}")
        
        return {
            "agent_id": agent_id,
            "status": "deregistered",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @router.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        metrics_text = f"""# HELP a2a_registered_agents_total Total number of registered agents
# TYPE a2a_registered_agents_total counter
a2a_registered_agents_total {agent.metrics['total_agents_registered']}

# HELP a2a_active_agents Number of currently active agents
# TYPE a2a_active_agents gauge
a2a_active_agents {len([a for a in agent.registered_agents.values() if a.status.value == 'active'])}

# HELP a2a_workflows_total Total number of workflows processed
# TYPE a2a_workflows_total counter
a2a_workflows_total {agent.metrics['total_workflows_processed']}

# HELP a2a_workflows_successful_total Total number of successful workflows
# TYPE a2a_workflows_successful_total counter
a2a_workflows_successful_total {agent.metrics['successful_workflows']}

# HELP a2a_workflows_failed_total Total number of failed workflows
# TYPE a2a_workflows_failed_total counter
a2a_workflows_failed_total {agent.metrics['failed_workflows']}

# HELP a2a_health_checks_total Total number of health checks performed
# TYPE a2a_health_checks_total counter
a2a_health_checks_total {agent.metrics['health_checks_performed']}
"""
        return metrics_text
    
    return router
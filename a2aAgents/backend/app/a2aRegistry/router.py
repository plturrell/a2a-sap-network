from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional
import logging

from .models import (
    AgentCard, AgentRegistrationRequest, AgentRegistrationResponse, AgentUpdateResponse,
    AgentSearchRequest, AgentSearchResponse, AgentDetails,
    AgentHealthResponse, AgentMetricsResponse, SystemHealthResponse,
    WorkflowMatchRequest, WorkflowMatchResponse, WorkflowPlanRequest, WorkflowPlanResponse,
    WorkflowExecutionRequest, WorkflowExecutionResponse, WorkflowExecutionStatus,
    ServiceHealthResponse, HealthStatus, AgentType
)
from .service import A2ARegistryService

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/v1/a2a", tags=["A2A Registry"])

# Service instance (singleton for consistency)
_service_instance = None

def get_a2a_service() -> A2ARegistryService:
    """Get A2A Registry service instance"""
    global _service_instance
    if _service_instance is None:
        # Initialize with ORD Registry integration
        ord_registry_url = "http://localhost:8000/api/v1/ord"
        _service_instance = A2ARegistryService(ord_registry_url=ord_registry_url)
    return _service_instance


# Agent Registration Endpoints

@router.post("/agents/register",
             response_model=AgentRegistrationResponse,
             status_code=201,
             summary="Register A2A Agent",
             description="Register a new A2A agent in the registry")
async def register_agent(
    request: AgentRegistrationRequest,
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentRegistrationResponse:
    """Register a new A2A agent"""
    try:
        return await service.register_agent(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/register/{agent_id}",
            response_model=AgentDetails,
            summary="Get Agent Registration",
            description="Retrieve details of a specific agent registration")
async def get_agent_registration(
    agent_id: str = Path(..., description="Agent ID"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentDetails:
    """Get agent registration details"""
    try:
        return await service.get_agent_details(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/agents/register/{agent_id}",
            response_model=AgentUpdateResponse,
            summary="Update Agent Registration",
            description="Update an existing agent registration")
async def update_agent_registration(
    agent_id: str = Path(..., description="Agent ID"),
    agent_card: AgentCard = ...,
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentUpdateResponse:
    """Update agent registration"""
    try:
        result = await service.update_agent(agent_id, agent_card)
        return AgentUpdateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/agents/register/{agent_id}",
               status_code=204,
               summary="Deregister Agent",
               description="Remove an agent from the registry")
async def deregister_agent(
    agent_id: str = Path(..., description="Agent ID"),
    service: A2ARegistryService = Depends(get_a2a_service)
):
    """Deregister an agent"""
    try:
        await service.deregister_agent(agent_id)
        return None
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deregistering agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Agent Discovery Endpoints

@router.get("/agents/search",
            response_model=AgentSearchResponse,
            summary="Search Agents",
            description="Search for agents by capabilities, skills, and other criteria")
async def search_agents(
    skills: Optional[str] = Query(None, description="Required skills (comma-separated)"),
    tags: Optional[str] = Query(None, description="Tags filter (comma-separated)"),
    agent_type: Optional[AgentType] = Query(None, description="Agent type filter"),
    status: Optional[HealthStatus] = Query(None, description="Health status filter"),
    inputModes: Optional[str] = Query(None, description="Required input modes (comma-separated)"),
    outputModes: Optional[str] = Query(None, description="Required output modes (comma-separated)"),
    page: int = Query(1, ge=1, description="Page number"),
    pageSize: int = Query(20, ge=1, le=100, description="Page size"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentSearchResponse:
    """Search for agents"""
    try:
        # Parse comma-separated parameters
        search_request = AgentSearchRequest(
            skills=skills.split(",") if skills else None,
            tags=tags.split(",") if tags else None,
            agent_type=agent_type,
            status=status,
            inputModes=inputModes.split(",") if inputModes else None,
            outputModes=outputModes.split(",") if outputModes else None,
            page=page,
            pageSize=pageSize
        )

        return await service.search_agents(search_request)
    except Exception as e:
        logger.error(f"Error searching agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/{agent_id}",
            response_model=AgentDetails,
            summary="Get Agent Details",
            description="Retrieve detailed information about a specific agent")
async def get_agent_details(
    agent_id: str = Path(..., description="Agent ID"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentDetails:
    """Get detailed agent information"""
    try:
        return await service.get_agent_details(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting agent details {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/agents/match",
             response_model=WorkflowMatchResponse,
             summary="Find Agents for Workflow",
             description="Find agents that match specific workflow requirements")
async def match_agents_for_workflow(
    request: WorkflowMatchRequest,
    service: A2ARegistryService = Depends(get_a2a_service)
) -> WorkflowMatchResponse:
    """Find agents for workflow"""
    try:
        return await service.match_workflow_agents(request)
    except Exception as e:
        logger.error(f"Error matching workflow agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health Monitoring Endpoints

@router.get("/agents/{agent_id}/health",
            response_model=AgentHealthResponse,
            summary="Get Agent Health Status",
            description="Retrieve current health status of a specific agent")
async def get_agent_health(
    agent_id: str = Path(..., description="Agent ID"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentHealthResponse:
    """Get agent health status"""
    try:
        return await service.get_agent_health(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting agent health {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/{agent_id}/metrics",
            response_model=AgentMetricsResponse,
            summary="Get Agent Metrics",
            description="Retrieve performance and usage metrics for an agent")
async def get_agent_metrics(
    agent_id: str = Path(..., description="Agent ID"),
    period: str = Query("24h", pattern="^(1h|24h|7d|30d)$", description="Time period for metrics"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> AgentMetricsResponse:
    """Get agent metrics"""
    try:
        return await service.get_agent_metrics(agent_id, period)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting agent metrics {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/system/health",
            response_model=SystemHealthResponse,
            summary="Get System Health Overview",
            description="Retrieve overall health status of the registry system")
async def get_system_health(
    service: A2ARegistryService = Depends(get_a2a_service)
) -> SystemHealthResponse:
    """Get system health overview"""
    try:
        return await service.get_system_health()
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Workflow Orchestration Endpoints

@router.post("/orchestration/plan",
             response_model=WorkflowPlanResponse,
             status_code=201,
             summary="Create Workflow Plan",
             description="Create an execution plan for a multi-agent workflow")
async def create_workflow_plan(
    request: WorkflowPlanRequest,
    service: A2ARegistryService = Depends(get_a2a_service)
) -> WorkflowPlanResponse:
    """Create workflow plan"""
    try:
        return await service.create_workflow_plan(request)
    except Exception as e:
        logger.error(f"Error creating workflow plan: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/orchestration/execute/{workflow_id}",
             response_model=WorkflowExecutionResponse,
             status_code=202,
             summary="Execute Workflow",
             description="Execute a workflow plan with provided input data")
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: WorkflowExecutionRequest = ...,
    service: A2ARegistryService = Depends(get_a2a_service)
) -> WorkflowExecutionResponse:
    """Execute workflow"""
    try:
        return await service.execute_workflow(workflow_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orchestration/status/{execution_id}",
            response_model=WorkflowExecutionStatus,
            summary="Get Workflow Execution Status",
            description="Retrieve the status of a workflow execution")
async def get_workflow_execution_status(
    execution_id: str = Path(..., description="Execution ID"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> WorkflowExecutionStatus:
    """Get workflow execution status"""
    try:
        return await service.get_workflow_execution_status(execution_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting workflow status {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Service Health Endpoint

@router.get("/health",
            response_model=ServiceHealthResponse,
            summary="Health Check",
            description="Check service health status")
async def health_check() -> ServiceHealthResponse:
    """Service health check"""
    return ServiceHealthResponse(
        status="healthy",
        services={
            "registry": "healthy",
            "search": "healthy",
            "orchestration": "healthy"
        },
        metrics={
            "version": "1.0.0",
            "uptime": "running"
        }
    )


# Utility endpoints for integration

@router.get("/agents/discover/by-skill/{skill_id}",
            response_model=List[AgentDetails],
            summary="Discover Agents by Skill",
            description="Find all agents that have a specific skill")
async def discover_agents_by_skill(
    skill_id: str = Path(..., description="Skill ID to search for"),
    service: A2ARegistryService = Depends(get_a2a_service)
) -> List[AgentDetails]:
    """Discover agents by specific skill"""
    try:
        search_request = AgentSearchRequest(
            skills=[skill_id],
            status=HealthStatus.HEALTHY,
            pageSize=100
        )

        search_response = await service.search_agents(search_request)

        # Get detailed information for each agent
        detailed_agents = []
        for result in search_response.results:
            try:
                details = await service.get_agent_details(result.agent_id)
                detailed_agents.append(details)
            except Exception as e:
                logger.warning(f"Could not get details for agent {result.agent_id}: {e}")

        return detailed_agents
    except Exception as e:
        logger.error(f"Error discovering agents by skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/statistics",
            summary="Get Agent Statistics",
            description="Get statistical overview of registered agents")
async def get_agent_statistics(
    service: A2ARegistryService = Depends(get_a2a_service)
) -> dict:
    """Get agent statistics"""
    try:
        # Get all agents
        search_request = AgentSearchRequest(pageSize=1000)
        search_response = await service.search_agents(search_request)

        # Calculate statistics
        total_agents = search_response.total_count
        healthy_agents = len([a for a in search_response.results if a.status == HealthStatus.HEALTHY])
        skill_distribution = {}
        tag_distribution = {}

        for agent in search_response.results:
            # Count skills
            for skill in agent.skills:
                skill_distribution[skill] = skill_distribution.get(skill, 0) + 1

            # Count tags
            for tag in agent.tags:
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1

        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "health_percentage": (healthy_agents / max(total_agents, 1)) * 100,
            "top_skills": sorted(skill_distribution.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_tags": sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)[:10],
            "total_skills": len(skill_distribution),
            "total_tags": len(tag_distribution)
        }
    except Exception as e:
        logger.error(f"Error getting agent statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

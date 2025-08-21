"""
A2A Trust System Router
REST API endpoints for trust relationship management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status

from .models import (
    TrustAgentRegistrationRequest, TrustAgentRegistrationResponse,
    TrustScoreResponse, InteractionRecord, TrustWorkflowRequest,
    TrustWorkflowResponse, SLACreationRequest,
    SystemHealth, TrustMetrics
)
from .service import TrustSystemService
from ..registry.service import A2ARegistryService

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/a2a/trust", tags=["A2A Trust System"])

# Global service instances
trust_service: Optional[TrustSystemService] = None
registry_service: Optional[A2ARegistryService] = None


def get_registry_service() -> A2ARegistryService:
    """Get A2A Registry service instance"""
    global registry_service
    if registry_service is None:
        registry_service = A2ARegistryService()
    return registry_service


def get_trust_service() -> TrustSystemService:
    """Get trust system service instance"""
    global trust_service
    if trust_service is None:
        trust_service = TrustSystemService(registry_service=get_registry_service())
    return trust_service


@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    service: TrustSystemService = Depends(get_trust_service)
):
    """Get trust system health status"""
    try:
        health = await service.get_system_health()
        return health
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )


@router.post(
    "/agents/register", 
    response_model=TrustAgentRegistrationResponse,
    status_code=status.HTTP_201_CREATED
)
async def register_agent_trust(
    request: TrustAgentRegistrationRequest,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Register A2A Agent with Trust System"""
    try:
        logger.info(f"Registering agent with trust system: {request.agent_card.get('name')}")
        
        response = await service.register_agent_with_trust(request)
        
        logger.info(f"Agent registered successfully: {response.trust_agent_id}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error in agent registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error registering agent with trust system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register agent: {str(e)}"
        )


@router.get("/agents/{agent_id}/score", response_model=TrustScoreResponse)
async def get_agent_trust_score(
    agent_id: str,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Get Agent Trust Score"""
    try:
        trust_score = await service.get_agent_trust_score(agent_id)
        return trust_score
        
    except Exception as e:
        logger.error(f"Error getting trust score for {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trust score: {str(e)}"
        )


@router.post("/agents/{agent_id}/interaction")
async def record_agent_interaction(
    agent_id: str,
    interaction: InteractionRecord,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Record Agent Interaction for trust score calculation"""
    try:
        success = await service.record_interaction(agent_id, interaction)
        
        if success:
            return {"message": "Interaction recorded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to record interaction"
            )
            
    except ValueError as e:
        logger.error(f"Validation error recording interaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(e)}"
        )


@router.post(
    "/workflows", 
    response_model=TrustWorkflowResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_trust_workflow(
    request: TrustWorkflowRequest,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Create Trust-Managed Workflow"""
    try:
        logger.info(f"Creating trust workflow: {request.workflow_definition.get('workflow_name')}")
        
        response = await service.create_trust_workflow(request)
        
        logger.info(f"Trust workflow created: {response.trust_workflow_id}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error creating workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating trust workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.post("/sla", status_code=status.HTTP_201_CREATED)
async def create_sla_contract(
    request: SLACreationRequest,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Create Service Level Agreement"""
    try:
        sla_id = await service.create_sla_contract(request)
        
        return {
            "sla_id": sla_id,
            "message": "SLA contract created successfully"
        }
        
    except ValueError as e:
        logger.error(f"Validation error creating SLA: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating SLA contract: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create SLA contract: {str(e)}"
        )


@router.get("/leaderboard")
async def get_trust_leaderboard(
    limit: int = 10,
    service: TrustSystemService = Depends(get_trust_service)
):
    """Get trust score leaderboard"""
    try:
        leaderboard = await service.get_trust_leaderboard(limit)
        
        return {
            "leaderboard": leaderboard,
            "total_agents": len(service.trust_scores)
        }
        
    except Exception as e:
        logger.error(f"Error getting trust leaderboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )


@router.get("/metrics", response_model=TrustMetrics)
async def get_trust_metrics(
    period: str = "24h",
    service: TrustSystemService = Depends(get_trust_service)
):
    """Get trust system metrics"""
    try:
        metrics = await service.get_trust_metrics(period)
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting trust metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/agents")
async def list_trust_agents(
    service: TrustSystemService = Depends(get_trust_service)
):
    """List all agents in trust system"""
    try:
        agents = [
            {
                "agent_id": agent_data["agent_id"],
                "registry_id": agent_data["registry_id"],
                "commitment_level": agent_data["commitment_level"],
                "status": agent_data["status"],
                "registration_timestamp": datetime.fromtimestamp(agent_data["registration_timestamp"]).isoformat()
            }
            for agent_data in service.trust_agents.values()
        ]
        
        return {
            "agents": agents,
            "total_count": len(agents)
        }
        
    except Exception as e:
        logger.error(f"Error listing trust agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )
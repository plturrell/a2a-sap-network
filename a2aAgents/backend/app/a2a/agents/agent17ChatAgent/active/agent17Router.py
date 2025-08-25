"""
Router for Agent 17 Chat Agent
Integrates with the main A2A agent router system
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from agent17ChatAgentA2AHandler import (
    ChatRequest,
    IntentAnalysisRequest,
    MultiAgentRequest,
    submit_to_blockchain,
    analyze_intent,
    list_agents,
    list_conversations,
    get_statistics,
    compliance_info,
    health_check
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(
    prefix="/agent17",
    tags=["agent17-chat"],
    responses={404: {"description": "Not found"}}
)


# Route definitions
@router.get("/")
async def agent17_info():
    """Get Agent 17 information"""
    return {
        "agent_id": "agent17_chat",
        "name": "A2A Chat Interface Agent",
        "version": "1.0.0",
        "description": "Conversational interface for multi-agent coordination via blockchain",
        "status": "active",
        "compliance": "A2A v0.2.9",
        "capabilities": [
            "conversational_interface",
            "intent_analysis",
            "multi_agent_routing",
            "response_synthesis"
        ]
    }


@router.get("/health")
async def agent17_health():
    """Health check for Agent 17"""
    return await health_check()


@router.post("/chat")
async def agent17_chat(request: ChatRequest):
    """Submit chat message via blockchain"""
    return await submit_to_blockchain(request)


@router.post("/analyze_intent")
async def agent17_analyze_intent(request: IntentAnalysisRequest):
    """Analyze user intent"""
    return await analyze_intent(request)


@router.get("/agents")
async def agent17_list_agents():
    """List discovered agents"""
    return await list_agents()


@router.get("/conversations")
async def agent17_conversations():
    """List active conversations"""
    return await list_conversations()


@router.get("/statistics")
async def agent17_statistics():
    """Get agent statistics"""
    return await get_statistics()


@router.get("/compliance")
async def agent17_compliance():
    """Get compliance information"""
    return await compliance_info()


# Export router for inclusion in main agent router
agent17_router = router
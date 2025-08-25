#!/usr/bin/env python3
"""
A2A Communication Router API
REST API service that exposes the A2A Human Communication Configuration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the communication router
from ..core.a2aHumanCommunicationConfig import (
    get_communication_router,
    A2AHumanCommunicationRouter,
    CommunicationRule,
    AgentCommunicationProfile,
    CommunicationTarget,
    MessageType,
    UrgencyLevel
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class A2AMessage(BaseModel):
    messageId: Optional[str] = None
    message_type: str
    category: Optional[str] = None
    urgency: Optional[str] = "medium"
    description: Optional[str] = None
    data: Optional[Dict[str, Any]] = {}
    requires_human_judgment: Optional[bool] = None
    involves_external_party: Optional[bool] = None
    data_classification: Optional[str] = None
    amount_threshold: Optional[float] = None
    confidence_level: Optional[float] = None
    routine: Optional[bool] = None
    standard_operation: Optional[bool] = None

class RoutingRequest(BaseModel):
    message: A2AMessage
    from_agent: str
    context: Optional[Dict[str, Any]] = {}

class RoutingResponse(BaseModel):
    target: str
    rule_applied: Dict[str, Any]
    requires_human_confirmation: bool = False
    escalation_timeout_minutes: Optional[int] = None
    human_response_timeout_minutes: Optional[int] = None
    timestamp: str
    message_id: str
    from_agent: str
    human_routing: Optional[Dict[str, Any]] = None
    agent_routing: Optional[Dict[str, Any]] = None
    escalation: Optional[Dict[str, Any]] = None

class RuleRequest(BaseModel):
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    target: str
    priority: int = 100
    active: bool = True
    require_human_confirmation: bool = False
    escalation_timeout_minutes: int = 30
    human_response_timeout_minutes: int = 60

class AgentProfileRequest(BaseModel):
    agent_id: str
    agent_name: str
    autonomy_level: str
    allowed_operations: List[str]
    requires_approval: List[str]
    escalation_triggers: List[str]
    communication_preferences: Dict[str, Any]
    human_supervisors: List[str] = []

# FastAPI app
app = FastAPI(
    title="A2A Communication Router API",
    description="Routes A2A messages between agents and humans based on intelligent rules",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_router() -> A2AHumanCommunicationRouter:
    """Dependency to get the communication router"""
    return get_communication_router()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    router = get_router()
    return {
        "status": "healthy",
        "service": "A2A Communication Router",
        "rules_loaded": len(router.rules),
        "agent_profiles": len(router.agent_profiles),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/route", response_model=RoutingResponse)
async def route_message(request: RoutingRequest, router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Route an A2A message to appropriate target(s)"""
    try:
        # Convert Pydantic model to dict
        message_dict = request.message.dict()

        # Route the message
        routing_decision = router.determine_communication_target(
            message=message_dict,
            from_agent=request.from_agent,
            context=request.context
        )

        return RoutingResponse(**routing_decision)

    except Exception as e:
        logger.error(f"Error routing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/profiles")
async def get_agent_profiles(router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Get all agent communication profiles"""
    return {
        "profiles": {
            agent_id: router.get_agent_communication_settings(agent_id)
            for agent_id in router.agent_profiles.keys()
        },
        "total_profiles": len(router.agent_profiles),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents/{agent_id}/profile")
async def get_agent_profile(agent_id: str, router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Get communication profile for a specific agent"""
    profile = router.get_agent_communication_settings(agent_id)

    if "error" in profile:
        raise HTTPException(status_code=404, detail=f"Agent profile not found: {agent_id}")

    return profile

@app.put("/agents/{agent_id}/profile")
async def update_agent_profile(
    agent_id: str,
    profile_request: AgentProfileRequest,
    router: A2AHumanCommunicationRouter = Depends(get_router)
):
    """Update communication profile for an agent"""
    try:
        # Create AgentCommunicationProfile from request
        profile = AgentCommunicationProfile(
            agent_id=profile_request.agent_id,
            agent_name=profile_request.agent_name,
            autonomy_level=profile_request.autonomy_level,
            allowed_operations=profile_request.allowed_operations,
            requires_approval=profile_request.requires_approval,
            escalation_triggers=profile_request.escalation_triggers,
            communication_preferences=profile_request.communication_preferences,
            human_supervisors=profile_request.human_supervisors
        )

        router.update_agent_profile(agent_id, profile)

        return {
            "message": f"Profile updated for agent {agent_id}",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating agent profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rules")
async def get_communication_rules(router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Get all communication rules"""
    rules_data = []

    for rule in sorted(router.rules, key=lambda r: r.priority):
        rules_data.append({
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "conditions": rule.conditions,
            "target": rule.target.value,
            "priority": rule.priority,
            "active": rule.active,
            "require_human_confirmation": rule.require_human_confirmation,
            "escalation_timeout_minutes": rule.escalation_timeout_minutes,
            "human_response_timeout_minutes": rule.human_response_timeout_minutes
        })

    return {
        "rules": rules_data,
        "total_rules": len(rules_data),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/rules")
async def add_communication_rule(
    rule_request: RuleRequest,
    router: A2AHumanCommunicationRouter = Depends(get_router)
):
    """Add a new communication rule"""
    try:
        # Create CommunicationRule from request
        rule = CommunicationRule(
            rule_id=rule_request.rule_id,
            name=rule_request.name,
            description=rule_request.description,
            conditions=rule_request.conditions,
            target=CommunicationTarget(rule_request.target),
            priority=rule_request.priority,
            active=rule_request.active,
            require_human_confirmation=rule_request.require_human_confirmation,
            escalation_timeout_minutes=rule_request.escalation_timeout_minutes,
            human_response_timeout_minutes=rule_request.human_response_timeout_minutes
        )

        router.add_custom_rule(rule)

        return {
            "message": f"Rule '{rule_request.rule_id}' added successfully",
            "rule_id": rule_request.rule_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error adding rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-routing")
async def test_message_routing(
    request: RoutingRequest,
    router: A2AHumanCommunicationRouter = Depends(get_router)
):
    """Test message routing without actually processing the message"""
    try:
        message_dict = request.message.dict()

        # Get routing decision
        routing_decision = router.determine_communication_target(
            message=message_dict,
            from_agent=request.from_agent,
            context=request.context
        )

        # Add test information
        routing_decision["test_mode"] = True
        routing_decision["message_would_be_routed_to"] = routing_decision["target"]

        return routing_decision

    except Exception as e:
        logger.error(f"Error testing routing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_routing_statistics(router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Get routing statistics and analytics"""
    try:
        # Calculate statistics from message history
        total_messages = len(router.message_history)

        target_counts = {}
        rule_usage = {}
        agent_activity = {}

        for msg in router.message_history:
            # Count targets
            target = msg.get("routing_target", "unknown")
            target_counts[target] = target_counts.get(target, 0) + 1

            # Count rule usage
            rule_id = msg.get("rule_applied", {}).get("rule_id", "unknown")
            rule_usage[rule_id] = rule_usage.get(rule_id, 0) + 1

            # Count agent activity
            from_agent = msg.get("from_agent", "unknown")
            agent_activity[from_agent] = agent_activity.get(from_agent, 0) + 1

        return {
            "total_messages_routed": total_messages,
            "routing_targets": target_counts,
            "rule_usage": rule_usage,
            "agent_activity": agent_activity,
            "active_rules": len([r for r in router.rules if r.active]),
            "total_rules": len(router.rules),
            "agent_profiles": len(router.agent_profiles),
            "pending_human_responses": len(router.pending_human_responses),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/configuration")
async def export_configuration(router: A2AHumanCommunicationRouter = Depends(get_router)):
    """Export complete configuration for backup/sharing"""
    try:
        config = router.export_configuration()
        config["exported_at"] = datetime.utcnow().isoformat()
        config["export_version"] = "1.0.0"

        return config

    except Exception as e:
        logger.error(f"Error exporting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate/scenario")
async def simulate_routing_scenario(
    scenario: Dict[str, Any],
    router: A2AHumanCommunicationRouter = Depends(get_router)
):
    """Simulate a routing scenario with multiple messages"""
    try:
        results = []

        messages = scenario.get("messages", [])
        for msg_data in messages:
            message = A2AMessage(**msg_data["message"])
            from_agent = msg_data.get("from_agent", "test_agent")
            context = msg_data.get("context", {})

            routing_decision = router.determine_communication_target(
                message=message.dict(),
                from_agent=from_agent,
                context=context
            )

            results.append({
                "message_id": message.messageId or f"sim_{len(results)}",
                "from_agent": from_agent,
                "routing_decision": routing_decision,
                "message_summary": f"{message.message_type} from {from_agent}"
            })

        return {
            "scenario_name": scenario.get("name", "Unnamed Scenario"),
            "total_messages": len(messages),
            "results": results,
            "summary": {
                "routed_to_human": len([r for r in results if r["routing_decision"]["target"] == "human"]),
                "routed_to_agent": len([r for r in results if r["routing_decision"]["target"] == "agent"]),
                "escalated": len([r for r in results if r["routing_decision"]["target"] == "escalate"]),
                "both_targets": len([r for r in results if r["routing_decision"]["target"] == "both"])
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error simulating scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global router instance for the API
communication_router = None

def create_router_api():
    """Create and return the communication router API"""
    global communication_router
    if communication_router is None:
        communication_router = get_communication_router()
    return app

# FastAPI app for direct usage
router_app = create_router_api()

if __name__ == "__main__":
    # Run the communication router API server
    uvicorn.run(
        "a2aCommunicationRouter:router_app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

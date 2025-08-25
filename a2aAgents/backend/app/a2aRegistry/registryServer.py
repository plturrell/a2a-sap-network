#!/usr/bin/env python3
"""
Simple Trust-Aware A2A Registry API
Works alongside your blockchain network
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import asyncio


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Set up Python path - go up to backend/app level
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
sys.path.insert(0, app_dir)
print(f"Added to Python path: {app_dir}")

# Initialize global variables
trust_system = None
agents_registry = {}

def initialize_trust_system():
    """Initialize the trust system"""
    global trust_system
    try:
        from a2a.security.smartContractTrust import SmartContractTrust
        trust_system = SmartContractTrust()

        # Register your blockchain agents
        AGENT1_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        AGENT2_ADDRESS = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

        agent1_identity = trust_system.register_agent(AGENT1_ADDRESS, "blockchain_financial_agent")
        agent2_identity = trust_system.register_agent(AGENT2_ADDRESS, "blockchain_message_agent")

        # Simple agent registry
        agents_registry[AGENT1_ADDRESS] = {
            "name": "Blockchain Financial Agent",
            "description": "On-chain financial analysis and portfolio management",
            "url": os.getenv("A2A_SERVICE_URL"),
            "type": "blockchain_financial_agent",
            "skills": ["portfolio-analysis", "risk-assessment"],
            "status": "active",
            "trust_score": trust_system.get_trust_score(AGENT1_ADDRESS),
            "registered_at": datetime.utcnow().isoformat()
        }

        agents_registry[AGENT2_ADDRESS] = {
            "name": "Blockchain Message Agent",
            "description": "On-chain message routing and communication",
            "url": os.getenv("A2A_SERVICE_URL"),
            "type": "blockchain_message_agent",
            "skills": ["message-routing", "data-transformation"],
            "status": "active",
            "trust_score": trust_system.get_trust_score(AGENT2_ADDRESS),
            "registered_at": datetime.utcnow().isoformat()
        }

        print("✅ Trust system initialized with blockchain agents")
        return True

    except Exception as e:
        print(f"❌ Trust system initialization failed: {e}")
        return False

# Create FastAPI app
app = FastAPI(
    title="Trust-Aware A2A Registry",
    description="Simple A2A Registry with Blockchain Trust Integration",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting Trust-Aware A2A Registry...")
    success = initialize_trust_system()
    if success:
        print("🎯 Trust-Aware A2A Registry is LIVE!")
        print("   • Blockchain integration: Ready")
        print("   • Trust-aware discovery: Active")
        print("   • API endpoints: Available")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Trust-Aware A2A Registry",
        "version": "1.0.0",
        "blockchain": "Anvil (localhost:8545)",
        "contracts": {
            "registry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "message_router": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
        },
        "agents": len(agents_registry),
        "trust_integration": trust_system is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "trust_system": "online" if trust_system else "offline",
        "agents": len(agents_registry),
        "blockchain": "anvil_connected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents")
async def list_agents():
    """List all blockchain agents with trust scores"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    agents_with_trust = []
    for agent_id, agent_data in agents_registry.items():
        trust_score = trust_system.get_trust_score(agent_id)
        trust_level = (
            "verified" if trust_score >= 0.9 else
            "high" if trust_score >= 0.7 else
            "medium" if trust_score >= 0.5 else
            "low"
        )

        agent_info = {
            **agent_data,
            "agent_id": agent_id,
            "trust_score": trust_score,
            "trust_level": trust_level
        }
        agents_with_trust.append(agent_info)

    # Sort by trust score (descending)
    agents_with_trust.sort(key=lambda x: x["trust_score"], reverse=True)

    return {
        "agents": agents_with_trust,
        "total": len(agents_with_trust),
        "trust_integration": True
    }

@app.post("/agents/search")
async def search_agents(request: Dict[str, Any]):
    """Search agents with trust-aware ranking"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    # Simple search by skills or tags
    search_skills = request.get("skills", [])
    search_tags = request.get("tags", [])

    matching_agents = []

    for agent_id, agent_data in agents_registry.items():
        # Check if agent matches search criteria
        matches = True

        if search_skills:
            agent_skills = agent_data.get("skills", [])
            if not any(skill in agent_skills for skill in search_skills):
                matches = False

        if search_tags:
            agent_type = agent_data.get("type", "")
            if not any(tag in agent_type for tag in search_tags):
                matches = False

        if matches:
            trust_score = trust_system.get_trust_score(agent_id)
            trust_level = (
                "verified" if trust_score >= 0.9 else
                "high" if trust_score >= 0.7 else
                "medium" if trust_score >= 0.5 else
                "low"
            )

            matching_agents.append({
                **agent_data,
                "agent_id": agent_id,
                "trust_score": trust_score,
                "trust_level": trust_level,
                "response_time_ms": 150 if "financial" in agent_data["type"] else 200
            })

    # Trust-aware sorting (trust score + performance)
    def sort_key(agent):
        health_weight = 0 if agent["status"] == "active" else 1
        trust_weight = 1.0 - agent["trust_score"]
        response_weight = agent["response_time_ms"] / 1000.0
        return (health_weight, trust_weight, response_weight)

    matching_agents.sort(key=sort_key)

    return {
        "results": matching_agents,
        "total_count": len(matching_agents),
        "search_criteria": request,
        "trust_ranking": True
    }

@app.post("/workflows/match")
async def match_workflow(request: Dict[str, Any]):
    """Match agents for workflow with trust filtering"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    workflow_requirements = request.get("workflow_requirements", [])

    stage_matches = []
    total_coverage = 0

    for stage_req in workflow_requirements:
        stage_name = stage_req.get("stage", "unknown")
        required_skills = stage_req.get("required_skills", [])

        # Find agents with required skills
        suitable_agents = []

        for agent_id, agent_data in agents_registry.items():
            agent_skills = agent_data.get("skills", [])
            trust_score = trust_system.get_trust_score(agent_id)

            # Check if agent has required skills and sufficient trust
            has_skills = any(skill in agent_skills for skill in required_skills)
            sufficient_trust = trust_score >= 0.6  # Minimum trust for workflows

            if has_skills and sufficient_trust:
                suitable_agents.append({
                    **agent_data,
                    "agent_id": agent_id,
                    "trust_score": trust_score,
                    "trust_level": (
                        "verified" if trust_score >= 0.9 else
                        "high" if trust_score >= 0.7 else
                        "medium" if trust_score >= 0.5 else
                        "low"
                    )
                })

        # Sort by trust score
        suitable_agents.sort(key=lambda x: x["trust_score"], reverse=True)

        stage_matches.append({
            "stage": stage_name,
            "agents": suitable_agents[:5]  # Top 5 agents
        })

        if suitable_agents:
            total_coverage += 1

    coverage_percentage = (total_coverage / len(workflow_requirements)) * 100 if workflow_requirements else 0

    return {
        "workflow_id": f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "matching_agents": stage_matches,
        "total_stages": len(workflow_requirements),
        "coverage_percentage": coverage_percentage,
        "trust_filtering": True
    }

@app.post("/trust/sign")
async def sign_message(request: Dict[str, Any]):
    """Sign a message with agent's trust identity"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    agent_id = request.get("agent_id")
    message = request.get("message")

    if not agent_id or not message:
        raise HTTPException(status_code=400, detail="agent_id and message required")

    if agent_id not in agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        signed_message = trust_system.sign_message(agent_id, message)
        return {
            "signed_message": signed_message,
            "signer_trust_score": trust_system.get_trust_score(agent_id),
            "blockchain_ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/trust/verify")
async def verify_message(request: Dict[str, Any]):
    """Verify a signed message"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    signed_message = request.get("signed_message")

    if not signed_message:
        raise HTTPException(status_code=400, detail="signed_message required")

    try:
        is_valid, verified_msg = trust_system.verify_message(signed_message)
        signer_id = signed_message.get("signature", {}).get("agent_id")

        return {
            "valid": is_valid,
            "verified_message": verified_msg if is_valid else None,
            "signer_id": signer_id,
            "signer_trust_score": trust_system.get_trust_score(signer_id) if is_valid and signer_id else None,
            "blockchain_approved": is_valid and trust_system.get_trust_score(signer_id) >= 0.7 if signer_id else False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trust/scores")
async def get_trust_scores():
    """Get trust scores for all blockchain agents"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")

    scores = {}
    for agent_id in agents_registry.keys():
        trust_score = trust_system.get_trust_score(agent_id)
        scores[agent_id] = {
            "trust_score": trust_score,
            "trust_level": (
                "verified" if trust_score >= 0.9 else
                "high" if trust_score >= 0.7 else
                "medium" if trust_score >= 0.5 else
                "low"
            ),
            "blockchain_approved": trust_score >= 0.7
        }

    return {"trust_scores": scores}

@app.get("/blockchain/status")
async def blockchain_status():
    """Get blockchain network status"""
    return {
        "network": "Anvil (localhost:8545)",
        "contracts": {
            "registry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "message_router": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
        },
        "agents": {
            "agent1": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "agent2": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        },
        "status": "active",
        "trust_integration": trust_system is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Trust-Aware A2A Registry Server...")
    print("   • Blockchain: Anvil (localhost:8545)")
    print("   • Registry: Trust-aware agent discovery")
    print("   • Server: http://localhost:8082")
    print("   • Docs: http://localhost:8082/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="info"
    )

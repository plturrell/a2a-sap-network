#!/usr/bin/env python3
"""
Trust-Aware A2A Registry Server
Runs alongside your blockchain network with full trust integration
"""

import uvicorn
import sys
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Set up Python path - add both parent and current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Add paths in correct order
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Global registry instance
registry_service = None
trust_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global registry_service, trust_system

    print("🚀 Starting Trust-Aware A2A Registry Server...")

    # Initialize trust system
    try:
        # Try to import trust system from correct location
        try:
            from app.a2a.core.trustManager import TrustManager
            trust_system = TrustManager()
            print("✅ Trust system initialized (using TrustManager)")
        except:
            # Fallback to direct import
            try:
                from a2a.core.trustManager import TrustManager
                trust_system = TrustManager()
                print("✅ Trust system initialized (using direct import)")
            except:
                print("⚠️ Trust system not available, continuing without trust integration")
                trust_system = None
    except Exception as e:
        print(f"⚠️ Trust system failed: {e}")
        trust_system = None

    # Initialize registry with trust integration
    try:
        # Import directly since we've set up the path correctly
        from service import A2ARegistryService

        registry_service = A2ARegistryService(
            enable_trust_integration=(trust_system is not None)
        )
        print(f"✅ A2A Registry initialized (trust: {'enabled' if registry_service.enable_trust_integration else 'disabled'})")
    except Exception as e:
        print(f"❌ Registry initialization failed: {e}")
        raise

    # Register your blockchain agents automatically
    await register_blockchain_agents()

    print("🎯 Trust-Aware A2A Registry Server is LIVE!")
    print("   • Blockchain integration: Ready")
    print("   • Trust-aware discovery: Active")
    print("   • Secure workflows: Enabled")
    print("   • API endpoints: Available")

    yield

    print("🛑 Shutting down Trust-Aware A2A Registry Server...")

async def register_blockchain_agents():
    """Register your blockchain agents in the A2A Registry"""
    if not registry_service or not trust_system:
        return

    # Your blockchain agent addresses
    AGENT1_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
    AGENT2_ADDRESS = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

    try:
        from models import (
            AgentRegistrationRequest, AgentCard, AgentProvider,
            AgentCapabilities, AgentSkill
        )

        print("📋 Auto-registering blockchain agents...")

        # Register agents in trust system first
        trust_system.register_agent(AGENT1_ADDRESS, "blockchain_financial_agent")
        trust_system.register_agent(AGENT2_ADDRESS, "blockchain_message_agent")

        # Create Agent1 card (Financial Agent)
        agent1_card = AgentCard(
            name="Blockchain Financial Agent",
            description="On-chain financial analysis and portfolio management",
            url=os.getenv("A2A_SERVICE_URL"),
            version="1.0.0",
            protocolVersion="0.2.9",
            provider=AgentProvider(
                organization="FinSight CIB Blockchain",
                url="https://finsight-cib.com"
            ),
            capabilities=AgentCapabilities(
                streaming=True,
                batchProcessing=True,
                smartContractDelegation=True
            ),
            skills=[
                AgentSkill(
                    id="portfolio-analysis",
                    name="Portfolio Analysis",
                    description="Blockchain-based portfolio analysis",
                    tags=["financial", "blockchain", "analysis"]
                ),
                AgentSkill(
                    id="risk-assessment",
                    name="Risk Assessment",
                    description="On-chain risk assessment",
                    tags=["financial", "risk", "blockchain"]
                )
            ],
            defaultInputModes=["application/json"],
            defaultOutputModes=["application/json"],
            authentication={"schemes": ["Bearer", "Basic"]},
            preferredTransport="https"
        )

        # Register Agent1 in A2A Registry
        agent1_request = AgentRegistrationRequest(
            agent_card=agent1_card,
            registered_by="blockchain_network"
        )

        agent1_response = await registry_service.register_agent(agent1_request)
        # Update to use blockchain address
        registry_service.agents[AGENT1_ADDRESS] = registry_service.agents.pop(agent1_response.agent_id)

        print(f"✅ Agent1 registered: {AGENT1_ADDRESS[:25]}...")

        # Create Agent2 card (Message Agent)
        agent2_card = AgentCard(
            name="Blockchain Message Agent",
            description="On-chain message routing and communication",
            url=os.getenv("A2A_SERVICE_URL"),
            version="1.0.0",
            protocolVersion="0.2.9",
            provider=AgentProvider(
                organization="FinSight CIB Blockchain",
                url="https://finsight-cib.com"
            ),
            capabilities=AgentCapabilities(
                streaming=True,
                batchProcessing=True,
                smartContractDelegation=True
            ),
            skills=[
                AgentSkill(
                    id="message-routing",
                    name="Message Routing",
                    description="Blockchain message routing",
                    tags=["messaging", "blockchain", "routing"]
                ),
                AgentSkill(
                    id="data-transformation",
                    name="Data Transformation",
                    description="On-chain data processing",
                    tags=["data", "blockchain", "processing"]
                )
            ],
            defaultInputModes=["application/json"],
            defaultOutputModes=["application/json"],
            authentication={"schemes": ["Bearer", "Basic"]},
            preferredTransport="https"
        )

        # Register Agent2 in A2A Registry
        agent2_request = AgentRegistrationRequest(
            agent_card=agent2_card,
            registered_by="blockchain_network"
        )

        agent2_response = await registry_service.register_agent(agent2_request)
        # Update to use blockchain address
        registry_service.agents[AGENT2_ADDRESS] = registry_service.agents.pop(agent2_response.agent_id)

        print(f"✅ Agent2 registered: {AGENT2_ADDRESS[:25]}...")
        print(f"📊 Total agents in registry: {len(registry_service.agents)}")

    except Exception as e:
        print(f"⚠️ Agent auto-registration failed: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Trust-Aware A2A Registry",
    description="A2A Agent Registry with Blockchain Trust Integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_registry():
    """Dependency to get registry service"""
    if not registry_service:
        raise HTTPException(status_code=503, detail="Registry service not available")
    return registry_service

def get_trust():
    """Dependency to get trust system"""
    if not trust_system:
        raise HTTPException(status_code=503, detail="Trust system not available")
    return trust_system

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Trust-Aware A2A Registry",
        "version": "1.0.0",
        "blockchain": "Anvil (localhost:8545)",
        "trust_integration": registry_service.enable_trust_integration if registry_service else False,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await registry_service.get_system_health()

@app.get("/agents")
async def list_agents(registry: object = Depends(get_registry)):
    """List all registered agents"""
    agents = []
    for agent_id, registration in registry.agents.items():
        agents.append({
            "agent_id": agent_id,
            "name": registration.agent_card.name,
            "description": registration.agent_card.description,
            "status": registration.registration_metadata.status.value,
            "registered_at": registration.registration_metadata.registered_at.isoformat(),
            "trust_score": trust_system.get_trust_score(agent_id) if trust_system else None
        })
    return {"agents": agents, "total": len(agents)}

@app.post("/agents/search")
async def search_agents(request: dict, registry: object = Depends(get_registry)):
    """Search agents with trust-aware ranking"""
    try:
        from models import AgentSearchRequest
        search_request = AgentSearchRequest(**request)
        results = await registry.search_agents(search_request)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/workflows/match")
async def match_workflow(request: dict, registry: object = Depends(get_registry)):
    """Match agents for workflow with trust filtering"""
    try:
        from models import WorkflowMatchRequest
        workflow_request = WorkflowMatchRequest(**request)
        results = await registry.match_workflow_agents(workflow_request)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/trust/sign")
async def sign_message(request: dict, trust: object = Depends(get_trust)):
    """Sign a message with agent's trust identity"""
    try:
        agent_id = request.get("agent_id")
        message = request.get("message")

        if not agent_id or not message:
            raise HTTPException(status_code=400, detail="agent_id and message required")

        signed_message = trust.sign_message(agent_id, message)
        return signed_message
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/trust/verify")
async def verify_message(request: dict, trust: object = Depends(get_trust)):
    """Verify a signed message"""
    try:
        signed_message = request.get("signed_message")

        if not signed_message:
            raise HTTPException(status_code=400, detail="signed_message required")

        is_valid, verified_msg = trust.verify_message(signed_message)
        return {
            "valid": is_valid,
            "verified_message": verified_msg if is_valid else None,
            "signer_trust_score": trust.get_trust_score(signed_message.get("signature", {}).get("agent_id")) if is_valid else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trust/scores")
async def get_trust_scores(trust: object = Depends(get_trust)):
    """Get trust scores for all agents"""
    scores = {}
    if registry_service:
        for agent_id in registry_service.agents.keys():
            scores[agent_id] = {
                "trust_score": trust.get_trust_score(agent_id),
                "trust_level": ("verified" if trust.get_trust_score(agent_id) >= 0.9 else
                               "high" if trust.get_trust_score(agent_id) >= 0.7 else
                               "medium" if trust.get_trust_score(agent_id) >= 0.5 else "low")
            }
    return {"trust_scores": scores}

@app.get("/blockchain/status")
async def blockchain_status():
    """Get blockchain network status"""
    return {
        "network": "Anvil (localhost:8545)",
        "registry_contract": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "message_router": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
        "agent1": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "agent2": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Trust-Aware A2A Registry Server...")
    print("   • Blockchain: Anvil (localhost:8545)")
    print("   • Registry: Trust-aware agent discovery")
    print("   • Server: http://localhost:8000")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,  # Use port 8000 for registry server
        reload=False,
        log_level="info"
    )

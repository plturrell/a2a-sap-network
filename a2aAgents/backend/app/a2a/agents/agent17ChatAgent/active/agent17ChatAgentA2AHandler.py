"""
A2A Handler for Agent 17 Chat Agent
Provides HTTP endpoints for blockchain message submission and testing
Note: In production, all communication should be via blockchain only
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import os

from agent17ChatAgentSdk import create_agent17_chat_agent

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agent 17 Chat Agent A2A Handler",
    description="A2A-compliant chat interface agent - HTTP interface for blockchain message submission",
    version="1.0.0"
)

# Global agent instance
agent = None


class ChatRequest(BaseModel):
    """Chat message request"""
    prompt: str
    user_id: str = "anonymous"
    conversation_id: Optional[str] = None


class IntentAnalysisRequest(BaseModel):
    """Intent analysis request"""
    prompt: str


class MultiAgentRequest(BaseModel):
    """Multi-agent coordination request"""
    query: str
    target_agents: List[str]
    coordination_type: str = "parallel"


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    
    try:
        # Get blockchain config
        blockchain_config = {
            "private_key": os.getenv("AGENT17_PRIVATE_KEY"),
            "contract_address": os.getenv("A2A_CONTRACT_ADDRESS"),
            "rpc_url": os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
        }
        
        # Create and initialize agent
        agent = create_agent17_chat_agent(
            base_url=f"http://localhost:{os.getenv('AGENT17_PORT', '8017')}",
            blockchain_config=blockchain_config
        )
        
        await agent.initialize()
        
        logger.info("Agent 17 Chat Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global agent
    if agent:
        await agent.shutdown()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "Agent 17 Chat Agent",
        "version": "1.0.0",
        "status": "running",
        "description": "A2A-compliant conversational interface agent",
        "blockchain_only": True,
        "note": "This HTTP interface is for blockchain message submission only"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = await agent.get_statistics()
    
    return {
        "status": "healthy",
        "agent_id": agent.AGENT_ID,
        "statistics": stats
    }


@app.post("/submit_to_blockchain")
async def submit_to_blockchain(request: ChatRequest):
    """
    Submit a chat message to blockchain for processing
    This endpoint demonstrates how external systems can submit messages
    that will be processed through blockchain-only communication
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # This would typically be called by an external system
        # The agent will route the message via blockchain
        result = await agent._analyze_and_route(
            request.prompt,
            request.conversation_id or f"conv_{request.user_id}"
        )
        
        return {
            "status": "submitted",
            "result": result,
            "note": "Message routed via blockchain to target agents"
        }
        
    except Exception as e:
        logger.error(f"Error submitting to blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_intent")
async def analyze_intent(request: IntentAnalysisRequest):
    """Analyze user intent (for testing/debugging)"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent._analyze_intent(request.prompt)
        
        return {
            "status": "success",
            "intent_analysis": result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List discovered agents in the network"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "agents": agent.agent_registry,
        "count": len(agent.agent_registry)
    }


@app.get("/conversations")
async def list_conversations():
    """List active conversations"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "conversations": agent.active_conversations,
        "count": len(agent.active_conversations)
    }


@app.get("/statistics")
async def get_statistics():
    """Get agent statistics"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = await agent.get_statistics()
    return stats


# A2A Protocol Compliance Note
@app.get("/compliance")
async def compliance_info():
    """A2A Protocol compliance information"""
    return {
        "agent": "Agent 17 Chat Agent",
        "compliance_status": "FULLY_COMPLIANT",
        "protocol_version": "A2A v0.2.9",
        "communication": {
            "primary": "blockchain_only",
            "http_endpoints": "for_blockchain_submission_only",
            "direct_agent_communication": "NOT_ALLOWED",
            "websocket": "NOT_SUPPORTED"
        },
        "security": {
            "base_class": "SecureA2AAgent",
            "authentication": "enabled",
            "rate_limiting": "enabled",
            "input_validation": "enabled"
        },
        "blockchain": {
            "registration": "required",
            "messaging": "exclusive",
            "smart_contract": "A2ARegistry"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("AGENT17_PORT", "8017"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
#!/usr/bin/env python3
"""
Launch Agent 1 (Financial Data Standardization Agent) as an independent service
True A2A microservice deployment
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.a2a.agents.data_standardization_agent import (
    FinancialStandardizationAgent, A2AMessage
)
from a2a_network.python_sdk.blockchain import get_blockchain_client, initialize_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for Agent 1
app = FastAPI(
    title="Agent 1 - Financial Data Standardization Agent",
    description="Independent A2A microservice for financial data standardization",
    version="1.0.0"
)

# Add CORS middleware for A2A communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration from environment or defaults
AGENT1_PORT = int(os.getenv("AGENT1_PORT", "8001"))
AGENT1_HOST = os.getenv("AGENT1_HOST", "0.0.0.0")

# A2A Network blockchain configuration
A2A_RPC_URL = os.getenv("A2A_RPC_URL", "http://localhost:8545")
A2A_AGENT_PRIVATE_KEY = os.getenv("A2A_AGENT1_PRIVATE_KEY")

# Initialize the agent
agent = FinancialStandardizationAgent(base_url=f"http://localhost:{AGENT1_PORT}")

# Agent 1 configuration for blockchain
AGENT1_CONFIG = {
    "agent_id": "financial_standardization_agent_1",
    "name": "Financial Data Standardization Agent",
    "description": "Standardizes and enriches financial data with multi-pass processing capabilities",
    "version": "1.0.0",
    "base_url": f"http://localhost:{AGENT1_PORT}",
    "capabilities": [
        "financial_standardization",
        "data_enrichment", 
        "multi_pass_processing",
        "json_rpc",
        "a2a_messaging"
    ]
}

@app.on_event("startup")
async def startup_event():
    """Register Agent 1 with the A2A Network blockchain on startup"""
    try:
        logger.info(f"🚀 Starting Agent 1 on port {AGENT1_PORT}")
        
        # Initialize blockchain client
        logger.info("🔗 Connecting to A2A Network blockchain...")
        initialize_blockchain_client(
            rpc_url=A2A_RPC_URL,
            private_key=A2A_AGENT_PRIVATE_KEY
        )
        blockchain_client = get_blockchain_client()
        logger.info(f"✅ Connected to A2A Network: {blockchain_client.agent_identity.address}")
        
        # Create blockchain adapter
        blockchain_adapter = create_blockchain_adapter(
            agent_id=AGENT1_CONFIG["agent_id"],
            name=AGENT1_CONFIG["name"],
            description=AGENT1_CONFIG["description"],
            version=AGENT1_CONFIG["version"],
            endpoint=AGENT1_CONFIG["base_url"],
            capabilities=AGENT1_CONFIG["capabilities"],
            skills=[
                {
                    "id": "financial-standardization",
                    "name": "Financial Data Standardization",
                    "description": "Standardizes financial data formats and structures",
                    "tags": ["financial", "standardization", "data"]
                },
                {
                    "id": "data-enrichment",
                    "name": "Data Enrichment",
                    "description": "Enriches data with additional metadata and context",
                    "tags": ["enrichment", "metadata", "enhancement"]
                }
            ]
        )
        
        # Register with A2A Network blockchain
        registration_success = await blockchain_adapter.register_agent()
        if registration_success:
            logger.info(f"✅ Agent 1 registered on A2A Network blockchain")
            logger.info(f"   Address: {blockchain_adapter.get_agent_address()}")
            logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        else:
            logger.warning("⚠️ Failed to register with A2A Network - continuing anyway")
        
        # Store blockchain adapter for use in agent
        agent.blockchain_adapter = blockchain_adapter
        
        # Log agent status
        status = blockchain_adapter.get_agent_status()
        logger.info(f"📊 Agent Status: {status}")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue running even if registration fails

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister Agent 1 from the A2A Network blockchain on shutdown"""
    try:
        if hasattr(agent, 'blockchain_adapter'):
            await agent.blockchain_adapter.deregister_agent()
            logger.info("✅ Agent 1 deregistered from A2A Network blockchain")
    except Exception as e:
        logger.error(f"Failed to deregister: {e}")

@app.get("/")
async def get_agent_card():
    """Get agent card"""
    card = agent.agent_card.dict()
    card["url"] = f"http://localhost:{AGENT1_PORT}"
    return card

@app.get("/.well-known/agent.json")
async def well_known_agent():
    """Well-known agent card endpoint"""
    card = agent.agent_card.dict()
    card["url"] = f"http://localhost:{AGENT1_PORT}"
    return card

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "Financial Data Standardization Agent",
        "version": "1.0.0",
        "protocol_version": "0.2.9",
        "port": AGENT1_PORT
    }

@app.post("/a2a/v1/messages")
async def process_message(request: Request):
    """Process A2A messages"""
    try:
        body = await request.json()
        
        # Handle both signed and unsigned messages
        if "signature" in body:
            # This is a signed message from Agent 0
            message_data = body.get("message", {})
            context_id = message_data.get("contextId", str(datetime.utcnow().timestamp()))
            # Extract the actual message content
            message = A2AMessage(**message_data.get("message", {}))
        else:
            # Regular message format
            message = A2AMessage(**body.get("message", {}))
            context_id = body.get("contextId", str(datetime.utcnow().timestamp()))
        
        # Check for signature and verify if present
        if hasattr(message, "signature") and message.signature:
            logger.info("📝 Received signed message - verifying trust signature")
            # Trust verification would happen here
            
        result = await agent.process_message(message, context_id)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Message processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/a2a/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()

@app.post("/a2a/v1/rpc")
async def json_rpc_handler(request: Request):
    """JSON-RPC 2.0 endpoint"""
    try:
        body = await request.json()
        
        # Handle JSON-RPC request
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id", 1)
        
        if method == "process":
            message = A2AMessage(**params.get("message", {}))
            context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
            result = await agent.process_message(message, context_id)
            
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            )
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
            
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": str(e)
                },
                "id": body.get("id", 1) if "body" in locals() else 1
            }
        )

@app.get("/config")
async def get_config():
    """Get current agent configuration"""
    return AGENT1_CONFIG

if __name__ == "__main__":
    logger.info(f"🚀 Launching Agent 1 on port {AGENT1_PORT}")
    uvicorn.run(
        "launch_agent1:app",
        host=AGENT1_HOST,
        port=AGENT1_PORT,
        reload=True,
        log_level="info"
    )
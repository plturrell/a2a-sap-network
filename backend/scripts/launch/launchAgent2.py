#!/usr/bin/env python3
"""
Launch Agent 2 (AI Preparation Agent) as an independent service
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

from app.a2a.agents.agent2_router import router as agent2_router
from a2a_network.python_sdk.blockchain import get_blockchain_client, initialize_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter
from app.a2a.core.a2aTypes import A2AMessage


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for Agent 2
app = FastAPI(
    title="Agent 2 - AI Preparation Agent",
    description="Independent A2A microservice for AI data preparation and enhancement",
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
AGENT2_PORT = int(os.getenv("AGENT2_PORT", "8003"))
AGENT2_HOST = os.getenv("AGENT2_HOST", "0.0.0.0")

# A2A Network blockchain configuration
A2A_RPC_URL = os.getenv("A2A_RPC_URL", "os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))")
A2A_AGENT_PRIVATE_KEY = os.getenv("A2A_AGENT2_PRIVATE_KEY")

# Initialize the agent (will be created after blockchain setup)
agent2 = None

# Agent 2 configuration for blockchain
AGENT2_CONFIG = {
    "agent_id": "ai_preparation_agent_2",
    "name": "AI Preparation Agent",
    "description": "Prepares and enhances data for AI/ML training and inference with quality validation",
    "version": "1.0.0",
    "base_url": f"http://localhost:{AGENT2_PORT}",
    "capabilities": [
        "ai_data_preparation",
        "semantic_enrichment",
        "vector_embeddings",
        "knowledge_graph_structuring",
        "quality_validation",
        "json_rpc",
        "a2a_messaging"
    ]
}

@app.on_event("startup")
async def startup_event():
    """Register Agent 2 with the A2A Network blockchain on startup"""
    global agent2
    try:
        logger.info(f"🚀 Starting Agent 2 on port {AGENT2_PORT}")
        
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
            agent_id=AGENT2_CONFIG["agent_id"],
            name=AGENT2_CONFIG["name"],
            description=AGENT2_CONFIG["description"],
            version=AGENT2_CONFIG["version"],
            endpoint=AGENT2_CONFIG["base_url"],
            capabilities=AGENT2_CONFIG["capabilities"],
            skills=[
                {
                    "id": "semantic-context-enrichment",
                    "name": "Semantic Context Enrichment",
                    "description": "Add rich semantic context, business descriptions, and domain-specific terminology",
                    "tags": ["semantic", "context", "nlp", "business-intelligence"]
                },
                {
                    "id": "entity-relationship-discovery",
                    "name": "Entity Relationship Discovery",
                    "description": "Discover and map relationships between entities across different financial dimensions",
                    "tags": ["relationships", "graph", "discovery", "cross-entity"]
                },
                {
                    "id": "multi-dimensional-feature-extraction",
                    "name": "Multi-Dimensional Feature Extraction",
                    "description": "Extract semantic, hierarchical, contextual, and quality features for vector embeddings",
                    "tags": ["features", "vectorization", "multi-dimensional", "ai-ready"]
                },
                {
                    "id": "vector-embedding-generation",
                    "name": "Vector Embedding Generation",
                    "description": "Generate specialized vector embeddings optimized for financial domain understanding",
                    "tags": ["embeddings", "vectors", "neural", "domain-specific"]
                },
                {
                    "id": "knowledge-graph-structuring",
                    "name": "Knowledge Graph Structuring",
                    "description": "Structure entities and relationships for RDF knowledge graph representation",
                    "tags": ["rdf", "knowledge-graph", "ontology", "turtle"]
                },
                {
                    "id": "ai-readiness-validation",
                    "name": "AI Readiness Validation",
                    "description": "Validate data quality and completeness for AI processing and knowledge graph ingestion",
                    "tags": ["validation", "quality", "ai-readiness", "completeness"]
                }
            ]
        )
        
        # Register with A2A Network blockchain
        registration_success = await blockchain_adapter.register_agent()
        if registration_success:
            logger.info(f"✅ Agent 2 registered on A2A Network blockchain")
            logger.info(f"   Address: {blockchain_adapter.get_agent_address()}")
            logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        else:
            logger.warning("⚠️ Failed to register with A2A Network - continuing anyway")
        
        # Initialize Agent 2 with the actual AI preparation agent
        from app.a2a.agents.ai_preparation_agent import AIPreparationAgent
        agent2 = AIPreparationAgent(
            base_url=f"http://localhost:{AGENT2_PORT}",
            agent_id=AGENT2_CONFIG["agent_id"]
        )
        
        # Store blockchain adapter for use in agent
        agent2.blockchain_adapter = blockchain_adapter
        
        # Start the message queue processor
        await agent2.start_message_queue_processor()
        
        # Set the global agent instance in the router
        import app.a2a.agents.agent2_router as router_module
        router_module.agent2 = agent2
        
        # Now include the router
        app.include_router(agent2_router)
        logger.info("✅ Agent 2 router included")
        
        # Log agent status
        status = blockchain_adapter.get_agent_status()
        logger.info(f"📊 Agent Status: {status}")
        logger.info("✅ Agent 2 initialized with blockchain integration and message queue processor")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue running even if registration fails

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister Agent 2 from the A2A Network blockchain and stop message queue on shutdown"""
    try:
        # Stop message queue processor
        if agent2:
            await agent2.stop_message_queue_processor()
        
        # Deregister from blockchain
        if agent2 and hasattr(agent2, 'blockchain_adapter'):
            await agent2.blockchain_adapter.deregister_agent()
            logger.info("✅ Agent 2 deregistered from A2A Network blockchain")
    except Exception as e:
        logger.error(f"Failed to deregister: {e}")

@app.get("/")
async def get_agent_card():
    """Get agent card"""
    if agent2:
        card = agent2.get_agent_card()
        card["url"] = f"http://localhost:{AGENT2_PORT}"
        return card
    else:
        return {
            "agent_id": AGENT2_CONFIG["agent_id"],
            "name": AGENT2_CONFIG["name"],
            "description": AGENT2_CONFIG["description"],
            "version": AGENT2_CONFIG["version"],
            "url": f"http://localhost:{AGENT2_PORT}",
            "capabilities": AGENT2_CONFIG["capabilities"]
        }

@app.get("/.well-known/agent.json")
async def well_known_agent():
    """Well-known agent card endpoint"""
    if agent2:
        card = agent2.get_agent_card()
        card["url"] = f"http://localhost:{AGENT2_PORT}"
        return card
    else:
        return {
            "agent_id": AGENT2_CONFIG["agent_id"],
            "name": AGENT2_CONFIG["name"],
            "description": AGENT2_CONFIG["description"],
            "version": AGENT2_CONFIG["version"],
            "url": f"http://localhost:{AGENT2_PORT}",
            "capabilities": AGENT2_CONFIG["capabilities"]
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "AI Preparation Agent",
        "version": "1.0.0",
        "protocol_version": "0.2.9",
        "port": AGENT2_PORT
    }

@app.post("/a2a/v1/messages")
async def process_message(request: Request):
    """Process A2A messages"""
    try:
        body = await request.json()
        
        # Handle both signed and unsigned messages
        if "signature" in body:
            # This is a signed message from another agent
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
            
        if agent2:
            result = await agent2.process_message(message, context_id)
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
    except Exception as e:
        logger.error(f"Message processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/a2a/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    if agent2:
        task = agent2.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    else:
        raise HTTPException(status_code=503, detail="Agent not initialized")

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
            if agent2:
                message = A2AMessage(**params.get("message", {}))
                context_id = params.get("contextId", str(datetime.utcnow().timestamp()))
                result = await agent2.process_message(message, context_id)
                
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": request_id
                    }
                )
            else:
                raise HTTPException(status_code=503, detail="Agent not initialized")
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
    return AGENT2_CONFIG

if __name__ == "__main__":
    logger.info(f"🚀 Launching Agent 2 on port {AGENT2_PORT}")
    uvicorn.run(
        "launch_agent2:app",
        host=AGENT2_HOST,
        port=AGENT2_PORT,
        reload=True,
        log_level="info"
    )
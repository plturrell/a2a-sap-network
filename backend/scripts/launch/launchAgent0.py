#!/usr/bin/env python3
"""
Launch Agent 0 (Data Product Registration Agent) as an independent service
True A2A microservice deployment
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.a2a.agents.agent0_router import router as agent0_router
from a2a_network.python_sdk.blockchain import get_blockchain_client, initialize_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter
from app.a2a.agents.dataProductAgent import DataProductRegistrationAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for Agent 0
app = FastAPI(
    title="Agent 0 - Data Product Registration Agent",
    description="Independent A2A microservice for data product registration",
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
AGENT0_PORT = int(os.getenv("AGENT0_PORT", "8002"))
AGENT0_HOST = os.getenv("AGENT0_HOST", "0.0.0.0")
ORD_REGISTRY_URL = os.getenv("ORD_REGISTRY_URL", "http://localhost:8000/api/v1/ord")

# A2A Network blockchain configuration
A2A_RPC_URL = os.getenv("A2A_RPC_URL", "os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))")
A2A_AGENT_PRIVATE_KEY = os.getenv("A2A_AGENT_PRIVATE_KEY")

# Agent 0 configuration for blockchain
AGENT0_CONFIG = {
    "agent_id": "data_product_agent_0",
    "name": "Data Product Registration Agent",
    "description": "Processes raw data into CDS schema with ORD descriptors enhanced by Dublin Core metadata",
    "version": "1.1.0",
    "base_url": f"http://localhost:{AGENT0_PORT}",
    "ord_registry_url": ORD_REGISTRY_URL,
    "capabilities": [
        "dublin_core_extraction",
        "cds_csn_generation", 
        "ord_descriptor_creation",
        "catalog_registration",
        "metadata_extraction",
        "data_processing"
    ],
    "downstream_agent_url": None,  # Will be discovered via blockchain
}

# Don't include router yet - we need to initialize agent0 first

@app.on_event("startup")
async def startup_event():
    """Register Agent 0 with the A2A Network blockchain on startup"""
    try:
        logger.info(f"🚀 Starting Agent 0 on port {AGENT0_PORT}")
        
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
            agent_id=AGENT0_CONFIG["agent_id"],
            name=AGENT0_CONFIG["name"],
            description=AGENT0_CONFIG["description"],
            version=AGENT0_CONFIG["version"],
            endpoint=AGENT0_CONFIG["base_url"],
            capabilities=AGENT0_CONFIG["capabilities"],
            skills=[
                {
                    "id": "dublin-core-extraction",
                    "name": "Dublin Core Metadata Extraction",
                    "description": "Extract and generate Dublin Core metadata from raw data",
                    "tags": ["dublin-core", "metadata", "standards"]
                },
                {
                    "id": "cds-csn-generation", 
                    "name": "CDS CSN Generation",
                    "description": "Generate Core Data Services CSN from raw financial data",
                    "tags": ["cds", "csn", "schema"]
                },
                {
                    "id": "ord-descriptor-creation",
                    "name": "ORD Descriptor Creation", 
                    "description": "Generate Object Resource Discovery descriptors",
                    "tags": ["ord", "discovery", "metadata"]
                }
            ]
        )
        
        # Register with A2A Network blockchain
        registration_success = await blockchain_adapter.register_agent()
        if registration_success:
            logger.info(f"✅ Agent 0 registered on A2A Network blockchain")
            logger.info(f"   Address: {blockchain_adapter.get_agent_address()}")
            logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        else:
            logger.warning("⚠️ Failed to register with A2A Network - continuing anyway")
        
        # Discover downstream agent (Agent 1) via blockchain
        try:
            discovered_agents = await blockchain_adapter.discover_agents("financial_standardization")
            
            if discovered_agents:
                agent1 = discovered_agents[0]
                AGENT0_CONFIG["downstream_agent_url"] = agent1["endpoint"]
                logger.info(f"✅ Discovered downstream agent: {agent1['name']} at {agent1['endpoint']}")
                logger.info(f"   Blockchain address: {agent1['address']}")
                logger.info(f"   Reputation: {agent1['reputation']}")
            else:
                logger.warning("⚠️ No downstream agent found on blockchain")
                
        except Exception as e:
            logger.error(f"Failed to discover downstream agent via blockchain: {e}")
        
        # Initialize Agent 0 with blockchain integration
        from app.a2a.agents.dataProductAgent import DataProductRegistrationAgent
        import app.a2a.agents.agent0_router as agent0_module
        
        agent0_module.agent0 = DataProductRegistrationAgent(
            base_url=f"http://localhost:{AGENT0_PORT}/a2a/agent0/v1",
            ord_registry_url=ORD_REGISTRY_URL,
            downstream_agent_url=AGENT0_CONFIG.get("downstream_agent_url")
        )
        
        # Store blockchain adapter for use in agent
        agent0_module.agent0.blockchain_adapter = blockchain_adapter
        
        # Start the message queue processor
        await agent0_module.agent0.start_message_queue_processor()
        
        # Now include the router
        app.include_router(agent0_router)
        logger.info("✅ Agent 0 initialized with blockchain integration and router included")
        
        # Log agent status
        status = blockchain_adapter.get_agent_status()
        logger.info(f"📊 Agent Status: {status}")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue running even if registration fails

@app.on_event("shutdown")
async def shutdown_event():
    """Deregister Agent 0 from the A2A Network blockchain and stop message queue on shutdown"""
    try:
        # Stop message queue processor
        import app.a2a.agents.agent0_router as agent0_module
        if agent0_module.agent0:
            await agent0_module.agent0.stop_message_queue_processor()
        
        # Deregister from blockchain
        if hasattr(agent0_module.agent0, 'blockchain_adapter'):
            await agent0_module.agent0.blockchain_adapter.deregister_agent()
            logger.info("✅ Agent 0 deregistered from A2A Network blockchain")
    except Exception as e:
        logger.error(f"Failed to deregister: {e}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "Data Product Registration Agent",
        "version": "1.1.0",
        "port": AGENT0_PORT
    }

@app.get("/config")
async def get_config():
    """Get current agent configuration"""
    return AGENT0_CONFIG

if __name__ == "__main__":
    logger.info(f"🚀 Launching Agent 0 on port {AGENT0_PORT}")
    uvicorn.run(
        "launch_agent0:app",
        host=AGENT0_HOST,
        port=AGENT0_PORT,
        reload=True,
        log_level="info"
    )
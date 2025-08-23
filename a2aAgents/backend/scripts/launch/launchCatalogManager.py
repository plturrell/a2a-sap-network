#!/usr/bin/env python3
"""
Launch Catalog Manager Agent as an independent A2A service
AI-powered ORD repository management microservice
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.a2a.agents.catalog_manager_router import router as catalog_manager_router
from a2a_network.python_sdk.blockchain import get_blockchain_client, initialize_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter
from a2a_network.python_sdk.blockchain.ord_blockchain_adapter import create_ord_blockchain_adapter
from app.a2a.agents.catalog_manager_agent import create_catalog_manager_agent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app for Catalog Manager
app = FastAPI(
    title="Catalog Manager Agent",
    description="AI-powered ORD repository management and enhancement microservice",
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
CATALOG_MANAGER_PORT = int(os.getenv("CATALOG_MANAGER_PORT", "8005"))
CATALOG_MANAGER_HOST = os.getenv("CATALOG_MANAGER_HOST", "0.0.0.0")
ORD_REGISTRY_URL = os.getenv("ORD_REGISTRY_URL", "http://localhost:8000/api/v1/ord")

# A2A Network blockchain configuration
A2A_RPC_URL = os.getenv("A2A_RPC_URL", "os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))")
A2A_AGENT_PRIVATE_KEY = os.getenv("A2A_CATALOG_MANAGER_PRIVATE_KEY")

# Catalog Manager configuration
CATALOG_MANAGER_CONFIG = {
    "agent_id": "catalog_manager_agent",
    "name": "Catalog Manager Agent",
    "description": "AI-powered ORD repository management with advanced metadata enhancement",
    "version": "1.0.0",
    "base_url": f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}",
    "capabilities": [
        "ord_repository_management",
        "ai_metadata_enhancement", 
        "semantic_search",
        "quality_assessment",
        "compliance_validation",
        "dublin_core_enrichment",
        "multi_model_ai"
    ],
    "endpoints": {
        "agent_card": f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}/a2a/catalog_manager/v1/.well-known/agent.json",
        "rpc": f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}/a2a/catalog_manager/v1/rpc",
        "message": f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}/a2a/catalog_manager/v1/message",
        "health": f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}/a2a/catalog_manager/v1/health"
    }
}


@app.on_event("startup")
async def startup_event():
    """Register Catalog Manager with the A2A Registry on startup"""
    try:
        logger.info("🚀 Starting Catalog Manager Agent...")
        
        # Initialize blockchain client
        logger.info("🔗 Connecting to A2A Network blockchain...")
        initialize_blockchain_client(
            rpc_url=A2A_RPC_URL,
            private_key=A2A_AGENT_PRIVATE_KEY
        )
        blockchain_client = get_blockchain_client()
        logger.info(f"✅ Connected to A2A Network: {blockchain_client.agent_identity.address}")
        
        # Initialize Catalog Manager agent
        base_url = f"https://{CATALOG_MANAGER_HOST}:{CATALOG_MANAGER_PORT}"
        catalog_manager = create_catalog_manager_agent(
            base_url=base_url,
            ord_registry_url=ORD_REGISTRY_URL
        )
        
        # Initialize traditional ORD service
        await catalog_manager.initialize()
        
        # Create ORD blockchain adapter that bridges traditional and blockchain
        ord_blockchain_adapter = create_ord_blockchain_adapter(
            traditional_ord_service=catalog_manager.ord_service
        )
        
        # Store ORD blockchain adapter in catalog manager
        catalog_manager.ord_blockchain_adapter = ord_blockchain_adapter
        logger.info("✅ ORD blockchain adapter integrated with Catalog Manager")
        
        # Set the global agent instance in the router
        import app.a2a.agents.catalog_manager_router as router_module
        router_module.catalog_manager = catalog_manager
        
        logger.info("✅ Catalog Manager Agent initialized")
        
        # Create blockchain adapter for agent registration
        blockchain_adapter = create_blockchain_adapter(
            agent_id=CATALOG_MANAGER_CONFIG["agent_id"],
            name=CATALOG_MANAGER_CONFIG["name"],
            description=CATALOG_MANAGER_CONFIG["description"],
            version="1.0.0",
            endpoint=CATALOG_MANAGER_CONFIG["base_url"],
            capabilities=CATALOG_MANAGER_CONFIG["capabilities"],
            skills=[
                {"id": "ord-management", "name": "ORD Repository Management"},
                {"id": "ai-enhancement", "name": "AI-powered Metadata Enhancement"},
                {"id": "dublin-core", "name": "Dublin Core Metadata Processing"}
            ]
        )
        
        # Register with A2A Network blockchain
        registration_success = await blockchain_adapter.register_agent()
        if registration_success:
            logger.info(f"✅ Catalog Manager registered on A2A Network blockchain")
            logger.info(f"   Address: {blockchain_adapter.get_agent_address()}")
            logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        else:
            logger.warning("⚠️ Failed to register with A2A Network - continuing anyway")
        
        # Store blockchain adapter
        catalog_manager.blockchain_adapter = blockchain_adapter
        
        logger.info(f"🎯 Catalog Manager Agent ready on port {CATALOG_MANAGER_PORT}")
        logger.info(f"📋 Agent Card: {CATALOG_MANAGER_CONFIG['endpoints']['agent_card']}")
        logger.info(f"🔗 RPC Endpoint: {CATALOG_MANAGER_CONFIG['endpoints']['rpc']}")
        logger.info(f"💬 Message Endpoint: {CATALOG_MANAGER_CONFIG['endpoints']['message']}")
        logger.info(f"❤️ Health Check: {CATALOG_MANAGER_CONFIG['endpoints']['health']}")
        
        # Log ORD-specific endpoints
        logger.info("📚 ORD Repository Endpoints:")
        logger.info(f"   📝 Register: {base_url}/a2a/catalog_manager/v1/ord/register")
        logger.info(f"   ✨ Enhance: {base_url}/a2a/catalog_manager/v1/ord/enhance/{{registration_id}}")
        logger.info(f"   🔍 Search: {base_url}/a2a/catalog_manager/v1/ord/search")
        logger.info(f"   📊 Quality: {base_url}/a2a/catalog_manager/v1/ord/quality/{{registration_id}}")
        
    except Exception as e:
        logger.error(f"Failed to start Catalog Manager Agent: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Deregister Catalog Manager from the A2A Registry on shutdown"""
    try:
        logger.info("🛑 Shutting down Catalog Manager Agent...")
        
        # Deregister from A2A Network blockchain
        try:
            import app.a2a.agents.catalog_manager_router as router_module
            if router_module.catalog_manager and hasattr(router_module.catalog_manager, 'blockchain_adapter'):
                await router_module.catalog_manager.blockchain_adapter.deregister_agent()
                logger.info("✅ Catalog Manager deregistered from A2A Network blockchain")
        except Exception as e:
            logger.warning(f"Failed to deregister from A2A Network: {e}")
        
        logger.info("✅ Catalog Manager Agent shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "catalog_manager",
        "version": CATALOG_MANAGER_CONFIG["version"],
        "capabilities": CATALOG_MANAGER_CONFIG["capabilities"]
    }


@app.get("/config")
async def get_config():
    """Get current agent configuration"""
    return CATALOG_MANAGER_CONFIG


# Include the Catalog Manager router
app.include_router(catalog_manager_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with agent information"""
    return {
        "agent": "Catalog Manager",
        "description": "AI-powered ORD repository management and enhancement",
        "version": CATALOG_MANAGER_CONFIG["version"],
        "endpoints": CATALOG_MANAGER_CONFIG["endpoints"],
        "capabilities": CATALOG_MANAGER_CONFIG["capabilities"]
    }


if __name__ == "__main__":
    logger.info(f"🚀 Launching Catalog Manager Agent on port {CATALOG_MANAGER_PORT}")
    uvicorn.run(
        "launch_catalog_manager:app",
        host=CATALOG_MANAGER_HOST,
        port=CATALOG_MANAGER_PORT,
        reload=False,
        log_level="info"
    )

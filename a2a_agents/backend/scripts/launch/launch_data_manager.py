#!/usr/bin/env python3
"""
Launch Data Manager Agent on port 8003
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.a2a.agents.data_manager_agent import DataManagerAgent, create_data_manager_router
from a2a_network.python_sdk.blockchain import get_blockchain_client, initialize_blockchain_client
from a2a_network.python_sdk.blockchain.agent_adapter import create_blockchain_adapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Manager Agent",
    description="A2A Data Manager Agent - Manages data storage operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Data Manager Agent"}

# Create and initialize agent
data_manager_agent = None
registry_client = None

@app.on_event("startup")
async def startup_event():
    global data_manager_agent
    
    logger.info("🚀 Starting Data Manager Agent on port 8003")
    
    # Initialize blockchain client
    A2A_RPC_URL = os.getenv("A2A_RPC_URL", "http://localhost:8545")
    A2A_AGENT_PRIVATE_KEY = os.getenv("A2A_DATA_MANAGER_PRIVATE_KEY")
    
    logger.info("🔗 Connecting to A2A Network blockchain...")
    initialize_blockchain_client(
        rpc_url=A2A_RPC_URL,
        private_key=A2A_AGENT_PRIVATE_KEY
    )
    blockchain_client = get_blockchain_client()
    logger.info(f"✅ Connected to A2A Network: {blockchain_client.agent_identity.address}")
    
    # Initialize agent
    data_manager_agent = DataManagerAgent(
        base_url="http://localhost:8003",
        ord_registry_url="http://localhost:8000/api/v1/ord"
    )
    
    # Create blockchain adapter
    blockchain_adapter = create_blockchain_adapter(
        agent_id="data_manager_agent",
        name="Data Manager Agent",
        description="Manages data storage, archival, and retrieval operations",
        version="1.0.0",
        endpoint="http://localhost:8003",
        capabilities=[
            "data_storage",
            "data_archival", 
            "data_retrieval",
            "data_lifecycle",
            "database_operations"
        ],
        skills=[
            {"id": "data-storage", "name": "Data Storage Management"},
            {"id": "data-archival", "name": "Data Archival"},
            {"id": "data-retrieval", "name": "Data Retrieval"},
            {"id": "data-lifecycle", "name": "Data Lifecycle Management"}
        ]
    )
    
    # Register with A2A Network blockchain
    registration_success = await blockchain_adapter.register_agent()
    if registration_success:
        logger.info(f"✅ Data Manager registered on A2A Network blockchain")
        logger.info(f"   Address: {blockchain_adapter.get_agent_address()}")
        logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
    else:
        logger.warning("⚠️ Failed to register with A2A Network - continuing anyway")
    
    # Store blockchain adapter for use in agent
    data_manager_agent.blockchain_adapter = blockchain_adapter
    
    # Include agent router
    agent_router = create_data_manager_router(data_manager_agent)
    app.include_router(agent_router)
    
    logger.info("✅ Data Manager Agent initialized with blockchain integration")

@app.on_event("shutdown")
async def shutdown_event():
    global data_manager_agent
    
    if data_manager_agent and hasattr(data_manager_agent, 'blockchain_adapter'):
        try:
            await data_manager_agent.blockchain_adapter.deregister_agent()
            logger.info("✅ Data Manager deregistered from A2A Network blockchain")
        except Exception as e:
            logger.warning(f"⚠️ Failed to deregister: {e}")

if __name__ == "__main__":
    logger.info("🚀 Launching Data Manager Agent on port 8003")
    uvicorn.run(
        "launch_data_manager:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        reload_dirs=["app"]
    )
#!/usr/bin/env python3
"""
Launch Agent 3: SAP HANA Vector Engine Ingestion & Knowledge Graph Agent
Handles vector ingestion into SAP HANA Cloud Vector Engine with semantic search capabilities
"""

import asyncio
import os
import sys
import logging
from typing import Optional
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.a2a.agents.vector_processing_agent import VectorProcessingAgent
from app.a2a.agents import agent3_router
from a2a_network.python_sdk.blockchain.enhanced_agent_discovery import EnhancedBlockchainAgentRegistry
from src.a2a.core.message_queue import initialize_message_queue
from app.a2a.security.smart_contract_trust import initialize_agent_trust
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AGENT3_PORT = int(os.getenv("AGENT3_PORT", "8004"))
AGENT3_HOST = os.getenv("AGENT3_HOST", "0.0.0.0")
AGENT2_URL = os.getenv("AGENT2_URL", "http://localhost:8003")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("Starting Agent 3: SAP HANA Vector Engine Ingestion & Knowledge Graph Agent")
    
    try:
        # Initialize Agent 3
        base_url = f"http://{AGENT3_HOST}:{AGENT3_PORT}"
        agent3 = VectorProcessingAgent(
            base_url=base_url,
            agent_id="vector_processing_agent_3"
        )
        
        # Set global agent instance for router
        agent3_router.agent3 = agent3
        
        # Initialize enhanced blockchain registry for service discovery
        blockchain_registry = EnhancedBlockchainAgentRegistry(
            network_config={
                "type": "development",
                "consensus": "proof_of_authority",
                "validators": ["agent_manager"],
                "block_time": 5,
                "gas_limit": 8000000
            }
        )
        
        try:
            # Register agent with blockchain discovery (enhanced trust model)
            agent_card = await agent3.get_agent_card()
            agent_registration = {
                "agent_id": "vector_processing_agent_3",
                "agent_card": agent_card,
                "service_endpoint": f"http://{AGENT3_HOST}:{AGENT3_PORT}",
                "capabilities": {
                    "vector_database_ingestion": True,
                    "knowledge_graph_construction": True, 
                    "semantic_search_enablement": True,
                    "langchain_integration": True,
                    "sap_hana_cloud_integration": True
                },
                "trust_level": "verified",
                "performance_tier": "high",
                "specialization": "vector_processing"
            }
            
            registration_result = await blockchain_registry.register_agent(agent_registration)
            logger.info(f"Agent 3 registered with blockchain: transaction_hash={registration_result.get('transaction_hash')}")
            
        except Exception as e:
            logger.warning(f"Blockchain registration failed, continuing without it: {e}")
        
        # Initialize trust system for A2A message verification
        try:
            trust_identity = await initialize_agent_trust(
                agent_id="vector_processing_agent_3",
                agent_type="vector_processing",
                capabilities=[
                    "vector_database_ingestion",
                    "knowledge_graph_construction",
                    "semantic_search_enablement", 
                    "langchain_integration"
                ]
            )
            logger.info(f"Trust system initialized for Agent 3: {trust_identity}")
        except Exception as e:
            logger.warning(f"Trust system initialization failed: {e}")
        
        # Initialize message queue with enhanced streaming support
        try:
            message_queue = initialize_message_queue(
                agent_id="vector_processing_agent_3",
                config={
                    "max_queue_size": 500,
                    "max_processing_time_seconds": 1800,  # 30 minutes for vector processing
                    "enable_streaming": True,
                    "enable_batch_processing": True,
                    "priority_processing": True,
                    "vector_processing_optimized": True
                }
            )
            logger.info("Message queue initialized with vector processing optimizations")
        except Exception as e:
            logger.error(f"Message queue initialization failed: {e}")
            raise
        
        # Store references for cleanup
        app.state.agent3 = agent3
        app.state.blockchain_registry = blockchain_registry
        
        logger.info(f"Agent 3 successfully started on {base_url}")
        logger.info("Agent 3 is ready to process AI-ready entities from Agent 2")
        logger.info("Capabilities: SAP HANA Cloud Vector Ingestion, Knowledge Graph Construction, Semantic Search")
        
    except Exception as e:
        logger.error(f"Failed to start Agent 3: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent 3...")
    
    try:
        # Deregister from blockchain
        if hasattr(app.state, 'blockchain_registry'):
            await app.state.blockchain_registry.deregister_agent("vector_processing_agent_3")
            logger.info("Agent 3 deregistered from blockchain")
    except Exception as e:
        logger.warning(f"Blockchain deregistration failed: {e}")
    
    logger.info("Agent 3 shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Agent 3: SAP HANA Vector Engine Ingestion & Knowledge Graph Agent",
        description="A2A Protocol v0.2.9 compliant agent for vector database ingestion and knowledge graph construction",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Include agent router
    app.include_router(agent3_router.router)
    
    # Add health check
    @app.get("/")
    async def root():
        return {
            "agent": "Agent 3: SAP HANA Vector Engine Ingestion & Knowledge Graph Agent",
            "version": "1.0.0",
            "protocol": "A2A v0.2.9",
            "status": "running",
            "capabilities": [
                "vector_database_ingestion",
                "knowledge_graph_construction",
                "semantic_search_enablement",
                "langchain_integration",
                "sap_hana_cloud_integration"
            ],
            "upstream_agent": "Agent 2 (AI Data Readiness & Vectorization)",
            "port": AGENT3_PORT
        }
    
    return app


async def main():
    """Main entry point"""
    
    logger.info("=" * 80)
    logger.info("AGENT 3: SAP HANA VECTOR ENGINE INGESTION & KNOWLEDGE GRAPH AGENT")
    logger.info("=" * 80)
    logger.info(f"Port: {AGENT3_PORT}")
    logger.info(f"Host: {AGENT3_HOST}")
    logger.info(f"Upstream Agent 2 URL: {AGENT2_URL}")
    logger.info("A2A Protocol: v0.2.9")
    logger.info("Capabilities:")
    logger.info("  - SAP HANA Cloud Vector Engine Ingestion")
    logger.info("  - Knowledge Graph Construction with SPARQL")
    logger.info("  - Semantic Search Index Creation")
    logger.info("  - LangChain Integration for RAG Applications")
    logger.info("  - Data Lineage Preservation")
    logger.info("=" * 80)
    
    # Check SAP HANA Cloud availability
    try:
        from langchain_hana import HanaDB, HanaInternalEmbeddings
        from hdbcli import dbapi
        logger.info("✓ SAP HANA Cloud integration available")
    except ImportError as e:
        logger.warning("⚠ SAP HANA Cloud integration not available:")
        logger.warning(f"  {e}")
        logger.warning("  Install with: pip install langchain-hana hdbcli")
        logger.warning("  Agent will continue but vector ingestion will fail")
    
    # Check environment variables
    hana_config_vars = ["HANA_HOSTNAME", "HANA_PORT", "HANA_USERNAME", "HANA_PASSWORD"]
    missing_vars = [var for var in hana_config_vars if not os.getenv(var) and not os.getenv(var.replace("HANA_HOSTNAME", "HANA_HOST").replace("HANA_USERNAME", "HANA_USER"))]
    
    if missing_vars:
        logger.warning("⚠ Missing SAP HANA Cloud configuration:")
        for var in missing_vars:
            logger.warning(f"  - {var}")
        logger.warning("  Vector ingestion may fail without proper HANA configuration")
    else:
        logger.info("✓ SAP HANA Cloud configuration detected")
    
    # Create and run the application
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=AGENT3_HOST,
        port=AGENT3_PORT,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False
    )
    
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Agent 3 stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent 3 interrupted by user")
    except Exception as e:
        logger.error(f"Agent 3 failed to start: {e}")
        sys.exit(1)
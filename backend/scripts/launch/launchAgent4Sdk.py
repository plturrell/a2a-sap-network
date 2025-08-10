#!/usr/bin/env python3
"""
Launch Agent 4: Computation Quality Testing Agent (SDK Version)
Dynamic computation validation using template-based test generation
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.a2a.agents.agent4_calc_validation.active.agent4_router import initialize_agent, router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/agent4.log')
    ]
)

logger = logging.getLogger(__name__)

# Configuration
AGENT_HOST = os.getenv("AGENT4_HOST", "0.0.0.0")
AGENT_PORT = int(os.getenv("AGENT4_PORT", "8006"))
BASE_URL = os.getenv("AGENT4_BASE_URL", f"http://{AGENT_HOST}:{AGENT_PORT}")
TEMPLATE_REPOSITORY_URL = os.getenv("TEMPLATE_REPOSITORY_URL")

# A2A Agent Integration URLs
DATA_MANAGER_URL = os.getenv("DATA_MANAGER_URL", "http://localhost:8001")
CATALOG_MANAGER_URL = os.getenv("CATALOG_MANAGER_URL", "http://localhost:8002")

# Global agent reference
agent_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global agent_instance
    
    logger.info("🚀 Starting Agent 4: Computation Quality Testing Agent (SDK Version)")
    
    try:
        # Initialize the agent with A2A integration
        agent_instance = await initialize_agent(
            base_url=BASE_URL,
            template_repository_url=TEMPLATE_REPOSITORY_URL,
            data_manager_url=DATA_MANAGER_URL,
            catalog_manager_url=CATALOG_MANAGER_URL
        )
        
        logger.info(f"✅ Agent 4 initialized successfully")
        logger.info(f"   Agent ID: {agent_instance.agent_id}")
        logger.info(f"   Base URL: {BASE_URL}")
        logger.info(f"   Templates loaded: {len(agent_instance.test_templates)}")
        logger.info(f"   Trust system: {'✅ Enabled' if agent_instance.trust_identity else '⚠️  Disabled'}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Agent 4: {e}")
        raise
    finally:
        # Cleanup
        if agent_instance:
            try:
                await agent_instance.cleanup()
                logger.info("🧹 Agent 4 cleanup completed")
            except Exception as e:
                logger.error(f"❌ Cleanup failed: {e}")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Agent 4: Computation Quality Testing Agent",
        description="""
        A2A v0.2.9 compliant agent for dynamic computation quality testing using template-based test generation.
        
        ## Features
        - 🔍 **Service Discovery**: Automatically discover computational services
        - 🧪 **Template-Based Testing**: Generate tests from configurable templates
        - ⚡ **Parallel Execution**: Execute multiple tests concurrently
        - 📊 **Quality Analysis**: Comprehensive quality metrics and scoring
        - 🛡️ **Circuit Breakers**: Fault tolerance for service calls
        - 🔐 **Trust System**: RSA-based agent authentication
        - 📈 **Metrics**: Prometheus-compatible performance metrics
        
        ## Supported Computation Types
        - **Mathematical**: Arithmetic, algebraic, statistical operations
        - **Logical**: Boolean operations, truth tables, conditional logic
        - **Transformational**: Data processing, format conversion
        - **Performance**: Latency, throughput, resource utilization
        
        ## Test Methodologies
        - **Accuracy**: Numerical precision and correctness validation
        - **Performance**: Execution time and resource usage testing
        - **Stress**: Edge cases and boundary condition testing  
        - **Comprehensive**: Combined testing across all dimensions
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Include the agent router
    app.include_router(router)
    
    # Health check endpoint
    @app.get("/health")
    async def health():
        """Application health check"""
        return {
            "status": "healthy", 
            "agent": "Computation Quality Testing Agent",
            "version": "1.0.0",
            "sdk_version": "3.0.0"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with agent information"""
        return {
            "agent": "Computation Quality Testing Agent",
            "agent_id": "calc_validation_agent_4",
            "version": "1.0.0",
            "protocol_version": "0.2.9",
            "description": "A2A compliant agent for dynamic computation quality testing",
            "endpoints": {
                "agent_card": "/agent4/",
                "health": "/agent4/health",
                "a2a_tasks": "/agent4/a2a/tasks",
                "testing": "/agent4/testing/execute",
                "discovery": "/agent4/discovery/services",
                "services": "/agent4/services",
                "templates": "/agent4/templates",
                "metrics": "/agent4/metrics",
                "docs": "/docs"
            },
            "capabilities": [
                "dynamic_computation_testing",
                "service_discovery",
                "template_based_testing",
                "quality_metrics",
                "streaming",
                "trust_system",
                "circuit_breakers"
            ]
        }
    
    return app


async def main():
    """Main entry point"""
    logger.info("🎯 Agent 4: Computation Quality Testing Agent")
    logger.info(f"📍 Starting server at {BASE_URL}")
    
    app = create_app()
    
    config = uvicorn.Config(
        app,
        host=AGENT_HOST,
        port=AGENT_PORT,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
    
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Agent 4...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    # Environment setup
    os.environ.setdefault("PYTHONPATH", str(backend_dir))
    
    # Log startup configuration
    logger.info("🔧 Agent 4 Configuration:")
    logger.info(f"   Host: {AGENT_HOST}")
    logger.info(f"   Port: {AGENT_PORT}")
    logger.info(f"   Base URL: {BASE_URL}")
    logger.info(f"   Template Repository: {TEMPLATE_REPOSITORY_URL or 'Built-in only'}")
    logger.info(f"   Log file: /tmp/agent4.log")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Agent 4 shutdown complete")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
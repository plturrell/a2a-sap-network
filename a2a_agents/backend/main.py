from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from typing import Dict, Any

from app.core.config import settings
from app.core.sap_cloud_sdk import get_sap_cloud_sdk, SAPLogHandler
from app.api.router import api_router
from app.a2a.core.router import router as a2a_router
from app.ord_registry.router import router as ord_router
from app.a2a.agents.agent0_data_product.active.agent0_router import router as agent0_router
from app.a2a.agents.agent1_standardization.active.agent1_router import router as agent1_router
from app.a2a.agents.agent2_ai_preparation.active.agent2_router import router as agent2_router
from app.a2a.agents.agent3_vector_processing.active.agent3_router import router as agent3_router
from app.a2a.agents.agent_manager.active.agent_manager_router import router as agent_manager_router
from app.a2a.agents.catalog_manager.active.catalog_manager_router import router as catalog_manager_router
from app.a2a_registry.router import router as a2a_registry_router
from app.a2a_trustsystem.router import router as a2a_trustsystem_router
from app.a2a.core.workflow_router import router as workflow_router
from app.api.middleware.rate_limiting import rate_limit_middleware
from app.a2a.core.telemetry import init_telemetry, instrument_fastapi, instrument_httpx, instrument_redis
from app.a2a.config.telemetry_config import telemetry_config

# Configure logging with SAP integration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add SAP Application Logging handler if available
if os.getenv("SAP_ALS_CLIENT_ID"):
    sap_handler = SAPLogHandler(component="a2a-main")
    sap_handler.setLevel(logging.WARNING)  # Only send warnings and errors to SAP
    logging.getLogger().addHandler(sap_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize SAP Cloud SDK
    sap_sdk = get_sap_cloud_sdk()
    await sap_sdk.send_alert(
        subject=f"{settings.APP_NAME} Starting",
        body=f"Application {settings.APP_NAME} v{settings.APP_VERSION} is starting up",
        severity="INFO",
        category="NOTIFICATION"
    )
    
    # Initialize OpenTelemetry if enabled
    if telemetry_config.otel_enabled:
        service_name = os.getenv("OTEL_SERVICE_NAME", f"{settings.APP_NAME}-main")
        init_telemetry(
            service_name=service_name,
            agent_id="main-app",
            sampling_rate=telemetry_config.otel_traces_sampler_arg
        )
        instrument_fastapi(app)
        instrument_httpx()
        instrument_redis()
        logger.info(f"OpenTelemetry enabled for {service_name}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await sap_sdk.send_alert(
        subject=f"{settings.APP_NAME} Shutting Down",
        body=f"Application {settings.APP_NAME} v{settings.APP_VERSION} is shutting down",
        severity="INFO",
        category="NOTIFICATION"
    )
    await sap_sdk.close()


# Enhanced OpenAPI metadata for SAP compliance
openapi_tags = [
    {
        "name": "Authentication",
        "description": "Operations related to user authentication and authorization"
    },
    {
        "name": "Users",
        "description": "User management operations"
    },
    {
        "name": "Data",
        "description": "Data operations and transformations"
    },
    {
        "name": "A2A Core",
        "description": "Agent-to-Agent communication core functionality"
    },
    {
        "name": "ORD Registry",
        "description": "SAP Object Resource Discovery registry operations"
    },
    {
        "name": "Agents",
        "description": "Individual agent endpoints for the A2A network"
    },
    {
        "name": "Workflows",
        "description": "Workflow orchestration and management"
    },
    {
        "name": "Trust System",
        "description": "A2A trust and security management"
    },
    {
        "name": "Health",
        "description": "Health check and monitoring endpoints"
    }
]

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## FinSight CIB - Enterprise A2A Agent Platform
    
    This API provides comprehensive agent-to-agent (A2A) communication and orchestration 
    capabilities for financial data processing, built on SAP BTP with enterprise-grade 
    security and scalability.
    
    ### Key Features:
    - 🤖 **Agent-to-Agent Communication**: Secure, scalable inter-agent messaging
    - 📊 **Financial Data Processing**: Specialized agents for financial standardization
    - 🔐 **Enterprise Security**: SAP BTP integration with OAuth2 and JWT
    - 📈 **Real-time Monitoring**: OpenTelemetry and Prometheus metrics
    - 🏗️ **Microservices Architecture**: Distributed, resilient agent network
    - 🗄️ **Dual Database Strategy**: SAP HANA primary with PostgreSQL fallback
    
    ### Architecture:
    - **Agent 0**: Data Product Registration
    - **Agent 1**: Financial Data Standardization  
    - **Agent 2**: AI Data Preparation
    - **Agent 3**: Vector Processing and Embeddings
    - **Agent 4**: Calculation and Validation
    - **Agent 5**: Quality Assurance and Validation
    - **Agent Manager**: Orchestration and Coordination
    - **Agent Builder**: Dynamic Agent Creation and Configuration
    - **Data Manager**: Data Pipeline and Storage Management
    - **Catalog Manager**: Data Catalog and Metadata Management
    
    ### Authentication:
    All endpoints except health checks require authentication via:
    - Bearer token (JWT)
    - API Key (X-API-Key header)
    
    ### Rate Limiting:
    API calls are rate-limited based on authentication tier:
    - Anonymous: 10 requests/minute
    - Authenticated: 100 requests/minute
    - Premium: 1000 requests/minute
    """,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Resource not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add telemetry middleware
if telemetry_config.otel_enabled:
    from app.api.middleware.telemetry import TelemetryMiddleware
    app.add_middleware(TelemetryMiddleware)

# Add rate limiting middleware for A2A API protection
from fastapi.middleware.base import BaseHTTPMiddleware
class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        return await rate_limit_middleware(request, call_next)

app.add_middleware(RateLimitingMiddleware)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(a2a_router)  # A2A routes at /a2a/v1 (Agent 1)
app.include_router(ord_router)  # ORD Registry at /api/v1/ord
app.include_router(agent0_router)  # Agent 0 at /a2a/agent0/v1
app.include_router(agent1_router)  # Agent 1 at /a2a/agent1/v1
app.include_router(agent2_router)  # Agent 2 at /a2a/agent2/v1
app.include_router(agent3_router)  # Agent 3 at /a2a/agent3/v1
app.include_router(agent_manager_router)  # Agent Manager at /a2a/agent_manager/v1
app.include_router(catalog_manager_router)  # Catalog Manager at /a2a/catalog_manager/v1
app.include_router(a2a_registry_router)  # A2A Registry at /api/v1/a2a
app.include_router(a2a_trustsystem_router)  # A2A Trust System at /api/v1/a2a/trust
app.include_router(workflow_router)  # A2A Workflow Orchestration at /api/v1/a2a/workflows

# Health endpoint for Docker health checks
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": "2025-08-01T16:22:00Z"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": {
            "ord_registry": "/api/v1/ord/health",
            "a2a_registry": "/api/v1/a2a/health",
            "a2a_trust_system": "/api/v1/a2a/trust/health",
            "workflow_orchestration": "/api/v1/a2a/workflows/active"
        },
        "a2a_agents": {
            "agent0": "/a2a/agent0/v1/.well-known/agent.json",
            "agent1": "/a2a/v1/.well-known/agent.json",
            "agent_manager": "/a2a/agent_manager/v1/.well-known/agent.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
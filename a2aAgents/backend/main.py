import os
import sys
import warnings

# Suppress warnings about unrecognized blockchain networks from eth_utils
warnings.filterwarnings("ignore", message="Network 345 with name 'Yooldo Verse Mainnet'")
warnings.filterwarnings("ignore", message="Network 12611 with name 'Astar zkEVM'")

# Add project root to the Python path to resolve import issues
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from typing import Dict, Any

from app.core.config import settings
from app.core.dynamicConfig import get_config_manager, validate_production_config
from app.core.constants import AppConstants, StaticConstants
from app.core.sapCloudSdk import get_sap_cloud_sdk, SAPLogHandler
from app.core.loggingConfig import init_logging, get_logger, LogCategory
from app.api.router import api_router
from app.a2a.core.router import router as a2a_router
from app.ordRegistry.router import router as ord_router
from app.a2a.agents.agent0DataProduct.active.agent0Router import router as agent0_router
from app.a2a.agents.agent1Standardization.active.agent1Router import router as agent1_router
from app.a2a.agents.agent2AiPreparation.active.agent2Router import router as agent2_router
from app.a2a.agents.agent3VectorProcessing.active.agent3Router import router as agent3_router
from app.a2a.agents.agent4CalcValidation.active.agent4Router import router as agent4_router
from app.a2a.agents.agent5QaValidation.active.agent5Router import router as agent5_router
from app.a2a.agents.calculationAgent.active.calculationRouter import router as calculation_router
from app.a2a.agents.agentManager.active.agentManagerRouter import router as agent_manager_router
from app.a2a.agents.catalogManager.active.catalogManagerRouter import router as catalog_manager_router
from app.a2aRegistry.router import router as a2a_registry_router
from app.a2aTrustSystem.router import router as a2a_trustsystem_router
from app.a2a.core.workflowRouter import router as workflow_router
from app.core.rateLimiting import rate_limit_middleware
from app.core.errorHandling import global_exception_handler
from app.core.securityMonitoring import get_security_monitor
from app.api.middleware.logging import create_logging_middleware
from app.api.middleware.securityMiddleware import SecurityEventMiddleware
from app.a2a.core.telemetry import init_telemetry, instrument_fastapi, instrument_httpx, instrument_redis
from app.a2a.config.telemetryConfig import telemetry_config

# Initialize dynamic configuration and standardized logging
try:
    config_manager = get_config_manager()
    monitoring_config = config_manager.get_monitoring_config()
    
    # Initialize standardized A2A logging system
    init_logging(
        level=monitoring_config.log_level.upper(),
        format_type="structured" if monitoring_config.log_format == 'json' else "simple",
        console=True,
        file_logging=config_manager.is_production()
    )
    logger = get_logger(__name__, LogCategory.SYSTEM)
    
    # Validate production configuration if needed
    if config_manager.is_production():
        validate_production_config()
        logger.info("Production configuration validated successfully")
    
except Exception as e:
    # Fallback to basic A2A logging configuration
    init_logging(level="INFO", format_type="simple", console=True)
    logger = get_logger(__name__, LogCategory.SYSTEM)
    logger.error(f"Failed to initialize dynamic configuration: {e}")
    logger.warning("Using fallback configuration")
    config_manager = None

# Add SAP Application Logging handler if available
if os.getenv("SAP_ALS_CLIENT_ID"):
    sap_handler = SAPLogHandler(component="a2a-main")
    logger.info("SAP Application Logging Service integration enabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        # Log environment and configuration info
        if config_manager:
            logger.info(f"Environment: {config_manager.env.value}")
            logger.info(f"Configuration loaded: {type(config_manager).__name__}")
            
            # Log important configuration values (without secrets)
            timeouts = AppConstants.get_timeout_constants()
            limits = AppConstants.get_limits_constants()
            logger.info(f"Agent timeouts configured: {timeouts.get('AGENT_DEFAULT_TIMEOUT', 30)}s default")
            logger.info(f"Max concurrent requests: {limits.get('MAX_CONCURRENT_REQUESTS', 100)}")
        
        # Initialize SAP Cloud SDK
        sap_sdk = get_sap_cloud_sdk()
        await sap_sdk.send_alert(
            subject=f"{settings.APP_NAME} Starting",
            body=f"Application {settings.APP_NAME} v{settings.APP_VERSION} is starting up (Environment: {config_manager.env.value if config_manager else 'unknown'})",
            severity="INFO",
            category="NOTIFICATION"
        )
        
        # Initialize OpenTelemetry with dynamic config
        if config_manager:
            monitoring_config = config_manager.get_monitoring_config()
            telemetry_enabled = monitoring_config.enable_tracing and monitoring_config.otel_endpoint
        else:
            telemetry_enabled = telemetry_config.otel_enabled
        
        if telemetry_enabled:
            service_name = monitoring_config.otel_service_name if config_manager else os.getenv("OTEL_SERVICE_NAME", f"{settings.APP_NAME}-main")
            init_telemetry(
                service_name=service_name,
                agent_id="main-app",
                sampling_rate=telemetry_config.otel_traces_sampler_arg
            )
            instrument_fastapi(app)
            instrument_httpx()
            instrument_redis()
            logger.info(f"OpenTelemetry enabled for {service_name}")
        
        # Initialize Security Monitoring System
        try:
            security_monitor = get_security_monitor()
            await security_monitor.start_monitoring()
            logger.info("🔍 Security monitoring system started")
        except Exception as security_error:
            logger.error(f"Failed to start security monitoring: {security_error}")
            logger.warning("Application will continue without security monitoring")
        
    except Exception as e:
        logger.error(f"Failed to initialize with dynamic configuration: {e}")
        logger.warning("Continuing with fallback configuration")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    try:
        # Stop security monitoring
        try:
            security_monitor = get_security_monitor()
            await security_monitor.stop_monitoring()
            logger.info("🔍 Security monitoring system stopped")
        except Exception as security_error:
            logger.error(f"Error stopping security monitoring: {security_error}")
        
        # SAP SDK cleanup
        sap_sdk = get_sap_cloud_sdk()
        await sap_sdk.send_alert(
            subject=f"{settings.APP_NAME} Shutting Down",
            body=f"Application {settings.APP_NAME} v{settings.APP_VERSION} is shutting down",
            severity="INFO",
            category="NOTIFICATION"
        )
        await sap_sdk.close()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


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
    ## A2A Platform - Enterprise Agent-to-Agent Network
    
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
    
    ### Request Signing (Optional):
    API supports HMAC-based request signing for enhanced security:
    - SHA-256 HMAC signatures over canonical request data
    - Timestamp validation to prevent replay attacks
    - Nonce tracking for additional replay protection
    - Body integrity verification with hash validation
    - API key-based permissions system
    
    ### Session Management:
    Secure session handling with advanced security features:
    - JWT access tokens with short expiration (15 minutes)
    - Refresh token rotation for enhanced security
    - Session-based authentication with device fingerprinting
    - Automatic session cleanup and maximum session limits
    - Token family revocation for breach detection
    - Replay attack protection with nonce tracking
    
    ### Rate Limiting:
    API calls are rate-limited based on authentication tier:
    - Anonymous: 60 requests/minute, 10 burst capacity
    - Authenticated: 120 requests/minute, 20 burst capacity  
    - Premium: 300 requests/minute, 50 burst capacity
    - Enterprise: 1000 requests/minute, 100 burst capacity
    
    ### DDoS Protection:
    - Automatic detection of attack patterns
    - IP-based blocking for suspicious activity
    - Burst protection with token bucket algorithm
    
    ### Security Monitoring:
    - Real-time threat detection and alerting
    - Automated incident response
    - Comprehensive security event logging
    - Multi-channel alerting (email, Slack, logs)
    - Blockchain security auditing
    - Pattern-based threat detection
    
    ### Automated Security Testing:
    - Static code analysis for vulnerabilities
    - Dependency vulnerability scanning
    - Configuration security auditing
    - API security testing suite
    - Injection vulnerability detection
    - Cryptography strength validation
    - Comprehensive security reporting
    
    ### Compliance & Audit Trails:
    - Comprehensive audit logging with integrity verification
    - Support for SOX, GDPR, SOC2, HIPAA, PCI-DSS, ISO27001, NIST
    - Automated compliance reporting and analysis
    - Audit event correlation and risk assessment
    - Tamper-evident audit trails with cryptographic checksums
    - Real-time compliance monitoring and alerting
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

# Set CORS with dynamic configuration
try:
    if config_manager:
        cors_config = config_manager.get_cors_config()
        # Security: Never use "*" for origins when credentials are allowed
        # Get CORS origins from environment with fallback to localhost for development
        default_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000").split(",")
        allowed_origins = cors_config.origins if cors_config.origins else [origin.strip() for origin in default_origins]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=cors_config.credentials,
            allow_methods=cors_config.methods,
            allow_headers=cors_config.headers,
            max_age=cors_config.max_age
        )
        logger.info(f"CORS configured with {len(cors_config.origins)} allowed origins")
    elif settings.BACKEND_CORS_ORIGINS:
        # Security warning for overly permissive CORS
        logger.warning("⚠️ CORS configured with wildcard methods and headers - consider restricting in production")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],  # More restrictive
            allow_headers=["*"],  # Consider restricting this too
        )
        logger.info("CORS configured with settings BACKEND_CORS_ORIGINS")
except Exception as e:
    logger.error(f"Failed to configure CORS with dynamic config: {e}")
    # Fallback CORS configuration
    if settings.BACKEND_CORS_ORIGINS:
        logger.warning("⚠️ Fallback CORS with wildcard methods - consider restricting in production")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],  # More restrictive
            allow_headers=["*"],  # Consider restricting this too
        )
        logger.warning("Using fallback CORS configuration")

# Add structured logging middleware
create_logging_middleware(
    app,
    log_requests=True,
    log_responses=True,
    log_bodies=False,  # Set to True for debugging only
    exclude_paths=["/health", "/docs", "/openapi.json", "/favicon.ico"]
)

# Add telemetry middleware
if telemetry_config.otel_enabled:
    from app.api.middleware.telemetry import TelemetryMiddleware
    app.add_middleware(TelemetryMiddleware)

# Add rate limiting middleware for A2A API protection
from starlette.middleware.base import BaseHTTPMiddleware
class RateLimitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        return await rate_limit_middleware(request, call_next)

app.add_middleware(RateLimitingMiddleware)

# Add security event monitoring middleware
app.add_middleware(SecurityEventMiddleware)

# Add request signing middleware (optional signing by default)
from app.api.middleware.requestSigning import RequestSigningMiddleware, APIKeyPermissionMiddleware
app.add_middleware(APIKeyPermissionMiddleware)  # Check permissions for signed requests
app.add_middleware(RequestSigningMiddleware, enforce_signing=False)  # Optional signing

# Add global exception handler for secure error handling (disabled in development for debugging)
if not (os.getenv("ENVIRONMENT", "development") == "development"):
    app.add_exception_handler(Exception, global_exception_handler)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(a2a_router)  # A2A routes at /a2a/v1 (Agent 1)
app.include_router(ord_router)  # ORD Registry at /api/v1/ord
app.include_router(agent0_router)  # Agent 0 at /a2a/agent0/v1
app.include_router(agent1_router)  # Agent 1 at /a2a/agent1/v1
app.include_router(agent2_router)  # Agent 2 at /a2a/agent2/v1
app.include_router(agent3_router)  # Agent 3 at /a2a/agent3/v1
app.include_router(agent4_router)  # Agent 4 at /a2a/agent4/v1
app.include_router(agent5_router)  # Agent 5 at /a2a/agent5/v1
app.include_router(calculation_router)  # Calculation Agent at /a2a/calculation/v1
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
    
    try:
        # Get server configuration dynamically
        server_host = os.getenv('HOST', '0.0.0.0')
        server_port = int(os.getenv('PORT', '8888'))
        
        # Determine reload and log level based on environment
        if config_manager:
            is_development = config_manager.is_development()
            reload_enabled = is_development and os.getenv('HOT_RELOAD', 'true').lower() == 'true'
            
            monitoring_config = config_manager.get_monitoring_config()
            log_level = monitoring_config.log_level.lower()
        else:
            reload_enabled = True
            log_level = "info"
        
        logger.info(f"Starting server on {server_host}:{server_port}")
        if config_manager:
            logger.info(f"Environment: {config_manager.env.value}")
        logger.info(f"Reload enabled: {reload_enabled}")
        logger.info(f"Log level: {log_level}")
        
        uvicorn.run(
            "main:app",
            host=server_host,
            port=server_port,
            reload=reload_enabled,
            log_level=log_level
        )
        
    except Exception as e:
        logger.error(f"Failed to start server with dynamic configuration: {e}")
        logger.info("Starting with fallback configuration")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8888,
            reload=True,
            log_level="info"
        )
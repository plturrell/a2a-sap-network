"
# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
""
Standalone API Gateway Server for A2A Network
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from app.api.gateway.router import router as gateway_router
from app.api.middleware.telemetry import TelemetryMiddleware
from app.api.middleware.auth import JWTMiddleware
from app.a2a.core.telemetry import init_telemetry, instrument_fastapi, instrument_httpx
from app.a2a.config.telemetryConfig import telemetry_config
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting A2A API Gateway...")
    
    # Initialize OpenTelemetry if enabled
    if telemetry_config.otel_enabled:
        init_telemetry(
            service_name="a2a-gateway",
            agent_id="gateway",
            sampling_rate=telemetry_config.otel_traces_sampler_arg
        )
        instrument_fastapi(app)
        instrument_httpx()
        logger.info("OpenTelemetry enabled for API Gateway")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")


# Create FastAPI app
app = FastAPI(
    title="A2A Network API Gateway",
    description="Centralized API Gateway for Agent-to-Agent Network",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware with secure configuration
# CRITICAL SECURITY: Never use "*" for origins when credentials are allowed
allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
    "os.getenv("A2A_FRONTEND_URL")",  # Default development frontend
    "os.getenv("A2A_GATEWAY_URL")",  # Alternative dev port
    os.getenv("A2A_SERVICE_URL"),  # localhost alternative
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],  # More restrictive
    allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization", "X-API-Key", "X-Requested-With"],  # Restrictive headers
)

# Telemetry middleware
if telemetry_config.otel_enabled:
    app.add_middleware(TelemetryMiddleware)

# Include gateway router
app.include_router(gateway_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "A2A API Gateway",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "gateway": "/gateway/{service_name}/{path}",
            "health": "/gateway/health",
            "services": "/gateway/services",
            "rate_limits": "/gateway/rate-limits",
            "metrics": "/gateway/metrics"
        },
        "documentation": "/docs"
    }

# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "a2a-gateway",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("GATEWAY_PORT", "8080"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


# Import datetime
from datetime import datetime
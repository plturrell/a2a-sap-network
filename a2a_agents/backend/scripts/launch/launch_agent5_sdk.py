#!/usr/bin/env python3
"""
QA Validation Agent (Agent 5) - SDK Version Launcher
Production-ready launcher with comprehensive configuration and monitoring
"""

import asyncio
import os
import sys
import logging
import signal
import uvicorn
from pathlib import Path
from typing import Dict, Any

# Add backend to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GzipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Import Agent 5 components
from app.a2a.agents.agent5_qa_validation.active.qa_validation_agent_sdk import QAValidationAgentSDK
from app.a2a.agents.agent5_qa_validation.active.agent5_router import router, initialize_agent

# Configuration
from app.a2a.config.production_config import get_agent_config
from src.a2a.core.telemetry import setup_telemetry
from src.a2a.core.auth_manager import AuthManager
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/agent5_qa_validation.log')
    ]
)

logger = logging.getLogger(__name__)

# Global state
agent_instance: QAValidationAgentSDK = None
app_metrics = {
    'requests_total': Counter('a2a_agent5_requests_total', 'Total requests', ['method', 'endpoint']),
    'request_duration': Histogram('a2a_agent5_request_duration_seconds', 'Request duration'),
    'active_connections': Gauge('a2a_agent5_active_connections', 'Active connections'),
    'agent_status': Gauge('a2a_agent5_status', 'Agent status (1=healthy, 0=unhealthy)')
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global agent_instance
    
    try:
        logger.info("🚀 Starting QA Validation Agent (Agent 5)")
        
        # Load configuration
        config = get_agent_config("agent5")
        logger.info(f"📋 Loaded configuration: {config}")
        
        # Setup telemetry
        if config.get('telemetry', {}).get('enabled', True):
            setup_telemetry("agent5-qa-validation", config.get('telemetry', {}))
            logger.info("📊 Telemetry initialized")
        
        # Initialize agent with A2A integration
        agent_instance = await initialize_agent(
            base_url=config.get('base_url', 'http://localhost:8007'),
            data_manager_url=config.get('data_manager_url', 'http://localhost:8001'),
            catalog_manager_url=config.get('catalog_manager_url', 'http://localhost:8002'),
            cache_ttl=config.get('cache_ttl', 3600),
            max_tests_per_product=config.get('max_tests_per_product', 50)
        )
        
        # Set agent status to healthy
        app_metrics['agent_status'].set(1)
        
        logger.info(f"✅ QA Validation Agent initialized successfully")
        logger.info(f"   Agent ID: {agent_instance.agent_id}")
        logger.info(f"   Name: {agent_instance.name}")
        logger.info(f"   Version: {agent_instance.version}")
        logger.info(f"   Base URL: {agent_instance.base_url}")
        logger.info(f"   Data Manager: {agent_instance.data_manager_url}")
        logger.info(f"   Catalog Manager: {agent_instance.catalog_manager_url}")
        logger.info(f"   Trust System: {'✅ Enabled' if agent_instance.trust_identity else '⚠️  Disabled'}")
        
        # Start Prometheus metrics server
        metrics_port = config.get('metrics_port', 9097)
        start_http_server(metrics_port)
        logger.info(f"📈 Prometheus metrics server started on port {metrics_port}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Agent 5: {e}")
        app_metrics['agent_status'].set(0)
        raise
    
    finally:
        # Cleanup
        if agent_instance:
            try:
                await agent_instance.shutdown()
                logger.info("🛑 Agent shutdown completed")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        
        app_metrics['agent_status'].set(0)


# Create FastAPI application with lifespan
app = FastAPI(
    title="A2A QA Validation Agent",
    description="Agent 5: ORD-integrated factuality testing with dynamic question generation using SimpleQA methodology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GzipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and update metrics"""
    start_time = asyncio.get_event_loop().time()
    
    # Update active connections
    app_metrics['active_connections'].inc()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = asyncio.get_event_loop().time() - start_time
        
        # Update metrics
        app_metrics['requests_total'].labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        app_metrics['request_duration'].observe(duration)
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - {duration:.3f}s"
        )
        
        return response
        
    except Exception as e:
        # Log errors
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(f"{request.method} {request.url.path} - ERROR: {e} - {duration:.3f}s")
        raise
    
    finally:
        # Update active connections
        app_metrics['active_connections'].dec()


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "qa-validation-agent", "version": "1.0.0"}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check"""
    global agent_instance
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "status": "ready",
        "agent_id": agent_instance.agent_id,
        "name": agent_instance.name,
        "data_manager_url": agent_instance.data_manager_url,
        "catalog_manager_url": agent_instance.catalog_manager_url,
        "trust_system_enabled": agent_instance.trust_identity is not None
    }


@app.get("/metrics-summary", tags=["Monitoring"])
async def metrics_summary():
    """Get metrics summary"""
    global agent_instance
    if agent_instance is None:
        return {"error": "Agent not initialized"}
    
    return {
        "agent_metrics": {
            "active_test_suites": len(agent_instance.test_suites),
            "websocket_connections": len(agent_instance.websocket_connections),
            "total_tests_generated": sum(
                len(suite.generated_tests) 
                for suite in agent_instance.test_suites.values()
            ),
            "processing_stats": agent_instance.processing_stats
        },
        "system_metrics": {
            "requests_total": app_metrics['requests_total']._value._value,
            "active_connections": app_metrics['active_connections']._value._value,
            "agent_status": app_metrics['agent_status']._value._value
        }
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    global agent_instance
    return {
        "service": "A2A QA Validation Agent",
        "version": "1.0.0",
        "description": "ORD-integrated factuality testing with dynamic question generation",
        "agent": {
            "id": agent_instance.agent_id if agent_instance else None,
            "name": agent_instance.name if agent_instance else None,
            "initialized": agent_instance is not None
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "agent": "/agent5",
            "a2a_tasks": "/agent5/a2a/tasks",
            "websocket_stream": "/agent5/a2a/stream/{task_id}"
        },
        "capabilities": [
            "A2A microservice architecture",
            "ORD registry discovery via Data Manager",
            "SimpleQA test generation", 
            "Dynamic factuality testing",
            "Self-contained validation logic",
            "Vector similarity via Agent 3 integration",
            "WebSocket streaming",
            "A2A protocol compliance",
            "Trust system integration"
        ]
    }


# Include agent router
app.include_router(router)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
            "path": request.url.path
        }
    )


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration
    config = get_agent_config("agent5")
    
    # Server configuration
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 8007)
    workers = config.get('workers', 1)
    reload = config.get('reload', False)
    log_level = config.get('log_level', 'info')
    
    logger.info(f"🚀 Starting QA Validation Agent server on {host}:{port}")
    
    try:
        # Run the server
        uvicorn.run(
            "launch_agent5_sdk:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level,
            access_log=True,
            server_header=False,
            date_header=False
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️  Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)
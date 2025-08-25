"""
A2A Developer Portal Launcher - Complete Integration
Unified entry point for the A2A Developer Portal with full integration
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Portal core components
from .portalServer import DeveloperPortalServer, create_developer_portal
from .api.a2aNetworkIntegration import router as network_router, initialize_a2a_client, NetworkConfig
from .sapBtp.authApi import router as auth_router
from .sapBtp.rbacService import initialize_rbac_service
from .sapBtp.sessionService import initialize_session_service
from .sapBtp.destinationService import initialize_destination_service
from .sapBtp.notificationApi import router as notification_router
from .deployment.deploymentPipeline import DeploymentPipeline

# Agent Builder integration
from .agentBuilder.enhancedAgentBuilder import EnhancedAgentBuilder

# Configuration
from .config.deployment import ProductionConfig as DeploymentConfig

logger = logging.getLogger(__name__)


class PortalIntegrationManager:
    """
    Manages complete integration of all portal components
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = None
        self.portal_server = None
        self.agent_builder = None
        self.deployment_pipeline = None
        self.network_client = None

        # Integration status
        self.integration_status = {
            "portal_server": False,
            "agent_builder": False,
            "network_integration": False,
            "sap_btp_services": False,
            "deployment_pipeline": False
        }

    async def initialize(self) -> bool:
        """
        Initialize all portal integration components
        """
        try:
            logger.info("🚀 Starting A2A Developer Portal Integration...")

            # Create FastAPI app
            self.app = await self._create_application()

            # Initialize portal server
            await self._initialize_portal_server()

            # Initialize agent builder
            await self._initialize_agent_builder()

            # Initialize A2A network integration
            await self._initialize_network_integration()

            # Initialize SAP BTP services
            await self._initialize_sap_btp_services()

            # Initialize deployment pipeline
            await self._initialize_deployment_pipeline()

            # Setup API routes
            await self._setup_api_routes()

            # Validate integration
            await self._validate_integration()

            logger.info("✅ Portal integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Portal integration failed: {e}")
            return False

    async def _create_application(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title="A2A Developer Portal",
            description="Comprehensive IDE for A2A Agent Development",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add security headers middleware
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            security_headers = getattr(DeploymentConfig, 'SECURITY_HEADERS', {})
            for header, value in security_headers.items():
                response.headers[header] = value
            return response

        return app

    async def _initialize_portal_server(self):
        """Initialize portal server component"""
        try:
            self.portal_server = DeveloperPortalServer(self.config)
            await self.portal_server.initialize()
            self.integration_status["portal_server"] = True
            logger.info("✅ Portal server initialized")
        except Exception as e:
            logger.error(f"❌ Portal server initialization failed: {e}")
            raise

    async def _initialize_agent_builder(self):
        """Initialize enhanced agent builder"""
        try:
            builder_config = {
                "templates_path": str(Path(self.config.get("workspace_path", "/tmp/a2a_workspace")) / "templates"),
                "output_path": str(Path(self.config.get("workspace_path", "/tmp/a2a_workspace")) / "agents"),
                "validation_enabled": True,
                "auto_deployment": self.config.get("auto_deployment", False)
            }

            self.agent_builder = EnhancedAgentBuilder(builder_config)
            await self.agent_builder.initialize()
            self.integration_status["agent_builder"] = True
            logger.info("✅ Agent builder initialized")
        except Exception as e:
            logger.error(f"❌ Agent builder initialization failed: {e}")
            raise

    async def _initialize_network_integration(self):
        """Initialize A2A network integration"""
        try:
            network_config = NetworkConfig(
                network=self.config.get("blockchain", {}).get("network", "mainnet"),
                rpc_url=self.config.get("blockchain", {}).get("rpc_url"),
                websocket_url=self.config.get("blockchain", {}).get("websocket_url")
            )

            await initialize_a2a_client(network_config)
            self.integration_status["network_integration"] = True
            logger.info("✅ A2A network integration initialized")
        except Exception as e:
            logger.warning(f"⚠️ A2A network integration failed (optional): {e}")
            # Network integration is optional in development

    async def _initialize_sap_btp_services(self):
        """Initialize SAP BTP services"""
        try:
            # Initialize RBAC service
            await initialize_rbac_service(self.config.get("sap_btp", {}).get("rbac", {}))

            # Initialize session service
            await initialize_session_service(self.config.get("sap_btp", {}).get("session", {}))

            # Initialize destination service (optional)
            if self.config.get("sap_btp", {}).get("destinations"):
                await initialize_destination_service(self.config.get("sap_btp", {}).get("destinations", {}))

            self.integration_status["sap_btp_services"] = True
            logger.info("✅ SAP BTP services initialized")
        except Exception as e:
            logger.warning(f"⚠️ SAP BTP services initialization failed (optional): {e}")
            # SAP BTP is optional in development

    async def _initialize_deployment_pipeline(self):
        """Initialize deployment pipeline"""
        try:
            deployment_config = {
                "deployments_path": str(Path(self.config.get("workspace_path", "/tmp/a2a_workspace")) / "deployments"),
                "artifacts_path": str(Path(self.config.get("workspace_path", "/tmp/a2a_workspace")) / "artifacts"),
                "logs_path": str(Path(self.config.get("workspace_path", "/tmp/a2a_workspace")) / "logs"),
                "portal_url": f"http://localhost:{self.config.get('port', 3001)}",
                "email": self.config.get("email", {})
            }

            self.deployment_pipeline = DeploymentPipeline(deployment_config)
            await self.deployment_pipeline.initialize()
            self.integration_status["deployment_pipeline"] = True
            logger.info("✅ Deployment pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Deployment pipeline initialization failed: {e}")
            raise

    async def _setup_api_routes(self):
        """Setup all API routes"""
        try:
            # Include all routers
            self.app.include_router(auth_router, prefix="/api")
            self.app.include_router(network_router, prefix="/api")
            self.app.include_router(notification_router, prefix="/api")

            # Portal server routes
            if self.portal_server:
                self.app.mount("/api/portal", self.portal_server.get_router())

            # Agent builder routes
            if self.agent_builder:
                self.app.mount("/api/agent-builder", self.agent_builder.get_router())

            # Deployment pipeline routes
            if self.deployment_pipeline:
                self.app.mount("/api/deployment", self.deployment_pipeline.get_router())

            # Static files
            static_path = Path(__file__).parent / "static"
            if static_path.exists():
                self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

            # Root endpoint
            @self.app.get("/", response_class=HTMLResponse)
            async def root():
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>A2A Developer Portal</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                </head>
                <body>
                    <h1>🚀 A2A Developer Portal</h1>
                    <p>Welcome to the A2A Agent Development IDE</p>
                    <ul>
                        <li><a href="/static/index.html">Portal Dashboard</a></li>
                        <li><a href="/docs">API Documentation</a></li>
                        <li><a href="/api/health">Health Check</a></li>
                    </ul>
                </body>
                </html>
                """

            # Health check endpoint
            @self.app.get("/api/health")
            async def health_check():
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "integration_status": self.integration_status,
                    "version": "1.0.0"
                }

            logger.info("✅ API routes configured")
        except Exception as e:
            logger.error(f"❌ API routes setup failed: {e}")
            raise

    async def _validate_integration(self):
        """Validate complete integration"""
        try:
            # Check critical components
            critical_components = ["portal_server", "agent_builder", "deployment_pipeline"]
            failed_components = [comp for comp in critical_components if not self.integration_status[comp]]

            if failed_components:
                raise Exception(f"Critical components failed: {failed_components}")

            # Log integration summary
            logger.info("🎯 Integration Summary:")
            for component, status in self.integration_status.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"  {status_icon} {component}: {'OK' if status else 'FAILED'}")

            logger.info("✅ Portal integration validation completed")
        except Exception as e:
            logger.error(f"❌ Integration validation failed: {e}")
            raise


def create_portal_launcher(config: Dict[str, Any] = None) -> PortalIntegrationManager:
    """
    Create portal launcher with default configuration
    """
    if config is None:
        config = {
            "workspace_path": os.getenv("A2A_WORKSPACE_PATH", "/tmp/a2a_workspace"),
            "port": int(os.getenv("PORTAL_PORT", "3001")),
            "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
            "blockchain": {
                "network": os.getenv("BLOCKCHAIN_NETWORK", "mainnet"),
                "rpc_url": os.getenv("BLOCKCHAIN_RPC_URL"),
                "websocket_url": os.getenv("BLOCKCHAIN_WS_URL")
            },
            "sap_btp": {
                "rbac": {
                    "development_mode": os.getenv("SAP_RBAC_DEV_MODE", "true").lower() == "true"
                },
                "session": {
                    "session_timeout_minutes": int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
                }
            },
            "email": {
                "smtp_host": os.getenv("SMTP_HOST"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD")
            },
            "auto_deployment": os.getenv("AUTO_DEPLOYMENT", "false").lower() == "true"
        }

    return PortalIntegrationManager(config)


async def launch_portal(config: Dict[str, Any] = None):
    """
    Launch the complete A2A Developer Portal
    """
    # Create and initialize portal
    portal_launcher = create_portal_launcher(config)

    success = await portal_launcher.initialize()
    if not success:
        logger.error("❌ Portal initialization failed")
        return None

    return portal_launcher.app


async def main():
    """
    Main entry point for launching the portal
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Launch portal
        app = await launch_portal()
        if not app:
            sys.exit(1)

        # Run server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=int(os.getenv("PORTAL_PORT", "3001")),
            log_level="info",
            reload=os.getenv("PORTAL_RELOAD", "false").lower() == "true"
        )

        server = uvicorn.Server(config)

        logger.info("🌐 Portal URL: http://localhost:3001")
        logger.info("📚 API Docs: http://localhost:3001/docs")
        logger.info("❤️ Health Check: http://localhost:3001/api/health")

        await server.serve()

    except KeyboardInterrupt:
        logger.info("✋ Portal stopped by user")
    except Exception as e:
        logger.error(f"❌ Portal launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

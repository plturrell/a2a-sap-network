#!/usr/bin/env python3
"""
Launch A2A Developer Portal with Complete Integration
Unified launcher for the fully integrated portal system
"""

import asyncio
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add portal to path
portal_path = Path(__file__).parent.parent.parent / "app" / "a2a" / "developerPortal"
sys.path.insert(0, str(portal_path))

from portalLauncher import create_portal_launcher, launch_portal

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('portal.log', mode='a')
        ]
    )


def load_environment_config() -> Dict[str, Any]:
    """Load configuration from environment and .env files"""
    # Try to load from .env.portal file
    env_file = portal_path / ".env.portal"
    if env_file.exists():
        logger.info(f"Loading configuration from: {env_file}")
        # Simple .env parser
        env_vars = {}
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Set environment variables
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
    
    # Build configuration from environment
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
            },
            "xsuaa": {
                "service_url": os.getenv("XSUAA_SERVICE_URL"),
                "client_id": os.getenv("XSUAA_CLIENT_ID"),
                "client_secret": os.getenv("XSUAA_CLIENT_SECRET")
            }
        },
        "database": {
            "url": os.getenv("DATABASE_URL", "sqlite:///./a2a_portal.db"),
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10"))
        },
        "email": {
            "service": os.getenv("EMAIL_SERVICE", "smtp"),
            "from_address": os.getenv("EMAIL_FROM", "noreply@a2a-portal.com"),
            "smtp_host": os.getenv("SMTP_HOST"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_username": os.getenv("SMTP_USERNAME"),
            "smtp_password": os.getenv("SMTP_PASSWORD"),
            "smtp_use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        },
        "security": {
            "secret_key": os.getenv("SECRET_KEY", "dev-secret-key"),
            "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "jwt_access_token_expires": int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))
        },
        "features": {
            "agent_builder": os.getenv("FEATURE_AGENT_BUILDER", "true").lower() == "true",
            "workflow_designer": os.getenv("FEATURE_WORKFLOW_DESIGNER", "true").lower() == "true",
            "deployment_pipeline": os.getenv("FEATURE_DEPLOYMENT_PIPELINE", "true").lower() == "true",
            "a2a_network_integration": os.getenv("FEATURE_A2A_NETWORK_INTEGRATION", "true").lower() == "true",
            "collaboration_tools": os.getenv("FEATURE_COLLABORATION_TOOLS", "true").lower() == "true",
            "analytics_dashboard": os.getenv("FEATURE_ANALYTICS_DASHBOARD", "true").lower() == "true"
        },
        "performance": {
            "worker_processes": int(os.getenv("WORKER_PROCESSES", "1")),
            "worker_threads": int(os.getenv("WORKER_THREADS", "4")),
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "300"))
        },
        "auto_deployment": os.getenv("AUTO_DEPLOYMENT", "false").lower() == "true",
        "development_mode": os.getenv("DEVELOPMENT_MODE", "true").lower() == "true",
        "enable_telemetry": os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
    }
    
    return config


def check_prerequisites() -> bool:
    """Check if all prerequisites are met"""
    logger.info("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        "fastapi", "uvicorn", "httpx", "jinja2", 
        "pydantic", "web3", "sqlalchemy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All prerequisites met")
    return True


def create_workspace(workspace_path: str):
    """Create workspace directory structure"""
    workspace = Path(workspace_path)
    
    directories = [
        "projects",
        "templates",
        "agents",
        "workflows",
        "deployments",
        "artifacts",
        "logs",
        "backups"
    ]
    
    for directory in directories:
        dir_path = workspace / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Workspace created at: {workspace}")


def print_startup_banner(config: Dict[str, Any]):
    """Print startup banner with configuration info"""
    print("\n" + "="*80)
    print("🚀 A2A DEVELOPER PORTAL - COMPLETE INTEGRATION")
    print("="*80)
    print(f"\n📍 Portal URL: http://localhost:{config['port']}")
    print(f"📚 API Documentation: http://localhost:{config['port']}/docs")
    print(f"❤️ Health Check: http://localhost:{config['port']}/api/health")
    print(f"💾 Workspace: {config['workspace_path']}")
    
    print("\n🎯 ENABLED FEATURES:")
    for feature, enabled in config['features'].items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    print("\n🔧 INTEGRATIONS:")
    integrations = [
        ("Agent Builder", "Core agent development tools"),
        ("BPMN Workflow Designer", "Visual workflow creation"),
        ("A2A Network", f"Blockchain network ({config['blockchain']['network']})"),
        ("SAP BTP Services", "Enterprise authentication & services"),
        ("Deployment Pipeline", "Automated testing & deployment"),
        ("Real-time Collaboration", "Multi-user WebSocket communication")
    ]
    
    for name, description in integrations:
        print(f"  🔌 {name}: {description}")
    
    print("\n⚙️ QUICK START:")
    print("  1. Navigate to the Portal URL above")
    print("  2. Create a new project or open an existing one")
    print("  3. Use the Agent Builder to create your first agent")
    print("  4. Configure A2A Network integration in settings")
    print("  5. Deploy your agent using the deployment pipeline")
    
    print("\n📖 DOCUMENTATION:")
    print("  • Integration Guide: PORTAL_INTEGRATION.md")
    print("  • Environment Config: .env.portal.template")
    print("  • API Reference: http://localhost:3001/docs")
    
    print("\n" + "="*80 + "\n")


async def main():
    """Main entry point"""
    setup_logging()
    
    logger.info("🎯 Starting A2A Developer Portal with Complete Integration")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_environment_config()
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create workspace
    try:
        create_workspace(config["workspace_path"])
    except Exception as e:
        logger.error(f"❌ Failed to create workspace: {e}")
        sys.exit(1)
    
    # Print startup banner
    print_startup_banner(config)
    
    # Launch portal
    try:
        app = await launch_portal(config)
        if not app:
            logger.error("❌ Failed to initialize portal")
            sys.exit(1)
        
        # Import uvicorn here to avoid import issues
        import uvicorn
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=config["port"],
            log_level="info",
            reload=config.get("development_mode", False),
            workers=1 if config.get("development_mode", False) else config["performance"]["worker_processes"]
        )
        
        server = uvicorn.Server(uvicorn_config)
        
        logger.info(f"🌐 Portal starting on http://localhost:{config['port']}")
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("✋ Portal stopped by user")
    except Exception as e:
        logger.error(f"❌ Portal launch failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
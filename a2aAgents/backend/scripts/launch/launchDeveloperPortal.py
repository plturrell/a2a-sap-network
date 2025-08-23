#!/usr/bin/env python3
"""
Launch script for A2A Developer Portal
SAP UI5 Fiori-based IDE for A2A agent development
"""

import asyncio
import logging
import uvicorn
from typing import Dict, Any

from app.a2a.developer_portal.portal_server import create_developer_portal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Launch the A2A Developer Portal"""
    
    # Portal configuration
    config = {
        "workspace_path": "/tmp/a2a_workspace",
        "templates_path": "app/a2a/developer_portal/templates", 
        "static_path": "app/a2a/developer_portal/static",
        "port": 8090
    }
    
    logger.info("🚀 Starting A2A Developer Portal...")
    logger.info("📱 SAP UI5 Fiori Interface")
    
    # Create portal instance
    portal = create_developer_portal(config)
    
    # Initialize portal
    await portal.initialize()
    
    try:
        # Run the portal server
        uvicorn_config = uvicorn.Config(
            app=portal.app,
            host="127.0.0.1",
            port=config["port"],
            log_level="info",
            access_log=True,
            reload=False
        )
        
        server = uvicorn.Server(uvicorn_config)
        
        logger.info(f"🌐 A2A Developer Portal available at: http://localhost:{config['port']}")
        logger.info("🎨 SAP UI5 Fiori Design System")
        logger.info("📋 Features:")
        logger.info("  • SAP Fiori Launchpad-style project management")  
        logger.info("  • Visual Agent Builder with templates")
        logger.info("  • BPMN 2.0 Workflow Designer")
        logger.info("  • Integrated Code Editor (Monaco)")
        logger.info("  • Real-time collaboration")
        logger.info("  • Built-in testing & deployment")
        logger.info("  • A2A Network monitoring")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down A2A Developer Portal...")
    finally:
        logger.info("✅ A2A Developer Portal stopped")


if __name__ == "__main__":
    asyncio.run(main())
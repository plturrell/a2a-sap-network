#!/usr/bin/env python3
"""
Launch script for A2A Health Dashboard
Real-time monitoring of A2A agent network health
"""

import asyncio
import logging
import uvicorn
from typing import Dict, Any

from app.a2a.dashboard.health_dashboard import create_health_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Launch the health dashboard"""
    
    # Dashboard configuration
    config = {
        "check_interval": 30,      # Health check interval in seconds
        "timeout": 5,              # HTTP request timeout
        "history_retention": 24,   # Hours to retain history
        "port": 8888              # Dashboard port
    }
    
    logger.info("🏥 Starting A2A Health Dashboard...")
    
    # Create dashboard instance
    dashboard = create_health_dashboard(config)
    
    # Start health monitoring
    await dashboard.start_monitoring()
    
    try:
        # Run the dashboard server
        uvicorn_config = uvicorn.Config(
            app=dashboard.app,
            host="0.0.0.0",
            port=config["port"],
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(uvicorn_config)
        
        logger.info(f"🌐 Health Dashboard available at: http://localhost:{config['port']}")
        logger.info("🔍 Monitoring the following services:")
        for service_id, service_config in dashboard.service_registry.items():
            logger.info(f"  • {service_config['name']} ({service_id}) - {service_config['endpoint']}")
        
        logger.info("📊 Dashboard features:")
        logger.info("  • Real-time health monitoring")
        logger.info("  • WebSocket-based live updates")
        logger.info("  • Service metrics visualization") 
        logger.info("  • Historical data tracking")
        logger.info("  • Alert management")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Health Dashboard...")
    finally:
        # Stop monitoring
        await dashboard.stop_monitoring()
        logger.info("✅ Health Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(main())
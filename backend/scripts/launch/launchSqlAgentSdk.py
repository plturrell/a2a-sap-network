#!/usr/bin/env python3
"""
Launch SQL Agent with SDK
Natural Language to SQL and SQL to Natural Language conversion
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import SQL Agent
from app.a2a.agents.sqlAgent.active.sqlAgentSdk import SQLAgentSDK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def launch_sql_agent(port: int = 8150, host: str = "0.0.0.0"):
    """Launch the SQL Agent"""
    
    logger.info("=" * 80)
    logger.info("🚀 Launching A2A SQL Agent (SDK Version)")
    logger.info("=" * 80)
    
    # Create base URL
    base_url = f"https://{host}:{port}"
    
    # Initialize agent
    logger.info(f"📡 Initializing SQL Agent on {base_url}")
    agent = SQLAgentSDK(
        base_url=base_url,
        enable_monitoring=True
    )
    
    try:
        # Initialize agent resources
        logger.info("🔧 Initializing agent resources...")
        await agent.initialize()
        logger.info("✅ Agent resources initialized successfully")
        
        # Create FastAPI app
        logger.info("🌐 Creating FastAPI application...")
        app = agent.create_fastapi_app()
        
        # Add health check endpoint
        @app.get("/")
        async def root():
            return {
                "name": "A2A SQL Agent",
                "version": agent.version,
                "status": "running",
                "capabilities": [
                    "Natural Language to SQL conversion",
                    "SQL to Natural Language explanation", 
                    "HANA SQL support",
                    "Graph query support",
                    "Vector similarity search",
                    "Hybrid queries"
                ],
                "endpoints": {
                    "agent_card": "/.well-known/agent.json",
                    "rpc": "/rpc",
                    "messages": "/messages",
                    "skills": "/skills",
                    "health": "/health"
                }
            }
        
        # Run the server
        logger.info(f"🚀 Starting SQL Agent server on {base_url}")
        logger.info("=" * 80)
        logger.info("📋 Agent Information:")
        logger.info(f"   ID: {agent.agent_id}")
        logger.info(f"   Name: {agent.name}")
        logger.info(f"   Version: {agent.version}")
        logger.info("=" * 80)
        logger.info("🔧 Available Skills:")
        for skill in agent.list_skills():
            logger.info(f"   - {skill['name']}: {skill['description']}")
        logger.info("=" * 80)
        logger.info("📡 API Endpoints:")
        logger.info(f"   Agent Card: {base_url}/.well-known/agent.json")
        logger.info(f"   JSON-RPC: {base_url}/rpc")
        logger.info(f"   Messages: {base_url}/messages")
        logger.info(f"   Skills: {base_url}/skills")
        logger.info(f"   Health: {base_url}/health")
        logger.info("=" * 80)
        logger.info("💡 Example Usage:")
        logger.info("   Natural Language to SQL:")
        logger.info('     curl -X POST http://localhost:8150/rpc \\')
        logger.info('       -H "Content-Type: application/json" \\')
        logger.info('       -d \'{"jsonrpc": "2.0", "method": "agent.processMessage", ')
        logger.info('            "params": {"message": {"parts": [{"kind": "data", ')
        logger.info('            "data": {"method": "nl2sql", "query": "show all customers where country is USA"}}]}}, ')
        logger.info('            "id": 1}\'')
        logger.info("")
        logger.info("   SQL to Natural Language:")
        logger.info('     curl -X POST http://localhost:8150/rpc \\')
        logger.info('       -H "Content-Type: application/json" \\')
        logger.info('       -d \'{"jsonrpc": "2.0", "method": "agent.processMessage", ')
        logger.info('            "params": {"message": {"parts": [{"kind": "data", ')
        logger.info('            "data": {"method": "sql2nl", "query": "SELECT * FROM customers WHERE country = \'USA\'"}}]}}, ')
        logger.info('            "id": 1}\'')
        logger.info("=" * 80)
        logger.info("✅ SQL Agent is ready!")
        logger.info("")
        
        # Run with uvicorn
        import uvicorn
        await uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="info"
            )
        ).serve()
        
    except Exception as e:
        logger.error(f"❌ Failed to start SQL Agent: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🧹 Shutting down SQL Agent...")
        await agent.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Launch A2A SQL Agent")
    parser.add_argument(
        "--port",
        type=int,
        default=8150,
        help="Port to run the agent on (default: 8150)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Run the agent
    try:
        asyncio.run(launch_sql_agent(port=args.port, host=args.host))
    except KeyboardInterrupt:
        logger.info("\n👋 SQL Agent shutdown requested")
    except Exception as e:
        logger.error(f"❌ SQL Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
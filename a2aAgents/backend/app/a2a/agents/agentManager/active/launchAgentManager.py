import asyncio
import uvicorn
import logging
import os
import sys
from pathlib import Path

from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Agent Manager Launcher - Starts the Agent Manager A2A Agent
"""

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.a2a.agents.agent_manager_agent import AgentManagerAgent
from app.a2a.agents import agent_manager_router


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_agent_manager_app():
    """Create and configure the Agent Manager FastAPI application"""

    # Initialize the Agent Manager
    agent_id = "agent_manager"
    agent_name = "Agent Manager"
    base_url = os.getenv("A2A_AGENT_BASE_URL", os.getenv("SERVICE_BASE_URL"))

    # Agent Manager capabilities
    capabilities = {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
        "batchProcessing": True,
        "smartContractDelegation": True,
        "aiAdvisor": True,
        "helpSeeking": True,
        "taskTracking": True,
        "agentRegistration": True,
        "trustContractManagement": True,
        "workflowOrchestration": True,
        "systemMonitoring": True
    }

    # Agent Manager skills
    skills = [
        {
            "id": "agent-registration",
            "name": "Agent Registration",
            "description": "Register and manage A2A agents in the ecosystem",
            "tags": ["registration", "management", "a2a"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "trust-contract-management",
            "name": "Trust Contract Management",
            "description": "Create and manage trust contracts between agents",
            "tags": ["trust", "contracts", "delegation"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "workflow-orchestration",
            "name": "Workflow Orchestration",
            "description": "Orchestrate complex workflows across multiple agents",
            "tags": ["workflow", "orchestration", "coordination"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        },
        {
            "id": "system-monitoring",
            "name": "System Monitoring",
            "description": "Monitor health and performance of the A2A ecosystem",
            "tags": ["monitoring", "health", "metrics"],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"]
        }
    ]

    # Create the Agent Manager instance
    agent_manager = AgentManagerAgent(
        agent_id=agent_id,
        agent_name=agent_name,
        base_url=base_url,
        capabilities=capabilities,
        skills=skills
    )

    # Set the agent instance in the router module
    agent_manager_router.agent_manager = agent_manager

    # Create FastAPI app
    app = FastAPI(
        title="Agent Manager A2A Agent",
        description="Orchestrates A2A ecosystem registration, trust contracts, and workflow management",
        version="2.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the Agent Manager router
    app.include_router(agent_manager_router.router)

    # Add root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Agent Manager A2A Agent",
            "version": "2.0.0",
            "protocol_version": "0.2.9",
            "status": "operational"
        }

    # Health check endpoint at root level
    @app.get("/health")
    async def health():
        return await agent_manager_router.health_check()

    logger.info(f"Agent Manager initialized:")
    logger.info(f"  - Agent ID: {agent_id}")
    logger.info(f"  - Agent Name: {agent_name}")
    logger.info(f"  - Base URL: {base_url}")
    logger.info(f"  - Capabilities: {len(capabilities)} features")
    logger.info(f"  - Skills: {len(skills)} skills")

    return app


def main():
    """Main entry point for the Agent Manager launcher"""
    logger.info("Starting Agent Manager A2A Agent...")

    # Get configuration from environment
    host = os.getenv("AGENT_MANAGER_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_MANAGER_PORT", "8005"))
    log_level = os.getenv("LOG_LEVEL", "info")

    # Create the app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(create_agent_manager_app())

    logger.info(f"Agent Manager starting on {host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  - Agent Card: GET /.well-known/agent.json")
    logger.info("  - JSON-RPC: POST /a2a/agent_manager/v1/rpc")
    logger.info("  - REST Messages: POST /a2a/agent_manager/v1/messages")
    logger.info("  - Health Check: GET /a2a/agent_manager/v1/health")
    logger.info("  - Queue Status: GET /a2a/agent_manager/v1/queue/status")
    logger.info("  - Agent Registration: POST /a2a/agent_manager/v1/agents/register")
    logger.info("  - Trust Contracts: POST /a2a/agent_manager/v1/trust/contracts")
    logger.info("  - Workflows: POST /a2a/agent_manager/v1/workflows")
    logger.info("  - System Health: GET /a2a/agent_manager/v1/system/health")

    # Start the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()

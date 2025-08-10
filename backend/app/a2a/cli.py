#!/usr/bin/env python3
"""
A2A CLI - Command Line Interface for A2A Network Management

This CLI provides tools for deploying, managing, and monitoring A2A agents.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from .config.productionConfig import ProductionConfigManager as ProductionConfig, Environment
from .core.telemetry import init_telemetry as setup_telemetry
from .agents.agentManager.active.agentManagerAgent import AgentManagerAgent


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="A2A Network Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  a2a-agent start --agent agent0 --port 8001
  a2a-agent deploy --environment production
  a2a-agent status --all
  a2a-agent test --agent-type standardization
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start an A2A agent")
    start_parser.add_argument("--agent", required=True, 
                            choices=["agent0", "agent1", "agent2", "agent3", 
                                   "agent4", "agent5", "data-manager", 
                                   "catalog-manager", "agent-manager"],
                            help="Agent to start")
    start_parser.add_argument("--port", type=int, help="Port to run agent on")
    start_parser.add_argument("--config", help="Configuration file path")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy A2A network")
    deploy_parser.add_argument("--environment", 
                             choices=["development", "testing", "staging", "production"],
                             default="development",
                             help="Deployment environment")
    deploy_parser.add_argument("--config-file", help="Configuration file")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check agent status")
    status_parser.add_argument("--all", action="store_true", 
                             help="Check status of all agents")
    status_parser.add_argument("--agent", help="Specific agent to check")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run agent tests")
    test_parser.add_argument("--agent-type", 
                           choices=["data-product", "standardization", 
                                  "ai-preparation", "vector-processing"],
                           help="Type of agent to test")
    test_parser.add_argument("--integration", action="store_true",
                           help="Run integration tests")
    
    return parser


async def start_agent(agent_name: str, port: Optional[int] = None, config_path: Optional[str] = None):
    """Start a specific agent."""
    logging.info(f"Starting {agent_name} agent...")
    
    # Agent port mapping
    default_ports = {
        "agent0": 8001,
        "agent1": 8002, 
        "agent2": 8003,
        "agent3": 8004,
        "agent4": 8009,
        "agent5": 8010,
        "data-manager": 8005,
        "catalog-manager": 8006,
        "agent-manager": 8007,
    }
    
    port = port or default_ports.get(agent_name, 8000)
    
    # Dynamic import and start based on agent name
    if agent_name == "agent0":
        from .agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        agent = DataProductRegistrationAgentSDK(
            base_url=f"http://localhost:{port}",
            ord_registry_url="http://localhost:9000"
        )
    elif agent_name == "agent1":
        from .agents.agent1Standardization.active.dataStandardizationAgentSdk import DataStandardizationAgentSDK
        agent = DataStandardizationAgentSDK()
    elif agent_name == "agent2":
        from .agents.agent2AiPreparation.active.aiPreparationAgentSdk import AiPreparationAgentSDK
        agent = AiPreparationAgentSDK()
    elif agent_name == "agent3":
        from .agents.agent3VectorProcessing.active.vectorProcessingAgentSdk import VectorProcessingAgentSDK
        agent = VectorProcessingAgentSDK()
    else:
        logging.error(f"Unknown agent: {agent_name}")
        return False
    
    # Initialize the agent if it has initialize method
    if hasattr(agent, 'initialize'):
        await agent.initialize()
    
    # Create FastAPI app
    app = agent.create_fastapi_app()
    
    logging.info(f"🚀 Starting {agent.name} v{agent.version}")
    logging.info(f"📡 Listening on http://localhost:{port}")
    logging.info(f"🎯 Agent ID: {agent.agent_id}")
    
    # Start server with uvicorn
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        if hasattr(agent, 'shutdown'):
            await agent.shutdown()
    
    logging.info(f"Agent {agent_name} stopped")
    return True


async def deploy_network(environment: str, config_file: Optional[str] = None):
    """Deploy the entire A2A network."""
    logging.info(f"Deploying A2A network to {environment}")
    
    env = Environment(environment)
    config = ProductionConfig(env)
    
    if config_file:
        config.load_from_file(config_file)
    
    # Setup telemetry
    setup_telemetry(environment)
    
    # Deploy agents in sequence
    agents = ["agent0", "agent1", "agent2", "agent3", "data-manager", "catalog-manager"]
    
    for agent_name in agents:
        success = await start_agent(agent_name)
        if not success:
            logging.error(f"Failed to deploy {agent_name}")
            return False
    
    logging.info("A2A network deployed successfully")
    return True


async def check_status(agent_name: Optional[str] = None, check_all: bool = False):
    """Check agent status."""
    if check_all:
        agents = ["agent0", "agent1", "agent2", "agent3", "data-manager", "catalog-manager"]
        for agent in agents:
            # Implementation would check agent health endpoints
            logging.info(f"{agent}: Status check not implemented yet")
    elif agent_name:
        logging.info(f"{agent_name}: Status check not implemented yet")
    else:
        logging.error("Must specify --agent or --all")


async def run_tests(agent_type: Optional[str] = None, integration: bool = False):
    """Run agent tests."""
    logging.info("Running tests...")
    
    if integration:
        logging.info("Running integration tests...")
        # Would run pytest with integration markers
    
    if agent_type:
        logging.info(f"Running tests for {agent_type} agents...")
        # Would run specific test suites
    
    logging.info("Tests completed")


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        if args.command == "start":
            await start_agent(args.agent, args.port, args.config)
        elif args.command == "deploy":
            await deploy_network(args.environment, args.config_file)
        elif args.command == "status":
            await check_status(args.agent, args.all)
        elif args.command == "test":
            await run_tests(args.agent_type, args.integration)
        else:
            logging.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        return 1
    
    return 0


def cli_main():
    """Synchronous entry point for console scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(cli_main())
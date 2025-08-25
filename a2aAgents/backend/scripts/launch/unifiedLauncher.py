#!/usr/bin/env python3
"""
Unified A2A Agent Launcher

This script replaces all individual launch scripts with a single parameterized launcher.
Usage: python unified_launcher.py --agent=agent0 --port=8001
"""

import argparse
import asyncio
import logging
import sys
import os
import subprocess
from pathlib import Path

# Add project directories to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "services" / "shared"))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def launch_agent(agent_name: str, port: int, config_path: str = None):
    """Launch a specific agent using subprocess calls to actual service main.py files."""
    logger = logging.getLogger(f"launcher.{agent_name}")
    
    try:
        # Map agent names to their actual service directories
        service_map = {
            "agent0": "services/agent0DataProduct/src/main.py",
            "agent1": "services/agent1Standardization/src/main.py", 
            "agent2": "services/agent2AiPreparation/src/main.py",
            "agent3": "services/agent3VectorProcessing/src/main.py",
            "agent4": "services/agent4CalcValidation/src/main.py",
            "agent5": "services/agent5QaValidation/src/main.py",
            "agent6": "services/agent6QualityControl/src/main.py",
            "data-manager": "services/dataManager/src/main.py",
            "agent-manager": "services/agentManager/src/main.py",
            "catalog-manager": "services/catalogManager/src/main.py",
            "glean-agent": "scripts/launch/launchGleanAgent.py",
            "agent17-chat": "main.py",  # Agent 17 uses main FastAPI app
        }
        
        # Handle agents that still need to be created (use FastAPI template)
        missing_agents = {
            "reasoning-agent": ("Reasoning Agent", "services/reasoningAgent/src/main.py"),
            "sql-agent": ("SQL Agent", "services/sqlAgent/src/main.py"),
            "calculation-agent": ("Calculation Agent", "services/calculationAgent/src/main.py"),
            "agent-builder": ("Agent Builder", "services/agentBuilder/src/main.py"),
            "embedding-finetuner": ("Embedding Fine-Tuner", "services/embeddingFineTuner/src/main.py"),
            "registry-server": ("Registry Server", "app/a2aRegistry/runRegistryServer.py"),
        }
        
        project_root = Path(__file__).parent.parent.parent
        
        if agent_name in service_map:
            service_path = project_root / service_map[agent_name]
            if service_path.exists():
                logger.info(f"Starting {agent_name} from {service_path} on port {port}")
                
                # Set environment variables for the service
                env = os.environ.copy()
                env["A2A_AGENT_PORT"] = str(port)
                env["A2A_AGENT_HOST"] = "0.0.0.0"
                env["A2A_AGENT_URL"] = f"http://0.0.0.0:{port}"
                env["A2A_SERVICE_URL"] = f"http://localhost:{port}"
                env["A2A_SERVICE_HOST"] = "localhost"
                env["A2A_BASE_URL"] = f"http://localhost:{port}"
                env["A2A_AGENT_BASE_URL"] = f"http://localhost:{port}"
                env["PORT"] = str(port)
                # Add downstream and manager URLs for agent communication
                env["A2A_DOWNSTREAM_URL"] = "http://localhost:8010"  # Agent Manager
                env["A2A_AGENT_MANAGER_URL"] = "http://localhost:8010"
                # Inherit blockchain settings from parent environment
                env["A2A_RPC_URL"] = env.get("A2A_RPC_URL", "http://localhost:8545")
                env["A2A_AGENT_REGISTRY_ADDRESS"] = env.get("A2A_AGENT_REGISTRY_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
                env["A2A_MESSAGE_ROUTER_ADDRESS"] = env.get("A2A_MESSAGE_ROUTER_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
                if config_path:
                    env["A2A_CONFIG_PATH"] = config_path
                
                # Launch the service
                process = subprocess.Popen([
                    sys.executable, str(service_path)
                ], env=env, cwd=str(project_root))
                
                logger.info(f"Agent {agent_name} started with PID {process.pid}")
                return True
            else:
                logger.error(f"Service file not found: {service_path}")
                return False
                
        elif agent_name in missing_agents:
            description, service_path = missing_agents[agent_name]
            full_service_path = project_root / service_path
            
            if full_service_path.exists():
                # Service file exists, launch it
                logger.info(f"Starting {agent_name} from existing {service_path} on port {port}")
                
                # Set environment variables for the service
                env = os.environ.copy()
                env["A2A_AGENT_PORT"] = str(port)
                env["A2A_AGENT_HOST"] = "0.0.0.0"
                env["A2A_AGENT_URL"] = f"http://0.0.0.0:{port}"
                env["A2A_SERVICE_URL"] = f"http://localhost:{port}"
                env["A2A_SERVICE_HOST"] = "localhost"
                env["A2A_BASE_URL"] = f"http://localhost:{port}"
                env["A2A_AGENT_BASE_URL"] = f"http://localhost:{port}"
                env["PORT"] = str(port)
                # Add downstream and manager URLs for agent communication
                env["A2A_DOWNSTREAM_URL"] = "http://localhost:8010"  # Agent Manager
                env["A2A_AGENT_MANAGER_URL"] = "http://localhost:8010"
                # Inherit blockchain settings from parent environment
                env["A2A_RPC_URL"] = env.get("A2A_RPC_URL", "http://localhost:8545")
                env["A2A_AGENT_REGISTRY_ADDRESS"] = env.get("A2A_AGENT_REGISTRY_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
                env["A2A_MESSAGE_ROUTER_ADDRESS"] = env.get("A2A_MESSAGE_ROUTER_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
                if config_path:
                    env["A2A_CONFIG_PATH"] = config_path
                
                # Launch the service
                process = subprocess.Popen([
                    sys.executable, str(full_service_path)
                ], env=env, cwd=str(project_root))
                
                logger.info(f"Agent {agent_name} started with PID {process.pid}")
                return True
            else:
                logger.warning(f"{description} service not implemented yet at {service_path}")
                logger.info(f"Creating placeholder service for {agent_name}")
                
                # Create the missing service directory and basic FastAPI service
                service_dir = full_service_path.parent
                service_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a basic FastAPI service template
                template_content = f'''"""
{description} Service
Auto-generated placeholder service
"""

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="{description}", version="1.0.0")

@app.get("/health")
async def health():
    return {{"status": "healthy", "service": "{description}"}}

@app.get("/")
async def root():
    return {{"message": "Hello from {description}"}}

if __name__ == "__main__":
    port = int(os.getenv("A2A_AGENT_PORT", {port}))
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
'''
                
                with open(full_service_path, "w") as f:
                    f.write(template_content)
                
                logger.info(f"Created placeholder service: {service_path}")
                
                # Now launch the newly created service
                env = os.environ.copy()
                env["A2A_AGENT_PORT"] = str(port)
                env["A2A_AGENT_HOST"] = "0.0.0.0"
                
                process = subprocess.Popen([
                    sys.executable, str(full_service_path)
                ], env=env, cwd=str(project_root))
                
                logger.info(f"Placeholder agent {agent_name} started with PID {process.pid}")
                return True
        else:
            logger.error(f"Unknown agent: {agent_name}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start {agent_name}: {e}")
        raise
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified A2A Agent Launcher")
    
    parser.add_argument(
        "--agent",
        required=True,
        choices=[
            "agent0", "agent1", "agent2", "agent3", "agent4", "agent5", "agent6",
            "data-manager", "catalog-manager", "agent-manager", "agent-builder",
            "reasoning-agent", "sql-agent", "calculation-agent", "embedding-finetuner",
            "registry-server", "glean-agent", "agent17-chat"
        ],
        help="Agent to launch"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run agent on"
    )
    
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Default ports (fixed port conflicts for 100% coverage)
    default_ports = {
        "registry-server": 8000,
        "agent0": 8001,
        "agent1": 8002,
        "agent2": 8003,
        "agent3": 8004,
        "agent4": 8005,
        "agent5": 8006,
        "agent6": 8007,
        "glean-agent": 8016,  # New port for Glean Agent
        "reasoning-agent": 8008,
        "sql-agent": 8009,
        "agent-manager": 8010,
        "data-manager": 8011,
        "catalog-manager": 8012,
        "calculation-agent": 8013,
        "agent-builder": 8014,
        "embedding-finetuner": 8015,
        "agent17-chat": 8017,
    }
    
    port = args.port or default_ports[args.agent]
    
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        success = launch_agent(args.agent, port, args.config)
        if success:
            logging.info(f"Agent {args.agent} launched successfully on port {port}")
        else:
            logging.error(f"Failed to launch agent {args.agent}")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception as e:
        logging.error(f"Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
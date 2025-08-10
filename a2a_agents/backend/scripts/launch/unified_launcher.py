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
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from a2a.core.telemetry import setup_telemetry


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def launch_agent(agent_name: str, port: int, config_path: str = None):
    """Launch a specific agent."""
    setup_telemetry("development")
    logger = logging.getLogger(f"launcher.{agent_name}")
    
    try:
        if agent_name == "agent0":
            from a2a.agents.agent0_data_product.active.data_product_agent_sdk import DataProductRegistrationAgentSDK
            agent = DataProductRegistrationAgentSDK()
            logger.info(f"Starting Data Product Agent on port {port}")
            
        elif agent_name == "agent1":
            from a2a.agents.agent1_standardization.active.data_standardization_agent_sdk import DataStandardizationAgentSDK
            agent = DataStandardizationAgentSDK()
            logger.info(f"Starting Data Standardization Agent on port {port}")
            
        elif agent_name == "agent2":
            from a2a.agents.agent2_ai_preparation.active.ai_preparation_agent_sdk import AiPreparationAgentSDK
            agent = AiPreparationAgentSDK()
            logger.info(f"Starting AI Preparation Agent on port {port}")
            
        elif agent_name == "agent3":
            from a2a.agents.agent3_vector_processing.active.vector_processing_agent_sdk import VectorProcessingAgentSDK
            agent = VectorProcessingAgentSDK()
            logger.info(f"Starting Vector Processing Agent on port {port}")
            
        elif agent_name == "agent4":
            from a2a.agents.agent4_calc_validation.active.calc_validation_agent_sdk import CalcValidationAgentSDK
            agent = CalcValidationAgentSDK()
            logger.info(f"Starting Calculation Validation Agent on port {port}")
            
        elif agent_name == "agent5":
            from a2a.agents.agent5_qa_validation.active.qa_validation_agent_sdk import QAValidationAgentSDK
            agent = QAValidationAgentSDK()
            logger.info(f"Starting QA Validation Agent on port {port}")
            
        elif agent_name == "data-manager":
            from a2a.agents.data_manager.active.data_manager_agent_sdk import DataManagerAgentSDK
            agent = DataManagerAgentSDK()
            logger.info(f"Starting Data Manager Agent on port {port}")
            
        elif agent_name == "catalog-manager":
            from a2a.agents.catalog_manager.active.catalog_manager_agent_sdk import CatalogManagerAgentSDK
            agent = CatalogManagerAgentSDK()
            logger.info(f"Starting Catalog Manager Agent on port {port}")
            
        elif agent_name == "agent-manager":
            from a2a.agents.agent_manager.active.agent_manager_agent import AgentManagerAgent
            agent = AgentManagerAgent()
            logger.info(f"Starting Agent Manager on port {port}")
            
        elif agent_name == "agent-builder":
            from a2a.agents.agent_builder.active.agent_builder_agent_sdk import AgentBuilderAgentSDK
            agent = AgentBuilderAgentSDK()
            logger.info(f"Starting Agent Builder on port {port}")
            
        else:
            logger.error(f"Unknown agent: {agent_name}")
            return False
        
        # Load config if provided
        if config_path:
            # Implementation would load agent-specific config
            logger.info(f"Loading config from: {config_path}")
        
        # Start the agent
        await agent.start(host="0.0.0.0", port=port)
        
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
            "agent0", "agent1", "agent2", "agent3", "agent4", "agent5",
            "data-manager", "catalog-manager", "agent-manager", "agent-builder"
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
    
    # Default ports
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
        "agent-builder": 8008,
    }
    
    port = args.port or default_ports[args.agent]
    
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        asyncio.run(launch_agent(args.agent, port, args.config))
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception as e:
        logging.error(f"Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
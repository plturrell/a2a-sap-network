"""
A2A Network Main Application
Blockchain-based agent communication system

A2A PROTOCOL COMPLIANCE:
This application replaces the traditional FastAPI HTTP server with
blockchain event monitoring. All agent communication goes through
the A2A blockchain messaging system.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import blockchain listener
from .sdk.a2aBlockchainListener import A2ABlockchainListener

# Import all agent handlers
from .agents.agent0DataProduct.active.agent0A2AHandler import create_agent0_a2a_handler
from .agents.agent1Standardization.active.agent1StandardizationA2AHandler import create_agent1_a2a_handler
from .agents.agent2AiPreparation.active.agent2AiPreparationA2AHandler import create_agent2_a2a_handler
from .agents.agent3VectorProcessing.active.agent3VectorProcessingA2AHandler import create_agent3_a2a_handler
from .agents.agent4CalcValidation.active.agent4CalcValidationA2AHandler import create_agent4_a2a_handler
from .agents.agent5QaValidation.active.agent5QaValidationA2AHandler import create_agent5_a2a_handler
from .agents.agentManager.active.agent_managerA2AHandler import create_agent_manager_a2a_handler
from .agents.calculationAgent.active.calculation_agentA2AHandler import create_calculation_agent_a2a_handler
from .agents.catalogManager.active.catalog_managerA2AHandler import create_catalog_manager_a2a_handler
from .agents.reasoningAgent.active.agent9RouterA2AHandler import create_agent9Router_a2a_handler

# Import agent SDKs
from .agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
from .agents.agent1Standardization.active.dataStandardizationAgentSdk import DataStandardizationAgentSDK
from .agents.agent2AiPreparation.active.aiPreparationAgentSdk import AiPreparationAgentSDK
from .agents.agent3VectorProcessing.active.vectorProcessingAgentSdk import VectorProcessingAgentSDK
from .agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
from .agents.agent5QaValidation.active.qaValidationAgentSdk import QaValidationAgentSDK
from .agents.agentManager.active.agentManagerAgent import AgentManagerSDK
from .agents.calculationAgent.active.comprehensiveCalculationAgentSdk import CalculationAgentSDK
from .agents.catalogManager.active.comprehensiveCatalogManagerSdk import CatalogManagerSDK
from .agents.reasoningAgent.active.reasoningAgentSdk import ReasoningAgentSDK


class A2ANetworkApplication:
    """Main A2A Network Application"""

    def __init__(self):
        self.listener: Optional[A2ABlockchainListener] = None
        self.running = False
        self.handlers = []

    async def initialize_agents(self) -> List:
        """Initialize all agent SDKs and handlers"""
        logger.info("Initializing A2A agents...")

        # Initialize agent SDKs
        agent_sdks = {
            'agent0': DataProductRegistrationAgentSDK(
                agent_id="agent0_data_product",
                downstream_agent_url=None  # No HTTP URLs in A2A protocol
            ),
            'agent1': DataStandardizationAgentSDK(
                agent_id="agent1_standardization",
                downstream_agent_url=None
            ),
            'agent2': AiPreparationAgentSDK(
                agent_id="agent2_ai_preparation",
                downstream_agent_url=None
            ),
            'agent3': VectorProcessingAgentSDK(
                agent_id="agent3_vector_processing",
                downstream_agent_url=None
            ),
            'agent4': CalcValidationAgentSDK(
                agent_id="agent4_calc_validation",
                downstream_agent_url=None
            ),
            'agent5': QaValidationAgentSDK(
                agent_id="agent5_qa_validation",
                downstream_agent_url=None
            ),
            'agent_manager': AgentManagerSDK(
                agent_id="agent_manager"
            ),
            'calculation_agent': CalculationAgentSDK(
                agent_id="calculation_agent"
            ),
            'catalog_manager': CatalogManagerSDK(
                agent_id="catalog_manager"
            ),
            'reasoning_agent': ReasoningAgentSDK(
                agent_id="reasoning_agent"
            )
        }

        # Create A2A handlers
        handlers = [
            create_agent0_a2a_handler(agent_sdks['agent0']),
            create_agent1_a2a_handler(agent_sdks['agent1']),
            create_agent2_a2a_handler(agent_sdks['agent2']),
            create_agent3_a2a_handler(agent_sdks['agent3']),
            create_agent4_a2a_handler(agent_sdks['agent4']),
            create_agent5_a2a_handler(agent_sdks['agent5']),
            create_agent_manager_a2a_handler(agent_sdks['agent_manager']),
            create_calculation_agent_a2a_handler(agent_sdks['calculation_agent']),
            create_catalog_manager_a2a_handler(agent_sdks['catalog_manager']),
            create_agent9Router_a2a_handler(agent_sdks['reasoning_agent'])
        ]

        logger.info(f"Initialized {len(handlers)} A2A agent handlers")
        return handlers

    async def start(self):
        """Start the A2A network application"""
        logger.info("Starting A2A Network Application...")

        # Check required environment variables
        required_env = ['A2A_PRIVATE_KEY', 'A2A_MESSAGE_ROUTER_ADDRESS']
        missing = [var for var in required_env if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            logger.error("Please set up your environment using the setup script")
            sys.exit(1)

        # Initialize agents
        self.handlers = await self.initialize_agents()

        # Create blockchain listener
        self.listener = A2ABlockchainListener(self.handlers)
        await self.listener.start()

        self.running = True
        logger.info("A2A Network Application started successfully")
        logger.info("Listening for blockchain messages...")

        # Display status
        await self.display_status()

    async def stop(self):
        """Stop the A2A network application"""
        logger.info("Stopping A2A Network Application...")
        self.running = False

        if self.listener:
            await self.listener.stop()

        logger.info("A2A Network Application stopped")

    async def display_status(self):
        """Display application status"""
        if self.listener:
            status = await self.listener.get_status()

            print("\n" + "="*60)
            print("A2A NETWORK STATUS")
            print("="*60)
            print(f"Blockchain Connected: {status['connected']}")
            print(f"Active Agents: {len(status['handlers'])}")
            print(f"Agents: {', '.join(status['handlers'])}")
            print(f"RPC URL: {status['config']['rpc_url']}")
            print(f"Poll Interval: {status['config']['poll_interval']}s")
            print("="*60)
            print("All agent communication now goes through blockchain messaging")
            print("No REST endpoints are exposed - full A2A protocol compliance")
            print("="*60 + "\n")

    async def run(self):
        """Main application loop"""
        await self.start()

        # Run event listening loop
        try:
            await self.listener.listen()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await self.stop()


def setup_signal_handlers(app: A2ANetworkApplication):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(app.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                A2A Network Application                    ║
    ║            Blockchain-Based Agent Communication           ║
    ║                                                           ║
    ║  Protocol: A2A Blockchain Messaging                       ║
    ║  Mode: Full Protocol Compliance                           ║
    ║  HTTP Endpoints: DISABLED                                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Create application
    app = A2ANetworkApplication()

    # Setup signal handlers
    setup_signal_handlers(app)

    # Run application
    await app.run()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("Python 3.7+ is required")
        sys.exit(1)

    # Run the application
    asyncio.run(main())


# Migration instructions for deployment
"""
MIGRATION FROM REST TO A2A:

1. Stop existing FastAPI servers:
   - Kill all processes using ports 8000-8015
   - Stop any REST API services

2. Deploy smart contracts:
   - Deploy MessageRouter contract
   - Deploy AgentRegistry contract
   - Note contract addresses

3. Configure environment:
   export A2A_PRIVATE_KEY="your-private-key"
   export A2A_RPC_URL="http://localhost:8545"
   export A2A_MESSAGE_ROUTER_ADDRESS="0x..."
   export A2A_AGENT_REGISTRY_ADDRESS="0x..."

4. Start A2A application:
   python -m app.a2a.main_a2a

5. Client migration:
   - Update all client code to use blockchain messaging
   - Remove HTTP client libraries
   - Use A2ANetworkClient for all communication

6. Monitoring:
   - Monitor blockchain events
   - Check transaction confirmations
   - Verify message delivery

ROLLBACK PROCEDURE:
- Keep REST routers as backup (but don't deploy)
- Can switch back by running old main.py if needed
- Dual-mode operation NOT recommended for security
"""

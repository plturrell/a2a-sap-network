#!/usr/bin/env python3
"""
Integrate Calculation Testing Flow with BDC Smart Contract
Sets up trust relationships between CalcTesting, CalculationAgent, and Data_Manager
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import logging

# Set up blockchain contract addresses from deployment with correct prefix
os.environ['A2A_AGENT_REGISTRY_ADDRESS'] = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
os.environ['A2A_MESSAGE_ROUTER_ADDRESS'] = '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512'
os.environ['A2A_NETWORK_RPC_URL'] = "os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))"
os.environ['A2A_NETWORK_CHAIN_ID'] = '31337'
os.environ['A2A_NETWORK'] = 'localhost'

# Set the artifacts path so contract config can find deployment
os.environ['A2A_ARTIFACTS_PATH'] = '/Users/apple/projects/a2a/a2aNetwork/broadcast/Deploy.s.sol'

# Set the ABI path for contract loading
os.environ['A2A_ABI_PATH'] = '/Users/apple/projects/a2a/a2aNetwork/out'

# Add backend to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Add a2aNetwork to path for trust system
a2a_network_dir = Path(__file__).parent.parent.parent.parent / "a2aNetwork"
sys.path.insert(0, str(a2a_network_dir))

# Import agents
from app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
from app.a2a.agents.calculationAgent.active.calculationAgentSdk import CalculationAgentSDK
from app.a2a.agents.dataManager.active.dataManagerAgentSdk import DataManagerAgentSDK

# Import trust system
from trustSystem.smartContractTrust import SmartContractTrust, initialize_agent_trust


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalculationTestingIntegrator:
    """Integrates calculation testing flow with BDC smart contract"""
    
    def __init__(self):
        self.agents = {}
        self.trust_system = SmartContractTrust("bdc_calc_testing_contract")
        
    async def initialize_agents(self):
        """Initialize all agents in the calculation testing flow"""
        logger.info("🚀 Initializing agents for calculation testing flow...")
        
        # Get agent URLs from environment or use localhost for local development
        calc_validation_url = os.getenv("CALC_VALIDATION_AGENT_URL")
        calculation_agent_url = os.getenv("CALCULATION_AGENT_URL")
        data_manager_url = os.getenv("DATA_MANAGER_URL", "os.getenv("DATA_MANAGER_URL")")
        
        # Initialize CalcValidation Agent (Agent 4)
        calc_validation_agent = CalcValidationAgentSDK(
            base_url=calc_validation_url,
            data_manager_url=data_manager_url,
            enable_monitoring=True
        )
        await calc_validation_agent.initialize()
        self.agents["calc_validation"] = calc_validation_agent
        
        # Initialize Calculation Agent
        calculation_agent = CalculationAgentSDK(
            base_url=calculation_agent_url,
            enable_monitoring=True
        )
        await calculation_agent.initialize()
        self.agents["calculation"] = calculation_agent
        
        # Initialize Data Manager
        storage_config = {
            "default_backend": "filesystem",
            "filesystem_path": "/tmp/a2a_data_manager",
            "hana_config": None,
            "service_levels": {
                "gold": {"backup_enabled": True, "compression": True},
                "silver": {"backup_enabled": True, "compression": False},
                "bronze": {"backup_enabled": False, "compression": False}
            }
        }
        
        data_manager_agent = DataManagerAgentSDK(
            base_url=data_manager_url,
            storage_config=storage_config
        )
        await data_manager_agent.initialize()
        self.agents["data_manager"] = data_manager_agent
        
        logger.info("✅ All agents initialized successfully")
        
    async def setup_trust_relationships(self):
        """Setup trust relationships between agents in BDC smart contract"""
        logger.info("🔐 Setting up trust relationships in BDC smart contract...")
        
        # Register agents in trust system
        for agent_name, agent in self.agents.items():
            identity = self.trust_system.register_agent(
                agent_id=agent.agent_id,
                agent_type=agent_name
            )
            logger.info(f"   Registered {agent_name}: {agent.agent_id}")
            
        # Establish trust channels between agents
        trust_pairs = [
            ("calc_validation_agent_4", "calculation_agent"),
            ("calculation_agent", "data_manager_agent"),
            ("calc_validation_agent_4", "data_manager_agent")
        ]
        
        for agent1, agent2 in trust_pairs:
            trust_channel = self.trust_system.establish_trust_channel(agent1, agent2)
            logger.info(f"   Established trust channel: {agent1} <-> {agent2} (Channel ID: {trust_channel['channel_id']})")
            
        logger.info("✅ Trust relationships established")
        
    async def create_test_templates(self):
        """Create and store test templates in Data Manager"""
        logger.info("📝 Creating test templates...")
        
        test_templates = {
            "calculation_tests": {
                "mathematical": {
                    "easy": [
                        {
                            "question": "Calculate the derivative of x^2 + 3x + 5",
                            "methodology": "Power rule differentiation",
                            "steps": ["Apply power rule to each term", "d/dx(x^n) = nx^(n-1)", "Combine results"],
                            "expected_answer": "2x + 3"
                        },
                        {
                            "question": "Solve the equation 2x + 5 = 15",
                            "methodology": "Linear equation solving",
                            "steps": ["Subtract 5 from both sides", "Divide by 2", "Simplify"],
                            "expected_answer": "5"
                        }
                    ],
                    "medium": [
                        {
                            "question": "Find the integral of sin(x) * cos(x)",
                            "methodology": "Substitution method or trigonometric identity",
                            "steps": ["Use identity: sin(x)cos(x) = (1/2)sin(2x)", "Integrate", "Add constant"],
                            "expected_answer": "-(1/4)cos(2x) + C"
                        },
                        {
                            "question": "Calculate the limit of (x^2 - 4)/(x - 2) as x approaches 2",
                            "methodology": "Factoring and simplification",
                            "steps": ["Factor numerator", "Cancel common terms", "Evaluate limit"],
                            "expected_answer": "4"
                        }
                    ],
                    "hard": [
                        {
                            "question": "Solve the system: x + y = 10, x^2 + y^2 = 58",
                            "methodology": "Substitution and quadratic solving",
                            "steps": ["Express y = 10 - x", "Substitute into second equation", "Solve quadratic", "Find both solutions"],
                            "expected_answer": "[(3, 7), (7, 3)]"
                        }
                    ]
                },
                "financial": {
                    "easy": [
                        {
                            "question": "Calculate compound interest on $1000 at 5% annual rate for 3 years",
                            "methodology": "Compound interest formula",
                            "steps": ["Apply A = P(1 + r)^t", "Substitute values", "Calculate"],
                            "expected_answer": {"final_amount": 1157.63, "interest_earned": 157.63}
                        }
                    ],
                    "medium": [
                        {
                            "question": "Price a 5-year bond with 4% coupon and 3% yield",
                            "methodology": "Bond pricing formula",
                            "steps": ["Calculate PV of coupons", "Calculate PV of face value", "Sum present values"],
                            "expected_answer": {"bond_price": 1045.80}
                        }
                    ]
                },
                "graph": {
                    "easy": [
                        {
                            "question": "Find the shortest path in a graph from node A to node E",
                            "methodology": "Dijkstra's algorithm",
                            "steps": ["Initialize distances", "Process nodes by shortest distance", "Update neighbors", "Track path"],
                            "graph_data": {
                                "nodes": ["A", "B", "C", "D", "E"],
                                "edges": [["A", "B", 4], ["A", "C", 2], ["B", "C", 1], ["B", "D", 5], ["C", "D", 8], ["C", "E", 10], ["D", "E", 2]]
                            }
                        }
                    ]
                }
            }
        }
        
        # Store templates in Data Manager
        data_manager = self.agents["data_manager"]
        
        # Create storage request using the direct skill method
        storage_result = await data_manager.data_create_skill(
            data=test_templates,
            storage_backend="filesystem",
            service_level="gold",
            metadata={
                "data_type": "test_templates",
                "category": "calculation_testing",
                "created_by": "integration_script",
                "version": "1.0.0"
            }
        )
        
        if hasattr(storage_result, 'success') and storage_result.success:
            logger.info(f"✅ Test templates stored successfully: {storage_result.data_id}")
        else:
            error_msg = getattr(storage_result, 'message', 'Unknown error')
            logger.error(f"❌ Failed to store test templates: {error_msg}")
            
    async def run_sample_test_flow(self):
        """Run a sample test flow to verify integration"""
        logger.info("🧪 Running sample test flow...")
        
        calc_validation = self.agents["calc_validation"]
        
        # Test 1: Simple calculation test
        logger.info("\n📊 Test 1: Simple mathematical calculation")
        result1 = await calc_validation.dispatch_calculation_test(
            "Calculate the derivative of x^3 + 2x^2 - 5x + 7",
            "mathematical",
            "easy"
        )
        logger.info(f"   Dispatch result: {result1.get('status')}")
        
        if result1.get("status") == "success":
            # Evaluate the result
            calc_result = result1.get("calculation_result", {})
            evaluation = await calc_validation.evaluate_calculation_result(
                result1.get("question_id"),
                calc_result,
                "3x^2 + 4x - 5"
            )
            logger.info(f"   Evaluation score: {evaluation.get('overall_score')}/100")
            logger.info(f"   Feedback: {evaluation.get('feedback')}")
        
        # Test 2: Financial calculation test
        logger.info("\n💰 Test 2: Financial calculation")
        result2 = await calc_validation.dispatch_calculation_test(
            "Calculate the present value of $5000 received in 2 years with 6% annual discount rate",
            "financial",
            "medium"
        )
        logger.info(f"   Dispatch result: {result2.get('status')}")
        
        # Get scoreboard
        logger.info("\n📈 Getting calculation test scoreboard...")
        scoreboard = await calc_validation.get_calculation_test_scoreboard()
        logger.info(f"   Total questions: {scoreboard['summary']['total_questions']}")
        logger.info(f"   Accuracy rate: {scoreboard['summary']['accuracy_rate']}")
        logger.info(f"   Average methodology score: {scoreboard['summary']['average_methodology_score']}")
        
    async def run(self):
        """Run the complete integration"""
        try:
            # Initialize agents
            await self.initialize_agents()
            
            # Setup trust relationships
            await self.setup_trust_relationships()
            
            # Create test templates
            await self.create_test_templates()
            
            # Run sample test flow
            await self.run_sample_test_flow()
            
            logger.info("\n✅ Calculation testing flow integration completed successfully!")
            
            # Display trust relationships
            logger.info("\n🔐 Trust System Status:")
            for agent_id, relationships in self.trust_system.trust_relationships.items():
                logger.info(f"   {agent_id}:")
                for peer_id, trust_score in relationships.items():
                    logger.info(f"      -> {peer_id}: {trust_score}")
                    
        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            raise
        finally:
            # Cleanup
            logger.info("\n🧹 Cleaning up...")
            for agent_name, agent in self.agents.items():
                await agent.shutdown()


async def main():
    """Main entry point"""
    integrator = CalculationTestingIntegrator()
    await integrator.run()


if __name__ == "__main__":
    asyncio.run(main())
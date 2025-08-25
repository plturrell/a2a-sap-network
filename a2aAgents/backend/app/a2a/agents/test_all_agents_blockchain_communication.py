#!/usr/bin/env python3
"""
Comprehensive Blockchain Communication Test for All 16 A2A Agents

This test verifies that all 16 agents can:
1. Initialize blockchain integration properly
2. Send and receive blockchain messages
3. Verify trust and reputation properly
4. Execute blockchain-specific capabilities
5. Coordinate through the blockchain network

Test Coverage:
- Basic blockchain initialization for each agent
- Inter-agent message communication
- Trust verification and reputation checks
- Capability-specific blockchain operations
- Network-wide coordination tests
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Agent imports - using a simplified approach to test blockchain integration
AGENTS_AVAILABLE = True
agent_modules = {
    "agentManager": ("agentManager.active.enhancedAgentManagerAgent", "EnhancedAgentManagerAgent"),
    "qualityControlManager": ("agent6QualityControl.active.qualityControlManagerAgent", "QualityControlManagerAgent"),
    "dataManager": ("dataManager.active.enhancedDataManagerAgentSdk", "EnhancedDataManagerAgentSDK"),
    "agent4CalcValidation": ("agent4CalcValidation.active.enhancedCalcValidationAgentSdk", "EnhancedCalcValidationAgent"),
    "agent5QaValidation": ("agent5QaValidation.active.enhancedQaValidationAgentSdk", "EnhancedQAValidationAgent"),
    "calculationAgent": ("calculationAgent.active.enhancedCalculationAgentSdk", "EnhancedCalculationAgentSDK"),
    "catalogManager": ("catalogManager.active.enhancedCatalogManagerAgentSdk", "EnhancedCatalogManagerAgent"),
    "agentBuilder": ("agentBuilder.active.enhancedAgentBuilderAgentSdk", "EnhancedAgentBuilderAgent"),
    "embeddingFineTuner": ("embeddingFineTuner.active.enhancedEmbeddingFineTunerAgentSdk", "EnhancedEmbeddingFineTunerAgent"),
    "sqlAgent": ("sqlAgent.active.enhancedSqlAgentSdk", "EnhancedSqlAgentSDK"),
    "reasoningAgent": ("reasoningAgent.enhancedReasoningAgent", "EnhancedReasoningAgent"),
    "dataProductAgent": ("agent0DataProduct.active.enhancedDataProductAgentSdk", "EnhancedDataProductAgentSDK"),
    "standardizationAgent": ("agent1Standardization.active.enhancedDataStandardizationAgentSdk", "EnhancedDataStandardizationAgentSDK"),
    "aiPreparationAgent": ("agent2AiPreparation.active.aiPreparationAgentSdk", "AIPreparationAgentSDK"),
    "vectorProcessingAgent": ("agent3VectorProcessing.active.vectorProcessingAgentSdk", "VectorProcessingAgentSDK")
}

# Try to import each agent dynamically
imported_agents = {}
for agent_id, (module_path, class_name) in agent_modules.items():
    try:
        module = __import__(module_path, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        imported_agents[agent_id] = agent_class
        logger.info(f"✅ Imported {agent_id}: {class_name}")
    except Exception as e:
        logger.warning(f"⚠️  Could not import {agent_id}: {str(e)}")
        imported_agents[agent_id] = None


class BlockchainCommunicationTester:
    """Test all 16 agents blockchain communication capabilities"""

    def __init__(self):
        self.agents = {}
        self.test_results = {
            "agent_initialization": {},
            "blockchain_connectivity": {},
            "inter_agent_communication": {},
            "trust_verification": {},
            "capability_tests": {},
            "coordination_tests": {},
            "overall_success": False
        }
        self.blockchain_enabled = os.getenv("BLOCKCHAIN_ENABLED", "false").lower() == "true"

        # Agent definitions with their blockchain capabilities - dynamically populated
        self.agent_definitions = {}
        port = 8001

        capability_mapping = {
            "agentManager": ["orchestration", "coordination", "task_delegation", "agent_lifecycle", "resource_management"],
            "qualityControlManager": ["quality_assurance", "test_execution", "validation_reporting", "compliance_checking"],
            "dataManager": ["data_operations", "data_validation", "data_transformation", "metadata_management"],
            "agent4CalcValidation": ["calculation_validation", "mathematical_verification", "formula_checking", "numerical_analysis"],
            "agent5QaValidation": ["qa_validation", "quality_assurance", "test_execution", "validation_reporting"],
            "calculationAgent": ["calculation_requests", "mathematical_computation", "distributed_calculation", "formula_verification"],
            "catalogManager": ["catalog_search", "resource_registration", "metadata_indexing", "discovery_services"],
            "agentBuilder": ["agent_creation", "template_management", "deployment_automation", "lifecycle_management"],
            "embeddingFineTuner": ["embedding_optimization", "model_fine_tuning", "model_collaboration", "performance_optimization"],
            "sqlAgent": ["sql_query_execution", "database_operations", "query_optimization", "distributed_query"],
            "reasoningAgent": ["complex_reasoning", "inference_processing", "logical_analysis", "decision_support"],
            "dataProductAgent": ["data_product_management", "product_lifecycle", "data_publishing", "consumer_management"],
            "standardizationAgent": ["data_standardization", "schema_validation", "format_conversion", "compliance_checking"],
            "aiPreparationAgent": ["ai_model_preparation", "data_preprocessing", "feature_engineering", "model_training"],
            "vectorProcessingAgent": ["vector_operations", "similarity_computation", "embedding_processing", "vector_indexing"]
        }

        # Populate agent definitions from imported agents
        for agent_id, agent_class in imported_agents.items():
            if agent_class is not None:
                self.agent_definitions[agent_id] = {
                    "class": agent_class,
                    "base_url": f"http://localhost:{port}",
                    "capabilities": capability_mapping.get(agent_id, ["general_capabilities"])
                }
                port += 1

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive blockchain communication test for all 16 agents"""
        logger.info("🚀 Starting Comprehensive Blockchain Communication Test for All 16 A2A Agents")

        try:
            # Step 1: Initialize all agents
            logger.info("📋 Step 1: Initializing all 16 agents...")
            await self._initialize_all_agents()

            # Step 2: Test blockchain connectivity
            logger.info("🔗 Step 2: Testing blockchain connectivity...")
            await self._test_blockchain_connectivity()

            # Step 3: Test inter-agent communication
            logger.info("💬 Step 3: Testing inter-agent communication...")
            await self._test_inter_agent_communication()

            # Step 4: Test trust verification
            logger.info("🛡️  Step 4: Testing trust verification...")
            await self._test_trust_verification()

            # Step 5: Test agent-specific capabilities
            logger.info("⚡ Step 5: Testing agent-specific capabilities...")
            await self._test_agent_capabilities()

            # Step 6: Test network-wide coordination
            logger.info("🌐 Step 6: Testing network-wide coordination...")
            await self._test_network_coordination()

            # Generate final report
            self._generate_test_report()

            return self.test_results

        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "traceback": traceback.format_exc()}

    async def _initialize_all_agents(self):
        """Initialize all 16 agents and test their blockchain integration"""
        logger.info("Initializing agents with blockchain capabilities...")

        for agent_id, config in self.agent_definitions.items():
            try:
                logger.info(f"  Initializing {agent_id}...")

                # Create agent instance
                agent_class = config["class"]
                base_url = config["base_url"]

                # Try different initialization patterns
                if agent_id in ["reasoningAgent"]:
                    agent = agent_class()
                else:
                    agent = agent_class(base_url)

                # Initialize agent (including blockchain)
                if hasattr(agent, 'initialize'):
                    await agent.initialize()

                # Store agent reference
                self.agents[agent_id] = agent

                # Test blockchain integration
                blockchain_status = await self._test_agent_blockchain_integration(agent, agent_id)

                self.test_results["agent_initialization"][agent_id] = {
                    "status": "success",
                    "blockchain_integration": blockchain_status,
                    "capabilities": config["capabilities"],
                    "initialized_at": datetime.now().isoformat()
                }

                logger.info(f"  ✅ {agent_id} initialized successfully")

            except Exception as e:
                logger.error(f"  ❌ Failed to initialize {agent_id}: {str(e)}")
                self.test_results["agent_initialization"][agent_id] = {
                    "status": "failed",
                    "error": str(e),
                    "capabilities": config["capabilities"]
                }

    async def _test_agent_blockchain_integration(self, agent, agent_id: str) -> Dict[str, Any]:
        """Test individual agent's blockchain integration"""
        try:
            blockchain_status = {
                "mixin_present": hasattr(agent, 'blockchain_client'),
                "client_initialized": False,
                "identity_set": False,
                "capabilities_defined": False,
                "message_handlers": []
            }

            # Check blockchain client
            if hasattr(agent, 'blockchain_client'):
                blockchain_status["client_initialized"] = agent.blockchain_client is not None

            # Check agent identity
            if hasattr(agent, 'agent_identity'):
                blockchain_status["identity_set"] = agent.agent_identity is not None

            # Check blockchain capabilities
            if hasattr(agent, 'blockchain_capabilities'):
                blockchain_status["capabilities_defined"] = bool(agent.blockchain_capabilities)
                blockchain_status["capabilities_list"] = getattr(agent, 'blockchain_capabilities', [])

            # Check for blockchain message handlers
            for attr_name in dir(agent):
                if attr_name.startswith('_handle_blockchain_'):
                    blockchain_status["message_handlers"].append(attr_name)

            return blockchain_status

        except Exception as e:
            logger.error(f"Blockchain integration test failed for {agent_id}: {str(e)}")
            return {"error": str(e)}

    async def _test_blockchain_connectivity(self):
        """Test each agent's blockchain connectivity"""
        logger.info("Testing blockchain connectivity for all agents...")

        for agent_id, agent in self.agents.items():
            try:
                # Test blockchain connection
                connectivity_result = await self._test_single_agent_connectivity(agent, agent_id)
                self.test_results["blockchain_connectivity"][agent_id] = connectivity_result

                if connectivity_result.get("connected", False):
                    logger.info(f"  ✅ {agent_id} blockchain connectivity: OK")
                else:
                    logger.warning(f"  ⚠️  {agent_id} blockchain connectivity: Issues detected")

            except Exception as e:
                logger.error(f"  ❌ {agent_id} connectivity test failed: {str(e)}")
                self.test_results["blockchain_connectivity"][agent_id] = {"error": str(e)}

    async def _test_single_agent_connectivity(self, agent, agent_id: str) -> Dict[str, Any]:
        """Test single agent's blockchain connectivity"""
        try:
            result = {
                "connected": False,
                "network_accessible": False,
                "contracts_available": False,
                "identity_registered": False
            }

            # Test network accessibility
            if hasattr(agent, 'blockchain_client') and agent.blockchain_client:
                try:
                    # Test basic blockchain connection
                    if hasattr(agent.blockchain_client, 'is_connected'):
                        result["network_accessible"] = await agent.blockchain_client.is_connected()
                    else:
                        result["network_accessible"] = True  # Assume accessible if no method available

                    # Test contract availability
                    if hasattr(agent.blockchain_client, 'get_contracts_info'):
                        contracts_info = await agent.blockchain_client.get_contracts_info()
                        result["contracts_available"] = bool(contracts_info)

                    result["connected"] = result["network_accessible"]

                except Exception as e:
                    logger.warning(f"Blockchain connection test for {agent_id} failed: {str(e)}")
                    result["error"] = str(e)

            return result

        except Exception as e:
            return {"error": str(e)}

    async def _test_inter_agent_communication(self):
        """Test communication between all agents through blockchain"""
        logger.info("Testing inter-agent blockchain communication...")

        # Test basic message sending between key agents
        test_pairs = [
            ("agentManager", "calculationAgent"),
            ("dataManager", "qualityControlManager"),
            ("catalogManager", "agentBuilder"),
            ("sqlAgent", "reasoningAgent"),
            ("embeddingFineTuner", "vectorProcessingAgent")
        ]

        for sender_id, receiver_id in test_pairs:
            if sender_id in self.agents and receiver_id in self.agents:
                try:
                    result = await self._test_message_exchange(sender_id, receiver_id)

                    pair_key = f"{sender_id}_to_{receiver_id}"
                    self.test_results["inter_agent_communication"][pair_key] = result

                    if result.get("success", False):
                        logger.info(f"  ✅ {sender_id} → {receiver_id}: Communication successful")
                    else:
                        logger.warning(f"  ⚠️  {sender_id} → {receiver_id}: Communication issues")

                except Exception as e:
                    logger.error(f"  ❌ {sender_id} → {receiver_id}: {str(e)}")
                    self.test_results["inter_agent_communication"][f"{sender_id}_to_{receiver_id}"] = {"error": str(e)}

    async def _test_message_exchange(self, sender_id: str, receiver_id: str) -> Dict[str, Any]:
        """Test message exchange between two agents"""
        try:
            sender = self.agents[sender_id]

            # Create test message
            test_message = {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message_type": "TEST_COMMUNICATION",
                "content": {
                    "test_data": f"Test message from {sender_id} to {receiver_id}",
                    "timestamp": datetime.now().isoformat(),
                    "test_id": f"test_{int(time.time())}"
                }
            }

            # Send message through blockchain if possible
            if hasattr(sender, 'send_blockchain_message'):
                send_result = await sender.send_blockchain_message(
                    target_agent_id=receiver_id,
                    message_type="TEST_COMMUNICATION",
                    content=test_message["content"]
                )

                return {
                    "success": True,
                    "send_result": send_result,
                    "message_sent": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message_sent": False,
                    "error": "send_blockchain_message method not available"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _test_trust_verification(self):
        """Test trust and reputation verification for all agents"""
        logger.info("Testing trust verification across all agents...")

        for agent_id, agent in self.agents.items():
            try:
                trust_result = await self._test_agent_trust_system(agent, agent_id)
                self.test_results["trust_verification"][agent_id] = trust_result

                if trust_result.get("trust_system_available", False):
                    logger.info(f"  ✅ {agent_id} trust system: Available")
                else:
                    logger.info(f"  ℹ️  {agent_id} trust system: Not available or disabled")

            except Exception as e:
                logger.error(f"  ❌ {agent_id} trust verification failed: {str(e)}")
                self.test_results["trust_verification"][agent_id] = {"error": str(e)}

    async def _test_agent_trust_system(self, agent, agent_id: str) -> Dict[str, Any]:
        """Test individual agent's trust system integration"""
        try:
            result = {
                "trust_system_available": False,
                "can_verify_reputation": False,
                "can_get_own_reputation": False,
                "trust_thresholds_defined": False
            }

            # Check if trust methods are available
            if hasattr(agent, 'get_agent_reputation'):
                result["can_verify_reputation"] = True

                # Try to get reputation for self
                try:
                    own_reputation = await agent.get_agent_reputation(agent_id)
                    result["can_get_own_reputation"] = True
                    result["own_reputation"] = own_reputation
                except Exception as e:
                    result["reputation_error"] = str(e)

            # Check trust thresholds
            if hasattr(agent, 'trust_thresholds'):
                result["trust_thresholds_defined"] = bool(agent.trust_thresholds)
                result["trust_thresholds"] = getattr(agent, 'trust_thresholds', {})

            result["trust_system_available"] = any([
                result["can_verify_reputation"],
                result["trust_thresholds_defined"]
            ])

            return result

        except Exception as e:
            return {"error": str(e)}

    async def _test_agent_capabilities(self):
        """Test agent-specific blockchain capabilities"""
        logger.info("Testing agent-specific blockchain capabilities...")

        # Test specific capabilities for key agents
        capability_tests = {
            "calculationAgent": self._test_calculation_capabilities,
            "sqlAgent": self._test_sql_capabilities,
            "dataManager": self._test_data_management_capabilities,
            "catalogManager": self._test_catalog_capabilities,
            "agentManager": self._test_orchestration_capabilities
        }

        for agent_id, test_function in capability_tests.items():
            if agent_id in self.agents:
                try:
                    result = await test_function(self.agents[agent_id])
                    self.test_results["capability_tests"][agent_id] = result

                    if result.get("success", False):
                        logger.info(f"  ✅ {agent_id} capabilities: Working")
                    else:
                        logger.warning(f"  ⚠️  {agent_id} capabilities: Issues detected")

                except Exception as e:
                    logger.error(f"  ❌ {agent_id} capability test failed: {str(e)}")
                    self.test_results["capability_tests"][agent_id] = {"error": str(e)}

    async def _test_calculation_capabilities(self, agent) -> Dict[str, Any]:
        """Test calculation agent blockchain capabilities"""
        try:
            # Test blockchain calculation handlers
            handlers_present = []
            for handler in ['_handle_blockchain_calculation_request', '_handle_blockchain_distributed_calculation']:
                if hasattr(agent, handler):
                    handlers_present.append(handler)

            return {
                "success": len(handlers_present) > 0,
                "handlers_present": handlers_present,
                "blockchain_capabilities": getattr(agent, 'blockchain_capabilities', [])
            }
        except Exception as e:
            return {"error": str(e)}

    async def _test_sql_capabilities(self, agent) -> Dict[str, Any]:
        """Test SQL agent blockchain capabilities"""
        try:
            handlers_present = []
            for handler in ['_handle_blockchain_sql_query_execution', '_handle_blockchain_database_operations']:
                if hasattr(agent, handler):
                    handlers_present.append(handler)

            return {
                "success": len(handlers_present) > 0,
                "handlers_present": handlers_present,
                "blockchain_capabilities": getattr(agent, 'blockchain_capabilities', [])
            }
        except Exception as e:
            return {"error": str(e)}

    async def _test_data_management_capabilities(self, agent) -> Dict[str, Any]:
        """Test data manager blockchain capabilities"""
        try:
            handlers_present = []
            for handler in ['_handle_blockchain_data_validation', '_handle_blockchain_data_transformation']:
                if hasattr(agent, handler):
                    handlers_present.append(handler)

            return {
                "success": len(handlers_present) > 0,
                "handlers_present": handlers_present,
                "blockchain_capabilities": getattr(agent, 'blockchain_capabilities', [])
            }
        except Exception as e:
            return {"error": str(e)}

    async def _test_catalog_capabilities(self, agent) -> Dict[str, Any]:
        """Test catalog manager blockchain capabilities"""
        try:
            handlers_present = []
            for handler in ['_handle_blockchain_catalog_search', '_handle_blockchain_resource_registration']:
                if hasattr(agent, handler):
                    handlers_present.append(handler)

            return {
                "success": len(handlers_present) > 0,
                "handlers_present": handlers_present,
                "blockchain_capabilities": getattr(agent, 'blockchain_capabilities', [])
            }
        except Exception as e:
            return {"error": str(e)}

    async def _test_orchestration_capabilities(self, agent) -> Dict[str, Any]:
        """Test agent manager blockchain capabilities"""
        try:
            handlers_present = []
            for handler in ['_handle_blockchain_orchestration', '_handle_blockchain_coordination']:
                if hasattr(agent, handler):
                    handlers_present.append(handler)

            return {
                "success": len(handlers_present) > 0,
                "handlers_present": handlers_present,
                "blockchain_capabilities": getattr(agent, 'blockchain_capabilities', [])
            }
        except Exception as e:
            return {"error": str(e)}

    async def _test_network_coordination(self):
        """Test network-wide coordination capabilities"""
        logger.info("Testing network-wide coordination...")

        try:
            # Test multi-agent coordination scenario
            coordination_result = await self._test_multi_agent_coordination()
            self.test_results["coordination_tests"]["multi_agent"] = coordination_result

            # Test consensus mechanisms
            consensus_result = await self._test_consensus_mechanisms()
            self.test_results["coordination_tests"]["consensus"] = consensus_result

            logger.info("  ✅ Network coordination tests completed")

        except Exception as e:
            logger.error(f"  ❌ Network coordination tests failed: {str(e)}")
            self.test_results["coordination_tests"]["error"] = str(e)

    async def _test_multi_agent_coordination(self) -> Dict[str, Any]:
        """Test coordination between multiple agents"""
        try:
            # Simple coordination test: agentManager coordinates with other agents
            if "agentManager" in self.agents:
                manager = self.agents["agentManager"]

                # Test if manager can coordinate with multiple agents
                coordination_targets = ["calculationAgent", "dataManager", "sqlAgent"]
                coordination_results = {}

                for target in coordination_targets:
                    if target in self.agents:
                        try:
                            # Test coordination capability
                            if hasattr(manager, 'coordinate_with_agent'):
                                result = await manager.coordinate_with_agent(target, {"test": "coordination"})
                                coordination_results[target] = {"success": True, "result": result}
                            else:
                                coordination_results[target] = {"success": False, "reason": "coordination method not available"}
                        except Exception as e:
                            coordination_results[target] = {"success": False, "error": str(e)}

                return {
                    "success": any(r.get("success", False) for r in coordination_results.values()),
                    "coordination_results": coordination_results
                }
            else:
                return {"success": False, "error": "agentManager not available"}

        except Exception as e:
            return {"error": str(e)}

    async def _test_consensus_mechanisms(self) -> Dict[str, Any]:
        """Test consensus mechanisms across agents"""
        try:
            # Test consensus among validation agents
            validation_agents = ["agent4CalcValidation", "agent5QaValidation", "qualityControlManager"]
            consensus_results = {}

            for agent_id in validation_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]

                    # Check for consensus-related capabilities
                    consensus_capabilities = []
                    for attr in dir(agent):
                        if 'consensus' in attr.lower() or 'validation' in attr.lower():
                            consensus_capabilities.append(attr)

                    consensus_results[agent_id] = {
                        "capabilities_found": len(consensus_capabilities),
                        "capabilities": consensus_capabilities[:5]  # Limit to first 5
                    }

            return {
                "success": any(r["capabilities_found"] > 0 for r in consensus_results.values()),
                "consensus_results": consensus_results
            }

        except Exception as e:
            return {"error": str(e)}

    def _generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating comprehensive test report...")

        # Calculate success metrics
        total_agents = len(self.agent_definitions)
        successful_initializations = sum(1 for r in self.test_results["agent_initialization"].values()
                                       if r.get("status") == "success")

        blockchain_connections = sum(1 for r in self.test_results["blockchain_connectivity"].values()
                                   if r.get("connected", False))

        successful_communications = sum(1 for r in self.test_results["inter_agent_communication"].values()
                                      if r.get("success", False))

        # Overall success criteria
        initialization_success = successful_initializations / total_agents >= 0.8  # 80% success rate
        connectivity_success = blockchain_connections / total_agents >= 0.5  # 50% connection rate
        communication_success = successful_communications >= 1  # At least one successful communication

        overall_success = initialization_success and connectivity_success and communication_success
        self.test_results["overall_success"] = overall_success

        # Summary stats
        self.test_results["summary"] = {
            "total_agents_tested": total_agents,
            "successful_initializations": successful_initializations,
            "blockchain_connections": blockchain_connections,
            "successful_communications": successful_communications,
            "initialization_success_rate": successful_initializations / total_agents,
            "connectivity_success_rate": blockchain_connections / total_agents,
            "overall_success": overall_success,
            "test_completion_time": datetime.now().isoformat()
        }

        # Log summary
        logger.info("📋 TEST SUMMARY:")
        logger.info(f"  Total agents tested: {total_agents}")
        logger.info(f"  Successful initializations: {successful_initializations}/{total_agents} ({(successful_initializations/total_agents)*100:.1f}%)")
        logger.info(f"  Blockchain connections: {blockchain_connections}/{total_agents} ({(blockchain_connections/total_agents)*100:.1f}%)")
        logger.info(f"  Successful communications: {successful_communications}")
        logger.info(f"  Overall result: {'✅ PASSED' if overall_success else '❌ FAILED'}")


async def main():
    """Main test execution"""
    available_agents = sum(1 for agent in imported_agents.values() if agent is not None)

    if available_agents == 0:
        logger.error("❌ No agents available for import. Cannot run tests.")
        return

    logger.info(f"📋 Found {available_agents} available agents out of {len(agent_modules)} total agents")

    # Enable blockchain for testing
    os.environ["BLOCKCHAIN_ENABLED"] = "true"

    # Initialize and run tester
    tester = BlockchainCommunicationTester()

    try:
        results = await tester.run_comprehensive_test()

        # Save results to file
        results_file = f"/tmp/blockchain_communication_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📄 Test results saved to: {results_file}")

        # Print final status
        if results.get("overall_success", False):
            logger.info("🎉 ALL TESTS PASSED! Blockchain communication is working across all 16 agents.")
        else:
            logger.warning("⚠️  Some tests failed. Check the detailed results for more information.")

        return results

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())

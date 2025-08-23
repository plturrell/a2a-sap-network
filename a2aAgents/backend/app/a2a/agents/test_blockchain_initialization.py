#!/usr/bin/env python3
"""
Proper Blockchain Initialization Test for All A2A Agents

This test properly initializes each agent with their correct parameters
and verifies blockchain integration step by step.
"""

import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Enable blockchain for testing
os.environ["BLOCKCHAIN_ENABLED"] = "true"

class BlockchainInitializationTester:
    """Test proper initialization and blockchain integration for all agents"""
    
    def __init__(self):
        self.test_results = {
            "agent_imports": {},
            "agent_initializations": {},
            "blockchain_integrations": {},
            "summary": {}
        }
        
        # Agent definitions with correct initialization patterns
        self.agent_specs = {
            "agentManager": {
                "module_path": "agentManager.active.enhancedAgentManagerAgent",
                "class_name": "EnhancedAgentManagerAgent",
                "init_pattern": "no_args",  # __init__(self)
                "base_url": None
            },
            "qualityControlManager": {
                "module_path": "agent6QualityControl.active.qualityControlManagerAgent", 
                "class_name": "QualityControlManagerAgent",
                "init_pattern": "base_url_only",  # __init__(self, base_url)
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "dataManager": {
                "module_path": "dataManager.active.enhancedDataManagerAgentSdk",
                "class_name": "EnhancedDataManagerAgentSDK", 
                "init_pattern": "base_url_only",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "agent4CalcValidation": {
                "module_path": "agent4CalcValidation.active.enhancedCalcValidationAgentSdk",
                "class_name": "EnhancedCalcValidationAgent",
                "init_pattern": "base_url_config",  # __init__(self, base_url, config=None)
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "agent5QaValidation": {
                "module_path": "agent5QaValidation.active.enhancedQaValidationAgentSdk",
                "class_name": "EnhancedQAValidationAgent",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "calculationAgent": {
                "module_path": "calculationAgent.active.enhancedCalculationAgentSdk",
                "class_name": "EnhancedCalculationAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "catalogManager": {
                "module_path": "catalogManager.active.enhancedCatalogManagerAgentSdk",
                "class_name": "EnhancedCatalogManagerAgent",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "agentBuilder": {
                "module_path": "agentBuilder.active.enhancedAgentBuilderAgentSdk",
                "class_name": "EnhancedAgentBuilderAgent",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "embeddingFineTuner": {
                "module_path": "embeddingFineTuner.active.enhancedEmbeddingFineTunerAgentSdk",
                "class_name": "EnhancedEmbeddingFineTunerAgent",
                "init_pattern": "base_url_config", 
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "sqlAgent": {
                "module_path": "sqlAgent.active.enhancedSqlAgentSdk",
                "class_name": "EnhancedSqlAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "reasoningAgent": {
                "module_path": "reasoningAgent.enhancedReasoningAgent",
                "class_name": "EnhancedReasoningAgent",
                "init_pattern": "no_args",
                "base_url": None
            },
            "dataProductAgent": {
                "module_path": "agent0DataProduct.active.enhancedDataProductAgentSdk",
                "class_name": "EnhancedDataProductAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "standardizationAgent": {
                "module_path": "agent1Standardization.active.enhancedDataStandardizationAgentSdk",
                "class_name": "EnhancedDataStandardizationAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "aiPreparationAgent": {
                "module_path": "agent2AiPreparation.active.aiPreparationAgentSdk",
                "class_name": "AIPreparationAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            },
            "vectorProcessingAgent": {
                "module_path": "agent3VectorProcessing.active.vectorProcessingAgentSdk",
                "class_name": "VectorProcessingAgentSDK",
                "init_pattern": "base_url_config",
                "base_url": os.getenv("A2A_SERVICE_URL")
            }
        }
        
        self.initialized_agents = {}
    
    async def run_full_test(self) -> Dict[str, Any]:
        """Run complete initialization and blockchain test"""
        logger.info("🚀 Starting Full Agent Initialization and Blockchain Test")
        
        try:
            # Step 1: Import all agent classes
            logger.info("📦 Step 1: Importing all agent classes...")
            await self._import_all_agents()
            
            # Step 2: Initialize each agent properly
            logger.info("🔧 Step 2: Initializing all agents...")
            await self._initialize_all_agents()
            
            # Step 3: Test blockchain integration for each agent
            logger.info("⛓️  Step 3: Testing blockchain integration...")
            await self._test_blockchain_integration()
            
            # Step 4: Generate summary report
            logger.info("📊 Step 4: Generating summary report...")
            self._generate_summary()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def _import_all_agents(self):
        """Import all agent classes with proper error handling"""
        logger.info("Importing agent classes...")
        
        for agent_id, spec in self.agent_specs.items():
            try:
                logger.info(f"  Importing {agent_id}...")
                
                module_path = spec["module_path"]
                class_name = spec["class_name"]
                
                # Import the module
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                
                self.test_results["agent_imports"][agent_id] = {
                    "status": "success",
                    "class": agent_class,
                    "module_path": module_path,
                    "class_name": class_name
                }
                
                logger.info(f"  ✅ {agent_id} imported successfully")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to import {agent_id}: {str(e)}")
                self.test_results["agent_imports"][agent_id] = {
                    "status": "failed",
                    "error": str(e),
                    "module_path": spec["module_path"],
                    "class_name": spec["class_name"]
                }
    
    async def _initialize_all_agents(self):
        """Initialize all successfully imported agents"""
        logger.info("Initializing agents with proper parameters...")
        
        for agent_id, import_result in self.test_results["agent_imports"].items():
            if import_result["status"] != "success":
                logger.info(f"  Skipping {agent_id} (import failed)")
                continue
                
            try:
                logger.info(f"  Initializing {agent_id}...")
                
                agent_class = import_result["class"]
                spec = self.agent_specs[agent_id]
                init_pattern = spec["init_pattern"]
                base_url = spec["base_url"]
                
                # Initialize based on the pattern
                if init_pattern == "no_args":
                    agent = agent_class()
                elif init_pattern == "base_url_only":
                    agent = agent_class(base_url)
                elif init_pattern == "base_url_config":
                    # Create basic config for agents that need it
                    config = {
                        "capabilities": {},
                        "skills": [],
                        "blockchain_enabled": True
                    }
                    agent = agent_class(base_url, config)
                else:
                    raise ValueError(f"Unknown initialization pattern: {init_pattern}")
                
                # Call initialize if available
                if hasattr(agent, 'initialize'):
                    logger.info(f"    Calling initialize() for {agent_id}...")
                    await agent.initialize()
                    logger.info(f"    ✅ {agent_id} initialized successfully")
                else:
                    logger.info(f"    ℹ️  {agent_id} has no initialize() method")
                
                # Store the initialized agent
                self.initialized_agents[agent_id] = agent
                
                self.test_results["agent_initializations"][agent_id] = {
                    "status": "success",
                    "init_pattern": init_pattern,
                    "base_url": base_url,
                    "has_initialize": hasattr(agent, 'initialize'),
                    "initialized_at": datetime.now().isoformat()
                }
                
                logger.info(f"  ✅ {agent_id} initialization complete")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to initialize {agent_id}: {str(e)}")
                self.test_results["agent_initializations"][agent_id] = {
                    "status": "failed", 
                    "error": str(e),
                    "init_pattern": spec["init_pattern"],
                    "base_url": spec["base_url"]
                }
    
    async def _test_blockchain_integration(self):
        """Test blockchain integration for each initialized agent"""
        logger.info("Testing blockchain integration for initialized agents...")
        
        for agent_id, agent in self.initialized_agents.items():
            try:
                logger.info(f"  Testing blockchain integration for {agent_id}...")
                
                blockchain_test = await self._test_agent_blockchain(agent, agent_id)
                self.test_results["blockchain_integrations"][agent_id] = blockchain_test
                
                if blockchain_test.get("blockchain_integration_present", False):
                    logger.info(f"  ✅ {agent_id} has blockchain integration")
                else:
                    logger.warning(f"  ⚠️  {agent_id} missing blockchain integration")
                    
            except Exception as e:
                logger.error(f"  ❌ Blockchain test failed for {agent_id}: {str(e)}")
                self.test_results["blockchain_integrations"][agent_id] = {
                    "status": "failed",
                    "error": str(e)
                }
    
    async def _test_agent_blockchain(self, agent, agent_id: str) -> Dict[str, Any]:
        """Test individual agent's blockchain integration"""
        try:
            blockchain_test = {
                "status": "success",
                "blockchain_integration_present": False,
                "blockchain_mixin": False,
                "blockchain_client": False,
                "blockchain_capabilities": False,
                "blockchain_handlers": [],
                "agent_identity": False,
                "trust_thresholds": False,
                "initialization_methods": []
            }
            
            # Check for BlockchainIntegrationMixin
            blockchain_test["blockchain_mixin"] = hasattr(agent, 'blockchain_client')
            
            # Check for blockchain client
            if hasattr(agent, 'blockchain_client'):
                blockchain_test["blockchain_client"] = agent.blockchain_client is not None
            
            # Check for blockchain capabilities
            if hasattr(agent, 'blockchain_capabilities'):
                blockchain_test["blockchain_capabilities"] = bool(agent.blockchain_capabilities)
                blockchain_test["capabilities_list"] = getattr(agent, 'blockchain_capabilities', [])
            
            # Check for agent identity
            if hasattr(agent, 'agent_identity'):
                blockchain_test["agent_identity"] = agent.agent_identity is not None
            
            # Check for trust thresholds
            if hasattr(agent, 'trust_thresholds'):
                blockchain_test["trust_thresholds"] = bool(agent.trust_thresholds)
            
            # Find blockchain message handlers
            for attr_name in dir(agent):
                if attr_name.startswith('_handle_blockchain_'):
                    blockchain_test["blockchain_handlers"].append(attr_name)
            
            # Find blockchain initialization methods
            for attr_name in dir(agent):
                if 'blockchain' in attr_name.lower() and 'init' in attr_name.lower():
                    blockchain_test["initialization_methods"].append(attr_name)
            
            # Overall blockchain integration assessment
            blockchain_test["blockchain_integration_present"] = any([
                blockchain_test["blockchain_mixin"],
                len(blockchain_test["blockchain_handlers"]) > 0,
                blockchain_test["blockchain_capabilities"]
            ])
            
            return blockchain_test
            
        except Exception as e:
            return {
                "status": "failed", 
                "error": str(e)
            }
    
    def _generate_summary(self):
        """Generate comprehensive summary report"""
        logger.info("Generating comprehensive summary...")
        
        # Calculate metrics
        total_agents = len(self.agent_specs)
        imported_agents = sum(1 for r in self.test_results["agent_imports"].values() 
                             if r["status"] == "success")
        initialized_agents = sum(1 for r in self.test_results["agent_initializations"].values() 
                               if r["status"] == "success")
        blockchain_integrated = sum(1 for r in self.test_results["blockchain_integrations"].values() 
                                  if r.get("blockchain_integration_present", False))
        
        # Generate summary
        self.test_results["summary"] = {
            "total_agents": total_agents,
            "successful_imports": imported_agents,
            "successful_initializations": initialized_agents,
            "blockchain_integrated": blockchain_integrated,
            "import_success_rate": imported_agents / total_agents,
            "initialization_success_rate": initialized_agents / total_agents,
            "blockchain_integration_rate": blockchain_integrated / total_agents,
            "test_completion_time": datetime.now().isoformat()
        }
        
        # Detailed breakdown by agent
        agent_breakdown = {}
        for agent_id in self.agent_specs.keys():
            agent_breakdown[agent_id] = {
                "imported": self.test_results["agent_imports"].get(agent_id, {}).get("status") == "success",
                "initialized": self.test_results["agent_initializations"].get(agent_id, {}).get("status") == "success",
                "blockchain_integrated": self.test_results["blockchain_integrations"].get(agent_id, {}).get("blockchain_integration_present", False),
                "blockchain_handlers": len(self.test_results["blockchain_integrations"].get(agent_id, {}).get("blockchain_handlers", [])),
                "capabilities_defined": self.test_results["blockchain_integrations"].get(agent_id, {}).get("blockchain_capabilities", False)
            }
        
        self.test_results["agent_breakdown"] = agent_breakdown
        
        # Log summary
        logger.info("📋 INITIALIZATION & BLOCKCHAIN INTEGRATION SUMMARY:")
        logger.info(f"  Total agents: {total_agents}")
        logger.info(f"  Successful imports: {imported_agents}/{total_agents} ({(imported_agents/total_agents)*100:.1f}%)")
        logger.info(f"  Successful initializations: {initialized_agents}/{total_agents} ({(initialized_agents/total_agents)*100:.1f}%)")
        logger.info(f"  Blockchain integrated: {blockchain_integrated}/{total_agents} ({(blockchain_integrated/total_agents)*100:.1f}%)")
        
        logger.info("\n📊 AGENT-BY-AGENT BREAKDOWN:")
        for agent_id, breakdown in agent_breakdown.items():
            status_indicators = []
            if breakdown["imported"]:
                status_indicators.append("📦 Imported")
            if breakdown["initialized"]: 
                status_indicators.append("🔧 Initialized")
            if breakdown["blockchain_integrated"]:
                status_indicators.append(f"⛓️  Blockchain ({breakdown['blockchain_handlers']} handlers)")
            
            status_str = " | ".join(status_indicators) if status_indicators else "❌ Failed"
            logger.info(f"  {agent_id}: {status_str}")


async def main():
    """Main test execution"""
    logger.info("🚀 Starting comprehensive agent initialization and blockchain integration test")
    
    # Initialize tester
    tester = BlockchainInitializationTester()
    
    try:
        # Run the full test
        results = await tester.run_full_test()
        
        # Save results
        results_file = f"/tmp/agent_initialization_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            # Convert agent classes to string for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📄 Test results saved to: {results_file}")
        
        # Final assessment
        summary = results.get("summary", {})
        initialization_rate = summary.get("initialization_success_rate", 0)
        blockchain_rate = summary.get("blockchain_integration_rate", 0)
        
        if initialization_rate >= 0.8 and blockchain_rate >= 0.8:
            logger.info("🎉 EXCELLENT! Most agents initialized and blockchain-integrated successfully!")
        elif initialization_rate >= 0.5 and blockchain_rate >= 0.5:
            logger.info("✅ GOOD! Majority of agents working with blockchain integration.")
        else:
            logger.warning("⚠️  NEEDS WORK! Some agents need fixing for proper blockchain integration.")
            
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
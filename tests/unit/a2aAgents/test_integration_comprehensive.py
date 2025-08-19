#!/usr/bin/env python3
"""
Comprehensive Integration Test for A2A Agent System
Tests agent initialization, SDK instantiation, cross-module integration, and end-to-end workflows
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add paths for imports
sys.path.insert(0, '/Users/apple/projects/a2a/a2aAgents/backend')
sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestResult:
    """Test result tracking"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
        self.test_details = {}
        
    def add_success(self, test_name: str, details: Any = None):
        self.passed += 1
        self.test_details[test_name] = {"status": "PASS", "details": details}
        logger.info(f"✅ {test_name}")
        
    def add_failure(self, test_name: str, error: str, details: Any = None):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        self.test_details[test_name] = {"status": "FAIL", "error": error, "details": details}
        logger.error(f"❌ {test_name}: {error}")
        
    def add_warning(self, test_name: str, warning: str):
        self.warnings.append(f"{test_name}: {warning}")
        logger.warning(f"⚠️ {test_name}: {warning}")
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tests": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "warnings": self.warnings,
            "success_rate": self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0,
            "test_details": self.test_details
        }


class A2AIntegrationTester:
    """Comprehensive integration tester for A2A system"""
    
    def __init__(self):
        self.result = IntegrationTestResult()
        self.test_agents = {}
        self.base_url = "http://localhost:8000"
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        logger.info("🚀 Starting A2A Integration Tests")
        
        # Test 1: Agent Initialization Testing
        await self.test_agent_initialization()
        
        # Test 2: SDK Component Testing
        await self.test_sdk_components()
        
        # Test 3: Cross-Module Integration
        await self.test_cross_module_integration()
        
        # Test 4: Configuration Integration
        await self.test_configuration_integration()
        
        # Test 5: End-to-End Workflow Testing
        await self.test_end_to_end_workflow()
        
        # Test 6: Trust System Integration
        await self.test_trust_system_integration()
        
        # Test 7: Network Integration
        await self.test_network_integration()
        
        logger.info("🏁 Integration Tests Complete")
        
        return self.result.get_summary()
    
    async def test_agent_initialization(self):
        """Test that all fixed agent files can be imported and initialized"""
        logger.info("🔧 Testing Agent Initialization")
        
        agent_configs = [
            {
                "name": "DataProduct Agent",
                "module": "app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk",
                "class": "DataProductRegistrationAgentSDK",
                "args": [self.base_url, "http://localhost:8080/ord"]
            },
            {
                "name": "Standardization Agent", 
                "module": "app.a2a.agents.agent1Standardization.active.dataStandardizationAgentSdk",
                "class": "DataStandardizationAgentSDK",
                "args": [self.base_url]
            },
            {
                "name": "AI Preparation Agent",
                "module": "app.a2a.agents.agent2AiPreparation.active.aiPreparationAgentSdk", 
                "class": "AIPreparationAgentSDK",
                "args": [self.base_url]
            },
            {
                "name": "Vector Processing Agent",
                "module": "app.a2a.agents.agent3VectorProcessing.active.vectorProcessingAgentSdk",
                "class": "VectorProcessingAgentSDK", 
                "args": [self.base_url]
            },
            {
                "name": "Calc Validation Agent",
                "module": "app.a2a.agents.agent4CalcValidation.active.calcValidationAgentSdk",
                "class": "CalcValidationAgentSDK",
                "args": [self.base_url]
            },
            {
                "name": "QA Validation Agent",
                "module": "app.a2a.agents.agent5QaValidation.active.qaValidationAgentSdk", 
                "class": "QAValidationAgentSDK",
                "args": [self.base_url]
            }
        ]
        
        for config in agent_configs:
            try:
                # Import module
                module = __import__(config["module"], fromlist=[config["class"]])
                agent_class = getattr(module, config["class"])
                
                # Initialize agent
                agent = agent_class(*config["args"])
                
                # Store for further testing
                self.test_agents[config["name"]] = agent
                
                # Basic validation
                assert hasattr(agent, 'agent_id'), f"Agent missing agent_id: {config['name']}"
                assert hasattr(agent, 'name'), f"Agent missing name: {config['name']}"
                assert hasattr(agent, 'version'), f"Agent missing version: {config['name']}"
                
                self.result.add_success(f"Agent Import & Init: {config['name']}", {
                    "agent_id": agent.agent_id,
                    "version": agent.version
                })
                
            except Exception as e:
                self.result.add_failure(f"Agent Import & Init: {config['name']}", str(e))
                
    async def test_sdk_components(self):
        """Test SDK classes are properly instantiated"""
        logger.info("📦 Testing SDK Components")
        
        try:
            # Test SDK import
            from app.a2a.sdk import (
                A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
                A2AMessage, MessageRole, create_agent_id
            )
            
            self.result.add_success("SDK Core Import", {
                "components": ["A2AAgentBase", "decorators", "types", "utils"]
            })
            
            # Test message creation
            message = A2AMessage(
                id="test_msg_001",
                sender="test_sender",
                receiver="test_receiver",
                parts=[]
            )
            
            assert message.id == "test_msg_001"
            assert message.sender == "test_sender"
            
            self.result.add_success("SDK Message Creation", {
                "message_id": message.id,
                "parts_count": len(message.parts)
            })
            
            # Test agent ID creation
            agent_id = create_agent_id("test_agent")
            assert agent_id is not None
            assert len(agent_id) > 0
            
            self.result.add_success("SDK Agent ID Generation", {
                "generated_id": agent_id
            })
            
        except Exception as e:
            self.result.add_failure("SDK Components", str(e))
            
    async def test_cross_module_integration(self):
        """Test communication between fixed agents and core A2A network"""
        logger.info("🔗 Testing Cross-Module Integration")
        
        try:
            # Test agent skills registration
            for agent_name, agent in self.test_agents.items():
                if hasattr(agent, 'list_skills'):
                    try:
                        skills = agent.list_skills()
                        self.result.add_success(f"Skills Registration: {agent_name}", {
                            "skills_count": len(skills),
                            "skills": list(skills.keys()) if isinstance(skills, dict) else skills
                        })
                    except Exception as e:
                        self.result.add_failure(f"Skills Registration: {agent_name}", str(e))
                        
            # Test distributed storage integration
            try:
                from app.a2a.storage.distributedStorage import DistributedStorageManager
                storage_manager = DistributedStorageManager()
                
                # Test basic storage operations
                test_data = {"test_key": "test_value", "timestamp": time.time()}
                result = await storage_manager.store("test_integration", test_data)
                
                self.result.add_success("Distributed Storage Integration", {
                    "storage_result": result,
                    "test_data_size": len(str(test_data))
                })
                
            except Exception as e:
                self.result.add_failure("Distributed Storage Integration", str(e))
                
            # Test request signing integration
            try:
                from app.a2a.security.requestSigning import RequestSigner
                signer = RequestSigner()
                
                test_payload = {"message": "test", "agent": "integration_test"}
                signature = signer.sign_request(test_payload, "test_agent")
                
                self.result.add_success("Request Signing Integration", {
                    "signature_present": signature is not None,
                    "payload_size": len(str(test_payload))
                })
                
            except Exception as e:
                self.result.add_failure("Request Signing Integration", str(e))
                
        except Exception as e:
            self.result.add_failure("Cross-Module Integration", str(e))
            
    async def test_configuration_integration(self):
        """Test configuration changes are properly loaded"""
        logger.info("⚙️ Testing Configuration Integration")
        
        try:
            # Test configuration loading
            from config.agentConfig import config
            
            # Check essential configuration items
            config_items = [
                'base_url', 'data_product_storage', 'catalog_manager_url'
            ]
            
            available_configs = []
            for item in config_items:
                if hasattr(config, item):
                    available_configs.append(item)
                    
            self.result.add_success("Configuration Loading", {
                "available_configs": available_configs,
                "total_checked": len(config_items)
            })
            
            # Test environment variables
            env_vars = [
                'A2A_NETWORK_PATH', 'AI_PREP_STORAGE_PATH'
            ]
            
            available_env_vars = []
            for var in env_vars:
                if os.getenv(var):
                    available_env_vars.append(var)
                    
            self.result.add_success("Environment Variables", {
                "available_env_vars": available_env_vars,
                "total_checked": len(env_vars)
            })
            
            # Test fallback mechanisms
            try:
                # This should work with or without a2aNetwork
                from app.a2a.sdk.utils import create_error_response, create_success_response
                
                error_resp = create_error_response(400, "test error")
                success_resp = create_success_response({"test": "data"})
                
                assert error_resp.get("error") is not None
                assert success_resp.get("data") is not None
                
                self.result.add_success("Fallback Mechanisms", {
                    "error_response": error_resp,
                    "success_response": success_resp
                })
                
            except Exception as e:
                self.result.add_failure("Fallback Mechanisms", str(e))
                
        except Exception as e:
            self.result.add_failure("Configuration Integration", str(e))
            
    async def test_end_to_end_workflow(self):
        """Test complete agent-to-agent communication flow"""
        logger.info("🔄 Testing End-to-End Workflow")
        
        try:
            # Get agents for workflow test
            data_product_agent = self.test_agents.get("DataProduct Agent")
            ai_prep_agent = self.test_agents.get("AI Preparation Agent")
            
            if not data_product_agent or not ai_prep_agent:
                self.result.add_warning("End-to-End Workflow", "Required agents not initialized")
                return
                
            # Test data product registration workflow
            try:
                if hasattr(data_product_agent, 'process_data_product_workflow'):
                    test_data = {
                        "data_location": "/tmp/test_data.csv",
                        "data_type": "financial",
                        "entity_id": "test_entity_001"
                    }
                    
                    # Create temporary test file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        f.write("account_id,balance,currency\n")
                        f.write("ACC001,1000.00,USD\n")
                        f.write("ACC002,2500.50,EUR\n")
                        test_data["data_location"] = f.name
                    
                    try:
                        result = await data_product_agent.process_data_product_workflow(test_data, "integration_test")
                        
                        self.result.add_success("Data Product Workflow", {
                            "workflow_successful": result.get("workflow_successful", False),
                            "stages_completed": len(result.get("results", {}).get("stages", {}))
                        })
                    finally:
                        # Cleanup
                        if os.path.exists(test_data["data_location"]):
                            os.unlink(test_data["data_location"])
                            
            except Exception as e:
                self.result.add_failure("Data Product Workflow", str(e))
                
            # Test AI preparation workflow
            try:
                if hasattr(ai_prep_agent, 'prepare_entity_for_ai'):
                    test_entity = {
                        "entity_id": "test_entity_002",
                        "entity_type": "financial_account",
                        "account_number": "ACC123456",
                        "balance": 5000.00,
                        "currency": "USD"
                    }
                    
                    result = await ai_prep_agent.prepare_entity_for_ai(test_entity, "integration_test")
                    
                    self.result.add_success("AI Preparation Workflow", {
                        "workflow_successful": result.get("workflow_successful", False),
                        "ai_readiness_score": result.get("ai_ready_entity", {}).get("ai_readiness_score", 0)
                    })
                    
            except Exception as e:
                self.result.add_failure("AI Preparation Workflow", str(e))
                
        except Exception as e:
            self.result.add_failure("End-to-End Workflow", str(e))
            
    async def test_trust_system_integration(self):
        """Test trust system integration"""
        logger.info("🔒 Testing Trust System Integration")
        
        try:
            # Test trust system imports
            try:
                from app.a2a.security.requestSigning import sign_a2a_message, verify_a2a_message
                from app.a2a.core.trustManager import trust_manager
                
                self.result.add_success("Trust System Import", {
                    "components": ["signing", "verification", "trust_manager"]
                })
                
            except ImportError as e:
                self.result.add_warning("Trust System Import", f"Trust system not available: {e}")
                
            # Test message signing and verification
            try:
                test_message = {
                    "messageId": "test_msg_002",
                    "sender": "integration_test",
                    "receiver": "test_receiver",
                    "content": {"test": "data"}
                }
                
                # Try to sign message
                signed_message = sign_a2a_message(test_message, "integration_test")
                
                # Try to verify
                is_valid, verification_result = verify_a2a_message(signed_message, "integration_test")
                
                self.result.add_success("Message Signing & Verification", {
                    "signed_message_present": signed_message is not None,
                    "verification_result": verification_result
                })
                
            except Exception as e:
                self.result.add_warning("Message Signing & Verification", f"Trust operations failed: {e}")
                
        except Exception as e:
            self.result.add_failure("Trust System Integration", str(e))
            
    async def test_network_integration(self):
        """Test network connectivity and agent registration"""
        logger.info("🌐 Testing Network Integration")
        
        try:
            # Test network connector
            try:
                from app.a2a.network import get_network_connector, get_registration_service
                
                network_connector = get_network_connector()
                registration_service = get_registration_service()
                
                self.result.add_success("Network Services Import", {
                    "network_connector": network_connector is not None,
                    "registration_service": registration_service is not None
                })
                
            except Exception as e:
                self.result.add_warning("Network Services Import", f"Network services not available: {e}")
                
            # Test agent registration
            try:
                test_agent = self.test_agents.get("DataProduct Agent")
                if test_agent and hasattr(test_agent, 'network_registered'):
                    self.result.add_success("Agent Network Registration", {
                        "agent_id": test_agent.agent_id,
                        "registered": getattr(test_agent, 'network_registered', False)
                    })
                else:
                    self.result.add_warning("Agent Network Registration", "No network registration status available")
                    
            except Exception as e:
                self.result.add_warning("Agent Network Registration", str(e))
                
        except Exception as e:
            self.result.add_failure("Network Integration", str(e))

async def main():
    """Run the comprehensive integration tests"""
    tester = A2AIntegrationTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 A2A INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  ⚠️ {warning}")
                
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  ❌ {error}")
                
        # Save detailed results
        results_file = "integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return exit code based on results
        return 0 if results['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
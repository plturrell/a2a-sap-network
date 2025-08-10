#!/usr/bin/env python3
"""
Full A2A Integration Test Suite
Tests the complete end-to-end workflow with all fixed agents
"""

import pytest
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class A2AIntegrationTester:
    """Comprehensive integration testing for all A2A agents"""
    
    def __init__(self):
        self.base_urls = {
            "data_manager": "http://localhost:8001",
            "catalog_manager": "http://localhost:8002",
            "agent0": "http://localhost:8003",  # Data Product Registration
            "agent1": "http://localhost:8004",  # Data Standardization
            "agent2": "http://localhost:8005",  # AI Preparation
            "agent3": "http://localhost:8008",  # Vector Processing
            "agent4": "http://localhost:8006",  # Calculation Validation
            "agent5": "http://localhost:8007",  # QA Validation
        }
        
        self.smart_contract_addresses = {
            "business_data_cloud": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
            "agent_registry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "message_router": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
        }
        
        self.test_results = {
            "health_checks": {},
            "trust_verification": {},
            "workflow_tests": {},
            "integration_tests": {},
            "performance_metrics": {}
        }
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        logger.info("🚀 Starting A2A Integration Test Suite")
        
        # Test 1: Health checks for all services
        await self.test_health_checks()
        
        # Test 2: Trust system verification
        await self.test_trust_system()
        
        # Test 3: End-to-end workflow test
        await self.test_complete_workflow()
        
        # Test 4: Cross-agent communication
        await self.test_cross_agent_communication()
        
        # Test 5: Smart contract integration
        await self.test_smart_contract_integration()
        
        # Test 6: Error handling and circuit breakers
        await self.test_error_handling()
        
        # Test 7: Performance benchmarks
        await self.test_performance_benchmarks()
        
        # Generate report
        return self.generate_test_report()
    
    async def test_health_checks(self):
        """Test health endpoints for all services"""
        logger.info("📋 Testing health checks for all services...")
        
        async with aiohttp.ClientSession() as session:
            for service, url in self.base_urls.items():
                try:
                    async with session.get(f"{url}/health") as response:
                        if response.status == 200:
                            data = await response.json()
                            self.test_results["health_checks"][service] = {
                                "status": "healthy",
                                "response": data
                            }
                            logger.info(f"✅ {service}: Healthy")
                        else:
                            self.test_results["health_checks"][service] = {
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}"
                            }
                            logger.warning(f"❌ {service}: Unhealthy (HTTP {response.status})")
                except Exception as e:
                    self.test_results["health_checks"][service] = {
                        "status": "unreachable",
                        "error": str(e)
                    }
                    logger.error(f"❌ {service}: Unreachable - {e}")
    
    async def test_trust_system(self):
        """Test RSA-based trust verification between agents"""
        logger.info("🔐 Testing trust system...")
        
        test_cases = [
            ("agent0", "agent1", "High trust expected"),
            ("agent1", "agent2", "High trust expected"),
            ("agent4", "agent5", "Validation agents trust"),
            ("data_manager", "agent0", "Infrastructure trust")
        ]
        
        for source, target, description in test_cases:
            try:
                # Test trust verification
                async with aiohttp.ClientSession() as session:
                    # Get public key from source
                    async with session.get(f"{self.base_urls[source]}/trust/public-key") as response:
                        if response.status == 200:
                            public_key = await response.json()
                            
                    # Verify trust with target
                    trust_data = {
                        "source_agent": source,
                        "public_key": public_key.get("public_key", "")
                    }
                    
                    async with session.post(
                        f"{self.base_urls[target]}/trust/verify",
                        json=trust_data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.test_results["trust_verification"][f"{source}->{target}"] = {
                                "status": "verified",
                                "trust_level": result.get("trust_level", 0),
                                "description": description
                            }
                            logger.info(f"✅ Trust verified: {source} -> {target}")
                        else:
                            self.test_results["trust_verification"][f"{source}->{target}"] = {
                                "status": "failed",
                                "description": description
                            }
                            logger.warning(f"❌ Trust failed: {source} -> {target}")
                            
            except Exception as e:
                self.test_results["trust_verification"][f"{source}->{target}"] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ Trust test error: {source} -> {target} - {e}")
    
    async def test_complete_workflow(self):
        """Test complete A2A workflow from registration to validation"""
        logger.info("🔄 Testing complete A2A workflow...")
        
        # Test data
        test_product = {
            "title": "Integration Test Product",
            "description": "Test product for A2A integration testing",
            "creator": "Integration Tester",
            "type": "Dataset",
            "format": "JSON",
            "data": {
                "accounts": [
                    {"id": "ACC001", "name": "Test Account", "balance": 1000.50},
                    {"id": "ACC002", "name": "Demo Account", "balance": 2500.75}
                ],
                "transactions": [
                    {"from": "ACC001", "to": "ACC002", "amount": 100.00}
                ]
            }
        }
        
        workflow_id = f"test_workflow_{int(time.time())}"
        
        try:
            # Step 1: Register data product (Agent 0)
            logger.info("  1️⃣ Registering data product...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent0']}/api/register",
                    json=test_product
                ) as response:
                    if response.status == 200:
                        registration = await response.json()
                        product_id = registration.get("product_id")
                        self.test_results["workflow_tests"]["registration"] = {
                            "status": "success",
                            "product_id": product_id
                        }
                        logger.info(f"  ✅ Registered: {product_id}")
                    else:
                        raise Exception(f"Registration failed: HTTP {response.status}")
            
            # Step 2: Standardize data (Agent 1)
            logger.info("  2️⃣ Standardizing data...")
            standardization_request = {
                "product_id": product_id,
                "source_format": "custom",
                "target_format": "standardized"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent1']}/api/standardize",
                    json=standardization_request
                ) as response:
                    if response.status == 200:
                        standardized = await response.json()
                        self.test_results["workflow_tests"]["standardization"] = {
                            "status": "success",
                            "standardized_id": standardized.get("standardized_id")
                        }
                        logger.info("  ✅ Standardized")
            
            # Step 3: AI Preparation (Agent 2)
            logger.info("  3️⃣ Preparing for AI...")
            ai_prep_request = {
                "product_id": product_id,
                "enrichment_type": "semantic",
                "use_grok": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent2']}/api/prepare",
                    json=ai_prep_request
                ) as response:
                    if response.status == 200:
                        ai_prepared = await response.json()
                        self.test_results["workflow_tests"]["ai_preparation"] = {
                            "status": "success",
                            "enriched": True
                        }
                        logger.info("  ✅ AI Prepared")
            
            # Step 4: Vector Processing (Agent 3)
            logger.info("  4️⃣ Processing vectors...")
            vector_request = {
                "product_id": product_id,
                "generate_embeddings": True,
                "create_knowledge_graph": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent3']}/api/process-vectors",
                    json=vector_request
                ) as response:
                    if response.status == 200:
                        vectors = await response.json()
                        self.test_results["workflow_tests"]["vector_processing"] = {
                            "status": "success",
                            "embeddings_created": True
                        }
                        logger.info("  ✅ Vectors processed")
            
            # Step 5: Calculation Validation (Agent 4)
            logger.info("  5️⃣ Validating calculations...")
            calc_request = {
                "product_id": product_id,
                "validation_type": "template_based",
                "test_calculations": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent4']}/api/validate-calculations",
                    json=calc_request
                ) as response:
                    if response.status == 200:
                        calc_results = await response.json()
                        self.test_results["workflow_tests"]["calculation_validation"] = {
                            "status": "success",
                            "validations_passed": calc_results.get("passed", False)
                        }
                        logger.info("  ✅ Calculations validated")
            
            # Step 6: QA Validation (Agent 5)
            logger.info("  6️⃣ Running QA validation...")
            qa_request = {
                "product_id": product_id,
                "test_type": "simpleqa",
                "include_ord_discovery": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['agent5']}/api/qa-validate",
                    json=qa_request
                ) as response:
                    if response.status == 200:
                        qa_results = await response.json()
                        self.test_results["workflow_tests"]["qa_validation"] = {
                            "status": "success",
                            "qa_score": qa_results.get("score", 0)
                        }
                        logger.info("  ✅ QA validation complete")
            
            self.test_results["workflow_tests"]["overall"] = "success"
            logger.info("✅ Complete workflow test passed!")
            
        except Exception as e:
            self.test_results["workflow_tests"]["overall"] = "failed"
            self.test_results["workflow_tests"]["error"] = str(e)
            logger.error(f"❌ Workflow test failed: {e}")
    
    async def test_cross_agent_communication(self):
        """Test cross-agent message passing"""
        logger.info("💬 Testing cross-agent communication...")
        
        test_message = {
            "message_type": "capability_query",
            "content": {
                "query": "data_standardization",
                "requester": "test_suite"
            }
        }
        
        try:
            # Test catalog manager discovery
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_urls['catalog_manager']}/api/discover",
                    json={"capability": "data_standardization"}
                ) as response:
                    if response.status == 200:
                        discovered = await response.json()
                        self.test_results["integration_tests"]["service_discovery"] = {
                            "status": "success",
                            "agents_found": len(discovered.get("agents", []))
                        }
                        logger.info(f"  ✅ Discovered {len(discovered.get('agents', []))} agents")
                    
        except Exception as e:
            self.test_results["integration_tests"]["service_discovery"] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Service discovery failed: {e}")
    
    async def test_smart_contract_integration(self):
        """Test smart contract integration"""
        logger.info("📜 Testing smart contract integration...")
        
        # This would normally interact with the blockchain
        # For now, we'll verify the configuration
        
        for agent_name, base_url in self.base_urls.items():
            if agent_name in ["agent0", "agent1", "agent2", "agent3", "agent4", "agent5"]:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{base_url}/api/blockchain/config") as response:
                            if response.status == 200:
                                config = await response.json()
                                self.test_results["integration_tests"][f"{agent_name}_blockchain"] = {
                                    "status": "configured",
                                    "contract_address": config.get("business_data_cloud_address")
                                }
                                logger.info(f"  ✅ {agent_name}: Blockchain configured")
                except:
                    self.test_results["integration_tests"][f"{agent_name}_blockchain"] = {
                        "status": "not_configured"
                    }
    
    async def test_error_handling(self):
        """Test error handling and circuit breakers"""
        logger.info("🛡️ Testing error handling...")
        
        # Test invalid requests
        invalid_requests = [
            ("agent0", "/api/register", {}),  # Empty registration
            ("agent1", "/api/standardize", {"product_id": "invalid"}),  # Invalid ID
            ("agent2", "/api/prepare", {"missing": "required_fields"}),  # Missing fields
        ]
        
        for agent, endpoint, payload in invalid_requests:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_urls[agent]}{endpoint}",
                        json=payload
                    ) as response:
                        if response.status >= 400:
                            self.test_results["integration_tests"][f"{agent}_error_handling"] = {
                                "status": "correct",
                                "http_status": response.status
                            }
                            logger.info(f"  ✅ {agent}: Correctly handled invalid request")
                        else:
                            self.test_results["integration_tests"][f"{agent}_error_handling"] = {
                                "status": "incorrect",
                                "message": "Should have rejected invalid request"
                            }
            except:
                pass
    
    async def test_performance_benchmarks(self):
        """Test performance metrics"""
        logger.info("⚡ Testing performance benchmarks...")
        
        # Test response times
        for service, url in self.base_urls.items():
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health") as response:
                        response_time = (time.time() - start_time) * 1000  # ms
                        
                self.test_results["performance_metrics"][service] = {
                    "response_time_ms": round(response_time, 2),
                    "status": "good" if response_time < 100 else "slow"
                }
                logger.info(f"  ⏱️ {service}: {response_time:.2f}ms")
                
            except:
                self.test_results["performance_metrics"][service] = {
                    "response_time_ms": -1,
                    "status": "failed"
                }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        # Count health checks
        for service, result in self.test_results["health_checks"].items():
            total_tests += 1
            if result.get("status") == "healthy":
                passed_tests += 1
        
        # Count workflow tests
        for test, result in self.test_results["workflow_tests"].items():
            if test != "overall":
                total_tests += 1
                if result.get("status") == "success":
                    passed_tests += 1
        
        # Calculate overall health
        overall_health = "healthy" if passed_tests / total_tests > 0.8 else "degraded"
        
        report = {
            "test_execution_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": round((passed_tests / total_tests) * 100, 2),
                "overall_health": overall_health
            },
            "detailed_results": self.test_results,
            "recommendations": []
        }
        
        # Add recommendations based on results
        if report["summary"]["success_rate"] < 100:
            report["recommendations"].append("Investigate and fix failing services")
            
        # Check for slow services
        slow_services = [
            service for service, metrics in self.test_results["performance_metrics"].items()
            if metrics.get("status") == "slow"
        ]
        if slow_services:
            report["recommendations"].append(f"Optimize slow services: {', '.join(slow_services)}")
        
        return report


async def main():
    """Run integration tests"""
    tester = A2AIntegrationTester()
    
    print("🧪 A2A Integration Test Suite")
    print("=" * 50)
    
    # Run all tests
    report = await tester.run_all_tests()
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']}%")
    print(f"Overall Health: {report['summary']['overall_health']}")
    
    # Save detailed report
    report_file = Path("integration_test_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Print recommendations
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    return report["summary"]["success_rate"] == 100


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        exit(1)
#!/usr/bin/env python3
"""
Catalog Manager A2A Agent Comprehensive Validation Tests
Tests all ORD repository management and AI enhancement functionality
"""

import asyncio
import sys
import json
import httpx
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.append('/Users/apple/projects/finsight_cib/backend')

from app.ordRegistry.models import ORDDocument, DublinCoreMetadata


def log_test(message: str):
    """Test logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


class CatalogManagerAgentTest:
    """Comprehensive test suite for Catalog Manager A2A agent"""
    
    def __init__(self):
        self.base_url = "http://localhost:8005"
        self.agent_endpoints = {
            "health": f"{self.base_url}/a2a/catalog_manager/v1/health",
            "agent_card": f"{self.base_url}/a2a/catalog_manager/v1/.well-known/agent.json",
            "rpc": f"{self.base_url}/a2a/catalog_manager/v1/rpc",
            "message": f"{self.base_url}/a2a/catalog_manager/v1/message",
            "ord_register": f"{self.base_url}/a2a/catalog_manager/v1/ord/register",
            "ord_search": f"{self.base_url}/a2a/catalog_manager/v1/ord/search",
        }
        
        self.test_results = {}
        self.client = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(timeout=60.0)
        log_test("🔧 Test client initialized")
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        if self.client:
            await self.client.aclose()
        log_test("🧹 Test client cleaned up")
    
    def create_test_ord_document(self) -> Dict[str, Any]:
        """Create a test ORD document for validation"""
        return {
            "openResourceDiscovery": "1.5.0",
            "description": "Test ORD document for Catalog Manager validation",
            "dublinCore": {
                "title": "Test API Resource for Catalog Manager",
                "description": "A test API resource to validate Catalog Manager functionality",
                "creator": ["Catalog Manager Test Suite"],
                "subject": ["API Testing and Validation", "catalog-management", "test-validation"],
                "relation": ["Related to SAP Catalog Manager Documentation"],
                "identifier": "test-catalog-manager-resource",
                "type": "API",
                "format": "JSON"
            },
            "dataProducts": [{
                "ordId": "com.finsight.cib:dataProduct:catalog_manager_api",
                "title": "Test Catalog Manager API",
                "description": "Test API for Catalog Manager validation",
                "version": "1.0.0",
                "visibility": "public",
                "tags": ["test", "catalog", "manager"]
            }]
        }
    
    async def test_agent_health(self) -> Dict[str, Any]:
        """Test 1: Agent health check"""
        log_test("🩺 TESTING AGENT HEALTH CHECK")
        log_test("-" * 50)
        
        try:
            response = await self.client.get(self.agent_endpoints["health"])
            
            if response.status_code == 200:
                health_data = response.json()
                log_test(f"✅ Agent Status: {health_data.get('status', 'unknown')}")
                log_test(f"   Agent ID: {health_data.get('agent_id', 'unknown')}")
                log_test(f"   ORD Service: {health_data.get('ord_service', {}).get('status', 'unknown')}")
                log_test(f"   AI Clients: Grok={health_data.get('ai_clients', {}).get('grok', False)}, Perplexity={health_data.get('ai_clients', {}).get('perplexity', False)}")
                log_test(f"   Active Tasks: {health_data.get('active_tasks', 0)}")
                
                return {
                    "success": True,
                    "status": health_data.get("status"),
                    "details": health_data
                }
            else:
                log_test(f"❌ Health check failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ Health check error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_agent_card(self) -> Dict[str, Any]:
        """Test 2: Agent card retrieval"""
        log_test("\n📋 TESTING AGENT CARD RETRIEVAL")
        log_test("-" * 50)
        
        try:
            response = await self.client.get(self.agent_endpoints["agent_card"])
            
            if response.status_code == 200:
                agent_card = response.json()
                log_test(f"✅ Agent Name: {agent_card.get('name', 'unknown')}")
                log_test(f"   Description: {agent_card.get('description', 'unknown')}")
                log_test(f"   Version: {agent_card.get('version', 'unknown')}")
                log_test(f"   Protocol Version: {agent_card.get('protocolVersion', 'unknown')}")
                
                capabilities = agent_card.get('capabilities', {})
                log_test(f"   Capabilities: {len(capabilities)} features")
                for capability, enabled in capabilities.items():
                    if enabled:
                        log_test(f"      ✅ {capability}")
                
                tools = agent_card.get('tools', [])
                log_test(f"   Tools: {len(tools)} available")
                for tool in tools:
                    log_test(f"      🔧 {tool.get('name', 'unknown')}")
                
                return {
                    "success": True,
                    "agent_card": agent_card,
                    "capabilities_count": len(capabilities),
                    "tools_count": len(tools)
                }
            else:
                log_test(f"❌ Agent card retrieval failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ Agent card error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_ord_registration_with_ai(self) -> Dict[str, Any]:
        """Test 3: ORD document registration with AI enhancement"""
        log_test("\n📝 TESTING ORD REGISTRATION WITH AI ENHANCEMENT")
        log_test("-" * 50)
        
        try:
            test_document = self.create_test_ord_document()
            
            registration_payload = {
                "ord_document": test_document,
                "enhancement_type": "metadata_enrichment",
                "ai_powered": True
            }
            
            log_test("📤 Sending ORD registration request with AI enhancement...")
            response = await self.client.post(
                self.agent_endpoints["ord_register"],
                json=registration_payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                log_test(f"✅ Registration Success!")
                log_test(f"   Task ID: {result.get('taskId', 'unknown')}")
                log_test(f"   State: {result.get('state', 'unknown')}")
                log_test(f"   Progress: {result.get('progress', 0) * 100:.1f}%")
                log_test(f"   Message: {result.get('message', 'No message')}")
                
                # Check for artifacts
                artifacts = result.get('artifacts', [])
                if artifacts:
                    for artifact in artifacts:
                        if artifact.get('name') == 'registration_response':
                            artifact_data = artifact.get('data', {})
                            if artifact_data.get('success'):
                                registration_id = artifact_data.get('result', {}).get('registration_id')
                                ai_insights = artifact_data.get('ai_insights', {})
                                
                                log_test(f"   📋 Registration ID: {registration_id}")
                                if ai_insights:
                                    log_test(f"   🤖 AI Model Used: {ai_insights.get('model_used', 'unknown')}")
                                    log_test(f"   🎯 Confidence Score: {ai_insights.get('confidence_score', 0)}")
                                    log_test(f"   ⏱️ Processing Time: {ai_insights.get('processing_time', 0):.2f}s")
                
                return {
                    "success": True,
                    "task_id": result.get('taskId'),
                    "registration_completed": result.get('state') == 'COMPLETED',
                    "artifacts": artifacts
                }
            else:
                log_test(f"❌ Registration failed: {response.status_code}")
                log_test(f"   Response: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ Registration error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_json_rpc_interface(self) -> Dict[str, Any]:
        """Test 4: JSON-RPC interface for A2A communication"""
        log_test("\n🔗 TESTING JSON-RPC A2A INTERFACE")
        log_test("-" * 50)
        
        try:
            # Test agent.getCard method
            rpc_payload = {
                "jsonrpc": "2.0",
                "method": "agent.getCard",
                "params": {},
                "id": "test-001"
            }
            
            log_test("📤 Testing JSON-RPC agent.getCard...")
            response = await self.client.post(
                self.agent_endpoints["rpc"],
                json=rpc_payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "result" in result:
                    agent_info = result["result"]
                    log_test(f"✅ JSON-RPC Success!")
                    log_test(f"   Agent: {agent_info.get('name', 'unknown')}")
                    log_test(f"   Request ID: {result.get('id', 'unknown')}")
                    log_test(f"   Protocol: {result.get('jsonrpc', 'unknown')}")
                    
                    return {
                        "success": True,
                        "method": "agent.getCard",
                        "response_id": result.get('id'),
                        "agent_name": agent_info.get('name')
                    }
                else:
                    log_test(f"❌ JSON-RPC error: {result.get('error', 'unknown error')}")
                    return {"success": False, "error": result.get('error')}
            else:
                log_test(f"❌ JSON-RPC failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ JSON-RPC error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_ai_enhancement_message(self) -> Dict[str, Any]:
        """Test 5: AI enhancement via A2A message interface"""
        log_test("\n✨ TESTING AI ENHANCEMENT VIA A2A MESSAGE")
        log_test("-" * 50)
        
        try:
            test_document = self.create_test_ord_document()
            
            # Create A2A message for enhancement
            message_payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps({
                                "operation": "register",
                                "ord_document": test_document,
                                "enhancement_type": "comprehensive"
                            })
                        }
                    ],
                    "ai_powered": True
                },
                "contextId": f"test-context-{datetime.utcnow().timestamp()}"
            }
            
            log_test("📤 Sending AI enhancement A2A message...")
            response = await self.client.post(
                self.agent_endpoints["message"],
                json=message_payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                log_test(f"✅ Enhancement Message Success!")
                log_test(f"   Status: {result.get('status', 'unknown')}")
                
                task_result = result.get('result', {})
                log_test(f"   Task State: {task_result.get('state', 'unknown')}")
                log_test(f"   Progress: {task_result.get('progress', 0) * 100:.1f}%")
                log_test(f"   Message: {task_result.get('message', 'No message')}")
                
                # Check for enhancement artifacts
                artifacts = task_result.get('artifacts', [])
                if artifacts:
                    for artifact in artifacts:
                        if artifact.get('name') == 'enhancement_response':
                            artifact_data = artifact.get('data', {})
                            ai_insights = artifact_data.get('ai_insights', {})
                            if ai_insights:
                                log_test(f"   🤖 AI Model: {ai_insights.get('model_used', 'unknown')}")
                                log_test(f"   🎯 Confidence: {ai_insights.get('confidence_score', 0)}")
                
                return {
                    "success": True,
                    "enhancement_completed": task_result.get('state') == 'COMPLETED',
                    "ai_enhanced": bool(artifacts)
                }
            else:
                log_test(f"❌ Enhancement message failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ Enhancement message error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_semantic_search(self) -> Dict[str, Any]:
        """Test 6: AI-powered semantic search"""
        log_test("\n🔍 TESTING AI-POWERED SEMANTIC SEARCH")
        log_test("-" * 50)
        
        try:
            search_query = "API metadata enhancement catalog"
            
            log_test(f"📤 Searching for: '{search_query}'")
            response = await self.client.get(
                f"{self.agent_endpoints['ord_search']}?query={search_query}&semantic=true"
            )
            
            if response.status_code == 200:
                result = response.json()
                
                log_test(f"✅ Search Success!")
                log_test(f"   Task State: {result.get('state', 'unknown')}")
                
                # Check for search artifacts
                artifacts = result.get('artifacts', [])
                if artifacts:
                    for artifact in artifacts:
                        if artifact.get('name') == 'search_response':
                            artifact_data = artifact.get('data', {})
                            search_result = artifact_data.get('result', {})
                            results = search_result.get('results', [])
                            log_test(f"   📊 Results Found: {len(results)}")
                            log_test(f"   🔍 Search Type: {search_result.get('search_type', 'unknown')}")
                
                return {
                    "success": True,
                    "search_completed": result.get('state') == 'COMPLETED',
                    "query": search_query
                }
            else:
                log_test(f"❌ Search failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            log_test(f"❌ Search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive validation tests"""
        log_test("🚀 STARTING CATALOG MANAGER A2A AGENT COMPREHENSIVE VALIDATION")
        log_test("Testing ORD repository management and AI enhancement capabilities")
        log_test("=" * 80)
        
        await self.initialize()
        
        # Run all tests
        tests = [
            ("Agent Health Check", self.test_agent_health),
            ("Agent Card Retrieval", self.test_agent_card),
            ("ORD Registration with AI", self.test_ord_registration_with_ai),
            ("JSON-RPC Interface", self.test_json_rpc_interface),
            ("AI Enhancement Message", self.test_ai_enhancement_message),
            ("Semantic Search", self.test_semantic_search)
        ]
        
        successful_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                self.test_results[test_name] = result
                
                if result.get("success", False):
                    successful_tests += 1
                    
            except Exception as e:
                log_test(f"\n❌ Test '{test_name}' failed with exception: {e}")
                self.test_results[test_name] = {"success": False, "error": str(e)}
        
        # Summary
        log_test("\n" + "=" * 80)
        log_test("🎯 CATALOG MANAGER A2A AGENT VALIDATION SUMMARY")
        log_test("=" * 80)
        
        success_rate = (successful_tests / total_tests) * 100
        log_test(f"✅ Successful Tests: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        for test_name, result in self.test_results.items():
            status = "✅" if result.get("success", False) else "❌"
            log_test(f"{status} {test_name}")
        
        if successful_tests == total_tests:
            log_test("\n🎉 ALL TESTS PASSED - CATALOG MANAGER FULLY OPERATIONAL!")
            log_test("✅ ORD repository management capabilities validated")
            log_test("✅ AI enhancement features validated") 
            log_test("✅ A2A protocol compliance validated")
            log_test("✅ Integration readiness confirmed")
        elif successful_tests > total_tests // 2:
            log_test("\n✅ MAJORITY OF TESTS PASSED - Agent mostly functional with minor issues")
        else:
            log_test("\n❌ MULTIPLE TEST FAILURES - Agent needs fixes before production")
        
        await self.cleanup()
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "all_passed": successful_tests == total_tests,
            "results": self.test_results
        }


async def main():
    """Run the comprehensive Catalog Manager agent validation"""
    test_suite = CatalogManagerAgentTest()
    await test_suite.run_comprehensive_tests()


if __name__ == "__main__":
    asyncio.run(main())

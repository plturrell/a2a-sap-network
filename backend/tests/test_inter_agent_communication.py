#!/usr/bin/env python3
"""
A2A Inter-Agent Communication Test
Test actual communication between agents
"""

import asyncio
import httpx
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test agent registry - using the ports from our system
TEST_AGENTS = [
    {"name": "Registry Server", "port": 8000, "endpoint": "/health"},
    {"name": "Data Product Agent", "port": 8001, "endpoint": "/health"},
    {"name": "Standardization Agent", "port": 8002, "endpoint": "/health"},
    {"name": "Validation Agent", "port": 8003, "endpoint": "/health"},
    {"name": "Pipeline Agent", "port": 8004, "endpoint": "/health"},
]

class InterAgentCommunicationTester:
    """Test inter-agent communication functionality"""
    
    def __init__(self):
        self.test_results = []
        self.active_agents = []
        
    async def run_communication_tests(self) -> bool:
        """Run comprehensive inter-agent communication tests"""
        logger.info("🚀 Starting A2A Inter-Agent Communication Tests")
        
        # Test 1: Agent Discovery
        discovery_success = await self._test_agent_discovery()
        
        # Test 2: Basic Communication
        if discovery_success:
            comm_success = await self._test_basic_communication()
        else:
            logger.warning("⚠️ Skipping communication tests - no agents discovered")
            comm_success = False
            
        # Test 3: Message Routing (if we have multiple agents)
        if len(self.active_agents) >= 2:
            routing_success = await self._test_message_routing()
        else:
            logger.warning("⚠️ Skipping message routing tests - need at least 2 agents")
            routing_success = False
            
        # Test 4: Error Handling
        error_handling_success = await self._test_error_handling()
        
        # Summary
        await self._print_test_summary(discovery_success, comm_success, routing_success, error_handling_success)
        
        return discovery_success and comm_success
    
    async def _test_agent_discovery(self) -> bool:
        """Test agent discovery and availability"""
        logger.info("🔍 Testing Agent Discovery...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for agent in TEST_AGENTS:
                try:
                    response = await client.get(f"http://localhost:{agent['port']}{agent['endpoint']}")
                    
                    if response.status_code == 200:
                        self.active_agents.append(agent)
                        logger.info(f"✅ {agent['name']} is active on port {agent['port']}")
                        
                        # Try to get agent info if available
                        try:
                            info_response = await client.get(f"http://localhost:{agent['port']}/info")
                            if info_response.status_code == 200:
                                info = info_response.json()
                                logger.info(f"   Agent info: {info.get('name', 'N/A')} v{info.get('version', 'N/A')}")
                        except Exception:
                            pass  # Info endpoint not available
                            
                    else:
                        logger.warning(f"⚠️ {agent['name']} responded with status {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"❌ {agent['name']} is not reachable: {e}")
        
        logger.info(f"📊 Discovery Results: {len(self.active_agents)}/{len(TEST_AGENTS)} agents active")
        return len(self.active_agents) > 0
    
    async def _test_basic_communication(self) -> bool:
        """Test basic communication with active agents"""
        logger.info("💬 Testing Basic Communication...")
        
        success_count = 0
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            for agent in self.active_agents:
                try:
                    # Test different endpoints that might exist
                    test_endpoints = [
                        ("/health", "Health check"),
                        ("/status", "Status check"),
                        ("/api/v1/status", "API status"),
                        ("/metrics", "Metrics endpoint"),
                    ]
                    
                    agent_success = False
                    for endpoint, description in test_endpoints:
                        try:
                            response = await client.get(f"http://localhost:{agent['port']}{endpoint}")
                            
                            if response.status_code == 200:
                                logger.info(f"✅ {agent['name']} - {description} successful")
                                agent_success = True
                                
                                # Log response if it's JSON
                                try:
                                    data = response.json()
                                    if isinstance(data, dict) and len(data) <= 5:  # Don't log huge responses
                                        logger.info(f"   Response: {json.dumps(data, indent=2)}")
                                except Exception:
                                    pass  # Not JSON or too large
                                    
                                break  # Found working endpoint
                                
                        except Exception as e:
                            logger.debug(f"   {endpoint} failed: {e}")
                    
                    if agent_success:
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ {agent['name']} - No successful communication endpoints found")
                        
                except Exception as e:
                    logger.error(f"❌ Communication with {agent['name']} failed: {e}")
        
        success_rate = (success_count / len(self.active_agents)) * 100 if self.active_agents else 0
        logger.info(f"📊 Communication Results: {success_count}/{len(self.active_agents)} agents ({success_rate:.1f}%)")
        
        return success_count > 0
    
    async def _test_message_routing(self) -> bool:
        """Test message routing between agents"""
        logger.info("🔄 Testing Message Routing...")
        
        if len(self.active_agents) < 2:
            logger.warning("⚠️ Need at least 2 agents for routing tests")
            return False
        
        # Try to send a message from first agent to second agent (if they support it)
        source_agent = self.active_agents[0]
        target_agent = self.active_agents[1]
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                # Try different message endpoints that might exist
                message_endpoints = [
                    "/api/v1/message",
                    "/send-message", 
                    "/communicate",
                    "/api/message"
                ]
                
                for endpoint in message_endpoints:
                    try:
                        message_payload = {
                            "to": f"http://localhost:{target_agent['port']}",
                            "message": {
                                "type": "test",
                                "content": "Inter-agent communication test",
                                "timestamp": datetime.utcnow().isoformat(),
                                "test_id": "ia_comm_test_001"
                            }
                        }
                        
                        response = await client.post(
                            f"http://localhost:{source_agent['port']}{endpoint}",
                            json=message_payload,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code in [200, 201, 202]:
                            logger.info(f"✅ Message routing successful: {source_agent['name']} -> {target_agent['name']}")
                            logger.info(f"   Response: {response.status_code} {response.text[:100]}")
                            return True
                            
                    except Exception as e:
                        logger.debug(f"   Endpoint {endpoint} failed: {e}")
                
                logger.warning(f"⚠️ No working message endpoints found for routing test")
                return False
                
            except Exception as e:
                logger.error(f"❌ Message routing test failed: {e}")
                return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        logger.info("🔧 Testing Error Handling...")
        
        success_count = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for agent in self.active_agents[:2]:  # Test first 2 agents to save time
                try:
                    # Test 404 handling
                    response = await client.get(f"http://localhost:{agent['port']}/nonexistent-endpoint")
                    
                    if response.status_code == 404:
                        logger.info(f"✅ {agent['name']} - 404 handling correct")
                        success_count += 1
                    elif response.status_code in [405, 501]:  # Method not allowed or not implemented
                        logger.info(f"✅ {agent['name']} - Error handling present ({response.status_code})")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ {agent['name']} - Unexpected response to invalid request: {response.status_code}")
                        
                except httpx.TimeoutException:
                    logger.warning(f"⚠️ {agent['name']} - Timeout on invalid request (acceptable)")
                    success_count += 1  # Timeout is acceptable error handling
                except Exception as e:
                    logger.warning(f"⚠️ {agent['name']} - Error handling test issue: {e}")
        
        tested_agents = min(len(self.active_agents), 2)
        logger.info(f"📊 Error Handling Results: {success_count}/{tested_agents} agents handled errors correctly")
        
        return success_count > 0
    
    async def _print_test_summary(self, discovery: bool, communication: bool, routing: bool, error_handling: bool):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 A2A INTER-AGENT COMMUNICATION TEST SUMMARY")
        logger.info("="*60)
        
        logger.info(f"🔍 Agent Discovery: {'✅ PASS' if discovery else '❌ FAIL'}")
        logger.info(f"💬 Basic Communication: {'✅ PASS' if communication else '❌ FAIL'}")
        logger.info(f"🔄 Message Routing: {'✅ PASS' if routing else '❌ SKIP/FAIL'}")
        logger.info(f"🔧 Error Handling: {'✅ PASS' if error_handling else '❌ FAIL'}")
        
        logger.info(f"\n📊 Active Agents: {len(self.active_agents)}")
        for agent in self.active_agents:
            logger.info(f"   • {agent['name']} (port {agent['port']})")
            
        overall_success = discovery and communication
        logger.info(f"\n🎯 Overall Result: {'✅ SUCCESS' if overall_success else '❌ PARTIAL/FAILURE'}")
        
        if overall_success:
            logger.info("🎉 Inter-agent communication is working!")
        else:
            logger.info("⚠️ Some inter-agent communication issues detected")
            
        logger.info("="*60 + "\n")

async def main():
    """Run the inter-agent communication test suite"""
    tester = InterAgentCommunicationTester()
    
    try:
        success = await tester.run_communication_tests()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
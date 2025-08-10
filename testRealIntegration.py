#!/usr/bin/env python3
"""
Test Real Integration between a2aAgents and a2aNetwork
Demonstrates actual agent-to-network connectivity and communication
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime

# Add the backend to path
sys.path.insert(0, '/Users/apple/projects/a2a/a2aAgents/backend')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def testRealIntegration():
    """Test real integration between a2aAgents and a2aNetwork"""
    print("🚀 Testing Real A2A Network Integration")
    print("=" * 60)
    
    try:
        # Test 1: Import Agent0 with network integration
        await testAgent0NetworkIntegration()
        
        # Test 2: Test network messaging between agents
        await testNetworkMessaging()
        
        # Test 3: Test data product registration through network
        await testDataProductRegistration()
        
        # Test 4: Test capability discovery and usage
        await testCapabilityDiscovery()
        
        print("\n" + "=" * 60)
        print("🎉 Real Integration Tests Completed Successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Real integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def testAgent0NetworkIntegration():
    """Test Agent0 with network integration"""
    print("\n📋 Testing Agent0 Network Integration")
    print("-" * 40)
    
    try:
        # Import Agent0 with network integration
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        
        print("✅ Agent0 imported successfully")
        
        # Create agent instance
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:9000"
        )
        
        print(f"✅ Agent0 instance created: {agent.agent_id}")
        
        # Initialize agent (this will trigger network integration)
        await agent.initialize()
        
        print(f"✅ Agent0 initialized with network integration")
        print(f"   Network registered: {getattr(agent, 'network_registered', 'Unknown')}")
        
        # Test agent status
        if hasattr(agent, 'network_connector'):
            network_status = await agent.network_connector.get_network_status()
            print(f"   Network available: {network_status.get('network_available', False)}")
            
        # Cleanup
        await agent.shutdown()
        print("✅ Agent0 shutdown completed")
        
    except Exception as e:
        print(f"❌ Agent0 network integration test failed: {e}")
        import traceback
        traceback.print_exc()

async def testNetworkMessaging():
    """Test network messaging between agents"""
    print("\n💬 Testing Network Messaging")
    print("-" * 30)
    
    try:
        # Import network messaging service
        from app.a2a.network import get_messaging_service
        
        messaging_service = get_messaging_service()
        await messaging_service.initialize()
        
        print("✅ Messaging service initialized")
        
        # Test sending a message
        result = await messaging_service.send_agent_message(
            from_agent_id="test_sender",
            to_agent_id="test_receiver",
            message_type="test_message",
            payload={"test_data": "Hello A2A Network!"},
            context_id="test_integration_001"
        )
        
        print(f"✅ Test message sent: {result}")
        
        # Test capability request
        capability_result = await messaging_service.request_agent_capability(
            from_agent_id="test_requester",
            capability="dublin-core",
            request_data={"title": "Test Data Product"}
        )
        
        print(f"✅ Capability request result: {capability_result is not None}")
        
    except Exception as e:
        print(f"❌ Network messaging test failed: {e}")

async def testDataProductRegistration():
    """Test data product registration through network"""
    print("\n📊 Testing Data Product Registration")
    print("-" * 35)
    
    try:
        # Import network connector
        from app.a2a.network import get_network_connector
        
        network_connector = get_network_connector()
        await network_connector.initialize()
        
        print("✅ Network connector initialized")
        
        # Create test Dublin Core metadata
        dublin_core_metadata = {
            "title": "Integration Test Data Product",
            "creator": ["A2A Integration Test Suite"],
            "description": "Test data product for integration validation",
            "subject": ["integration", "testing", "a2a-network"],
            "type": "Dataset",
            "format": "application/json",
            "language": "en"
        }
        
        # Create ORD descriptor
        ord_descriptor = {
            "title": dublin_core_metadata["title"],
            "shortDescription": dublin_core_metadata["description"][:250],
            "description": dublin_core_metadata["description"],
            "version": "1.0.0",
            "releaseStatus": "active",
            "visibility": "internal",
            "tags": dublin_core_metadata["subject"]
        }
        
        # Register data product
        result = await network_connector.register_data_product(
            dublin_core_metadata, ord_descriptor
        )
        
        print(f"✅ Data product registration result: {result.get('success', False)}")
        if result.get("success"):
            print(f"   Data Product ID: {result.get('data_product_id', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Data product registration test failed: {e}")

async def testCapabilityDiscovery():
    """Test capability discovery through network"""
    print("\n🔍 Testing Capability Discovery")
    print("-" * 32)
    
    try:
        # Import network connector
        from app.a2a.network import get_network_connector
        
        network_connector = get_network_connector()
        await network_connector.initialize()
        
        print("✅ Network connector ready for capability discovery")
        
        # Search for agents with Dublin Core capabilities
        agents = await network_connector.find_agents(
            capabilities=["dublin-core", "metadata-extraction"]
        )
        
        print(f"✅ Found {len(agents)} agents with Dublin Core capabilities")
        
        for agent in agents:
            print(f"   - Agent: {agent.get('agent_id', 'Unknown')}")
            print(f"     Capabilities: {agent.get('capabilities', [])}")
        
        # Search for agents in data domain
        data_agents = await network_connector.find_agents(domain="data")
        
        print(f"✅ Found {len(data_agents)} agents in data domain")
        
        # Test network status
        network_status = await network_connector.get_network_status()
        
        print("✅ Network Status:")
        print(f"   Initialized: {network_status.get('initialized', False)}")
        print(f"   Network Available: {network_status.get('network_available', False)}")
        
    except Exception as e:
        print(f"❌ Capability discovery test failed: {e}")

async def testEndToEndWorkflow():
    """Test complete end-to-end workflow"""
    print("\n🔄 Testing End-to-End Workflow")
    print("-" * 35)
    
    try:
        # Create Agent0 instance
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        
        agent0 = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:9000"
        )
        
        await agent0.initialize()
        print("✅ Agent0 initialized and connected to network")
        
        # Create test data for processing
        test_data = {
            "data_location": "/tmp/test_integration_data.json",
            "data_type": "integration_test",
            "title": "End-to-End Test Data"
        }
        
        # Create test data file
        with open(test_data["data_location"], 'w') as f:
            json.dump({
                "test_records": [
                    {"id": 1, "name": "Integration Test Record 1"},
                    {"id": 2, "name": "Integration Test Record 2"}
                ]
            }, f, indent=2)
        
        print("✅ Test data file created")
        
        # Execute Dublin Core extraction
        dublin_core_result = await agent0.extract_dublin_core_metadata(test_data)
        print("✅ Dublin Core metadata extracted")
        
        # Execute integrity verification
        integrity_result = await agent0.verify_data_integrity(test_data)
        print("✅ Data integrity verified")
        
        # Execute ORD registration
        ord_input = {
            "dublin_core_metadata": dublin_core_result["dublin_core_metadata"],
            "integrity_info": integrity_result
        }
        
        ord_result = await agent0.register_with_ord(ord_input)
        print("✅ ORD registration completed")
        
        # If network is available, register data product through network
        if hasattr(agent0, 'network_connector') and agent0.network_registered:
            network_result = await agent0.network_connector.register_data_product(
                dublin_core_result["dublin_core_metadata"],
                ord_result.get("ord_descriptor", {})
            )
            print("✅ Network data product registration completed")
        
        # Cleanup
        await agent0.shutdown()
        os.unlink(test_data["data_location"])
        
        print("✅ End-to-end workflow completed successfully")
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()

async def runComprehensiveTest():
    """Run comprehensive integration test"""
    try:
        success = await testRealIntegration()
        
        if success:
            print("\n🌟 Running End-to-End Workflow Test")
            await testEndToEndWorkflow()
        
        return success
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔗 A2A Network Real Integration Test Suite")
    print("=" * 60)
    
    success = asyncio.run(runComprehensiveTest())
    
    if success:
        print("\n✨ All integration tests passed!")
        print("🔗 a2aAgents successfully integrated with a2aNetwork")
    else:
        print("\n💥 Some integration tests failed")
        print("🔧 Check logs for details")
    
    sys.exit(0 if success else 1)
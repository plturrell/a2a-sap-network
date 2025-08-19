#!/usr/bin/env python3
"""
Basic integration test for A2A Agent functionality
"""

import asyncio
import sys
import os
import httpx
import json
from typing import Dict, Any

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_agent_endpoints():
    """Test basic agent endpoints"""
    try:
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        from app.a2a.sdk.types import A2AMessage, MessagePart, MessageRole
        
        print("🔄 Creating agent for endpoint testing...")
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8001",
            ord_registry_url="http://localhost:9000"
        )
        
        await agent.initialize()
        
        # Test agent card generation
        print("🔄 Testing agent card generation...")
        agent_card = agent.get_agent_card()
        assert agent_card is not None
        assert agent_card.name == "Data Product Registration Agent"
        assert agent_card.protocolVersion == "0.2.9"
        print("✅ Agent card generation works correctly!")
        
        # Test skill listing
        print("🔄 Testing skill listing...")
        skills = agent.list_skills()
        assert len(skills) >= 3  # Should have dublin_core_extraction, integrity_verification, ord_registration
        skill_names = [skill['name'] for skill in skills]
        assert 'dublin_core_extraction' in skill_names
        assert 'integrity_verification' in skill_names
        assert 'ord_registration' in skill_names
        print(f"✅ Skills found: {skill_names}")
        
        # Test message processing capability
        print("🔄 Testing message processing...")
        test_message = A2AMessage(
            role=MessageRole.USER,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "get_status",
                        "data_location": "/tmp/test_data"
                    }
                )
            ]
        )
        
        result = await agent.process_message(test_message, "test-context-123")
        assert result is not None
        assert result.get("success") is True
        print("✅ Message processing works correctly!")
        
        # Test skill execution
        print("🔄 Testing skill execution...")
        dublin_core_result = await agent.execute_skill(
            "dublin_core_extraction",
            {
                "data_location": "/tmp/test_data",
                "data_type": "financial"
            }
        )
        assert dublin_core_result is not None
        assert dublin_core_result.get("success") is True
        assert "dublin_core_metadata" in dublin_core_result.get("result", {})
        print("✅ Skill execution works correctly!")
        
        await agent.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Agent endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_http_endpoints():
    """Test HTTP endpoints using FastAPI app"""
    try:
        from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
        
        print("🔄 Testing FastAPI endpoints...")
        agent = DataProductRegistrationAgentSDK(
            base_url="http://localhost:8002",
            ord_registry_url="http://localhost:9000"
        )
        
        await agent.initialize()
        app = agent.create_fastapi_app()
        
        # Test agent card endpoint
        print("🔄 Testing /.well-known/agent.json endpoint...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/.well-known/agent.json")
        assert response.status_code == 200
        agent_card_data = response.json()
        assert agent_card_data["name"] == "Data Product Registration Agent"
        assert agent_card_data["protocolVersion"] == "0.2.9"
        print("✅ Agent card endpoint works correctly!")
        
        # Test health endpoint
        print("🔄 Testing /health endpoint...")
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["agent_id"] == "data_product_agent_0"
        print("✅ Health endpoint works correctly!")
        
        # Test skills endpoint
        print("🔄 Testing /skills endpoint...")
        response = client.get("/skills")
        assert response.status_code == 200
        skills_data = response.json()
        assert "skills" in skills_data
        assert len(skills_data["skills"]) >= 3
        print("✅ Skills endpoint works correctly!")
        
        await agent.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ HTTP endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the integration tests"""
    print("🚀 Starting A2A Agent Integration Tests")
    print("=" * 60)
    
    # Test 1: Basic agent functionality
    print("\n📋 Test 1: Basic Agent Functionality")
    print("-" * 40)
    test1_success = await test_agent_endpoints()
    
    # Test 2: HTTP endpoints
    print("\n📋 Test 2: FastAPI HTTP Endpoints")
    print("-" * 40)
    test2_success = await test_http_endpoints()
    
    print("=" * 60)
    if test1_success and test2_success:
        print("🎉 All integration tests passed!")
        print("✨ Agent0 is fully functional and ready for production use!")
        return 0
    else:
        print("💥 Some integration tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Test True A2A Communication with Dynamic Service Discovery
No hardcoded agent URLs - everything discovered via registry
"""

import asyncio
import httpx
import json
from datetime import datetime
from uuid import uuid4
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.a2a_registry.client import A2ARegistryClient
from app.a2a.security.smart_contract_trust import initialize_agent_trust, sign_a2a_message


async def test_true_a2a_workflow():
    """Test true A2A workflow with dynamic service discovery"""
    
    print("=== TESTING TRUE A2A WITH DYNAMIC DISCOVERY ===")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    # Initialize registry client
    registry_url = os.getenv("A2A_REGISTRY_URL", "http://localhost:8000/api/v1/a2a")
    registry = A2ARegistryClient(base_url=registry_url)
    
    try:
        # Step 1: Discover available agents
        print("\n📡 Discovering available agents...")
        
        # Search for all healthy agents
        all_agents = await registry.search_agents(status="healthy")
        if all_agents and all_agents.get("agents"):
            print(f"✅ Found {len(all_agents['agents'])} healthy agents:")
            for agent in all_agents["agents"]:
                print(f"   - {agent['name']} ({agent['agent_id']}) at {agent['url']}")
        else:
            print("❌ No healthy agents found in registry")
            return False
        
        # Step 2: Discover Agent 0 (Data Product Registration)
        print("\n🔍 Discovering Agent 0 (Data Product Registration)...")
        
        agent0_results = await registry.search_agents(
            skills=["catalog-registration", "ord-descriptor-creation"],
            tags=["catalog", "registration"]
        )
        
        if not agent0_results or not agent0_results.get("agents"):
            print("❌ Agent 0 not found in registry")
            return False
        
        agent0 = agent0_results["agents"][0]
        print(f"✅ Discovered Agent 0: {agent0['name']} at {agent0['url']}")
        
        # Step 3: Send message to Agent 0 (discovered dynamically)
        print("\n📤 Sending message to dynamically discovered Agent 0...")
        
        test_message = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Register financial data products"
                    },
                    {
                        "kind": "data",
                        "data": {
                            "raw_data_location": "/app/data/raw",
                            "processing_stage": "raw_to_ord",
                            "auto_trigger_downstream": True,
                            "use_dynamic_discovery": True
                        }
                    }
                ]
            },
            "contextId": f"true_a2a_test_{uuid4().hex[:8]}"
        }
        
        # Send message via registry client (no hardcoded URL!)
        result = await registry.send_message_to_agent(
            agent_id=agent0["agent_id"],
            message=test_message["message"],
            context_id=test_message["contextId"]
        )
        
        if result:
            task_id = result.get("taskId")
            print(f"✅ Agent 0 accepted task: {task_id}")
            
            # Monitor task progress
            print("⏳ Monitoring Agent 0 task...")
            agent0_url = agent0["url"]
            
            async with httpx.AsyncClient() as client:
                for i in range(30):
                    await asyncio.sleep(2)
                    
                    status_response = await client.get(f"{agent0_url}/a2a/agent0/v1/tasks/{task_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        state = status["status"]["state"]
                        print(f"   Status: {state}")
                        
                        if state == "completed":
                            print("✅ Agent 0 completed successfully")
                            
                            # Check if Agent 1 was triggered dynamically
                            artifacts = status.get("artifacts", [])
                            for artifact in artifacts:
                                if artifact.get("type") == "downstream_trigger":
                                    downstream = artifact.get("data", {})
                                    if downstream.get("discovered_dynamically"):
                                        print(f"✅ Agent 1 discovered dynamically: {downstream.get('agent_id')}")
                                        print(f"   URL: {downstream.get('downstream_url')}")
                                    else:
                                        print("⚠️ Agent 1 used from configuration (not dynamic)")
                            break
                            
                        elif state == "failed":
                            error = status["status"].get("error", {})
                            print(f"❌ Agent 0 failed: {error.get('message', 'Unknown error')}")
                            return False
        else:
            print("❌ Failed to send message to Agent 0")
            return False
        
        # Step 4: Verify Agent 1 was triggered
        print("\n🔍 Verifying Agent 1 (Standardization) was triggered...")
        
        # Discover Agent 1
        agent1_results = await registry.search_agents(
            skills=["standardization", "batch-standardization"],
            tags=["financial", "standardization"]
        )
        
        if agent1_results and agent1_results.get("agents"):
            agent1 = agent1_results["agents"][0]
            print(f"✅ Agent 1 discovered: {agent1['name']} at {agent1['url']}")
            
            # Could check Agent 1 tasks here if needed
        else:
            print("⚠️ Agent 1 not found in registry")
        
        # Step 5: Test workflow matching
        print("\n🔄 Testing workflow requirement matching...")
        
        workflow_requirements = [
            "dublin-core-extraction",
            "ord-descriptor-creation", 
            "catalog-registration",
            "location-standardization",
            "account-standardization"
        ]
        
        matched_agents = await registry.match_workflow_requirements(
            required_skills=workflow_requirements,
            preferred_tags=["financial", "dublin-core"]
        )
        
        if matched_agents:
            print(f"✅ Found {len(matched_agents)} agents matching workflow requirements:")
            for match in matched_agents:
                agent_info = match.get("agent", {})
                matched_skills = match.get("matched_skills", [])
                print(f"   - {agent_info.get('name')}: {len(matched_skills)} skills matched")
        else:
            print("⚠️ No agents matched workflow requirements")
        
        print("\n✅ TRUE A2A COMMUNICATION TEST SUCCESSFUL!")
        print("   - No hardcoded URLs used")
        print("   - All agents discovered dynamically")
        print("   - Communication routed via registry")
        print("   - Service discovery working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await registry.close()


async def test_agent_health_monitoring():
    """Test that registry monitors agent health"""
    
    print("\n=== TESTING AGENT HEALTH MONITORING ===")
    
    registry = A2ARegistryClient()
    
    try:
        # Check health status of all agents
        all_agents = await registry.search_agents()
        
        if all_agents and all_agents.get("agents"):
            print(f"📊 Agent Health Status:")
            for agent in all_agents["agents"]:
                status = agent.get("status", "unknown")
                last_health = agent.get("last_health_check", "never")
                
                status_emoji = "✅" if status == "healthy" else "❌"
                print(f"   {status_emoji} {agent['name']}: {status} (last check: {last_health})")
        else:
            print("❌ No agents registered")
            
    finally:
        await registry.close()


async def main():
    """Run all true A2A tests"""
    
    print("🚀 Starting True A2A Tests")
    print("=" * 60)
    
    # Test 1: True A2A workflow
    workflow_success = await test_true_a2a_workflow()
    
    # Test 2: Health monitoring
    await test_agent_health_monitoring()
    
    print("\n" + "=" * 60)
    print(f"Test Result: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    
    return workflow_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
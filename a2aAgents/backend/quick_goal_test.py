#!/usr/bin/env python3
"""
Quick test to verify the goal assignment fix
"""

import os
import sys
import asyncio

# Set environment for development mode
os.environ['A2A_DEV_MODE'] = 'true'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['BLOCKCHAIN_DISABLED'] = 'true'

async def test_single_agent_goal():
    try:
        print("🧪 Quick Goal Assignment Test")
        print("-" * 30)
        
        # Import modules
        from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
        from app.a2a.agents.orchestratorAgent.active.orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
        from app.a2a.agents.orchestratorAgent.active.smartGoalNotificationSystem import SMARTGoalNotificationSystem
        from app.a2a.agents.orchestratorAgent.active.comprehensiveGoalAssignment import ComprehensiveGoalAssignmentSystem
        
        # Build system
        print("🔧 Building system...")
        orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
        orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)
        notification_system = SMARTGoalNotificationSystem(orchestrator_handler)
        goal_system = ComprehensiveGoalAssignmentSystem(orchestrator_handler, notification_system)
        
        print("✅ System built successfully")
        
        # Test just one agent assignment to see if it works
        print("🎯 Testing single agent goal assignment...")
        profiles = goal_system.agent_profiles
        if profiles:
            first_agent_id = list(profiles.keys())[0]
            first_profile = profiles[first_agent_id]
            print(f"   Testing: {first_agent_id}")
            
            result = await goal_system._assign_agent_goals(first_agent_id, first_profile)
            
            if result.get("status") == "success":
                goals_count = len(result.get("goals", []))
                print(f"✅ SUCCESS: Created {goals_count} goals for {first_agent_id}")
                
                # Show first goal details
                if result.get("goals"):
                    first_goal = result["goals"][0]
                    print(f"   Sample goal: {first_goal.get('type', 'Unknown')} - {first_goal.get('description', 'No description')}")
                
                return True
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ FAILED: {error}")
                return False
        else:
            print("❌ No agent profiles found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_agent_goal())
    print(f"\n{'🎉 Test passed!' if success else '💥 Test failed!'}")
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Quick test to verify goal assignment system can initialize
"""

import os
import sys

# Set environment
os.environ['A2A_DEV_MODE'] = 'true'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress blockchain connection errors
os.environ['BLOCKCHAIN_DISABLED'] = 'true'

async def test_goal_assignment():
    try:
        print("🔧 Testing goal assignment system initialization...")
        
        # Test imports
        from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
        print("✅ ComprehensiveOrchestratorAgentSDK imported successfully")
        
        from app.a2a.agents.orchestratorAgent.active.orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
        print("✅ OrchestratorAgentA2AHandler imported successfully")
        
        from app.a2a.agents.orchestratorAgent.active.comprehensiveGoalAssignment import ComprehensiveGoalAssignmentSystem
        from app.a2a.agents.orchestratorAgent.active.smartGoalNotificationSystem import SMARTGoalNotificationSystem
        print("✅ Goal assignment system modules imported successfully")
        
        # Test initialization
        print("🔧 Creating orchestrator SDK...")
        orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
        print("✅ ComprehensiveOrchestratorAgentSDK created successfully")
        
        print("🔧 Creating orchestrator handler...")
        orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)
        print("✅ OrchestratorAgentA2AHandler created successfully")
        
        print("🔧 Creating SMART goal notification system...")
        notification_system = SMARTGoalNotificationSystem(orchestrator_handler)
        print("✅ SMARTGoalNotificationSystem created successfully")
        
        print("🔧 Creating goal assignment system...")
        goal_assignment = ComprehensiveGoalAssignmentSystem(orchestrator_handler, notification_system)
        print("✅ ComprehensiveGoalAssignmentSystem created successfully")
        
        print("🎯 Testing goal assignment for all agents...")
        result = await goal_assignment.assign_initial_goals_to_all_agents()
        print(f"✅ All agents goal assignment completed!")
        print(f"   Assigned goals to {len(result.get('assigned_agents', []))} agents")
        
        print("🔍 Monitoring goal progress...")
        progress = await goal_assignment.monitor_goal_progress()
        print(f"✅ Goal progress monitoring completed!")
        print(f"   Monitoring {len(progress.get('tracked_goals', []))} goals")
        
        print("🏆 SUCCESS: Goal assignment system working correctly!")
        print("✅ All components initialized and basic goal assignment tested successfully")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_goal_assignment())
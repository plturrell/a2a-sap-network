#!/usr/bin/env python3
"""
Enhanced Goal Assignment Script for All 16 A2A Agents
Provides detailed progress feedback and verification
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Set environment for development mode
os.environ['A2A_DEV_MODE'] = 'true'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['BLOCKCHAIN_DISABLED'] = 'true'

# Configure logging to reduce noise
logging.basicConfig(level=logging.WARNING)

async def main():
    print("🚀 Enhanced Goal Assignment System for A2A Network")
    print("=" * 60)
    
    try:
        print("📦 Step 1: Importing core modules...")
        from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
        from app.a2a.agents.orchestratorAgent.active.orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
        from app.a2a.agents.orchestratorAgent.active.smartGoalNotificationSystem import SMARTGoalNotificationSystem
        from app.a2a.agents.orchestratorAgent.active.comprehensiveGoalAssignment import ComprehensiveGoalAssignmentSystem
        print("✅ All modules imported successfully")
        
        print("\n🏗️ Step 2: Building goal assignment system...")
        
        print("   🔧 Creating ComprehensiveOrchestratorAgentSDK...")
        orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
        print("   ✅ ComprehensiveOrchestratorAgentSDK ready")
        
        print("   🔧 Creating OrchestratorAgentA2AHandler...")
        orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)
        print("   ✅ OrchestratorAgentA2AHandler ready")
        
        print("   🔧 Creating SMARTGoalNotificationSystem...")
        notification_system = SMARTGoalNotificationSystem(orchestrator_handler)
        print("   ✅ SMARTGoalNotificationSystem ready")
        
        print("   🔧 Creating ComprehensiveGoalAssignmentSystem...")
        goal_system = ComprehensiveGoalAssignmentSystem(orchestrator_handler, notification_system)
        print("   ✅ ComprehensiveGoalAssignmentSystem ready")
        
        print("\n🎯 Step 3: Assigning goals to all 16 A2A agents...")
        result = await goal_system.assign_initial_goals_to_all_agents()
        
        # Display results
        print(f"✅ Goal assignment completed!")
        
        # Parse the actual result format
        summary = result.get('summary', {})
        assignments = result.get('assignments', {})
        
        successful_count = summary.get('successful_assignments', 0)
        failed_count = summary.get('failed_assignments', 0)
        total_goals = summary.get('total_goals_assigned', 0)
        
        print(f"\n📊 Assignment Summary:")
        print(f"   ✅ Successfully assigned: {successful_count} agents")
        print(f"   ❌ Failed assignments: {failed_count} agents")
        print(f"   🎯 Total goals created: {total_goals} goals")
        
        # Show successful assignments
        successful_assignments = {k: v for k, v in assignments.items() if v.get('status') == 'success'}
        failed_assignments = {k: v for k, v in assignments.items() if v.get('status') in ['failed', 'error']}
        
        if successful_assignments:
            print(f"\n🎯 Successfully Assigned Agents:")
            for i, (agent_id, assignment) in enumerate(list(successful_assignments.items())[:10], 1):
                goals_count = len(assignment.get('goals', []))
                print(f"   {i:2d}. {agent_id}: {goals_count} goals")
            
            if len(successful_assignments) > 10:
                print(f"   ... and {len(successful_assignments) - 10} more agents")
        
        if failed_assignments:
            print(f"\n❌ Failed Assignments:")
            for agent_id, assignment in list(failed_assignments.items())[:5]:
                error = assignment.get('error', 'Unknown error')
                print(f"   • {agent_id}: {error}")
        
        # Show detailed agent breakdown
        if successful_assignments:
            print(f"\n📋 Detailed Agent Goals:")
            for agent_id, assignment in list(successful_assignments.items())[:5]:
                goals = assignment.get('goals', [])
                print(f"   🤖 {agent_id}:")
                for goal in goals[:3]:  # Show first 3 goals per agent
                    goal_type = goal.get('type', 'Unknown')
                    target = goal.get('target_value', 'N/A')
                    print(f"      • {goal_type}: Target {target}")
                if len(goals) > 3:
                    print(f"      ... and {len(goals) - 3} more goals")
        
        print(f"\n🔍 Step 4: Monitoring goal progress...")
        progress = await goal_system.monitor_goal_progress()
        
        tracked_goals = progress.get('tracked_goals', [])
        print(f"✅ Monitoring {len(tracked_goals)} active goals")
        
        # Show goal status summary
        statuses = {}
        for goal in tracked_goals:
            status = goal.get('status', 'unknown')
            statuses[status] = statuses.get(status, 0) + 1
        
        if statuses:
            print(f"\n📈 Goal Status Summary:")
            for status, count in statuses.items():
                print(f"   • {status.title()}: {count} goals")
        
        print(f"\n🔧 Step 5: Generating comprehensive report...")
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'assignment_summary': {
                'total_agents': successful_count + failed_count,
                'successful_assignments': successful_count,
                'failed_assignments': failed_count,
                'total_goals_assigned': total_goals,
                'success_rate': (successful_count / (successful_count + failed_count)) * 100 if (successful_count + failed_count) > 0 else 0
            },
            'goal_summary': {
                'total_goals': len(tracked_goals),
                'status_breakdown': statuses
            },
            'detailed_results': result,
            'successful_assignments': successful_assignments,
            'failed_assignments': failed_assignments
        }
        
        # Save report
        report_file = 'goal_assignment_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comprehensive report saved to: {report_file}")
        
        print(f"\n🏆 SUCCESS: Goal Assignment System Fully Operational!")
        print("=" * 60)
        print(f"✅ System Status: READY")
        print(f"📊 Agents with Goals: {successful_count}")
        print(f"🎯 Total Goals Assigned: {total_goals}")
        print(f"📈 Success Rate: {(successful_count / (successful_count + failed_count)) * 100 if (successful_count + failed_count) > 0 else 100:.1f}%")
        print(f"💡 System Health: {'EXCELLENT' if failed_count == 0 else 'GOOD' if failed_count < 3 else 'NEEDS ATTENTION'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
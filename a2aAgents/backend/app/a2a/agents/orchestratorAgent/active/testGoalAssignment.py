#!/usr/bin/env python3
"""
Test script to verify goal assignment works correctly
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append('../../../../../')

from app.a2a.agents.orchestratorAgent.active.assignGoalsToAllAgents import (
    assign_goals_to_all_agents, verify_goal_assignments
)

async def test_goal_assignment():
    """Test the goal assignment system"""
    print("=== Testing Goal Assignment System ===")
    
    # Run assignment
    print("\n1. Running goal assignment...")
    result = await assign_goals_to_all_agents()
    
    if result and isinstance(result, tuple):
        assignment_results, orchestrator_handler = result
        
        if assignment_results:
            summary = assignment_results["summary"]
            print(f"✓ Successfully assigned goals to {summary['successful_assignments']}/{summary['total_agents']} agents")
            print(f"✓ Total goals created: {summary['total_goals_assigned']}")
            
            # Test verification
            print("\n2. Running verification...")
            verification_results = await verify_goal_assignments(orchestrator_handler)
            
            if verification_results:
                agents_with_goals = sum(1 for v in verification_results.values() if v["has_goals"])
                print(f"✓ Verification complete: {agents_with_goals}/16 agents have goals")
                
                # Show sample goals
                print("\n3. Sample goals assigned:")
                for agent_id, assignment in list(assignment_results["assignments"].items())[:3]:
                    if assignment["status"] == "success":
                        print(f"\n{agent_id}:")
                        for goal in assignment["goals"]:
                            print(f"  - {goal['goal_type']}: {goal['specific'][:60]}...")
            else:
                print("✗ Verification failed")
        else:
            print("✗ Goal assignment failed")
    else:
        print("✗ Failed to run goal assignment")

if __name__ == "__main__":
    # Set development mode
    os.environ["A2A_DEV_MODE"] = "true"
    
    # Run test
    asyncio.run(test_goal_assignment())
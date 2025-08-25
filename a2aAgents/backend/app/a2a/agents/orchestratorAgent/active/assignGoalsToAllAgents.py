#!/usr/bin/env python3
"""
Script to assign goals to all A2A agents
Run this to initialize goals for the entire A2A network

Usage:
    # Development mode (uses default localhost values)
    A2A_DEV_MODE=true python3 assignGoalsToAllAgents.py

    # Production mode (requires all environment variables)
    python3 assignGoalsToAllAgents.py
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import json

# Check if we're in development mode
if os.getenv("A2A_DEV_MODE", "false").lower() == "true":
    print("Running in DEVELOPMENT mode")
else:
    print("Running in PRODUCTION mode")
    print("Set A2A_DEV_MODE=true for development mode with default values")

# Add parent directory to path for imports
sys.path.append('../../../../../')

from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
from app.a2a.agents.orchestratorAgent.active.orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler
from app.a2a.agents.orchestratorAgent.active.smartGoalNotificationSystem import SMARTGoalNotificationSystem
from app.a2a.agents.orchestratorAgent.active.comprehensiveGoalAssignment import create_comprehensive_goal_assignment_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def assign_goals_to_all_agents():
    """Main function to assign goals to all agents"""
    try:
        logger.info("=== Starting A2A Network Goal Assignment ===")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")

        # Initialize orchestrator components
        logger.info("Initializing orchestrator components...")
        orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
        orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)

        # Skip starting orchestrator handler for development mode to avoid blockchain dependency
        if os.getenv("A2A_DEV_MODE", "false").lower() != "true":
            await orchestrator_handler.start()
            logger.info("Orchestrator handler started successfully")
        else:
            logger.info("Skipping orchestrator handler startup in development mode")

        # Initialize notification system
        notification_system = SMARTGoalNotificationSystem(orchestrator_handler)
        logger.info("SMART goal notification system initialized")

        # Create comprehensive goal assignment system
        goal_assignment_system = create_comprehensive_goal_assignment_system(
            orchestrator_handler,
            notification_system
        )
        logger.info("Comprehensive goal assignment system created")

        # Assign initial goals to all agents
        logger.info("\n=== Assigning Goals to All 16 A2A Agents ===")
        assignment_results = await goal_assignment_system.assign_initial_goals_to_all_agents()

        # Print results summary
        logger.info("\n=== Assignment Results Summary ===")
        summary = assignment_results["summary"]
        logger.info(f"Total Agents: {summary['total_agents']}")
        logger.info(f"Successful Assignments: {summary['successful_assignments']}")
        logger.info(f"Failed Assignments: {summary['failed_assignments']}")
        logger.info(f"Total Goals Assigned: {summary['total_goals_assigned']}")

        # Print individual agent results
        logger.info("\n=== Individual Agent Results ===")
        for agent_id, result in assignment_results["assignments"].items():
            if result["status"] == "success":
                logger.info(f"✓ {agent_id}: {len(result['goals'])} goals assigned")
                for goal in result["goals"]:
                    logger.info(f"  - {goal['goal_type']} ({goal['goal_id']})")
            else:
                logger.error(f"✗ {agent_id}: Failed - {result.get('error', 'Unknown error')}")

        # Print collaborative goals
        if "collaborative_goals" in assignment_results:
            collab_summary = assignment_results["collaborative_goals"]
            logger.info(f"\n=== Collaborative Goals Created ===")
            logger.info(f"Total Collaborative Goals: {collab_summary['total_created']}")

            for collab_goal in collab_summary["collaborative_goals"]:
                logger.info(f"\n{collab_goal['title']}:")
                logger.info(f"  Participants: {', '.join(collab_goal['participating_agents'])}")
                logger.info(f"  Pattern: {collab_goal['collaboration_pattern']}")
                logger.info(f"  Duration: {collab_goal['duration']}")

        # Save results to file
        output_file = f"goal_assignment_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(assignment_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_file}")

        # Monitor initial progress
        logger.info("\n=== Initial Progress Check ===")
        await asyncio.sleep(2)  # Give agents time to acknowledge goals

        progress_report = await goal_assignment_system.monitor_goal_progress()
        logger.info(f"Goals on track: {progress_report['summary']['goals_on_track']}")
        logger.info(f"Goals at risk: {progress_report['summary']['goals_at_risk']}")
        logger.info(f"Goals completed: {progress_report['summary']['goals_completed']}")

        # Get recommendations
        logger.info("\n=== Goal Recommendations ===")
        recommendations = await goal_assignment_system.recommend_goal_adjustments()
        if recommendations:
            logger.info(f"Found {len(recommendations)} recommendations:")
            for rec in recommendations[:5]:  # Show first 5
                logger.info(f"  {rec['agent_id']} - {rec['goal_id']}: {len(rec['recommendations'])} suggestions")
        else:
            logger.info("No recommendations at this time (all goals on track)")

        # Stop orchestrator handler
        await orchestrator_handler.stop()
        logger.info("\n=== Goal Assignment Complete ===")

        return assignment_results

    except Exception as e:
        logger.error(f"Failed to assign goals: {e}", exc_info=True)
        return None

async def verify_goal_assignments():
    """Verify that goals were properly assigned"""
    try:
        logger.info("\n=== Verifying Goal Assignments ===")

        # Initialize orchestrator
        orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
        orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)
        await orchestrator_handler.start()

        # Check each agent
        agent_ids = [
            "agent0_data_product", "agent1_standardization", "agent2_ai_preparation",
            "agent3_vector_processing", "agent4_calc_validation", "agent5_qa_validation",
            "agent6_quality_control", "agent7_agent_manager", "agent8_data_manager",
            "agent9_reasoning", "agent10_calculation", "agent11_sql",
            "agent12_catalog_manager", "agent13_agent_builder", "agent14_embedding_finetuner",
            "agent15_orchestrator"
        ]

        verification_results = {}

        for agent_id in agent_ids:
            # Query goals for agent
            from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole

            message = A2AMessage(
                role=MessageRole.USER,
                parts=[MessagePart(
                    kind="goal_verification",
                    data={
                        "operation": "get_agent_goals",
                        "data": {"agent_id": agent_id}
                    }
                )]
            )

            result = await orchestrator_handler.process_a2a_message(message)

            if result.get("status") == "success":
                goals_data = result.get("data", {}).get("goals", {})
                goal_count = len(goals_data.get("goals", {}).get("primary_objectives", []))
                verification_results[agent_id] = {
                    "has_goals": goal_count > 0,
                    "goal_count": goal_count,
                    "status": goals_data.get("status", "unknown")
                }
                logger.info(f"✓ {agent_id}: {goal_count} goals found")
            else:
                verification_results[agent_id] = {
                    "has_goals": False,
                    "goal_count": 0,
                    "error": result.get("message", "Unknown error")
                }
                logger.warning(f"✗ {agent_id}: No goals found")

        # Summary
        agents_with_goals = sum(1 for v in verification_results.values() if v["has_goals"])
        logger.info(f"\n=== Verification Summary ===")
        logger.info(f"Agents with goals: {agents_with_goals}/{len(agent_ids)}")

        await orchestrator_handler.stop()
        return verification_results

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Run goal assignment
    results = asyncio.run(assign_goals_to_all_agents())

    if results:
        # Verify assignments
        asyncio.run(verify_goal_assignments())
    else:
        logger.error("Goal assignment failed, skipping verification")
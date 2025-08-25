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
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import sys
import os
from datetime import datetime, timedelta
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

        # In development mode, directly assign goals without A2A messaging
        if os.getenv("A2A_DEV_MODE", "false").lower() == "true":
            logger.info("Using direct assignment for development mode")

            # Use direct assignment to bypass messaging issues
            assignment_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "assignments": {},
                "summary": {
                    "total_agents": 16,
                    "successful_assignments": 0,
                    "failed_assignments": 0,
                    "total_goals_assigned": 0
                }
            }

            # Process each agent profile directly
            for agent_id, profile in goal_assignment_system.agent_profiles.items():
                try:
                    assigned_goals = []

                    # Assign top 2 goal types per agent
                    for goal_type in profile.primary_goal_types[:2]:
                        agent_key = agent_id.split('_')[0] if '_' in agent_id else agent_id
                        goal_template_key = goal_assignment_system._map_goal_template_key(agent_key, goal_type)

                        if goal_template_key in notification_system.goal_templates:
                            template = notification_system.goal_templates[goal_template_key]

                            # Calculate target metrics
                            target_metrics = {}
                            for metric in template.measurable_metrics[:3]:
                                if metric in profile.performance_baseline:
                                    baseline = profile.performance_baseline[metric]
                                    if metric in ["error_rate", "false_positive_rate", "false_alarm_rate"]:
                                        target = max(template.achievable_criteria[metric]["min"], baseline * 0.85)
                                    else:
                                        target = min(template.achievable_criteria[metric]["max"], baseline * 1.15)
                                    target_metrics[metric] = round(target, 2)

                            # Create template params
                            template_params = goal_assignment_system._create_template_params(template, target_metrics, goal_type)

                            # Create SMART goal
                            goal = {
                                "goal_id": f"{agent_id}_{goal_type}_{int(datetime.now().timestamp())}",
                                "agent_id": agent_id,
                                "goal_type": goal_type,
                                "title": f"{profile.agent_name} {goal_type.replace('_', ' ').title()} Goal",
                                "specific": template.specific_template.format(**template_params),
                                "measurable": target_metrics,
                                "achievable": True,
                                "relevant": template.relevant_context,
                                "time_bound": "30 days",
                                "created_at": datetime.now().isoformat(),
                                "target_date": (datetime.now() + timedelta(days=30)).isoformat(),
                                "assigned_date": datetime.now().isoformat(),
                                "status": "active",
                                "progress": 0
                            }

                            assigned_goals.append(goal)
                            logger.info(f"  ✓ Created {goal_type} goal for {agent_id}")

                    # Store goals in handler's memory for verification
                    if assigned_goals:
                        goals_data = {
                            "primary_objectives": assigned_goals,
                            "success_criteria": [],
                            "purpose_statement": f"Optimize {profile.agent_name} performance",
                            "kpis": list(set([metric for goal in assigned_goals for metric in goal["measurable"].keys()])),
                            "tracking_config": {
                                "frequency": "daily",
                                "alert_thresholds": {}
                            }
                        }

                        # Store directly in handler's agent_goals dict
                        orchestrator_handler.agent_goals[agent_id] = {
                            "agent_id": agent_id,
                            "goals": goals_data,
                            "created_at": datetime.utcnow().isoformat(),
                            "created_by": "goal_assignment_system",
                            "status": "active"
                        }

                        assignment_results["assignments"][agent_id] = {"status": "success", "goals": assigned_goals}
                        assignment_results["summary"]["successful_assignments"] += 1
                        assignment_results["summary"]["total_goals_assigned"] += len(assigned_goals)
                    else:
                        assignment_results["assignments"][agent_id] = {"status": "failed", "error": "No goals created"}
                        assignment_results["summary"]["failed_assignments"] += 1

                except Exception as e:
                    logger.error(f"Failed to assign goals to {agent_id}: {e}")
                    assignment_results["assignments"][agent_id] = {"status": "error", "error": str(e)}
                    assignment_results["summary"]["failed_assignments"] += 1
        else:
            # Production mode - use normal assignment
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

        # Stop orchestrator handler only if not in dev mode (we'll need it for verification)
        if os.getenv("A2A_DEV_MODE", "false").lower() != "true":
            await orchestrator_handler.stop()
        logger.info("\n=== Goal Assignment Complete ===")

        return assignment_results, orchestrator_handler

    except Exception as e:
        logger.error(f"Failed to assign goals: {e}", exc_info=True)
        return None, None

async def verify_goal_assignments(orchestrator_handler=None):
    """Verify that goals were properly assigned"""
    try:
        logger.info("\n=== Verifying Goal Assignments ===")

        # Initialize orchestrator if not provided
        if orchestrator_handler is None:
            orchestrator_sdk = ComprehensiveOrchestratorAgentSDK()
            orchestrator_handler = OrchestratorAgentA2AHandler(orchestrator_sdk)
        
        # In development mode, check in-memory storage directly
        if os.getenv("A2A_DEV_MODE", "false").lower() == "true":
            logger.info("Using direct verification for development mode")
            
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

            # Check goals directly from handler's memory
            for agent_id in agent_ids:
                if hasattr(orchestrator_handler, 'agent_goals') and agent_id in orchestrator_handler.agent_goals:
                    goals_data = orchestrator_handler.agent_goals[agent_id]
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
                        "error": "No goals in memory"
                    }
                    logger.warning(f"✗ {agent_id}: No goals found")
        else:
            # Production mode - use A2A messaging
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

            await orchestrator_handler.stop()

        # Summary
        agents_with_goals = sum(1 for v in verification_results.values() if v["has_goals"])
        logger.info(f"\n=== Verification Summary ===")
        logger.info(f"Agents with goals: {agents_with_goals}/{len(agent_ids)}")

        return verification_results

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Run goal assignment
    result = asyncio.run(assign_goals_to_all_agents())

    if result and isinstance(result, tuple):
        results, orchestrator_handler = result
        # Verify assignments with the same handler instance
        asyncio.run(verify_goal_assignments(orchestrator_handler))
    else:
        logger.error("Goal assignment failed, skipping verification")

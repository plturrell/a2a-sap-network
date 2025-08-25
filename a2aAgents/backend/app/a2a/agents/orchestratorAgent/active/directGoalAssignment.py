#!/usr/bin/env python3
"""
Direct Goal Assignment Script for A2A Agents
Bypasses A2A messaging and directly assigns goals for development mode
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from uuid import uuid4

# Add parent directory to path for imports
sys.path.append('../../../../../')

from app.a2a.agents.orchestratorAgent.active.comprehensiveGoalAssignment import ComprehensiveGoalAssignmentSystem
from app.a2a.agents.orchestratorAgent.active.smartGoalNotificationSystem import SMARTGoalNotificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectGoalAssigner:
    """Directly assigns goals without A2A messaging"""

    def __init__(self):
        # Create mock orchestrator handler
        self.orchestrator_handler = self._create_mock_handler()

        # Initialize systems
        self.notification_system = SMARTGoalNotificationSystem(self.orchestrator_handler)
        self.goal_system = ComprehensiveGoalAssignmentSystem(
            self.orchestrator_handler,
            self.notification_system
        )

        # Storage for assigned goals
        self.assigned_goals = {}

    def _create_mock_handler(self):
        """Create a mock orchestrator handler that works without A2A messaging"""
        class MockHandler:
            def __init__(self):
                self.agent_goals = {}

            async def process_a2a_message(self, message):
                """Mock message processing that directly handles goal operations"""
                try:
                    if message.parts and len(message.parts) > 0:
                        part = message.parts[0]
                        if hasattr(part, 'data') and part.data:
                            operation = part.data.get("operation")
                            data = part.data.get("data", {})

                            if operation == "set_agent_goals":
                                agent_id = data.get("agent_id")
                                goals = data.get("goals")
                                if agent_id and goals:
                                    self.agent_goals[agent_id] = goals
                                    logger.info(f"✓ Assigned goals to {agent_id}")
                                    return {"status": "success", "message": f"Goals assigned to {agent_id}"}

                            elif operation == "get_agent_goals":
                                agent_id = data.get("agent_id")
                                if agent_id in self.agent_goals:
                                    return {
                                        "status": "success",
                                        "data": {
                                            "goals": self.agent_goals[agent_id],
                                            "progress": {"overall_progress": 0}
                                        }
                                    }
                                else:
                                    return {"status": "not_found", "message": "No goals found"}

                    return {"status": "error", "message": "Invalid message format"}
                except Exception as e:
                    logger.error(f"Mock handler error: {e}")
                    return {"status": "error", "message": str(e)}

        return MockHandler()

    async def assign_all_goals(self):
        """Assign goals to all agents directly"""
        logger.info("=== Starting Direct Goal Assignment ===")

        try:
            # Get agent profiles
            agent_profiles = self.goal_system.agent_profiles
            logger.info(f"Processing {len(agent_profiles)} agent profiles")

            successful_assignments = 0
            failed_assignments = 0
            total_goals = 0

            for agent_id, profile in agent_profiles.items():
                logger.info(f"\n--- Processing {agent_id} ({profile.agent_name}) ---")

                try:
                    # Create goals directly
                    assigned_goals = []

                    # Process first 2 goal types for each agent
                    for goal_type in profile.primary_goal_types[:2]:
                        agent_key = agent_id.split('_')[0] if '_' in agent_id else agent_id
                        goal_template_key = self.goal_system._map_goal_template_key(agent_key, goal_type)

                        if goal_template_key in self.notification_system.goal_templates:
                            template = self.notification_system.goal_templates[goal_template_key]

                            # Calculate target metrics
                            target_metrics = {}
                            for metric in template.measurable_metrics[:3]:  # Limit to 3 metrics
                                if metric in profile.performance_baseline:
                                    baseline = profile.performance_baseline[metric]
                                    if metric in ["error_rate", "false_positive_rate", "false_alarm_rate"]:
                                        target = max(template.achievable_criteria[metric]["min"], baseline * 0.85)
                                    else:
                                        target = min(template.achievable_criteria[metric]["max"], baseline * 1.15)
                                    target_metrics[metric] = round(target, 2)

                            # Create goal
                            goal_id = f"{agent_id}_{goal_type}_{int(datetime.now().timestamp())}"

                            # Create template params
                            template_params = self.goal_system._create_template_params(template, target_metrics, goal_type)

                            goal = {
                                "goal_id": goal_id,
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
                            total_goals += 1
                            logger.info(f"  ✓ Created {goal_type} goal: {goal['specific']}")

                    # Store goals directly in mock handler
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

                        self.orchestrator_handler.agent_goals[agent_id] = goals_data
                        self.assigned_goals[agent_id] = assigned_goals
                        successful_assignments += 1
                        logger.info(f"  ✓ {agent_id}: {len(assigned_goals)} goals assigned successfully")
                    else:
                        failed_assignments += 1
                        logger.warning(f"  ✗ {agent_id}: No goals created")

                except Exception as e:
                    failed_assignments += 1
                    logger.error(f"  ✗ {agent_id}: Failed - {str(e)}")

            # Print summary
            logger.info(f"\n=== Assignment Complete ===")
            logger.info(f"Total Agents: {len(agent_profiles)}")
            logger.info(f"Successful Assignments: {successful_assignments}")
            logger.info(f"Failed Assignments: {failed_assignments}")
            logger.info(f"Total Goals Created: {total_goals}")

            # Save results to file
            results = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_agents": len(agent_profiles),
                    "successful_assignments": successful_assignments,
                    "failed_assignments": failed_assignments,
                    "total_goals_assigned": total_goals
                },
                "assigned_goals": self.assigned_goals
            }

            output_file = "goal_assignment_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

            # Verification
            await self.verify_assignments()

            return results

        except Exception as e:
            logger.error(f"Goal assignment failed: {e}")
            raise

    async def verify_assignments(self):
        """Verify goal assignments"""
        logger.info(f"\n=== Verifying Goal Assignments ===")

        for agent_id in self.assigned_goals:
            goals = self.orchestrator_handler.agent_goals.get(agent_id)
            if goals and goals.get("primary_objectives"):
                goal_count = len(goals["primary_objectives"])
                logger.info(f"✓ {agent_id}: {goal_count} goals verified")

                # Show first goal as example
                first_goal = goals["primary_objectives"][0]
                logger.info(f"  Example: {first_goal['specific']}")
            else:
                logger.warning(f"✗ {agent_id}: No goals found in verification")

async def main():
    """Main function"""
    if os.getenv("A2A_DEV_MODE", "false").lower() == "true":
        print("Running Direct Goal Assignment in DEVELOPMENT mode")
    else:
        print("Set A2A_DEV_MODE=true to run this script")
        return

    try:
        assigner = DirectGoalAssigner()
        await assigner.assign_all_goals()

        logger.info("\n🎉 Goal assignment completed successfully!")

    except Exception as e:
        logger.error(f"Goal assignment failed: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())

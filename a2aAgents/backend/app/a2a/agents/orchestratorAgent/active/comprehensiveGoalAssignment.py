"""
Comprehensive Goal Assignment System for A2A Network
Automatically assigns and manages goals for all 16 A2A agents
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from .smartGoalNotificationSystem import SMARTGoalNotificationSystem

logger = logging.getLogger(__name__)

@dataclass
class AgentGoalProfile:
    """Profile defining goal preferences for each agent type"""
    agent_id: str
    agent_name: str
    primary_goal_types: List[str]
    performance_baseline: Dict[str, float]
    goal_priorities: List[str]  # ordered by priority
    collaboration_preferences: List[str]  # preferred collaboration partners

class ComprehensiveGoalAssignmentSystem:
    """Manages goal assignment across all A2A agents"""

    def __init__(self, orchestrator_handler, notification_system: SMARTGoalNotificationSystem):
        self.orchestrator = orchestrator_handler
        self.notification_system = notification_system
        self.agent_profiles = self._initialize_agent_profiles()
        self.assignment_history = {}
        self.active_assignments = {}

    def _initialize_agent_profiles(self) -> Dict[str, AgentGoalProfile]:
        """Initialize goal profiles for all 16 agents"""
        profiles = {
            # Data Pipeline Agents (0-5)
            "agent0_data_product": AgentGoalProfile(
                agent_id="agent0_data_product",
                agent_name="Data Product Agent",
                primary_goal_types=["performance", "quality", "reliability"],
                performance_baseline={
                    "registration_success_rate": 90.0,
                    "avg_registration_time": 3.0,
                    "validation_accuracy": 95.0,
                    "data_quality_score": 85.0
                },
                goal_priorities=["registration_efficiency", "quality_improvement", "reliability"],
                collaboration_preferences=["agent1_standardization", "agent5_qa_validation"]
            ),

            "agent1_standardization": AgentGoalProfile(
                agent_id="agent1_standardization",
                agent_name="Data Standardization Agent",
                primary_goal_types=["transformation", "compliance"],
                performance_baseline={
                    "standardization_success_rate": 92.0,
                    "avg_transformation_time": 2.0,
                    "canonical_compliance_rate": 96.0,
                    "schema_mapping_accuracy": 97.0
                },
                goal_priorities=["transformation_efficiency", "compliance_adherence"],
                collaboration_preferences=["agent0_data_product", "agent2_ai_preparation"]
            ),

            "agent2_ai_preparation": AgentGoalProfile(
                agent_id="agent2_ai_preparation",
                agent_name="AI Data Preparation Agent",
                primary_goal_types=["feature_engineering", "privacy_preservation"],
                performance_baseline={
                    "feature_quality_score": 88.0,
                    "ml_readiness_score": 92.0,
                    "privacy_preservation_score": 97.0,
                    "feature_generation_speed": 300
                },
                goal_priorities=["feature_quality", "privacy_compliance", "ml_readiness"],
                collaboration_preferences=["agent1_standardization", "agent3_vector_processing"]
            ),

            "agent3_vector_processing": AgentGoalProfile(
                agent_id="agent3_vector_processing",
                agent_name="Vector Processing Agent",
                primary_goal_types=["embedding_generation", "vector_indexing"],
                performance_baseline={
                    "embedding_quality_score": 90.0,
                    "vector_generation_throughput": 1000,
                    "index_coverage_rate": 97.0,
                    "avg_query_time_ms": 25
                },
                goal_priorities=["embedding_quality", "indexing_performance", "search_accuracy"],
                collaboration_preferences=["agent2_ai_preparation", "agent14_embedding_finetuner"]
            ),

            "agent4_calc_validation": AgentGoalProfile(
                agent_id="agent4_calc_validation",
                agent_name="Calculation Validation Agent",
                primary_goal_types=["calculation_validation"],
                performance_baseline={
                    "mathematical_validation_accuracy": 99.5,
                    "validation_throughput": 100,
                    "false_positive_rate": 1.0,
                    "symbolic_computation_success": 97.0
                },
                goal_priorities=["validation_accuracy", "throughput_optimization"],
                collaboration_preferences=["agent10_calculation", "agent5_qa_validation"]
            ),

            "agent5_qa_validation": AgentGoalProfile(
                agent_id="agent5_qa_validation",
                agent_name="QA Validation Agent",
                primary_goal_types=["quality_assurance"],
                performance_baseline={
                    "qa_pass_rate": 94.0,
                    "avg_review_time_hours": 4.0,
                    "defect_detection_rate": 97.0,
                    "compliance_check_score": 99.0
                },
                goal_priorities=["quality_gates", "compliance_verification", "defect_prevention"],
                collaboration_preferences=["agent0_data_product", "agent6_quality_control"]
            ),

            # Management Agents (6-8)
            "agent6_quality_control": AgentGoalProfile(
                agent_id="agent6_quality_control",
                agent_name="Quality Control Manager",
                primary_goal_types=["continuous_monitoring"],
                performance_baseline={
                    "issue_detection_rate": 96.0,
                    "mean_time_to_repair": 15,
                    "false_alarm_rate": 2.0,
                    "monitoring_coverage": 95.0
                },
                goal_priorities=["proactive_monitoring", "rapid_remediation", "coverage_expansion"],
                collaboration_preferences=["agent5_qa_validation", "agent7_agent_manager"]
            ),

            "agent7_agent_manager": AgentGoalProfile(
                agent_id="agent7_agent_manager",
                agent_name="Agent Manager",
                primary_goal_types=["agent_management"],
                performance_baseline={
                    "agent_uptime_percentage": 99.5,
                    "deployment_success_rate": 97.0,
                    "health_check_response_time": 100,
                    "resource_utilization": 60.0
                },
                goal_priorities=["uptime_maximization", "deployment_reliability", "resource_optimization"],
                collaboration_preferences=["agent15_orchestrator", "agent13_agent_builder"]
            ),

            "agent8_data_manager": AgentGoalProfile(
                agent_id="agent8_data_manager",
                agent_name="Data Manager",
                primary_goal_types=["data_storage"],
                performance_baseline={
                    "storage_efficiency_ratio": 82.0,
                    "avg_retrieval_time_ms": 50,
                    "data_compression_ratio": 4.0,
                    "cache_hit_rate": 88.0
                },
                goal_priorities=["storage_optimization", "retrieval_performance", "cache_efficiency"],
                collaboration_preferences=["agent0_data_product", "agent12_catalog_manager"]
            ),

            # Specialized Agents (9-11)
            "agent9_reasoning": AgentGoalProfile(
                agent_id="agent9_reasoning",
                agent_name="Reasoning Agent",
                primary_goal_types=["logical_reasoning"],
                performance_baseline={
                    "reasoning_accuracy": 93.0,
                    "inference_throughput": 20,
                    "logic_consistency_score": 97.0,
                    "explanation_quality": 90.0
                },
                goal_priorities=["reasoning_accuracy", "inference_speed", "explanation_clarity"],
                collaboration_preferences=["agent10_calculation", "agent11_sql"]
            ),

            "agent10_calculation": AgentGoalProfile(
                agent_id="agent10_calculation",
                agent_name="Calculation Agent",
                primary_goal_types=["complex_calculation"],
                performance_baseline={
                    "calculation_accuracy": 99.95,
                    "calculation_throughput": 500,
                    "numerical_stability_score": 98.0,
                    "self_healing_success_rate": 95.0
                },
                goal_priorities=["computational_accuracy", "numerical_stability", "self_healing"],
                collaboration_preferences=["agent4_calc_validation", "agent9_reasoning"]
            ),

            "agent11_sql": AgentGoalProfile(
                agent_id="agent11_sql",
                agent_name="SQL Agent",
                primary_goal_types=["sql_operations"],
                performance_baseline={
                    "nl2sql_accuracy": 90.0,
                    "query_optimization_rate": 80.0,
                    "avg_query_execution_time": 200,
                    "sql_injection_prevention": 99.95
                },
                goal_priorities=["nl2sql_accuracy", "query_optimization", "security"],
                collaboration_preferences=["agent9_reasoning", "agent8_data_manager"]
            ),

            # Infrastructure Agents (12-15)
            "agent12_catalog_manager": AgentGoalProfile(
                agent_id="agent12_catalog_manager",
                agent_name="Catalog Manager",
                primary_goal_types=["service_catalog"],
                performance_baseline={
                    "catalog_completeness": 97.0,
                    "avg_discovery_time": 5.0,
                    "ord_compliance_rate": 99.0,
                    "metadata_accuracy": 97.0
                },
                goal_priorities=["catalog_completeness", "discovery_speed", "metadata_quality"],
                collaboration_preferences=["agent8_data_manager", "agent7_agent_manager"]
            ),

            "agent13_agent_builder": AgentGoalProfile(
                agent_id="agent13_agent_builder",
                agent_name="Agent Builder",
                primary_goal_types=["agent_creation"],
                performance_baseline={
                    "agent_build_success_rate": 94.0,
                    "avg_deployment_time": 20,
                    "code_quality_score": 90.0,
                    "test_coverage": 85.0
                },
                goal_priorities=["build_reliability", "code_quality", "deployment_speed"],
                collaboration_preferences=["agent7_agent_manager", "agent15_orchestrator"]
            ),

            "agent14_embedding_finetuner": AgentGoalProfile(
                agent_id="agent14_embedding_finetuner",
                agent_name="Embedding Fine-Tuner",
                primary_goal_types=["model_finetuning"],
                performance_baseline={
                    "model_performance_improvement": 15.0,
                    "training_efficiency": 82.0,
                    "embedding_quality_score": 91.0,
                    "convergence_speed": 100
                },
                goal_priorities=["model_improvement", "training_efficiency", "embedding_quality"],
                collaboration_preferences=["agent3_vector_processing", "agent2_ai_preparation"]
            ),

            "agent15_orchestrator": AgentGoalProfile(
                agent_id="agent15_orchestrator",
                agent_name="Orchestrator Agent",
                primary_goal_types=["workflow_orchestration"],
                performance_baseline={
                    "workflow_success_rate": 97.0,
                    "scheduling_efficiency": 90.0,
                    "pipeline_throughput": 100,
                    "resource_optimization_score": 88.0
                },
                goal_priorities=["workflow_reliability", "scheduling_optimization", "resource_efficiency"],
                collaboration_preferences=["agent7_agent_manager", "all_agents"]
            )
        }

        return profiles

    async def assign_initial_goals_to_all_agents(self) -> Dict[str, Any]:
        """Assign initial goals to all 16 agents based on their profiles"""
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

        for agent_id, profile in self.agent_profiles.items():
            try:
                # Assign primary goal for each agent
                agent_goals = await self._assign_agent_goals(agent_id, profile)

                if agent_goals["status"] == "success":
                    assignment_results["assignments"][agent_id] = agent_goals
                    assignment_results["summary"]["successful_assignments"] += 1
                    assignment_results["summary"]["total_goals_assigned"] += len(agent_goals["goals"])
                else:
                    assignment_results["assignments"][agent_id] = {"status": "failed", "error": agent_goals.get("error")}
                    assignment_results["summary"]["failed_assignments"] += 1

            except Exception as e:
                logger.error(f"Failed to assign goals to {agent_id}: {e}")
                assignment_results["assignments"][agent_id] = {"status": "error", "error": str(e)}
                assignment_results["summary"]["failed_assignments"] += 1

        # Create collaborative goals for compatible agents
        collab_results = await self._create_collaborative_goals()
        assignment_results["collaborative_goals"] = collab_results

        return assignment_results

    async def _assign_agent_goals(self, agent_id: str, profile: AgentGoalProfile) -> Dict[str, Any]:
        """Assign specific goals to an agent based on its profile"""
        try:
            assigned_goals = []

            # Register agent for notifications first
            await self.notification_system.register_agent_for_notifications(
                agent_id,
                {
                    "metrics_capabilities": list(profile.performance_baseline.keys()),
                    "collection_frequency": "hourly",
                    "notification_preferences": {"email": False, "a2a_message": True}
                }
            )

            # Assign primary goal based on first goal type
            for goal_type in profile.primary_goal_types[:2]:  # Assign top 2 goal types
                # Extract agent number/type from agent_id (e.g., agent0_data_product -> agent0)
                agent_key = agent_id.split('_')[0] if '_' in agent_id else agent_id
                goal_template_key = self._map_goal_template_key(agent_key, goal_type)

                if goal_template_key in self.notification_system.goal_templates:
                    # Calculate target metrics based on baseline + improvement
                    target_metrics = {}
                    template = self.notification_system.goal_templates[goal_template_key]

                    for metric in template.measurable_metrics:
                        if metric in profile.performance_baseline:
                            baseline = profile.performance_baseline[metric]
                            # Set target as 10-20% improvement from baseline
                            if metric in ["error_rate", "false_positive_rate", "false_alarm_rate"]:
                                # Lower is better
                                target = max(template.achievable_criteria[metric]["min"], baseline * 0.85)
                            else:
                                # Higher is better
                                target = min(template.achievable_criteria[metric]["max"], baseline * 1.15)
                            target_metrics[metric] = round(target, 2)

                    # Create template parameter mapping based on the goal template
                    template_params = self._create_template_params(template, target_metrics, goal_type)

                    goal_params = {
                        "specific": template.specific_template.format(**template_params),
                        "measurable": target_metrics,
                        "time_bound": "30 days",
                        "tracking_frequency": "daily"
                    }

                    smart_goal = self.notification_system.create_smart_goal(
                        agent_id, goal_type, goal_params
                    )

                    # Send goal to orchestrator
                    goal_message = {
                        "operation": "set_agent_goals",
                        "data": {
                            "agent_id": agent_id,
                            "goals": {
                                "primary_objectives": [smart_goal],
                                "success_criteria": self._generate_success_criteria(smart_goal),
                                "purpose_statement": f"Optimize {profile.agent_name} {goal_type} performance",
                                "kpis": list(target_metrics.keys()),
                                "tracking_config": {
                                    "frequency": "daily",
                                    "alert_thresholds": self._generate_alert_thresholds(target_metrics)
                                }
                            }
                        }
                    }

                    # Process through orchestrator
                    message = A2AMessage(
                        role=MessageRole.USER,
                        parts=[MessagePart(
                            kind="goal_assignment",
                            data=goal_message
                        )]
                    )

                    result = await self.orchestrator.process_a2a_message(message)

                    if result.get("status") == "success":
                        assigned_goals.append({
                            "goal_id": smart_goal["goal_id"],
                            "goal_type": goal_type,
                            "status": "assigned"
                        })

                        # Store in active assignments
                        if agent_id not in self.active_assignments:
                            self.active_assignments[agent_id] = []
                        self.active_assignments[agent_id].append(smart_goal)

            return {
                "status": "success",
                "agent_id": agent_id,
                "goals": assigned_goals,
                "total_assigned": len(assigned_goals)
            }

        except Exception as e:
            logger.error(f"Failed to assign goals to {agent_id}: {e}")
            return {"status": "failed", "error": str(e)}

    def _generate_success_criteria(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate success criteria for a goal"""
        criteria = []

        for metric, target in goal["measurable"].items():
            criterion = {
                "metric_name": metric,
                "target_value": target,
                "comparison": ">=" if metric not in ["error_rate", "false_positive_rate"] else "<=",
                "weight": 1.0 / len(goal["measurable"])  # Equal weight for all metrics
            }
            criteria.append(criterion)

        return criteria

    def _generate_alert_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Generate alert thresholds for metrics"""
        thresholds = {}

        for metric, target in metrics.items():
            if metric in ["error_rate", "false_positive_rate", "false_alarm_rate"]:
                # Lower is better
                thresholds[metric] = {
                    "warning": target * 1.2,
                    "critical": target * 1.5
                }
            else:
                # Higher is better
                thresholds[metric] = {
                    "warning": target * 0.9,
                    "critical": target * 0.8
                }

        return thresholds

    async def _create_collaborative_goals(self) -> Dict[str, Any]:
        """Create collaborative goals for compatible agent groups"""
        collab_goals = []

        # Define collaborative goal opportunities
        collaboration_opportunities = [
            {
                "agents": ["agent0_data_product", "agent1_standardization", "agent2_ai_preparation"],
                "goal": "End-to-End Data Pipeline Optimization",
                "pattern": "sequential",
                "metrics": {
                    "pipeline_throughput": 500,  # items/hour
                    "end_to_end_latency": 30,    # seconds
                    "data_quality_score": 95.0
                }
            },
            {
                "agents": ["agent3_vector_processing", "agent14_embedding_finetuner"],
                "goal": "Advanced Embedding Quality Enhancement",
                "pattern": "parallel",
                "metrics": {
                    "embedding_quality_improvement": 20.0,  # percentage
                    "vector_accuracy": 95.0,
                    "optimization_efficiency": 85.0
                }
            },
            {
                "agents": ["agent4_calc_validation", "agent10_calculation"],
                "goal": "Mathematical Computation Excellence",
                "pattern": "parallel",
                "metrics": {
                    "combined_accuracy": 99.99,
                    "validation_coverage": 100.0,
                    "computation_reliability": 99.9
                }
            },
            {
                "agents": ["agent5_qa_validation", "agent6_quality_control"],
                "goal": "Comprehensive Quality Assurance Framework",
                "pattern": "hierarchical",
                "metrics": {
                    "quality_gate_effectiveness": 98.0,
                    "defect_escape_rate": 0.1,
                    "mttr_improvement": 50.0  # percentage reduction
                }
            },
            {
                "agents": ["agent7_agent_manager", "agent13_agent_builder", "agent15_orchestrator"],
                "goal": "Agent Lifecycle Management Excellence",
                "pattern": "hierarchical",
                "metrics": {
                    "deployment_automation": 95.0,
                    "agent_availability": 99.9,
                    "orchestration_efficiency": 92.0
                }
            }
        ]

        for collab_config in collaboration_opportunities:
            try:
                # Create collaborative goal
                collab_goal = {
                    "goal_id": f"collab_{datetime.utcnow().timestamp()}",
                    "title": collab_config["goal"],
                    "participating_agents": collab_config["agents"],
                    "collaboration_pattern": collab_config["pattern"],
                    "measurable_targets": collab_config["metrics"],
                    "duration": "60 days",
                    "status": "proposed"
                }

                # Assign roles based on pattern
                if collab_config["pattern"] == "sequential":
                    roles = {agent: f"stage_{i+1}" for i, agent in enumerate(collab_config["agents"])}
                elif collab_config["pattern"] == "parallel":
                    roles = {agent: "co-contributor" for agent in collab_config["agents"]}
                else:  # hierarchical
                    roles = {
                        collab_config["agents"][0]: "lead",
                        **{agent: "contributor" for agent in collab_config["agents"][1:]}
                    }

                collab_goal["agent_roles"] = roles
                collab_goals.append(collab_goal)

            except Exception as e:
                logger.error(f"Failed to create collaborative goal: {e}")

        return {
            "total_created": len(collab_goals),
            "collaborative_goals": collab_goals
        }

    async def monitor_goal_progress(self) -> Dict[str, Any]:
        """Monitor progress across all agent goals"""
        progress_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_progress": {},
            "summary": {
                "total_goals": 0,
                "goals_on_track": 0,
                "goals_at_risk": 0,
                "goals_completed": 0
            }
        }

        for agent_id, goals in self.active_assignments.items():
            agent_progress = {
                "goals": [],
                "overall_health": "good"
            }

            for goal in goals:
                # Get progress from orchestrator
                progress_message = {
                    "operation": "get_agent_goals",
                    "data": {
                        "agent_id": agent_id,
                        "include_progress": True
                    }
                }

                message = A2AMessage(
                    role=MessageRole.USER,
                    parts=[MessagePart(
                        kind="goal_monitoring",
                        data=progress_message
                    )]
                )

                result = await self.orchestrator.process_a2a_message(message)

                if result.get("status") == "success" and "progress" in result["data"]:
                    progress_data = result["data"]["progress"]
                    goal_status = self._assess_goal_status(goal, progress_data)

                    agent_progress["goals"].append({
                        "goal_id": goal["goal_id"],
                        "progress": progress_data.get("overall_progress", 0),
                        "status": goal_status,
                        "metrics": progress_data.get("objective_progress", {})
                    })

                    # Update summary
                    progress_report["summary"]["total_goals"] += 1
                    if goal_status == "on_track":
                        progress_report["summary"]["goals_on_track"] += 1
                    elif goal_status == "at_risk":
                        progress_report["summary"]["goals_at_risk"] += 1
                        agent_progress["overall_health"] = "warning"
                    elif goal_status == "completed":
                        progress_report["summary"]["goals_completed"] += 1

            progress_report["agent_progress"][agent_id] = agent_progress

        return progress_report

    def _assess_goal_status(self, goal: Dict[str, Any], progress: Dict[str, Any]) -> str:
        """Assess if a goal is on track, at risk, or completed"""
        overall_progress = progress.get("overall_progress", 0)

        # Calculate expected progress based on time elapsed
        start_date = datetime.fromisoformat(goal["assigned_date"])
        target_date = datetime.fromisoformat(goal["target_date"])
        current_date = datetime.utcnow()

        total_duration = (target_date - start_date).days
        elapsed_duration = (current_date - start_date).days
        expected_progress = (elapsed_duration / total_duration) * 100 if total_duration > 0 else 0

        if overall_progress >= 100:
            return "completed"
        elif overall_progress >= expected_progress * 0.9:  # Within 10% of expected
            return "on_track"
        else:
            return "at_risk"

    async def recommend_goal_adjustments(self) -> List[Dict[str, Any]]:
        """Recommend adjustments to goals based on performance"""
        recommendations = []

        # Get current progress
        progress_report = await self.monitor_goal_progress()

        for agent_id, agent_data in progress_report["agent_progress"].items():
            for goal_data in agent_data["goals"]:
                if goal_data["status"] == "at_risk":
                    # Generate recommendations for at-risk goals
                    recommendation = {
                        "agent_id": agent_id,
                        "goal_id": goal_data["goal_id"],
                        "current_progress": goal_data["progress"],
                        "recommendations": []
                    }

                    # Analyze underperforming metrics
                    for metric, value in goal_data["metrics"].items():
                        if value < 80:  # Metric is underperforming
                            recommendation["recommendations"].append({
                                "type": "metric_improvement",
                                "metric": metric,
                                "current_value": value,
                                "suggestion": f"Focus resources on improving {metric}"
                            })

                    # Suggest timeline adjustment if severely behind
                    if goal_data["progress"] < 50:
                        recommendation["recommendations"].append({
                            "type": "timeline_extension",
                            "suggestion": "Consider extending deadline by 14 days"
                        })

                    # Suggest collaboration if available
                    profile = self.agent_profiles.get(agent_id)
                    if profile and profile.collaboration_preferences:
                        recommendation["recommendations"].append({
                            "type": "collaboration",
                            "suggestion": f"Consider collaboration with {profile.collaboration_preferences[0]}"
                        })

                    recommendations.append(recommendation)

        return recommendations

    def _create_template_params(self, template: 'SMARTGoalTemplate', target_metrics: Dict[str, float], goal_type: str) -> Dict[str, Any]:
        """Create template parameter mapping based on goal template and metrics"""
        params = {
            "time_constraint": "optimal"
        }

        # Map common template placeholders to actual metric values
        if goal_type == "transformation":
            params.update({
                "success_rate": target_metrics.get("standardization_success_rate", 95.0),
                "processing_time": target_metrics.get("avg_transformation_time", 2.0)
            })
        elif goal_type == "compliance":
            params.update({
                "compliance_rate": target_metrics.get("canonical_compliance_rate", 96.0),
                "validation_accuracy": target_metrics.get("validation_accuracy", 98.0)
            })
        elif goal_type == "feature_engineering":
            params.update({
                "feature_quality": target_metrics.get("feature_quality_score", 88.0),
                "feature_coverage": target_metrics.get("feature_coverage_rate", 85.0)
            })
        elif goal_type == "privacy_preservation":
            params.update({
                "privacy_score": target_metrics.get("privacy_preservation_score", 97.0),
                "anonymization_rate": target_metrics.get("anonymization_success_rate", 98.0)
            })
        elif goal_type == "performance":
            # Get first metric as primary target
            first_metric = list(target_metrics.keys())[0] if target_metrics else "performance"
            params.update({
                "target_rate": target_metrics.get(first_metric, 95.0),
                "metric_name": first_metric.replace('_', ' ').title()
            })
        elif goal_type == "quality":
            params.update({
                "target_score": target_metrics.get("data_quality_score", 85.0),
                "quality_metric": "data quality score",
                "compliance_rate": target_metrics.get("schema_compliance_rate", 95.0)
            })
        elif goal_type == "reliability":
            params.update({
                "availability": target_metrics.get("api_availability", 99.0),
                "error_rate": target_metrics.get("error_rate", 2.0)
            })
        elif goal_type == "embedding_generation":
            params.update({
                "quality_score": target_metrics.get("embedding_quality_score", 90.0),
                "throughput": target_metrics.get("vector_generation_throughput", 1000)
            })
        elif goal_type == "vector_indexing":
            params.update({
                "index_coverage": target_metrics.get("index_coverage_rate", 97.0),
                "query_speed": target_metrics.get("avg_query_time_ms", 25)
            })
        elif goal_type == "calculation_validation":
            params.update({
                "validation_accuracy": target_metrics.get("mathematical_validation_accuracy", 99.5),
                "validation_speed": target_metrics.get("validation_throughput", 100)
            })
        elif goal_type == "quality_assurance":
            params.update({
                "qa_pass_rate": target_metrics.get("qa_pass_rate", 94.0),
                "review_time": target_metrics.get("avg_review_time_hours", 4.0)
            })
        elif goal_type == "continuous_monitoring":
            params.update({
                "detection_rate": target_metrics.get("issue_detection_rate", 96.0),
                "mttr": target_metrics.get("mean_time_to_repair", 15)
            })
        elif goal_type == "agent_management":
            params.update({
                "agent_uptime": target_metrics.get("agent_uptime_percentage", 99.5),
                "deployment_success": target_metrics.get("deployment_success_rate", 97.0)
            })
        elif goal_type == "data_storage":
            params.update({
                "storage_efficiency": target_metrics.get("storage_efficiency_ratio", 82.0),
                "retrieval_time": target_metrics.get("avg_retrieval_time_ms", 50)
            })
        elif goal_type == "logical_reasoning":
            params.update({
                "reasoning_accuracy": target_metrics.get("reasoning_accuracy", 93.0),
                "inference_speed": target_metrics.get("inference_throughput", 20)
            })
        elif goal_type == "complex_calculation":
            params.update({
                "calculation_accuracy": target_metrics.get("calculation_accuracy", 99.95),
                "calc_throughput": target_metrics.get("calculation_throughput", 500)
            })
        elif goal_type == "sql_operations":
            params.update({
                "nl2sql_accuracy": target_metrics.get("nl2sql_accuracy", 90.0),
                "query_optimization": target_metrics.get("query_optimization_rate", 80.0)
            })
        elif goal_type == "service_catalog":
            params.update({
                "catalog_completeness": target_metrics.get("catalog_completeness", 97.0),
                "discovery_time": target_metrics.get("avg_discovery_time", 5.0)
            })
        elif goal_type == "agent_creation":
            params.update({
                "build_success": target_metrics.get("agent_build_success_rate", 94.0),
                "deployment_time": target_metrics.get("avg_deployment_time", 20)
            })
        elif goal_type == "model_finetuning":
            params.update({
                "model_improvement": target_metrics.get("model_performance_improvement", 15.0),
                "training_efficiency": target_metrics.get("training_efficiency", 82.0)
            })
        elif goal_type == "workflow_orchestration":
            params.update({
                "workflow_success": target_metrics.get("workflow_success_rate", 97.0),
                "scheduling_efficiency": target_metrics.get("scheduling_efficiency", 90.0)
            })
        else:
            # Default parameters for other goal types
            if target_metrics:
                first_metric = list(target_metrics.keys())[0]
                params.update({
                    "target_rate": target_metrics[first_metric],
                    "metric_name": first_metric.replace('_', ' ').title()
                })
            else:
                params.update({
                    "target_rate": 95.0,
                    "metric_name": "Performance"
                })

        return params

    def _map_goal_template_key(self, agent_key: str, goal_type: str) -> str:
        """Map goal type to correct template key"""
        # Direct mapping for known mismatches
        goal_type_mapping = {
            ("agent3", "embedding_generation"): "agent3_embedding",
            ("agent3", "vector_indexing"): "agent3_indexing",
            ("agent4", "calculation_validation"): "agent4_validation",
            ("agent5", "quality_assurance"): "agent5_quality",
            ("agent6", "continuous_monitoring"): "agent6_monitoring",
            ("agent7", "agent_management"): "agent7_management",
            ("agent8", "data_storage"): "agent8_storage",
            ("agent9", "logical_reasoning"): "agent9_reasoning",
            ("agent10", "complex_calculation"): "agent10_computation",
            ("agent11", "sql_operations"): "agent11_query",
            ("agent12", "service_catalog"): "agent12_catalog",
            ("agent13", "agent_creation"): "agent13_builder",
            ("agent14", "model_finetuning"): "agent14_finetuning",
            ("agent15", "workflow_orchestration"): "agent15_orchestration"
        }

        # Check if we have a specific mapping
        if (agent_key, goal_type) in goal_type_mapping:
            return goal_type_mapping[(agent_key, goal_type)]

        # Default behavior
        return f"{agent_key}_{goal_type}"


# Factory function
def create_comprehensive_goal_assignment_system(
    orchestrator_handler,
    notification_system: SMARTGoalNotificationSystem
) -> ComprehensiveGoalAssignmentSystem:
    """Create comprehensive goal assignment system"""
    return ComprehensiveGoalAssignmentSystem(orchestrator_handler, notification_system)

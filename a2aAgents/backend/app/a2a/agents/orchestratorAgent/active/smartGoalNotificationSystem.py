"""
SMART Goal Notification System for Orchestrator Agent
Handles goal assignment notifications and metric mapping for agents
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.networkClient import A2ANetworkClient

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of goal conflicts"""
    RESOURCE = "resource"  # Competing for same resources/metrics
    TIMELINE = "timeline"  # Same completion dates
    PRIORITY = "priority"  # Multiple critical priorities
    OBJECTIVE = "objective"  # Conflicting objectives
    COLLABORATIVE = "collaborative"  # Conflicts in collaborative goals

class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    LOW = "low"  # Can coexist with minor adjustments
    MEDIUM = "medium"  # Requires coordination
    HIGH = "high"  # Requires resolution before proceeding
    CRITICAL = "critical"  # Blocks goal execution

@dataclass
class GoalConflict:
    """Represents a conflict between goals"""
    conflict_id: str
    goal1_id: str
    goal2_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    detected_at: datetime
    resolution_suggestions: List[Dict[str, Any]]
    is_resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

@dataclass
class AgentCapabilities:
    """Agent's metric collection capabilities"""
    agent_id: str
    available_metrics: List[str]
    collection_frequency: str
    last_updated: datetime
    notification_preferences: Dict[str, Any]
    resource_constraints: Dict[str, Any] = None  # CPU, memory, bandwidth limits
    current_load: Dict[str, float] = None  # Current resource utilization

@dataclass
class SMARTGoalTemplate:
    """Template for creating SMART goals"""
    goal_type: str
    specific_template: str
    measurable_metrics: List[str]
    achievable_criteria: Dict[str, Any]
    relevant_context: str
    time_bound_options: List[str]

class SMARTGoalNotificationSystem:
    """Manages SMART goal assignments and agent notifications"""

    def __init__(self, orchestrator_handler):
        self.orchestrator = orchestrator_handler
        self.registered_agents: Dict[str, AgentCapabilities] = {}
        self.goal_templates: Dict[str, SMARTGoalTemplate] = {}
        self.active_notifications: Dict[str, Dict[str, Any]] = {}
        self.detected_conflicts: Dict[str, GoalConflict] = {}
        self.resolution_history: List[Dict[str, Any]] = []

        # Initialize goal templates
        self._initialize_goal_templates()

    def _initialize_goal_templates(self):
        """Initialize SMART goal templates for different agent types"""

        # Agent 0 (Data Product Agent) Templates
        self.goal_templates["agent0_performance"] = SMARTGoalTemplate(
            goal_type="performance",
            specific_template="Achieve {target_rate}% {metric_name} with {time_constraint} processing time",
            measurable_metrics=[
                "registration_success_rate",
                "avg_registration_time",
                "validation_accuracy",
                "throughput_per_hour"
            ],
            achievable_criteria={
                "registration_success_rate": {"min": 85.0, "max": 99.9},
                "avg_registration_time": {"min": 0.5, "max": 10.0},
                "validation_accuracy": {"min": 90.0, "max": 99.9},
                "throughput_per_hour": {"min": 50, "max": 500}
            },
            relevant_context="Critical for data product registration and validation efficiency",
            time_bound_options=["7 days", "14 days", "30 days", "45 days", "60 days"]
        )

        self.goal_templates["agent0_quality"] = SMARTGoalTemplate(
            goal_type="quality",
            specific_template="Maintain {target_score}+ {quality_metric} with {compliance_rate}% compliance",
            measurable_metrics=[
                "data_quality_score",
                "schema_compliance_rate",
                "dublin_core_compliance",
                "compliance_violations"
            ],
            achievable_criteria={
                "data_quality_score": {"min": 70.0, "max": 100.0},
                "schema_compliance_rate": {"min": 90.0, "max": 100.0},
                "dublin_core_compliance": {"min": 85.0, "max": 100.0},
                "compliance_violations": {"min": 0, "max": 5}
            },
            relevant_context="Ensures high-quality data products meet enterprise standards",
            time_bound_options=["14 days", "30 days", "45 days", "90 days"]
        )

        self.goal_templates["agent0_reliability"] = SMARTGoalTemplate(
            goal_type="reliability",
            specific_template="Achieve {availability}% uptime with <{error_rate}% error rate",
            measurable_metrics=[
                "api_availability",
                "error_rate",
                "queue_depth",
                "processing_time_p95"
            ],
            achievable_criteria={
                "api_availability": {"min": 95.0, "max": 99.99},
                "error_rate": {"min": 0.1, "max": 10.0},
                "queue_depth": {"min": 0, "max": 20},
                "processing_time_p95": {"min": 1.0, "max": 30.0}
            },
            relevant_context="Ensures reliable service for enterprise data management",
            time_bound_options=["30 days", "60 days", "90 days"]
        )

        # Agent 1 (Data Standardization) Templates
        self.goal_templates["agent1_transformation"] = SMARTGoalTemplate(
            goal_type="transformation",
            specific_template="Achieve {success_rate}% standardization success with {processing_time}s avg time",
            measurable_metrics=[
                "standardization_success_rate",
                "avg_transformation_time",
                "schema_mapping_accuracy",
                "data_enrichment_rate"
            ],
            achievable_criteria={
                "standardization_success_rate": {"min": 90.0, "max": 99.9},
                "avg_transformation_time": {"min": 0.1, "max": 5.0},
                "schema_mapping_accuracy": {"min": 95.0, "max": 100.0},
                "data_enrichment_rate": {"min": 70.0, "max": 95.0}
            },
            relevant_context="Critical for maintaining data consistency across the enterprise",
            time_bound_options=["7 days", "14 days", "30 days", "60 days"]
        )

        self.goal_templates["agent1_compliance"] = SMARTGoalTemplate(
            goal_type="compliance",
            specific_template="Maintain {compliance_rate}% canonical format compliance with {validation_accuracy}% accuracy",
            measurable_metrics=[
                "canonical_compliance_rate",
                "validation_accuracy",
                "format_consistency_score",
                "metadata_completeness"
            ],
            achievable_criteria={
                "canonical_compliance_rate": {"min": 95.0, "max": 100.0},
                "validation_accuracy": {"min": 98.0, "max": 100.0},
                "format_consistency_score": {"min": 90.0, "max": 100.0},
                "metadata_completeness": {"min": 85.0, "max": 100.0}
            },
            relevant_context="Ensures all data conforms to enterprise canonical standards",
            time_bound_options=["14 days", "30 days", "45 days", "90 days"]
        )

        # Agent 2 (AI Data Preparation) Templates
        self.goal_templates["agent2_feature_engineering"] = SMARTGoalTemplate(
            goal_type="feature_engineering",
            specific_template="Generate {feature_quality}% quality features with {feature_coverage}% coverage",
            measurable_metrics=[
                "feature_quality_score",
                "feature_coverage_rate",
                "feature_generation_speed",
                "ml_readiness_score"
            ],
            achievable_criteria={
                "feature_quality_score": {"min": 85.0, "max": 99.0},
                "feature_coverage_rate": {"min": 80.0, "max": 100.0},
                "feature_generation_speed": {"min": 100, "max": 1000},  # features/minute
                "ml_readiness_score": {"min": 90.0, "max": 100.0}
            },
            relevant_context="Prepares data for advanced ML/AI model training and inference",
            time_bound_options=["7 days", "14 days", "30 days"]
        )

        self.goal_templates["agent2_privacy"] = SMARTGoalTemplate(
            goal_type="privacy_preservation",
            specific_template="Achieve {privacy_score}% privacy preservation with {anonymization_rate}% anonymization",
            measurable_metrics=[
                "privacy_preservation_score",
                "anonymization_success_rate",
                "pii_detection_accuracy",
                "differential_privacy_epsilon"
            ],
            achievable_criteria={
                "privacy_preservation_score": {"min": 95.0, "max": 100.0},
                "anonymization_success_rate": {"min": 98.0, "max": 100.0},
                "pii_detection_accuracy": {"min": 99.0, "max": 100.0},
                "differential_privacy_epsilon": {"min": 0.1, "max": 10.0}
            },
            relevant_context="Ensures AI-ready data maintains privacy and regulatory compliance",
            time_bound_options=["14 days", "30 days", "60 days"]
        )

        # Agent 3 (Vector Processing) Templates
        self.goal_templates["agent3_embedding"] = SMARTGoalTemplate(
            goal_type="embedding_generation",
            specific_template="Generate embeddings with {quality_score}% quality at {throughput} vectors/sec",
            measurable_metrics=[
                "embedding_quality_score",
                "vector_generation_throughput",
                "dimensionality_reduction_ratio",
                "semantic_accuracy"
            ],
            achievable_criteria={
                "embedding_quality_score": {"min": 85.0, "max": 99.0},
                "vector_generation_throughput": {"min": 100, "max": 10000},
                "dimensionality_reduction_ratio": {"min": 0.1, "max": 0.9},
                "semantic_accuracy": {"min": 90.0, "max": 99.5}
            },
            relevant_context="Enables semantic search and similarity-based operations",
            time_bound_options=["7 days", "14 days", "30 days"]
        )

        self.goal_templates["agent3_indexing"] = SMARTGoalTemplate(
            goal_type="vector_indexing",
            specific_template="Maintain {index_coverage}% index coverage with {query_speed}ms query time",
            measurable_metrics=[
                "index_coverage_rate",
                "avg_query_time_ms",
                "index_freshness_hours",
                "similarity_search_accuracy"
            ],
            achievable_criteria={
                "index_coverage_rate": {"min": 95.0, "max": 100.0},
                "avg_query_time_ms": {"min": 1, "max": 100},
                "index_freshness_hours": {"min": 0.1, "max": 24},
                "similarity_search_accuracy": {"min": 90.0, "max": 99.9}
            },
            relevant_context="Enables fast and accurate vector similarity searches",
            time_bound_options=["14 days", "30 days", "45 days"]
        )

        # Agent 4 (Calculation Validation) Templates
        self.goal_templates["agent4_validation"] = SMARTGoalTemplate(
            goal_type="calculation_validation",
            specific_template="Achieve {validation_accuracy}% accuracy with {validation_speed} validations/sec",
            measurable_metrics=[
                "mathematical_validation_accuracy",
                "validation_throughput",
                "false_positive_rate",
                "symbolic_computation_success"
            ],
            achievable_criteria={
                "mathematical_validation_accuracy": {"min": 99.0, "max": 100.0},
                "validation_throughput": {"min": 10, "max": 1000},
                "false_positive_rate": {"min": 0.01, "max": 5.0},
                "symbolic_computation_success": {"min": 95.0, "max": 100.0}
            },
            relevant_context="Ensures mathematical correctness and computational integrity",
            time_bound_options=["7 days", "14 days", "30 days"]
        )

        # Agent 5 (QA Validation) Templates
        self.goal_templates["agent5_quality"] = SMARTGoalTemplate(
            goal_type="quality_assurance",
            specific_template="Maintain {qa_pass_rate}% QA pass rate with {review_time}hr avg review time",
            measurable_metrics=[
                "qa_pass_rate",
                "avg_review_time_hours",
                "defect_detection_rate",
                "compliance_check_score"
            ],
            achievable_criteria={
                "qa_pass_rate": {"min": 90.0, "max": 99.9},
                "avg_review_time_hours": {"min": 0.5, "max": 48},
                "defect_detection_rate": {"min": 95.0, "max": 100.0},
                "compliance_check_score": {"min": 98.0, "max": 100.0}
            },
            relevant_context="Final quality gate ensuring data meets all standards",
            time_bound_options=["14 days", "30 days", "60 days"]
        )

        # Agent 6 (Quality Control Manager) Templates
        self.goal_templates["agent6_monitoring"] = SMARTGoalTemplate(
            goal_type="continuous_monitoring",
            specific_template="Monitor with {detection_rate}% issue detection and {mttr}min mean time to repair",
            measurable_metrics=[
                "issue_detection_rate",
                "mean_time_to_repair",
                "false_alarm_rate",
                "monitoring_coverage"
            ],
            achievable_criteria={
                "issue_detection_rate": {"min": 95.0, "max": 99.9},
                "mean_time_to_repair": {"min": 1, "max": 60},
                "false_alarm_rate": {"min": 0.1, "max": 5.0},
                "monitoring_coverage": {"min": 90.0, "max": 100.0}
            },
            relevant_context="Proactive quality management across the data pipeline",
            time_bound_options=["30 days", "60 days", "90 days"]
        )

        # Agent 7 (Agent Manager) Templates
        self.goal_templates["agent7_management"] = SMARTGoalTemplate(
            goal_type="agent_management",
            specific_template="Maintain {agent_uptime}% agent uptime with {deployment_success}% deployment success",
            measurable_metrics=[
                "agent_uptime_percentage",
                "deployment_success_rate",
                "health_check_response_time",
                "resource_utilization"
            ],
            achievable_criteria={
                "agent_uptime_percentage": {"min": 99.0, "max": 99.99},
                "deployment_success_rate": {"min": 95.0, "max": 100.0},
                "health_check_response_time": {"min": 10, "max": 1000},  # ms
                "resource_utilization": {"min": 20.0, "max": 80.0}
            },
            relevant_context="Central management of all A2A network agents",
            time_bound_options=["30 days", "60 days", "90 days", "180 days"]
        )

        # Agent 8 (Data Manager) Templates
        self.goal_templates["agent8_storage"] = SMARTGoalTemplate(
            goal_type="data_storage",
            specific_template="Achieve {storage_efficiency}% efficiency with {retrieval_time}ms retrieval time",
            measurable_metrics=[
                "storage_efficiency_ratio",
                "avg_retrieval_time_ms",
                "data_compression_ratio",
                "cache_hit_rate"
            ],
            achievable_criteria={
                "storage_efficiency_ratio": {"min": 70.0, "max": 95.0},
                "avg_retrieval_time_ms": {"min": 1, "max": 500},
                "data_compression_ratio": {"min": 2.0, "max": 10.0},
                "cache_hit_rate": {"min": 80.0, "max": 99.0}
            },
            relevant_context="Centralized data storage and retrieval optimization",
            time_bound_options=["14 days", "30 days", "60 days"]
        )

        # Agent 9 (Reasoning Agent) Templates
        self.goal_templates["agent9_reasoning"] = SMARTGoalTemplate(
            goal_type="logical_reasoning",
            specific_template="Achieve {reasoning_accuracy}% accuracy with {inference_speed} inferences/sec",
            measurable_metrics=[
                "reasoning_accuracy",
                "inference_throughput",
                "logic_consistency_score",
                "explanation_quality"
            ],
            achievable_criteria={
                "reasoning_accuracy": {"min": 90.0, "max": 99.5},
                "inference_throughput": {"min": 1, "max": 100},
                "logic_consistency_score": {"min": 95.0, "max": 100.0},
                "explanation_quality": {"min": 85.0, "max": 100.0}
            },
            relevant_context="Advanced reasoning and decision-making capabilities",
            time_bound_options=["14 days", "30 days", "45 days"]
        )

        # Agent 10 (Calculation Agent) Templates
        self.goal_templates["agent10_computation"] = SMARTGoalTemplate(
            goal_type="complex_calculation",
            specific_template="Process {calculation_accuracy}% accurately with {calc_throughput} calculations/sec",
            measurable_metrics=[
                "calculation_accuracy",
                "calculation_throughput",
                "numerical_stability_score",
                "self_healing_success_rate"
            ],
            achievable_criteria={
                "calculation_accuracy": {"min": 99.9, "max": 100.0},
                "calculation_throughput": {"min": 10, "max": 10000},
                "numerical_stability_score": {"min": 95.0, "max": 100.0},
                "self_healing_success_rate": {"min": 90.0, "max": 100.0}
            },
            relevant_context="High-precision mathematical and statistical computations",
            time_bound_options=["7 days", "14 days", "30 days"]
        )

        # Agent 11 (SQL Agent) Templates
        self.goal_templates["agent11_query"] = SMARTGoalTemplate(
            goal_type="sql_operations",
            specific_template="Convert {nl2sql_accuracy}% accurately with {query_optimization}% optimization",
            measurable_metrics=[
                "nl2sql_accuracy",
                "query_optimization_rate",
                "avg_query_execution_time",
                "sql_injection_prevention"
            ],
            achievable_criteria={
                "nl2sql_accuracy": {"min": 85.0, "max": 98.0},
                "query_optimization_rate": {"min": 70.0, "max": 95.0},
                "avg_query_execution_time": {"min": 10, "max": 5000},  # ms
                "sql_injection_prevention": {"min": 99.9, "max": 100.0}
            },
            relevant_context="Natural language to SQL conversion and database operations",
            time_bound_options=["14 days", "30 days", "45 days"]
        )

        # Agent 12 (Catalog Manager) Templates
        self.goal_templates["agent12_catalog"] = SMARTGoalTemplate(
            goal_type="service_catalog",
            specific_template="Maintain {catalog_completeness}% completeness with {discovery_time}s discovery time",
            measurable_metrics=[
                "catalog_completeness",
                "avg_discovery_time",
                "ord_compliance_rate",
                "metadata_accuracy"
            ],
            achievable_criteria={
                "catalog_completeness": {"min": 95.0, "max": 100.0},
                "avg_discovery_time": {"min": 0.1, "max": 30.0},
                "ord_compliance_rate": {"min": 98.0, "max": 100.0},
                "metadata_accuracy": {"min": 95.0, "max": 100.0}
            },
            relevant_context="Service discovery and resource catalog management",
            time_bound_options=["30 days", "60 days", "90 days"]
        )

        # Agent 13 (Agent Builder) Templates
        self.goal_templates["agent13_builder"] = SMARTGoalTemplate(
            goal_type="agent_creation",
            specific_template="Build agents with {build_success}% success and {deployment_time}min deployment",
            measurable_metrics=[
                "agent_build_success_rate",
                "avg_deployment_time",
                "code_quality_score",
                "test_coverage"
            ],
            achievable_criteria={
                "agent_build_success_rate": {"min": 90.0, "max": 100.0},
                "avg_deployment_time": {"min": 1, "max": 60},
                "code_quality_score": {"min": 85.0, "max": 100.0},
                "test_coverage": {"min": 80.0, "max": 100.0}
            },
            relevant_context="Dynamic agent creation and deployment automation",
            time_bound_options=["7 days", "14 days", "30 days"]
        )

        # Agent 14 (Embedding Fine-Tuner) Templates
        self.goal_templates["agent14_finetuning"] = SMARTGoalTemplate(
            goal_type="model_finetuning",
            specific_template="Achieve {model_improvement}% improvement with {training_efficiency}% efficiency",
            measurable_metrics=[
                "model_performance_improvement",
                "training_efficiency",
                "embedding_quality_score",
                "convergence_speed"
            ],
            achievable_criteria={
                "model_performance_improvement": {"min": 5.0, "max": 50.0},
                "training_efficiency": {"min": 70.0, "max": 95.0},
                "embedding_quality_score": {"min": 85.0, "max": 99.0},
                "convergence_speed": {"min": 10, "max": 1000}  # epochs
            },
            relevant_context="Optimize embedding models for domain-specific tasks",
            time_bound_options=["14 days", "30 days", "60 days"]
        )

        # Agent 15 (Orchestrator) Templates
        self.goal_templates["agent15_orchestration"] = SMARTGoalTemplate(
            goal_type="workflow_orchestration",
            specific_template="Orchestrate with {workflow_success}% success and {scheduling_efficiency}% efficiency",
            measurable_metrics=[
                "workflow_success_rate",
                "scheduling_efficiency",
                "pipeline_throughput",
                "resource_optimization_score"
            ],
            achievable_criteria={
                "workflow_success_rate": {"min": 95.0, "max": 99.9},
                "scheduling_efficiency": {"min": 85.0, "max": 99.0},
                "pipeline_throughput": {"min": 10, "max": 1000},  # workflows/hour
                "resource_optimization_score": {"min": 80.0, "max": 100.0}
            },
            relevant_context="Central workflow coordination and task scheduling",
            time_bound_options=["30 days", "60 days", "90 days"]
        )

    async def register_agent_for_notifications(self, agent_id: str, capabilities_data: Dict[str, Any]):
        """Register an agent for goal notifications"""
        try:
            capabilities = AgentCapabilities(
                agent_id=agent_id,
                available_metrics=capabilities_data.get("metrics_capabilities", []),
                collection_frequency=capabilities_data.get("collection_frequency", "hourly"),
                last_updated=datetime.utcnow(),
                notification_preferences=capabilities_data.get("notification_preferences", {})
            )

            self.registered_agents[agent_id] = capabilities

            logger.info(f"Registered {agent_id} for goal notifications")
            logger.info(f"Available metrics: {len(capabilities.available_metrics)}")

            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def create_smart_goal(self, agent_id: str, goal_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a SMART goal for an agent"""
        try:
            # Get agent capabilities
            agent_capabilities = self.registered_agents.get(agent_id)
            if not agent_capabilities:
                raise ValueError(f"Agent {agent_id} not registered for notifications")

            # Get goal template
            template_key = f"{agent_id.split('_')[0]}_{goal_type}"
            template = self.goal_templates.get(template_key)
            if not template:
                raise ValueError(f"No template found for {template_key}")

            # Validate metrics are available
            required_metrics = parameters.get("measurable", {}).keys()
            available_metrics = set(agent_capabilities.available_metrics)
            missing_metrics = set(required_metrics) - available_metrics

            if missing_metrics:
                raise ValueError(f"Agent {agent_id} cannot provide metrics: {missing_metrics}")

            # Create SMART goal
            goal_id = f"{agent_id}_{goal_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Validate achievability
            measurable_targets = parameters.get("measurable", {})
            for metric, target in measurable_targets.items():
                if metric in template.achievable_criteria:
                    criteria = template.achievable_criteria[metric]
                    if not (criteria["min"] <= target <= criteria["max"]):
                        logger.warning(f"Target {target} for {metric} may not be achievable (range: {criteria['min']}-{criteria['max']})")

            smart_goal = {
                "goal_id": goal_id,
                "agent_id": agent_id,
                "goal_type": goal_type,

                # SMART Criteria
                "specific": parameters.get("specific", template.specific_template),
                "measurable": measurable_targets,
                "achievable": self._assess_achievability(agent_id, measurable_targets, template),
                "relevant": parameters.get("relevant", template.relevant_context),
                "time_bound": parameters.get("time_bound", "30 days"),

                # Tracking Information
                "assigned_date": datetime.utcnow().isoformat(),
                "target_date": self._calculate_target_date(parameters.get("time_bound", "30 days")),
                "tracking_frequency": parameters.get("tracking_frequency", agent_capabilities.collection_frequency),
                "created_by": "orchestrator_agent",
                "status": "assigned"
            }

            return smart_goal

        except Exception as e:
            logger.error(f"Failed to create SMART goal: {e}")
            raise

    def _assess_achievability(self, agent_id: str, targets: Dict[str, Any], template: SMARTGoalTemplate) -> bool:
        """Assess if the goal targets are achievable"""
        try:
            for metric, target in targets.items():
                if metric in template.achievable_criteria:
                    criteria = template.achievable_criteria[metric]
                    if not (criteria["min"] <= target <= criteria["max"]):
                        return False
            return True
        except Exception:
            return False

    def _calculate_target_date(self, time_bound: str) -> str:
        """Calculate target date from time bound string"""
        try:
            if "day" in time_bound:
                days = int(time_bound.split()[0])
                target_date = datetime.utcnow() + timedelta(days=days)
            elif "week" in time_bound:
                weeks = int(time_bound.split()[0])
                target_date = datetime.utcnow() + timedelta(weeks=weeks)
            elif "month" in time_bound:
                months = int(time_bound.split()[0])
                target_date = datetime.utcnow() + timedelta(days=months * 30)
            else:
                # Default to 30 days
                target_date = datetime.utcnow() + timedelta(days=30)

            return target_date.isoformat()

        except Exception:
            return (datetime.utcnow() + timedelta(days=30)).isoformat()

    async def send_goal_assignment(self, agent_id: str, smart_goal: Dict[str, Any]) -> bool:
        """Send SMART goal assignment to agent"""
        try:
            # Create A2A message for goal assignment
            assignment_message = {
                "operation": "goal_assignment",
                "data": smart_goal
            }

            # Send through orchestrator's A2A client
            await self.orchestrator.a2a_client.send_message(
                recipient_id=agent_id,
                message_data=assignment_message
            )

            # Track notification
            self.active_notifications[smart_goal["goal_id"]] = {
                "agent_id": agent_id,
                "goal_id": smart_goal["goal_id"],
                "sent_at": datetime.utcnow().isoformat(),
                "status": "sent",
                "acknowledged": False
            }

            logger.info(f"Sent SMART goal assignment {smart_goal['goal_id']} to {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send goal assignment to {agent_id}: {e}")
            return False

    async def handle_goal_acknowledgment(self, agent_id: str, goal_id: str, ack_data: Dict[str, Any]):
        """Handle goal assignment acknowledgment from agent"""
        try:
            if goal_id in self.active_notifications:
                self.active_notifications[goal_id].update({
                    "acknowledged": True,
                    "acknowledged_at": ack_data.get("acknowledged_at"),
                    "baseline_metrics_collected": ack_data.get("baseline_metrics_collected", False),
                    "tracking_active": ack_data.get("tracking_active", False)
                })

                logger.info(f"Goal {goal_id} acknowledged by {agent_id}")

                # Update goal status in orchestrator
                if hasattr(self.orchestrator, 'agent_goals') and agent_id in self.orchestrator.agent_goals:
                    goals = self.orchestrator.agent_goals[agent_id].get("goals", {})
                    if goal_id in goals:
                        goals[goal_id]["status"] = "acknowledged"
                        goals[goal_id]["acknowledged_at"] = ack_data.get("acknowledged_at")

                return True
            else:
                logger.warning(f"Received acknowledgment for unknown goal {goal_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to handle goal acknowledgment: {e}")
            return False

    def get_agent_metric_mapping(self, agent_id: str) -> Dict[str, Any]:
        """Get available metrics mapping for an agent"""
        agent_capabilities = self.registered_agents.get(agent_id)
        if not agent_capabilities:
            return {}

        # Map metrics to their descriptions and types
        metric_mapping = {}

        if agent_id.startswith("agent0"):
            metric_mapping = {
                # Performance Metrics
                "data_products_registered": {
                    "description": "Total number of data products registered",
                    "type": "counter",
                    "unit": "count",
                    "goal_relevance": ["performance", "business"]
                },
                "registration_success_rate": {
                    "description": "Percentage of successful registrations",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["performance", "quality"]
                },
                "avg_registration_time": {
                    "description": "Average time to register a data product",
                    "type": "duration",
                    "unit": "seconds",
                    "goal_relevance": ["performance", "efficiency"]
                },
                "validation_accuracy": {
                    "description": "Accuracy of data validation processes",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["quality", "reliability"]
                },

                # Quality Metrics
                "schema_compliance_rate": {
                    "description": "Percentage of data meeting schema requirements",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["quality", "compliance"]
                },
                "data_quality_score": {
                    "description": "Overall data quality assessment score",
                    "type": "score",
                    "unit": "0-100",
                    "goal_relevance": ["quality", "business"]
                },
                "dublin_core_compliance": {
                    "description": "Dublin Core metadata compliance percentage",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["compliance", "quality"]
                },

                # System Metrics
                "api_availability": {
                    "description": "API uptime percentage",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["reliability", "performance"]
                },
                "error_rate": {
                    "description": "Percentage of requests resulting in errors",
                    "type": "percentage",
                    "unit": "%",
                    "goal_relevance": ["reliability", "quality"]
                },
                "throughput_per_hour": {
                    "description": "Number of registrations processed per hour",
                    "type": "rate",
                    "unit": "per hour",
                    "goal_relevance": ["performance", "scalability"]
                }
            }

        # Filter to only include available metrics
        available_mapping = {
            metric: details
            for metric, details in metric_mapping.items()
            if metric in agent_capabilities.available_metrics
        }

        return {
            "agent_id": agent_id,
            "total_available_metrics": len(available_mapping),
            "collection_frequency": agent_capabilities.collection_frequency,
            "last_updated": agent_capabilities.last_updated.isoformat(),
            "metrics": available_mapping
        }

    def suggest_smart_goals(self, agent_id: str, current_metrics: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Suggest SMART goals based on agent capabilities and current performance"""
        suggestions = []

        try:
            agent_capabilities = self.registered_agents.get(agent_id)
            if not agent_capabilities:
                return suggestions

            # Get relevant templates for this agent
            agent_prefix = agent_id.split('_')[0]
            relevant_templates = {
                k: v for k, v in self.goal_templates.items()
                if k.startswith(agent_prefix)
            }

            for template_key, template in relevant_templates.items():
                goal_type = template.goal_type

                # Create suggestion based on template
                suggestion = {
                    "goal_type": goal_type,
                    "template": template_key,
                    "specific": f"Improve {goal_type} metrics for {agent_id}",
                    "measurable_options": {},
                    "time_bound_options": template.time_bound_options,
                    "relevant": template.relevant_context,
                    "achievable": True
                }

                # Suggest realistic targets based on current metrics or defaults
                for metric in template.measurable_metrics:
                    if metric in agent_capabilities.available_metrics:
                        criteria = template.achievable_criteria.get(metric, {})

                        if current_metrics and metric in current_metrics:
                            current_value = current_metrics[metric]
                            # Suggest 10-20% improvement from current
                            if metric in ["error_rate", "compliance_violations"]:
                                # Lower is better
                                suggested_target = max(criteria.get("min", 0), current_value * 0.8)
                            else:
                                # Higher is better
                                suggested_target = min(criteria.get("max", 100), current_value * 1.15)
                        else:
                            # Use middle of achievable range
                            min_val = criteria.get("min", 0)
                            max_val = criteria.get("max", 100)
                            suggested_target = (min_val + max_val) / 2

                        suggestion["measurable_options"][metric] = {
                            "suggested_target": suggested_target,
                            "achievable_range": criteria,
                            "current_value": current_metrics.get(metric) if current_metrics else None
                        }

                suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"Failed to suggest SMART goals for {agent_id}: {e}")
            return suggestions

    def get_notification_status(self) -> Dict[str, Any]:
        """Get status of all goal notifications"""
        return {
            "registered_agents": len(self.registered_agents),
            "active_notifications": len(self.active_notifications),
            "acknowledged_goals": len([
                n for n in self.active_notifications.values()
                if n.get("acknowledged", False)
            ]),
            "goal_templates": len(self.goal_templates),
            "agents": {
                agent_id: {
                    "available_metrics": len(caps.available_metrics),
                    "collection_frequency": caps.collection_frequency,
                    "last_updated": caps.last_updated.isoformat()
                }
                for agent_id, caps in self.registered_agents.items()
            },
            "notifications": list(self.active_notifications.values()),
            "conflicts": {
                "total": len(self.detected_conflicts),
                "unresolved": len([c for c in self.detected_conflicts.values() if not c.is_resolved]),
                "by_severity": self._group_conflicts_by_severity(),
                "by_type": self._group_conflicts_by_type()
            }
        }

    async def detect_goal_conflicts(self, new_goal: Dict[str, Any], agent_id: str) -> List[GoalConflict]:
        """Detect conflicts between a new goal and existing goals"""
        conflicts = []

        try:
            # Get all active goals across agents
            all_goals = await self._get_all_active_goals()

            for existing_goal in all_goals:
                if existing_goal["agent_id"] == agent_id:
                    # Check intra-agent conflicts
                    conflict = self._check_intra_agent_conflict(new_goal, existing_goal)
                else:
                    # Check inter-agent conflicts
                    conflict = self._check_inter_agent_conflict(new_goal, existing_goal, agent_id)

                if conflict:
                    conflicts.append(conflict)

            # Store detected conflicts
            for conflict in conflicts:
                self.detected_conflicts[conflict.conflict_id] = conflict

            return conflicts

        except Exception as e:
            logger.error(f"Failed to detect goal conflicts: {e}")
            return conflicts

    def _check_intra_agent_conflict(self, new_goal: Dict[str, Any], existing_goal: Dict[str, Any]) -> Optional[GoalConflict]:
        """Check for conflicts within the same agent"""
        conflict_checks = [
            self._check_resource_conflict,
            self._check_timeline_conflict,
            self._check_priority_conflict,
            self._check_objective_conflict
        ]

        for check in conflict_checks:
            conflict = check(new_goal, existing_goal)
            if conflict:
                return conflict

        return None

    def _check_inter_agent_conflict(self, new_goal: Dict[str, Any], existing_goal: Dict[str, Any], agent_id: str) -> Optional[GoalConflict]:
        """Check for conflicts between different agents"""
        # Check if goals involve collaboration
        new_collaborators = new_goal.get("collaborative_agents", [])
        existing_collaborators = existing_goal.get("collaborative_agents", [])

        if existing_goal["agent_id"] in new_collaborators or agent_id in existing_collaborators:
            # Check collaborative conflicts
            return self._check_collaborative_conflict(new_goal, existing_goal, agent_id)

        # Check for shared resource conflicts
        return self._check_shared_resource_conflict(new_goal, existing_goal, agent_id)

    def _check_resource_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> Optional[GoalConflict]:
        """Check if goals compete for the same resources/metrics"""
        metrics1 = set(goal1.get("measurable", {}).keys())
        metrics2 = set(goal2.get("measurable", {}).keys())

        overlapping_metrics = metrics1 & metrics2

        if overlapping_metrics:
            # Check if the overlapping metrics have conflicting targets
            conflicting_metrics = []
            for metric in overlapping_metrics:
                target1 = goal1["measurable"][metric]
                target2 = goal2["measurable"][metric]

                # If targets differ significantly (>20%), it's a conflict
                if abs(target1 - target2) / max(target1, target2) > 0.2:
                    conflicting_metrics.append(metric)

            if conflicting_metrics:
                return GoalConflict(
                    conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                    goal1_id=goal1.get("goal_id", "new"),
                    goal2_id=goal2.get("goal_id"),
                    conflict_type=ConflictType.RESOURCE,
                    severity=ConflictSeverity.MEDIUM if len(conflicting_metrics) < 2 else ConflictSeverity.HIGH,
                    description=f"Conflicting targets for metrics: {', '.join(conflicting_metrics)}",
                    detected_at=datetime.utcnow(),
                    resolution_suggestions=[
                        {"action": "adjust_targets", "description": "Align metric targets between goals"},
                        {"action": "prioritize", "description": "Set one goal as higher priority"},
                        {"action": "sequence", "description": "Execute goals sequentially instead of parallel"}
                    ]
                )

        return None

    def _check_timeline_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> Optional[GoalConflict]:
        """Check if goals have conflicting timelines"""
        time1 = goal1.get("time_bound")
        time2 = goal2.get("time_bound")

        if time1 == time2 and goal1.get("priority") == "critical" and goal2.get("priority") == "critical":
            return GoalConflict(
                conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                goal1_id=goal1.get("goal_id", "new"),
                goal2_id=goal2.get("goal_id"),
                conflict_type=ConflictType.TIMELINE,
                severity=ConflictSeverity.HIGH,
                description="Both critical goals have the same deadline",
                detected_at=datetime.utcnow(),
                resolution_suggestions=[
                    {"action": "stagger_deadlines", "description": "Adjust one deadline by 7-14 days"},
                    {"action": "allocate_resources", "description": "Ensure sufficient resources for both"},
                    {"action": "delegate", "description": "Assign to different team members"}
                ]
            )

        return None

    def _check_priority_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> Optional[GoalConflict]:
        """Check if there are too many high-priority goals"""
        if goal1.get("priority") == "critical" and goal2.get("priority") == "critical":
            if goal1.get("status") == "active" and goal2.get("status") != "completed":
                return GoalConflict(
                    conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                    goal1_id=goal1.get("goal_id", "new"),
                    goal2_id=goal2.get("goal_id"),
                    conflict_type=ConflictType.PRIORITY,
                    severity=ConflictSeverity.MEDIUM,
                    description="Multiple critical priority goals active simultaneously",
                    detected_at=datetime.utcnow(),
                    resolution_suggestions=[
                        {"action": "reprioritize", "description": "Review and adjust priority levels"},
                        {"action": "focus", "description": "Complete one critical goal before starting another"},
                        {"action": "resources", "description": "Allocate additional resources"}
                    ]
                )

        return None

    def _check_objective_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> Optional[GoalConflict]:
        """Check if goals have conflicting objectives"""
        # Example: One goal optimizes for speed, another for accuracy
        objective_pairs = [
            ({"response_time", "throughput"}, {"accuracy", "quality_score"}),
            ({"cost_reduction"}, {"feature_expansion", "quality_improvement"}),
            ({"automation_rate"}, {"manual_review_rate"})
        ]

        metrics1 = set(goal1.get("measurable", {}).keys())
        metrics2 = set(goal2.get("measurable", {}).keys())

        for speed_metrics, quality_metrics in objective_pairs:
            if (metrics1 & speed_metrics and metrics2 & quality_metrics) or \
               (metrics2 & speed_metrics and metrics1 & quality_metrics):
                return GoalConflict(
                    conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                    goal1_id=goal1.get("goal_id", "new"),
                    goal2_id=goal2.get("goal_id"),
                    conflict_type=ConflictType.OBJECTIVE,
                    severity=ConflictSeverity.LOW,
                    description="Goals have potentially conflicting objectives (speed vs quality trade-off)",
                    detected_at=datetime.utcnow(),
                    resolution_suggestions=[
                        {"action": "balance", "description": "Find optimal balance between objectives"},
                        {"action": "phase", "description": "Phase goals to focus on one objective at a time"},
                        {"action": "segment", "description": "Apply different objectives to different segments"}
                    ]
                )

        return None

    def _check_collaborative_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any], agent_id: str) -> Optional[GoalConflict]:
        """Check for conflicts in collaborative goals"""
        # Check if agents have conflicting roles or responsibilities
        role1 = goal1.get("collaborative_role")
        role2 = goal2.get("collaborative_role")

        if role1 == "leader" and role2 == "leader":
            return GoalConflict(
                conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                goal1_id=goal1.get("goal_id", "new"),
                goal2_id=goal2.get("goal_id"),
                conflict_type=ConflictType.COLLABORATIVE,
                severity=ConflictSeverity.HIGH,
                description="Multiple agents assigned as leaders for collaborative goals",
                detected_at=datetime.utcnow(),
                resolution_suggestions=[
                    {"action": "reassign_roles", "description": "Designate clear leader and contributor roles"},
                    {"action": "co_leadership", "description": "Implement co-leadership with clear divisions"},
                    {"action": "hierarchy", "description": "Establish clear hierarchy for decision making"}
                ]
            )

        return None

    def _check_shared_resource_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any], agent_id: str) -> Optional[GoalConflict]:
        """Check if goals from different agents compete for shared resources"""
        # Check agent capabilities and current load
        agent1_caps = self.registered_agents.get(goal1["agent_id"])
        agent2_caps = self.registered_agents.get(agent_id)

        if agent1_caps and agent2_caps:
            # Check if both agents are near capacity
            if agent1_caps.current_load and agent2_caps.current_load:
                avg_load1 = sum(agent1_caps.current_load.values()) / len(agent1_caps.current_load)
                avg_load2 = sum(agent2_caps.current_load.values()) / len(agent2_caps.current_load)

                if avg_load1 > 0.8 and avg_load2 > 0.8:
                    return GoalConflict(
                        conflict_id=f"conflict_{datetime.utcnow().timestamp()}",
                        goal1_id=goal1.get("goal_id", "new"),
                        goal2_id=goal2.get("goal_id"),
                        conflict_type=ConflictType.RESOURCE,
                        severity=ConflictSeverity.CRITICAL,
                        description="Both agents near capacity, new goals may overload system",
                        detected_at=datetime.utcnow(),
                        resolution_suggestions=[
                            {"action": "defer", "description": "Defer one goal until resources are available"},
                            {"action": "scale", "description": "Scale up agent resources"},
                            {"action": "optimize", "description": "Optimize current workloads to free resources"}
                        ]
                    )

        return None

    async def resolve_conflict(self, conflict_id: str, resolution_action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a detected conflict"""
        try:
            conflict = self.detected_conflicts.get(conflict_id)
            if not conflict:
                return {"status": "error", "message": "Conflict not found"}

            if conflict.is_resolved:
                return {"status": "error", "message": "Conflict already resolved"}

            # Apply resolution based on action
            resolution_result = await self._apply_resolution(conflict, resolution_action, parameters)

            if resolution_result["status"] == "success":
                # Mark conflict as resolved
                conflict.is_resolved = True
                conflict.resolution = resolution_result["description"]
                conflict.resolved_at = datetime.utcnow()

                # Add to resolution history
                self.resolution_history.append({
                    "conflict_id": conflict_id,
                    "resolution_action": resolution_action,
                    "parameters": parameters,
                    "resolved_at": conflict.resolved_at,
                    "result": resolution_result
                })

            return resolution_result

        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def _apply_resolution(self, conflict: GoalConflict, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific resolution action"""
        resolution_handlers = {
            "adjust_targets": self._resolve_by_adjusting_targets,
            "prioritize": self._resolve_by_prioritizing,
            "sequence": self._resolve_by_sequencing,
            "stagger_deadlines": self._resolve_by_staggering_deadlines,
            "allocate_resources": self._resolve_by_allocating_resources,
            "reassign_roles": self._resolve_by_reassigning_roles
        }

        handler = resolution_handlers.get(action)
        if not handler:
            return {"status": "error", "message": f"Unknown resolution action: {action}"}

        return await handler(conflict, parameters)

    async def _resolve_by_adjusting_targets(self, conflict: GoalConflict, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by adjusting metric targets"""
        # Send notifications to affected agents to adjust targets
        adjustment_plan = parameters.get("adjustment_plan", {})

        notifications_sent = await self._send_target_adjustment_notifications(
            conflict.goal1_id,
            conflict.goal2_id,
            adjustment_plan
        )

        return {
            "status": "success",
            "description": f"Sent target adjustment notifications to affected agents",
            "details": {"notifications_sent": notifications_sent}
        }

    async def _resolve_by_prioritizing(self, conflict: GoalConflict, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by setting priority order"""
        priority_order = parameters.get("priority_order", [])

        if not priority_order:
            return {"status": "error", "message": "Priority order not specified"}

        # Update goal priorities through orchestrator
        for idx, goal_id in enumerate(priority_order):
            new_priority = "critical" if idx == 0 else "high" if idx == 1 else "medium"
            await self.orchestrator._update_goal_priority(goal_id, new_priority)

        return {
            "status": "success",
            "description": f"Updated goal priorities based on specified order",
            "details": {"priority_order": priority_order}
        }

    async def _resolve_by_sequencing(self, conflict: GoalConflict, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by creating sequential dependencies"""
        # Add dependency so goal2 depends on goal1 completion
        dependency_added = await self.orchestrator._add_goal_dependency(
            conflict.goal2_id,
            conflict.goal1_id,
            "prerequisite"
        )

        return {
            "status": "success" if dependency_added else "error",
            "description": f"Created sequential dependency: {conflict.goal2_id} depends on {conflict.goal1_id}",
            "details": {"dependency_added": dependency_added}
        }

    async def _get_all_active_goals(self) -> List[Dict[str, Any]]:
        """Get all active goals from orchestrator"""
        all_goals = []
        for agent_id, goal_record in self.orchestrator.agent_goals.items():
            if goal_record.get("status") == "active":
                goals = goal_record.get("goals", {}).get("primary_objectives", [])
                for goal in goals:
                    goal["agent_id"] = agent_id
                    all_goals.append(goal)
        return all_goals

    def _group_conflicts_by_severity(self) -> Dict[str, int]:
        """Group conflicts by severity level"""
        severity_counts = {s.value: 0 for s in ConflictSeverity}
        for conflict in self.detected_conflicts.values():
            if not conflict.is_resolved:
                severity_counts[conflict.severity.value] += 1
        return severity_counts

    def _group_conflicts_by_type(self) -> Dict[str, int]:
        """Group conflicts by type"""
        type_counts = {t.value: 0 for t in ConflictType}
        for conflict in self.detected_conflicts.values():
            if not conflict.is_resolved:
                type_counts[conflict.conflict_type.value] += 1
        return type_counts

    async def _send_target_adjustment_notifications(self, goal1_id: str, goal2_id: str, adjustment_plan: Dict[str, Any]) -> int:
        """Send notifications to agents about target adjustments"""
        # Implementation would send A2A messages to affected agents
        # For now, return mock success
        return 2  # Number of notifications sent

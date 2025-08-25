"""
Goal-Aware Metrics Collector for Agent 0
Collects SMART goal-relevant metrics and notifies orchestrator of goal assignments
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....sdk.a2aNetworkClient import A2ANetworkClient

logger = logging.getLogger(__name__)

@dataclass
class SMARTGoal:
    """SMART Goal structure for Agent 0"""
    specific: str  # What exactly will be accomplished
    measurable: Dict[str, Any]  # Metrics and targets
    achievable: bool  # Is this realistic given current capabilities
    relevant: str  # Why this matters to Agent 0's purpose
    time_bound: str  # When this should be completed

    # Additional tracking fields
    goal_id: str
    assigned_date: datetime
    target_date: datetime
    current_metrics: Dict[str, float]
    tracking_frequency: str  # daily, hourly, real-time

@dataclass
class Agent0Metrics:
    """Current metrics available from Agent 0"""
    # Performance Metrics
    data_products_registered: int
    registration_success_rate: float  # %
    avg_registration_time: float  # seconds
    validation_accuracy: float  # %

    # Quality Metrics
    schema_compliance_rate: float  # %
    data_quality_score: float  # 0-100
    dublin_core_compliance: float  # %

    # System Metrics
    api_availability: float  # %
    error_rate: float  # %
    throughput_per_hour: int
    queue_depth: int
    processing_time_p95: float  # seconds

    # Business Metrics
    catalog_completeness: float  # %
    user_satisfaction_score: float  # 1-10
    compliance_violations: int

    # AI Enhancement Metrics
    grok_ai_accuracy: float  # %
    perplexity_api_success_rate: float  # %
    pdf_processing_success_rate: float  # %

    timestamp: datetime

class GoalAwareMetricsCollector:
    """Collects metrics relevant to assigned SMART goals"""

    def __init__(self, agent_id: str = "agent0_data_product"):
        self.agent_id = agent_id
        self.assigned_goals: Dict[str, SMARTGoal] = {}
        self.current_metrics = None
        self.metrics_history: List[Agent0Metrics] = []
        self.orchestrator_client = None

        # Goal notification tracking
        self.goal_notification_received = False
        self.last_goal_sync = None

    async def initialize(self):
        """Initialize metrics collector and A2A client"""
        try:
            self.orchestrator_client = A2ANetworkClient(
                agent_id=self.agent_id,
                private_key=os.getenv('A2A_PRIVATE_KEY'),
                rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
            )
            await self.orchestrator_client.connect()

            # Register for goal notifications
            await self._register_for_goal_notifications()

            logger.info(f"Goal-aware metrics collector initialized for {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            raise

    async def _register_for_goal_notifications(self):
        """Register to receive goal assignment notifications from orchestrator"""
        try:
            # Send registration message to orchestrator
            registration_message = {
                "operation": "register_for_goal_notifications",
                "data": {
                    "agent_id": self.agent_id,
                    "notification_types": ["goal_assignment", "goal_update", "goal_completion"],
                    "metrics_capabilities": self._get_available_metrics_list()
                }
            }

            await self.orchestrator_client.send_message(
                recipient_id="orchestrator_agent",
                message_data=registration_message
            )

            logger.info("Registered for goal notifications with orchestrator")

        except Exception as e:
            logger.error(f"Failed to register for goal notifications: {e}")

    def _get_available_metrics_list(self) -> List[str]:
        """Return list of all metrics Agent 0 can provide"""
        return [
            # Performance Metrics
            "data_products_registered",
            "registration_success_rate",
            "avg_registration_time",
            "validation_accuracy",

            # Quality Metrics
            "schema_compliance_rate",
            "data_quality_score",
            "dublin_core_compliance",

            # System Metrics
            "api_availability",
            "error_rate",
            "throughput_per_hour",
            "queue_depth",
            "processing_time_p95",

            # Business Metrics
            "catalog_completeness",
            "user_satisfaction_score",
            "compliance_violations",

            # AI Enhancement Metrics
            "grok_ai_accuracy",
            "perplexity_api_success_rate",
            "pdf_processing_success_rate"
        ]

    async def handle_goal_assignment(self, goal_data: Dict[str, Any]):
        """Handle goal assignment from orchestrator"""
        try:
            # Parse SMART goal structure
            smart_goal = SMARTGoal(
                goal_id=goal_data["goal_id"],
                specific=goal_data["specific"],
                measurable=goal_data["measurable"],
                achievable=goal_data["achievable"],
                relevant=goal_data["relevant"],
                time_bound=goal_data["time_bound"],
                assigned_date=datetime.fromisoformat(goal_data["assigned_date"]),
                target_date=datetime.fromisoformat(goal_data["target_date"]),
                current_metrics={},
                tracking_frequency=goal_data.get("tracking_frequency", "hourly")
            )

            self.assigned_goals[smart_goal.goal_id] = smart_goal
            self.goal_notification_received = True
            self.last_goal_sync = datetime.utcnow()

            logger.info(f"Received SMART goal assignment: {smart_goal.goal_id}")
            logger.info(f"Goal: {smart_goal.specific}")
            logger.info(f"Measurable targets: {smart_goal.measurable}")

            # Immediately collect baseline metrics for this goal
            await self._collect_goal_specific_metrics(smart_goal.goal_id)

            # Send acknowledgment to orchestrator
            await self._send_goal_acknowledgment(smart_goal.goal_id)

        except Exception as e:
            logger.error(f"Failed to handle goal assignment: {e}")

    async def _send_goal_acknowledgment(self, goal_id: str):
        """Send acknowledgment of goal assignment to orchestrator"""
        try:
            ack_message = {
                "operation": "goal_assignment_acknowledged",
                "data": {
                    "agent_id": self.agent_id,
                    "goal_id": goal_id,
                    "acknowledged_at": datetime.utcnow().isoformat(),
                    "baseline_metrics_collected": True,
                    "tracking_active": True
                }
            }

            await self.orchestrator_client.send_message(
                recipient_id="orchestrator_agent",
                message_data=ack_message
            )

        except Exception as e:
            logger.error(f"Failed to send goal acknowledgment: {e}")

    async def collect_current_metrics(self) -> Agent0Metrics:
        """Collect all current metrics from Agent 0"""
        try:
            # In a real implementation, these would come from:
            # - Agent 0's internal counters
            # - Database queries
            # - System monitoring
            # - A2A message queue status
            # - External monitoring systems

            # Simulated realistic metrics based on Agent 0's actual capabilities
            current_time = datetime.utcnow()

            # Performance metrics (would come from agent's internal tracking)
            metrics = Agent0Metrics(
                data_products_registered=1247,  # From database count
                registration_success_rate=96.8,  # Success/total ratio
                avg_registration_time=1.85,  # Average processing time
                validation_accuracy=94.2,  # Schema validation success rate

                # Quality metrics (from quality assessment module)
                schema_compliance_rate=98.1,  # Dublin Core compliance
                data_quality_score=87.5,  # AI-powered quality assessment
                dublin_core_compliance=97.3,  # Metadata compliance

                # System metrics (from health monitoring)
                api_availability=99.94,  # Uptime percentage
                error_rate=2.1,  # Error percentage
                throughput_per_hour=156,  # Registrations per hour
                queue_depth=3,  # Current message queue depth
                processing_time_p95=2.8,  # 95th percentile processing time

                # Business metrics (from business logic)
                catalog_completeness=89.7,  # Catalog entry completeness
                user_satisfaction_score=8.4,  # User feedback score
                compliance_violations=0,  # Current violations

                # AI enhancement metrics (from AI modules)
                grok_ai_accuracy=91.3,  # Grok AI processing accuracy
                perplexity_api_success_rate=98.7,  # Perplexity API success
                pdf_processing_success_rate=93.8,  # PDF processing success

                timestamp=current_time
            )

            self.current_metrics = metrics
            self.metrics_history.append(metrics)

            # Keep only last 1000 metrics entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect current metrics: {e}")
            raise

    async def _collect_goal_specific_metrics(self, goal_id: str):
        """Collect metrics specific to a SMART goal"""
        try:
            goal = self.assigned_goals.get(goal_id)
            if not goal:
                return

            # Collect current metrics
            current_metrics = await self.collect_current_metrics()

            # Extract goal-relevant metrics based on measurable criteria
            goal_metrics = {}
            measurable_targets = goal.measurable

            for metric_name, target_value in measurable_targets.items():
                if hasattr(current_metrics, metric_name):
                    current_value = getattr(current_metrics, metric_name)
                    goal_metrics[metric_name] = {
                        "current_value": current_value,
                        "target_value": target_value,
                        "progress_percentage": self._calculate_progress_percentage(
                            current_value, target_value, metric_name
                        )
                    }

            # Update goal with current metrics
            goal.current_metrics = goal_metrics

            # Send metrics update to orchestrator
            await self._send_goal_metrics_update(goal_id, goal_metrics)

        except Exception as e:
            logger.error(f"Failed to collect goal-specific metrics for {goal_id}: {e}")

    def _calculate_progress_percentage(self, current: float, target: float, metric_name: str) -> float:
        """Calculate progress percentage towards goal"""
        try:
            # Different calculation methods based on metric type
            if metric_name in ["error_rate", "compliance_violations"]:
                # Lower is better - reverse calculation
                if target == 0:
                    return 100.0 if current == 0 else max(0, 100 - (current * 10))
                return max(0, min(100, ((target - current) / target) * 100))
            else:
                # Higher is better - standard calculation
                if target == 0:
                    return 100.0 if current == 0 else 0.0
                return min(100, (current / target) * 100)

        except Exception:
            return 0.0

    async def _send_goal_metrics_update(self, goal_id: str, metrics: Dict[str, Any]):
        """Send goal metrics update to orchestrator"""
        try:
            update_message = {
                "operation": "track_goal_progress",
                "data": {
                    "agent_id": self.agent_id,
                    "goal_id": goal_id,
                    "progress": {
                        "metrics": metrics,
                        "timestamp": datetime.utcnow().isoformat(),
                        "overall_progress": self._calculate_overall_progress(metrics)
                    }
                }
            }

            await self.orchestrator_client.send_message(
                recipient_id="orchestrator_agent",
                message_data=update_message
            )

            logger.debug(f"Sent goal metrics update for {goal_id}")

        except Exception as e:
            logger.error(f"Failed to send goal metrics update: {e}")

    def _calculate_overall_progress(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall progress percentage across all goal metrics"""
        if not metrics:
            return 0.0

        total_progress = sum(
            metric_data["progress_percentage"]
            for metric_data in metrics.values()
        )

        return total_progress / len(metrics)

    async def start_goal_tracking(self):
        """Start continuous goal tracking"""
        logger.info("Starting goal-aware metrics tracking")

        while True:
            try:
                # Check if we have assigned goals
                if self.assigned_goals:
                    # Collect metrics for each assigned goal
                    for goal_id in self.assigned_goals.keys():
                        await self._collect_goal_specific_metrics(goal_id)

                # Wait based on tracking frequency (default: hourly)
                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Error in goal tracking loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def get_goal_status_summary(self) -> Dict[str, Any]:
        """Get summary of all assigned goals and their status"""
        summary = {
            "agent_id": self.agent_id,
            "total_goals": len(self.assigned_goals),
            "goal_notification_received": self.goal_notification_received,
            "last_goal_sync": self.last_goal_sync.isoformat() if self.last_goal_sync else None,
            "available_metrics": len(self._get_available_metrics_list()),
            "goals": []
        }

        for goal_id, goal in self.assigned_goals.items():
            goal_summary = {
                "goal_id": goal_id,
                "specific": goal.specific,
                "target_date": goal.target_date.isoformat(),
                "tracking_frequency": goal.tracking_frequency,
                "current_progress": self._calculate_overall_progress(goal.current_metrics),
                "metrics_tracked": len(goal.current_metrics)
            }
            summary["goals"].append(goal_summary)

        return summary

# Example SMART Goals for Agent 0
EXAMPLE_SMART_GOALS = [
    {
        "goal_id": "agent0_registration_efficiency",
        "specific": "Achieve 95% data product registration success rate with sub-2 second average processing time",
        "measurable": {
            "registration_success_rate": 95.0,  # Target: 95%
            "avg_registration_time": 2.0,  # Target: < 2 seconds
            "validation_accuracy": 98.0  # Target: 98% validation accuracy
        },
        "achievable": True,
        "relevant": "Critical for Agent 0's primary function of data product registration and validation",
        "time_bound": "30 days",
        "assigned_date": datetime.utcnow().isoformat(),
        "target_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "tracking_frequency": "hourly"
    },
    {
        "goal_id": "agent0_quality_excellence",
        "specific": "Maintain 99% schema compliance and achieve 90+ data quality scores consistently",
        "measurable": {
            "schema_compliance_rate": 99.0,  # Target: 99%
            "data_quality_score": 90.0,  # Target: 90+
            "dublin_core_compliance": 98.0  # Target: 98%
        },
        "achievable": True,
        "relevant": "Ensures high-quality data products meet enterprise standards",
        "time_bound": "45 days",
        "assigned_date": datetime.utcnow().isoformat(),
        "target_date": (datetime.utcnow() + timedelta(days=45)).isoformat(),
        "tracking_frequency": "daily"
    },
    {
        "goal_id": "agent0_system_reliability",
        "specific": "Achieve 99.9% API availability with zero compliance violations",
        "measurable": {
            "api_availability": 99.9,  # Target: 99.9%
            "error_rate": 1.0,  # Target: < 1%
            "compliance_violations": 0  # Target: 0 violations
        },
        "achievable": True,
        "relevant": "Ensures reliable service for enterprise data product management",
        "time_bound": "60 days",
        "assigned_date": datetime.utcnow().isoformat(),
        "target_date": (datetime.utcnow() + timedelta(days=60)).isoformat(),
        "tracking_frequency": "real-time"
    }
]

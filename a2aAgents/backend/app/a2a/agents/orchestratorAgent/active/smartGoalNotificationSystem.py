"""
SMART Goal Notification System for Orchestrator Agent
Handles goal assignment notifications and metric mapping for agents
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
class AgentCapabilities:
    """Agent's metric collection capabilities"""
    agent_id: str
    available_metrics: List[str]
    collection_frequency: str
    last_updated: datetime
    notification_preferences: Dict[str, Any]

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
            "notifications": list(self.active_notifications.values())
        }

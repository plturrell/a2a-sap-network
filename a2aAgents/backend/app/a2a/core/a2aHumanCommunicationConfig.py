#!/usr/bin/env python3
"""
A2A Human Communication Configuration
Determines when agents should communicate with humans vs other agents
Provides rules, escalation paths, and message routing logic
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CommunicationTarget(str, Enum):
    """Communication target types"""
    AGENT = "agent"
    HUMAN = "human" 
    BOTH = "both"
    ESCALATE = "escalate"


class MessageType(str, Enum):
    """A2A Message types that can be routed"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification" 
    ALERT = "alert"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    APPROVAL_REQUEST = "approval_request"
    DATA_REQUEST = "data_request"


class UrgencyLevel(str, Enum):
    """Message urgency levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CommunicationRule:
    """Rule for determining communication routing"""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    target: CommunicationTarget
    priority: int = 100  # Lower number = higher priority
    active: bool = True
    require_human_confirmation: bool = False
    escalation_timeout_minutes: int = 30
    human_response_timeout_minutes: int = 60


@dataclass  
class AgentCommunicationProfile:
    """Communication profile for an agent"""
    agent_id: str
    agent_name: str
    autonomy_level: str  # "low", "medium", "high", "full"
    allowed_operations: List[str]
    requires_approval: List[str]  # Operations that need human approval
    escalation_triggers: List[str]  # Conditions that trigger human escalation
    communication_preferences: Dict[str, Any]
    human_supervisors: List[str] = field(default_factory=list)


class A2AHumanCommunicationRouter:
    """Routes A2A messages between agents and humans based on configuration rules"""
    
    def __init__(self):
        self.rules: List[CommunicationRule] = []
        self.agent_profiles: Dict[str, AgentCommunicationProfile] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.pending_human_responses: Dict[str, Dict[str, Any]] = {}
        
        # Load default configuration
        self.load_default_configuration()
        
        logger.info("A2A Human Communication Router initialized")
    
    def load_default_configuration(self):
        """Load default communication rules and agent profiles"""
        
        # Default Communication Rules (in priority order)
        default_rules = [
            # CRITICAL - Security threats always to humans immediately
            CommunicationRule(
                rule_id="security_critical",
                name="Critical Security Threats",
                description="All critical security events must be reported to humans",
                conditions={
                    "message_type": ["alert", "error"],
                    "urgency": ["critical", "emergency"],
                    "category": ["security", "threat_detected", "breach_attempt"],
                    "threat_level": ["high", "critical"]
                },
                target=CommunicationTarget.HUMAN,
                priority=1,
                require_human_confirmation=True,
                escalation_timeout_minutes=5
            ),
            
            # CRITICAL - Agent failures/crashes to humans
            CommunicationRule(
                rule_id="agent_failure_critical",
                name="Agent Failure/Crash",
                description="Agent crashes and critical failures require human intervention",
                conditions={
                    "message_type": ["error", "alert"],
                    "urgency": ["critical", "high"],
                    "category": ["agent_failure", "system_failure", "crash"],
                    "involves_data_loss": True
                },
                target=CommunicationTarget.HUMAN,
                priority=2,
                require_human_confirmation=True,
                escalation_timeout_minutes=10
            ),
            
            # HIGH - Financial/Business Critical Operations
            CommunicationRule(
                rule_id="financial_operations",
                name="Financial Operations",
                description="Financial transactions and business-critical operations need human approval",
                conditions={
                    "message_type": ["approval_request", "request"],
                    "category": ["financial", "transaction", "payment", "contract"],
                    "amount_threshold": 1000,  # Over $1000
                    "involves_external_party": True
                },
                target=CommunicationTarget.HUMAN,
                priority=10,
                require_human_confirmation=True,
                human_response_timeout_minutes=120
            ),
            
            # HIGH - Data Access Requests (Sensitive)
            CommunicationRule(
                rule_id="sensitive_data_access",
                name="Sensitive Data Access",
                description="Requests for sensitive or restricted data require human approval",
                conditions={
                    "message_type": ["data_request", "approval_request"],
                    "data_classification": ["confidential", "restricted", "pii"],
                    "external_access": True
                },
                target=CommunicationTarget.HUMAN,
                priority=15,
                require_human_confirmation=True,
                human_response_timeout_minutes=240
            ),
            
            # MEDIUM - Workflow Approvals
            CommunicationRule(
                rule_id="workflow_approvals",
                name="Workflow Approvals",
                description="Business workflow approvals that require human decision",
                conditions={
                    "message_type": ["approval_request"],
                    "category": ["workflow", "business_process"],
                    "requires_human_judgment": True
                },
                target=CommunicationTarget.HUMAN,
                priority=20,
                human_response_timeout_minutes=480  # 8 hours
            ),
            
            # MEDIUM - Cross-domain Agent Coordination
            CommunicationRule(
                rule_id="cross_domain_coordination", 
                name="Cross-domain Coordination",
                description="Complex operations requiring coordination across multiple agent domains",
                conditions={
                    "message_type": ["request", "notification"],
                    "involves_multiple_agents": True,
                    "cross_domain": True,
                    "complexity_level": ["high", "very_high"]
                },
                target=CommunicationTarget.BOTH,  # Notify human but allow agents to proceed
                priority=30
            ),
            
            # MEDIUM - New/Unknown Scenarios
            CommunicationRule(
                rule_id="unknown_scenarios",
                name="Unknown Scenarios",
                description="Scenarios not covered by agent training require human guidance",
                conditions={
                    "confidence_level": {"$lt": 0.7},  # Less than 70% confidence
                    "scenario_known": False,
                    "requires_creative_solution": True
                },
                target=CommunicationTarget.ESCALATE,
                priority=25,
                escalation_timeout_minutes=60
            ),
            
            # LOW - Routine Status Updates
            CommunicationRule(
                rule_id="routine_status",
                name="Routine Status Updates", 
                description="Regular status updates can be handled by agents unless specifically requested",
                conditions={
                    "message_type": ["status_update", "notification"],
                    "urgency": ["low", "medium"],
                    "routine": True
                },
                target=CommunicationTarget.AGENT,
                priority=50
            ),
            
            # LOW - Standard Data Processing
            CommunicationRule(
                rule_id="standard_data_processing",
                name="Standard Data Processing",
                description="Standard data processing operations between agents",
                conditions={
                    "message_type": ["request", "response"],
                    "category": ["data_processing", "calculation", "transformation"],
                    "data_classification": ["public", "internal"],
                    "standard_operation": True
                },
                target=CommunicationTarget.AGENT,
                priority=60
            ),
            
            # FALLBACK - Default escalation rule
            CommunicationRule(
                rule_id="default_escalation",
                name="Default Escalation",
                description="When no other rule matches, escalate to human after timeout",
                conditions={"default": True},
                target=CommunicationTarget.ESCALATE,
                priority=1000,  # Lowest priority
                escalation_timeout_minutes=15
            )
        ]
        
        self.rules = default_rules
        
        # Default Agent Profiles
        self.agent_profiles = {
            "data_product_agent_0": AgentCommunicationProfile(
                agent_id="data_product_agent_0",
                agent_name="Data Product Agent",
                autonomy_level="high",
                allowed_operations=["data_registration", "metadata_management", "catalog_updates"],
                requires_approval=["external_data_access", "schema_changes"],
                escalation_triggers=["data_quality_failure", "schema_validation_error"],
                communication_preferences={"notify_on_completion": True}
            ),
            
            "security_monitor": AgentCommunicationProfile(
                agent_id="security_monitor", 
                agent_name="Security Monitor",
                autonomy_level="medium",
                allowed_operations=["threat_detection", "alert_generation", "log_analysis"],
                requires_approval=["ip_blocking", "account_suspension", "system_lockdown"],
                escalation_triggers=["critical_threat", "breach_detected", "anomalous_behavior"],
                communication_preferences={"immediate_escalation": True},
                human_supervisors=["security_admin", "system_admin"]
            ),
            
            "workflow_engine": AgentCommunicationProfile(
                agent_id="workflow_engine",
                agent_name="Workflow Engine", 
                autonomy_level="low",  # Workflows often need human input
                allowed_operations=["task_execution", "status_updates"],
                requires_approval=["workflow_modifications", "user_task_completion", "deadline_extensions"],
                escalation_triggers=["approval_timeout", "workflow_stuck", "exception_occurred"],
                communication_preferences={"human_in_loop": True}
            ),
            
            "reasoning_agent": AgentCommunicationProfile(
                agent_id="reasoning_agent",
                agent_name="Reasoning Agent",
                autonomy_level="high",
                allowed_operations=["logical_inference", "analysis", "recommendation_generation"],
                requires_approval=["high_impact_decisions", "policy_recommendations"],
                escalation_triggers=["low_confidence_result", "contradictory_evidence", "ethical_concern"],
                communication_preferences={"explain_reasoning": True}
            ),
            
            "sql_agent": AgentCommunicationProfile(
                agent_id="sql_agent", 
                agent_name="SQL Agent",
                autonomy_level="medium",
                allowed_operations=["query_execution", "data_retrieval", "report_generation"],
                requires_approval=["write_operations", "schema_modifications", "bulk_deletes"],
                escalation_triggers=["query_timeout", "data_inconsistency", "permission_denied"],
                communication_preferences={"query_validation": True}
            )
        }
        
        logger.info(f"Loaded {len(self.rules)} communication rules and {len(self.agent_profiles)} agent profiles")
    
    def determine_communication_target(
        self,
        message: Dict[str, Any],
        from_agent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Determine where an A2A message should be routed based on rules"""
        
        try:
            # Enhance message with context
            enhanced_message = {**message}
            if context:
                enhanced_message.update(context)
            
            # Add agent profile information
            agent_profile = self.agent_profiles.get(from_agent)
            if agent_profile:
                enhanced_message["agent_autonomy_level"] = agent_profile.autonomy_level
                enhanced_message["agent_requires_approval"] = agent_profile.requires_approval
                enhanced_message["agent_escalation_triggers"] = agent_profile.escalation_triggers
            
            # Find matching rule (highest priority first)
            matching_rule = None
            for rule in sorted(self.rules, key=lambda r: r.priority):
                if not rule.active:
                    continue
                    
                if self._message_matches_rule(enhanced_message, rule):
                    matching_rule = rule
                    break
            
            if not matching_rule:
                logger.warning(f"No matching rule found for message from {from_agent}")
                matching_rule = self.rules[-1]  # Default escalation rule
            
            # Determine routing decision
            routing_decision = {
                "target": matching_rule.target.value,
                "rule_applied": {
                    "rule_id": matching_rule.rule_id,
                    "name": matching_rule.name,
                    "description": matching_rule.description,
                    "priority": matching_rule.priority
                },
                "requires_human_confirmation": matching_rule.require_human_confirmation,
                "escalation_timeout_minutes": matching_rule.escalation_timeout_minutes,
                "human_response_timeout_minutes": matching_rule.human_response_timeout_minutes,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": enhanced_message.get("messageId", "unknown"),
                "from_agent": from_agent
            }
            
            # Add specific routing instructions
            if matching_rule.target == CommunicationTarget.HUMAN:
                routing_decision["human_routing"] = self._prepare_human_routing(enhanced_message, agent_profile)
            elif matching_rule.target == CommunicationTarget.AGENT:
                routing_decision["agent_routing"] = self._prepare_agent_routing(enhanced_message)
            elif matching_rule.target == CommunicationTarget.BOTH:
                routing_decision["human_routing"] = self._prepare_human_routing(enhanced_message, agent_profile)
                routing_decision["agent_routing"] = self._prepare_agent_routing(enhanced_message)
            elif matching_rule.target == CommunicationTarget.ESCALATE:
                routing_decision["escalation"] = self._prepare_escalation_routing(enhanced_message, matching_rule)
            
            # Log routing decision
            logger.info(f"Message from {from_agent} routed to {matching_rule.target.value} using rule {matching_rule.rule_id}")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error determining communication target: {e}")
            return {
                "target": "human",  # Fail-safe to human
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _message_matches_rule(self, message: Dict[str, Any], rule: CommunicationRule) -> bool:
        """Check if a message matches a rule's conditions"""
        
        conditions = rule.conditions
        
        # Handle default rule
        if conditions.get("default"):
            return True
        
        # Check each condition
        for condition_key, condition_value in conditions.items():
            message_value = message.get(condition_key)
            
            # Handle list conditions (OR logic)
            if isinstance(condition_value, list):
                if message_value not in condition_value:
                    return False
            
            # Handle dict conditions (operators like $lt, $gt)
            elif isinstance(condition_value, dict):
                if not self._evaluate_condition_operators(message_value, condition_value):
                    return False
            
            # Handle boolean conditions
            elif isinstance(condition_value, bool):
                if bool(message_value) != condition_value:
                    return False
            
            # Handle direct value comparison
            else:
                if message_value != condition_value:
                    return False
        
        return True
    
    def _evaluate_condition_operators(self, message_value: Any, operators: Dict[str, Any]) -> bool:
        """Evaluate condition operators like $lt, $gt, etc."""
        
        for operator, expected_value in operators.items():
            if operator == "$lt" and message_value >= expected_value:
                return False
            elif operator == "$lte" and message_value > expected_value:
                return False  
            elif operator == "$gt" and message_value <= expected_value:
                return False
            elif operator == "$gte" and message_value < expected_value:
                return False
            elif operator == "$eq" and message_value != expected_value:
                return False
            elif operator == "$ne" and message_value == expected_value:
                return False
            elif operator == "$in" and message_value not in expected_value:
                return False
            elif operator == "$nin" and message_value in expected_value:
                return False
        
        return True
    
    def _prepare_human_routing(self, message: Dict[str, Any], agent_profile: Optional[AgentCommunicationProfile]) -> Dict[str, Any]:
        """Prepare human routing information"""
        
        human_routing = {
            "notification_required": True,
            "urgency_level": message.get("urgency", "medium"),
            "requires_immediate_attention": message.get("urgency") in ["critical", "emergency"],
            "suggested_supervisors": [],
            "human_readable_message": self._convert_a2a_to_human_readable(message),
            "recommended_actions": self._suggest_human_actions(message),
            "context_information": {
                "agent_name": agent_profile.agent_name if agent_profile else "Unknown Agent",
                "agent_autonomy": agent_profile.autonomy_level if agent_profile else "unknown",
                "message_category": message.get("category", "general"),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Add suggested supervisors from agent profile
        if agent_profile and agent_profile.human_supervisors:
            human_routing["suggested_supervisors"] = agent_profile.human_supervisors
        
        return human_routing
    
    def _prepare_agent_routing(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare agent-to-agent routing information"""
        
        return {
            "target_agents": self._determine_target_agents(message),
            "routing_priority": message.get("urgency", "medium"),
            "requires_response": message.get("message_type") == "request",
            "broadcast_to_all": message.get("broadcast", False),
            "routing_context": {
                "original_message_id": message.get("messageId"),
                "routing_timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _prepare_escalation_routing(self, message: Dict[str, Any], rule: CommunicationRule) -> Dict[str, Any]:
        """Prepare escalation routing information"""
        
        return {
            "escalation_type": "timeout_based",
            "initial_target": "agent",
            "escalation_delay_minutes": rule.escalation_timeout_minutes,
            "escalation_target": "human", 
            "escalation_reason": f"No agent response within {rule.escalation_timeout_minutes} minutes",
            "escalation_context": {
                "rule_id": rule.rule_id,
                "original_message": message,
                "scheduled_escalation_time": (
                    datetime.utcnow() + timedelta(minutes=rule.escalation_timeout_minutes)
                ).isoformat()
            }
        }
    
    def _convert_a2a_to_human_readable(self, message: Dict[str, Any]) -> str:
        """Convert A2A protocol message to human-readable format"""
        
        message_type = message.get("message_type", "message")
        category = message.get("category", "general")
        urgency = message.get("urgency", "medium")
        
        # Generate human-friendly message based on type and content
        if message_type == "approval_request":
            return f"Agent is requesting approval for: {message.get('description', 'an operation')}"
        elif message_type == "data_request":
            return f"Agent needs access to data: {message.get('data_description', 'unspecified data')}"
        elif message_type == "error":
            return f"Agent encountered an error: {message.get('error_description', 'unknown error')}"
        elif message_type == "alert" and category == "security":
            return f"Security alert: {message.get('description', 'potential threat detected')}"
        elif message_type == "status_update":
            return f"Agent status update: {message.get('status', 'status changed')}"
        else:
            return f"Agent {message_type}: {message.get('description', 'requires attention')}"
    
    def _suggest_human_actions(self, message: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest possible human actions based on message content"""
        
        actions = []
        message_type = message.get("message_type")
        category = message.get("category")
        
        if message_type == "approval_request":
            actions.extend([
                {"action": "approve", "label": "Approve Request", "type": "positive"},
                {"action": "deny", "label": "Deny Request", "type": "negative"}, 
                {"action": "request_more_info", "label": "Request More Information", "type": "neutral"}
            ])
        
        elif category == "security":
            actions.extend([
                {"action": "investigate", "label": "Investigate Further", "type": "neutral"},
                {"action": "block", "label": "Block/Quarantine", "type": "negative"},
                {"action": "escalate_security", "label": "Escalate to Security Team", "type": "attention"}
            ])
        
        elif message_type == "error":
            actions.extend([
                {"action": "restart_agent", "label": "Restart Agent", "type": "neutral"},
                {"action": "view_logs", "label": "View Error Logs", "type": "neutral"},
                {"action": "escalate_technical", "label": "Escalate to Technical Team", "type": "attention"}
            ])
        
        # Always add generic actions
        actions.extend([
            {"action": "acknowledge", "label": "Acknowledge", "type": "neutral"},
            {"action": "delegate", "label": "Delegate to Another Agent", "type": "neutral"},
            {"action": "dismiss", "label": "Dismiss", "type": "neutral"}
        ])
        
        return actions
    
    def _determine_target_agents(self, message: Dict[str, Any]) -> List[str]:
        """Determine which agents should receive this message"""
        
        # Simple logic - could be enhanced with agent capability matching
        category = message.get("category", "")
        message_type = message.get("message_type", "")
        
        if category == "data_processing":
            return ["data_product_agent_0", "sql_agent"]
        elif category == "security":
            return ["security_monitor"]
        elif category == "workflow":
            return ["workflow_engine"]
        elif message_type == "status_update":
            return ["agent_manager"]  # Broadcast status updates to manager
        else:
            return []  # Let routing service decide
    
    def add_custom_rule(self, rule: CommunicationRule):
        """Add a custom communication rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)  # Maintain priority order
        logger.info(f"Added custom rule: {rule.rule_id}")
    
    def update_agent_profile(self, agent_id: str, profile: AgentCommunicationProfile):
        """Update an agent's communication profile"""
        self.agent_profiles[agent_id] = profile
        logger.info(f"Updated communication profile for agent: {agent_id}")
    
    def get_agent_communication_settings(self, agent_id: str) -> Dict[str, Any]:
        """Get communication settings for a specific agent"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return {"error": f"No profile found for agent {agent_id}"}
        
        return {
            "agent_id": profile.agent_id,
            "agent_name": profile.agent_name,
            "autonomy_level": profile.autonomy_level,
            "allowed_operations": profile.allowed_operations,
            "requires_approval": profile.requires_approval,
            "escalation_triggers": profile.escalation_triggers,
            "communication_preferences": profile.communication_preferences,
            "human_supervisors": profile.human_supervisors
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for backup/sharing"""
        return {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "conditions": rule.conditions,
                    "target": rule.target.value,
                    "priority": rule.priority,
                    "active": rule.active,
                    "require_human_confirmation": rule.require_human_confirmation,
                    "escalation_timeout_minutes": rule.escalation_timeout_minutes,
                    "human_response_timeout_minutes": rule.human_response_timeout_minutes
                }
                for rule in self.rules
            ],
            "agent_profiles": {
                agent_id: {
                    "agent_id": profile.agent_id,
                    "agent_name": profile.agent_name,
                    "autonomy_level": profile.autonomy_level,
                    "allowed_operations": profile.allowed_operations,
                    "requires_approval": profile.requires_approval,
                    "escalation_triggers": profile.escalation_triggers,
                    "communication_preferences": profile.communication_preferences,
                    "human_supervisors": profile.human_supervisors
                }
                for agent_id, profile in self.agent_profiles.items()
            }
        }


# Global instance
_communication_router: Optional[A2AHumanCommunicationRouter] = None

def get_communication_router() -> A2AHumanCommunicationRouter:
    """Get global communication router instance"""
    global _communication_router
    if _communication_router is None:
        _communication_router = A2AHumanCommunicationRouter()
    return _communication_router


# Convenience functions for agents to use
def should_message_human(
    message: Dict[str, Any], 
    from_agent: str, 
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Check if a message should be sent to a human"""
    router = get_communication_router()
    decision = router.determine_communication_target(message, from_agent, context)
    return decision["target"] in ["human", "both", "escalate"]


def route_a2a_message(
    message: Dict[str, Any],
    from_agent: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Route an A2A message to appropriate target(s)"""
    router = get_communication_router()
    return router.determine_communication_target(message, from_agent, context)


# Export main classes and functions
__all__ = [
    'A2AHumanCommunicationRouter',
    'CommunicationRule', 
    'AgentCommunicationProfile',
    'CommunicationTarget',
    'MessageType',
    'UrgencyLevel',
    'get_communication_router',
    'should_message_human',
    'route_a2a_message'
]
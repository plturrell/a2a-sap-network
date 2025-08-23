"""
Comprehensive A2A Protocol Validation System
Validates message format, agent interactions, and protocol compliance
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import hashlib
import uuid
from collections import defaultdict, deque

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..a2a.sdk.types import A2AMessage, MessageType, AgentCapability
from ..a2a.sdk.agentBase import A2AAgentBase

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation"""
    MESSAGE_FORMAT = "message_format"
    PROTOCOL_COMPLIANCE = "protocol_compliance"
    AGENT_BEHAVIOR = "agent_behavior"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TRUST = "trust"


class ProtocolVersion(str, Enum):
    """Supported A2A protocol versions"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_id: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class MessageValidationContext:
    """Context for message validation"""
    message: A2AMessage
    sender_agent: Optional[str] = None
    receiver_agent: Optional[str] = None
    conversation_id: Optional[str] = None
    protocol_version: ProtocolVersion = ProtocolVersion.V2_0
    trust_level: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInteractionContext:
    """Context for agent interaction validation"""
    agent_id: str
    interaction_type: str
    start_time: datetime
    messages: List[A2AMessage] = field(default_factory=list)
    expected_capabilities: Set[AgentCapability] = field(default_factory=set)
    actual_capabilities: Set[AgentCapability] = field(default_factory=set)
    trust_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, rule_id: str, category: ValidationCategory, severity: ValidationSeverity):
        self.rule_id = rule_id
        self.category = category
        self.severity = severity
        self.enabled = True
        
    @abstractmethod
    async def validate(self, context: Union[MessageValidationContext, AgentInteractionContext]) -> List[ValidationResult]:
        """Validate the given context"""
        pass
    
    def create_result(self, message: str, details: Dict[str, Any] = None, location: str = None, suggestion: str = None) -> ValidationResult:
        """Create a validation result"""
        return ValidationResult(
            rule_id=self.rule_id,
            category=self.category,
            severity=self.severity,
            message=message,
            details=details or {},
            location=location,
            suggestion=suggestion
        )


class MessageFormatRule(ValidationRule):
    """Validates A2A message format compliance"""
    
    def __init__(self):
        super().__init__("msg_format", ValidationCategory.MESSAGE_FORMAT, ValidationSeverity.ERROR)
    
    async def validate(self, context: MessageValidationContext) -> List[ValidationResult]:
        results = []
        message = context.message
        
        # Required fields validation
        required_fields = ['id', 'type', 'sender', 'timestamp', 'payload']
        for field in required_fields:
            if not hasattr(message, field) or getattr(message, field) is None:
                results.append(self.create_result(
                    f"Missing required field: {field}",
                    {"field": field},
                    suggestion=f"Ensure message includes '{field}' field"
                ))
        
        # Message ID format validation
        if hasattr(message, 'id') and message.id:
            if not re.match(r'^[a-zA-Z0-9\-_]{1,64}$', message.id):
                results.append(self.create_result(
                    "Invalid message ID format",
                    {"message_id": message.id, "pattern": "^[a-zA-Z0-9\\-_]{1,64}$"},
                    suggestion="Use alphanumeric characters, hyphens, and underscores only"
                ))
        
        # Timestamp validation
        if hasattr(message, 'timestamp') and message.timestamp:
            try:
                # Validate timestamp is not too old or in future
                msg_time = datetime.fromisoformat(message.timestamp.replace('Z', '+00:00'))
                now = datetime.utcnow()
                
                if msg_time > now + timedelta(minutes=5):
                    results.append(self.create_result(
                        "Message timestamp is in the future",
                        {"timestamp": message.timestamp, "drift_minutes": (msg_time - now).total_seconds() / 60}
                    ))
                
                if now - msg_time > timedelta(hours=24):
                    results.append(self.create_result(
                        "Message timestamp is too old",
                        {"timestamp": message.timestamp, "age_hours": (now - msg_time).total_seconds() / 3600}
                    ))
                    
            except (ValueError, TypeError) as e:
                results.append(self.create_result(
                    "Invalid timestamp format",
                    {"timestamp": message.timestamp, "error": str(e)},
                    suggestion="Use ISO 8601 format: YYYY-MM-DDTHH:MM:SS.sssZ"
                ))
        
        # Payload size validation
        if hasattr(message, 'payload') and message.payload:
            try:
                payload_size = len(json.dumps(message.payload))
                if payload_size > 1024 * 1024:  # 1MB limit
                    results.append(self.create_result(
                        "Message payload exceeds size limit",
                        {"payload_size": payload_size, "limit": 1024 * 1024},
                        suggestion="Consider using message streaming for large payloads"
                    ))
            except (TypeError, ValueError):
                results.append(self.create_result(
                    "Message payload is not JSON serializable",
                    {"payload_type": type(message.payload).__name__}
                ))
        
        return results


class ProtocolComplianceRule(ValidationRule):
    """Validates A2A protocol compliance"""
    
    def __init__(self):
        super().__init__("protocol_compliance", ValidationCategory.PROTOCOL_COMPLIANCE, ValidationSeverity.ERROR)
    
    async def validate(self, context: MessageValidationContext) -> List[ValidationResult]:
        results = []
        message = context.message
        
        # Message type validation
        if hasattr(message, 'type') and message.type:
            valid_types = [mt.value for mt in MessageType]
            if message.type not in valid_types:
                results.append(self.create_result(
                    f"Invalid message type: {message.type}",
                    {"message_type": message.type, "valid_types": valid_types},
                    suggestion=f"Use one of: {', '.join(valid_types)}"
                ))
        
        # Conversation ID validation for dialogue messages
        if (hasattr(message, 'type') and message.type in ['request', 'response', 'dialogue'] and
            context.conversation_id is None):
            results.append(self.create_result(
                "Conversation ID required for dialogue messages",
                {"message_type": message.type},
                suggestion="Include conversation_id in message context"
            ))
        
        # Response correlation validation
        if hasattr(message, 'type') and message.type == 'response':
            if not hasattr(message, 'in_reply_to') or not message.in_reply_to:
                results.append(self.create_result(
                    "Response message missing 'in_reply_to' field",
                    {"message_type": message.type},
                    suggestion="Include original message ID in 'in_reply_to' field"
                ))
        
        # Capability declaration validation
        if hasattr(message, 'payload') and isinstance(message.payload, dict):
            if 'capabilities' in message.payload:
                capabilities = message.payload['capabilities']
                if isinstance(capabilities, list):
                    valid_capabilities = [cap.value for cap in AgentCapability]
                    invalid_caps = [cap for cap in capabilities if cap not in valid_capabilities]
                    if invalid_caps:
                        results.append(self.create_result(
                            f"Invalid agent capabilities declared: {invalid_caps}",
                            {"invalid_capabilities": invalid_caps, "valid_capabilities": valid_capabilities},
                            suggestion="Use only standard A2A capabilities"
                        ))
        
        # Protocol version compatibility
        if context.protocol_version == ProtocolVersion.V1_0:
            # V1.0 doesn't support certain features
            if hasattr(message, 'payload') and isinstance(message.payload, dict):
                v2_features = ['streaming', 'multi_agent_coordination', 'trust_metrics']
                used_features = [f for f in v2_features if f in message.payload]
                if used_features:
                    results.append(self.create_result(
                        f"Protocol v1.0 doesn't support features: {used_features}",
                        {"protocol_version": context.protocol_version.value, "unsupported_features": used_features},
                        suggestion="Upgrade to protocol v2.0 or remove unsupported features"
                    ))
        
        return results


class SecurityValidationRule(ValidationRule):
    """Validates security aspects of A2A messages"""
    
    def __init__(self):
        super().__init__("security_validation", ValidationCategory.SECURITY, ValidationSeverity.CRITICAL)
    
    async def validate(self, context: MessageValidationContext) -> List[ValidationResult]:
        results = []
        message = context.message
        
        # Check for sensitive data in payload
        if hasattr(message, 'payload') and isinstance(message.payload, dict):
            payload_str = json.dumps(message.payload).lower()
            
            # Common sensitive patterns
            sensitive_patterns = {
                'password': r'password["\s]*[:=]["\s]*[^\s,"}{]+',
                'api_key': r'api[_\s]*key["\s]*[:=]["\s]*[^\s,"}{]+',
                'token': r'token["\s]*[:=]["\s]*[^\s,"}{]+',
                'secret': r'secret["\s]*[:=]["\s]*[^\s,"}{]+',
                'private_key': r'private[_\s]*key["\s]*[:=]["\s]*[^\s,"}{]+',
                'ssh_key': r'ssh[_\s]*key["\s]*[:=]["\s]*[^\s,"}{]+'
            }
            
            for pattern_name, pattern in sensitive_patterns.items():
                if re.search(pattern, payload_str):
                    results.append(self.create_result(
                        f"Potential sensitive data detected: {pattern_name}",
                        {"pattern": pattern_name, "severity": "critical"},
                        suggestion="Remove sensitive data from message payload or use encryption"
                    ))
        
        # Validate sender identity
        if hasattr(message, 'sender') and message.sender:
            # Check for suspicious sender patterns
            if message.sender.startswith('anonymous_') or message.sender == 'unknown':
                results.append(self.create_result(
                    "Message from anonymous or unknown sender",
                    {"sender": message.sender},
                    suggestion="Implement proper agent authentication"
                ))
        
        # Trust level validation
        if context.trust_level < 0.3:
            results.append(self.create_result(
                "Message from low-trust source",
                {"trust_level": context.trust_level, "threshold": 0.3},
                suggestion="Verify sender identity and message authenticity"
            ))
        
        # Message integrity check
        if hasattr(message, 'signature') and message.signature:
            # Validate signature format
            if not re.match(r'^[a-zA-Z0-9+/]+=*$', message.signature):
                results.append(self.create_result(
                    "Invalid message signature format",
                    {"signature_length": len(message.signature)},
                    suggestion="Use base64-encoded signatures"
                ))
        elif context.trust_level < 0.7:
            results.append(self.create_result(
                "Message lacks digital signature",
                {"trust_level": context.trust_level},
                suggestion="Sign messages for integrity verification"
            ))
        
        return results


class AgentBehaviorRule(ValidationRule):
    """Validates agent behavior patterns"""
    
    def __init__(self):
        super().__init__("agent_behavior", ValidationCategory.AGENT_BEHAVIOR, ValidationSeverity.WARNING)
    
    async def validate(self, context: AgentInteractionContext) -> List[ValidationResult]:
        results = []
        
        # Capability consistency check
        declared_caps = context.expected_capabilities
        demonstrated_caps = context.actual_capabilities
        
        missing_caps = declared_caps - demonstrated_caps
        if missing_caps:
            results.append(self.create_result(
                f"Agent hasn't demonstrated declared capabilities: {[cap.value for cap in missing_caps]}",
                {"missing_capabilities": [cap.value for cap in missing_caps]},
                suggestion="Ensure agent implements all declared capabilities"
            ))
        
        unexpected_caps = demonstrated_caps - declared_caps
        if unexpected_caps:
            results.append(self.create_result(
                f"Agent demonstrated undeclared capabilities: {[cap.value for cap in unexpected_caps]}",
                {"unexpected_capabilities": [cap.value for cap in unexpected_caps]},
                suggestion="Update agent capability declarations"
            ))
        
        # Message frequency analysis
        if len(context.messages) > 100:
            time_span = (datetime.utcnow() - context.start_time).total_seconds()
            if time_span > 0:
                message_rate = len(context.messages) / time_span
                if message_rate > 10:  # More than 10 messages per second
                    results.append(self.create_result(
                        f"High message frequency detected: {message_rate:.2f} msg/sec",
                        {"message_rate": message_rate, "threshold": 10},
                        suggestion="Implement rate limiting or batch processing"
                    ))
        
        # Response time analysis
        request_response_pairs = []
        for i, msg in enumerate(context.messages):
            if msg.type == 'request':
                # Look for corresponding response
                for j in range(i + 1, len(context.messages)):
                    if (context.messages[j].type == 'response' and 
                        hasattr(context.messages[j], 'in_reply_to') and 
                        context.messages[j].in_reply_to == msg.id):
                        
                        request_time = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                        response_time = datetime.fromisoformat(context.messages[j].timestamp.replace('Z', '+00:00'))
                        response_delay = (response_time - request_time).total_seconds()
                        
                        request_response_pairs.append(response_delay)
                        break
        
        if request_response_pairs:
            avg_response_time = sum(request_response_pairs) / len(request_response_pairs)
            if avg_response_time > 30:  # More than 30 seconds average
                results.append(self.create_result(
                    f"Slow agent response time: {avg_response_time:.2f}s average",
                    {"average_response_time": avg_response_time, "threshold": 30},
                    suggestion="Optimize agent processing or implement asynchronous responses"
                ))
        
        return results


class PerformanceValidationRule(ValidationRule):
    """Validates performance aspects of A2A interactions"""
    
    def __init__(self):
        super().__init__("performance_validation", ValidationCategory.PERFORMANCE, ValidationSeverity.INFO)
    
    async def validate(self, context: Union[MessageValidationContext, AgentInteractionContext]) -> List[ValidationResult]:
        results = []
        
        if isinstance(context, MessageValidationContext):
            # Message-level performance validation
            message = context.message
            
            # Payload efficiency check
            if hasattr(message, 'payload') and isinstance(message.payload, dict):
                payload_str = json.dumps(message.payload)
                
                # Check for redundant data
                if 'data' in message.payload and 'payload' in message.payload:
                    results.append(self.create_result(
                        "Redundant data fields in message payload",
                        {"fields": ["data", "payload"]},
                        suggestion="Use consistent field naming to avoid redundancy"
                    ))
                
                # Check for large string repetitions
                import collections
                words = payload_str.split()
                word_counts = collections.Counter(words)
                most_common = word_counts.most_common(1)
                if most_common and most_common[0][1] > 20:
                    results.append(self.create_result(
                        f"High repetition in payload: '{most_common[0][0]}' appears {most_common[0][1]} times",
                        {"repeated_word": most_common[0][0], "count": most_common[0][1]},
                        suggestion="Consider data compression or structure optimization"
                    ))
        
        elif isinstance(context, AgentInteractionContext):
            # Interaction-level performance validation
            
            # Message batching efficiency
            if len(context.messages) > 50:
                # Check for potential batching opportunities
                message_types = [msg.type for msg in context.messages]
                type_counts = collections.Counter(message_types)
                
                if type_counts.get('notification', 0) > 10:
                    results.append(self.create_result(
                        f"Multiple notification messages ({type_counts['notification']}) could be batched",
                        {"notification_count": type_counts['notification']},
                        suggestion="Consider batching notifications to reduce overhead"
                    ))
        
        return results


class A2AProtocolValidator:
    """Main A2A protocol validation system"""
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_history: deque = deque(maxlen=10000)
        self.agent_interactions: Dict[str, AgentInteractionContext] = {}
        self.enabled = True
        self.severity_threshold = ValidationSeverity.WARNING
        
        # Statistics
        self.validation_stats = {
            "total_validations": 0,
            "failed_validations": 0,
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "by_rule": defaultdict(int)
        }
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules"""
        self.register_rule(MessageFormatRule())
        self.register_rule(ProtocolComplianceRule())
        self.register_rule(SecurityValidationRule())
        self.register_rule(AgentBehaviorRule())
        self.register_rule(PerformanceValidationRule())
    
    def register_rule(self, rule: ValidationRule):
        """Register a validation rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Registered validation rule: {rule.rule_id} ({rule.category.value})")
    
    def unregister_rule(self, rule_id: str):
        """Unregister a validation rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Unregistered validation rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a validation rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable a validation rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    @trace_async("validate_message")
    async def validate_message(
        self,
        message: A2AMessage,
        sender_agent: Optional[str] = None,
        receiver_agent: Optional[str] = None,
        conversation_id: Optional[str] = None,
        protocol_version: ProtocolVersion = ProtocolVersion.V2_0,
        trust_level: float = 0.5
    ) -> List[ValidationResult]:
        """Validate an A2A message"""
        
        if not self.enabled:
            return []
        
        add_span_attributes({
            "message.id": message.id if hasattr(message, 'id') else "unknown",
            "message.type": message.type if hasattr(message, 'type') else "unknown",
            "sender.agent": sender_agent or "unknown",
            "protocol.version": protocol_version.value
        })
        
        context = MessageValidationContext(
            message=message,
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            conversation_id=conversation_id,
            protocol_version=protocol_version,
            trust_level=trust_level
        )
        
        all_results = []
        
        # Run all applicable rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
                
            # Only run message-applicable rules
            if rule.category in [ValidationCategory.MESSAGE_FORMAT, ValidationCategory.PROTOCOL_COMPLIANCE, 
                               ValidationCategory.SECURITY, ValidationCategory.PERFORMANCE]:
                try:
                    results = await rule.validate(context)
                    all_results.extend(results)
                    
                    # Update statistics
                    self.validation_stats["by_rule"][rule.rule_id] += len(results)
                    
                except Exception as e:
                    logger.error(f"Validation rule {rule.rule_id} failed: {e}")
        
        # Filter by severity threshold
        filtered_results = [
            result for result in all_results
            if self._severity_level(result.severity) >= self._severity_level(self.severity_threshold)
        ]
        
        # Update statistics
        self.validation_stats["total_validations"] += 1
        if filtered_results:
            self.validation_stats["failed_validations"] += 1
        
        for result in filtered_results:
            self.validation_stats["by_severity"][result.severity.value] += 1
            self.validation_stats["by_category"][result.category.value] += 1
        
        # Store in history
        self.validation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message_id": message.id if hasattr(message, 'id') else None,
            "sender_agent": sender_agent,
            "results_count": len(filtered_results),
            "max_severity": max([result.severity for result in filtered_results], 
                              default=ValidationSeverity.INFO).value
        })
        
        return filtered_results
    
    @trace_async("validate_agent_interaction")
    async def validate_agent_interaction(self, agent_id: str) -> List[ValidationResult]:
        """Validate an agent's interaction patterns"""
        
        if not self.enabled or agent_id not in self.agent_interactions:
            return []
        
        context = self.agent_interactions[agent_id]
        all_results = []
        
        # Run agent behavior rules
        for rule in self.rules.values():
            if (rule.enabled and 
                rule.category in [ValidationCategory.AGENT_BEHAVIOR, ValidationCategory.PERFORMANCE]):
                try:
                    results = await rule.validate(context)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Agent validation rule {rule.rule_id} failed: {e}")
        
        return all_results
    
    def start_agent_interaction(
        self,
        agent_id: str,
        interaction_type: str,
        expected_capabilities: Set[AgentCapability] = None
    ):
        """Start tracking an agent interaction"""
        
        context = AgentInteractionContext(
            agent_id=agent_id,
            interaction_type=interaction_type,
            start_time=datetime.utcnow(),
            expected_capabilities=expected_capabilities or set()
        )
        
        self.agent_interactions[agent_id] = context
        logger.debug(f"Started tracking agent interaction: {agent_id}")
    
    def add_message_to_interaction(self, agent_id: str, message: A2AMessage):
        """Add a message to an agent interaction"""
        
        if agent_id in self.agent_interactions:
            self.agent_interactions[agent_id].messages.append(message)
            
            # Update demonstrated capabilities based on message content
            if hasattr(message, 'payload') and isinstance(message.payload, dict):
                if 'capability_used' in message.payload:
                    try:
                        capability = AgentCapability(message.payload['capability_used'])
                        self.agent_interactions[agent_id].actual_capabilities.add(capability)
                    except ValueError:
                        pass
    
    def end_agent_interaction(self, agent_id: str) -> Optional[AgentInteractionContext]:
        """End tracking an agent interaction"""
        
        return self.agent_interactions.pop(agent_id, None)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        return {
            "enabled": self.enabled,
            "severity_threshold": self.severity_threshold.value,
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "active_interactions": len(self.agent_interactions),
            "validation_history_size": len(self.validation_history),
            "statistics": dict(self.validation_stats)
        }
    
    def get_validation_report(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a validation report"""
        
        recent_validations = list(self.validation_history)[-100:]  # Last 100 validations
        
        if agent_id:
            recent_validations = [
                v for v in recent_validations 
                if v.get("sender_agent") == agent_id
            ]
        
        # Analyze trends
        severity_trend = defaultdict(int)
        category_trend = defaultdict(int)
        
        for validation in recent_validations:
            severity_trend[validation.get("max_severity", "info")] += 1
        
        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "validation_count": len(recent_validations),
            "severity_distribution": dict(severity_trend),
            "statistics": self.get_validation_stats(),
            "recommendations": self._generate_recommendations(recent_validations)
        }
    
    def _severity_level(self, severity: ValidationSeverity) -> int:
        """Convert severity to numeric level"""
        levels = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 2,
            ValidationSeverity.CRITICAL: 3
        }
        return levels.get(severity, 0)
    
    def _generate_recommendations(self, validations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation history"""
        recommendations = []
        
        # High failure rate
        failed_count = sum(1 for v in validations if v.get("results_count", 0) > 0)
        if failed_count > len(validations) * 0.5:
            recommendations.append(
                "High validation failure rate detected. Review message format and protocol compliance."
            )
        
        # Security issues
        security_issues = sum(1 for v in validations if v.get("max_severity") == "critical")
        if security_issues > 0:
            recommendations.append(
                "Critical security issues detected. Implement proper authentication and remove sensitive data from messages."
            )
        
        # Performance issues
        perf_issues = [v for v in validations if "performance" in str(v)]
        if len(perf_issues) > 5:
            recommendations.append(
                "Performance issues detected. Consider message batching and payload optimization."
            )
        
        return recommendations


# Global validator instance
_protocol_validator = None


def initialize_protocol_validator() -> A2AProtocolValidator:
    """Initialize global protocol validator"""
    global _protocol_validator
    
    if _protocol_validator is None:
        _protocol_validator = A2AProtocolValidator()
    
    return _protocol_validator


def get_protocol_validator() -> Optional[A2AProtocolValidator]:
    """Get the global protocol validator"""
    return _protocol_validator


def shutdown_protocol_validator():
    """Shutdown global protocol validator"""
    global _protocol_validator
    _protocol_validator = None


# Convenience functions
async def validate_message(message: A2AMessage, **kwargs) -> List[ValidationResult]:
    """Validate a message using the global validator"""
    validator = get_protocol_validator()
    if not validator:
        validator = initialize_protocol_validator()
    return await validator.validate_message(message, **kwargs)


async def validate_agent(agent_id: str) -> List[ValidationResult]:
    """Validate an agent using the global validator"""
    validator = get_protocol_validator()
    if not validator:
        return []
    return await validator.validate_agent_interaction(agent_id)


# Decorator for automatic message validation
def validate_a2a_message(trust_level: float = 0.5):
    """Decorator to automatically validate A2A messages"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract message from function arguments
            message = None
            for arg in args:
                if isinstance(arg, A2AMessage):
                    message = arg
                    break
            
            if message:
                validator = get_protocol_validator()
                if validator:
                    results = await validator.validate_message(message, trust_level=trust_level)
                    
                    # Log critical issues
                    critical_issues = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
                    if critical_issues:
                        logger.error(f"Critical validation issues in {func.__name__}: {[r.message for r in critical_issues]}")
                        # Optionally raise exception for critical issues
                        # raise ValueError(f"Message validation failed: {critical_issues[0].message}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
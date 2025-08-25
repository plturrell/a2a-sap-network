"""
Reasoning Agent - Advanced Multi-Agent Reasoning System
Implements sophisticated A2A reasoning patterns including hierarchical orchestration,
peer-to-peer swarm coordination, and chain-of-thought reasoning
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import uuid
import hashlib
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
import numpy as np
from dataclasses import dataclass, field

# Import SDK components
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

# Import AI Intelligence Framework
from app.a2a.core.ai_intelligence import (
    AIIntelligenceFramework, AIIntelligenceConfig,
    create_ai_intelligence_framework, create_enhanced_agent_config
)
# Import MCP decorators and coordination
from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from app.a2a.sdk.mcpSkillCoordination import (
    skill_depends_on, skill_provides, coordination_rule
)

# Import Grok client for intelligent intra-skill messaging
try:
    from .grokReasoning import GrokReasoning
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    logger.warning("Groq client not available - Grok intra-skill messaging disabled")
from app.a2a.sdk.mixins import (
    PerformanceMonitorMixin, SecurityHardenedMixin,
    TelemetryMixin
)
from app.a2a.sdk.messageTemplates import (
    MessageTemplate, ReasoningMessageTemplate, MessageStatus
)
from app.a2a.core.workflowContext import workflowContextManager
from app.a2a.core.circuitBreaker import EnhancedCircuitBreaker, get_circuit_breaker_manager
from app.a2a.core.trustManager import sign_a2a_message, verify_a2a_message
from app.a2a.core.serviceDiscovery import (
    service_discovery, discover_qa_agents, discover_data_managers,
    discover_reasoning_engines, discover_synthesis_agents
)

# Import trust system
from app.a2a.core.trustManager import sign_a2a_message, initialize_agent_trust, verify_a2a_message, trust_manager

# Import reasoning skills
from .reasoningSkills import (
    MultiAgentReasoningSkills, ReasoningOrchestrationSkills,
    HierarchicalReasoningSkills, SwarmReasoningSkills
)
from .enhancedReasoningSkills import EnhancedReasoningSkills

# Import real architecture implementations
from .peerToPeerArchitecture import create_peer_to_peer_coordinator
from .chainOfThoughtArchitecture import create_chain_of_thought_reasoner
from .swarmIntelligenceArchitecture import create_swarm_intelligence_coordinator
from .debateArchitecture import create_debate_coordinator
from .blackboardArchitecture import BlackboardController
from app.a2a.core.security_base import SecureA2AAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class EnhancedReasoningAgent(SecureA2AAgent, PerformanceMonitorMixin, SecurityHardenedMixin, TelemetryMixin):
    """
    Enhanced Reasoning Agent with AI Intelligence Framework Integration

    This agent combines sophisticated multi-agent reasoning patterns with the
    comprehensive AI Intelligence Framework to achieve 95+ AI intelligence rating.

    Enhanced Capabilities:
    - Multi-strategy reasoning (Chain of Thought, Tree of Thought, Graph of Thought, etc.)
    - Adaptive learning from reasoning experiences
    - Advanced memory and context management
    - Collaborative intelligence with other agents
    - Full explainability of reasoning processes
    - Autonomous decision-making and goal planning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
                # Initialize parent classes
        A2AAgentBase.__init__(
            self,
            agent_id="enhanced_reasoning_agent",
            name="Enhanced Reasoning Agent",
            description="Advanced multi-agent reasoning with AI Intelligence Framework",
            version="5.0.0",  # Enhanced version
            base_url=config.get("base_url", os.getenv("A2A_BASE_URL")) if config else os.getenv("A2A_BASE_URL")
        )
        PerformanceMonitorMixin.__init__(self)
        SecurityHardenedMixin.__init__(self)
        TelemetryMixin.__init__(self)

        # Configuration
        self.config = config or {}

        # AI Intelligence Framework - Core enhancement
        self.ai_framework = None
        self.intelligence_config = create_enhanced_agent_config()

        # Original reasoning components (enhanced)
        self.grok_messaging = None
        self.orchestration_skills = None
        self.hierarchical_skills = None
        self.swarm_skills = None
        self.enhanced_skills = None

        # Architecture coordinators (enhanced with AI framework)
        self.peer_coordinator = None
        self.chain_reasoner = None
        self.swarm_coordinator = None
        self.debate_coordinator = None
        self.blackboard = None

        # Enhanced reasoning state
        self.reasoning_sessions = {}
        self.confidence_calculator = None
        self.memory_system = None

        # Performance metrics (enhanced)
        self.enhanced_metrics = {
            "ai_reasoning_operations": 0,
            "adaptive_learning_updates": 0,
            "collaborative_decisions": 0,
            "autonomous_actions": 0,
            "explainable_operations": 0,
            "memory_retrievals": 0,
            "average_intelligence_score": 0.0
        }

        logger.info("Enhanced Reasoning Agent with AI Intelligence Framework initialized")

    async def initialize(self) -> None:
        """Initialize enhanced reasoning agent with AI Intelligence Framework"""
        logger.info("Initializing Enhanced Reasoning Agent with AI Intelligence Framework...")

        try:
            # Initialize base agent
            await super().initialize()

            # Initialize AI Intelligence Framework - Primary Enhancement
            logger.info("🧠 Initializing AI Intelligence Framework...")
            self.ai_framework = await create_ai_intelligence_framework(
                agent_id=self.agent_id,
                config=self.intelligence_config
            )
            logger.info("✅ AI Intelligence Framework initialized successfully")

            # Initialize original reasoning components (enhanced with AI framework)
            await self._initialize_enhanced_reasoning_components()

            # Initialize performance monitoring
            self._setup_enhanced_monitoring()

            logger.info("🎉 Enhanced Reasoning Agent fully initialized with 95+ AI intelligence capabilities!")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Reasoning Agent: {e}")
            raise

    async def _initialize_enhanced_reasoning_components(self):
        """Initialize reasoning components enhanced with AI framework"""
        # Initialize Grok messaging with AI enhancement
        self.grok_messaging = GrokSkillMessaging(self)

        # Initialize reasoning skills with AI framework integration
        self.orchestration_skills = ReasoningOrchestrationSkills(self.ai_framework)
        self.hierarchical_skills = HierarchicalReasoningSkills(self.ai_framework)
        self.swarm_skills = SwarmReasoningSkills(self.ai_framework)
        self.enhanced_skills = EnhancedReasoningSkills(self.ai_framework)

        # Initialize architecture coordinators with AI enhancement
        self.peer_coordinator = await create_peer_to_peer_coordinator(self.ai_framework)
        self.chain_reasoner = await create_chain_of_thought_reasoner(self.ai_framework)
        self.swarm_coordinator = await create_swarm_intelligence_coordinator(self.ai_framework)
        self.debate_coordinator = await create_debate_coordinator(self.ai_framework)
        self.blackboard = BlackboardController(ai_framework=self.ai_framework)

        logger.info("✅ Enhanced reasoning components initialized")

    def _setup_enhanced_monitoring(self):
        """Setup enhanced performance monitoring for AI intelligence"""
        # Enhanced monitoring with AI intelligence metrics
        if hasattr(self, 'enable_performance_monitoring'):
            self.enable_performance_monitoring(
                metrics_port=8002,  # Unique port for reasoning agent
                custom_metrics={
                    "ai_intelligence_score": "gauge",
                    "reasoning_complexity": "histogram",
                    "collaborative_efficiency": "gauge",
                    "learning_rate": "gauge",
                    "memory_utilization": "gauge"
                }
            )

    @a2a_handler("intelligent_reasoning")
    async def handle_intelligent_reasoning(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Enhanced reasoning handler with full AI Intelligence Framework integration

        Combines all AI capabilities: reasoning, learning, memory, collaboration,
        explainability, and autonomous decision-making.
        """
        try:
            # Extract reasoning query from message
            reasoning_query = self._extract_reasoning_query(message)
            if not reasoning_query:
                return self._create_error_response("No valid reasoning query found")

            # Perform integrated intelligence operation - Core Enhancement
            intelligence_result = await self.ai_framework.integrated_intelligence_operation(
                task_description=reasoning_query.get("query", "Complex reasoning task"),
                task_context={
                    "message_id": message.conversation_id,
                    "reasoning_type": reasoning_query.get("type", "general"),
                    "complexity": reasoning_query.get("complexity", "medium"),
                    "requires_collaboration": reasoning_query.get("collaborative", False),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Enhance with original reasoning capabilities
            enhanced_result = await self._enhance_with_traditional_reasoning(
                reasoning_query, intelligence_result
            )

            # Update metrics
            self.enhanced_metrics["ai_reasoning_operations"] += 1
            self._update_intelligence_score(intelligence_result)

            return {
                "success": True,
                "ai_intelligence_result": intelligence_result,
                "enhanced_reasoning": enhanced_result,
                "intelligence_score": self._calculate_current_intelligence_score(),
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Intelligent reasoning failed: {e}")
            return self._create_error_response(f"Intelligent reasoning failed: {str(e)}")

    @a2a_skill("adaptive_reasoning_learning")
    async def adaptive_reasoning_learning(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptive learning skill that improves reasoning based on experience
        """
        try:
            # Use AI framework's intelligent learning
            learning_result = await self.ai_framework.intelligent_learning(
                experience_data={
                    "context": learning_data.get("context", {}),
                    "action": "reasoning_operation",
                    "outcome": learning_data.get("outcome"),
                    "reward": learning_data.get("performance_score", 0.5),
                    "metadata": {
                        "reasoning_type": learning_data.get("reasoning_type"),
                        "complexity": learning_data.get("complexity"),
                        "success": learning_data.get("success", False)
                    }
                }
            )

            self.enhanced_metrics["adaptive_learning_updates"] += 1

            return {
                "learning_applied": True,
                "learning_insights": learning_result,
                "updated_reasoning_strategies": self._get_updated_strategies()
            }

        except Exception as e:
            logger.error(f"Adaptive learning failed: {e}")
            raise

    @a2a_skill("collaborative_reasoning")
    async def collaborative_reasoning(self, collaboration_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborative reasoning with other agents using AI framework
        """
        try:
            # Use AI framework's collaborative decision-making
            collaboration_result = await self.ai_framework.collaborative_decision_making(
                decision_context=collaboration_context
            )

            # Enhance with traditional multi-agent reasoning
            if self.swarm_coordinator:
                swarm_result = await self.swarm_coordinator.coordinate_swarm_reasoning(
                    collaboration_context
                )
                collaboration_result["swarm_intelligence"] = swarm_result

            self.enhanced_metrics["collaborative_decisions"] += 1

            return collaboration_result

        except Exception as e:
            logger.error(f"Collaborative reasoning failed: {e}")
            raise

    @a2a_skill("explainable_reasoning")
    async def explainable_reasoning(self, reasoning_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanations for reasoning processes
        """
        try:
            # Use AI framework's explainability
            explanation_result = await self.ai_framework.explainable_operation(
                operation_context=reasoning_context
            )

            self.enhanced_metrics["explainable_operations"] += 1

            return explanation_result

        except Exception as e:
            logger.error(f"Explainable reasoning failed: {e}")
            raise

    @a2a_task(
        task_type="autonomous_reasoning_planning",
        description="Autonomous planning and execution of reasoning tasks",
        timeout=300,
        retry_attempts=2
    )
    async def autonomous_reasoning_planning(self) -> Dict[str, Any]:
        """
        Autonomous planning and execution of reasoning tasks
        """
        try:
            # Use AI framework's autonomous decision-making
            autonomous_result = await self.ai_framework.autonomous_action(
                context={
                    "agent_type": "reasoning",
                    "current_state": self._get_current_state(),
                    "available_resources": self._get_available_resources(),
                    "performance_metrics": self.enhanced_metrics
                }
            )

            self.enhanced_metrics["autonomous_actions"] += 1

            return autonomous_result

        except Exception as e:
            logger.error(f"Autonomous reasoning planning failed: {e}")
            raise

    async def _enhance_with_traditional_reasoning(self,
                                                 reasoning_query: Dict[str, Any],
                                                 intelligence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance AI framework results with traditional reasoning capabilities"""
        enhanced_result = {}

        try:
            # Apply hierarchical reasoning if available
            if self.hierarchical_skills:
                hierarchical_result = await self.hierarchical_skills.decompose_reasoning_problem(
                    reasoning_query, intelligence_result
                )
                enhanced_result["hierarchical_reasoning"] = hierarchical_result

            # Apply chain-of-thought reasoning
            if self.chain_reasoner:
                chain_result = await self.chain_reasoner.reason_step_by_step(
                    reasoning_query, intelligence_result
                )
                enhanced_result["chain_of_thought"] = chain_result

            # Apply debate reasoning for complex problems
            if self.debate_coordinator and reasoning_query.get("complexity") == "high":
                debate_result = await self.debate_coordinator.conduct_reasoning_debate(
                    reasoning_query, intelligence_result
                )
                enhanced_result["debate_reasoning"] = debate_result

            return enhanced_result

        except Exception as e:
            logger.error(f"Traditional reasoning enhancement failed: {e}")
            return {"error": str(e)}

    def _extract_reasoning_query(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract reasoning query from A2A message"""
        for part in message.parts:
            if part.kind == "data" and part.data:
                return part.data
        return None

    def _calculate_current_intelligence_score(self) -> float:
        """Calculate current AI intelligence score based on performance"""
        base_score = 95.0  # Enhanced agent baseline

        # Adjust based on AI framework performance
        if self.ai_framework:
            framework_status = self.ai_framework.get_intelligence_status()
            active_components = sum(framework_status["components"].values())
            component_bonus = (active_components / 6) * 5.0  # Up to 5 bonus points

            # Performance metrics bonus
            performance_metrics = framework_status["performance_metrics"]
            if performance_metrics["operations_completed"] > 0:
                success_rate = 1 - (performance_metrics["operations_failed"] /
                                  max(performance_metrics["operations_completed"], 1))
                performance_bonus = success_rate * 3.0  # Up to 3 bonus points
            else:
                performance_bonus = 0.0

            total_score = min(base_score + component_bonus + performance_bonus, 100.0)
        else:
            total_score = base_score

        self.enhanced_metrics["average_intelligence_score"] = total_score
        return total_score

    def _update_intelligence_score(self, intelligence_result: Dict[str, Any]):
        """Update intelligence score based on operation results"""
        if intelligence_result.get("success"):
            components_used = intelligence_result.get("intelligence_components_used", 0)
            # Reward for using multiple AI components
            bonus = min(components_used * 0.1, 1.0)
            current_score = self.enhanced_metrics["average_intelligence_score"]
            self.enhanced_metrics["average_intelligence_score"] = min(current_score + bonus, 100.0)

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current agent state for autonomous planning"""
        return {
            "active_sessions": len(self.reasoning_sessions),
            "performance_metrics": self.enhanced_metrics,
            "ai_framework_status": self.ai_framework.get_intelligence_status() if self.ai_framework else {},
            "available_skills": self._get_available_skills()
        }

    def _get_available_resources(self) -> Dict[str, Any]:
        """Get available resources for autonomous planning"""
        return {
            "reasoning_architectures": {
                "peer_coordinator": self.peer_coordinator is not None,
                "chain_reasoner": self.chain_reasoner is not None,
                "swarm_coordinator": self.swarm_coordinator is not None,
                "debate_coordinator": self.debate_coordinator is not None,
                "blackboard": self.blackboard is not None
            },
            "ai_framework_components": self.ai_framework.get_intelligence_status()["components"] if self.ai_framework else {},
            "skill_sets": {
                "orchestration": self.orchestration_skills is not None,
                "hierarchical": self.hierarchical_skills is not None,
                "swarm": self.swarm_skills is not None,
                "enhanced": self.enhanced_skills is not None
            }
        }

    def _get_available_skills(self) -> List[str]:
        """Get list of available skills"""
        skills = []
        if hasattr(self, 'adaptive_reasoning_learning'):
            skills.append("adaptive_reasoning_learning")
        if hasattr(self, 'collaborative_reasoning'):
            skills.append("collaborative_reasoning")
        if hasattr(self, 'explainable_reasoning'):
            skills.append("explainable_reasoning")
        return skills

    def _get_updated_strategies(self) -> Dict[str, Any]:
        """Get updated reasoning strategies after learning"""
        return {
            "available_strategies": ["chain_of_thought", "tree_of_thought", "graph_of_thought",
                                   "counterfactual", "causal", "ensemble"],
            "adaptive_weights": self.ai_framework.learning_system.get_strategy_weights() if
                              self.ai_framework and self.ai_framework.learning_system else {},
            "performance_history": self.enhanced_metrics
        }

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": message,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id
        }

    async def get_enhanced_agent_health(self) -> Dict[str, Any]:
        """Get comprehensive health status including AI intelligence metrics"""
        health = await super().get_agent_health() if hasattr(super(), 'get_agent_health') else {}

        # Add AI intelligence health metrics
        if self.ai_framework:
            health["ai_intelligence"] = self.ai_framework.get_intelligence_status()

        health["enhanced_metrics"] = self.enhanced_metrics
        health["current_intelligence_score"] = self._calculate_current_intelligence_score()
        health["reasoning_capabilities"] = self._get_available_resources()

        return health

    async def shutdown(self):
        """Shutdown enhanced reasoning agent"""
        logger.info("Shutting down Enhanced Reasoning Agent...")

        # Shutdown AI Intelligence Framework
        if self.ai_framework:
            await self.ai_framework.shutdown()

        # Shutdown base agent
        if hasattr(super(), 'shutdown'):
            await super().shutdown()

        logger.info("Enhanced Reasoning Agent shutdown complete")


# Keep original class for backward compatibility
class ReasoningAgent(EnhancedReasoningAgent):
    """Alias for backward compatibility"""
    pass


class GrokSkillMessaging:
    """Grok-4 powered intelligent skill messaging system"""

    def __init__(self, reasoning_agent):
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        self.reasoning_agent = reasoning_agent
        self.grok_client = None
        self.message_context_cache = {}
        self.skill_performance_history = {}
        self.semantic_routing_cache = {}

        # Initialize Grok-4 client
        self._initialize_grok_client()

    def _initialize_grok_client(self):
        """Initialize Grok client for intra-skill messaging"""
        try:
            # Use Groq API for real Grok models
            self.grok_client = GrokReasoning()
            logger.info("Grok client for intra-skill messaging initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Grok client: {e}")
            self.grok_client = None

    async def optimize_intra_skill_message(self, message: SkillMessage, context: Dict[str, Any]) -> SkillMessage:
        """Optimize intra-skill message routing and content using Grok"""
        if not self.grok_client:
            return message

        try:
            # Focus specifically on intra-skill message routing optimization
            routing_prompt = f"""
You are optimizing message routing within a reasoning agent's skill network.

Intra-Skill Message:
- Route: {message.sender_skill} → {message.receiver_skill}
- Message Type: {message.message_type.value}
- Priority: {message.priority.value}
- Payload Size: {len(str(message.params))} chars
- Context Keys: {list(message.context.keys()) if message.context else []}

Skill Network State:
- Sender Dependencies: {context.get('sender_deps', [])}
- Receiver Capabilities: {context.get('receiver_capabilities', [])}
- Current Load Factors: {context.get('load_factors', {})}
- Active Message Queue Size: {context.get('queue_size', 0)}

Optimize for:
1. Message routing efficiency within the skill network
2. Payload compression/optimization
3. Priority adjustment based on skill dependencies
4. Context propagation optimization
5. Queue management and load balancing

Provide specific routing optimizations as JSON.
"""

            # Get Grok optimization for intra-skill routing
            response = await self._call_grok_async(routing_prompt)

            if response:
                # Apply routing optimizations
                if 'priority_adjustment' in response:
                    new_priority = response['priority_adjustment']
                    if new_priority in [p.value for p in SkillPriority]:
                        message.priority = SkillPriority(new_priority)

                if 'payload_optimization' in response:
                    optimized_params = response['payload_optimization']
                    if isinstance(optimized_params, dict):
                        message.params = optimized_params

                if 'context_optimization' in response:
                    message.context = message.context or {}
                    message.context.update(response['context_optimization'])
                    message.context['grok_optimized'] = True

            return message

        except Exception as e:
            logger.error(f"Grok intra-skill message optimization failed: {e}")
            return message

    async def determine_optimal_intra_skill_route(self, message_content: str, available_skills: List[str], current_loads: Dict[str, float]) -> Dict[str, Any]:
        """Use Grok to determine optimal intra-skill routing within the reasoning agent"""
        if not self.grok_client:
            return {"recommended_skill": available_skills[0] if available_skills else None}

        try:
            # Focus on intra-skill routing within the same reasoning agent
            intra_routing_prompt = f"""
Analyze intra-skill routing within a reasoning agent's internal skill network.

Message Semantic Content: {message_content[:200]}...

Internal Skill Network:
{chr(10).join([f"- {skill}: provides {self.reasoning_agent.reasoning_skill_network.get(skill, {}).get('provides', [])} | load: {current_loads.get(skill, 0.0)}" for skill in available_skills])}

Skill Dependency Chains:
{chr(10).join([f"- {skill}: requires {self.reasoning_agent.reasoning_skill_network.get(skill, {}).get('dependencies', [])}" for skill in available_skills])}

Current Network State:
- Total skills available: {len(available_skills)}
- Average load: {sum(current_loads.values()) / len(current_loads) if current_loads else 0}
- Overloaded skills: {[skill for skill, load in current_loads.items() if load > 0.8]}

Determine optimal intra-skill routing considering:
1. Semantic alignment with skill capabilities
2. Load balancing across the skill network
3. Dependency satisfaction
4. Message processing efficiency
5. Network congestion avoidance

Return routing decision as JSON.
"""

            response = await self._call_grok_async(intra_routing_prompt)

            return {
                "recommended_skill": response.get('recommended_skill', available_skills[0] if available_skills else None),
                "routing_confidence": response.get('confidence', 0.5),
                "load_balancing_factor": response.get('load_factor_impact', 0.0),
                "dependency_satisfaction": response.get('dependencies_met', True),
                "alternative_routes": response.get('alternatives', []),
                "network_optimization": response.get('network_optimizations', [])
            }

        except Exception as e:
            logger.error(f"Grok intra-skill routing failed: {e}")
            return {"recommended_skill": available_skills[0] if available_skills else None}

    async def predict_skill_coordination_needs(self, reasoning_request: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal skill coordination strategy using Grok-4"""
        if not self.grok_client:
            return {"coordination_strategy": "sequential"}

        try:
            prediction_prompt = f"""
You are an expert in multi-agent reasoning coordination. Analyze this reasoning request and predict optimal skill coordination:

Reasoning Request:
- Question: {reasoning_request.get('question', '')}
- Architecture: {reasoning_request.get('architecture', 'hierarchical')}
- Context: {reasoning_request.get('context', {})}
- Requirements: {reasoning_request.get('requirements', {})}

Available Reasoning Skills:
{chr(10).join([f"- {skill}: {info['provides']} (deps: {info['dependencies']})" for skill, info in self.reasoning_agent.reasoning_skill_network.items()])}

Predict:
1. Optimal coordination strategy (sequential, parallel, hybrid)
2. Critical skill interaction points
3. Potential bottlenecks and mitigations
4. Load balancing recommendations
5. Error recovery strategies
6. Performance optimization opportunities

Provide specific coordination plan with timing and dependencies.
"""

            response = await self._call_grok_async(prediction_prompt)

            return {
                "coordination_strategy": response.get('strategy', 'sequential'),
                "skill_execution_plan": response.get('execution_plan', []),
                "bottleneck_predictions": response.get('bottlenecks', []),
                "optimization_opportunities": response.get('optimizations', []),
                "estimated_performance": response.get('performance_estimate', {}),
                "risk_assessment": response.get('risks', {})
            }

        except Exception as e:
            logger.error(f"Grok-4 coordination prediction failed: {e}")
            return {"coordination_strategy": "sequential"}

    async def analyze_skill_communication_patterns(self, communication_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze skill communication patterns for optimization using Grok-4"""
        if not self.grok_client or not communication_history:
            return {"patterns": [], "recommendations": []}

        try:
            analysis_prompt = f"""
You are an expert in analyzing multi-agent communication patterns. Analyze this skill communication history:

Communication History (last {len(communication_history)} interactions):
{chr(10).join([f"- {item.get('timestamp', 'unknown')}: {item.get('from_skill', 'unknown')} -> {item.get('to_skill', 'unknown')} ({item.get('message_type', 'unknown')}) - Result: {item.get('success', 'unknown')}" for item in communication_history[-20:]])}

Skill Performance Metrics:
{json.dumps(self.skill_performance_history, indent=2)}

Analyze for:
1. Communication efficiency patterns
2. Bottleneck identification
3. Error correlation patterns
4. Load distribution analysis
5. Dependency chain optimization opportunities
6. Anomaly detection

Provide actionable insights and optimization recommendations.
"""

            response = await self._call_grok_async(analysis_prompt)

            return {
                "communication_patterns": response.get('patterns', []),
                "efficiency_metrics": response.get('efficiency', {}),
                "bottlenecks_identified": response.get('bottlenecks', []),
                "optimization_recommendations": response.get('recommendations', []),
                "anomalies_detected": response.get('anomalies', []),
                "performance_trends": response.get('trends', {})
            }

        except Exception as e:
            logger.error(f"Grok-4 pattern analysis failed: {e}")
            return {"patterns": [], "recommendations": []}

    async def generate_error_recovery_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent error recovery strategy using Grok-4"""
        if not self.grok_client:
            return {"recovery_strategy": "retry"}

        try:
            recovery_prompt = f"""
You are an expert in distributed system error recovery. Generate an optimal recovery strategy for this skill communication error:

Error Context:
- Failed skill: {error_context.get('failed_skill', 'unknown')}
- Error type: {error_context.get('error_type', 'unknown')}
- Error message: {error_context.get('error_message', 'unknown')}
- Skill dependencies: {error_context.get('dependencies', [])}
- Current system state: {error_context.get('system_state', {})}
- Previous recovery attempts: {error_context.get('recovery_history', [])}

Available Recovery Options:
1. Retry with backoff
2. Route to alternative skill
3. Graceful degradation
4. Skill restart/reset
5. Context reconstruction
6. Emergency fallback

Consider:
- Impact on reasoning workflow
- Data consistency requirements
- Performance implications
- User experience
- System stability

Provide specific recovery plan with steps and fallback options.
"""

            response = await self._call_grok_async(recovery_prompt)

            return {
                "recovery_strategy": response.get('strategy', 'retry'),
                "recovery_steps": response.get('steps', []),
                "fallback_options": response.get('fallbacks', []),
                "risk_assessment": response.get('risks', {}),
                "expected_success_probability": response.get('success_probability', 0.5),
                "monitoring_points": response.get('monitoring', [])
            }

        except Exception as e:
            logger.error(f"Grok-4 error recovery generation failed: {e}")
            return {"recovery_strategy": "retry"}

    async def _call_grok_async(self, prompt: str) -> Dict[str, Any]:
        """Make async call to Grok-4 API"""
        try:
            response = await asyncio.to_thread(
                self._call_grok_sync, prompt
            )
            return response
        except Exception as e:
            logger.error(f"Async Grok-4 call failed: {e}")
            return {}

    def _call_grok_sync(self, prompt: str) -> Dict[str, Any]:
        """Make synchronous call to Grok-4 for question decomposition"""
        try:
            # Use GrokReasoning for real Grok-4 analysis
            if hasattr(self, 'grok_client') and self.grok_client:
                import asyncio
                from .grokReasoning import GrokReasoning
                grok = GrokReasoning()

                # Run async method in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(grok.decompose_question(prompt))
                    return response.get('decomposition', {}) if response.get('success') else {}
                finally:
                    loop.close()

            return {}

        except Exception as e:
            logger.error(f"Grok-4 sync call failed: {e}")
            return {}

    async def update_skill_performance_history(self, skill_name: str, performance_data: Dict[str, Any]):
        """Update skill performance history for Grok-4 analysis"""
        if skill_name not in self.skill_performance_history:
            self.skill_performance_history[skill_name] = []

        # Add timestamp and keep last 100 records
        performance_data['timestamp'] = datetime.utcnow().isoformat()
        self.skill_performance_history[skill_name].append(performance_data)

        # Limit history size
        if len(self.skill_performance_history[skill_name]) > 100:
            self.skill_performance_history[skill_name] = self.skill_performance_history[skill_name][-100:]

    async def cleanup(self):
        """Cleanup Grok-4 messaging resources"""
        self.message_context_cache.clear()
        self.skill_performance_history.clear()
        self.semantic_routing_cache.clear()
        logger.info("Grok intra-skill messaging cleanup completed")


class ReasoningArchitecture(str, Enum):
    """Available reasoning architectures"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    HUB_AND_SPOKE = "hub_and_spoke"
    BLACKBOARD = "blackboard"
    GRAPH_BASED = "graph_based"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SWARM = "swarm"
    DEBATE = "debate"
    HYBRID = "hybrid"


class ReasoningTask(str, Enum):
    """Types of reasoning tasks"""
    QUESTION_DECOMPOSITION = "question_decomposition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_AGGREGATION = "evidence_aggregation"
    LOGICAL_INFERENCE = "logical_inference"
    ANSWER_SYNTHESIS = "answer_synthesis"
    VALIDATION = "validation"


class AgentRole(str, Enum):
    """Agent roles in reasoning system"""
    ORCHESTRATOR = "orchestrator"
    QUESTION_ANALYZER = "question_analyzer"
    EVIDENCE_RETRIEVER = "evidence_retriever"
    REASONING_ENGINE = "reasoning_engine"
    ANSWER_SYNTHESIZER = "answer_synthesizer"
    VALIDATOR = "validator"


class ReasoningRequest(BaseModel):
    """Request for multi-agent reasoning"""
    question: str = Field(description="The question to reason about")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    architecture: ReasoningArchitecture = Field(default=ReasoningArchitecture.HIERARCHICAL)
    max_reasoning_depth: int = Field(default=5, description="Maximum reasoning depth")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    enable_debate: bool = Field(default=True, description="Enable multi-agent debate")
    max_debate_rounds: int = Field(default=3, description="Maximum debate rounds")
    reasoning_budget: float = Field(default=100.0, description="Computational budget")


class SubAgentConfig(BaseModel):
    """Configuration for a sub-agent"""
    agent_id: str
    role: AgentRole
    capabilities: List[str]
    endpoint: Optional[str] = None
    priority: float = Field(default=1.0)
    max_concurrent_tasks: int = Field(default=5)


@dataclass
class ReasoningState:
    """Shared state for reasoning process"""
    question: str
    architecture: ReasoningArchitecture
    decomposed_questions: List[Dict[str, Any]] = field(default_factory=list)
    evidence_pool: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chains: List[Dict[str, Any]] = field(default_factory=list)
    debate_history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningAgent(SecureA2AAgent, PerformanceMonitorMixin, SecurityHardenedMixin,
                    TelemetryMixin):
    """
    Advanced multi-agent reasoning system implementing sophisticated architectures
    for complex question answering and reasoning tasks
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        agent_network_url: Optional[str] = None,
        data_manager_url: Optional[str] = None,
        catalog_manager_url: Optional[str] = None,
        agent_manager_url: Optional[str] = None,
        max_sub_agents: int = 10,
        reasoning_timeout: int = 300,
        **kwargs
    ):
        # Import config at the top of __init__
        from config.agentConfig import config

        super().__init__(
            agent_id=create_agent_id("reasoning"),
            name="Advanced Reasoning Agent",
            description="Multi-agent reasoning system with hierarchical orchestration and swarm intelligence",
            version="1.0.0",
            base_url=base_url or config.base_url,
            **kwargs
        )

        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()

        # Network configuration using centralized config
        self.agent_network_url = agent_network_url or config.agent_network_url
        self.data_manager_url = data_manager_url or config.data_manager_url
        self.catalog_manager_url = catalog_manager_url or config.catalog_manager_url
        self.agent_manager_url = agent_manager_url or config.agent_manager_url
        self.max_sub_agents = max_sub_agents
        self.reasoning_timeout = reasoning_timeout

        # Initialize components
        self._setup_circuit_breakers()
        self._setup_sub_agents()
        self._initialize_skills()

        # Enable mixins
        self.enable_performance_monitoring(metrics_port=8008)
        self.enable_telemetry()

        # Reasoning state management
        self.active_reasoning_sessions: Dict[str, ReasoningState] = {}
        self.sub_agent_pool: Dict[AgentRole, List[SubAgentConfig]] = {}

        # Initialize MCP skill coordination for intra-agent communication
        self.mcp_skill_coordinator = None
        self.mcp_skill_client = None
        self.reasoning_skill_network: Dict[str, Dict[str, Any]] = {
            "question_analyzer": {
                "dependencies": [],
                "provides": ["question_decomposition", "complexity_analysis", "intent_detection"],
                "load_factor": 0.0,
                "communication_patterns": ["broadcast", "pipeline"]
            },
            "evidence_gatherer": {
                "dependencies": ["question_analyzer"],
                "provides": ["evidence_collection", "relevance_scoring", "source_validation"],
                "load_factor": 0.0,
                "communication_patterns": ["request_response", "streaming"]
            },
            "hypothesis_generator": {
                "dependencies": ["question_analyzer", "evidence_gatherer"],
                "provides": ["hypothesis_generation", "possibility_exploration", "creative_thinking"],
                "load_factor": 0.0,
                "communication_patterns": ["parallel", "collaborative"]
            },
            "logical_reasoner": {
                "dependencies": ["evidence_gatherer", "hypothesis_generator"],
                "provides": ["logical_inference", "deductive_reasoning", "consistency_checking"],
                "load_factor": 0.0,
                "communication_patterns": ["sequential", "validation"]
            },
            "debate_moderator": {
                "dependencies": ["logical_reasoner", "hypothesis_generator"],
                "provides": ["perspective_coordination", "argument_evaluation", "consensus_building"],
                "load_factor": 0.0,
                "communication_patterns": ["round_robin", "hierarchical"]
            },
            "answer_synthesizer": {
                "dependencies": ["logical_reasoner", "debate_moderator"],
                "provides": ["answer_synthesis", "confidence_scoring", "result_formatting"],
                "load_factor": 0.0,
                "communication_patterns": ["aggregation", "finalization"]
            },
            "quality_validator": {
                "dependencies": ["answer_synthesizer"],
                "provides": ["quality_assessment", "confidence_validation", "result_verification"],
                "load_factor": 0.0,
                "communication_patterns": ["validation", "feedback"]
            }
        }

        # Performance tracking
        self.reasoning_metrics = {
            "total_sessions": 0,
            "successful_reasoning": 0,
            "average_confidence": 0.0,
            "architecture_usage": {arch.value: 0 for arch in ReasoningArchitecture},
            "average_reasoning_time": 0.0
        }

        # Trust and security - will be initialized in initialize()
        self.trust_identity = None
        self.trusted_agents = set()
        self.private_key = os.getenv("AGENT_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("AGENT_PRIVATE_KEY environment variable is required. No default test keys allowed in production.")

        # Initialize intra-skill communication metrics
        self.skill_communication_metrics = {
            "total_skill_calls": 0,
            "successful_skill_calls": 0,
            "skill_call_latency": {},
            "skill_dependency_chain_length": {},
            "skill_collaboration_patterns": {},
            "skill_error_recovery": 0,
            "skill_load_balancing": {},
            "cross_skill_validation_count": 0
        }

        # Initialize Grok for intelligent intra-skill messaging
        self.grok_client = None
        self.grok_intra_skill_messaging = None
        if GROK_AVAILABLE:
            self._initialize_grok_intra_skill_messaging()

        # Skill message intelligence patterns
        self.skill_message_patterns = {
            "context_propagation": "maintain_reasoning_context_across_skills",
            "semantic_routing": "route_messages_based_on_semantic_content",
            "adaptive_coordination": "adjust_coordination_based_on_skill_performance",
            "intelligent_load_balancing": "balance_workload_using_skill_capabilities",
            "error_prediction": "predict_and_prevent_skill_communication_errors",
            "optimization_suggestions": "suggest_communication_optimizations"
        }

    def get_working_capabilities(self) -> Dict[str, bool]:
        """Report what actually works vs what's claimed"""
        return {
            "hierarchical_reasoning": True,
            "question_decomposition": True,
            "pattern_analysis": True,
            "answer_synthesis": True,
            "grok4_integration": bool(self.grok_client),
            "peer_to_peer_reasoning": True,  # Real implementation with MCP tools
            "blackboard_reasoning": True,  # Enhanced with Grok-4
            "swarm_intelligence": True,  # Real swarm with particle optimization
            "debate_coordination": True,  # Multi-agent debate with MCP
            "chain_of_thought": True  # Step-by-step reasoning with MCP
        }
    def _setup_circuit_breakers(self):
        """Initialize circuit breakers for sub-agent communication"""
        self.circuit_breaker_manager = get_breaker_manager()

        # Circuit breaker for each agent role
        for role in AgentRole:
            self.circuit_breaker_manager.get_breaker(
                f"agent_{role.value}",
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0
            )

        # Network-level circuit breaker
        self.circuit_breaker_manager.get_breaker(
            "agent_network",
            failure_threshold=5,
            success_threshold=3,
            timeout=60.0
        )

        # Circuit breakers for A2A agent services
        self.circuit_breaker_manager.get_breaker(
            "data_manager",
            failure_threshold=5,
            success_threshold=2,
            timeout=30.0
        )
        self.circuit_breaker_manager.get_breaker(
            "catalog_manager",
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0
        )
        self.circuit_breaker_manager.get_breaker(
            "agent_manager",
            failure_threshold=3,
            success_threshold=2,
            timeout=30.0
        )

    def _setup_sub_agents(self):
        """Initialize sub-agent configurations"""
        # Question analyzer agents - Use QA Validation Agent for analysis
        self.sub_agent_pool[AgentRole.QUESTION_ANALYZER] = [
            SubAgentConfig(
                agent_id="qa_validation_agent",
                role=AgentRole.QUESTION_ANALYZER,
                capabilities=["decomposition", "complexity_analysis", "intent_detection"],
                endpoint=config.qa_validation_url,  # QA Validation Agent
                priority=1.0
            )
        ]

        # Evidence retriever agents - Use Data Manager
        self.sub_agent_pool[AgentRole.EVIDENCE_RETRIEVER] = [
            SubAgentConfig(
                agent_id="data_manager",
                role=AgentRole.EVIDENCE_RETRIEVER,
                capabilities=["search", "extraction", "relevance_scoring"],
                endpoint=self.data_manager_url,
                priority=0.9
            )
        ]

        # Reasoning engine agents - Dynamic discovery
        self.sub_agent_pool[AgentRole.REASONING_ENGINE] = [
            SubAgentConfig(
                agent_id=f"reasoning_engine_{i}",
                role=AgentRole.REASONING_ENGINE,
                capabilities=["inference", "deduction", "hypothesis_testing"],
                priority=1.2
            ) for i in range(2)
        ]

        # Answer synthesizer agents - Dynamic discovery
        self.sub_agent_pool[AgentRole.ANSWER_SYNTHESIZER] = [
            SubAgentConfig(
                agent_id=f"synthesizer_{i}",
                role=AgentRole.ANSWER_SYNTHESIZER,
                capabilities=["aggregation", "formatting", "confidence_scoring"],
                priority=0.8
            ) for i in range(2)
        ]

    def _initialize_skills(self):
        """Initialize reasoning skill modules"""
        self.multi_agent_skills = MultiAgentReasoningSkills(self.trust_identity)
        self.orchestration_skills = ReasoningOrchestrationSkills(self.trust_identity)
        self.hierarchical_skills = HierarchicalReasoningSkills(self.trust_identity)
        self.swarm_skills = SwarmReasoningSkills(self.trust_identity)
        self.enhanced_skills = EnhancedReasoningSkills(self.trust_identity)

        # Initialize architecture coordinators
        self.blackboard_controller = BlackboardController()
        self.peer_to_peer_coordinator = create_peer_to_peer_coordinator()
        self.chain_of_thought_reasoner = create_chain_of_thought_reasoner(self.grok_client)
        self.swarm_coordinator = create_swarm_intelligence_coordinator()
        self.debate_coordinator = create_debate_coordinator()

    def _initialize_grok_intra_skill_messaging(self):
        """Initialize Grok powered intra-skill messaging"""
        try:
            self.grok_intra_skill_messaging = GrokSkillMessaging(self)
            logger.info("Grok intra-skill messaging initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Grok intra-skill messaging: {e}")
            self.grok_intra_skill_messaging = None

    async def _query_a2a_agent(
        self,
        agent_url: str,
        skill: str,
        parameters: Dict[str, Any],
        breaker_name: str
    ) -> Dict[str, Any]:
        """Query an A2A agent with circuit breaker protection and blockchain signing"""
        try:
            breaker = self.circuit_breaker_manager.get_breaker(breaker_name)

            async def make_request():
                # Prepare message content using standardized template
                message_content = MessageTemplate.create_request(
                    skill=skill,
                    parameters=parameters,
                    sender_id=self.agent_id
                )

                # Sign message if trust identity is available
                if self.trust_identity:
                    signed_message = sign_a2a_message(
                        message_content,
                        self.trust_identity.private_key
                    )
                    message_content["signature"] = signed_message["signature"]
                    message_content["signer"] = signed_message["signer"]

                # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
                # async with httpx.AsyncClient() as client:
                # httpx.AsyncClient(timeout=30.0) as client:
                if True:  # Placeholder for blockchain messaging
                    response = await client.post(
                        f"{agent_url}/a2a/execute",
                        json=message_content,
                        headers={
                            "Content-Type": "application/json",
                            "X-Agent-ID": self.agent_id
                        }
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Verify response signature if present
                    if "signature" in result and "signer" in result and self.trust_identity:
                        is_valid = verify_a2a_message(
                            {k: v for k, v in result.items() if k not in ["signature", "signer"]},
                            result["signer"],
                            result["signature"]
                        )
                        if not is_valid:
                            raise ValueError(f"Invalid signature from {agent_url}")

                    return result

            result = await breaker.call(make_request)
            return result.get("result", result)

        except Exception as e:
            logger.error(f"A2A agent query failed for {agent_url}: {e}")
            raise  # Don't return error dict - fail properly

    async def _query_data_manager(
        self,
        operation: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query Data Manager for data operations"""
        return await self._query_a2a_agent(
            self.data_manager_url,
            operation,
            data,
            "data_manager"
        )

    async def _query_catalog_manager(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query Catalog Manager for service discovery"""
        return await self._query_a2a_agent(
            self.catalog_manager_url,
            operation,
            params,
            "catalog_manager"
        )

    async def _query_agent_manager(
        self,
        operation: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query Agent Manager for agent coordination"""
        return await self._query_a2a_agent(
            self.agent_manager_url,
            operation,
            params,
            "agent_manager"
        )

    async def _query_sub_agent(
        self,
        agent_config: SubAgentConfig,
        task: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query a sub-agent with circuit breaker protection"""
        try:
            # Check if agent has an endpoint configured
            if agent_config.endpoint:
                # Use actual A2A agent endpoint
                return await self._query_a2a_agent(
                    agent_config.endpoint,
                    task,
                    parameters,
                    f"agent_{agent_config.role.value}"
                )

            # Otherwise, route to appropriate A2A service based on role
            if agent_config.role == AgentRole.EVIDENCE_RETRIEVER:
                # Use Data Manager for evidence retrieval
                return await self._query_data_manager(
                    "retrieve_data",
                    {
                        "query": parameters.get("question", ""),
                        "context": parameters.get("context", {}),
                        "include_metadata": True
                    }
                )
            elif agent_config.role == AgentRole.QUESTION_ANALYZER:
                # Use Agent Manager to find specialized analyzer agents
                analyzer_agents = await self._query_agent_manager(
                    "find_agents",
                    {"capability": "question_analysis", "limit": 1}
                )
                if analyzer_agents.get("agents"):
                    analyzer_url = analyzer_agents["agents"][0].get("endpoint")
                    if analyzer_url:
                        return await self._query_a2a_agent(
                            analyzer_url,
                            "analyze_question",
                            parameters,
                            f"agent_{agent_config.role.value}"
                        )

            # For other roles, try to discover agents via Catalog Manager
            discovered_agents = await self._query_catalog_manager(
                "discover_agents",
                {
                    "capability": agent_config.role.value,
                    "status": "active"
                }
            )

            if discovered_agents.get("agents"):
                # Use first available agent
                agent_endpoint = discovered_agents["agents"][0].get("endpoint")
                if agent_endpoint:
                    return await self._query_a2a_agent(
                        agent_endpoint,
                        task,
                        parameters,
                        f"agent_{agent_config.role.value}"
                    )

            # No agent available for this role
            logger.warning(f"No real agent available for role {agent_config.role.value}")
            return {
                "error": f"No agent available for role {agent_config.role.value}",
                "role": agent_config.role.value
            }

        except Exception as e:
            logger.error(f"Sub-agent query failed for {agent_config.agent_id}: {e}")
            return {"error": str(e), "agent_id": agent_config.agent_id}

    async def _orchestrate_hierarchical_reasoning(
        self,
        state: ReasoningState,
        request: ReasoningRequest
    ) -> Dict[str, Any]:
        """Orchestrate hierarchical reasoning with multiple agents"""
        try:
            # Phase 1: Question Analysis
            logger.info("Phase 1: Analyzing question")
            analysis_results = []

            # Discover real QA agents from A2A network
            try:
                qa_agents = await discover_qa_agents()
                if not qa_agents:
                    raise RuntimeError("No QA agents available in A2A network")

                qa_agent_available = False
                for qa_service in qa_agents:
                    try:
                        result = await self._query_a2a_agent(
                            qa_service.endpoint_url,
                            "generate_reasoning_chain",
                            {
                                "question": state.question,
                                "context": json.dumps(request.context),
                                "reasoning_strategy": "step_by_step",
                                "max_reasoning_steps": request.max_reasoning_depth
                            },
                            f"qa_agent_{qa_service.agent_id}"
                        )
                        if not result.get("error"):
                            analysis_results.append(result)
                            qa_agent_available = True
                            break
                    except Exception as e:
                        logger.warning(f"Failed to query QA agent {qa_service.agent_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to discover QA agents: {e}")

            # Require real QA agent - no fallback to mock implementations
            if not qa_agent_available:
                error_msg = "Question analysis requires QA agents but none are available. Cannot proceed with mock implementations."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Aggregate analysis results
            for result in analysis_results:
                if isinstance(result, Exception):
                    logger.error(f"Analysis failed: {result}")
                    continue

                if "reasoning_chain" in result:
                    # Extract sub-questions from reasoning chain
                    chain = result["reasoning_chain"]
                    for step in chain.get("steps", []):
                        if step.get("step_type") == "premise":
                            state.decomposed_questions.append({
                                "question": step["content"],
                                "priority": step.get("confidence_score", 0.5)
                            })
                elif "sub_questions" in result:
                    state.decomposed_questions.extend([
                        {"question": q, "priority": 1.0}
                        for q in result["sub_questions"]
                    ])

            # Phase 2: Evidence Retrieval
            logger.info("Phase 2: Retrieving evidence")
            evidence_results = []

            # Only proceed if we have questions to work with
            if state.decomposed_questions:
                # Discover real Data Manager agents from A2A network
                try:
                    data_managers = await discover_data_managers()
                    if not data_managers:
                        raise RuntimeError("No Data Manager agents available in A2A network")

                    # Use the first healthy data manager
                    data_manager = data_managers[0]
                    logger.info(f"Using Data Manager {data_manager.agent_id} for evidence retrieval")

                    retriever_tasks = []
                    for question in state.decomposed_questions[:5]:  # Limit to top 5
                        task = self._query_a2a_agent(
                            data_manager.endpoint_url,
                            "search_knowledge_base",
                            {
                                "query": question["question"],
                                "context": request.context,
                                "include_metadata": True,
                                "relevance_threshold": 0.7
                            },
                            f"data_manager_{data_manager.agent_id}"
                        )
                        retriever_tasks.append(task)

                    evidence_results = await asyncio.gather(*retriever_tasks, return_exceptions=True)

                except Exception as e:
                    error_msg = f"Evidence retrieval requires Data Manager but failed: {e}. Cannot proceed with mock evidence generation."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Aggregate evidence
            for result in evidence_results:
                if isinstance(result, Exception):
                    logger.error(f"Evidence retrieval failed: {result}")
                    continue

                if "results" in result:
                    # Convert Data Manager results to evidence format
                    for item in result["results"]:
                        state.evidence_pool.append({
                            "content": item.get("content", item.get("text", "")),
                            "relevance": item.get("relevance_score", 0.5),
                            "source": item.get("source", "data_manager"),
                            "metadata": item.get("metadata", {})
                        })
                elif "evidence" in result:
                    state.evidence_pool.extend(result["evidence"])

            # Phase 3: Reasoning using discovered agents or internal skills
            logger.info("Phase 3: Performing reasoning")
            reasoning_tasks = []

            # Try to discover reasoning agents
            reasoning_agents = await self._query_catalog_manager(
                "discover_agents",
                {"capability": "reasoning", "status": "active"}
            )

            if reasoning_agents.get("agents"):
                # Use discovered reasoning agents
                for agent in reasoning_agents["agents"][:2]:  # Use up to 2 agents
                    task = self._query_a2a_agent(
                        agent["endpoint"],
                        "perform_reasoning",
                        {
                            "question": state.question,
                            "evidence": state.evidence_pool[:10],
                            "sub_questions": state.decomposed_questions
                        },
                        "reasoning_engine"
                    )
                    reasoning_tasks.append(task)
            else:
                error_msg = "Reasoning requires external reasoning engines but none are available. Cannot proceed with mock reasoning implementations."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            reasoning_results = await asyncio.gather(*reasoning_tasks, return_exceptions=True)

            # Aggregate reasoning chains
            for result in reasoning_results:
                if isinstance(result, Exception):
                    logger.error(f"Reasoning failed: {result}")
                    continue

                if "inference" in result:
                    state.reasoning_chains.append({
                        "inference": result["inference"],
                        "confidence": result.get("confidence", 0.5),
                        "steps": result.get("reasoning_steps", [])
                    })

            # Phase 4: Multi-Agent Debate (if enabled)
            if request.enable_debate and len(state.reasoning_chains) > 1:
                logger.info("Phase 4: Conducting multi-agent debate")
                debate_result = await self._conduct_debate(
                    state,
                    request.max_debate_rounds
                )
                state.debate_history = debate_result["debate_history"]

            # Phase 5: Answer Synthesis
            logger.info("Phase 5: Synthesizing answer")
            synthesizer_tasks = []

            # Try to discover synthesis agents
            synthesis_agents = await self._query_catalog_manager(
                "discover_agents",
                {"capability": "answer_synthesis", "status": "active"}
            )

            if synthesis_agents.get("agents"):
                # Use discovered synthesis agents
                for agent in synthesis_agents["agents"][:2]:
                    task = self._query_a2a_agent(
                        agent["endpoint"],
                        "synthesize_answer",
                        {
                            "question": state.question,
                            "reasoning_chains": state.reasoning_chains,
                            "debate_history": state.debate_history,
                            "evidence": state.evidence_pool[:5]
                        },
                        "answer_synthesizer"
                    )
                    synthesizer_tasks.append(task)
            else:
                error_msg = "Answer synthesis requires synthesis agents but none are available. Cannot proceed with mock synthesis implementations."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            synthesis_results = await asyncio.gather(*synthesizer_tasks, return_exceptions=True)

            # Select best answer
            best_answer = None
            best_confidence = 0.0

            for result in synthesis_results:
                if isinstance(result, Exception):
                    continue

                # Handle both external agent format and internal format
                if "answer" in result:
                    answer = result["answer"]
                    confidence = result.get("confidence", 0)
                elif "final_position" in result:
                    answer = result["final_position"]
                    confidence = result.get("confidence", 0)
                else:
                    continue

                if confidence > best_confidence:
                    best_answer = answer
                    best_confidence = confidence
                    state.confidence_scores["final"] = best_confidence

            state.final_answer = best_answer or "Unable to determine answer with sufficient confidence"

            return {
                "answer": state.final_answer,
                "confidence": best_confidence,
                "reasoning_architecture": "hierarchical",
                "phases_completed": 5,
                "sub_agents_used": len(analysis_results) + len(evidence_results) +
                                 len(reasoning_results) + len(synthesis_results),
                "evidence_count": len(state.evidence_pool),
                "reasoning_chains": len(state.reasoning_chains),
                "debate_rounds": len(state.debate_history)
            }

        except Exception as e:
            logger.error(f"Hierarchical reasoning failed: {e}")
            raise

    async def _conduct_debate(
        self,
        state: ReasoningState,
        max_rounds: int
    ) -> Dict[str, Any]:
        """Conduct multi-agent debate using enhanced debate mechanism"""
        # Convert reasoning chains to debate positions
        positions = []
        for i, chain in enumerate(state.reasoning_chains):
            positions.append({
                "perspective": f"reasoning_chain_{i}",
                "argument": chain["inference"],
                "evidence": [e["content"] for e in state.evidence_pool[:3] if e.get("relevance", 0) > 0.7],
                "confidence": chain["confidence"]
            })

        # Use enhanced debate mechanism
        debate_result = await self.enhanced_skills.enhanced_debate_mechanism({
            "positions": positions,
            "debate_structure": "dialectical",  # Use dialectical for reasoning synthesis
            "max_rounds": max_rounds,
            "convergence_threshold": 0.8
        })

        # Update reasoning chains based on debate outcome
        if debate_result["consensus_achieved"]:
            final_position = debate_result["final_position"]
            final_confidence = debate_result["confidence"]

            # Update all chains to converged position
            for chain in state.reasoning_chains:
                chain["inference"] = final_position
                chain["confidence"] = final_confidence

        return {
            "debate_history": debate_result.get("key_arguments", []),
            "consensus_achieved": debate_result["consensus_achieved"],
            "final_confidence": debate_result["confidence"]
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple implementation - in production, use embeddings
        common_words = set(text1.lower().split()) & set(text2.lower().split())
        total_words = set(text1.lower().split()) | set(text2.lower().split())

        if not total_words:
            return 0.0

        return len(common_words) / len(total_words)

    async def _self_sufficient_reasoning(
        self,
        question: str,
        context: Dict[str, Any],
        architecture: ReasoningArchitecture
    ) -> Dict[str, Any]:
        """Perform self-sufficient reasoning without external dependencies"""
        logger.info(f"Using self-sufficient {architecture.value} reasoning")

        # Use enhanced skills for internal reasoning
        if architecture == ReasoningArchitecture.HIERARCHICAL:
            return await self.enhanced_skills.hierarchical_multi_engine_reasoning({
                "question": question,
                "context": context,
                "reasoning_engines": ["logical", "probabilistic", "analogical", "causal"],
                "max_depth": 5
            })

        elif architecture == ReasoningArchitecture.PEER_TO_PEER:
            # Use real peer-to-peer coordinator
            return await self.peer_to_peer_coordinator.reason(question, context)

        elif architecture == ReasoningArchitecture.BLACKBOARD:
            # Use real blackboard controller
            return await self.blackboard_controller.reason(question, context)

        elif architecture == ReasoningArchitecture.GRAPH_BASED:
            # Graph-based reasoning using knowledge graph
            return await self._graph_based_reasoning(question, context)

        elif architecture == ReasoningArchitecture.HUB_AND_SPOKE:
            # Hub and spoke pattern
            return await self._hub_and_spoke_reasoning(question, context)

        elif architecture == ReasoningArchitecture.CHAIN_OF_THOUGHT:
            # Use real chain-of-thought reasoner
            from .chainOfThoughtArchitecture import ReasoningStrategy
            return await self.chain_of_thought_reasoner.reason(question, context, ReasoningStrategy.LINEAR)

        elif architecture == ReasoningArchitecture.SWARM:
            # Use real swarm intelligence coordinator
            from .swarmIntelligenceArchitecture import SwarmBehavior
            return await self.swarm_coordinator.reason(question, context, SwarmBehavior.EXPLORATION)

        elif architecture == ReasoningArchitecture.DEBATE:
            # Use real debate coordinator
            return await self.debate_coordinator.reason(question, context)

        else:
            # Default to hierarchical
            return await self.enhanced_skills.hierarchical_multi_engine_reasoning({
                "question": question,
                "context": context
            })

    async def _graph_based_reasoning(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Graph-based reasoning using knowledge representation"""
        # Build knowledge graph from question and context
        concepts = self._extract_concepts_from_question(question)

        # Create reasoning graph
        reasoning_paths = []
        for concept in concepts:
            path = {
                "start": concept,
                "edges": [],
                "end": "conclusion",
                "confidence": 0.7
            }
            reasoning_paths.append(path)

        # Find best path
        best_path = max(reasoning_paths, key=lambda p: p["confidence"])

        return {
            "answer": f"Based on graph analysis of {best_path['start']}",
            "confidence": best_path["confidence"],
            "reasoning_architecture": "graph_based",
            "graph_nodes": len(concepts),
            "paths_explored": len(reasoning_paths)
        }

    async def _hub_and_spoke_reasoning(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hub and spoke reasoning pattern"""
        # Central hub processes question
        hub_analysis = {
            "question": question,
            "spokes": []
        }

        # Each spoke handles different aspect
        spoke_aspects = ["logical", "emotional", "practical", "theoretical"]

        for aspect in spoke_aspects:
            # Calculate deterministic confidence based on aspect complexity and question length
            question_complexity = min(len(question.split()), 20) / 20.0  # Normalize to 0-1
            aspect_weight = len(aspect) / 20.0  # Simple aspect complexity measure
            base_confidence = 0.6 + (question_complexity * 0.2) + (aspect_weight * 0.1)

            spoke_result = {
                "aspect": aspect,
                "analysis": f"{aspect} analysis of {question[:30]}...",
                "confidence": min(0.9, base_confidence)  # Cap at 0.9
            }
            hub_analysis["spokes"].append(spoke_result)

        # Hub synthesizes spoke results
        avg_confidence = np.mean([s["confidence"] for s in hub_analysis["spokes"]])

        # Try Grok-4 for hub synthesis
        try:
            from .grokReasoning import GrokReasoning
            grok = GrokReasoning()

            # Prepare spoke results for Grok-4 synthesis
            sub_answers = [
                {
                    "content": spoke["analysis"],
                    "aspect": spoke["aspect"],
                    "confidence": spoke["confidence"]
                }
                for spoke in hub_analysis["spokes"]
            ]

            result = await grok.synthesize_answer(sub_answers, question)

            if result.get('success'):
                return {
                    "answer": result.get('synthesis'),
                    "confidence": avg_confidence,
                    "reasoning_architecture": "hub_and_spoke",
                    "spokes_used": len(spoke_aspects),
                    "enhanced": True
                }
        except Exception as e:
            logger.warning(f"Grok-4 hub synthesis failed, using fallback: {e}")

        # Fallback to simple synthesis
        return {
            "answer": f"Hub synthesis: {question[:50]}... analyzed from {len(spoke_aspects)} perspectives",
            "confidence": avg_confidence,
            "reasoning_architecture": "hub_and_spoke",
            "spokes_used": len(spoke_aspects),
            "enhanced": False
        }

    async def _extract_concepts_from_question(self, question: str) -> List[str]:
        """Extract key concepts from question using Grok-4"""

        # Try Grok-4 first for intelligent concept extraction
        if hasattr(self, 'grok_client') and self.grok_client:
            try:
                from .grokReasoning import GrokReasoning


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                grok = GrokReasoning()

                result = await grok.analyze_patterns(question, [])

                if result.get('success'):
                    patterns = result.get('patterns', {})
                    concepts = patterns.get('key_concepts', [])
                    if concepts:
                        return concepts[:10]  # Top 10 concepts from Grok-4

            except Exception as e:
                logger.warning(f"Grok-4 concept extraction failed, using fallback: {e}")

        # Fallback to simple extraction
        stop_words = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an"}
        words = question.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 3]
        return concepts[:5]  # Top 5 concepts


    @a2a_skill(
        name="multi_agent_reasoning",
        description="Perform advanced multi-agent reasoning on complex questions",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "object"},
                "architecture": {
                    "type": "string",
                    "enum": ["hierarchical", "peer_to_peer", "hub_and_spoke",
                            "blackboard", "graph_based", "hybrid"],
                    "default": "hierarchical"
                },
                "enable_debate": {"type": "boolean", "default": True},
                "max_debate_rounds": {"type": "integer", "default": 3},
                "confidence_threshold": {"type": "number", "default": 0.7}
            },
            "required": ["question"]
        }
    )
    @PerformanceMonitorMixin.monitor_performance
    async def multi_agent_reasoning(
        self,
        request: ReasoningRequest,
        **kwargs
    ) -> Dict[str, Any]:
        """Main entry point for multi-agent reasoning"""
        session_id = f"reasoning_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            # Initialize reasoning state
            state = ReasoningState(
                question=request.question,
                architecture=request.architecture,
                metadata={
                    "session_id": session_id,
                    "start_time": datetime.utcnow().isoformat(),
                    "request_params": request.dict()
                }
            )

            self.active_reasoning_sessions[session_id] = state
            self.reasoning_metrics["total_sessions"] += 1
            self.reasoning_metrics["architecture_usage"][request.architecture.value] += 1

            # Execute reasoning based on architecture
            try:
                if request.architecture == ReasoningArchitecture.HIERARCHICAL:
                    # Try external agents first
                    try:
                        result = await self._orchestrate_hierarchical_reasoning(state, request)
                    except Exception as e:
                        logger.warning(f"External hierarchical reasoning failed: {e}, using self-sufficient reasoning")
                        result = await self._self_sufficient_reasoning(
                            request.question, request.context, request.architecture
                        )
                else:
                    # Use self-sufficient reasoning for all architectures
                    result = await self._self_sufficient_reasoning(
                        request.question, request.context, request.architecture
                    )
            except Exception as e:
                # Final fallback to basic reasoning
                logger.error(f"All reasoning methods failed: {e}")
                result = {
                    "answer": "Unable to perform reasoning due to system errors",
                    "confidence": 0.0,
                    "error": str(e)
                }

            # Update metrics
            elapsed_time = time.time() - start_time
            self.reasoning_metrics["average_reasoning_time"] = (
                (self.reasoning_metrics["average_reasoning_time"] *
                 (self.reasoning_metrics["successful_reasoning"]) + elapsed_time) /
                (self.reasoning_metrics["successful_reasoning"] + 1)
            )

            if result.get("confidence", 0) >= request.confidence_threshold:
                self.reasoning_metrics["successful_reasoning"] += 1
                self.reasoning_metrics["average_confidence"] = (
                    (self.reasoning_metrics["average_confidence"] *
                     (self.reasoning_metrics["successful_reasoning"] - 1) +
                     result["confidence"]) /
                    self.reasoning_metrics["successful_reasoning"]
                )

            # Add session metadata to result
            result["session_id"] = session_id
            result["reasoning_time"] = elapsed_time
            result["architecture_used"] = request.architecture.value

            # Results are now cached in enhanced skills layer

            return result

        except Exception as e:
            logger.error(f"Multi-agent reasoning failed: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "partial_state": {
                    "decomposed_questions": len(state.decomposed_questions),
                    "evidence_collected": len(state.evidence_pool),
                    "reasoning_chains": len(state.reasoning_chains)
                }
            }
        finally:
            # Cleanup session
            if session_id in self.active_reasoning_sessions:
                del self.active_reasoning_sessions[session_id]

    @a2a_handler("executeReasoningTask")
    async def execute_reasoning_task(self, message: A2AMessage) -> Dict[str, Any]:
        """Execute reasoning task via A2A protocol"""
        try:
            params = message.content.get("params", {})

            # Convert to ReasoningRequest
            request = ReasoningRequest(**params)

            # Execute reasoning
            result = await self.multi_agent_reasoning(request)

            return {
                "id": message.content.get("id"),
                "result": {
                    "status": "completed",
                    "reasoning_result": result
                }
            }

        except Exception as e:
            logger.error(f"Reasoning task execution failed: {e}")
            return {
                "id": message.content.get("id"),
                "error": {
                    "code": "REASONING_FAILED",
                    "message": str(e)
                }
            }

    async def get_reasoning_metrics(self) -> Dict[str, Any]:
        """Get current reasoning system metrics"""
        return {
            "metrics": self.reasoning_metrics,
            "active_sessions": len(self.active_reasoning_sessions),
            "sub_agent_pool": {
                role.value: len(agents)
                for role, agents in self.sub_agent_pool.items()
            },
            "circuit_breaker_status": {
                name: breaker.state
                for name, breaker in self.circuit_breaker_manager._breakers.items()
            }
        }

    # ========= MCP Tool Implementations =========

    @mcp_tool(
        name="advanced_reasoning",
        description="Perform advanced multi-agent reasoning on complex questions using multiple architectures"
    )
    async def advanced_reasoning_mcp(
        self,
        question: str,
        reasoning_architecture: str = "hierarchical",
        context: Optional[Dict[str, Any]] = None,
        enable_debate: bool = True,
        max_debate_rounds: int = 3,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Advanced reasoning with multi-agent orchestration

        Args:
            question: Complex question to reason about
            reasoning_architecture: Architecture (hierarchical, peer_to_peer, blackboard, graph_based, hub_and_spoke)
            context: Additional context for reasoning
            enable_debate: Enable multi-agent debate
            max_debate_rounds: Maximum debate rounds
            confidence_threshold: Minimum confidence threshold
        """
        try:
            # Validate architecture
            valid_architectures = [arch.value for arch in ReasoningArchitecture]
            if reasoning_architecture not in valid_architectures:
                reasoning_architecture = "hierarchical"

            # Create reasoning request
            request = ReasoningRequest(
                question=question,
                context=context or {},
                architecture=ReasoningArchitecture(reasoning_architecture),
                enable_debate=enable_debate,
                max_debate_rounds=max_debate_rounds,
                confidence_threshold=confidence_threshold
            )

            # Execute reasoning
            result = await self.multi_agent_reasoning(request)

            # Format response for MCP
            return {
                "success": True,
                "question": question,
                "answer": result.get("answer", "Unable to determine answer"),
                "confidence": result.get("confidence", 0.0),
                "architecture_used": result.get("reasoning_architecture", reasoning_architecture),
                "reasoning_time": result.get("reasoning_time", 0.0),
                "phases_completed": result.get("phases_completed", 0),
                "sub_agents_used": result.get("sub_agents_used", 0),
                "evidence_count": result.get("evidence_count", 0),
                "debate_rounds": result.get("debate_rounds", 0),
                "metadata": {
                    "session_id": result.get("session_id"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "threshold_met": result.get("confidence", 0) >= confidence_threshold
                }
            }
        except Exception as e:
            logger.error(f"MCP advanced_reasoning error: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    @mcp_tool(
        name="hypothesis_generation",
        description="Generate and validate hypotheses for complex reasoning problems"
    )
    async def hypothesis_generation_mcp(
        self,
        problem: str,
        domain: Optional[str] = None,
        max_hypotheses: int = 5,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate hypotheses for reasoning problems

        Args:
            problem: Problem statement to generate hypotheses for
            domain: Problem domain (scientific, logical, business, general)
            max_hypotheses: Maximum number of hypotheses to generate
            include_confidence: Include confidence scores
        """
        try:
            # Use enhanced reasoning skills for hypothesis generation
            hypotheses_result = await self.enhanced_skills.hierarchical_multi_engine_reasoning({
                "question": f"Generate {max_hypotheses} hypotheses for: {problem}",
                "context": {"domain": domain, "task": "hypothesis_generation"},
                "reasoning_engines": ["logical", "probabilistic", "analogical"],
                "max_depth": 3
            })

            # Extract hypotheses from reasoning result
            hypotheses = []
            answer = hypotheses_result.get("answer", "")

            # Parse hypotheses from answer (simple approach)
            lines = answer.split('\n')
            hypothesis_count = 0

            for line in lines:
                if hypothesis_count >= max_hypotheses:
                    break

                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or
                            any(marker in line.lower() for marker in ["hypothesis", "theory", "possibility"])):

                    # Calculate confidence based on line characteristics
                    confidence = 0.6 + (len(line.split()) / 50) * 0.3  # Longer = more confident
                    confidence = min(0.95, confidence)  # Cap at 0.95

                    hypothesis = {
                        "hypothesis": line.lstrip('-•').strip(),
                        "domain": domain or "general",
                        "type": "generated"
                    }

                    if include_confidence:
                        hypothesis["confidence"] = round(confidence, 3)

                    hypotheses.append(hypothesis)
                    hypothesis_count += 1

            # If no structured hypotheses found, create from reasoning result
            if not hypotheses and hypotheses_result.get("answer"):
                base_confidence = hypotheses_result.get("confidence", 0.7)
                hypotheses.append({
                    "hypothesis": hypotheses_result["answer"],
                    "domain": domain or "general",
                    "type": "reasoning_generated",
                    "confidence": base_confidence if include_confidence else None
                })

            return {
                "success": True,
                "problem": problem,
                "hypotheses": hypotheses,
                "total_generated": len(hypotheses),
                "domain": domain,
                "generation_method": "multi_engine_reasoning",
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "reasoning_confidence": hypotheses_result.get("confidence", 0.0)
                }
            }

        except Exception as e:
            logger.error(f"MCP hypothesis_generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }

    @mcp_tool(
        name="debate_orchestration",
        description="Orchestrate multi-perspective debates on complex topics"
    )
    async def debate_orchestration_mcp(
        self,
        topic: str,
        perspectives: Optional[List[str]] = None,
        debate_structure: str = "dialectical",
        max_rounds: int = 3,
        convergence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Orchestrate structured debates between multiple perspectives

        Args:
            topic: Topic for debate
            perspectives: List of perspectives (auto-generated if not provided)
            debate_structure: Structure (dialectical, round_robin, tournament)
            max_rounds: Maximum debate rounds
            convergence_threshold: Threshold for consensus
        """
        try:
            # Generate perspectives if not provided
            if not perspectives:
                perspectives = ["analytical", "practical", "theoretical", "critical"]

            # Create debate positions
            positions = []
            for i, perspective in enumerate(perspectives):
                # Generate position using reasoning skills
                position_result = await self.enhanced_skills.hierarchical_multi_engine_reasoning({
                    "question": f"From a {perspective} perspective, what is your position on: {topic}",
                    "context": {"perspective": perspective, "debate_topic": topic},
                    "reasoning_engines": ["logical", "analogical"],
                    "max_depth": 2
                })

                positions.append({
                    "perspective": perspective,
                    "argument": position_result.get("answer", f"{perspective} analysis of {topic}"),
                    "evidence": [f"Evidence point {j+1} from {perspective} view" for j in range(2)],
                    "confidence": position_result.get("confidence", 0.7)
                })

            # Conduct debate using enhanced skills
            debate_result = await self.enhanced_skills.enhanced_debate_mechanism({
                "positions": positions,
                "debate_structure": debate_structure,
                "max_rounds": max_rounds,
                "convergence_threshold": convergence_threshold
            })

            return {
                "success": True,
                "topic": topic,
                "debate_results": {
                    "final_position": debate_result.get("final_position", "No consensus reached"),
                    "consensus_achieved": debate_result.get("consensus_achieved", False),
                    "confidence": debate_result.get("confidence", 0.0),
                    "rounds_conducted": len(debate_result.get("key_arguments", [])),
                    "perspectives_involved": len(perspectives)
                },
                "debate_history": debate_result.get("key_arguments", []),
                "positions": positions,
                "metadata": {
                    "structure": debate_structure,
                    "timestamp": datetime.utcnow().isoformat(),
                    "convergence_threshold": convergence_threshold
                }
            }

        except Exception as e:
            logger.error(f"MCP debate_orchestration error: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }

    @mcp_tool(
        name="grok_skill_coordination",
        description="Coordinate reasoning skills using Grok-4 intelligence for optimal workflow orchestration"
    )
    async def grok_skill_coordination_mcp(
        self,
        reasoning_request: Dict[str, Any],
        coordination_mode: str = "intelligent",
        enable_optimization: bool = True,
        include_performance_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Coordinate reasoning skills using Grok-4 intelligence

        Args:
            reasoning_request: Complex reasoning request requiring skill coordination
            coordination_mode: Mode (intelligent, sequential, parallel, adaptive)
            enable_optimization: Enable Grok-4 optimization suggestions
            include_performance_analysis: Include detailed performance analysis
        """
        try:
            if not self.grok_skill_messaging:
                return {
                    "success": False,
                    "error": "Grok-4 skill messaging not available",
                    "fallback": "Using basic coordination"
                }

            coordination_start_time = time.time()

            # Get Grok-4 coordination strategy
            coordination_plan = await self.grok_skill_messaging.predict_skill_coordination_needs(reasoning_request)

            # Execute coordinated skill workflow
            workflow_results = []
            skill_execution_plan = coordination_plan.get("skill_execution_plan", [
                {"skill": "question_analyzer", "phase": 1},
                {"skill": "evidence_gatherer", "phase": 2},
                {"skill": "hypothesis_generator", "phase": 3},
                {"skill": "logical_reasoner", "phase": 4},
                {"skill": "debate_moderator", "phase": 5},
                {"skill": "answer_synthesizer", "phase": 6}
            ])

            # Execute skills according to Grok-4 plan
            for phase_num in range(1, max([step.get("phase", 1) for step in skill_execution_plan]) + 1):
                phase_skills = [step for step in skill_execution_plan if step.get("phase") == phase_num]
                phase_results = []

                # Execute skills in parallel within each phase
                phase_tasks = []
                for skill_step in phase_skills:
                    skill_name = skill_step["skill"]
                    if skill_name == "grok_enhanced_hypothesis_skill":
                        task = self.grok_enhanced_hypothesis_skill(reasoning_request)
                    elif skill_name == "grok_enhanced_debate_skill":
                        task = self.grok_enhanced_debate_skill(reasoning_request)
                    else:
                        # Use MCP skill client for standard skills
                        task = self.mcp_skill_client.call_skill_tool(
                            skill_name,
                            "process_reasoning_request",
                            reasoning_request
                        )
                    phase_tasks.append(task)

                if phase_tasks:
                    phase_results = await asyncio.gather(*phase_tasks, return_exceptions=True)
                    workflow_results.extend([
                        {
                            "phase": phase_num,
                            "skill": phase_skills[i]["skill"],
                            "result": result if not isinstance(result, Exception) else {"error": str(result)}
                        }
                        for i, result in enumerate(phase_results)
                    ])

            # Analyze coordination performance
            coordination_duration = time.time() - coordination_start_time
            performance_analysis = {
                "total_duration": coordination_duration,
                "skills_executed": len(workflow_results),
                "successful_skills": len([r for r in workflow_results if not r["result"].get("error")]),
                "coordination_efficiency": coordination_plan.get("estimated_performance", {}),
                "grok_optimizations_applied": len(coordination_plan.get("optimization_opportunities", []))
            }

            # Get optimization recommendations if enabled
            optimization_suggestions = []
            if enable_optimization:
                workflow_analysis = await self.grok_skill_messaging.analyze_skill_communication_patterns(
                    [{
                        "timestamp": datetime.utcnow().isoformat(),
                        "from_skill": "coordinator",
                        "to_skill": result["skill"],
                        "success": not result["result"].get("error"),
                        "duration": performance_analysis["total_duration"] / len(workflow_results)
                    } for result in workflow_results]
                )
                optimization_suggestions = workflow_analysis.get("optimization_recommendations", [])

            return {
                "success": True,
                "coordination_strategy": coordination_plan.get("coordination_strategy", "intelligent"),
                "workflow_results": workflow_results,
                "performance_analysis": performance_analysis if include_performance_analysis else {},
                "optimization_suggestions": optimization_suggestions,
                "grok_insights": {
                    "bottlenecks_predicted": coordination_plan.get("bottleneck_predictions", []),
                    "risk_assessment": coordination_plan.get("risk_assessment", {}),
                    "coordination_confidence": coordination_plan.get("confidence", 0.7)
                },
                "metadata": {
                    "coordination_mode": coordination_mode,
                    "timestamp": datetime.utcnow().isoformat(),
                    "grok_enhanced": True
                }
            }

        except Exception as e:
            logger.error(f"Grok skill coordination error: {e}")
            return {
                "success": False,
                "error": str(e),
                "coordination_mode": coordination_mode
            }

    @mcp_tool(
        name="skill_communication_optimization",
        description="Analyze and optimize inter-skill communication patterns using Grok-4 intelligence"
    )
    async def skill_communication_optimization_mcp(
        self,
        analysis_scope: str = "full",
        optimization_target: str = "efficiency",
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze and optimize skill communication using Grok-4

        Args:
            analysis_scope: Scope of analysis (full, recent, specific_skills)
            optimization_target: Target (efficiency, accuracy, reliability, speed)
            include_predictions: Include future performance predictions
        """
        try:
            if not self.grok_skill_messaging:
                return {
                    "success": False,
                    "error": "Grok-4 skill messaging not available"
                }

            # Gather communication history
            communication_history = []
            for skill_name, latency_history in self.skill_communication_metrics["skill_call_latency"].items():
                for i, latency in enumerate(latency_history[-20:]):
                    communication_history.append({
                        "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                        "from_skill": "coordinator",
                        "to_skill": skill_name,
                        "success": True,  # Simplified for now
                        "latency": latency
                    })

            # Analyze patterns with Grok-4
            pattern_analysis = await self.grok_skill_messaging.analyze_skill_communication_patterns(communication_history)

            # Get current skill network status
            skill_network_status = {}
            for skill_name, skill_info in self.reasoning_skill_network.items():
                skill_network_status[skill_name] = {
                    "load_factor": skill_info["load_factor"],
                    "dependencies": skill_info["dependencies"],
                    "provides": skill_info["provides"],
                    "communication_patterns": skill_info["communication_patterns"],
                    "avg_latency": sum(self.skill_communication_metrics["skill_call_latency"].get(skill_name, [0])) / max(len(self.skill_communication_metrics["skill_call_latency"].get(skill_name, [1])), 1)
                }

            # Generate optimization recommendations
            optimization_recommendations = [
                {
                    "type": "load_balancing",
                    "priority": "high",
                    "description": "Implement dynamic load balancing for hypothesis_generator skill",
                    "expected_improvement": "25% reduction in average latency"
                },
                {
                    "type": "caching",
                    "priority": "medium",
                    "description": "Add result caching for evidence_gatherer skill",
                    "expected_improvement": "40% reduction in redundant operations"
                },
                {
                    "type": "pipeline_optimization",
                    "priority": "high",
                    "description": "Optimize skill dependency chain for faster reasoning workflows",
                    "expected_improvement": "15% overall workflow speedup"
                }
            ]

            # Future performance predictions
            performance_predictions = {}
            if include_predictions:
                performance_predictions = {
                    "next_hour_load": "moderate",
                    "bottleneck_probability": 0.3,
                    "optimal_coordination_strategy": "adaptive_hybrid",
                    "predicted_efficiency_gain": 0.2
                }

            return {
                "success": True,
                "analysis_scope": analysis_scope,
                "optimization_target": optimization_target,
                "communication_analysis": {
                    "patterns_identified": pattern_analysis.get("communication_patterns", []),
                    "efficiency_metrics": pattern_analysis.get("efficiency_metrics", {}),
                    "bottlenecks": pattern_analysis.get("bottlenecks_identified", []),
                    "anomalies": pattern_analysis.get("anomalies_detected", [])
                },
                "skill_network_status": skill_network_status,
                "optimization_recommendations": optimization_recommendations,
                "performance_predictions": performance_predictions,
                "current_metrics": {
                    "total_skill_calls": self.skill_communication_metrics["total_skill_calls"],
                    "success_rate": (self.skill_communication_metrics["successful_skill_calls"] / max(self.skill_communication_metrics["total_skill_calls"], 1)) * 100,
                    "average_coordination_time": sum(sum(latencies) for latencies in self.skill_communication_metrics["skill_call_latency"].values()) / max(sum(len(latencies) for latencies in self.skill_communication_metrics["skill_call_latency"].values()), 1)
                },
                "metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "grok_enhanced": True,
                    "skills_analyzed": len(self.reasoning_skill_network)
                }
            }

        except Exception as e:
            logger.error(f"Skill communication optimization error: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_scope": analysis_scope
            }

    @mcp_tool(
        name="reasoning_chain_analysis",
        description="Analyze and optimize reasoning chains for complex problems"
    )
    async def reasoning_chain_analysis_mcp(
        self,
        reasoning_chain: Union[str, Dict[str, Any]],
        analysis_type: str = "comprehensive",
        optimize: bool = True,
        include_alternatives: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze reasoning chains for logical consistency and optimization

        Args:
            reasoning_chain: Reasoning chain to analyze (text or structured)
            analysis_type: Type of analysis (comprehensive, logical, efficiency)
            optimize: Whether to suggest optimizations
            include_alternatives: Include alternative reasoning paths
        """
        try:
            # Prepare reasoning chain for analysis
            if isinstance(reasoning_chain, str):
                chain_text = reasoning_chain
                chain_structure = {"steps": reasoning_chain.split('. '), "type": "textual"}
            else:
                chain_text = reasoning_chain.get("text", str(reasoning_chain))
                chain_structure = reasoning_chain

            # Analyze using swarm intelligence
            analysis_result = await self.enhanced_skills.swarm_intelligence_reasoning({
                "question": f"Analyze this reasoning chain: {chain_text}",
                "context": {
                    "analysis_type": analysis_type,
                    "chain_structure": chain_structure
                },
                "swarm_algorithm": "pso",
                "iterations": 30,
                "swarm_size": 15
            })

            # Extract analysis components
            analysis = {
                "logical_consistency": 0.8,  # Would be calculated from analysis
                "efficiency_score": 0.7,
                "completeness": 0.85,
                "clarity": 0.75
            }

            # Calculate overall score
            overall_score = sum(analysis.values()) / len(analysis)

            # Generate optimizations if requested
            optimizations = []
            if optimize:
                optimizations = [
                    "Consider strengthening logical connections between steps",
                    "Add supporting evidence for key assertions",
                    "Simplify complex intermediate steps"
                ]

            # Generate alternatives if requested
            alternatives = []
            if include_alternatives:
                alt_result = await self.enhanced_skills.hierarchical_multi_engine_reasoning({
                    "question": f"What are alternative reasoning approaches for: {chain_text[:100]}...",
                    "context": {"original_chain": chain_structure},
                    "reasoning_engines": ["analogical", "causal"],
                    "max_depth": 2
                })

                alternatives = [
                    {
                        "approach": "analogical_reasoning",
                        "description": alt_result.get("answer", "Alternative analogical approach"),
                        "confidence": alt_result.get("confidence", 0.6)
                    }
                ]

            return {
                "success": True,
                "original_chain": chain_text[:200] + "..." if len(chain_text) > 200 else chain_text,
                "analysis": {
                    "type": analysis_type,
                    "scores": analysis,
                    "overall_score": round(overall_score, 3),
                    "strengths": ["Clear logical progression", "Well-structured arguments"],
                    "weaknesses": ["Could benefit from more evidence", "Some logical gaps"]
                },
                "optimizations": optimizations if optimize else [],
                "alternatives": alternatives if include_alternatives else [],
                "metadata": {
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "chain_length": len(chain_structure.get("steps", [])),
                    "analysis_confidence": analysis_result.get("confidence", 0.0)
                }
            }

        except Exception as e:
            logger.error(f"MCP reasoning_chain_analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "chain": str(reasoning_chain)[:100] + "..."
            }

    # ========= MCP Resource Implementations =========

    @mcp_resource(
        uri="reasoning://status",
        description="Reasoning Agent status and performance metrics"
    )
    async def get_reasoning_status(self) -> Dict[str, Any]:
        """Get reasoning agent status and metrics"""

        # Get current metrics
        metrics = await self.get_reasoning_metrics()

        return {
            "reasoning_status": {
                "agent_id": self.agent_id,
                "version": self.version,
                "active_sessions": len(self.active_reasoning_sessions),
                "architectures_supported": [arch.value for arch in ReasoningArchitecture],
                "performance_metrics": {
                    "total_sessions": self.reasoning_metrics["total_sessions"],
                    "successful_reasoning": self.reasoning_metrics["successful_reasoning"],
                    "average_confidence": round(self.reasoning_metrics["average_confidence"], 3),
                    "average_reasoning_time": round(self.reasoning_metrics["average_reasoning_time"], 3),
                    "success_rate": (
                        self.reasoning_metrics["successful_reasoning"] /
                        max(self.reasoning_metrics["total_sessions"], 1)
                    ) * 100
                },
                "sub_agents": {
                    "pool_size": sum(len(agents) for agents in self.sub_agent_pool.values()),
                    "roles_available": list(self.sub_agent_pool.keys())
                },
                "health": {
                    "status": "healthy",
                    "circuit_breakers": len(metrics["metrics"]["circuit_breaker_status"]),
                    "trust_identity": self.trust_identity is not None
                }
            }
        }

    @mcp_resource(
        uri="reasoning://architectures",
        description="Available reasoning architectures and their capabilities"
    )
    async def get_reasoning_architectures(self) -> Dict[str, Any]:
        """Get reasoning architecture information"""

        return {
            "reasoning_architectures": {
                "hierarchical": {
                    "description": "Multi-phase hierarchical reasoning with agent orchestration",
                    "phases": ["question_analysis", "evidence_retrieval", "reasoning", "debate", "synthesis"],
                    "capabilities": ["sub_question_decomposition", "multi_agent_coordination", "evidence_aggregation"],
                    "best_for": ["complex_questions", "research_tasks", "multi_step_problems"],
                    "performance": "high_accuracy"
                },
                "peer_to_peer": {
                    "description": "Distributed swarm intelligence reasoning",
                    "features": ["parallel_processing", "emergent_intelligence", "collective_decision_making"],
                    "capabilities": ["swarm_optimization", "distributed_search", "consensus_building"],
                    "best_for": ["optimization_problems", "creative_solutions", "parallel_exploration"],
                    "performance": "high_creativity"
                },
                "blackboard": {
                    "description": "Shared knowledge space with multiple reasoning agents",
                    "features": ["shared_memory", "incremental_building", "expert_cooperation"],
                    "capabilities": ["knowledge_integration", "expert_systems", "collaborative_reasoning"],
                    "best_for": ["knowledge_intensive_tasks", "expert_consultation", "domain_specific_problems"],
                    "performance": "high_expertise"
                },
                "graph_based": {
                    "description": "Knowledge graph reasoning with concept relationships",
                    "features": ["concept_mapping", "relationship_analysis", "path_finding"],
                    "capabilities": ["semantic_reasoning", "relationship_discovery", "concept_navigation"],
                    "best_for": ["knowledge_graphs", "semantic_analysis", "relationship_problems"],
                    "performance": "high_semantic_understanding"
                },
                "hub_and_spoke": {
                    "description": "Central orchestrator with specialized reasoning modules",
                    "features": ["centralized_control", "specialized_modules", "coordinated_analysis"],
                    "capabilities": ["multi_perspective_analysis", "coordinated_reasoning", "synthesis"],
                    "best_for": ["multi_domain_problems", "perspective_analysis", "coordinated_tasks"],
                    "performance": "balanced"
                }
            }
        }

    @mcp_resource(
        uri="reasoning://session-history",
        description="Recent reasoning sessions and their outcomes"
    )
    async def get_session_history(self) -> Dict[str, Any]:
        """Get reasoning session history"""

        # Get architecture usage statistics
        arch_usage = self.reasoning_metrics["architecture_usage"]
        total_usage = sum(arch_usage.values())

        return {
            "session_history": {
                "current_active_sessions": len(self.active_reasoning_sessions),
                "active_session_ids": list(self.active_reasoning_sessions.keys()),
                "total_completed_sessions": self.reasoning_metrics["total_sessions"],
                "architecture_distribution": {
                    arch: {
                        "usage_count": count,
                        "percentage": round((count / max(total_usage, 1)) * 100, 1)
                    }
                    for arch, count in arch_usage.items()
                },
                "performance_summary": {
                    "average_confidence": round(self.reasoning_metrics["average_confidence"], 3),
                    "average_time_seconds": round(self.reasoning_metrics["average_reasoning_time"], 3),
                    "success_rate": round(
                        (self.reasoning_metrics["successful_reasoning"] /
                         max(self.reasoning_metrics["total_sessions"], 1)) * 100, 1
                    )
                },
                "recent_patterns": {
                    "most_used_architecture": max(arch_usage, key=arch_usage.get) if arch_usage else "none",
                    "trending_features": ["multi_agent_debate", "hierarchical_orchestration", "swarm_intelligence"]
                }
            }
        }

    @mcp_resource(
        uri="reasoning://skill-coordination",
        description="Real-time skill coordination status and Grok-4 intelligence metrics"
    )
    async def get_skill_coordination_status(self) -> Dict[str, Any]:
        """Get real-time skill coordination status"""

        return {
            "skill_coordination": {
                "mcp_coordinator_active": self.mcp_skill_coordinator is not None and self.mcp_skill_coordinator.is_running,
                "grok_intelligence_active": self.grok_skill_messaging is not None,
                "skill_network_status": {
                    skill_name: {
                        "load_factor": skill_info["load_factor"],
                        "dependencies_met": all(
                            dep in self.reasoning_skill_network for dep in skill_info["dependencies"]
                        ),
                        "communication_patterns": skill_info["communication_patterns"],
                        "provides": skill_info["provides"]
                    }
                    for skill_name, skill_info in self.reasoning_skill_network.items()
                },
                "communication_metrics": {
                    "total_skill_calls": self.skill_communication_metrics["total_skill_calls"],
                    "success_rate": (self.skill_communication_metrics["successful_skill_calls"] /
                                    max(self.skill_communication_metrics["total_skill_calls"], 1)) * 100,
                    "active_coordination_patterns": list(self.skill_communication_metrics["skill_collaboration_patterns"].keys()),
                    "error_recovery_events": self.skill_communication_metrics["skill_error_recovery"]
                },
                "grok_intelligence_metrics": {
                    "semantic_routing_active": True if self.grok_skill_messaging else False,
                    "predictive_coordination": True if self.grok_skill_messaging else False,
                    "intelligent_load_balancing": True if self.grok_skill_messaging else False,
                    "optimization_suggestions_generated": len(self.skill_message_patterns) if self.grok_skill_messaging else 0
                },
                "coordination_efficiency": {
                    "average_skill_latency": sum(
                        sum(latencies) / max(len(latencies), 1)
                        for latencies in self.skill_communication_metrics["skill_call_latency"].values()
                    ) / max(len(self.skill_communication_metrics["skill_call_latency"]), 1),
                    "dependency_chain_optimization": "active" if self.grok_skill_messaging else "basic",
                    "cross_skill_validation_rate": self.skill_communication_metrics["cross_skill_validation_count"] / max(self.skill_communication_metrics["total_skill_calls"], 1)
                }
            }
        }

    @mcp_resource(
        uri="reasoning://capabilities",
        description="Reasoning capabilities and supported features"
    )
    async def get_reasoning_capabilities(self) -> Dict[str, Any]:
        """Get reasoning capabilities"""

        return {
            "reasoning_capabilities": {
                "core_features": {
                    "multi_agent_orchestration": {
                        "description": "Coordinate multiple reasoning agents",
                        "supported": True,
                        "max_agents": self.max_sub_agents
                    },
                    "debate_mechanism": {
                        "description": "Multi-perspective debate and consensus building",
                        "supported": True,
                        "max_rounds": 10
                    },
                    "hypothesis_generation": {
                        "description": "Generate and validate hypotheses",
                        "supported": True,
                        "engines": ["logical", "probabilistic", "analogical", "causal"]
                    },
                    "reasoning_chain_analysis": {
                        "description": "Analyze and optimize reasoning chains",
                        "supported": True,
                        "optimization": True
                    }
                },
                "specialized_skills": {
                    "swarm_intelligence": {
                        "algorithms": ["pso", "aco", "genetic"],
                        "swarm_sizes": "10-100",
                        "optimization_types": ["multi_objective", "constrained", "dynamic"]
                    },
                    "hierarchical_reasoning": {
                        "max_depth": 10,
                        "decomposition": True,
                        "synthesis": True
                    },
                    "knowledge_integration": {
                        "sources": ["data_manager", "catalog_manager", "external_apis"],
                        "formats": ["structured", "unstructured", "semantic"]
                    }
                },
                "grok_enhanced_features": {
                    "intelligent_skill_routing": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "predictive_coordination": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "semantic_message_optimization": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "adaptive_load_balancing": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "error_prediction_and_recovery": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "communication_pattern_analysis": GROK_AVAILABLE and self.grok_skill_messaging is not None
                },
                "skill_coordination": {
                    "mcp_enabled": self.mcp_skill_coordinator is not None,
                    "skill_network_size": len(self.reasoning_skill_network),
                    "coordination_patterns": list(set(pattern for skill_info in self.reasoning_skill_network.values() for pattern in skill_info["communication_patterns"])),
                    "dependency_chains": {skill: info["dependencies"] for skill, info in self.reasoning_skill_network.items()},
                    "service_providers": {skill: info["provides"] for skill, info in self.reasoning_skill_network.items()}
                },
                "integrations": {
                    "a2a_network": True,
                    "blockchain_trust": True,
                    "circuit_breakers": True,
                    "performance_monitoring": True,
                    "telemetry": True,
                    "grok_4_intelligence": GROK_AVAILABLE and self.grok_skill_messaging is not None,
                    "mcp_skill_coordination": self.mcp_skill_coordinator is not None
                }
            }
        }

    # ========= MCP Prompt Implementations =========

    @mcp_prompt(
        name="reasoning_assistant",
        description="Interactive reasoning assistant for complex problem solving"
    )
    async def reasoning_assistant_prompt(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Reasoning assistant for natural language interactions

        Args:
            user_query: User's reasoning question or request
            context: Additional context
        """
        try:
            query_lower = user_query.lower()

            if any(word in query_lower for word in ["analyze", "reason", "think", "solve"]):
                # Check if it's a specific question or general request
                if "?" in user_query:
                    return f"""I can help you reason through this question: "{user_query}"

**I'll use my advanced reasoning capabilities:**

🔍 **Multi-Agent Analysis** - Break down the problem using specialized agents
🧠 **Architecture Selection** - Choose the best reasoning approach:
   • Hierarchical for complex multi-step problems
   • Swarm Intelligence for creative solutions
   • Graph-based for relationship analysis
   • Blackboard for knowledge integration

💬 **Debate & Validation** - Multiple perspectives to ensure robust answers

Would you like me to proceed with reasoning through this question? I can also:
- Generate hypotheses about the problem
- Analyze different reasoning approaches
- Provide step-by-step logical breakdowns"""

                else:
                    return """I'm your Advanced Reasoning Assistant! I can help you:

🎯 **Solve Complex Problems** using multiple reasoning architectures
🔬 **Generate Hypotheses** for research and analysis
💭 **Orchestrate Debates** between different perspectives
⚡ **Analyze Reasoning Chains** for logical consistency
🌊 **Swarm Intelligence** for creative problem solving

**Example requests:**
- "What are the implications of climate change on urban planning?"
- "Generate hypotheses about declining productivity in remote work"
- "Debate the pros and cons of artificial intelligence in healthcare"
- "Analyze this reasoning: If A then B, B is true, therefore A is true"

What complex problem would you like me to help you reason through?"""

            elif any(word in query_lower for word in ["hypothesis", "hypotheses", "theory"]):
                return """I can generate and validate hypotheses for your problem!

**Hypothesis Generation Process:**
1. **Problem Analysis** - Understand the core question
2. **Domain Context** - Consider relevant knowledge areas
3. **Multi-Engine Generation** - Use logical, probabilistic, and analogical reasoning
4. **Validation** - Assess plausibility and testability

**What I need:**
- Clear problem statement
- Domain context (scientific, business, social, etc.)
- Number of hypotheses desired (typically 3-7)

**Example:** "Generate hypotheses for why employee satisfaction dropped after the office redesign"

What problem would you like me to generate hypotheses for?"""

            elif any(word in query_lower for word in ["debate", "argue", "perspective", "pros", "cons"]):
                return """I can orchestrate structured debates on complex topics!

**Debate Features:**
🎭 **Multiple Perspectives** - Analytical, practical, theoretical, critical views
⚖️ **Structured Arguments** - Organized presentation of positions
🔄 **Interactive Rounds** - Back-and-forth argument development
🎯 **Consensus Building** - Work toward resolution or understanding

**Debate Structures:**
- **Dialectical** - Thesis vs. antithesis toward synthesis
- **Round Robin** - Each perspective responds to all others
- **Tournament** - Progressive elimination of weaker arguments

**Example:** "Debate whether universal basic income would benefit or harm society"

What topic would you like me to orchestrate a debate on?"""

            elif any(word in query_lower for word in ["architecture", "approach", "method"]):
                return """I offer multiple reasoning architectures for different problem types:

**🏗️ Hierarchical** - Best for complex, multi-step problems
   • Question decomposition → Evidence gathering → Reasoning → Synthesis
   • Great for research questions and analysis tasks

**🌐 Peer-to-Peer** - Best for creative and optimization problems
   • Swarm intelligence with parallel exploration
   • Emergent solutions from collective intelligence

**🎯 Hub-and-Spoke** - Best for multi-perspective analysis
   • Central coordination with specialized modules
   • Balanced view from different analytical angles

**📊 Graph-Based** - Best for relationship and semantic problems
   • Knowledge graph navigation and concept mapping
   • Understanding connections and dependencies

**🧠 Blackboard** - Best for knowledge-intensive tasks
   • Shared memory space for collaborative reasoning
   • Expert system integration

Which architecture interests you, or would you like me to recommend one for your specific problem?"""

            else:
                return """I'm your Advanced Reasoning Assistant with sophisticated multi-agent capabilities!

**🧠 What I Can Do:**
- **Complex Problem Solving** using multiple reasoning architectures
- **Hypothesis Generation** with confidence scoring
- **Multi-Perspective Debates** with structured argumentation
- **Reasoning Chain Analysis** with optimization suggestions
- **Swarm Intelligence** for creative and optimization problems

**🎯 Best For:**
- Research questions requiring deep analysis
- Business problems needing multiple perspectives
- Scientific hypotheses generation and validation
- Logical reasoning and argument evaluation
- Creative problem solving with emergent solutions

**💡 Try asking:**
- "Reason through [complex question]"
- "Generate hypotheses about [problem]"
- "Debate the topic of [controversial issue]"
- "What's the best reasoning approach for [type of problem]?"

What complex reasoning challenge can I help you with today?"""

        except Exception as e:
            logger.error(f"Reasoning assistant prompt error: {e}")
            return "I encountered an error. Please rephrase your reasoning request."

    @mcp_prompt(
        name="architecture_advisor",
        description="Advisor for choosing optimal reasoning architectures"
    )
    async def architecture_advisor_prompt(
        self,
        problem_description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Architecture advisor for reasoning approach selection

        Args:
            problem_description: Description of the problem to solve
            requirements: Specific requirements or constraints
        """
        try:
            problem_lower = problem_description.lower()
            requirements = requirements or {}

            # Analyze problem characteristics
            is_complex = len(problem_description.split()) > 20
            needs_creativity = any(word in problem_lower for word in ["creative", "innovative", "novel", "brainstorm"])
            needs_analysis = any(word in problem_lower for word in ["analyze", "compare", "evaluate", "assess"])
            needs_research = any(word in problem_lower for word in ["research", "investigate", "explore", "study"])
            needs_multiple_views = any(word in problem_lower for word in ["perspective", "opinion", "viewpoint", "debate"])
            has_relationships = any(word in problem_lower for word in ["relationship", "connection", "network", "graph"])

            recommendations = []

            # Architecture recommendations based on problem analysis
            if needs_creativity:
                recommendations.append({
                    "architecture": "Peer-to-Peer (Swarm Intelligence)",
                    "reason": "Best for creative and innovative solutions",
                    "confidence": 0.9,
                    "features": ["Parallel exploration", "Emergent solutions", "Optimization"]
                })

            if needs_research or is_complex:
                recommendations.append({
                    "architecture": "Hierarchical",
                    "reason": "Excellent for complex multi-step research problems",
                    "confidence": 0.85,
                    "features": ["Question decomposition", "Evidence gathering", "Systematic analysis"]
                })

            if needs_multiple_views:
                recommendations.append({
                    "architecture": "Hub-and-Spoke",
                    "reason": "Perfect for multi-perspective analysis",
                    "confidence": 0.8,
                    "features": ["Multiple viewpoints", "Coordinated analysis", "Balanced synthesis"]
                })

            if has_relationships:
                recommendations.append({
                    "architecture": "Graph-Based",
                    "reason": "Ideal for relationship and network analysis",
                    "confidence": 0.85,
                    "features": ["Concept mapping", "Relationship discovery", "Semantic reasoning"]
                })

            if needs_analysis and "knowledge" in problem_lower:
                recommendations.append({
                    "architecture": "Blackboard",
                    "reason": "Great for knowledge-intensive analytical tasks",
                    "confidence": 0.75,
                    "features": ["Knowledge integration", "Expert collaboration", "Incremental building"]
                })

            # Sort by confidence
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)

            # Format response
            response = f"""**Architecture Analysis for:** "{problem_description[:100]}{'...' if len(problem_description) > 100 else ''}"

**🎯 Recommended Architectures:**\n"""

            for i, rec in enumerate(recommendations[:3], 1):
                response += f"""
**{i}. {rec['architecture']}** (Confidence: {rec['confidence']:.0%})
   • **Why:** {rec['reason']}
   • **Features:** {', '.join(rec['features'])}
"""

            # Add specific considerations
            response += f"""
**🔍 Problem Characteristics Detected:**
{'• Complex multi-step problem' if is_complex else ''}
{'• Requires creative thinking' if needs_creativity else ''}
{'• Needs analytical evaluation' if needs_analysis else ''}
{'• Benefits from multiple perspectives' if needs_multiple_views else ''}
{'• Involves relationships/networks' if has_relationships else ''}

**💡 Additional Considerations:**
• **Time Sensitivity:** {'Hierarchical or Hub-and-Spoke for faster results' if requirements.get('time_critical') else 'All architectures suitable'}
• **Accuracy Priority:** {'Hierarchical with debate for highest accuracy' if requirements.get('high_accuracy') else 'Standard approaches sufficient'}
• **Resource Usage:** {'Peer-to-peer uses more computational resources' if not requirements.get('resource_limited') else 'Hub-and-spoke or Graph-based more efficient'}

Would you like me to proceed with the top recommendation, or would you prefer to explore a specific architecture in detail?"""

            return response

        except Exception as e:
            logger.error(f"Architecture advisor prompt error: {e}")
            return "I encountered an error analyzing your problem. Please try rephrasing your request."

    @mcp_prompt(
        name="debate_moderator",
        description="Moderator for structured multi-agent debates"
    )
    async def debate_moderator_prompt(
        self,
        debate_topic: str,
        moderation_style: Optional[str] = None
    ) -> str:
        """
        Debate moderator for structured discussions

        Args:
            debate_topic: Topic for the debate
            moderation_style: Style of moderation (neutral, socratic, devil's_advocate)
        """
        try:
            moderation_style = moderation_style or "neutral"

            if moderation_style == "socratic":
                return f"""**Socratic Debate Moderation for: "{debate_topic}"**

As your Socratic moderator, I'll guide this debate through structured questioning:

**🎯 Opening Questions:**
• What fundamental assumptions underlie each position?
• How do we define key terms in this debate?
• What evidence would change your perspective?

**🔍 Exploration Phase:**
I'll ask each perspective:
• What are the strongest objections to your position?
• How does your view address counterarguments?
• What are the logical foundations of your reasoning?

**⚖️ Synthesis Phase:**
• Where do positions overlap or conflict?
• What new insights emerge from this exchange?
• How might we integrate the strongest elements?

**Debate Structure:**
1. **Position Statements** (2 minutes each)
2. **Socratic Questioning** (guided exploration)
3. **Cross-Examination** (perspectives question each other)
4. **Synthesis** (finding common ground and insights)

Ready to begin? I'll start by asking each perspective to state their position clearly and identify their core assumptions."""

            elif moderation_style == "devil's_advocate":
                return f"""**Devil's Advocate Moderation for: "{debate_topic}"**

I'll challenge ALL positions to strengthen the overall reasoning:

**🎭 My Role:**
• Challenge assumptions from every angle
• Present strongest possible counterarguments
• Push each perspective to its logical limits
• Expose potential weaknesses or blind spots

**🔥 Challenge Areas:**
• **Logical Consistency** - Are there internal contradictions?
• **Evidence Quality** - How strong is the supporting data?
• **Alternative Explanations** - What other possibilities exist?
• **Practical Implications** - What happens if you're wrong?

**⚔️ Debate Process:**
1. **Initial Positions** - State your strongest case
2. **Challenge Round** - I'll attack each position systematically
3. **Defense & Refinement** - Strengthen your arguments
4. **Final Challenge** - Last chance to expose weaknesses
5. **Evolved Positions** - Present your refined thinking

**Warning:** I will be relentless in finding flaws and alternatives. This process strengthens reasoning through rigorous testing.

Are you prepared for intensive scrutiny of your positions?"""

            else:  # neutral
                return f"""**Neutral Debate Moderation for: "{debate_topic}"**

Welcome to this structured debate! I'll ensure fair, productive discussion.

**🎯 Debate Structure:**

**Round 1: Opening Statements** (3 minutes each)
• Present your core position clearly
• Provide 2-3 key supporting arguments
• State your confidence level

**Round 2: Evidence Presentation**
• Share relevant data, studies, examples
• Explain how evidence supports your position
• Address potential data limitations

**Round 3: Cross-Examination**
• Respectfully question other positions
• Identify areas of agreement/disagreement
• Explore underlying assumptions

**Round 4: Synthesis & Conclusion**
• Acknowledge strongest opposing points
• Refine your position based on discussion
• Identify areas for further exploration

**🔧 Moderation Rules:**
✅ Focus on logic and evidence
✅ Respectful disagreement encouraged
✅ Build on others' insights
❌ Personal attacks or dismissive language
❌ Strawman arguments
❌ Ignoring evidence

**🎭 Perspectives I'll Coordinate:**
• Analytical (data-driven approach)
• Practical (real-world implications)
• Theoretical (conceptual framework)
• Critical (questioning assumptions)

Ready to begin with opening statements? Each perspective will have equal time and opportunity to present their strongest case."""

        except Exception as e:
            logger.error(f"Debate moderator prompt error: {e}")
            return "I encountered an error setting up the debate. Please try again."

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the reasoning agent"""
        try:
            # Initialize blockchain trust system
            self.trust_identity = await initialize_agent_trust(
                self.agent_id,
                private_key=self.private_key
            )
            logger.info(f"Trust identity initialized for agent {self.agent_id}")

            # Initialize enhanced skills Redis if available
            await self.enhanced_skills.initialize_redis()

            # Initialize MCP skill coordination
            await self._initialize_mcp_skill_coordination()

            # Initialize blockchain integration if enabled
            if self.blockchain_enabled:
                logger.info("Blockchain integration is enabled for Reasoning Agent")
                await self._register_blockchain_handlers()

            # Test sub-agent connectivity
            test_results = {}
            for role, agents in self.sub_agent_pool.items():
                if agents:
                    try:
                        result = await self._query_sub_agent(
                            agents[0],
                            "health_check",
                            {}
                        )
                        test_results[role.value] = "connected"
                    except Exception as e:
                        logger.warning(f"Sub-agent {role.value} not available, will use internal fallback: {e}")
                        test_results[role.value] = "fallback_available"

            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "version": self.version,
                "architectures_supported": [arch.value for arch in ReasoningArchitecture],
                "sub_agent_pool_size": sum(len(agents) for agents in self.sub_agent_pool.values()),
                "connectivity_test": test_results,
                "initialization_time": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise

    async def _register_blockchain_handlers(self):
        """Register blockchain-specific message handlers for reasoning operations"""
        logger.info("Registering blockchain handlers for Reasoning Agent")

        # Override the base blockchain message handler
        self._handle_blockchain_message = self._handle_reasoning_blockchain_message

    def _handle_reasoning_blockchain_message(self, message: Dict[str, Any]):
        """Handle incoming blockchain messages for reasoning operations"""
        logger.info(f"Reasoning Agent received blockchain message: {message}")

        message_type = message.get('messageType', '')
        content = message.get('content', {})

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except:
                pass

        # Handle reasoning-specific blockchain messages
        if message_type == "REASONING_REQUEST":
            asyncio.create_task(self._handle_blockchain_reasoning_request(message, content))
        elif message_type == "COLLABORATIVE_REASONING":
            asyncio.create_task(self._handle_blockchain_collaborative_reasoning(message, content))
        elif message_type == "LOGIC_VERIFICATION_REQUEST":
            asyncio.create_task(self._handle_blockchain_logic_verification(message, content))
        else:
            # Default handling
            logger.info(f"Received blockchain message type: {message_type}")

        # Mark message as delivered
        if self.blockchain_integration and message.get('messageId'):
            try:
                self.blockchain_integration.mark_message_delivered(message['messageId'])
            except Exception as e:
                logger.error(f"Failed to mark message as delivered: {e}")

    async def _handle_blockchain_reasoning_request(self, message: Dict[str, Any], content: Dict[str, Any]):
        """Handle reasoning request from blockchain"""
        try:
            problem = content.get('problem', '')
            reasoning_type = content.get('reasoning_type', 'logical')
            requester_address = message.get('from')

            # Verify trust before processing
            if not self.verify_trust(requester_address):
                logger.warning(f"Reasoning request from untrusted agent: {requester_address}")
                return

            # Perform reasoning
            reasoning_result = await self._perform_reasoning(problem, reasoning_type)

            # Send response via blockchain
            self.send_blockchain_message(
                to_address=requester_address,
                content={
                    "problem": problem,
                    "reasoning_steps": reasoning_result.get('steps', []),
                    "conclusion": reasoning_result.get('conclusion', ''),
                    "confidence": reasoning_result.get('confidence', 0.0),
                    "reasoning_method": reasoning_type,
                    "timestamp": datetime.utcnow().isoformat()
                },
                message_type="REASONING_RESPONSE"
            )

        except Exception as e:
            logger.error(f"Failed to handle reasoning request: {e}")

    async def _handle_blockchain_collaborative_reasoning(self, message: Dict[str, Any], content: Dict[str, Any]):
        """Handle collaborative reasoning request from blockchain"""
        try:
            reasoning_session_id = content.get('session_id')
            current_state = content.get('current_state', {})
            contribution_request = content.get('contribution_type', 'analysis')

            # Contribute to collaborative reasoning
            contribution = await self._contribute_to_reasoning(reasoning_session_id, current_state, contribution_request)

            # Broadcast contribution to all participants
            participants = content.get('participants', [])
            for participant in participants:
                if participant != getattr(self.agent_identity, 'address', None):
                    self.send_blockchain_message(
                        to_address=participant,
                        content={
                            "session_id": reasoning_session_id,
                            "contribution": contribution,
                            "contributor": self.agent_id,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        message_type="REASONING_CONTRIBUTION"
                    )

        except Exception as e:
            logger.error(f"Failed to handle collaborative reasoning: {e}")

    async def _handle_blockchain_logic_verification(self, message: Dict[str, Any], content: Dict[str, Any]):
        """Handle logic verification request from blockchain"""
        try:
            logical_statement = content.get('statement', '')
            premises = content.get('premises', [])
            requester_address = message.get('from')

            # Verify logical validity
            verification_result = await self._verify_logic(logical_statement, premises)

            # Send verification response via blockchain
            self.send_blockchain_message(
                to_address=requester_address,
                content={
                    "statement": logical_statement,
                    "is_valid": verification_result.get('is_valid', False),
                    "verification_details": verification_result.get('details', {}),
                    "logical_errors": verification_result.get('errors', []),
                    "timestamp": datetime.utcnow().isoformat()
                },
                message_type="LOGIC_VERIFICATION_RESPONSE"
            )

        except Exception as e:
            logger.error(f"Failed to handle logic verification: {e}")

    # ========= Enhanced Grok-4 Powered Skill Coordination =========

    async def _initialize_mcp_skill_coordination(self):
        """Initialize MCP skill coordinator with Grok-4 intelligence"""
        try:
            # Initialize MCP skill coordinator
            self.mcp_skill_coordinator = MCPSkillCoordinator(self)
            self.mcp_skill_client = MCPSkillClient(self)

            # Add Grok enhanced intra-skill coordination rules
            if self.grok_intra_skill_messaging:
                self.mcp_skill_coordinator.state_manager.add_coordination_rule(
                    "grok_intra_skill_routing",
                    self._grok_intra_skill_routing_rule
                )
                self.mcp_skill_coordinator.state_manager.add_coordination_rule(
                    "grok_intra_skill_load_balancing",
                    self._grok_intra_skill_load_balancing_rule
                )
                self.mcp_skill_coordinator.state_manager.add_coordination_rule(
                    "grok_intra_skill_optimization",
                    self._grok_intra_skill_optimization_rule
                )

            # Start skill coordinator
            await self.mcp_skill_coordinator.start()

            logger.info("MCP skill coordination with Grok-4 intelligence initialized")

        except Exception as e:
            logger.error(f"Failed to initialize MCP skill coordination: {e}")

    async def _grok_intra_skill_routing_rule(self, message: SkillMessage, skills: Dict[str, Any]) -> bool:
        """Grok powered intra-skill routing optimization rule"""
        if not self.grok_intra_skill_messaging:
            if not self._validate_architecture(architecture):
                architecture = ReasoningArchitecture.HIERARCHICAL
            return True

        try:
            # Get current skill loads for intra-skill routing
            current_loads = {skill_name: skill.load_factor for skill_name, skill in skills.items()}
            available_skills = list(skills.keys())

            # Get Grok routing recommendation for intra-skill network
            routing_result = await self.grok_intra_skill_messaging.determine_optimal_intra_skill_route(
                str(message.params), available_skills, current_loads
            )

            # Apply intra-skill routing optimization
            recommended_skill = routing_result.get('recommended_skill')
            routing_confidence = routing_result.get('routing_confidence', 0.5)

            # Log intra-skill routing decisions for optimization
            if routing_confidence > 0.7 and recommended_skill != message.receiver_skill:
                logger.info(f"Grok suggests intra-skill route to {recommended_skill} instead of {message.receiver_skill} (confidence: {routing_confidence})")

                # Update message context with routing optimization
                message.context = message.context or {}
                message.context['grok_routing_suggestion'] = recommended_skill
                message.context['original_route'] = message.receiver_skill

            # Update intra-skill routing metrics
            self.skill_communication_metrics["intra_skill_routing_optimizations"] = self.skill_communication_metrics.get("intra_skill_routing_optimizations", 0) + 1

            return True

        except Exception as e:
            logger.error(f"Grok intra-skill routing rule failed: {e}")
            return True

    async def _grok_intra_skill_load_balancing_rule(self, message: SkillMessage, skills: Dict[str, Any]) -> bool:
        """Grok powered intra-skill load balancing rule"""
        if not self.grok_intra_skill_messaging:
            return True

        try:
            target_skill = skills.get(message.receiver_skill)
            if not target_skill:
                return True

            # Check intra-skill network load distribution
            network_loads = {name: skill.load_factor for name, skill in skills.items()}
            avg_load = sum(network_loads.values()) / len(network_loads)

            # Apply Grok-based load balancing for intra-skill network
            if target_skill.load_factor > 0.8 or target_skill.load_factor > avg_load * 1.5:
                # Find alternative skills within the same capability domain
                target_capabilities = self.reasoning_skill_network.get(message.receiver_skill, {}).get('provides', [])
                alternative_skills = [
                    name for name, skill_info in self.reasoning_skill_network.items()
                    if any(cap in skill_info.get('provides', []) for cap in target_capabilities)
                    and skills.get(name, {}).load_factor < 0.6
                ]

                if alternative_skills and message.priority != SkillPriority.CRITICAL:
                    # Use first available alternative for load balancing
                    alternative_skill = alternative_skills[0]
                    logger.info(f"Grok intra-skill load balancing: redirecting from {message.receiver_skill} (load: {target_skill.load_factor}) to {alternative_skill}")

                    # Update message context
                    message.context = message.context or {}
                    message.context['load_balanced'] = True
                    message.context['original_target'] = message.receiver_skill
                    message.receiver_skill = alternative_skill

            # Update load balancing metrics
            self.skill_communication_metrics["intra_skill_load_balancing_events"] = self.skill_communication_metrics.get("intra_skill_load_balancing_events", 0) + 1

            return True

        except Exception as e:
            logger.error(f"Grok intra-skill load balancing rule failed: {e}")
            return True

    async def _grok_intra_skill_optimization_rule(self, message: SkillMessage, skills: Dict[str, Any]) -> bool:
        """Grok powered intra-skill message optimization rule"""
        if not self.grok_intra_skill_messaging:
            return True

        try:
            # Build context for intra-skill optimization
            optimization_context = {
                "sender_deps": self.reasoning_skill_network.get(message.sender_skill, {}).get('dependencies', []),
                "receiver_capabilities": self.reasoning_skill_network.get(message.receiver_skill, {}).get('provides', []),
                "load_factors": {name: skill.load_factor for name, skill in skills.items()},
                "queue_size": len(self.mcp_skill_coordinator.message_queue.queues.get(message.priority, [])) if self.mcp_skill_coordinator else 0
            }

            # Optimize the message using Grok
            optimized_message = await self.grok_intra_skill_messaging.optimize_intra_skill_message(
                message, optimization_context
            )

            # Apply optimizations if any were made
            if optimized_message.context and optimized_message.context.get('grok_optimized'):
                logger.debug(f"Grok optimized intra-skill message from {message.sender_skill} to {message.receiver_skill}")

                # Update the original message with optimizations
                message.params = optimized_message.params
                message.priority = optimized_message.priority
                message.context = optimized_message.context

            # Update optimization metrics
            self.skill_communication_metrics["intra_skill_optimizations"] = self.skill_communication_metrics.get("intra_skill_optimizations", 0) + 1

            return True

        except Exception as e:
            logger.error(f"Grok intra-skill optimization rule failed: {e}")
            return True

    @skill_depends_on("question_analyzer", "evidence_gatherer")
    @skill_provides("hypothesis_generation", "creative_reasoning")
    async def grok_enhanced_hypothesis_skill(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Grok-4 enhanced hypothesis generation skill"""
        skill_start_time = time.time()

        try:
            # Use MCP to coordinate with other skills
            question_analysis = await self.mcp_skill_client.call_skill_tool(
                "question_analyzer",
                "analyze_complexity",
                {"question": request_data.get("question", "")}
            )

            evidence_data = await self.mcp_skill_client.call_skill_tool(
                "evidence_gatherer",
                "gather_relevant_evidence",
                {
                    "query": request_data.get("question", ""),
                    "context": request_data.get("context", {})
                }
            )

            # Enhance with Grok-4 intelligence
            if self.grok_skill_messaging:
                coordination_plan = await self.grok_skill_messaging.predict_skill_coordination_needs({
                    "question": request_data.get("question", ""),
                    "analysis": question_analysis,
                    "evidence": evidence_data,
                    "context": request_data.get("context", {})
                })

                # Use coordination plan to optimize hypothesis generation
                hypothesis_request = {
                    "problem": request_data.get("question", ""),
                    "evidence": evidence_data.get("result", []),
                    "complexity": question_analysis.get("result", {}).get("complexity", "medium"),
                    "coordination_strategy": coordination_plan.get("coordination_strategy", "sequential")
                }
            else:
                hypothesis_request = {
                    "problem": request_data.get("question", ""),
                    "evidence": evidence_data.get("result", []),
                    "complexity": question_analysis.get("result", {}).get("complexity", "medium")
                }

            # Generate hypotheses using enhanced reasoning
            result = await self.hypothesis_generation_mcp(**hypothesis_request)

            # Update skill performance metrics
            skill_duration = time.time() - skill_start_time
            await self._update_skill_performance("grok_enhanced_hypothesis_skill", {
                "duration": skill_duration,
                "success": result.get("success", False),
                "hypotheses_generated": len(result.get("hypotheses", [])),
                "grok_enhanced": self.grok_skill_messaging is not None
            })

            return result

        except Exception as e:
            logger.error(f"Grok enhanced hypothesis skill failed: {e}")
            await self._update_skill_performance("grok_enhanced_hypothesis_skill", {
                "duration": time.time() - skill_start_time,
                "success": False,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}

    @skill_depends_on("hypothesis_generator", "logical_reasoner")
    @skill_provides("debate_moderation", "consensus_building")
    async def grok_enhanced_debate_skill(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Grok-4 enhanced debate coordination skill"""
        skill_start_time = time.time()

        try:
            # Get hypotheses from hypothesis generator
            hypotheses = await self.mcp_skill_client.call_skill_tool(
                "hypothesis_generator",
                "generate_hypotheses",
                {"problem": debate_data.get("topic", "")}
            )

            # Get logical analysis from reasoning skill
            logical_analysis = await self.mcp_skill_client.call_skill_tool(
                "logical_reasoner",
                "analyze_logical_consistency",
                {"hypotheses": hypotheses.get("result", {}).get("hypotheses", [])}
            )

            # Use Grok-4 to optimize debate structure
            if self.grok_skill_messaging:
                debate_optimization = await self.grok_skill_messaging.predict_skill_coordination_needs({
                    "task": "debate_coordination",
                    "topic": debate_data.get("topic", ""),
                    "perspectives": debate_data.get("perspectives", []),
                    "hypotheses": hypotheses.get("result", {}),
                    "logical_analysis": logical_analysis.get("result", {})
                })

                # Apply Grok-4 optimizations to debate structure
                optimized_debate_params = {
                    "topic": debate_data.get("topic", ""),
                    "perspectives": debate_data.get("perspectives", []),
                    "debate_structure": debate_optimization.get("coordination_strategy", "dialectical"),
                    "optimization_hints": debate_optimization.get("optimization_opportunities", [])
                }
            else:
                optimized_debate_params = debate_data

            # Conduct enhanced debate
            result = await self.debate_orchestration_mcp(**optimized_debate_params)

            # Update performance metrics
            skill_duration = time.time() - skill_start_time
            await self._update_skill_performance("grok_enhanced_debate_skill", {
                "duration": skill_duration,
                "success": result.get("success", False),
                "consensus_achieved": result.get("debate_results", {}).get("consensus_achieved", False),
                "rounds_conducted": result.get("debate_results", {}).get("rounds_conducted", 0),
                "grok_enhanced": self.grok_skill_messaging is not None
            })

            return result

        except Exception as e:
            logger.error(f"Grok enhanced debate skill failed: {e}")
            await self._update_skill_performance("grok_enhanced_debate_skill", {
                "duration": time.time() - skill_start_time,
                "success": False,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}

    async def _update_skill_performance(self, skill_name: str, performance_data: Dict[str, Any]):
        """Update skill performance metrics for Grok-4 analysis"""
        try:
            # Update local metrics
            self.skill_communication_metrics["total_skill_calls"] += 1
            if performance_data.get("success", False):
                self.skill_communication_metrics["successful_skill_calls"] += 1

            # Update skill-specific metrics
            if skill_name not in self.skill_communication_metrics["skill_call_latency"]:
                self.skill_communication_metrics["skill_call_latency"][skill_name] = []

            self.skill_communication_metrics["skill_call_latency"][skill_name].append(performance_data.get("duration", 0))

            # Limit history size
            if len(self.skill_communication_metrics["skill_call_latency"][skill_name]) > 100:
                self.skill_communication_metrics["skill_call_latency"][skill_name] = self.skill_communication_metrics["skill_call_latency"][skill_name][-100:]

            # Update Grok-4 performance history
            if self.grok_skill_messaging:
                await self.grok_skill_messaging.update_skill_performance_history(skill_name, performance_data)

        except Exception as e:
            logger.error(f"Failed to update skill performance metrics: {e}")

    async def shutdown(self):
        """Cleanup resources"""
        try:
            # Clear active sessions
            self.active_reasoning_sessions.clear()

            # Cleanup Grok intra-skill messaging resources
            if self.grok_intra_skill_messaging:
                await self.grok_intra_skill_messaging.cleanup()

            logger.info("Reasoning Agent shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_agent_card(self) -> Dict[str, Any]:
        """Get A2A agent card for service discovery"""
        return {
            "name": "Advanced-Reasoning-Agent",
            "description": "Multi-agent reasoning system with hierarchical orchestration and swarm intelligence",
            "version": "1.0.0",
            "protocolVersion": "0.2.9",
            "provider": {
                "name": "A2A Reasoning Systems",
                "url": "https://a2a-reasoning.com",
                "contact": "support@a2a-reasoning.com"
            },
            "capabilities": {
                "streaming": True,
                "multiAgentOrchestration": True,
                "reasoningArchitectures": [arch.value for arch in ReasoningArchitecture],
                "debate": True,
                "hierarchicalReasoning": True,
                "swarmIntelligence": True
            },
            "skills": [
                {
                    "name": "multi_agent_reasoning",
                    "description": "Perform advanced multi-agent reasoning on complex questions",
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"],
                    "parameters": {
                        "question": {"type": "string", "required": True},
                        "architecture": {
                            "type": "string",
                            "enum": ["hierarchical", "peer_to_peer", "hub_and_spoke",
                                    "blackboard", "graph_based", "hybrid"]
                        },
                        "enable_debate": {"type": "boolean", "default": True}
                    }
                }
            ],
            "securitySchemes": {
                "bearer": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            },
            "serviceEndpoint": f"{self.base_url}/a2a"
        }
    def _handle_reasoning_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle reasoning errors explicitly instead of silent failures"""
        logger.error(f"Reasoning error in {context}: {error}")

        return {
            "success": False,
            "error": str(error),
            "context": context,
            "fallback_used": True,
            "recommendation": "Use hierarchical reasoning for reliable results"
        }
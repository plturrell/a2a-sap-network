"""
AI Intelligence Framework for A2A Agents
Phase 1: Core AI Framework - Integration Module

This module integrates all AI intelligence components to provide enhanced
capabilities for A2A agents including reasoning, learning, memory, collaboration,
explainability, and autonomous decision-making.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Import core AI intelligence components
from .reasoning_engine import ReasoningStrategy, create_reasoning_engine
from .adaptive_learning import LearningStrategy, LearningExperience, create_adaptive_learning_system
from .memory_context import MemoryType, create_memory_context_manager
from .collaborative_intelligence import (
    CollaborationType,
    create_collaborative_intelligence_framework,
)
from .explainability import ExplanationType, ExplanationLevel, create_explainability_framework
from .autonomous_decisions import (
    DecisionType,
    PlanningAlgorithm,
    create_autonomous_decision_framework,
)

logger = logging.getLogger(__name__)


@dataclass
class AIIntelligenceConfig:
    """Configuration for AI Intelligence Framework"""

    # Reasoning configuration
    reasoning_enabled: bool = True
    default_reasoning_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    # Learning configuration
    learning_enabled: bool = True
    default_learning_strategy: LearningStrategy = LearningStrategy.REINFORCEMENT
    learning_storage_path: Optional[str] = None

    # Memory configuration
    memory_enabled: bool = True
    memory_embedding_dim: int = 768
    memory_storage_path: Optional[str] = None

    # Collaboration configuration
    collaboration_enabled: bool = True
    default_collaboration_type: CollaborationType = CollaborationType.PEER_TO_PEER

    # Explainability configuration
    explainability_enabled: bool = True
    default_explanation_type: ExplanationType = ExplanationType.NATURAL_LANGUAGE
    default_explanation_level: ExplanationLevel = ExplanationLevel.DETAILED

    # Autonomous decisions configuration
    autonomy_enabled: bool = True
    default_decision_type: DecisionType = DecisionType.REACTIVE
    default_planning_algorithm: PlanningAlgorithm = PlanningAlgorithm.FORWARD_CHAINING

    # Performance settings
    enable_performance_monitoring: bool = True
    cache_size: int = 1000
    max_concurrent_operations: int = 10


class AIIntelligenceFramework:
    """
    Integrated AI Intelligence Framework

    Combines all AI intelligence components into a unified system that provides
    enhanced reasoning, learning, memory, collaboration, explainability, and
    autonomous decision-making capabilities for A2A agents.
    """

    def __init__(self, agent_id: str, config: Optional[AIIntelligenceConfig] = None):
        self.agent_id = agent_id
        self.config = config or AIIntelligenceConfig()

        # Core components
        self.reasoning_engine = None
        self.learning_system = None
        self.memory_manager = None
        self.collaboration_framework = None
        self.explainability_framework = None
        self.decision_framework = None

        # Integration state
        self.initialized = False
        self.active_operations = {}
        self.performance_metrics = {
            "operations_completed": 0,
            "operations_failed": 0,
            "average_operation_time": 0.0,
            "cache_hit_rate": 0.0,
        }

        logger.info(f"Created AI Intelligence Framework for agent {agent_id}")

    async def initialize(self) -> bool:
        """
        Initialize all AI intelligence components

        Returns:
            Success status of initialization
        """
        try:
            logger.info(f"Initializing AI Intelligence Framework for {self.agent_id}...")

            # Initialize reasoning engine
            if self.config.reasoning_enabled:
                self.reasoning_engine = create_reasoning_engine(self.agent_id)
                logger.info("✅ Reasoning engine initialized")

            # Initialize adaptive learning system
            if self.config.learning_enabled:
                self.learning_system = create_adaptive_learning_system(self.agent_id)
                logger.info("✅ Adaptive learning system initialized")

            # Initialize memory and context manager
            if self.config.memory_enabled:
                self.memory_manager = create_memory_context_manager(self.agent_id)
                await self.memory_manager.start_consolidation_loop()
                logger.info("✅ Memory and context manager initialized")

            # Initialize collaborative intelligence
            if self.config.collaboration_enabled:
                self.collaboration_framework = create_collaborative_intelligence_framework(
                    self.agent_id
                )
                logger.info("✅ Collaborative intelligence framework initialized")

            # Initialize explainability framework
            if self.config.explainability_enabled:
                self.explainability_framework = create_explainability_framework(self.agent_id)
                logger.info("✅ Explainability framework initialized")

            # Initialize autonomous decision framework
            if self.config.autonomy_enabled:
                self.decision_framework = create_autonomous_decision_framework(self.agent_id)
                logger.info("✅ Autonomous decision framework initialized")

            self.initialized = True
            logger.info("🎉 AI Intelligence Framework fully initialized!")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AI Intelligence Framework: {e}")
            return False

    async def enhance_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[ReasoningStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced reasoning with multiple strategies

        Args:
            query: The reasoning query
            context: Optional context information
            strategy: Reasoning strategy to use

        Returns:
            Reasoning result with enhanced capabilities
        """
        if not self.reasoning_engine:
            raise RuntimeError("Reasoning engine not initialized")

        start_time = datetime.utcnow()
        operation_id = f"reasoning_{start_time.timestamp()}"

        try:
            # Use specified strategy or default
            strategy = strategy or self.config.default_reasoning_strategy

            # Perform reasoning
            reasoning_result = await self.reasoning_engine.reason(
                query=query, context=context or {}, strategy=strategy
            )

            # Store in memory if available
            if self.memory_manager:
                await self.memory_manager.store_memory(
                    content={
                        "query": query,
                        "result": reasoning_result,
                        "strategy": strategy.value,
                    },
                    memory_type=MemoryType.EPISODIC,
                    importance=0.8,
                    metadata={"type": "reasoning", "operation_id": operation_id},
                )

            # Learn from the reasoning process if learning is enabled
            if self.learning_system:
                experience = LearningExperience(
                    experience_id=operation_id,
                    timestamp=start_time,
                    context=context or {},
                    action=f"reasoning_{strategy.value}",
                    outcome=reasoning_result,
                    reward=reasoning_result.get("confidence", 0.5),
                )
                await self.learning_system.learn(experience)

            # Update performance metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["operations_completed"] += 1
            self._update_average_operation_time(duration)

            return {
                "success": True,
                "result": reasoning_result,
                "strategy_used": strategy.value,
                "operation_id": operation_id,
                "duration": duration,
            }

        except Exception as e:
            self.performance_metrics["operations_failed"] += 1
            logger.error(f"Enhanced reasoning failed: {e}")
            return {"success": False, "error": str(e), "operation_id": operation_id}

    async def intelligent_learning(
        self, experience_data: Dict[str, Any], strategy: Optional[LearningStrategy] = None
    ) -> Dict[str, Any]:
        """
        Intelligent learning with multiple strategies

        Args:
            experience_data: Data to learn from
            strategy: Learning strategy to use

        Returns:
            Learning result
        """
        if not self.learning_system:
            raise RuntimeError("Learning system not initialized")

        strategy = strategy or self.config.default_learning_strategy

        # Create learning experience
        experience = LearningExperience(
            experience_id=str(datetime.utcnow().timestamp()),
            timestamp=datetime.utcnow(),
            context=experience_data.get("context", {}),
            action=experience_data.get("action", "unknown"),
            outcome=experience_data.get("outcome"),
            reward=experience_data.get("reward", 0.0),
            metadata=experience_data.get("metadata", {}),
        )

        # Learn and get insights
        learning_result = await self.learning_system.learn(experience, strategy)

        # Store learning insights in memory
        if self.memory_manager:
            await self.memory_manager.store_memory(
                content=learning_result,
                memory_type=MemoryType.SEMANTIC,
                importance=0.7,
                metadata={"type": "learning_insight"},
            )

        return learning_result

    async def contextual_memory_retrieval(
        self, query: Any, memory_types: Optional[List[MemoryType]] = None, top_k: int = 5
    ) -> List[Any]:
        """
        Retrieve relevant memories with context awareness

        Args:
            query: Query for memory retrieval
            memory_types: Types of memory to search
            top_k: Number of memories to retrieve

        Returns:
            Retrieved memories
        """
        if not self.memory_manager:
            raise RuntimeError("Memory manager not initialized")

        return await self.memory_manager.retrieve_memory(
            query=query, memory_types=memory_types, top_k=top_k
        )

    async def collaborative_decision_making(
        self,
        decision_context: Dict[str, Any],
        collaboration_type: Optional[CollaborationType] = None,
    ) -> Dict[str, Any]:
        """
        Make collaborative decisions with other agents

        Args:
            decision_context: Context for the decision
            collaboration_type: Type of collaboration

        Returns:
            Collaborative decision result
        """
        if not self.collaboration_framework:
            raise RuntimeError("Collaboration framework not initialized")

        collaboration_type = collaboration_type or self.config.default_collaboration_type

        # This would involve actual agent collaboration
        # For now, return a framework response
        return {
            "collaboration_type": collaboration_type.value,
            "decision": "collaborative_decision_made",
            "participants": [self.agent_id],
            "confidence": 0.8,
        }

    async def explainable_operation(
        self,
        operation_context: Dict[str, Any],
        explanation_type: Optional[ExplanationType] = None,
        explanation_level: Optional[ExplanationLevel] = None,
    ) -> Dict[str, Any]:
        """
        Perform operation with explainability

        Args:
            operation_context: Context of the operation to explain
            explanation_type: Type of explanation
            explanation_level: Level of explanation detail

        Returns:
            Operation result with explanation
        """
        if not self.explainability_framework:
            raise RuntimeError("Explainability framework not initialized")

        explanation_type = explanation_type or self.config.default_explanation_type
        explanation_level = explanation_level or self.config.default_explanation_level

        # Generate explanation
        explanation = await self.explainability_framework.explain_decision(
            decision_context=operation_context,
            explanation_type=explanation_type,
            level=explanation_level,
        )

        return {
            "operation_result": operation_context,
            "explanation": explanation,
            "explainable": True,
        }

    async def autonomous_action(
        self, decision_type: Optional[DecisionType] = None, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Take autonomous action based on current state

        Args:
            decision_type: Type of autonomous decision
            context: Decision context

        Returns:
            Autonomous action result
        """
        if not self.decision_framework:
            raise RuntimeError("Decision framework not initialized")

        decision_type = decision_type or self.config.default_decision_type

        # Make autonomous decision
        decision_result = await self.decision_framework.make_autonomous_decision(
            context=context, decision_type=decision_type
        )

        # Learn from the autonomous action
        if self.learning_system and decision_result.get("success"):
            experience = LearningExperience(
                experience_id=str(datetime.utcnow().timestamp()),
                timestamp=datetime.utcnow(),
                context=context or {},
                action="autonomous_decision",
                outcome=decision_result,
                reward=1.0 if decision_result.get("success") else 0.0,
            )
            await self.learning_system.learn(experience)

        return decision_result

    async def integrated_intelligence_operation(
        self, task_description: str, task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform integrated intelligence operation using multiple AI components

        Args:
            task_description: Description of the task
            task_context: Context for the task

        Returns:
            Integrated operation result
        """
        operation_id = f"integrated_{datetime.utcnow().timestamp()}"
        results = {}

        try:
            # Step 1: Reason about the task
            if self.reasoning_engine:
                reasoning_result = await self.enhance_reasoning(
                    query=task_description, context=task_context
                )
                results["reasoning"] = reasoning_result

            # Step 2: Retrieve relevant memories
            if self.memory_manager:
                memories = await self.contextual_memory_retrieval(query=task_description, top_k=3)
                results["relevant_memories"] = memories

            # Step 3: Make autonomous decision
            if self.decision_framework:
                decision = await self.autonomous_action(context=task_context)
                results["autonomous_decision"] = decision

            # Step 4: Generate explanation
            if self.explainability_framework:
                explanation = await self.explainable_operation(
                    operation_context={
                        "task": task_description,
                        "context": task_context,
                        "reasoning": results.get("reasoning"),
                        "decision": results.get("autonomous_decision"),
                    }
                )
                results["explanation"] = explanation

            # Step 5: Learn from the integrated operation
            if self.learning_system:
                learning_result = await self.intelligent_learning(
                    {
                        "context": task_context,
                        "action": "integrated_operation",
                        "outcome": results,
                        "reward": 0.8,  # High reward for successful integration
                    }
                )
                results["learning"] = learning_result

            return {
                "success": True,
                "operation_id": operation_id,
                "task_description": task_description,
                "results": results,
                "intelligence_components_used": len(results),
            }

        except Exception as e:
            logger.error(f"Integrated intelligence operation failed: {e}")
            return {"success": False, "error": str(e), "operation_id": operation_id}

    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get status of all AI intelligence components"""
        return {
            "framework_initialized": self.initialized,
            "agent_id": self.agent_id,
            "components": {
                "reasoning_engine": self.reasoning_engine is not None,
                "learning_system": self.learning_system is not None,
                "memory_manager": self.memory_manager is not None,
                "collaboration_framework": self.collaboration_framework is not None,
                "explainability_framework": self.explainability_framework is not None,
                "decision_framework": self.decision_framework is not None,
            },
            "performance_metrics": self.performance_metrics,
            "config": {
                "reasoning_enabled": self.config.reasoning_enabled,
                "learning_enabled": self.config.learning_enabled,
                "memory_enabled": self.config.memory_enabled,
                "collaboration_enabled": self.config.collaboration_enabled,
                "explainability_enabled": self.config.explainability_enabled,
                "autonomy_enabled": self.config.autonomy_enabled,
            },
        }

    async def shutdown(self):
        """Shutdown AI intelligence framework"""
        logger.info(f"Shutting down AI Intelligence Framework for {self.agent_id}...")

        # Stop memory consolidation
        if self.memory_manager:
            self.memory_manager.stop_consolidation_loop()

        self.initialized = False
        logger.info("AI Intelligence Framework shutdown complete")

    def _update_average_operation_time(self, duration: float):
        """Update average operation time metric"""
        total_ops = self.performance_metrics["operations_completed"]
        current_avg = self.performance_metrics["average_operation_time"]

        # Calculate new average
        new_avg = ((current_avg * (total_ops - 1)) + duration) / total_ops
        self.performance_metrics["average_operation_time"] = new_avg


# Factory functions for easy integration
async def create_ai_intelligence_framework(
    agent_id: str, config: Optional[AIIntelligenceConfig] = None
) -> AIIntelligenceFramework:
    """
    Create and initialize AI Intelligence Framework

    Args:
        agent_id: ID of the agent
        config: Configuration for the framework

    Returns:
        Initialized AI Intelligence Framework
    """
    framework = AIIntelligenceFramework(agent_id, config)
    await framework.initialize()
    return framework


def create_enhanced_agent_config() -> AIIntelligenceConfig:
    """
    Create configuration for enhanced AI agent with all features enabled

    Returns:
        Enhanced configuration
    """
    return AIIntelligenceConfig(
        reasoning_enabled=True,
        learning_enabled=True,
        memory_enabled=True,
        collaboration_enabled=True,
        explainability_enabled=True,
        autonomy_enabled=True,
        enable_performance_monitoring=True,
    )


# Version and component information
__version__ = "1.0.0"
__components__ = [
    "reasoning_engine",
    "adaptive_learning",
    "memory_context",
    "collaborative_intelligence",
    "explainability",
    "autonomous_decisions",
]

logger.info(f"AI Intelligence Framework v{__version__} loaded with components: {__components__}")

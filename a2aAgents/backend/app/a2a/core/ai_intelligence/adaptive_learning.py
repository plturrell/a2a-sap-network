"""
Adaptive Learning System for A2A Agents
Part of Phase 1: Core AI Framework

This module provides adaptive learning capabilities with self-improvement,
pattern learning, and feedback loops.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod
import pickle
import os

logger = logging.getLogger(__name__)


def create_performance_history_deque():
    """Create a deque for performance history with max length of 1000"""
    return deque(maxlen=1000)

def get_strategy_score(x):
    """Get strategy score for sorting"""
    return x[1]


class LearningStrategy(str, Enum):
    """Available learning strategies"""

    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META = "meta"
    FEDERATED = "federated"
    CONTINUAL = "continual"
    ACTIVE = "active"


@dataclass
class LearningExperience:
    """Represents a learning experience"""

    experience_id: str
    timestamp: datetime
    context: Dict[str, Any]
    action: str
    outcome: Any
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningModel:
    """Represents a learned model"""

    model_id: str
    strategy: LearningStrategy
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    version: int = 1


class AdaptiveLearningSystem:
    """
    Advanced adaptive learning system for continuous improvement
    Supports multiple learning strategies and self-optimization
    """

    def __init__(self, agent_id: str, storage_path: Optional[str] = None):
        self.agent_id = agent_id
        self.storage_path = storage_path or f"/tmp/a2a_learning/{agent_id}"
        os.makedirs(self.storage_path, exist_ok=True)

        # Learning components
        self.experience_buffer = deque(maxlen=10000)
        self.models = {}
        self.active_strategy = LearningStrategy.REINFORCEMENT

        # Performance tracking
        self.performance_history = defaultdict(create_performance_history_deque)
        self.learning_curves = defaultdict(list)

        # Strategy implementations
        self.strategies = {
            LearningStrategy.REINFORCEMENT: ReinforcementLearner(),
            LearningStrategy.SUPERVISED: SupervisedLearner(),
            LearningStrategy.META: MetaLearner(),
            LearningStrategy.TRANSFER: TransferLearner(),
            LearningStrategy.CONTINUAL: ContinualLearner(),
            LearningStrategy.ACTIVE: ActiveLearner(),
        }

        # Hyperparameter optimization
        self.hyperparameter_optimizer = HyperparameterOptimizer()

        # Load existing models
        self._load_models()

        logger.info(f"Initialized adaptive learning system for agent {agent_id}")

    async def learn(
        self, experience: LearningExperience, strategy: Optional[LearningStrategy] = None
    ) -> Dict[str, Any]:
        """
        Learn from an experience using specified or active strategy

        Args:
            experience: The learning experience
            strategy: Optional specific strategy to use

        Returns:
            Learning result with updates and insights
        """
        strategy = strategy or self.active_strategy

        # Add to experience buffer
        self.experience_buffer.append(experience)

        # Get the learner
        learner = self.strategies.get(strategy)
        if not learner:
            raise ValueError(f"Unknown learning strategy: {strategy}")

        # Perform learning
        result = await learner.learn(experience, self.experience_buffer)

        # Update performance metrics
        self._update_performance_metrics(strategy, result)

        # Check if we should optimize hyperparameters
        if len(self.experience_buffer) % 100 == 0:
            await self._optimize_hyperparameters(strategy)

        # Check if we should switch strategies
        if len(self.experience_buffer) % 500 == 0:
            await self._evaluate_strategy_switch()

        # Save models periodically
        if len(self.experience_buffer) % 1000 == 0:
            self._save_models()

        return result

    async def predict(
        self, context: Dict[str, Any], model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using learned models

        Args:
            context: Context for prediction
            model_id: Optional specific model to use

        Returns:
            Prediction result
        """
        if model_id:
            model = self.models.get(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
        else:
            # Use best performing model
            model = self._get_best_model()

        if not model:
            return {"prediction": None, "confidence": 0.0, "reason": "No trained models available"}

        # Get the strategy learner
        learner = self.strategies.get(model.strategy)

        # Make prediction
        prediction = await learner.predict(context, model)

        return {
            "prediction": prediction,
            "model_id": model.model_id,
            "strategy": model.strategy.value,
            "confidence": prediction.get("confidence", 0.0),
        }

    async def adapt(self, feedback: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt based on feedback

        Args:
            feedback: Feedback on previous predictions/actions
            context: Context of the feedback

        Returns:
            Adaptation result
        """
        # Create experience from feedback
        experience = LearningExperience(
            experience_id=f"feedback_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            context=context,
            action=feedback.get("action", "unknown"),
            outcome=feedback.get("outcome"),
            reward=feedback.get("reward", 0.0),
            metadata=feedback,
        )

        # Learn from feedback
        learning_result = await self.learn(experience)

        # Perform specific adaptations based on feedback type
        adaptation_result = await self._perform_adaptation(feedback, context)

        return {
            "learning_result": learning_result,
            "adaptation_result": adaptation_result,
            "new_strategy": self.active_strategy.value,
            "models_updated": len(self.models),
        }

    async def transfer_knowledge(
        self, source_domain: str, target_domain: str, knowledge_filter: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from one domain to another

        Args:
            source_domain: Source domain identifier
            target_domain: Target domain identifier
            knowledge_filter: Optional filter for knowledge transfer

        Returns:
            Transfer result
        """
        transfer_learner = self.strategies[LearningStrategy.TRANSFER]

        # Get relevant experiences from source domain
        source_experiences = [
            exp for exp in self.experience_buffer if exp.metadata.get("domain") == source_domain
        ]

        if knowledge_filter:
            source_experiences = [exp for exp in source_experiences if knowledge_filter(exp)]

        # Perform transfer learning
        transfer_result = await transfer_learner.transfer(
            source_experiences, target_domain, self.models
        )

        return transfer_result

    async def meta_learn(self, task_distribution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform meta-learning across task distribution

        Args:
            task_distribution: Distribution of tasks to learn from

        Returns:
            Meta-learning result
        """
        meta_learner = self.strategies[LearningStrategy.META]

        # Perform meta-learning
        meta_result = await meta_learner.meta_learn(
            task_distribution, self.experience_buffer, self.models
        )

        # Update models with meta-knowledge
        if meta_result.get("meta_model"):
            self.models["meta_model"] = meta_result["meta_model"]

        return meta_result

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            "total_experiences": len(self.experience_buffer),
            "active_strategy": self.active_strategy.value,
            "num_models": len(self.models),
            "performance_by_strategy": {},
        }

        # Calculate performance by strategy
        for strategy, history in self.performance_history.items():
            if history:
                recent_performance = list(history)[-100:]  # Last 100 entries
                stats["performance_by_strategy"][strategy] = {
                    "average_reward": np.mean([h["reward"] for h in recent_performance]),
                    "success_rate": np.mean([h["success"] for h in recent_performance]),
                    "learning_rate": self._calculate_learning_rate(history),
                }

        # Add model performance
        stats["model_performance"] = {}
        for model_id, model in self.models.items():
            stats["model_performance"][model_id] = {
                "strategy": model.strategy.value,
                "version": model.version,
                "performance_metrics": model.performance_metrics,
                "age": (datetime.utcnow() - model.created_at).total_seconds(),
            }

        return stats

    async def _optimize_hyperparameters(self, strategy: LearningStrategy):
        """Optimize hyperparameters for a strategy"""
        learner = self.strategies[strategy]

        # Get recent experiences
        recent_experiences = list(self.experience_buffer)[-500:]

        # Optimize hyperparameters
        optimal_params = await self.hyperparameter_optimizer.optimize(
            learner, recent_experiences, current_params=learner.get_hyperparameters()
        )

        # Update learner parameters
        learner.set_hyperparameters(optimal_params)

        logger.info(f"Optimized hyperparameters for {strategy}: {optimal_params}")

    async def _evaluate_strategy_switch(self):
        """Evaluate if we should switch learning strategies"""
        # Calculate recent performance for each strategy
        strategy_scores = {}

        for strategy, history in self.performance_history.items():
            if len(history) >= 10:
                recent = list(history)[-50:]
                score = np.mean([h["reward"] for h in recent])
                trend = self._calculate_trend(recent)

                # Combine score and trend
                strategy_scores[strategy] = score + trend * 0.3

        if strategy_scores:
            # Select best strategy
            best_strategy = max(strategy_scores.items(), key=get_strategy_score)[0]

            if (
                best_strategy != self.active_strategy
                and strategy_scores[best_strategy]
                > strategy_scores.get(self.active_strategy, 0) * 1.2
            ):
                logger.info(f"Switching strategy from {self.active_strategy} to {best_strategy}")
                self.active_strategy = LearningStrategy(best_strategy)

    async def _perform_adaptation(
        self, feedback: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform specific adaptations based on feedback"""
        adaptation_type = feedback.get("type", "general")

        if adaptation_type == "error_correction":
            # Adapt to correct errors
            return await self._adapt_error_correction(feedback, context)
        elif adaptation_type == "preference":
            # Adapt to user preferences
            return await self._adapt_preferences(feedback, context)
        elif adaptation_type == "performance":
            # Adapt for performance improvement
            return await self._adapt_performance(feedback, context)
        else:
            # General adaptation
            return {"adapted": True, "type": "general"}

    async def _adapt_error_correction(
        self, feedback: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt based on error feedback"""
        error_type = feedback.get("error_type")
        correction = feedback.get("correction")

        # Create corrective experience
        corrective_exp = LearningExperience(
            experience_id=f"correction_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            context=context,
            action=correction.get("action") if correction else "unknown",
            outcome=correction.get("outcome") if correction else None,
            reward=1.0,  # Positive reward for corrections
            metadata={"type": "correction", "error_type": error_type},
        )

        # Prioritize learning from corrections
        await self.learn(corrective_exp, strategy=LearningStrategy.SUPERVISED)

        return {
            "adapted": True,
            "type": "error_correction",
            "error_type": error_type,
            "correction_applied": True,
        }

    async def _adapt_preferences(
        self, feedback: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt based on preference feedback"""
        preference = feedback.get("preference")

        # Update preference model
        if "preference_model" not in self.models:
            self.models["preference_model"] = LearningModel(
                model_id="preference_model",
                strategy=LearningStrategy.SUPERVISED,
                parameters={"preferences": {}},
                performance_metrics={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        # Update preferences
        pref_model = self.models["preference_model"]
        pref_model.parameters["preferences"][context.get("user_id", "default")] = preference
        pref_model.updated_at = datetime.utcnow()

        return {"adapted": True, "type": "preference", "preference_updated": True}

    async def _adapt_performance(
        self, feedback: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt for performance improvement"""
        performance_metric = feedback.get("metric")
        target_value = feedback.get("target")
        current_value = feedback.get("current")

        # Calculate performance gap
        gap = target_value - current_value if target_value and current_value else 0

        # Adjust learning rate based on gap
        for learner in self.strategies.values():
            if hasattr(learner, "adjust_learning_rate"):
                learner.adjust_learning_rate(gap)

        return {
            "adapted": True,
            "type": "performance",
            "metric": performance_metric,
            "gap": gap,
            "adjustments_made": True,
        }

    def _update_performance_metrics(self, strategy: LearningStrategy, result: Dict[str, Any]):
        """Update performance metrics"""
        metric = {
            "timestamp": datetime.utcnow(),
            "reward": result.get("reward", 0.0),
            "success": result.get("success", False),
            "loss": result.get("loss", 0.0),
        }

        self.performance_history[strategy.value].append(metric)

        # Update learning curves
        self.learning_curves[strategy.value].append(
            {
                "iteration": len(self.learning_curves[strategy.value]),
                "performance": metric["reward"],
                "timestamp": metric["timestamp"],
            }
        )

    def _get_best_model(self) -> Optional[LearningModel]:
        """Get the best performing model"""
        if not self.models:
            return None

        best_model = None
        best_score = float("-inf")

        for model in self.models.values():
            # Calculate score based on performance metrics
            score = model.performance_metrics.get("accuracy", 0.0) * 0.5
            score += model.performance_metrics.get("reward", 0.0) * 0.3
            score += (1.0 / (model.version + 1)) * 0.2  # Prefer newer versions

            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    def _calculate_learning_rate(self, history: deque) -> float:
        """Calculate learning rate from performance history"""
        if len(history) < 2:
            return 0.0

        # Calculate improvement over time
        early = list(history)[:20]
        recent = list(history)[-20:]

        early_avg = np.mean([h["reward"] for h in early])
        recent_avg = np.mean([h["reward"] for h in recent])

        return (recent_avg - early_avg) / len(history)

    def _calculate_trend(self, history: List[Dict[str, Any]]) -> float:
        """Calculate performance trend"""
        if len(history) < 2:
            return 0.0

        # Simple linear regression
        x = np.arange(len(history))
        y = np.array([h["reward"] for h in history])

        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _save_models(self):
        """Save models to storage"""
        for model_id, model in self.models.items():
            model_path = os.path.join(self.storage_path, f"model_{model_id}.pkl")
            try:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                logger.error(f"Failed to save model {model_id}: {e}")

    def _load_models(self):
        """Load models from storage"""
        if not os.path.exists(self.storage_path):
            return

        for filename in os.listdir(self.storage_path):
            if filename.startswith("model_") and filename.endswith(".pkl"):
                model_path = os.path.join(self.storage_path, filename)
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                        self.models[model.model_id] = model
                except Exception as e:
                    logger.error(f"Failed to load model from {filename}: {e}")


class BaseLearner(ABC):
    """Base class for learning strategy implementations"""

    def __init__(self):
        self.hyperparameters = self.get_default_hyperparameters()

    @abstractmethod
    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn from experience"""

    @abstractmethod
    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction"""

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters"""

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters"""
        return self.hyperparameters.copy()

    def set_hyperparameters(self, params: Dict[str, Any]):
        """Set hyperparameters"""
        self.hyperparameters.update(params)


class ReinforcementLearner(BaseLearner):
    """Reinforcement learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": 0.01,
            "discount_factor": 0.95,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "batch_size": 32,
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn using reinforcement learning"""
        # Q-learning update (simplified)
        learning_rate = self.hyperparameters["learning_rate"]
        discount_factor = self.hyperparameters["discount_factor"]

        # Calculate temporal difference error
        td_error = experience.reward  # Simplified - would include future rewards

        # In a full implementation, we would use:
        # td_error = reward + discount_factor * max_future_q - current_q
        # q_update = current_q + learning_rate * td_error

        # Update would happen here in a real implementation
        # For now, return learning statistics
        return {
            "strategy": "reinforcement",
            "reward": experience.reward,
            "td_error": td_error,
            "epsilon": self.hyperparameters["epsilon"],
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction using RL model"""
        # Epsilon-greedy action selection
        epsilon = self.hyperparameters["epsilon"]

        if np.random.random() < epsilon:
            # Exploration
            action = "explore"
            confidence = epsilon
        else:
            # Exploitation (would use Q-values in real implementation)
            action = "exploit"
            confidence = 1.0 - epsilon

        return {"action": action, "confidence": confidence, "exploration_rate": epsilon}

    def adjust_learning_rate(self, performance_gap: float):
        """Adjust learning rate based on performance"""
        if performance_gap > 0.5:
            self.hyperparameters["learning_rate"] *= 1.1
        elif performance_gap < -0.5:
            self.hyperparameters["learning_rate"] *= 0.9

        # Decay epsilon
        self.hyperparameters["epsilon"] *= self.hyperparameters["epsilon_decay"]


class SupervisedLearner(BaseLearner):
    """Supervised learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "regularization": 0.01,
            "optimizer": "adam",
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn using supervised learning"""
        # In real implementation, would train a model
        # For now, simulate learning

        # Calculate loss (simplified)
        loss = np.random.random() * 0.1  # Simulated loss

        return {
            "strategy": "supervised",
            "loss": loss,
            "accuracy": 1.0 - loss,
            "batch_size": self.hyperparameters["batch_size"],
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction using supervised model"""
        # Simulate prediction
        prediction = "predicted_output"
        confidence = 0.85 + np.random.random() * 0.15

        return {"prediction": prediction, "confidence": confidence, "model_version": model.version}


class MetaLearner(BaseLearner):
    """Meta-learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "meta_learning_rate": 0.01,
            "task_learning_rate": 0.1,
            "adaptation_steps": 5,
            "meta_batch_size": 10,
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn using meta-learning"""
        # MAML-style meta-learning (simplified)
        # meta_lr = self.hyperparameters["meta_learning_rate"]  # For future meta-learning implementation

        # Would perform inner loop adaptation and outer loop meta-update
        meta_loss = np.random.random() * 0.05

        return {
            "strategy": "meta",
            "meta_loss": meta_loss,
            "adaptation_steps": self.hyperparameters["adaptation_steps"],
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction using meta-learned model"""
        # Quick adaptation to new task
        adapted_prediction = "meta_adapted_output"
        confidence = 0.9  # Meta-learning typically has high confidence after adaptation

        return {"prediction": adapted_prediction, "confidence": confidence, "adapted": True}

    async def meta_learn(
        self,
        task_distribution: List[Dict[str, Any]],
        buffer: deque,
        models: Dict[str, LearningModel],
    ) -> Dict[str, Any]:
        """Perform meta-learning across task distribution"""
        # Simulate meta-learning across tasks
        meta_model = LearningModel(
            model_id="meta_model",
            strategy=LearningStrategy.META,
            parameters={"meta_parameters": {}, "task_embeddings": {}},
            performance_metrics={"meta_accuracy": 0.92, "adaptation_efficiency": 0.88},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return {
            "meta_model": meta_model,
            "tasks_learned": len(task_distribution),
            "meta_performance": 0.9,
        }


class TransferLearner(BaseLearner):
    """Transfer learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "transfer_rate": 0.7,
            "domain_similarity_threshold": 0.6,
            "feature_extraction_layers": 3,
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn using transfer learning"""
        transfer_rate = self.hyperparameters["transfer_rate"]

        # Simulate transfer learning
        transfer_loss = np.random.random() * 0.08

        return {
            "strategy": "transfer",
            "transfer_loss": transfer_loss,
            "transfer_rate": transfer_rate,
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction using transferred knowledge"""
        prediction = "transferred_prediction"
        confidence = 0.82  # Slightly lower confidence due to domain shift

        return {
            "prediction": prediction,
            "confidence": confidence,
            "source_domain": model.parameters.get("source_domain", "unknown"),
        }

    async def transfer(
        self,
        source_experiences: List[LearningExperience],
        target_domain: str,
        models: Dict[str, LearningModel],
    ) -> Dict[str, Any]:
        """Transfer knowledge between domains"""
        # Extract transferable features
        transferable_knowledge = {
            "feature_representations": {},
            "domain_invariant_patterns": [],
            "adaptation_strategies": [],
        }

        # Create transfer model
        transfer_model = LearningModel(
            model_id=f"transfer_{target_domain}",
            strategy=LearningStrategy.TRANSFER,
            parameters={
                "source_domain": "source",
                "target_domain": target_domain,
                "transferred_knowledge": transferable_knowledge,
            },
            performance_metrics={"transfer_efficiency": 0.85, "target_performance": 0.78},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return {
            "transfer_model": transfer_model,
            "knowledge_transferred": len(transferable_knowledge),
            "expected_performance": 0.78,
        }


class ContinualLearner(BaseLearner):
    """Continual learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "memory_size": 1000,
            "replay_frequency": 10,
            "regularization_strength": 0.1,
            "task_boundary_detection": True,
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn continuously without forgetting"""
        # Elastic weight consolidation or similar
        # regularization = self.hyperparameters["regularization_strength"]  # For future regularization implementation

        # Simulate continual learning
        continual_loss = np.random.random() * 0.06

        return {
            "strategy": "continual",
            "continual_loss": continual_loss,
            "memory_usage": len(buffer) / self.hyperparameters["memory_size"],
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction using continual learning model"""
        prediction = "continual_prediction"
        confidence = 0.88

        # Check which task this belongs to
        task_id = context.get("task_id", "unknown")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "task_id": task_id,
            "memory_consolidated": True,
        }


class ActiveLearner(BaseLearner):
    """Active learning implementation"""

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
            "uncertainty_threshold": 0.3,
            "query_strategy": "uncertainty_sampling",
            "budget": 100,
            "diversity_weight": 0.2,
        }

    async def learn(self, experience: LearningExperience, buffer: deque) -> Dict[str, Any]:
        """Learn using active learning"""
        # Query strategy for selecting informative samples
        uncertainty_threshold = self.hyperparameters["uncertainty_threshold"]

        # Simulate active learning
        query_value = np.random.random()
        should_query = query_value < uncertainty_threshold

        return {
            "strategy": "active",
            "queried": should_query,
            "uncertainty": query_value,
            "budget_remaining": self.hyperparameters["budget"],
            "success": True,
        }

    async def predict(self, context: Dict[str, Any], model: LearningModel) -> Dict[str, Any]:
        """Make prediction with uncertainty estimation"""
        prediction = "active_prediction"

        # Estimate uncertainty
        uncertainty = np.random.random() * 0.4
        confidence = 1.0 - uncertainty

        # Suggest if labeling would be valuable
        worth_labeling = uncertainty > self.hyperparameters["uncertainty_threshold"]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "suggest_labeling": worth_labeling,
        }


class HyperparameterOptimizer:
    """Hyperparameter optimization using Bayesian optimization"""

    def __init__(self):
        self.optimization_history = []

    async def optimize(
        self,
        learner: BaseLearner,
        experiences: List[LearningExperience],
        current_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        # Simplified Bayesian optimization
        # In real implementation would use proper BO library

        # Try variations of current parameters
        param_variations = self._generate_variations(current_params)

        best_params = current_params
        best_score = 0.0

        for params in param_variations:
            # Evaluate parameters (simplified)
            score = await self._evaluate_params(learner, experiences, params)

            if score > best_score:
                best_score = score
                best_params = params

        # Record optimization
        self.optimization_history.append(
            {"timestamp": datetime.utcnow(), "best_params": best_params, "best_score": best_score}
        )

        return best_params

    def _generate_variations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter variations"""
        variations = [params.copy()]

        # Create variations for each parameter
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Numeric parameters
                var_up = params.copy()
                var_up[key] = value * 1.1
                variations.append(var_up)

                var_down = params.copy()
                var_down[key] = value * 0.9
                variations.append(var_down)

        return variations

    async def _evaluate_params(
        self, learner: BaseLearner, experiences: List[LearningExperience], params: Dict[str, Any]
    ) -> float:
        """Evaluate parameter set"""
        # Simplified evaluation
        # In real implementation would run actual evaluation

        # Simulate evaluation score
        base_score = 0.7

        # Prefer moderate learning rates
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if 0.001 <= lr <= 0.1:
                base_score += 0.1

        # Add some randomness
        score = base_score + np.random.random() * 0.2

        return min(score, 1.0)


# Utility functions
def create_adaptive_learning_system(agent_id: str) -> AdaptiveLearningSystem:
    """Factory function to create adaptive learning system"""
    return AdaptiveLearningSystem(agent_id)


async def learn_from_experience(
    system: AdaptiveLearningSystem, experience: LearningExperience
) -> Dict[str, Any]:
    """Convenience function for learning"""
    return await system.learn(experience)

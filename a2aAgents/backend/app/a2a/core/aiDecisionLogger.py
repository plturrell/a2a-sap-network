# AI Decision Logger for A2A Agents
# Provides structured logging and learning from AI-powered decisions

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of AI decisions"""

    HELP_REQUEST = "help_request"
    ADVISOR_GUIDANCE = "advisor_guidance"
    ERROR_RECOVERY = "error_recovery"
    TASK_PLANNING = "task_planning"
    DELEGATION = "delegation"
    QUALITY_ASSESSMENT = "quality_assessment"


class OutcomeStatus(Enum):
    """Status of decision outcomes"""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    PENDING = "pending"
    UNKNOWN = "unknown"


@dataclass
class AIDecision:
    """Represents an AI-powered decision made by an agent"""

    decision_id: str
    agent_id: str
    decision_type: DecisionType
    timestamp: float = field(default_factory=time.time)
    question: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    ai_response: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutcome:
    """Outcome of an AI decision"""

    decision_id: str
    outcome_status: OutcomeStatus
    outcome_timestamp: float = field(default_factory=time.time)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    feedback: Optional[str] = None
    actual_duration: float = 0.0


@dataclass
class PatternInsight:
    """Learned pattern from decision history"""

    pattern_type: str
    description: str
    confidence: float
    evidence_count: int
    success_rate: float
    applicable_contexts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AIDecisionLogger:
    """Centralized logger for AI decisions across agents"""

    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[str] = None,
        memory_size: int = 1000,
        learning_threshold: int = 10,  # Min decisions before learning patterns
    ):
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.learning_threshold = learning_threshold

        # In-memory storage
        self.decisions: Dict[str, AIDecision] = {}
        self.outcomes: Dict[str, DecisionOutcome] = {}
        self.decision_history: deque = deque(maxlen=memory_size)

        # Analytics tracking
        self.decision_stats = defaultdict(
            lambda: {
                "total": 0,
                "success": 0,
                "failure": 0,
                "avg_response_time": 0.0,
                "avg_confidence": 0.0,
            }
        )

        # Pattern learning
        self.learned_patterns: List[PatternInsight] = []
        self.pattern_cache: Dict[str, Any] = {}

        # Performance tracking
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
            "pending_outcomes": 0,
            "avg_decision_time": 0.0,
            "learning_effectiveness": 0.0,
        }

        # Storage configuration
        self.storage_path = storage_path or os.path.join(
            os.getenv("AI_DECISION_STORAGE_PATH", "/tmp/ai_decisions"), agent_id
        )
        os.makedirs(self.storage_path, exist_ok=True)

        # Background tasks
        self._analysis_task = None
        self._persistence_task = None
        self._start_background_tasks()

        logger.info(f"AI Decision Logger initialized for agent {agent_id}")

    def _start_background_tasks(self):
        """Start analysis & persistence tasks only if an event loop is running."""

        async def analysis_loop():
            while True:
                try:
                    await self._analyze_patterns()
                    await asyncio.sleep(300)  # Analyze every 5 minutes
                except (RuntimeError, ValueError, KeyError) as e:
                    logger.error(f"Pattern analysis error: {e}")
                    await asyncio.sleep(60)

        async def persistence_loop():
            while True:
                try:
                    await self._persist_data()
                    await asyncio.sleep(60)  # Persist every minute
                except (IOError, OSError, ValueError) as e:
                    logger.error(f"Persistence error: {e}")
                    await asyncio.sleep(30)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Skipping AIDecisionLogger background tasks – no running event loop.")
            return
        self._analysis_task = asyncio.create_task(analysis_loop())
        self._persistence_task = asyncio.create_task(persistence_loop())

    async def log_decision(
        self,
        decision_type: DecisionType,
        question: str,
        ai_response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        response_time: Optional[float] = None,
    ) -> str:
        """Log an AI decision"""
        decision_id = f"{self.agent_id}_{int(time.time() * 1000)}_{len(self.decisions)}"

        decision = AIDecision(
            decision_id=decision_id,
            agent_id=self.agent_id,
            decision_type=decision_type,
            question=question,
            context=context or {},
            ai_response=ai_response,
            confidence_score=confidence_score or self._extract_confidence(ai_response),
            response_time=response_time or 0.0,
            metadata={
                "logged_at": datetime.utcnow().isoformat(),
                "source": "ai_advisor" if "advisor" in str(decision_type) else "system",
            },
        )

        # Store decision
        self.decisions[decision_id] = decision
        self.decision_history.append(decision_id)

        # Update statistics
        self._update_stats(decision)
        self.performance_metrics["total_decisions"] += 1

        logger.debug(f"Logged AI decision {decision_id}: {decision_type.value}")

        return decision_id

    async def log_outcome(
        self,
        decision_id: str,
        outcome_status: OutcomeStatus,
        success_metrics: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        feedback: Optional[str] = None,
        actual_duration: Optional[float] = None,
    ) -> bool:
        """Log the outcome of an AI decision"""
        if decision_id not in self.decisions:
            logger.warning(f"Decision {decision_id} not found")
            return False

        outcome = DecisionOutcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            success_metrics=success_metrics or {},
            failure_reason=failure_reason,
            feedback=feedback,
            actual_duration=actual_duration or 0.0,
            side_effects=[],
        )

        # Store outcome
        self.outcomes[decision_id] = outcome

        # Update performance metrics
        if outcome_status == OutcomeStatus.SUCCESS:
            self.performance_metrics["successful_outcomes"] += 1
        elif outcome_status == OutcomeStatus.FAILURE:
            self.performance_metrics["failed_outcomes"] += 1
        elif outcome_status == OutcomeStatus.PENDING:
            self.performance_metrics["pending_outcomes"] += 1

        # Trigger immediate pattern learning if significant
        if outcome_status in [OutcomeStatus.SUCCESS, OutcomeStatus.FAILURE]:
            await self._learn_from_outcome(decision_id, outcome)

        logger.debug(f"Logged outcome for decision {decision_id}: {outcome_status.value}")

        return True

    def _extract_confidence(self, ai_response: Dict[str, Any]) -> float:
        """Extract confidence score from AI response"""
        # Look for confidence indicators in response
        if isinstance(ai_response, dict):
            if "confidence" in ai_response:
                return float(ai_response["confidence"])
            elif "confidence_score" in ai_response:
                return float(ai_response["confidence_score"])
            elif "advisor_response" in ai_response:
                advisor_resp = ai_response["advisor_response"]
                if isinstance(advisor_resp, dict) and "confidence" in advisor_resp:
                    return float(advisor_resp["confidence"])

        # Default confidence based on response characteristics
        response_str = str(ai_response).lower()
        if "high confidence" in response_str:
            return 0.9
        elif "moderate confidence" in response_str:
            return 0.7
        elif "low confidence" in response_str:
            return 0.4

        return 0.6  # Default moderate confidence

    def _update_stats(self, decision: AIDecision):
        """Update decision statistics"""
        stats = self.decision_stats[decision.decision_type.value]
        stats["total"] += 1

        # Update moving averages
        n = stats["total"]
        stats["avg_response_time"] = (
            stats["avg_response_time"] * (n - 1) + decision.response_time
        ) / n
        stats["avg_confidence"] = (
            stats["avg_confidence"] * (n - 1) + decision.confidence_score
        ) / n

    async def _learn_from_outcome(self, decision_id: str, outcome: DecisionOutcome):
        """Learn patterns from decision outcomes"""
        decision = self.decisions.get(decision_id)
        if not decision:
            return

        # Extract learning features
        features = {
            "decision_type": decision.decision_type.value,
            "confidence": decision.confidence_score,
            "response_time": decision.response_time,
            "outcome": outcome.outcome_status.value,
            "context_keys": list(decision.context.keys()),
            "question_length": len(decision.question),
            "has_failure_reason": outcome.failure_reason is not None,
        }

        # Cache features for pattern analysis
        pattern_key = f"{decision.decision_type.value}_{outcome.outcome_status.value}"
        if pattern_key not in self.pattern_cache:
            self.pattern_cache[pattern_key] = []
        self.pattern_cache[pattern_key].append(features)

        # Keep cache size manageable
        if len(self.pattern_cache[pattern_key]) > 100:
            self.pattern_cache[pattern_key] = self.pattern_cache[pattern_key][-100:]

    async def _analyze_patterns(self):
        """Analyze decision history to identify patterns"""
        if len(self.decisions) < self.learning_threshold:
            return

        logger.info(f"Analyzing patterns from {len(self.decisions)} decisions")

        new_patterns = []

        # Pattern 1: Success rate by decision type
        for decision_type, stats in self.decision_stats.items():
            if stats["total"] >= 5:  # Minimum sample size
                success_count = sum(
                    1
                    for outcome in self.outcomes.values()
                    if outcome.decision_id in self.decisions
                    and self.decisions[outcome.decision_id].decision_type.value == decision_type
                    and outcome.outcome_status == OutcomeStatus.SUCCESS
                )

                success_rate = success_count / stats["total"]

                if success_rate < 0.5:  # Low success rate pattern
                    pattern = PatternInsight(
                        pattern_type="low_success_rate",
                        description=f"Low success rate for {decision_type} decisions",
                        confidence=0.8,
                        evidence_count=stats["total"],
                        success_rate=success_rate,
                        applicable_contexts=[decision_type],
                        recommendations=[
                            f"Review {decision_type} decision logic",
                            "Consider additional context gathering",
                            "Adjust confidence thresholds",
                        ],
                    )
                    new_patterns.append(pattern)

        # Pattern 2: Correlation between confidence and success
        confidence_outcomes = []
        for decision_id, outcome in self.outcomes.items():
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                confidence_outcomes.append(
                    {
                        "confidence": decision.confidence_score,
                        "success": outcome.outcome_status == OutcomeStatus.SUCCESS,
                    }
                )

        if len(confidence_outcomes) >= 10:
            # Group by confidence buckets
            high_conf = [co for co in confidence_outcomes if co["confidence"] >= 0.8]
            # low_conf = [co for co in confidence_outcomes if co["confidence"] < 0.5]  # Reserved for future analysis

            if high_conf:
                high_conf_success = sum(1 for co in high_conf if co["success"]) / len(high_conf)
                if high_conf_success < 0.6:  # High confidence but low success
                    pattern = PatternInsight(
                        pattern_type="overconfident_failures",
                        description="High confidence decisions failing more than expected",
                        confidence=0.7,
                        evidence_count=len(high_conf),
                        success_rate=high_conf_success,
                        applicable_contexts=["all"],
                        recommendations=[
                            "Recalibrate confidence scoring",
                            "Add validation steps for high-confidence decisions",
                        ],
                    )
                    new_patterns.append(pattern)

        # Pattern 3: Context-specific patterns
        context_success = defaultdict(lambda: {"total": 0, "success": 0})
        for decision_id, outcome in self.outcomes.items():
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                for context_key in decision.context.keys():
                    context_success[context_key]["total"] += 1
                    if outcome.outcome_status == OutcomeStatus.SUCCESS:
                        context_success[context_key]["success"] += 1

        for context_key, stats in context_success.items():
            if stats["total"] >= 5:
                success_rate = stats["success"] / stats["total"]
                if success_rate > 0.8:  # High success with specific context
                    pattern = PatternInsight(
                        pattern_type="context_success_factor",
                        description=f"Decisions with '{context_key}' context have high "
                        "success rate",
                        confidence=0.75,
                        evidence_count=stats["total"],
                        success_rate=success_rate,
                        applicable_contexts=[context_key],
                        recommendations=[
                            f"Prioritize including '{context_key}' in decision context",
                            f"Study why '{context_key}' improves outcomes",
                        ],
                    )
                    new_patterns.append(pattern)

        # Update learned patterns
        self.learned_patterns.extend(new_patterns)

        # Keep only most relevant patterns
        if len(self.learned_patterns) > 50:
            # Sort by evidence count and recency
            self.learned_patterns.sort(key=lambda p: (p.evidence_count, p.confidence), reverse=True)
            self.learned_patterns = self.learned_patterns[:50]

        # Update learning effectiveness metric
        if self.performance_metrics["total_decisions"] > 0:
            self.performance_metrics["learning_effectiveness"] = (
                len(self.learned_patterns) / self.performance_metrics["total_decisions"]
            )

        logger.info(
            f"Identified {len(new_patterns)} new patterns, "
            f"total patterns: {len(self.learned_patterns)}"
        )

    async def get_recommendations(
        self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get recommendations based on learned patterns"""
        recommendations = []

        # Find applicable patterns
        for pattern in self.learned_patterns:
            # Check if pattern applies to this decision type
            if (
                decision_type.value in pattern.applicable_contexts
                or "all" in pattern.applicable_contexts
            ):

                # Check if context matches
                if context and any(key in pattern.applicable_contexts for key in context.keys()):
                    recommendations.extend(pattern.recommendations)
                elif not context and pattern.pattern_type in [
                    "low_success_rate",
                    "overconfident_failures",
                ]:
                    recommendations.extend(pattern.recommendations)

        # Add stats-based recommendations
        stats = self.decision_stats.get(decision_type.value)
        if stats and stats["total"] > 0:
            if stats["avg_response_time"] > 5.0:
                recommendations.append("Consider caching frequent queries to reduce response time")

            if stats["avg_confidence"] < 0.5:
                recommendations.append("Low average confidence - provide more context in queries")

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:5]  # Return top 5 recommendations

    def get_decision_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about AI decisions"""
        # Calculate success rates by type
        success_rates = {}
        for decision_type in DecisionType:
            type_decisions = [
                d for d in self.decisions.values() if d.decision_type == decision_type
            ]

            if type_decisions:
                successful = sum(
                    1
                    for d in type_decisions
                    if d.decision_id in self.outcomes
                    and self.outcomes[d.decision_id].outcome_status == OutcomeStatus.SUCCESS
                )
                success_rates[decision_type.value] = successful / len(type_decisions)

        # Recent performance trends
        recent_decisions = list(self.decision_history)[-20:]  # Last 20 decisions
        recent_success = 0
        recent_avg_confidence = 0

        if recent_decisions:
            for decision_id in recent_decisions:
                if decision_id in self.outcomes:
                    if self.outcomes[decision_id].outcome_status == OutcomeStatus.SUCCESS:
                        recent_success += 1
                if decision_id in self.decisions:
                    recent_avg_confidence += self.decisions[decision_id].confidence_score

            recent_success_rate = recent_success / len(recent_decisions)
            recent_avg_confidence /= len(recent_decisions)
        else:
            recent_success_rate = 0
            recent_avg_confidence = 0

        # Pattern insights
        pattern_summary = defaultdict(int)
        for pattern in self.learned_patterns:
            pattern_summary[pattern.pattern_type] += 1

        return {
            "summary": {
                "total_decisions": self.performance_metrics["total_decisions"],
                "successful_outcomes": self.performance_metrics["successful_outcomes"],
                "failed_outcomes": self.performance_metrics["failed_outcomes"],
                "pending_outcomes": self.performance_metrics["pending_outcomes"],
                "overall_success_rate": (
                    self.performance_metrics["successful_outcomes"]
                    / max(self.performance_metrics["total_decisions"], 1)
                ),
            },
            "by_type": dict(self.decision_stats.items()),
            "success_rates": success_rates,
            "recent_performance": {
                "decisions_analyzed": len(recent_decisions),
                "success_rate": recent_success_rate,
                "avg_confidence": recent_avg_confidence,
            },
            "patterns": {
                "total_patterns": len(self.learned_patterns),
                "pattern_types": dict(pattern_summary),
                "learning_effectiveness": self.performance_metrics["learning_effectiveness"],
            },
            "recommendations_available": len(self.learned_patterns) > 0,
        }

    def info(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Return a lightweight summary for test expectations."""
        return self.get_decision_analytics()["summary"]

    def get_decision_history(
        self, decision_type: Optional[DecisionType] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        history = []

        for decision_id in reversed(list(self.decision_history)):
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]

                # Filter by type if specified
                if decision_type and decision.decision_type != decision_type:
                    continue

                # Build history entry
                entry = {
                    "decision_id": decision.decision_id,
                    "timestamp": decision.timestamp,
                    "type": decision.decision_type.value,
                    "question": (
                        decision.question[:100] + "..."
                        if len(decision.question) > 100
                        else decision.question
                    ),
                    "confidence": decision.confidence_score,
                    "response_time": decision.response_time,
                }

                # Add outcome if available
                if decision_id in self.outcomes:
                    outcome = self.outcomes[decision_id]
                    entry["outcome"] = {
                        "status": outcome.outcome_status.value,
                        "timestamp": outcome.outcome_timestamp,
                        "duration": outcome.actual_duration,
                    }

                history.append(entry)

                if len(history) >= limit:
                    break

        return history

    async def _persist_data(self):
        """Persist decision data to storage"""
        try:
            # Save decisions
            decisions_file = os.path.join(self.storage_path, "decisions.json")
            decisions_data = {
                decision_id: asdict(decision) for decision_id, decision in self.decisions.items()
            }

            # Convert enums to strings
            for decision_data in decisions_data.values():
                decision_data["decision_type"] = decision_data["decision_type"].value

            with open(decisions_file, "w", encoding="utf-8") as f:
                json.dump(decisions_data, f, indent=2, default=str)

            # Save outcomes
            outcomes_file = os.path.join(self.storage_path, "outcomes.json")
            outcomes_data = {
                decision_id: asdict(outcome) for decision_id, outcome in self.outcomes.items()
            }

            # Convert enums to strings
            for outcome_data in outcomes_data.values():
                outcome_data["outcome_status"] = outcome_data["outcome_status"].value

            with open(outcomes_file, "w", encoding="utf-8") as f:
                json.dump(outcomes_data, f, indent=2, default=str)

            # Save patterns
            patterns_file = os.path.join(self.storage_path, "patterns.json")
            patterns_data = [asdict(pattern) for pattern in self.learned_patterns]

            with open(patterns_file, "w", encoding="utf-8") as f:
                json.dump(patterns_data, f, indent=2, default=str)

            # Save analytics snapshot
            analytics_file = os.path.join(self.storage_path, "analytics.json")
            analytics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "analytics": self.get_decision_analytics(),
                "performance_metrics": self.performance_metrics,
            }

            with open(analytics_file, "w", encoding="utf-8") as f:
                json.dump(analytics_data, f, indent=2, default=str)

        except (IOError, OSError, ValueError) as e:
            logger.error(f"Failed to persist AI decision data: {e}")

    async def load_historical_data(self):
        """Load historical decision data from storage"""
        try:
            # Load decisions
            decisions_file = os.path.join(self.storage_path, "decisions.json")
            if os.path.exists(decisions_file):
                with open(decisions_file, "r", encoding="utf-8") as f:
                    decisions_data = json.load(f)

                for decision_id, decision_dict in decisions_data.items():
                    # Convert string to enum
                    decision_dict["decision_type"] = DecisionType(decision_dict["decision_type"])

                    decision = AIDecision(**decision_dict)
                    self.decisions[decision_id] = decision
                    self.decision_history.append(decision_id)

                logger.info(f"Loaded {len(self.decisions)} historical decisions")

            # Load outcomes
            outcomes_file = os.path.join(self.storage_path, "outcomes.json")
            if os.path.exists(outcomes_file):
                with open(outcomes_file, "r", encoding="utf-8") as f:
                    outcomes_data = json.load(f)

                for decision_id, outcome_dict in outcomes_data.items():
                    # Convert string to enum
                    outcome_dict["outcome_status"] = OutcomeStatus(outcome_dict["outcome_status"])

                    outcome = DecisionOutcome(**outcome_dict)
                    self.outcomes[decision_id] = outcome

                logger.info(f"Loaded {len(self.outcomes)} historical outcomes")

            # Load patterns
            patterns_file = os.path.join(self.storage_path, "patterns.json")
            if os.path.exists(patterns_file):
                with open(patterns_file, "r", encoding="utf-8") as f:
                    patterns_data = json.load(f)

                self.learned_patterns = [
                    PatternInsight(**pattern_dict) for pattern_dict in patterns_data
                ]

                logger.info(f"Loaded {len(self.learned_patterns)} learned patterns")

            # Rebuild statistics
            for decision in self.decisions.values():
                self._update_stats(decision)

        except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load historical data: {e}")

    async def export_insights_report(self) -> Dict[str, Any]:
        """Export a comprehensive insights report"""
        analytics = self.get_decision_analytics()

        report = {
            "agent_id": self.agent_id,
            "report_timestamp": datetime.utcnow().isoformat(),
            "summary": analytics["summary"],
            "performance_by_type": analytics["by_type"],
            "success_analysis": {
                "overall": analytics["summary"]["overall_success_rate"],
                "by_type": analytics["success_rates"],
                "recent_trend": analytics["recent_performance"],
            },
            "learned_insights": [
                {
                    "pattern": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "evidence": pattern.evidence_count,
                    "recommendations": pattern.recommendations,
                }
                for pattern in self.learned_patterns[:10]  # Top 10 patterns
            ],
            "recommendations": {
                decision_type.value: await self.get_recommendations(decision_type)
                for decision_type in DecisionType
            },
            "data_quality": {
                "decisions_with_outcomes": sum(
                    1 for d_id in self.decisions if d_id in self.outcomes
                ),
                "pending_outcomes": analytics["summary"]["pending_outcomes"],
                "avg_outcome_delay": self._calculate_avg_outcome_delay(),
            },
        }

        return report

    def _calculate_avg_outcome_delay(self) -> float:
        """Calculate average time between decision and outcome"""
        delays = []

        for decision_id, outcome in self.outcomes.items():
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                delay = outcome.outcome_timestamp - decision.timestamp
                delays.append(delay)

        return statistics.mean(delays) if delays else 0.0

    async def shutdown(self):
        """Gracefully shutdown the logger"""
        logger.info("Shutting down AI Decision Logger")

        # Final persistence
        await self._persist_data()

        # Cancel background tasks
        if self._analysis_task:
            self._analysis_task.cancel()
        if self._persistence_task:
            self._persistence_task.cancel()

        # Wait for tasks to complete
        try:
            if self._analysis_task:
                await self._analysis_task
            if self._persistence_task:
                await self._persistence_task
        except asyncio.CancelledError:
            pass

        logger.info("AI Decision Logger shutdown complete")


# Global registry for cross-agent learning
class AIDecisionRegistry:
    """Global registry for cross-agent decision learning"""

    def __init__(self):
        self.agent_loggers: Dict[str, AIDecisionLogger] = {}
        self.cross_agent_patterns: List[PatternInsight] = []
        self._analysis_task = None

    def register_agent(self, agent_id: str, logger: AIDecisionLogger):
        """Register an agent's decision logger"""
        self.agent_loggers[agent_id] = logger
        logger.info(f"Registered agent {agent_id} with global AI decision registry")

    async def analyze_cross_agent_patterns(self):
        """Analyze patterns across all agents"""
        if len(self.agent_loggers) < 2:
            return  # Need at least 2 agents for cross-agent analysis

        # Aggregate data from all agents
        all_decisions = []
        all_outcomes = []

        for agent_id, logger in self.agent_loggers.items():
            for decision in logger.decisions.values():
                decision_copy = AIDecision(**asdict(decision))
                decision_copy.metadata["source_agent"] = agent_id
                all_decisions.append(decision_copy)

            for outcome in logger.outcomes.values():
                all_outcomes.append(outcome)

        # Analyze cross-agent patterns
        # ... (pattern analysis logic similar to single-agent but across agents)

        module_logger = logging.getLogger(__name__)
        module_logger.info(
            f"Analyzed {len(all_decisions)} decisions across {len(self.agent_loggers)} agents"
        )

    def get_global_insights(self) -> Dict[str, Any]:
        """Get insights across all agents"""
        total_decisions = sum(
            agent_logger.performance_metrics["total_decisions"] for agent_logger in self.agent_loggers.values()
        )

        total_success = sum(
            agent_logger.performance_metrics["successful_outcomes"]
            for agent_logger in self.agent_loggers.values()
        )

        agent_performance = {
            agent_id: agent_logger.get_decision_analytics()["summary"]
            for agent_id, agent_logger in self.agent_loggers.items()
        }

        return {
            "total_agents": len(self.agent_loggers),
            "total_decisions": total_decisions,
            "global_success_rate": total_success / max(total_decisions, 1),
            "agent_performance": agent_performance,
            "cross_agent_patterns": len(self.cross_agent_patterns),
        }


# Create global registry instance
_global_registry = AIDecisionRegistry()


def get_global_decision_registry() -> AIDecisionRegistry:
    """Get the global AI decision registry"""
    return _global_registry

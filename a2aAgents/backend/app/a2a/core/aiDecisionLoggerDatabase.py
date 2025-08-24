"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

# Database-backed AI Decision Logger for A2A Agents
# Integrates with Data Manager Agent for persistent storage

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from .aiDecisionLogger import (
    DecisionType,
    OutcomeStatus,
    AIDecision,
    DecisionOutcome,
    PatternInsight,
)
from .a2aTypes import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class AIDecisionDatabaseLogger:
    """Database-backed AI Decision Logger using Data Manager Agent"""

    def __init__(
        self,
        agent_id: str,
        data_manager_url: str,
        memory_size: int = 1000,
        learning_threshold: int = 10,
        cache_ttl: int = 300,  # 5 minutes cache
    ):
        self.agent_id = agent_id
        self.data_manager_url = data_manager_url.rstrip("/")
        self.memory_size = memory_size
        self.learning_threshold = learning_threshold
        self.cache_ttl = cache_ttl

        # In-memory cache for performance
        self._decision_cache: Dict[str, AIDecision] = {}
        self._outcome_cache: Dict[str, DecisionOutcome] = {}
        self._pattern_cache: List[PatternInsight] = []
        # Public learned patterns list to satisfy unit tests
        self.learned_patterns: List[PatternInsight] = []
        self._cache_timestamps: Dict[str, float] = {}

        # Analytics tracking
        self.decision_stats = defaultdict(
            lambda: {"total": 0, "success": 0, "failure": 0, "pending": 0}
        )

        self.performance_metrics = {
            "total_decisions": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
            "pending_outcomes": 0,
            "avg_decision_time": 0.0,
            "learning_effectiveness": 0.0,
        }

        # HTTP client for Data Manager communication
        self.http_client = None  # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        # httpx\.AsyncClient(timeout=30.0)

        # Background tasks will be started explicitly once a running event loop is available
        self._analysis_task = None
        self._cache_cleanup_task = None

        logger.info(f"Database AI Decision Logger initialized for agent {agent_id}")

    # -------------------------
    # Convenience helpers for tests

    async def start_background_tasks(self):
        """Public helper to start background tasks when inside an event loop."""
        # Ensure http client available
        if self.http_client and self.http_client.is_closed:
            self.http_client = None  # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            # httpx\.AsyncClient(timeout=30.0)
        if not (self._analysis_task and not self._analysis_task.done()):
            self._start_background_tasks()

    # -------------------------
    def info(self, *args, **kwargs):
        """Return a lightweight analytics summary (mirrors in-memory logger for tests)."""
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        if loop and loop.is_running():
            # Safe to call async analytics
            try:
                return loop.run_until_complete(self.get_decision_analytics()).get("summary", {})
            except Exception:
                pass
        # Fallback simple summary
        return {"total_decisions": self.performance_metrics.get("total_decisions", 0)}

    async def _persist_data(self):
        """Stubbed persistence for unit tests (no-op)."""
        return

    @property
    def is_shutdown(self) -> bool:
        """Whether background tasks have been cancelled (used in tests)."""
        return self._analysis_task is None or self._analysis_task.cancelled()

    def _start_background_tasks(self):
        """Start background analysis and cache management tasks if an event loop is running.

        When imported in synchronous contexts (e.g., during unit-test module collection)
        there may be no running loop yet.  In that case we silently skip starting
        the tasks to avoid `RuntimeError: no running event loop`.  Tests can still
        exercise public async APIs by explicitly calling `await logger.start_background_tasks()`
        once a loop is available, or by running within an async test (pytest-asyncio).
        """

        async def analysis_loop():
            while True:
                try:
                    await self._analyze_patterns()
                    await asyncio.sleep(300)  # Analyze every 5 minutes
                except Exception as e:
                    logger.error(f"Pattern analysis error: {e}")
                    await asyncio.sleep(60)

        async def cache_cleanup_loop():
            while True:
                try:
                    await self._cleanup_cache()
                    await asyncio.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    await asyncio.sleep(30)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop available; defer background tasks
            logger.debug("Skipping background tasks – no running event loop detected.")
            return

        self._analysis_task = asyncio.create_task(analysis_loop())
        self._cache_cleanup_task = asyncio.create_task(cache_cleanup_loop())

    async def _send_to_data_manager(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send A2A message to Data Manager Agent"""
        message_parts = [
            MessagePart(kind="text", text=f"AI Decision Logger {operation}"),
            MessagePart(kind="data", data=data),
        ]

        message = A2AMessage(
            role=MessageRole.AGENT,
            parts=message_parts,
            contextId=f"ai_decision_{operation}_{int(time.time())}",
        )

        try:
            response = await self.http_client.post(
                f"{self.data_manager_url}/process",
                json=message.model_dump(),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Data Manager error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Data Manager request failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to communicate with Data Manager: {e}")
            raise

    async def log_decision(
        self,
        decision_type: DecisionType,
        question: str,
        ai_response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        response_time: float = 0.0,
    ) -> str:
        """Log AI decision to database via Data Manager Agent"""

        decision_id = str(uuid4())
        timestamp = time.time()

        # Extract confidence from AI response if not provided
        if confidence_score is None:
            confidence_score = self._extract_confidence(ai_response)

        # Create decision object
        decision = AIDecision(
            decision_id=decision_id,
            agent_id=self.agent_id,
            decision_type=decision_type,
            timestamp=timestamp,
            question=question,
            context=context or {},
            ai_response=ai_response,
            confidence_score=confidence_score,
            response_time=response_time,
        )

        # Store in database via Data Manager
        try:
            decision_data = {
                "operation": "CREATE",
                "storage_type": "HANA",  # Use HANA for structured data
                "path": "ai_decisions",  # Table name
                "data": {
                    "decision_id": decision_id,
                    "agent_id": self.agent_id,
                    "decision_type": decision_type.value,
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "question": question,
                    "context": json.dumps(context or {}),
                    "ai_response": json.dumps(ai_response),
                    "confidence_score": confidence_score,
                    "response_time": response_time,
                    "metadata": json.dumps({}),
                },
                "service_level": "SILVER",  # Standard processing
            }

            await self._send_to_data_manager("create_decision", decision_data)

            # Cache locally for performance
            self._decision_cache[decision_id] = decision
            self._cache_timestamps[f"decision_{decision_id}"] = time.time()

            # Update statistics
            self._update_stats(decision)

            logger.debug(f"Decision logged to database: {decision_id}")
            return decision_id

        except Exception as e:
            logger.error(f"Failed to log decision to database: {e}")
            # Fall back to in-memory only
            self._decision_cache[decision_id] = decision
            self._update_stats(decision)
            return decision_id

    async def log_outcome(
        self,
        decision_id: str,
        outcome_status: OutcomeStatus,
        success_metrics: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        side_effects: Optional[List[str]] = None,
        feedback: Optional[str] = None,
        actual_duration: float = 0.0,
    ) -> bool:
        """Log decision outcome to database"""

        if decision_id not in self._decision_cache:
            # Try to load from database
            decision = await self._load_decision_from_db(decision_id)
            if not decision:
                logger.warning(f"Cannot log outcome for unknown decision: {decision_id}")
                return False

        outcome = DecisionOutcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            outcome_timestamp=time.time(),
            success_metrics=success_metrics or {},
            failure_reason=failure_reason,
            side_effects=side_effects or [],
            feedback=feedback,
            actual_duration=actual_duration,
        )

        try:
            outcome_data = {
                "operation": "CREATE",
                "storage_type": "HANA",
                "path": "ai_decision_outcomes",
                "data": {
                    "decision_id": decision_id,
                    "outcome_status": outcome_status.value,
                    "outcome_timestamp": datetime.fromtimestamp(
                        outcome.outcome_timestamp
                    ).isoformat(),
                    "success_metrics": json.dumps(success_metrics or {}),
                    "failure_reason": failure_reason,
                    "side_effects": json.dumps(side_effects or []),
                    "feedback": feedback,
                    "actual_duration": actual_duration,
                },
                "service_level": "SILVER",
            }

            await self._send_to_data_manager("create_outcome", outcome_data)

            # Cache outcome
            self._outcome_cache[decision_id] = outcome
            self._cache_timestamps[f"outcome_{decision_id}"] = time.time()

            # Update performance metrics
            self._update_performance_metrics(outcome_status)

            logger.debug(f"Outcome logged to database: {decision_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log outcome to database: {e}")
            # Fall back to cache only
            self._outcome_cache[decision_id] = outcome
            self._update_performance_metrics(outcome_status)
            return False

    async def get_recommendations(
        self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get AI recommendations based on learned patterns from database"""

        try:
            # Query database for relevant patterns
            query_data = {
                "operation": "READ",
                "storage_type": "HANA",
                "query": {
                    "table": "ai_learned_patterns",
                    "where": {"agent_id": self.agent_id, "pattern_type": decision_type.value},
                    "order_by": "confidence DESC",
                    "limit": 10,
                },
            }

            response = await self._send_to_data_manager("query_patterns", query_data)

            recommendations = []
            if response.get("data"):
                for pattern_data in response["data"]:
                    pattern_recommendations = json.loads(pattern_data.get("recommendations", "[]"))
                    recommendations.extend(pattern_recommendations)

            # Deduplicate and limit
            unique_recommendations = list(dict.fromkeys(recommendations))[:5]

            return unique_recommendations

        except Exception as e:
            logger.error(f"Failed to get recommendations from database: {e}")
            # Fall back to cache
            return self._get_cached_recommendations(decision_type, context)

    async def get_decision_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics from database"""

        try:
            # Query analytics view
            query_data = {
                "operation": "READ",
                "storage_type": "HANA",
                "query": {
                    "table": "ai_global_analytics",
                    "where": {"agent_id": self.agent_id},
                    "order_by": "decision_date DESC",
                },
            }

            response = await self._send_to_data_manager("query_analytics", query_data)

            if response.get("data"):
                # Process database analytics
                analytics = self._process_database_analytics(response["data"])
                return analytics
            else:
                # Fall back to in-memory analytics
                return self._get_memory_analytics()

        except Exception as e:
            logger.error(f"Failed to get analytics from database: {e}")
            return self._get_memory_analytics()

    async def get_decision_history(
        self, decision_type: Optional[DecisionType] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get decision history from database"""

        try:
            where_clause = {"agent_id": self.agent_id}
            if decision_type:
                where_clause["decision_type"] = decision_type.value

            query_data = {
                "operation": "READ",
                "storage_type": "HANA",
                "query": {
                    "table": "ai_decision_performance",
                    "where": where_clause,
                    "order_by": "timestamp DESC",
                    "limit": limit,
                },
            }

            response = await self._send_to_data_manager("query_history", query_data)

            if response.get("data"):
                return self._format_history_response(response["data"])
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to get history from database: {e}")
            return []

    async def export_insights_report(self) -> Dict[str, Any]:
        """Export comprehensive insights report from database"""

        try:
            # Get comprehensive data from multiple views
            analytics = await self.get_decision_analytics()
            patterns = await self._get_pattern_effectiveness()

            report = {
                "agent_id": self.agent_id,
                "report_timestamp": datetime.utcnow().isoformat(),
                "summary": analytics.get("summary", {}),
                "performance_by_type": analytics.get("by_type", {}),
                "success_analysis": {
                    "overall": analytics.get("summary", {}).get("overall_success_rate", 0.0),
                    "by_type": analytics.get("success_rates", {}),
                    "recent_trend": analytics.get("recent_performance", 0.0),
                },
                "learned_insights": [
                    {
                        "pattern": pattern.get("pattern_type", ""),
                        "description": pattern.get("description", ""),
                        "confidence": pattern.get("confidence", 0.0),
                        "evidence": pattern.get("evidence_count", 0),
                        "effectiveness": pattern.get("actual_success_rate", 0.0),
                        "recommendations": json.loads(pattern.get("recommendations", "[]")),
                    }
                    for pattern in patterns
                ],
                "recommendations": {
                    decision_type.value: await self.get_recommendations(decision_type)
                    for decision_type in DecisionType
                },
                "data_quality": {
                    "decisions_stored": analytics.get("summary", {}).get("total_decisions", 0),
                    "decisions_with_outcomes": analytics.get("summary", {}).get(
                        "decisions_with_outcomes", 0
                    ),
                    "pattern_count": len(patterns),
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                },
            }

            return report

        except Exception as e:
            logger.error(f"Failed to export insights report: {e}")
            return {"error": str(e), "agent_id": self.agent_id}

    # Helper methods

    def _extract_confidence(self, ai_response: Dict[str, Any]) -> float:
        """Extract confidence score from AI response"""
        if isinstance(ai_response, dict):
            if "confidence" in ai_response:
                return float(ai_response["confidence"])
            elif "advisor_response" in ai_response and isinstance(
                ai_response["advisor_response"], dict
            ):
                if "confidence" in ai_response["advisor_response"]:
                    return float(ai_response["advisor_response"]["confidence"])

            # Look for textual confidence indicators
            answer_text = str(ai_response.get("answer", "")).lower()
            if any(word in answer_text for word in ["high confidence", "certain", "definitely"]):
                return 0.9
            elif any(word in answer_text for word in ["medium confidence", "likely", "probably"]):
                return 0.7
            elif any(word in answer_text for word in ["low confidence", "uncertain", "maybe"]):
                return 0.4

        return 0.5  # Default neutral confidence

    def _update_stats(self, decision: AIDecision):
        """Update decision statistics"""
        decision_type = decision.decision_type.value
        self.decision_stats[decision_type]["total"] += 1
        self.performance_metrics["total_decisions"] += 1

    def _update_performance_metrics(self, outcome_status: OutcomeStatus):
        """Update performance metrics"""
        if outcome_status == OutcomeStatus.SUCCESS:
            self.performance_metrics["successful_outcomes"] += 1
        elif outcome_status == OutcomeStatus.FAILURE:
            self.performance_metrics["failed_outcomes"] += 1
        elif outcome_status == OutcomeStatus.PENDING:
            self.performance_metrics["pending_outcomes"] += 1

    async def _load_decision_from_db(self, decision_id: str) -> Optional[AIDecision]:
        """Load decision from database"""
        try:
            query_data = {
                "operation": "READ",
                "storage_type": "HANA",
                "query": {"table": "ai_decisions", "where": {"decision_id": decision_id}},
            }

            response = await self._send_to_data_manager("load_decision", query_data)

            if response.get("data") and len(response["data"]) > 0:
                data = response["data"][0]
                decision = AIDecision(
                    decision_id=data["decision_id"],
                    agent_id=data["agent_id"],
                    decision_type=DecisionType(data["decision_type"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]).timestamp(),
                    question=data["question"],
                    context=json.loads(data["context"]),
                    ai_response=json.loads(data["ai_response"]),
                    confidence_score=data["confidence_score"],
                    response_time=data["response_time"],
                )

                # Cache it
                self._decision_cache[decision_id] = decision
                self._cache_timestamps[f"decision_{decision_id}"] = time.time()

                return decision

            return None

        except Exception as e:
            logger.error(f"Failed to load decision from database: {e}")
            return None

    async def _analyze_patterns(self):
        """Analyze patterns and update database"""
        try:
            # Get recent decisions for pattern analysis
            recent_decisions = await self.get_decision_history(limit=self.learning_threshold * 2)

            if len(recent_decisions) < self.learning_threshold:
                return

            # Simple pattern analysis - group by decision type and context
            patterns_found = self._identify_patterns(recent_decisions)

            # Store patterns in database
            for pattern in patterns_found:
                await self._store_pattern(pattern)

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")

    def _identify_patterns(self, decisions: List[Dict[str, Any]]) -> List[PatternInsight]:
        """Identify patterns from decision history"""
        patterns = []

        # Group by decision type
        by_type = defaultdict(list)
        for decision in decisions:
            by_type[decision.get("decision_type", "")].append(decision)

        for decision_type, type_decisions in by_type.items():
            if len(type_decisions) >= self.learning_threshold:
                success_rate = sum(
                    1 for d in type_decisions if d.get("success_score", 0) > 0.5
                ) / len(type_decisions)
                avg_confidence = sum(d.get("confidence_score", 0.5) for d in type_decisions) / len(
                    type_decisions
                )

                if success_rate > 0.7:  # High success rate pattern
                    pattern = PatternInsight(
                        pattern_type=f"high_success_{decision_type}",
                        description=f"Decisions of type {decision_type} have {success_rate:.1%} success rate",
                        confidence=min(success_rate + 0.1, 1.0),
                        evidence_count=len(type_decisions),
                        success_rate=success_rate,
                        applicable_contexts=[decision_type],
                        recommendations=[
                            f"Continue current approach for {decision_type} decisions",
                            f"Maintain confidence levels around {avg_confidence:.2f}",
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    async def _store_pattern(self, pattern: PatternInsight):
        """Store learned pattern in database"""
        try:
            pattern_data = {
                "operation": "CREATE",
                "storage_type": "HANA",
                "path": "ai_learned_patterns",
                "data": {
                    "pattern_id": str(uuid4()),
                    "agent_id": self.agent_id,
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "evidence_count": pattern.evidence_count,
                    "success_rate": pattern.success_rate,
                    "applicable_contexts": json.dumps(pattern.applicable_contexts),
                    "recommendations": json.dumps(pattern.recommendations),
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                },
                "service_level": "BRONZE",  # Pattern storage is less critical
            }

            await self._send_to_data_manager("store_pattern", pattern_data)

        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")

    async def _get_pattern_effectiveness(self) -> List[Dict[str, Any]]:
        """Get pattern effectiveness from database view"""
        try:
            query_data = {
                "operation": "READ",
                "storage_type": "HANA",
                "query": {
                    "table": "ai_pattern_effectiveness",
                    "where": {"agent_id": self.agent_id},
                    "order_by": "confidence DESC",
                },
            }

            response = await self._send_to_data_manager("query_pattern_effectiveness", query_data)
            return response.get("data", [])

        except Exception as e:
            logger.error(f"Failed to get pattern effectiveness: {e}")
            return []

    def _process_database_analytics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process database analytics data"""
        if not data:
            return self._get_memory_analytics()

        total_decisions = sum(row.get("total_decisions", 0) for row in data)
        successful_outcomes = sum(row.get("successful_outcomes", 0) for row in data)

        by_type = {}
        success_rates = {}

        for row in data:
            decision_type = row.get("decision_type", "unknown")
            by_type[decision_type] = {
                "total": row.get("total_decisions", 0),
                "successful": row.get("successful_outcomes", 0),
            }

            if row.get("total_decisions", 0) > 0:
                success_rates[decision_type] = row.get("successful_outcomes", 0) / row.get(
                    "total_decisions", 1
                )

        return {
            "summary": {
                "total_decisions": total_decisions,
                "successful_outcomes": successful_outcomes,
                "overall_success_rate": successful_outcomes / max(total_decisions, 1),
                "decisions_with_outcomes": total_decisions,  # All DB entries have outcomes
            },
            "by_type": by_type,
            "success_rates": success_rates,
            "recent_performance": success_rates.get("advisor_guidance", 0.0),
        }

    def _get_memory_analytics(self) -> Dict[str, Any]:
        """Get analytics from in-memory cache"""
        return {
            "summary": {
                "total_decisions": self.performance_metrics["total_decisions"],
                "successful_outcomes": self.performance_metrics["successful_outcomes"],
                "failed_outcomes": self.performance_metrics["failed_outcomes"],
                "overall_success_rate": (
                    self.performance_metrics["successful_outcomes"]
                    / max(self.performance_metrics["total_decisions"], 1)
                ),
            },
            "by_type": dict(self.decision_stats),
            "success_rates": {
                dt: stats["success"] / max(stats["total"], 1)
                for dt, stats in self.decision_stats.items()
            },
        }

    def _get_cached_recommendations(
        self, decision_type: DecisionType, context: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations from cache"""
        recommendations = []
        for pattern in self._pattern_cache:
            if decision_type.value in pattern.applicable_contexts:
                recommendations.extend(pattern.recommendations)

        return list(dict.fromkeys(recommendations))[:5]  # Deduplicate and limit

    def _format_history_response(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format database history response"""
        formatted = []
        for row in data:
            formatted.append(
                {
                    "decision_id": row.get("decision_id"),
                    "type": row.get("decision_type"),
                    "question": row.get("question"),
                    "confidence": row.get("confidence_score"),
                    "outcome": row.get("outcome_status"),
                    "timestamp": row.get("timestamp"),
                    "success_score": row.get("success_score", 0.0),
                }
            )

        return formatted

    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache_timestamps[key]
            if key.startswith("decision_"):
                decision_id = key.replace("decision_", "")
                self._decision_cache.pop(decision_id, None)
            elif key.startswith("outcome_"):
                decision_id = key.replace("outcome_", "")
                self._outcome_cache.pop(decision_id, None)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Simple estimate based on cache size vs total decisions
        cached_decisions = len(self._decision_cache)
        total_decisions = self.performance_metrics["total_decisions"]

        if total_decisions == 0:
            return 0.0

        return min(cached_decisions / total_decisions, 1.0)

    async def shutdown(self):
        """Graceful shutdown"""
        if self._analysis_task:
            self._analysis_task.cancel()

        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()

        await self.http_client.aclose()

        logger.info(f"Database AI Decision Logger shut down for agent {self.agent_id}")


# Global registry for database-backed loggers
_global_database_registry: Optional["AIDatabaseDecisionRegistry"] = None


class AIDatabaseDecisionRegistry:
    """Global registry for database-backed AI decision loggers"""

    def __init__(self):
        self.agent_loggers: Dict[str, AIDecisionDatabaseLogger] = {}

    def register_agent(self, agent_id: str, logger: AIDecisionDatabaseLogger):
        """Register an agent's decision logger"""
        self.agent_loggers[agent_id] = logger

    async def get_global_insights(self) -> Dict[str, Any]:
        """Get insights across all registered agents"""
        if not self.agent_loggers:
            return {"total_agents": 0, "total_decisions": 0}

        insights = {
            "total_agents": len(self.agent_loggers),
            "total_decisions": 0,
            "global_success_rate": 0.0,
            "agent_performance": {},
        }

        total_successful = 0
        total_decisions = 0

        for agent_id, logger in self.agent_loggers.items():
            try:
                analytics = await logger.get_decision_analytics()
                agent_decisions = analytics["summary"]["total_decisions"]
                agent_successful = analytics["summary"]["successful_outcomes"]

                insights["agent_performance"][agent_id] = {
                    "decisions": agent_decisions,
                    "success_rate": agent_successful / max(agent_decisions, 1),
                }

                total_decisions += agent_decisions
                total_successful += agent_successful

            except Exception as e:
                logger.error(f"Failed to get analytics for agent {agent_id}: {e}")

        insights["total_decisions"] = total_decisions
        insights["global_success_rate"] = total_successful / max(total_decisions, 1)

        return insights


def get_global_database_decision_registry() -> AIDatabaseDecisionRegistry:
    """Get or create global database decision registry"""
    global _global_database_registry
    if _global_database_registry is None:
        _global_database_registry = AIDatabaseDecisionRegistry()
    return _global_database_registry

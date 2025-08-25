# A2A Message Protocols for AI Decision Logger Database Integration
# Defines standardized message formats for communication with Data Manager Agent

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .a2aTypes import A2AMessage, MessagePart, MessageRole
from .aiDecisionLogger import DecisionType, OutcomeStatus


class AIDecisionOperation(str, Enum):
    """AI Decision database operations"""

    LOG_DECISION = "log_decision"
    LOG_OUTCOME = "log_outcome"
    QUERY_PATTERNS = "query_patterns"
    STORE_PATTERN = "store_pattern"
    GET_ANALYTICS = "get_analytics"
    GET_HISTORY = "get_history"
    GET_RECOMMENDATIONS = "get_recommendations"
    EXPORT_INSIGHTS = "export_insights"
    INITIALIZE_SCHEMA = "initialize_schema"


class AIDecisionRequest(BaseModel):
    """Base request for AI decision operations"""

    operation: AIDecisionOperation
    agent_id: str
    timestamp: str = Field(default_factory=datetime.utcnow().isoformat)
    context_id: Optional[str] = None


class LogDecisionRequest(AIDecisionRequest):
    """Request to log an AI decision"""

    operation: AIDecisionOperation = AIDecisionOperation.LOG_DECISION
    decision_data: Dict[str, Any] = Field(
        description="Decision data including decision_id, decision_type, question, context, "
        "ai_response, confidence_score, response_time"
    )

    @classmethod
    def create(
        cls,
        agent_id: str,
        decision_id: str,
        decision_type: DecisionType,
        question: str,
        ai_response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        confidence_score: float = 0.5,
        response_time: float = 0.0,
    ) -> "LogDecisionRequest":
        """Create a log decision request"""
        return cls(
            agent_id=agent_id,
            decision_data={
                "decision_id": decision_id,
                "decision_type": decision_type.value,
                "question": question,
                "context": context or {},
                "ai_response": ai_response,
                "confidence_score": confidence_score,
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


class LogOutcomeRequest(AIDecisionRequest):
    """Request to log a decision outcome"""

    operation: AIDecisionOperation = AIDecisionOperation.LOG_OUTCOME
    outcome_data: Dict[str, Any] = Field(
        description="Outcome data including decision_id, outcome_status, success_metrics, "
        "failure_reason, feedback"
    )

    @classmethod
    def create(
        cls,
        agent_id: str,
        decision_id: str,
        outcome_status: OutcomeStatus,
        success_metrics: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        feedback: Optional[str] = None,
        actual_duration: float = 0.0,
    ) -> "LogOutcomeRequest":
        """Create a log outcome request"""
        return cls(
            agent_id=agent_id,
            outcome_data={
                "decision_id": decision_id,
                "outcome_status": outcome_status.value,
                "success_metrics": success_metrics or {},
                "failure_reason": failure_reason,
                "feedback": feedback,
                "actual_duration": actual_duration,
                "outcome_timestamp": datetime.utcnow().isoformat(),
            },
        )


class QueryPatternsRequest(AIDecisionRequest):
    """Request to query learned patterns"""

    operation: AIDecisionOperation = AIDecisionOperation.QUERY_PATTERNS
    query_filters: Dict[str, Any] = Field(
        description="Filters for pattern query (decision_type, confidence_threshold, etc.)"
    )
    limit: int = 10

    @classmethod
    def create(
        cls,
        agent_id: str,
        decision_type: Optional[DecisionType] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> "QueryPatternsRequest":
        """Create a query patterns request"""
        filters = {"agent_id": agent_id}
        if decision_type:
            filters["pattern_type"] = decision_type.value
        if min_confidence > 0:
            filters["confidence_gte"] = min_confidence

        return cls(agent_id=agent_id, query_filters=filters, limit=limit)


class StorePatternRequest(AIDecisionRequest):
    """Request to store a learned pattern"""

    operation: AIDecisionOperation = AIDecisionOperation.STORE_PATTERN
    pattern_data: Dict[str, Any] = Field(
        description="Pattern data including pattern_type, description, confidence, "
        "evidence_count, success_rate, recommendations"
    )

    @classmethod
    def create(
        cls,
        agent_id: str,
        pattern_type: str,
        description: str,
        confidence: float,
        evidence_count: int,
        success_rate: float,
        applicable_contexts: List[str],
        recommendations: List[str],
    ) -> "StorePatternRequest":
        """Create a store pattern request"""
        return cls(
            agent_id=agent_id,
            pattern_data={
                "pattern_id": str(uuid4()),
                "pattern_type": pattern_type,
                "description": description,
                "confidence": confidence,
                "evidence_count": evidence_count,
                "success_rate": success_rate,
                "applicable_contexts": applicable_contexts,
                "recommendations": recommendations,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
        )


class GetAnalyticsRequest(AIDecisionRequest):
    """Request to get decision analytics"""

    operation: AIDecisionOperation = AIDecisionOperation.GET_ANALYTICS
    time_range: Optional[Dict[str, str]] = Field(
        default=None, description="Time range for analytics (start_date, end_date)"
    )
    include_global: bool = False

    @classmethod
    def create(
        cls, agent_id: str, days_back: int = 30, include_global: bool = False
    ) -> "GetAnalyticsRequest":
        """Create an analytics request"""
        end_date = datetime.utcnow()
        start_date = datetime.utcnow().replace(day=end_date.day - days_back)

        return cls(
            agent_id=agent_id,
            time_range={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
            include_global=include_global,
        )


class GetHistoryRequest(AIDecisionRequest):
    """Request to get decision history"""

    operation: AIDecisionOperation = AIDecisionOperation.GET_HISTORY
    filters: Dict[str, Any] = Field(
        description="History filters (decision_type, outcome_status, etc.)"
    )
    limit: int = 50
    offset: int = 0

    @classmethod
    def create(
        cls,
        agent_id: str,
        decision_type: Optional[DecisionType] = None,
        outcome_status: Optional[OutcomeStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> "GetHistoryRequest":
        """Create a history request"""
        filters = {"agent_id": agent_id}
        if decision_type:
            filters["decision_type"] = decision_type.value
        if outcome_status:
            filters["outcome_status"] = outcome_status.value

        return cls(agent_id=agent_id, filters=filters, limit=limit, offset=offset)


class GetRecommendationsRequest(AIDecisionRequest):
    """Request to get AI recommendations"""

    operation: AIDecisionOperation = AIDecisionOperation.GET_RECOMMENDATIONS
    decision_type: str
    context: Dict[str, Any] = Field(default_factory=dict)
    max_recommendations: int = 5

    @classmethod
    def create(
        cls,
        agent_id: str,
        decision_type: DecisionType,
        context: Optional[Dict[str, Any]] = None,
        max_recommendations: int = 5,
    ) -> "GetRecommendationsRequest":
        """Create a recommendations request"""
        return cls(
            agent_id=agent_id,
            decision_type=decision_type.value,
            context=context or {},
            max_recommendations=max_recommendations,
        )


class AIDecisionResponse(BaseModel):
    """Base response for AI decision operations"""

    success: bool
    operation: AIDecisionOperation
    agent_id: str
    timestamp: str = Field(default_factory=datetime.utcnow().isoformat)
    context_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogDecisionResponse(AIDecisionResponse):
    """Response for log decision operation"""

    decision_id: Optional[str] = None
    storage_location: Optional[Dict[str, Any]] = None


class LogOutcomeResponse(AIDecisionResponse):
    """Response for log outcome operation"""

    decision_id: Optional[str] = None
    outcome_recorded: bool = False


class QueryPatternsResponse(AIDecisionResponse):
    """Response for query patterns operation"""

    patterns: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0


class GetAnalyticsResponse(AIDecisionResponse):
    """Response for get analytics operation"""

    analytics: Dict[str, Any] = Field(default_factory=dict)


class GetHistoryResponse(AIDecisionResponse):
    """Response for get history operation"""

    history: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False


class GetRecommendationsResponse(AIDecisionResponse):
    """Response for get recommendations operation"""

    recommendations: List[str] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    pattern_sources: List[str] = Field(default_factory=list)


def create_data_manager_message_for_decision_operation(request: AIDecisionRequest) -> A2AMessage:
    """Create A2A message for Data Manager Agent from AI decision request"""

    # Convert request to Data Manager format
    if request.operation == AIDecisionOperation.LOG_DECISION:
        # Create decision record in database
        log_request = request
        data_manager_data = {
            "operation": "CREATE",
            "storage_type": "HANA",
            "path": "ai_decisions",  # Table name
            "data": {
                "decision_id": log_request.decision_data["decision_id"],
                "agent_id": log_request.agent_id,
                "decision_type": log_request.decision_data["decision_type"],
                "timestamp": log_request.decision_data["timestamp"],
                "question": log_request.decision_data["question"],
                "context": log_request.decision_data["context"],
                "ai_response": log_request.decision_data["ai_response"],
                "confidence_score": log_request.decision_data["confidence_score"],
                "response_time": log_request.decision_data["response_time"],
                "metadata": {},
            },
            "service_level": "SILVER",
        }

    elif request.operation == AIDecisionOperation.LOG_OUTCOME:
        # Create outcome record in database
        outcome_request = request
        data_manager_data = {
            "operation": "CREATE",
            "storage_type": "HANA",
            "path": "ai_decision_outcomes",
            "data": {
                "decision_id": outcome_request.outcome_data["decision_id"],
                "outcome_status": outcome_request.outcome_data["outcome_status"],
                "outcome_timestamp": outcome_request.outcome_data["outcome_timestamp"],
                "success_metrics": outcome_request.outcome_data["success_metrics"],
                "failure_reason": outcome_request.outcome_data.get("failure_reason"),
                "side_effects": outcome_request.outcome_data.get("side_effects", []),
                "feedback": outcome_request.outcome_data.get("feedback"),
                "actual_duration": outcome_request.outcome_data.get("actual_duration", 0.0),
            },
            "service_level": "SILVER",
        }

    elif request.operation == AIDecisionOperation.QUERY_PATTERNS:
        # Query patterns from database
        query_request = request
        data_manager_data = {
            "operation": "READ",
            "storage_type": "HANA",
            "query": {
                "table": "ai_learned_patterns",
                "where": query_request.query_filters,
                "order_by": "confidence DESC",
                "limit": query_request.limit,
            },
        }

    elif request.operation == AIDecisionOperation.STORE_PATTERN:
        # Store pattern in database
        pattern_request = request
        data_manager_data = {
            "operation": "CREATE",
            "storage_type": "HANA",
            "path": "ai_learned_patterns",
            "data": pattern_request.pattern_data,
            "service_level": "BRONZE",  # Patterns are less critical
        }

    elif request.operation == AIDecisionOperation.GET_ANALYTICS:
        # Query analytics view
        analytics_request = request
        data_manager_data = {
            "operation": "READ",
            "storage_type": "HANA",
            "query": {
                "table": "ai_global_analytics",
                "where": {"agent_id": analytics_request.agent_id},
                "order_by": "decision_date DESC",
            },
        }

    elif request.operation == AIDecisionOperation.GET_HISTORY:
        # Query decision history
        history_request = request
        data_manager_data = {
            "operation": "READ",
            "storage_type": "HANA",
            "query": {
                "table": "ai_decision_performance",
                "where": history_request.filters,
                "order_by": "timestamp DESC",
                "limit": history_request.limit,
                "offset": history_request.offset,
            },
        }

    else:
        raise ValueError(f"Unsupported AI decision operation: {request.operation}")

    # Create A2A message parts
    message_parts = [
        MessagePart(kind="text", text=f"AI Decision Logger: {request.operation.value}"),
        MessagePart(kind="data", data=data_manager_data),
        MessagePart(
            kind="data",
            data={
                "ai_decision_metadata": {
                    "operation": request.operation.value,
                    "agent_id": request.agent_id,
                    "timestamp": request.timestamp,
                    "context_id": request.context_id,
                }
            },
        ),
    ]

    return A2AMessage(
        role=MessageRole.AGENT,
        parts=message_parts,
        contextId=request.context_id
        or f"ai_decision_{request.operation.value}_{int(datetime.utcnow().timestamp())}",
        timestamp=request.timestamp,
    )


def parse_data_manager_response_to_ai_decision_response(
    response_data: Dict[str, Any], original_request: AIDecisionRequest
) -> AIDecisionResponse:
    """Parse Data Manager response into AI Decision response"""

    success = response_data.get("overall_status") in ["SUCCESS", "PARTIAL_SUCCESS"]
    error = None if success else response_data.get("error", "Operation failed")

    if original_request.operation == AIDecisionOperation.LOG_DECISION:
        return LogDecisionResponse(
            success=success,
            operation=original_request.operation,
            agent_id=original_request.agent_id,
            context_id=original_request.context_id,
            error=error,
            decision_id=original_request.decision_data.get("decision_id") if success else None,
            storage_location=(
                response_data.get("primary_result", {}).get("location") if success else None
            ),
        )

    if original_request.operation == AIDecisionOperation.LOG_OUTCOME:
        return LogOutcomeResponse(
            success=success,
            operation=original_request.operation,
            agent_id=original_request.agent_id,
            context_id=original_request.context_id,
            error=error,
            decision_id=original_request.outcome_data.get("decision_id") if success else None,
            outcome_recorded=success,
        )

    if original_request.operation == AIDecisionOperation.QUERY_PATTERNS:
        patterns = response_data.get("data", []) if success else []
        return QueryPatternsResponse(
            success=success,
            operation=original_request.operation,
            agent_id=original_request.agent_id,
            context_id=original_request.context_id,
            error=error,
            patterns=patterns,
            total_count=len(patterns),
        )

    if original_request.operation == AIDecisionOperation.GET_ANALYTICS:
        analytics = response_data.get("data", {}) if success else {}
        return GetAnalyticsResponse(
            success=success,
            operation=original_request.operation,
            agent_id=original_request.agent_id,
            context_id=original_request.context_id,
            error=error,
            analytics=analytics,
        )

    if original_request.operation == AIDecisionOperation.GET_HISTORY:
        history = response_data.get("data", []) if success else []
        return GetHistoryResponse(
            success=success,
            operation=original_request.operation,
            agent_id=original_request.agent_id,
            context_id=original_request.context_id,
            error=error,
            history=history,
            total_count=len(history),
            has_more=len(history) >= original_request.limit,
        )

    return AIDecisionResponse(
        success=success,
        operation=original_request.operation,
        agent_id=original_request.agent_id,
        context_id=original_request.context_id,
        error=error,
    )


# Utility functions for creating common requests


def create_decision_log_message(
    agent_id: str,
    decision_id: str,
    decision_type: DecisionType,
    question: str,
    ai_response: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    confidence_score: float = 0.5,
    response_time: float = 0.0,
) -> A2AMessage:
    """Create A2A message to log a decision"""
    request = LogDecisionRequest.create(
        agent_id=agent_id,
        decision_id=decision_id,
        decision_type=decision_type,
        question=question,
        ai_response=ai_response,
        context=context,
        confidence_score=confidence_score,
        response_time=response_time,
    )
    return create_data_manager_message_for_decision_operation(request)


def create_outcome_log_message(
    agent_id: str,
    decision_id: str,
    outcome_status: OutcomeStatus,
    success_metrics: Optional[Dict[str, Any]] = None,
    failure_reason: Optional[str] = None,
    feedback: Optional[str] = None,
    actual_duration: float = 0.0,
) -> A2AMessage:
    """Create A2A message to log an outcome"""
    request = LogOutcomeRequest.create(
        agent_id=agent_id,
        decision_id=decision_id,
        outcome_status=outcome_status,
        success_metrics=success_metrics,
        failure_reason=failure_reason,
        feedback=feedback,
        actual_duration=actual_duration,
    )
    return create_data_manager_message_for_decision_operation(request)


def create_analytics_query_message(
    agent_id: str, days_back: int = 30, include_global: bool = False
) -> A2AMessage:
    """Create A2A message to get analytics"""
    request = GetAnalyticsRequest.create(
        agent_id=agent_id, days_back=days_back, include_global=include_global
    )
    return create_data_manager_message_for_decision_operation(request)


def create_recommendations_query_message(
    agent_id: str,
    decision_type: DecisionType,
    context: Optional[Dict[str, Any]] = None,
    max_recommendations: int = 5,
) -> A2AMessage:
    """Create A2A message to get recommendations"""
    request = GetRecommendationsRequest.create(
        agent_id=agent_id,
        decision_type=decision_type,
        context=context,
        max_recommendations=max_recommendations,
    )
    return create_data_manager_message_for_decision_operation(request)

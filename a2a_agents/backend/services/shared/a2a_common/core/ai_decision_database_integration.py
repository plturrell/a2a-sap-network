# Database Integration Mixin for AI Decision Logger
# Shows how to integrate database-backed AI Decision Logger with existing agents

import asyncio
import logging
from typing import Dict, Any, Optional, List

from .ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
from .ai_decision_logger import DecisionType, OutcomeStatus
from .a2a_types import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class AIDatabaseDecisionIntegrationMixin:
    """Mixin to add database-backed AI decision logging to existing agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get Data Manager URL from agent configuration
        data_manager_url = getattr(self, 'data_manager_url', None)
        if not data_manager_url:
            # Try to construct from base_url or use default
            base_url = getattr(self, 'base_url', 'http://localhost:8000')
            data_manager_url = f"{base_url.replace('/agents/', '/').rstrip('/')}/data-manager"
        
        # Initialize Database AI Decision Logger
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=getattr(self, 'agent_id', 'unknown_agent'),
            data_manager_url=data_manager_url,
            memory_size=1000,
            learning_threshold=10
        )
        
        # Register with global registry for cross-agent learning
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        logger.info(f"Database AI Decision Logger integrated for agent {self.agent_id}")
    
    async def _enhanced_handle_advisor_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Enhanced advisor request handler with database decision logging"""
        
        # Extract question from message
        question = ""
        for part in message.parts:
            if part.kind == "text" and part.text:
                question = part.text
                break
            elif part.kind == "data" and part.data and "question" in part.data:
                question = part.data["question"]
                break
        
        # Get context information
        context = {
            "context_id": context_id,
            "message_parts": len(message.parts),
            "has_data": any(part.kind == "data" for part in message.parts),
            "source_agent_id": getattr(message, 'from_agent_id', 'unknown'),
            "processing_stage": "advisor_request"
        }
        
        # Log the AI decision request
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.ADVISOR_GUIDANCE,
            question=question,
            ai_response={},  # Will update after getting response
            context=context
        )
        
        try:
            # Call original advisor method (assuming it exists)
            if hasattr(self, 'ai_advisor') and self.ai_advisor:
                start_time = asyncio.get_event_loop().time()
                
                advisor_response = await self.ai_advisor.process_a2a_help_message(
                    [part.dict() for part in message.parts],
                    context.get('source_agent_id')
                )
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Update decision with response details (cache only - DB already has initial record)
                if decision_id in self.ai_decision_logger._decision_cache:
                    decision = self.ai_decision_logger._decision_cache[decision_id]
                    decision.ai_response = advisor_response
                    decision.response_time = response_time
                
                # Determine if this was successful based on response
                has_answer = (
                    isinstance(advisor_response, dict) and 
                    advisor_response.get('answer') and 
                    len(advisor_response['answer']) > 10
                )
                
                confidence = advisor_response.get('confidence', 0.5) if isinstance(advisor_response, dict) else 0.5
                
                # Log outcome to database
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.SUCCESS if has_answer else OutcomeStatus.PARTIAL_SUCCESS,
                    success_metrics={
                        "response_length": len(str(advisor_response)),
                        "has_answer": has_answer,
                        "response_time": response_time,
                        "confidence": confidence,
                        "advisor_available": True
                    }
                )
                
                # Get recommendations from database patterns
                recommendations = await self.ai_decision_logger.get_recommendations(
                    DecisionType.ADVISOR_GUIDANCE,
                    context
                )
                
                enhanced_response = {
                    "message_type": "advisor_response",
                    "advisor_response": advisor_response,
                    "agent_id": self.agent_id,
                    "contextId": context_id,
                    "timestamp": asyncio.get_event_loop().time(),
                    "decision_metadata": {
                        "decision_id": decision_id,
                        "recommendations": recommendations,
                        "learning_active": True,
                        "database_backed": True,
                        "pattern_count": len(recommendations)
                    }
                }
                
                return enhanced_response
                
            else:
                # Fallback if no AI advisor
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.FAILURE,
                    failure_reason="No AI advisor available",
                    success_metrics={"advisor_available": False}
                )
                
                return {
                    "message_type": "advisor_error",
                    "error": "AI advisor not available",
                    "agent_id": self.agent_id,
                    "contextId": context_id,
                    "decision_metadata": {
                        "decision_id": decision_id,
                        "database_backed": True
                    }
                }
        
        except Exception as e:
            # Log failure outcome to database
            await self.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.FAILURE,
                failure_reason=str(e),
                success_metrics={"exception_occurred": True, "error_type": type(e).__name__}
            )
            
            logger.error(f"Enhanced advisor request failed: {e}")
            
            return {
                "message_type": "advisor_error",
                "error": str(e),
                "agent_id": self.agent_id,
                "contextId": context_id,
                "decision_metadata": {
                    "decision_id": decision_id,
                    "database_backed": True
                }
            }
    
    async def _enhanced_handle_error_with_help_seeking(
        self, 
        error: Exception, 
        operation_name: str,
        context_id: str
    ) -> Dict[str, Any]:
        """Enhanced error handling with database decision logging"""
        
        context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation_name,
            "context_id": context_id,
            "processing_stage": "error_recovery"
        }
        
        # Log help-seeking decision
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.HELP_REQUEST,
            question=f"Need help with {operation_name} error: {str(error)}",
            ai_response={},  # Will be filled after help response
            context=context
        )
        
        try:
            # Call original help-seeking method if available
            if hasattr(self, 'seek_help_for_error'):
                help_response = await self.seek_help_for_error(error, operation_name, context_id)
                
                # Update decision cache with response
                if decision_id in self.ai_decision_logger._decision_cache:
                    decision = self.ai_decision_logger._decision_cache[decision_id]
                    decision.ai_response = help_response
                
                # Determine success based on help response
                help_received = bool(help_response and help_response.get("advisor_response"))
                resolution_confidence = 0.8 if help_received else 0.2
                
                # Log outcome to database
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.SUCCESS if help_received else OutcomeStatus.FAILURE,
                    success_metrics={
                        "help_received": help_received,
                        "response_quality": len(str(help_response)) if help_response else 0,
                        "error_type": type(error).__name__,
                        "operation": operation_name,
                        "resolution_confidence": resolution_confidence
                    },
                    failure_reason=None if help_received else "No help response received"
                )
                
                return {
                    "success": help_received,
                    "help_response": help_response,
                    "decision_id": decision_id,
                    "database_backed": True
                }
            
            else:
                # No help-seeking capability
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.FAILURE,
                    failure_reason="No help-seeking capability available",
                    success_metrics={"help_seeking_available": False}
                )
                
                return {
                    "success": False,
                    "error": "No help-seeking capability",
                    "decision_id": decision_id,
                    "database_backed": True
                }
                
        except Exception as help_error:
            # Log help-seeking failure
            await self.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.FAILURE,
                failure_reason=f"Help-seeking failed: {str(help_error)}",
                success_metrics={
                    "help_seeking_exception": True,
                    "help_error_type": type(help_error).__name__
                }
            )
            
            logger.error(f"Help-seeking failed: {help_error}")
            
            return {
                "success": False,
                "error": str(help_error),
                "decision_id": decision_id,
                "database_backed": True
            }
    
    async def _log_task_planning_decision(self, task_description: str, plan_result: Dict[str, Any]) -> str:
        """Log task planning decisions to database"""
        
        context = {
            "task_description": task_description,
            "agent_id": self.agent_id,
            "planning_context": "task_execution",
            "plan_complexity": len(str(plan_result))
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.TASK_PLANNING,
            question=f"Planning task: {task_description}",
            ai_response=plan_result or {},
            context=context
        )
        
        # Initial outcome as pending (will be updated when task completes)
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=OutcomeStatus.PENDING,
            success_metrics={"task_planned": True, "pending_execution": True}
        )
        
        return decision_id
    
    async def _update_task_outcome(self, decision_id: str, task_successful: bool, task_duration: float = 0.0):
        """Update task outcome in database after completion"""
        
        outcome_status = OutcomeStatus.SUCCESS if task_successful else OutcomeStatus.FAILURE
        success_metrics = {
            "task_completed": task_successful,
            "execution_duration": task_duration,
            "task_finished": True
        }
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            success_metrics=success_metrics,
            actual_duration=task_duration
        )
    
    async def _log_delegation_decision(self, delegate_agent: str, actions: list, result: Dict[str, Any]) -> str:
        """Log delegation decisions to database"""
        
        context = {
            "delegate_agent": delegate_agent,
            "actions": actions,
            "agent_id": self.agent_id,
            "delegation_context": "agent_collaboration"
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.DELEGATION,
            question=f"Delegating {len(actions)} actions to {delegate_agent}",
            ai_response=result or {},
            context=context
        )
        
        # Determine outcome based on delegation result
        delegation_successful = result and result.get("success")
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=OutcomeStatus.SUCCESS if delegation_successful else OutcomeStatus.FAILURE,
            success_metrics={
                "delegation_successful": delegation_successful,
                "actions_delegated": len(actions),
                "delegate_agent": delegate_agent
            },
            failure_reason=result.get("error") if result and not delegation_successful else None
        )
        
        return decision_id
    
    async def _log_quality_assessment_decision(self, data_quality: Dict[str, Any], assessment_result: Dict[str, Any]) -> str:
        """Log quality assessment decisions to database"""
        
        context = {
            "data_metrics": data_quality,
            "agent_id": self.agent_id,
            "assessment_context": "data_quality"
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.QUALITY_ASSESSMENT,
            question=f"Assessing data quality with {len(data_quality)} metrics",
            ai_response=assessment_result,
            context=context
        )
        
        # Determine success based on quality thresholds
        quality_score = assessment_result.get("overall_score", 0.0)
        quality_passed = quality_score >= 0.7  # 70% threshold
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=OutcomeStatus.SUCCESS if quality_passed else OutcomeStatus.PARTIAL_SUCCESS,
            success_metrics={
                "quality_score": quality_score,
                "quality_passed": quality_passed,
                "metrics_evaluated": len(data_quality)
            }
        )
        
        return decision_id
    
    # Analytics and insights methods
    
    async def get_ai_decision_analytics(self) -> Dict[str, Any]:
        """Get AI decision analytics from database"""
        return await self.ai_decision_logger.get_decision_analytics()
    
    async def get_ai_decision_history(self, decision_type: Optional[DecisionType] = None, limit: int = 10) -> list:
        """Get AI decision history from database"""
        return await self.ai_decision_logger.get_decision_history(decision_type, limit)
    
    async def get_ai_recommendations(self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None) -> list:
        """Get AI recommendations from database patterns"""
        return await self.ai_decision_logger.get_recommendations(decision_type, context)
    
    async def export_ai_insights_report(self) -> Dict[str, Any]:
        """Export comprehensive AI insights report from database"""
        return await self.ai_decision_logger.export_insights_report()
    
    # Database-specific methods
    
    async def get_cross_agent_insights(self) -> Dict[str, Any]:
        """Get insights across all agents from global database registry"""
        global_registry = get_global_database_decision_registry()
        return await global_registry.get_global_insights()
    
    async def query_decision_patterns(self, pattern_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query specific decision patterns from database"""
        try:
            # Use the logger's database connection to query patterns
            from .ai_decision_protocols import QueryPatternsRequest, create_data_manager_message_for_decision_operation
            
            request = QueryPatternsRequest(
                agent_id=self.agent_id,
                query_filters=pattern_filters,
                limit=20
            )
            
            # This would use the Data Manager communication logic
            # For now, return from in-memory cache
            return []
            
        except Exception as e:
            logger.error(f"Failed to query decision patterns: {e}")
            return []
    
    async def get_decision_performance_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Get decision performance trends from database"""
        try:
            analytics = await self.get_ai_decision_analytics()
            
            # Extract trend information
            trends = {
                "period_days": days_back,
                "total_decisions": analytics.get("summary", {}).get("total_decisions", 0),
                "success_rate_trend": analytics.get("success_rates", {}),
                "performance_by_type": analytics.get("by_type", {}),
                "database_backed": True
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {"error": str(e)}
    
    async def shutdown_ai_decision_logger(self):
        """Gracefully shutdown the database AI decision logger"""
        if hasattr(self, 'ai_decision_logger'):
            await self.ai_decision_logger.shutdown()


# Database-specific utility functions

async def initialize_ai_decision_database_schema(data_manager_url: str) -> bool:
    """Initialize AI decision database schema via Data Manager Agent"""
    try:
        import httpx
        
        # Read schema file
        schema_path = "/Users/apple/projects/finsight_cib/backend/app/a2a/core/ai_decision_database_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Create initialization message
        message_parts = [
            MessagePart(kind="text", text="Initialize AI Decision Logger Database Schema"),
            MessagePart(kind="data", data={
                "operation": "EXECUTE_SQL",
                "storage_type": "HANA",
                "sql": schema_sql,
                "service_level": "GOLD"  # Schema creation is critical
            })
        ]
        
        message = A2AMessage(
            role=MessageRole.SYSTEM,
            parts=message_parts,
            contextId=f"ai_decision_schema_init_{int(asyncio.get_event_loop().time())}"
        )
        
        # Send to Data Manager
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{data_manager_url.rstrip('/')}/process",
                json=message.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("overall_status") in ["SUCCESS", "PARTIAL_SUCCESS"]
                logger.info(f"AI Decision schema initialization: {'SUCCESS' if success else 'FAILED'}")
                return success
            else:
                logger.error(f"Schema initialization failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Failed to initialize AI decision database schema: {e}")
        return False


async def migrate_json_data_to_database(
    json_storage_path: str,
    data_manager_url: str,
    agent_id: str
) -> Dict[str, Any]:
    """Migrate existing JSON data to database"""
    try:
        import os
        import json as json_lib
        from .ai_decision_logger import DecisionType, OutcomeStatus
        
        migration_stats = {
            "decisions_migrated": 0,
            "outcomes_migrated": 0,
            "patterns_migrated": 0,
            "errors": []
        }
        
        # Initialize database logger for migration
        db_logger = AIDecisionDatabaseLogger(
            agent_id=agent_id,
            data_manager_url=data_manager_url
        )
        
        # Migrate decisions
        decisions_file = os.path.join(json_storage_path, "decisions.json")
        if os.path.exists(decisions_file):
            with open(decisions_file, 'r') as f:
                decisions_data = json_lib.load(f)
            
            for decision_id, decision_dict in decisions_data.items():
                try:
                    await db_logger.log_decision(
                        decision_type=DecisionType(decision_dict["decision_type"]),
                        question=decision_dict["question"],
                        ai_response=decision_dict["ai_response"],
                        context=decision_dict["context"],
                        confidence_score=decision_dict["confidence_score"],
                        response_time=decision_dict["response_time"]
                    )
                    migration_stats["decisions_migrated"] += 1
                except Exception as e:
                    migration_stats["errors"].append(f"Decision {decision_id}: {str(e)}")
        
        # Migrate outcomes
        outcomes_file = os.path.join(json_storage_path, "outcomes.json")
        if os.path.exists(outcomes_file):
            with open(outcomes_file, 'r') as f:
                outcomes_data = json_lib.load(f)
            
            for decision_id, outcome_dict in outcomes_data.items():
                try:
                    await db_logger.log_outcome(
                        decision_id=decision_id,
                        outcome_status=OutcomeStatus(outcome_dict["outcome_status"]),
                        success_metrics=outcome_dict["success_metrics"],
                        failure_reason=outcome_dict.get("failure_reason"),
                        feedback=outcome_dict.get("feedback"),
                        actual_duration=outcome_dict.get("actual_duration", 0.0)
                    )
                    migration_stats["outcomes_migrated"] += 1
                except Exception as e:
                    migration_stats["errors"].append(f"Outcome {decision_id}: {str(e)}")
        
        await db_logger.shutdown()
        
        logger.info(f"Migration completed: {migration_stats}")
        return migration_stats
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {"error": str(e)}
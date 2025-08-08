# AI Decision Logger Integration Example
# Shows how to integrate AI Decision Logger with existing agents

import asyncio
import logging
from typing import Dict, Any, Optional

from .ai_decision_logger import AIDecisionLogger, DecisionType, OutcomeStatus, get_global_decision_registry
from .a2a_types import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class AIDecisionIntegrationMixin:
    """Mixin to add AI decision logging to existing agents"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize AI Decision Logger
        self.ai_decision_logger = AIDecisionLogger(
            agent_id=getattr(self, 'agent_id', 'unknown_agent'),
            storage_path=None,  # Uses default path
            memory_size=1000,
            learning_threshold=10
        )
        
        # Register with global registry for cross-agent learning
        global_registry = get_global_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Load historical data if available
        asyncio.create_task(self.ai_decision_logger.load_historical_data())
        
        logger.info(f"AI Decision Logger integrated for agent {self.agent_id}")
    
    async def _enhanced_handle_advisor_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Enhanced advisor request handler with decision logging"""
        
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
            "agent_id": getattr(message, 'from_agent_id', 'unknown')
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
                    context.get('agent_id')
                )
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Update decision with response details
                decision = self.ai_decision_logger.decisions[decision_id]
                decision.ai_response = advisor_response
                decision.response_time = response_time
                
                # Determine if this was successful based on response
                has_answer = (
                    isinstance(advisor_response, dict) and 
                    advisor_response.get('answer') and 
                    len(advisor_response['answer']) > 10
                )
                
                # Log immediate outcome (we assume success if we got a substantial answer)
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.SUCCESS if has_answer else OutcomeStatus.PARTIAL_SUCCESS,
                    success_metrics={
                        "response_length": len(str(advisor_response)),
                        "has_answer": has_answer,
                        "response_time": response_time
                    }
                )
                
                # Create enhanced response with recommendations
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
                        "learning_active": True
                    }
                }
                
                return enhanced_response
                
            else:
                # Fallback if no AI advisor
                await self.ai_decision_logger.log_outcome(
                    decision_id=decision_id,
                    outcome_status=OutcomeStatus.FAILURE,
                    failure_reason="No AI advisor available"
                )
                
                return {
                    "message_type": "advisor_error",
                    "error": "AI advisor not available",
                    "agent_id": self.agent_id,
                    "contextId": context_id
                }
        
        except Exception as e:
            # Log failure outcome
            await self.ai_decision_logger.log_outcome(
                decision_id=decision_id,
                outcome_status=OutcomeStatus.FAILURE,
                failure_reason=str(e)
            )
            
            logger.error(f"Enhanced advisor request failed: {e}")
            
            return {
                "message_type": "advisor_error",
                "error": str(e),
                "agent_id": self.agent_id,
                "contextId": context_id
            }
    
    async def _log_help_seeking_decision(self, problem_type: str, helper_agent: str, help_response: Dict[str, Any]) -> str:
        """Log help-seeking decisions"""
        
        context = {
            "problem_type": problem_type,
            "helper_agent": helper_agent,
            "agent_id": self.agent_id
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.HELP_REQUEST,
            question=f"Seeking help for {problem_type}",
            ai_response=help_response or {},
            context=context
        )
        
        # Determine outcome based on help response
        if help_response:
            if "advisor_response" in help_response and help_response["advisor_response"].get("answer"):
                outcome_status = OutcomeStatus.SUCCESS
                success_metrics = {
                    "help_received": True,
                    "response_quality": len(help_response["advisor_response"]["answer"]),
                    "helper_agent": helper_agent
                }
            else:
                outcome_status = OutcomeStatus.PARTIAL_SUCCESS
                success_metrics = {"help_received": True, "response_quality": 0}
        else:
            outcome_status = OutcomeStatus.FAILURE
            success_metrics = {"help_received": False}
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            success_metrics=success_metrics
        )
        
        return decision_id
    
    async def _log_delegation_decision(self, delegate_agent: str, actions: list, result: Dict[str, Any]) -> str:
        """Log delegation decisions"""
        
        context = {
            "delegate_agent": delegate_agent,
            "actions": actions,
            "agent_id": self.agent_id
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.DELEGATION,
            question=f"Delegating {len(actions)} actions to {delegate_agent}",
            ai_response=result or {},
            context=context
        )
        
        # Determine outcome based on delegation result
        if result and result.get("success"):
            outcome_status = OutcomeStatus.SUCCESS
            success_metrics = {
                "delegation_successful": True,
                "actions_delegated": len(actions),
                "delegate_agent": delegate_agent
            }
        else:
            outcome_status = OutcomeStatus.FAILURE
            success_metrics = {
                "delegation_successful": False,
                "error": result.get("error") if result else "No result"
            }
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            success_metrics=success_metrics,
            failure_reason=result.get("error") if result and not result.get("success") else None
        )
        
        return decision_id
    
    async def _log_task_planning_decision(self, task_description: str, plan_result: Dict[str, Any]) -> str:
        """Log task planning decisions"""
        
        context = {
            "task_description": task_description,
            "agent_id": self.agent_id,
            "planning_context": "task_execution"
        }
        
        decision_id = await self.ai_decision_logger.log_decision(
            decision_type=DecisionType.TASK_PLANNING,
            question=f"Planning task: {task_description}",
            ai_response=plan_result or {},
            context=context
        )
        
        # Outcome will be logged later when task completes
        # For now, log as pending
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=OutcomeStatus.PENDING,
            success_metrics={"task_planned": True}
        )
        
        return decision_id
    
    async def _update_task_outcome(self, decision_id: str, task_successful: bool, task_duration: float = 0.0):
        """Update task outcome after completion"""
        
        outcome_status = OutcomeStatus.SUCCESS if task_successful else OutcomeStatus.FAILURE
        success_metrics = {
            "task_completed": task_successful,
            "execution_duration": task_duration
        }
        
        await self.ai_decision_logger.log_outcome(
            decision_id=decision_id,
            outcome_status=outcome_status,
            success_metrics=success_metrics,
            actual_duration=task_duration
        )
    
    def get_ai_decision_analytics(self) -> Dict[str, Any]:
        """Get AI decision analytics for this agent"""
        return self.ai_decision_logger.get_decision_analytics()
    
    def get_ai_decision_history(self, decision_type: Optional[DecisionType] = None, limit: int = 10) -> list:
        """Get AI decision history for this agent"""
        return self.ai_decision_logger.get_decision_history(decision_type, limit)
    
    async def get_ai_recommendations(self, decision_type: DecisionType, context: Optional[Dict[str, Any]] = None) -> list:
        """Get AI recommendations based on learned patterns"""
        return await self.ai_decision_logger.get_recommendations(decision_type, context)
    
    async def export_ai_insights_report(self) -> Dict[str, Any]:
        """Export comprehensive AI insights report"""
        return await self.ai_decision_logger.export_insights_report()


# Example of how to modify an existing agent to include AI decision logging
class ExampleEnhancedAgent(AIDecisionIntegrationMixin):
    """Example showing how to enhance existing agent with AI decision logging"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__()
    
    async def process_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Example message processing with decision logging"""
        
        # Check if this is an advisor request
        if self._is_advisor_request(message):
            return await self._enhanced_handle_advisor_request(message, context_id)
        
        # Log task planning decision
        task_description = f"Processing message with {len(message.parts)} parts"
        planning_decision_id = await self._log_task_planning_decision(task_description, {"planned": True})
        
        try:
            # Simulate message processing
            await asyncio.sleep(0.1)  # Simulate work
            
            # Update task outcome as successful
            await self._update_task_outcome(planning_decision_id, True, 0.1)
            
            return {
                "status": "success",
                "message": "Message processed successfully",
                "decision_id": planning_decision_id
            }
            
        except Exception as e:
            # Update task outcome as failed
            await self._update_task_outcome(planning_decision_id, False, 0.1)
            
            return {
                "status": "error",
                "error": str(e),
                "decision_id": planning_decision_id
            }
    
    def _is_advisor_request(self, message: A2AMessage) -> bool:
        """Check if message is requesting AI advisor help"""
        for part in message.parts:
            if part.kind == "text" and part.text:
                text_lower = part.text.lower()
                if any(word in text_lower for word in ["help", "advisor", "question", "how", "what", "explain"]):
                    return True
            elif part.kind == "data" and part.data:
                if "advisor_request" in part.data or "help_request" in part.data:
                    return True
        return False


# Utility functions for easy integration

async def log_ai_advisor_interaction(
    logger: AIDecisionLogger, 
    question: str, 
    response: Dict[str, Any], 
    context: Dict[str, Any] = None
) -> str:
    """Utility function to log AI advisor interactions"""
    
    decision_id = await logger.log_decision(
        decision_type=DecisionType.ADVISOR_GUIDANCE,
        question=question,
        ai_response=response,
        context=context or {}
    )
    
    # Determine success based on response quality
    has_substantial_answer = (
        isinstance(response, dict) and 
        response.get('answer') and 
        len(response['answer']) > 20
    )
    
    await logger.log_outcome(
        decision_id=decision_id,
        outcome_status=OutcomeStatus.SUCCESS if has_substantial_answer else OutcomeStatus.PARTIAL_SUCCESS,
        success_metrics={
            "response_quality": len(str(response)),
            "has_answer": has_substantial_answer
        }
    )
    
    return decision_id


async def log_help_seeking_interaction(
    logger: AIDecisionLogger,
    problem_type: str,
    help_response: Dict[str, Any],
    context: Dict[str, Any] = None
) -> str:
    """Utility function to log help-seeking interactions"""
    
    decision_id = await logger.log_decision(
        decision_type=DecisionType.HELP_REQUEST,
        question=f"Seeking help for {problem_type}",
        ai_response=help_response,
        context=context or {"problem_type": problem_type}
    )
    
    # Determine success based on help received
    help_received = bool(help_response and help_response.get("advisor_response"))
    
    await logger.log_outcome(
        decision_id=decision_id,
        outcome_status=OutcomeStatus.SUCCESS if help_received else OutcomeStatus.FAILURE,
        success_metrics={
            "help_received": help_received,
            "problem_type": problem_type
        }
    )
    
    return decision_id


# Integration instructions for existing agents
"""
To integrate AI Decision Logger with existing agents:

1. Add the import:
   from ..core.ai_decision_logger import AIDecisionLogger, DecisionType, OutcomeStatus

2. Initialize in agent __init__:
   self.ai_decision_logger = AIDecisionLogger(agent_id=self.agent_id)

3. Log advisor interactions:
   decision_id = await log_ai_advisor_interaction(
       self.ai_decision_logger, 
       question, 
       advisor_response, 
       context
   )

4. Log help-seeking:
   decision_id = await log_help_seeking_interaction(
       self.ai_decision_logger,
       problem_type,
       help_response,
       context
   )

5. Get analytics:
   analytics = self.ai_decision_logger.get_decision_analytics()

6. Get recommendations:
   recommendations = await self.ai_decision_logger.get_recommendations(
       DecisionType.ADVISOR_GUIDANCE, 
       context
   )

This provides immediate value with minimal code changes to existing agents.
"""
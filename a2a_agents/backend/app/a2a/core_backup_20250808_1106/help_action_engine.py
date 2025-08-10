"""
A2A Help Action Engine
Enables agents to parse help responses and take concrete actions based on the advice received
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    RETRY_OPERATION = "retry_operation"
    UPDATE_CONFIG = "update_config"
    CHANGE_STRATEGY = "change_strategy"
    MODIFY_PARAMETERS = "modify_parameters"
    USE_ALTERNATIVE = "use_alternative"
    WAIT_AND_RETRY = "wait_and_retry"
    ESCALATE_ISSUE = "escalate_issue"
    DISABLE_FEATURE = "disable_feature"


class RetryStrategy(str, Enum):
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    ALTERNATIVE_ENDPOINT = "alternative_endpoint"
    FALLBACK_METHOD = "fallback_method"


class HelpAction(BaseModel):
    """Represents a specific action to take based on help advice"""
    action_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 1  # 1 = highest priority
    prerequisites: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    rollback_possible: bool = True


class RetryInstructions(BaseModel):
    """Specific retry instructions from help advice"""
    strategy: RetryStrategy
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    alternative_endpoint: Optional[str] = None
    modified_parameters: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)


class ConfigUpdate(BaseModel):
    """Configuration changes suggested by help"""
    config_path: str  # e.g., "database.timeout", "retry.max_attempts"
    new_value: Any
    old_value: Optional[Any] = None
    temporary: bool = False  # If True, revert after operation
    scope: str = "operation"  # "operation", "session", "permanent"


class ActionPlan(BaseModel):
    """Complete action plan derived from help response"""
    plan_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    problem_type: str
    help_source: str
    actions: List[HelpAction] = Field(default_factory=list)
    retry_instructions: Optional[RetryInstructions] = None
    config_updates: List[ConfigUpdate] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    estimated_time_seconds: Optional[int] = None
    risk_level: str = "low"  # "low", "medium", "high"


class HelpResponseParser:
    """Parses help responses from AI advisors into actionable plans"""
    
    def __init__(self):
        self.action_patterns = {
            # Network/Connection issues
            "connection": {
                "timeout": ActionType.UPDATE_CONFIG,
                "retry": ActionType.RETRY_OPERATION,
                "alternative": ActionType.USE_ALTERNATIVE,
                "endpoint": ActionType.CHANGE_STRATEGY
            },
            # Service availability
            "service": {
                "unavailable": ActionType.WAIT_AND_RETRY,
                "restart": ActionType.ESCALATE_ISSUE,
                "alternative": ActionType.USE_ALTERNATIVE
            },
            # Data integrity
            "integrity": {
                "checksum": ActionType.RETRY_OPERATION,
                "validation": ActionType.MODIFY_PARAMETERS,
                "corruption": ActionType.USE_ALTERNATIVE
            },
            # Performance issues
            "performance": {
                "timeout": ActionType.UPDATE_CONFIG,
                "memory": ActionType.MODIFY_PARAMETERS,
                "optimization": ActionType.CHANGE_STRATEGY
            }
        }
    
    def parse_advisor_response(self, help_response: Dict[str, Any]) -> ActionPlan:
        """Parse AI advisor response into concrete action plan"""
        try:
            action_plan = ActionPlan(
                problem_type=help_response.get("problem_type", "unknown"),
                help_source=help_response.get("source_agent", "unknown_advisor")
            )
            
            # Extract advisor response content
            advisor_content = help_response.get("advisor_response", {})
            if isinstance(advisor_content, dict):
                advisor_response = advisor_content.get("advisor_response", {})
                answer_text = advisor_response.get("answer", "")
                context_used = advisor_response.get("context_used", {})
                
                # Parse actions from answer text
                action_plan.actions = self._extract_actions_from_text(answer_text, context_used)
                
                # Parse retry instructions
                action_plan.retry_instructions = self._extract_retry_instructions(answer_text)
                
                # Parse configuration updates
                action_plan.config_updates = self._extract_config_updates(answer_text)
                
                # Set success criteria
                action_plan.success_criteria = self._extract_success_criteria(answer_text)
                
                # Assess risk level
                action_plan.risk_level = self._assess_risk_level(action_plan.actions)
            
            logger.info(f"📋 Parsed help response into action plan with {len(action_plan.actions)} actions")
            return action_plan
            
        except Exception as e:
            logger.error(f"❌ Error parsing help response: {e}")
            # Return minimal action plan for fallback
            return ActionPlan(
                problem_type="parse_error",
                help_source="parser_fallback",
                actions=[
                    HelpAction(
                        action_type=ActionType.RETRY_OPERATION,
                        description="Fallback: retry original operation",
                        priority=1
                    )
                ]
            )
    
    def _extract_actions_from_text(self, answer_text: str, context: Dict[str, Any]) -> List[HelpAction]:
        """Extract specific actions from advisor response text"""
        actions = []
        answer_lower = answer_text.lower()
        
        # Check for retry instructions
        if any(word in answer_lower for word in ["retry", "try again", "attempt again"]):
            actions.append(HelpAction(
                action_type=ActionType.RETRY_OPERATION,
                description="Retry the failed operation",
                priority=1
            ))
        
        # Check for timeout adjustments
        if any(word in answer_lower for word in ["timeout", "increase timeout", "extend timeout"]):
            timeout_value = self._extract_number_from_text(answer_text, "timeout")
            actions.append(HelpAction(
                action_type=ActionType.UPDATE_CONFIG,
                description="Increase timeout value",
                parameters={"config_path": "timeout", "new_value": timeout_value or 60},
                priority=2
            ))
        
        # Check for connection issues
        if any(word in answer_lower for word in ["connection", "connectivity", "network"]):
            actions.append(HelpAction(
                action_type=ActionType.CHANGE_STRATEGY,
                description="Use alternative connection strategy",
                parameters={"strategy": "fallback_connection"},
                priority=2
            ))
        
        # Check for service restart recommendations
        if any(word in answer_lower for word in ["restart", "reboot", "reset service"]):
            actions.append(HelpAction(
                action_type=ActionType.ESCALATE_ISSUE,
                description="Request service restart",
                parameters={"escalation_type": "service_restart"},
                priority=3
            ))
        
        # Check for alternative approaches
        if any(word in answer_lower for word in ["alternative", "different approach", "workaround"]):
            actions.append(HelpAction(
                action_type=ActionType.USE_ALTERNATIVE,
                description="Use alternative approach",
                parameters={"approach": "fallback_method"},
                priority=2
            ))
        
        # Check for parameter modifications
        if any(word in answer_lower for word in ["parameter", "setting", "configuration"]):
            actions.append(HelpAction(
                action_type=ActionType.MODIFY_PARAMETERS,
                description="Modify operation parameters",
                parameters={"scope": "current_operation"},
                priority=2
            ))
        
        # Extract actions from relevant issues in context
        relevant_issues = context.get("relevant_issues", [])
        for issue in relevant_issues:
            if isinstance(issue, dict) and issue.get("relevance_score", 0) >= 3:
                solution = issue.get("solution", "")
                if solution:
                    action = self._parse_solution_text(solution, issue.get("type", "unknown"))
                    if action:
                        actions.append(action)
        
        # If no specific actions found, add a generic retry
        if not actions:
            actions.append(HelpAction(
                action_type=ActionType.RETRY_OPERATION,
                description="Retry operation with default parameters",
                priority=1
            ))
        
        return actions
    
    def _extract_retry_instructions(self, answer_text: str) -> Optional[RetryInstructions]:
        """Extract retry strategy from help text"""
        answer_lower = answer_text.lower()
        
        strategy = RetryStrategy.IMMEDIATE
        max_attempts = 3
        delay_seconds = 1.0
        
        # Determine retry strategy
        if "exponential" in answer_lower or "backoff" in answer_lower:
            strategy = RetryStrategy.EXPONENTIAL_BACKOFF
            delay_seconds = 2.0
        elif "delay" in answer_lower or "wait" in answer_lower:
            strategy = RetryStrategy.FIXED_DELAY
            delay_seconds = self._extract_number_from_text(answer_text, "delay") or 5.0
        elif "alternative" in answer_lower or "different" in answer_lower:
            strategy = RetryStrategy.ALTERNATIVE_ENDPOINT
        
        # Extract retry count
        max_attempts = self._extract_number_from_text(answer_text, "attempt") or 3
        
        return RetryInstructions(
            strategy=strategy,
            max_attempts=max_attempts,
            delay_seconds=delay_seconds,
            backoff_multiplier=2.0
        )
    
    def _extract_config_updates(self, answer_text: str) -> List[ConfigUpdate]:
        """Extract configuration changes from help text"""
        updates = []
        answer_lower = answer_text.lower()
        
        # Extract timeout updates
        if "timeout" in answer_lower:
            timeout_value = self._extract_number_from_text(answer_text, "timeout")
            if timeout_value:
                updates.append(ConfigUpdate(
                    config_path="operation.timeout",
                    new_value=timeout_value,
                    temporary=True,
                    scope="operation"
                ))
        
        # Extract retry count updates
        if "retry" in answer_lower and "count" in answer_lower:
            retry_count = self._extract_number_from_text(answer_text, "retry")
            if retry_count:
                updates.append(ConfigUpdate(
                    config_path="retry.max_attempts",
                    new_value=retry_count,
                    temporary=True,
                    scope="operation"
                ))
        
        return updates
    
    def _extract_success_criteria(self, answer_text: str) -> Dict[str, Any]:
        """Extract success criteria from help text"""
        criteria = {}
        answer_lower = answer_text.lower()
        
        if "success" in answer_lower:
            criteria["check_success_response"] = True
        
        if "error" in answer_lower and "no error" in answer_lower:
            criteria["no_errors"] = True
        
        if "complete" in answer_lower:
            criteria["operation_completed"] = True
        
        return criteria
    
    def _parse_solution_text(self, solution_text: str, issue_type: str) -> Optional[HelpAction]:
        """Parse solution text into specific action"""
        solution_lower = solution_text.lower()
        
        if "check" in solution_lower:
            return HelpAction(
                action_type=ActionType.MODIFY_PARAMETERS,
                description=f"Check and verify {issue_type} parameters",
                parameters={"verification_type": issue_type}
            )
        elif "restart" in solution_lower:
            return HelpAction(
                action_type=ActionType.ESCALATE_ISSUE,
                description=f"Request restart for {issue_type}",
                parameters={"escalation_type": "restart", "component": issue_type}
            )
        elif "retry" in solution_lower:
            return HelpAction(
                action_type=ActionType.RETRY_OPERATION,
                description=f"Retry operation for {issue_type}",
                parameters={"focus": issue_type}
            )
        
        return None
    
    def _extract_number_from_text(self, text: str, context: str) -> Optional[float]:
        """Extract numeric values from text near context words"""
        import re
        
        # Look for numbers near the context word
        pattern = rf"{context}[:\s]*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
        
        # Look for standalone numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            return float(numbers[0])
        
        return None
    
    def _assess_risk_level(self, actions: List[HelpAction]) -> str:
        """Assess risk level of action plan"""
        high_risk_actions = [ActionType.ESCALATE_ISSUE, ActionType.DISABLE_FEATURE]
        medium_risk_actions = [ActionType.UPDATE_CONFIG, ActionType.CHANGE_STRATEGY]
        
        for action in actions:
            if action.action_type in high_risk_actions:
                return "high"
            elif action.action_type in medium_risk_actions:
                return "medium"
        
        return "low"


class HelpActionExecutor:
    """Executes action plans derived from help responses"""
    
    def __init__(self, agent_context: Dict[str, Any]):
        self.agent_context = agent_context
        self.execution_history = []
        self.config_rollbacks = []
    
    async def execute_action_plan(
        self, 
        action_plan: ActionPlan, 
        original_operation: Optional[Callable] = None,
        operation_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute complete action plan and return results"""
        
        execution_result = {
            "plan_id": action_plan.plan_id,
            "success": False,
            "actions_executed": [],
            "failed_actions": [],
            "final_outcome": None,
            "execution_time": None,
            "rollbacks_performed": []
        }
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"🚀 Executing action plan {action_plan.plan_id} with {len(action_plan.actions)} actions")
            
            # Sort actions by priority
            sorted_actions = sorted(action_plan.actions, key=lambda x: x.priority)
            
            # Execute configuration updates first
            for config_update in action_plan.config_updates:
                await self._apply_config_update(config_update)
            
            # Execute actions in priority order
            for action in sorted_actions:
                try:
                    action_result = await self._execute_single_action(
                        action, 
                        original_operation, 
                        operation_context or {}
                    )
                    
                    execution_result["actions_executed"].append({
                        "action_id": action.action_id,
                        "action_type": action.action_type,
                        "success": action_result["success"],
                        "result": action_result
                    })
                    
                    # If this action resolved the issue, we can stop
                    if action_result["success"] and action_result.get("resolved_issue"):
                        execution_result["success"] = True
                        execution_result["final_outcome"] = action_result
                        break
                        
                except Exception as action_error:
                    logger.error(f"❌ Action {action.action_id} failed: {action_error}")
                    execution_result["failed_actions"].append({
                        "action_id": action.action_id,
                        "error": str(action_error)
                    })
            
            # If we have retry instructions and nothing else worked, try the retry
            if not execution_result["success"] and action_plan.retry_instructions:
                retry_result = await self._execute_retry_with_instructions(
                    action_plan.retry_instructions,
                    original_operation,
                    operation_context or {}
                )
                
                execution_result["final_outcome"] = retry_result
                execution_result["success"] = retry_result.get("success", False)
            
            execution_result["execution_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Action plan execution completed. Success: {execution_result['success']}")
            
        except Exception as e:
            logger.error(f"❌ Action plan execution failed: {e}")
            execution_result["execution_error"] = str(e)
            
            # Rollback any temporary changes
            await self._rollback_temporary_changes()
            
        finally:
            self.execution_history.append(execution_result)
        
        return execution_result
    
    async def _execute_single_action(
        self, 
        action: HelpAction, 
        original_operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single action and return result"""
        
        logger.info(f"🔧 Executing action: {action.action_type} - {action.description}")
        
        try:
            if action.action_type == ActionType.RETRY_OPERATION:
                return await self._execute_retry_action(action, original_operation, context)
            
            elif action.action_type == ActionType.UPDATE_CONFIG:
                return await self._execute_config_update_action(action, context)
            
            elif action.action_type == ActionType.CHANGE_STRATEGY:
                return await self._execute_strategy_change_action(action, context)
            
            elif action.action_type == ActionType.MODIFY_PARAMETERS:
                return await self._execute_parameter_modification_action(action, context)
            
            elif action.action_type == ActionType.USE_ALTERNATIVE:
                return await self._execute_alternative_action(action, original_operation, context)
            
            elif action.action_type == ActionType.WAIT_AND_RETRY:
                return await self._execute_wait_and_retry_action(action, original_operation, context)
            
            elif action.action_type == ActionType.ESCALATE_ISSUE:
                return await self._execute_escalation_action(action, context)
            
            elif action.action_type == ActionType.DISABLE_FEATURE:
                return await self._execute_disable_feature_action(action, context)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action.action_type}",
                    "action_id": action.action_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_id": action.action_id
            }
    
    async def _execute_retry_action(
        self, 
        action: HelpAction, 
        original_operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute retry operation action"""
        
        if not original_operation:
            return {
                "success": False,
                "error": "No original operation provided for retry",
                "action_id": action.action_id
            }
        
        try:
            # Apply any focus or parameters from the action
            modified_context = context.copy()
            if "focus" in action.parameters:
                modified_context["focus"] = action.parameters["focus"]
            
            # Retry the original operation
            if asyncio.iscoroutinefunction(original_operation):
                result = await original_operation(**modified_context)
            else:
                result = original_operation(**modified_context)
            
            return {
                "success": True,
                "resolved_issue": True,
                "result": result,
                "action_id": action.action_id,
                "description": "Successfully retried original operation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_id": action.action_id,
                "description": "Retry operation failed"
            }
    
    async def _execute_config_update_action(
        self, 
        action: HelpAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute configuration update action"""
        
        try:
            config_path = action.parameters.get("config_path")
            new_value = action.parameters.get("new_value")
            
            if not config_path:
                return {
                    "success": False,
                    "error": "No config_path specified",
                    "action_id": action.action_id
                }
            
            # Store old value for rollback
            old_value = context.get(config_path)
            self.config_rollbacks.append({
                "path": config_path,
                "old_value": old_value,
                "new_value": new_value
            })
            
            # Apply configuration change
            context[config_path] = new_value
            
            logger.info(f"📝 Updated config {config_path}: {old_value} → {new_value}")
            
            return {
                "success": True,
                "action_id": action.action_id,
                "description": f"Updated {config_path} to {new_value}",
                "config_change": {"path": config_path, "old_value": old_value, "new_value": new_value}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_id": action.action_id
            }
    
    async def _execute_strategy_change_action(
        self, 
        action: HelpAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute strategy change action"""
        
        strategy = action.parameters.get("strategy", "default")
        
        # Update context with new strategy
        context["execution_strategy"] = strategy
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Changed strategy to {strategy}",
            "strategy_change": strategy
        }
    
    async def _execute_parameter_modification_action(
        self, 
        action: HelpAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute parameter modification action"""
        
        scope = action.parameters.get("scope", "general")
        verification_type = action.parameters.get("verification_type")
        
        # Modify parameters based on scope
        if scope == "current_operation":
            context["modified_by_help"] = True
            context["modification_scope"] = scope
        
        if verification_type:
            context["verification_focus"] = verification_type
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Modified parameters for {scope}",
            "parameter_changes": action.parameters
        }
    
    async def _execute_alternative_action(
        self, 
        action: HelpAction, 
        original_operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute alternative approach action"""
        
        approach = action.parameters.get("approach", "fallback_method")
        
        # Mark context for alternative approach
        context["use_alternative"] = True
        context["alternative_approach"] = approach
        
        # Try alternative if original operation is available
        if original_operation:
            try:
                if asyncio.iscoroutinefunction(original_operation):
                    result = await original_operation(**context)
                else:
                    result = original_operation(**context)
                
                return {
                    "success": True,
                    "resolved_issue": True,
                    "result": result,
                    "action_id": action.action_id,
                    "description": f"Successfully used alternative approach: {approach}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "action_id": action.action_id,
                    "description": f"Alternative approach {approach} failed"
                }
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Configured for alternative approach: {approach}",
            "alternative_configured": approach
        }
    
    async def _execute_wait_and_retry_action(
        self, 
        action: HelpAction, 
        original_operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute wait and retry action"""
        
        wait_seconds = action.parameters.get("wait_seconds", 5)
        
        logger.info(f"⏳ Waiting {wait_seconds} seconds before retry...")
        await asyncio.sleep(wait_seconds)
        
        # Try the operation again
        if original_operation:
            try:
                if asyncio.iscoroutinefunction(original_operation):
                    result = await original_operation(**context)
                else:
                    result = original_operation(**context)
                
                return {
                    "success": True,
                    "resolved_issue": True,
                    "result": result,
                    "action_id": action.action_id,
                    "description": f"Successfully retried after {wait_seconds}s wait"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "action_id": action.action_id,
                    "description": f"Retry failed after {wait_seconds}s wait"
                }
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Waited {wait_seconds} seconds as advised"
        }
    
    async def _execute_escalation_action(
        self, 
        action: HelpAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute escalation action"""
        
        escalation_type = action.parameters.get("escalation_type", "general")
        component = action.parameters.get("component", "unknown")
        
        # Log escalation (in real implementation, this would trigger alerts)
        logger.warning(f"🚨 ESCALATION: {escalation_type} for {component}")
        
        # Mark context for escalation
        context["escalated"] = True
        context["escalation_type"] = escalation_type
        context["escalation_component"] = component
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Escalated {escalation_type} for {component}",
            "escalation": {"type": escalation_type, "component": component}
        }
    
    async def _execute_disable_feature_action(
        self, 
        action: HelpAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute disable feature action"""
        
        feature = action.parameters.get("feature", "unknown")
        
        # Mark feature as disabled
        context["disabled_features"] = context.get("disabled_features", [])
        context["disabled_features"].append(feature)
        
        logger.warning(f"⚠️ Disabled feature: {feature}")
        
        return {
            "success": True,
            "action_id": action.action_id,
            "description": f"Disabled feature: {feature}",
            "disabled_feature": feature
        }
    
    async def _execute_retry_with_instructions(
        self, 
        retry_instructions: RetryInstructions,
        original_operation: Optional[Callable],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute retry with specific instructions"""
        
        if not original_operation:
            return {
                "success": False,
                "error": "No original operation for retry instructions"
            }
        
        max_attempts = retry_instructions.max_attempts
        delay_seconds = retry_instructions.delay_seconds
        backoff_multiplier = retry_instructions.backoff_multiplier
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"🔄 Retry attempt {attempt + 1}/{max_attempts}")
                
                # Apply modified parameters if any
                modified_context = context.copy()
                modified_context.update(retry_instructions.modified_parameters)
                
                # Execute operation
                if asyncio.iscoroutinefunction(original_operation):
                    result = await original_operation(**modified_context)
                else:
                    result = original_operation(**modified_context)
                
                logger.info(f"✅ Retry successful on attempt {attempt + 1}")
                return {
                    "success": True,
                    "resolved_issue": True,
                    "result": result,
                    "attempts_used": attempt + 1,
                    "strategy": retry_instructions.strategy
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Retry attempt {attempt + 1} failed: {e}")
                
                # Wait before next attempt (unless it's the last one)
                if attempt < max_attempts - 1:
                    if retry_instructions.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        wait_time = delay_seconds * (backoff_multiplier ** attempt)
                    else:
                        wait_time = delay_seconds
                    
                    logger.info(f"⏳ Waiting {wait_time}s before next retry...")
                    await asyncio.sleep(wait_time)
        
        return {
            "success": False,
            "error": f"All {max_attempts} retry attempts failed",
            "attempts_used": max_attempts,
            "strategy": retry_instructions.strategy
        }
    
    async def _apply_config_update(self, config_update: ConfigUpdate):
        """Apply a configuration update"""
        # Store for rollback if temporary
        if config_update.temporary:
            self.config_rollbacks.append({
                "path": config_update.config_path,
                "old_value": config_update.old_value,
                "new_value": config_update.new_value,
                "temporary": True
            })
        
        # Apply the configuration change to agent context
        if config_update.config_path in self.agent_context:
            config_update.old_value = self.agent_context[config_update.config_path]
        
        self.agent_context[config_update.config_path] = config_update.new_value
        
        logger.info(f"📝 Applied config update: {config_update.config_path} = {config_update.new_value}")
    
    async def _rollback_temporary_changes(self):
        """Rollback temporary configuration changes"""
        for rollback in self.config_rollbacks:
            if rollback.get("temporary", False):
                path = rollback["path"]
                old_value = rollback["old_value"]
                
                if old_value is not None:
                    self.agent_context[path] = old_value
                elif path in self.agent_context:
                    del self.agent_context[path]
                
                logger.info(f"🔄 Rolled back temporary config: {path}")
        
        self.config_rollbacks.clear()


class AgentHelpActionSystem:
    """Complete help action system for A2A agents"""
    
    def __init__(self, agent_id: str, agent_context: Dict[str, Any]):
        self.agent_id = agent_id
        self.parser = HelpResponseParser()
        self.executor = HelpActionExecutor(agent_context)
        self.action_history = []
    
    async def process_help_and_execute_actions(
        self, 
        help_response: Dict[str, Any],
        original_operation: Optional[Callable] = None,
        operation_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Complete pipeline: parse help response and execute actions"""
        
        try:
            # Parse help response into action plan
            action_plan = self.parser.parse_advisor_response(help_response)
            
            # Execute action plan
            execution_result = await self.executor.execute_action_plan(
                action_plan, 
                original_operation, 
                operation_context
            )
            
            # Store in history
            self.action_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "help_response": help_response,
                "action_plan": action_plan.model_dump(),
                "execution_result": execution_result
            })
            
            return {
                "success": execution_result["success"],
                "action_plan_id": action_plan.plan_id,
                "actions_executed": len(execution_result["actions_executed"]),
                "final_outcome": execution_result["final_outcome"],
                "execution_time": execution_result["execution_time"],
                "resolved_issue": execution_result.get("final_outcome", {}).get("resolved_issue", False)
            }
            
        except Exception as e:
            logger.error(f"❌ Help action processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "resolved_issue": False
            }
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get history of help actions taken"""
        return self.action_history.copy()
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about help action effectiveness"""
        if not self.action_history:
            return {
                "total_help_processed": 0,
                "success_rate": 0.0,
                "average_actions_per_help": 0.0,
                "issue_resolution_rate": 0.0
            }
        
        total_processed = len(self.action_history)
        successful_executions = sum(1 for h in self.action_history if h["execution_result"]["success"])
        resolved_issues = sum(1 for h in self.action_history 
                            if h["execution_result"].get("final_outcome", {}).get("resolved_issue", False))
        
        total_actions = sum(len(h["execution_result"]["actions_executed"]) for h in self.action_history)
        
        return {
            "total_help_processed": total_processed,
            "success_rate": (successful_executions / total_processed) * 100,
            "average_actions_per_help": total_actions / total_processed,
            "issue_resolution_rate": (resolved_issues / total_processed) * 100,
            "total_actions_executed": total_actions
        }
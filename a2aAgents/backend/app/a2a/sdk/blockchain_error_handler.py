"""
AI-Enhanced Blockchain Error Handler for A2A Agents
Provides intelligent error handling and recovery for blockchain operations
Enhanced with Grok AI for smart error analysis and recovery decisions
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# A2A Protocol Compliance: Import Grok AI for intelligent error handling
try:
    from ..ai.grokClient import GrokClient
    GROK_AVAILABLE = True
except ImportError:
    try:
        from app.a2a.ai.grokClient import GrokClient
        GROK_AVAILABLE = True
    except ImportError:
        GROK_AVAILABLE = False
        GrokClient = None
        logging.warning("GrokClient not available - using basic error handling without AI")

logger = logging.getLogger(__name__)


class BlockchainErrorType(Enum):
    """Types of blockchain errors"""
    CONNECTION_ERROR = "connection_error"
    TRANSACTION_FAILED = "transaction_failed"
    INSUFFICIENT_GAS = "insufficient_gas"
    NONCE_ERROR = "nonce_error"
    CONTRACT_ERROR = "contract_error"
    TIMEOUT_ERROR = "timeout_error"
    REGISTRATION_FAILED = "registration_failed"
    MESSAGE_SEND_FAILED = "message_send_failed"
    REPUTATION_ERROR = "reputation_error"


@dataclass
class BlockchainError:
    """Blockchain error details"""
    error_type: BlockchainErrorType
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True


class BlockchainErrorHandler:
    """
    Comprehensive error handler for blockchain operations
    Provides retry logic, fallback mechanisms, and recovery strategies
    """

    def __init__(self, agent_name: str, max_retries: int = 3):
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.error_history: List[BlockchainError] = []
        self.recovery_strategies: Dict[BlockchainErrorType, Callable] = {}
        self.circuit_breaker_open = False
        self.circuit_breaker_timeout = None

        # Initialize Grok AI for intelligent error analysis
        self.grok_client = None
        if GROK_AVAILABLE:
            try:
                self.grok_client = GrokClient()
                logger.info(f"✅ Grok AI initialized for {agent_name} blockchain error handling")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok AI for {agent_name}: {e}")
                self.grok_client = None
        else:
            logger.info(f"📋 Using basic error handling for {agent_name} (Grok AI not available)")

        # AI-enhanced metrics
        self.ai_decisions = []
        self.ai_success_rate = 0.0
        self.learning_enabled = True

        # Initialize default recovery strategies
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Setup default recovery strategies for common errors"""
        self.recovery_strategies = {
            BlockchainErrorType.CONNECTION_ERROR: self._handle_connection_error,
            BlockchainErrorType.TRANSACTION_FAILED: self._handle_transaction_error,
            BlockchainErrorType.INSUFFICIENT_GAS: self._handle_gas_error,
            BlockchainErrorType.NONCE_ERROR: self._handle_nonce_error,
            BlockchainErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            BlockchainErrorType.REGISTRATION_FAILED: self._handle_registration_error
        }

    async def handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        AI-Enhanced blockchain error handling with intelligent recovery decisions

        Args:
            error: The exception that occurred
            operation: The operation being performed
            context: Additional context about the operation

        Returns:
            Recovery result if successful, None otherwise
        """
        # Classify the error (enhanced with AI if available)
        error_type = await self._ai_classify_error(error, operation, context)

        # Create error record
        blockchain_error = BlockchainError(
            error_type=error_type,
            message=str(error),
            details={
                "operation": operation,
                "context": context or {},
                "agent": self.agent_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Add to history
        self.error_history.append(blockchain_error)

        # AI-powered recovery decision
        recovery_plan = await self._ai_generate_recovery_plan(blockchain_error)

        # Check circuit breaker (AI can influence this decision)
        if self.circuit_breaker_open:
            if self.circuit_breaker_timeout and datetime.utcnow() > self.circuit_breaker_timeout:
                # Reset circuit breaker
                self.circuit_breaker_open = False
                self.circuit_breaker_timeout = None
                logger.info(f"🔄 Circuit breaker reset for {self.agent_name}")
            else:
                # Check if AI recommends forcing through despite circuit breaker
                if recovery_plan.get("force_execution", False):
                    logger.warning(f"🤖 AI overriding circuit breaker for critical operation: {operation}")
                else:
                    logger.warning(f"⚠️ Circuit breaker open for {self.agent_name}, rejecting operation")
                    return None

        # Log the error with AI analysis
        logger.error(f"🔍 Blockchain error in {self.agent_name}: {error_type.value} - {error}")
        if recovery_plan.get("ai_analysis"):
            logger.info(f"🤖 AI Analysis: {recovery_plan['ai_analysis']}")

        # Apply AI-recommended recovery strategy
        recovery_result = await self._execute_ai_recovery_plan(recovery_plan, blockchain_error)

        # Learn from the outcome
        await self._ai_learn_from_outcome(blockchain_error, recovery_plan, recovery_result)

        return recovery_result

    def _classify_error(self, error: Exception) -> BlockchainErrorType:
        """Classify the error type based on exception"""
        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            return BlockchainErrorType.CONNECTION_ERROR
        elif "nonce" in error_str:
            return BlockchainErrorType.NONCE_ERROR
        elif "gas" in error_str or "out of gas" in error_str:
            return BlockchainErrorType.INSUFFICIENT_GAS
        elif "timeout" in error_str:
            return BlockchainErrorType.TIMEOUT_ERROR
        elif "register" in error_str:
            return BlockchainErrorType.REGISTRATION_FAILED
        elif "transaction" in error_str or "revert" in error_str:
            return BlockchainErrorType.TRANSACTION_FAILED
        elif "contract" in error_str:
            return BlockchainErrorType.CONTRACT_ERROR
        else:
            return BlockchainErrorType.TRANSACTION_FAILED

    async def _handle_connection_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle connection errors with exponential backoff"""
        if error.retry_count >= self.max_retries:
            logger.error(f"Max retries exceeded for connection error in {self.agent_name}")
            return None

        # Calculate backoff delay
        delay = min(2 ** error.retry_count, 60)  # Max 60 seconds
        logger.info(f"Retrying connection after {delay} seconds...")

        await asyncio.sleep(delay)
        error.retry_count += 1

        # Return retry signal
        return {"retry": True, "delay": delay}

    async def _handle_transaction_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle transaction errors"""
        # Check if error is recoverable
        if "revert" in error.message.lower():
            # Contract revert - likely not recoverable
            error.recoverable = False
            logger.error(f"Contract revert in {self.agent_name}: {error.message}")
            return None

        # For other transaction errors, retry with higher gas
        return {"retry": True, "increase_gas": 1.2}  # 20% gas increase

    async def _handle_gas_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle insufficient gas errors"""
        logger.warning(f"Insufficient gas for {self.agent_name}, increasing gas limit")

        # Suggest gas increase
        current_gas = error.details.get("context", {}).get("gas", 500000)
        new_gas = int(current_gas * 1.5)  # 50% increase

        return {
            "retry": True,
            "gas": new_gas,
            "gas_price_multiplier": 1.2
        }

    async def _handle_nonce_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle nonce errors by fetching fresh nonce"""
        logger.warning(f"Nonce error for {self.agent_name}, fetching fresh nonce")

        return {
            "retry": True,
            "refresh_nonce": True,
            "delay": 1  # Small delay to avoid race conditions
        }

    async def _handle_timeout_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle timeout errors"""
        if error.retry_count >= 2:
            logger.error(f"Persistent timeout for {self.agent_name}")
            return None

        return {
            "retry": True,
            "timeout": 120,  # Increase timeout to 2 minutes
            "delay": 5
        }

    async def _handle_registration_error(self, error: BlockchainError) -> Optional[Any]:
        """Handle agent registration errors"""
        logger.error(f"Registration failed for {self.agent_name}: {error.message}")

        # Check if agent might already be registered
        if "already registered" in error.message.lower():
            return {"already_registered": True}

        # For other registration errors, retry with delay
        if error.retry_count < 2:
            return {
                "retry": True,
                "delay": 10,
                "check_existing": True
            }

        return None

    def _open_circuit_breaker(self):
        """Open circuit breaker to prevent cascading failures"""
        self.circuit_breaker_open = True
        self.circuit_breaker_timeout = datetime.utcnow() + timedelta(minutes=5)
        logger.warning(f"Circuit breaker opened for {self.agent_name} until {self.circuit_breaker_timeout}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history
                        if e.timestamp > datetime.utcnow() - timedelta(hours=1)]

        error_by_type = {}
        for error in self.error_history:
            error_by_type[error.error_type.value] = error_by_type.get(error.error_type.value, 0) + 1

        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "errors_by_type": error_by_type,
            "circuit_breaker_open": self.circuit_breaker_open,
            "last_error": self.error_history[-1].message if self.error_history else None
        }

    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Execute an operation with automatic retry and error handling

        Args:
            operation: The async function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Operation result if successful, None otherwise
        """
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Execute the operation
                result = await operation(*args, **kwargs)

                # Reset error count on success
                if retries > 0:
                    logger.info(f"Operation succeeded after {retries} retries")

                return result

            except Exception as e:
                last_error = e

                # Handle the error
                recovery_result = await self.handle_error(
                    e,
                    operation.__name__,
                    {"args": args, "kwargs": kwargs}
                )

                if not recovery_result or not recovery_result.get("retry"):
                    break

                # Apply recovery suggestions
                if recovery_result.get("delay"):
                    await asyncio.sleep(recovery_result["delay"])

                if recovery_result.get("gas"):
                    kwargs["gas"] = recovery_result["gas"]

                if recovery_result.get("increase_gas"):
                    current_gas = kwargs.get("gas", 500000)
                    kwargs["gas"] = int(current_gas * recovery_result["increase_gas"])

                if recovery_result.get("refresh_nonce"):
                    kwargs["refresh_nonce"] = True

                retries += 1

        logger.error(f"Operation {operation.__name__} failed after {retries} retries: {last_error}")
        return None

    # AI-Enhanced Methods for Intelligent Error Handling

    async def _ai_classify_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> BlockchainErrorType:
        """AI-enhanced error classification with Grok intelligence (AI required for A2A compliance)"""

        try:
            # Prepare context for AI analysis
            error_context = {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "operation": operation,
                "agent": self.agent_name,
                "context": context or {},
                "recent_errors": [
                    {"type": e.error_type.value, "message": e.message, "timestamp": e.timestamp.isoformat()}
                    for e in self.error_history[-5:]  # Last 5 errors for pattern analysis
                ]
            }

            # Ask Grok for intelligent error classification
            ai_prompt = f"""
            Analyze this blockchain error and provide intelligent classification:

            Error Details: {json.dumps(error_context, indent=2)}

            Consider:
            1. Error patterns from recent history
            2. Operation context and criticality
            3. Network conditions and gas market
            4. Agent-specific behavior patterns

            Provide JSON response with:
            {{
                "error_type": "one of: connection_error, transaction_failed, insufficient_gas, nonce_error, contract_error, timeout_error, registration_failed, message_send_failed, reputation_error",
                "confidence": 0.95,
                "reasoning": "detailed explanation",
                "severity": "low|medium|high|critical",
                "pattern_detected": "description of any patterns"
            }}
            """

            ai_response = await self.grok_client.generate_response(ai_prompt)

            if ai_response and "error_type" in ai_response:
                ai_type_str = ai_response["error_type"]
                # Map AI response to enum
                for error_type in BlockchainErrorType:
                    if error_type.value == ai_type_str:
                        logger.info(f"🤖 AI classified error as {ai_type_str} (confidence: {ai_response.get('confidence', 'unknown')})")
                        return error_type

        except Exception as ai_error:
            logger.warning(f"AI error classification failed: {ai_error}")

        # Fallback to basic classification
        return self._classify_error(error)

    async def _ai_generate_recovery_plan(self, blockchain_error: BlockchainError) -> Dict[str, Any]:
        """AI-powered recovery plan generation (AI is required for A2A compliance)"""

        try:
            # Prepare comprehensive context for AI
            recovery_context = {
                "current_error": {
                    "type": blockchain_error.error_type.value,
                    "message": blockchain_error.message,
                    "retry_count": blockchain_error.retry_count,
                    "details": blockchain_error.details
                },
                "error_history": [
                    {
                        "type": e.error_type.value,
                        "message": e.message,
                        "retry_count": e.retry_count,
                        "timestamp": e.timestamp.isoformat(),
                        "recovered": getattr(e, 'recovered', False)
                    }
                    for e in self.error_history[-10:]  # Last 10 errors
                ],
                "agent_info": {
                    "name": self.agent_name,
                    "circuit_breaker_open": self.circuit_breaker_open,
                    "ai_success_rate": self.ai_success_rate
                },
                "network_conditions": {
                    "recent_error_count": len([e for e in self.error_history if e.timestamp > datetime.utcnow() - timedelta(minutes=10)]),
                    "error_frequency": "high" if len(self.error_history) > 20 else "normal"
                }
            }

            ai_prompt = f"""
            As a blockchain expert AI, analyze this error situation and generate an intelligent recovery plan:

            Context: {json.dumps(recovery_context, indent=2)}

            Consider:
            1. Error patterns and root causes
            2. Network congestion and gas market conditions
            3. Agent's operational history and success patterns
            4. Critical vs non-critical operations
            5. Resource optimization strategies

            Generate a comprehensive recovery plan as JSON:
            {{
                "strategy": "retry|fallback|escalate|abort",
                "ai_analysis": "detailed analysis of the situation",
                "recommended_actions": [
                    {{"action": "retry", "params": {{"delay": 30, "gas_multiplier": 1.5}}}},
                    {{"action": "fallback", "params": {{"queue_for_later": true}}}}
                ],
                "confidence": 0.85,
                "reasoning": "why this plan was chosen",
                "force_execution": false,
                "expected_success_rate": 0.7,
                "alternative_strategies": ["fallback options if primary fails"],
                "monitoring": {{"watch_for": ["specific conditions to monitor"]}},
                "learning_points": ["insights to remember for future"]
            }}
            """

            ai_response = await self.grok_client.generate_response(ai_prompt)

            if ai_response and "strategy" in ai_response:
                self.ai_decisions.append({
                    "timestamp": datetime.utcnow(),
                    "error_type": blockchain_error.error_type.value,
                    "ai_plan": ai_response,
                    "outcome": None  # Will be filled later
                })

                logger.info(f"🤖 AI generated recovery plan: {ai_response['strategy']} (confidence: {ai_response.get('confidence', 'unknown')})")
                return ai_response

        except Exception as ai_error:
            logger.warning(f"AI recovery plan generation failed: {ai_error}")

        # Fallback to basic recovery plan
        return await self._basic_recovery_plan(blockchain_error)

    async def _basic_recovery_plan(self, blockchain_error: BlockchainError) -> Dict[str, Any]:
        """Fallback recovery plan when AI is unavailable"""
        return {
            "strategy": "retry",
            "ai_analysis": "Basic fallback plan (AI unavailable)",
            "recommended_actions": [{"action": "retry", "params": {"delay": 5, "gas_multiplier": 1.2}}],
            "confidence": 0.5,
            "reasoning": "Standard retry logic",
            "force_execution": False
        }

    async def _execute_ai_recovery_plan(self, recovery_plan: Dict[str, Any], blockchain_error: BlockchainError) -> Optional[Any]:
        """Execute the AI-generated recovery plan"""
        strategy = recovery_plan.get("strategy", "retry")
        actions = recovery_plan.get("recommended_actions", [])

        for action_spec in actions:
            action = action_spec.get("action")
            params = action_spec.get("params", {})

            if action == "retry":
                if blockchain_error.retry_count >= self.max_retries:
                    continue

                delay = params.get("delay", 5)
                await asyncio.sleep(delay)

                return {
                    "retry": True,
                    "delay": delay,
                    "gas_multiplier": params.get("gas_multiplier", 1.2),
                    "ai_recommended": True
                }

            elif action == "fallback":
                if params.get("queue_for_later"):
                    # Implementation would depend on fallback handler
                    return {"retry": False, "queued": True, "ai_recommended": True}

            elif action == "escalate":
                # Could trigger help-seeking or admin notification
                logger.warning(f"🚨 AI recommends escalating error for {self.agent_name}: {blockchain_error.message}")
                return {"retry": False, "escalated": True, "ai_recommended": True}

        return None

    async def _ai_learn_from_outcome(self, blockchain_error: BlockchainError, recovery_plan: Dict[str, Any], result: Optional[Any]):
        """Learn from recovery outcomes to improve future decisions"""
        if not self.grok_client or not self.learning_enabled:
            return

        try:
            # Record the outcome
            outcome = {
                "timestamp": datetime.utcnow(),
                "error_type": blockchain_error.error_type.value,
                "recovery_strategy": recovery_plan.get("strategy"),
                "success": result is not None and result.get("retry", False),
                "result": result
            }

            # Update AI decision record
            if self.ai_decisions:
                self.ai_decisions[-1]["outcome"] = outcome

            # Calculate new success rate
            recent_decisions = [d for d in self.ai_decisions if d.get("outcome")][-20:]  # Last 20 decisions
            if recent_decisions:
                successful = len([d for d in recent_decisions if d["outcome"].get("success", False)])
                self.ai_success_rate = successful / len(recent_decisions)

            # Send learning feedback to AI
            if len(self.ai_decisions) % 10 == 0:  # Every 10 decisions
                await self._send_learning_feedback()

        except Exception as e:
            logger.warning(f"AI learning failed: {e}")

    async def _send_learning_feedback(self):
        """Send learning feedback to Grok for continuous improvement"""
        if not self.grok_client:
            return

        try:
            recent_decisions = self.ai_decisions[-20:]  # Last 20 decisions
            learning_data = {
                "agent": self.agent_name,
                "success_rate": self.ai_success_rate,
                "decisions": recent_decisions,
                "patterns": self._extract_patterns()
            }

            feedback_prompt = f"""
            Learning feedback for blockchain error handling:

            Data: {json.dumps(learning_data, indent=2, default=str)}

            Analyze performance patterns and suggest improvements for future error handling decisions.
            Focus on:
            1. Which strategies worked best for each error type
            2. Optimal timing and parameters
            3. Environmental factors that influence success
            4. Patterns in agent behavior

            Provide insights to improve future decisions.
            """

            await self.grok_client.generate_response(feedback_prompt, learning_mode=True)

        except Exception as e:
            logger.warning(f"Learning feedback failed: {e}")

    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from error history for learning"""
        patterns = {
            "most_common_errors": {},
            "success_rates_by_type": {},
            "time_patterns": {},
            "recovery_effectiveness": {}
        }

        # Analyze error frequency
        for error in self.error_history:
            error_type = error.error_type.value
            patterns["most_common_errors"][error_type] = patterns["most_common_errors"].get(error_type, 0) + 1

        # Analyze AI decision effectiveness
        for decision in self.ai_decisions:
            if decision.get("outcome"):
                strategy = decision.get("ai_plan", {}).get("strategy", "unknown")
                if strategy not in patterns["recovery_effectiveness"]:
                    patterns["recovery_effectiveness"][strategy] = {"total": 0, "successful": 0}

                patterns["recovery_effectiveness"][strategy]["total"] += 1
                if decision["outcome"].get("success"):
                    patterns["recovery_effectiveness"][strategy]["successful"] += 1

        return patterns

    def get_ai_metrics(self) -> Dict[str, Any]:
        """Get AI-enhanced metrics"""
        return {
            "ai_enabled": self.grok_client is not None,
            "ai_success_rate": self.ai_success_rate,
            "total_ai_decisions": len(self.ai_decisions),
            "learning_enabled": self.learning_enabled,
            "recent_decisions": len([d for d in self.ai_decisions if d["timestamp"] > datetime.utcnow() - timedelta(hours=1)]),
            "pattern_insights": self._extract_patterns()
        }


class BlockchainFallbackHandler:
    """
    Provides fallback mechanisms when blockchain is unavailable
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.fallback_queue: List[Dict[str, Any]] = []
        self.fallback_active = False

    async def queue_for_retry(self, operation: str, data: Dict[str, Any]):
        """Queue operation for retry when blockchain is available"""
        self.fallback_queue.append({
            "operation": operation,
            "data": data,
            "timestamp": datetime.utcnow(),
            "agent": self.agent_name
        })

        logger.info(f"Queued operation {operation} for retry ({len(self.fallback_queue)} in queue)")

    async def process_queued_operations(self, blockchain_client):
        """Process queued operations when blockchain becomes available"""
        if not self.fallback_queue:
            return

        logger.info(f"Processing {len(self.fallback_queue)} queued operations for {self.agent_name}")

        processed = []
        for item in self.fallback_queue:
            try:
                if item["operation"] == "register_agent":
                    await blockchain_client.register_agent(**item["data"])
                elif item["operation"] == "send_message":
                    await blockchain_client.send_message(**item["data"])

                processed.append(item)
                logger.info(f"Successfully processed queued operation: {item['operation']}")

            except Exception as e:
                logger.error(f"Failed to process queued operation {item['operation']}: {e}")

        # Remove processed items
        for item in processed:
            self.fallback_queue.remove(item)

    def activate_fallback(self):
        """Activate fallback mode"""
        self.fallback_active = True
        logger.warning(f"Fallback mode activated for {self.agent_name}")

    def deactivate_fallback(self):
        """Deactivate fallback mode"""
        self.fallback_active = False
        logger.info(f"Fallback mode deactivated for {self.agent_name}")

    def get_fallback_status(self) -> Dict[str, Any]:
        """Get fallback status"""
        return {
            "fallback_active": self.fallback_active,
            "queued_operations": len(self.fallback_queue),
            "oldest_queued": self.fallback_queue[0]["timestamp"].isoformat() if self.fallback_queue else None
        }

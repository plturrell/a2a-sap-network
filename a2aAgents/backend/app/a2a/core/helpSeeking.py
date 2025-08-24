"""
A2A Help-Seeking System
Enables agents to actively request help from other agents when encountering problems
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



import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from .a2aTypes import A2AMessage, MessagePart, MessageRole
from .helpActionEngine import AgentHelpActionSystem
from .skillScopeValidator import get_trust_based_help_validator, HelpRequestType
from .taskTracker import TaskStatus
from .messageQueue import AgentMessageQueue
from .circuitBreaker import EnhancedCircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class HelpRequest:
    """Represents a help request to another agent"""

    def __init__(
        self,
        requesting_agent: str,
        target_agent: str,
        problem_type: str,
        problem_description: str,
        context: Dict[str, Any] = None,
        urgency: str = "medium",
    ):
        self.request_id = str(uuid4())
        self.requesting_agent = requesting_agent
        self.target_agent = target_agent
        self.problem_type = problem_type
        self.problem_description = problem_description
        self.context = context or {}
        self.urgency = urgency  # low, medium, high, critical
        self.created_at = datetime.utcnow()
        self.status = "pending"
        self.response = None


class AgentHelpSeeker:
    """Mixin class that adds help-seeking capabilities to A2A agents"""

    def __init__(self):
        self.help_requests: Dict[str, HelpRequest] = {}
        self.help_thresholds = {
            "max_retries": 3,
            "error_count_threshold": 5,
            "timeout_threshold": 30.0,
            "consecutive_failures": 3,
        }
        # Initialize trust-based help validator
        self.trust_validator = get_trust_based_help_validator()
        # Initialize circuit breaker for agent-to-agent communication
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30,
            recovery_threshold=2,
            enable_exponential_backoff=True,
            initial_backoff=1.0,
            max_backoff=60.0,
        )
        self.circuit_breaker = EnhancedCircuitBreaker(
            name="agent_help_communication", config=circuit_config
        )
        # Agent registry must be configured via environment variables - no hardcoded URLs
        self.agent_registry = {}
        # Initialize task tracker (optional - set by agents that use task tracking)
        self.task_tracker = None

        # Load agent URLs from environment
        agents_config = [
            (
                "data_product_agent_0",
                "A2A_DATA_PRODUCT_AGENT_URL",
                ["data_processing", "ord_registration", "dublin_core", "crd_data"],
            ),
            (
                "financial_standardization_agent_1",
                "A2A_FINANCIAL_AGENT_URL",
                [
                    "data_standardization",
                    "l4_processing",
                    "integrity_verification",
                    "financial_entities",
                ],
            ),
            (
                "data_manager_agent",
                "A2A_DATA_MANAGER_URL",
                ["data_storage", "crud_operations", "database_management", "file_operations"],
            ),
        ]

        for agent_id, env_var, specialties in agents_config:
            agent_url = os.getenv(env_var)
            if agent_url:
                self.agent_registry[agent_id] = {"url": agent_url, "specialties": specialties}
            else:
                logger.warning(
                    f"Agent {agent_id} not configured - {env_var} environment variable not set"
                )

        if not self.agent_registry:
            logger.warning("No agent URLs configured. Help-seeking will be disabled until agents are available.")
        self.error_counters = {}
        self.consecutive_failures = {}

        # Initialize help action system (will be set by each agent)
        self.help_action_system: Optional[AgentHelpActionSystem] = None

        # Initialize message queue (will be configured by each agent)
        self.message_queue: Optional[AgentMessageQueue] = None

    def initialize_help_action_system(self, agent_id: str, agent_context: Dict[str, Any]):
        """Initialize the help action system for this agent"""
        self.help_action_system = AgentHelpActionSystem(agent_id, agent_context)
        logger.info(f"✅ Help action system initialized for {agent_id}")

    def initialize_message_queue(
        self,
        agent_id: str,
        max_concurrent_processing: int = 5,
        auto_mode_threshold: int = 10,
        enable_streaming: bool = True,
        enable_batch_processing: bool = True,
    ):
        """Initialize the independent message queue for this agent"""
        self.message_queue = AgentMessageQueue(
            agent_id=agent_id,
            max_concurrent_processing=max_concurrent_processing,
            auto_mode_threshold=auto_mode_threshold,
            enable_streaming=enable_streaming,
            enable_batch_processing=enable_batch_processing,
        )

        # Set this agent's original process_message method as the queue processor
        if hasattr(self, "_original_process_message"):
            self.message_queue.set_message_processor(self._original_process_message)

        logger.info(f"✅ Message queue initialized for {agent_id}")

    async def start_message_queue_processor(self):
        """Start the background message queue processor"""
        if self.message_queue:
            await self.message_queue.start_queue_processor()

    async def stop_message_queue_processor(self):
        """Stop the background message queue processor"""
        if self.message_queue:
            await self.message_queue.stop_queue_processor()

    def _should_seek_help(self, problem_type: str, error_context: Dict[str, Any] = None) -> bool:
        """Determine if agent should seek help based on error patterns and thresholds"""

        # Check consecutive failures
        failure_count = self.consecutive_failures.get(problem_type, 0)
        if failure_count >= self.help_thresholds["consecutive_failures"]:
            logger.info(
                f"🆘 Seeking help due to {failure_count} consecutive failures for {problem_type}"
            )
            return True

        # Check error count threshold
        error_count = self.error_counters.get(problem_type, 0)
        if error_count >= self.help_thresholds["error_count_threshold"]:
            logger.info(f"🆘 Seeking help due to {error_count} total errors for {problem_type}")
            return True

        # Check specific error conditions
        if error_context:
            # Network connectivity issues
            if (
                "connection" in str(error_context).lower()
                or "timeout" in str(error_context).lower()
            ):
                logger.info(f"🆘 Seeking help due to connectivity issues: {error_context}")
                return True

            # Service unavailable
            if (
                "unavailable" in str(error_context).lower()
                or "service" in str(error_context).lower()
            ):
                logger.info(f"🆘 Seeking help due to service issues: {error_context}")
                return True

            # Data integrity problems
            if (
                "integrity" in str(error_context).lower()
                or "corruption" in str(error_context).lower()
            ):
                logger.info(f"🆘 Seeking help due to data integrity issues: {error_context}")
                return True

        return False

    def _find_best_helper_agent(self, problem_type: str) -> Optional[str]:
        """Find the best agent to ask for help based on problem type"""

        # Map problem types to agent specialties
        problem_to_specialty = {
            "data_processing": ["data_processing", "ord_registration"],
            "data_storage": ["data_storage", "crud_operations"],
            "data_retrieval": ["data_storage", "database_management"],
            "standardization": ["data_standardization", "l4_processing"],
            "metadata": ["ord_repository", "metadata_enhancement"],
            "quality_assessment": ["quality_assessment", "ai_enhancement"],
            "ord_registration": ["ord_registration", "dublin_core"],
            "integrity_verification": ["integrity_verification", "financial_entities"],
            "database_connectivity": ["database_management", "crud_operations"],
            "file_operations": ["file_operations", "data_storage"],
            "ai_enhancement": ["ai_enhancement", "metadata_enhancement"],
        }

        # Find agents with relevant specialties
        relevant_specialties = problem_to_specialty.get(problem_type, [])

        for agent_id, agent_info in self.agent_registry.items():
            # Don't ask ourselves for help
            if agent_id == getattr(self, "agent_id", None):
                continue

            # Check if agent has relevant specialties
            agent_specialties = agent_info.get("specialties", [])
            if any(specialty in agent_specialties for specialty in relevant_specialties):
                logger.info(f"🎯 Found helper agent {agent_id} for {problem_type}")
                return agent_id

        # Fallback: ask the most general agent (Data Manager)
        if "data_manager_agent" in self.agent_registry:
            logger.info(f"📋 Using fallback helper: data_manager_agent for {problem_type}")
            return "data_manager_agent"

        return None

    async def _send_help_request(
        self,
        target_agent: str,
        problem_type: str,
        problem_description: str,
        context: Dict[str, Any] = None,
        urgency: str = "medium",
    ) -> Optional[Dict[str, Any]]:
        """Send a help request to another agent with skill scope validation"""

        try:
            # Validate skill scope before sending request
            requesting_agent_id = getattr(self, "agent_id", "unknown")

            # Determine help request type from problem context
            help_type = self._determine_help_request_type(problem_type, context)

            # Extract requested skill if applicable
            requested_skill = context.get("skill") if context else None
            if not requested_skill:
                requested_skill = self._map_problem_to_skill(problem_type)

            # Validate the help request against trust contract
            validation_result = self.trust_validator.validate_help_request(
                requesting_agent_id=requesting_agent_id,
                target_agent_id=target_agent,
                help_type=help_type,
                requested_skill=requested_skill,
                problem_context=context,
            )

            if not validation_result["valid"]:
                logger.warning(
                    f"❌ Help request rejected due to scope violation: {validation_result['reason']}"
                )
                return {
                    "error": "scope_validation_failed",
                    "reason": validation_result["reason"],
                    "allowed_help_types": validation_result["allowed_help_types"],
                    "skill_compatibility": validation_result["skill_compatibility"],
                }

            logger.info(f"✅ Help request validated: {validation_result['reason']}")

        except (ValueError, KeyError, AttributeError) as validation_error:
            logger.error(f"❌ Scope validation error: {validation_error}")
            return {"error": "validation_error", "reason": str(validation_error)}

        try:
            # Create help request
            help_request = HelpRequest(
                requesting_agent=getattr(self, "agent_id", "unknown"),
                target_agent=target_agent,
                problem_type=problem_type,
                problem_description=problem_description,
                context=context,
                urgency=urgency,
            )

            self.help_requests[help_request.request_id] = help_request

            # Get target agent URL
            agent_info = self.agent_registry.get(target_agent)
            if not agent_info:
                logger.error(f"❌ Unknown target agent: {target_agent}")
                return None

            target_url = agent_info["url"]

            # Construct help request message
            help_message = A2AMessage(
                role=MessageRole.USER,
                parts=[
                    MessagePart(kind="text", text=f"Help request: {problem_description}"),
                    MessagePart(
                        kind="data",
                        data={
                            "help_request": True,
                            "advisor_request": True,
                            "problem_type": problem_type,
                            "urgency": urgency,
                            "requesting_agent": getattr(self, "agent_id", "unknown"),
                            "context": context or {},
                            "request_id": help_request.request_id,
                            "question": f"I'm experiencing {problem_type}: {problem_description}. "
                            "Can you help me troubleshoot this?",
                        },
                    ),
                ],
            )

            # Send request to target agent with circuit breaker protection
            endpoint = f"{target_url}/a2a/v1/messages"

            async def make_help_request():
                # A2A Protocol Compliance: Use blockchain messaging instead of direct HTTP
                if hasattr(self, 'send_blockchain_message'):
                    # Use blockchain messaging if available
                    message_id = await self.send_blockchain_message(
                        to_address=target_agent,
                        content=help_message.model_dump(),
                        message_type="HELP_REQUEST"
                    )
                    if message_id:
                        return {"message_id": message_id, "status": "sent_via_blockchain"}
                    else:
                        raise RuntimeError("Failed to send help request via blockchain")
                else:
                    # Fallback: A2A Network Client (should route through blockchain)
                    from .networkClient import A2ANetworkClient
                    network_client = A2ANetworkClient(agent_id=getattr(self, 'agent_id', 'unknown'))
                    
                    response = await network_client.send_message(
                        to_agent=target_agent,
                        message=help_message.model_dump(),
                        message_type="HELP_REQUEST"
                    )
                    
                    if not response or response.get('error'):
                        raise RuntimeError(f"A2A network request failed: {response.get('error', 'Unknown error')}")
                    
                    return response

            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(make_help_request)

            if result is not None:
                help_request.status = "completed"
                help_request.response = result
                logger.info(f"✅ Help request sent successfully to {target_agent}")
                return result

            help_request.status = "failed"
            logger.error("❌ Help request failed (circuit breaker open or timeout)")
            return None

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"❌ Error sending help request: {e}")
            if help_request.request_id in self.help_requests:
                self.help_requests[help_request.request_id].status = "failed"
            return None

    async def seek_help_and_act(
        self,
        problem_type: str,
        error: Exception,
        original_operation: Optional[Callable] = None,
        context: Dict[str, Any] = None,
        urgency: str = "medium",
    ) -> Dict[str, Any]:
        """Seek help and automatically act on the response to resolve the issue"""

        # Track error occurrence
        self.error_counters[problem_type] = self.error_counters.get(problem_type, 0) + 1
        self.consecutive_failures[problem_type] = self.consecutive_failures.get(problem_type, 0) + 1

        # Check if we should seek help
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        if not self._should_seek_help(problem_type, error_context):
            logger.debug(f"📊 Not seeking help for {problem_type} yet (threshold not met)")
            return {"success": False, "reason": "threshold_not_met", "help_sought": False}

        # Find the best agent to help
        helper_agent = self._find_best_helper_agent(problem_type)
        if not helper_agent:
            logger.warning(f"⚠️ No suitable helper agent found for {problem_type}")
            return {"success": False, "reason": "no_helper_found", "help_sought": False}

        # Send help request
        problem_description = f"Error: {type(error).__name__}: {str(error)}"

        help_response = await self._send_help_request(
            target_agent=helper_agent,
            problem_type=problem_type,
            problem_description=problem_description,
            context=error_context,
            urgency=urgency,
        )

        if not help_response:
            logger.warning(f"⚠️ No help response received for {problem_type}")
            return {"success": False, "reason": "no_help_response", "help_sought": True}

        # Now the critical part: ACT ON THE HELP RESPONSE
        if self.help_action_system:
            try:
                action_result = await self.help_action_system.process_help_and_execute_actions(
                    help_response=help_response,
                    original_operation=original_operation,
                    operation_context=context or {},
                )

                # Reset consecutive failure counter on successful help application
                if action_result["success"] and action_result.get("resolved_issue"):
                    self.consecutive_failures[problem_type] = 0
                    logger.info(f"🎉 Help successfully resolved issue for {problem_type}")

                return {
                    "success": action_result["success"],
                    "help_sought": True,
                    "help_received": True,
                    "actions_taken": True,
                    "resolved_issue": action_result.get("resolved_issue", False),
                    "action_plan_id": action_result.get("action_plan_id"),
                    "actions_executed": action_result.get("actions_executed", 0),
                    "execution_time": action_result.get("execution_time"),
                    "final_outcome": action_result.get("final_outcome"),
                }

            except (RuntimeError, ValueError, KeyError) as action_error:
                logger.error(f"❌ Error acting on help response: {action_error}")
                return {
                    "success": False,
                    "help_sought": True,
                    "help_received": True,
                    "actions_taken": False,
                    "error": str(action_error),
                }
        else:
            logger.warning("⚠️ Help action system not initialized - cannot act on help")
            return {
                "success": False,
                "help_sought": True,
                "help_received": True,
                "actions_taken": False,
                "reason": "action_system_not_initialized",
            }

    async def seek_help_for_error(
        self,
        problem_type: str,
        error: Exception,
        context: Dict[str, Any] = None,
        urgency: str = "medium",
    ) -> Optional[Dict[str, Any]]:
        """Main method to seek help when encountering errors"""

        # Track error occurrence
        self.error_counters[problem_type] = self.error_counters.get(problem_type, 0) + 1
        self.consecutive_failures[problem_type] = self.consecutive_failures.get(problem_type, 0) + 1

        # Check if we should seek help
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        if not self._should_seek_help(problem_type, error_context):
            logger.debug(f"📊 Not seeking help for {problem_type} yet (threshold not met)")
            return None

        # Find the best agent to help
        helper_agent = self._find_best_helper_agent(problem_type)
        if not helper_agent:
            logger.warning(f"⚠️ No suitable helper agent found for {problem_type}")
            return None

        # Send help request
        problem_description = f"Error: {type(error).__name__}: {str(error)}"

        result = await self._send_help_request(
            target_agent=helper_agent,
            problem_type=problem_type,
            problem_description=problem_description,
            context=error_context,
            urgency=urgency,
        )

        # Reset consecutive failure counter on successful help request
        if result:
            self.consecutive_failures[problem_type] = 0
            logger.info(f"🔄 Reset failure counter for {problem_type} after getting help")

        return result

    async def seek_help_for_guidance(
        self, question: str, domain: str = "general", context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Seek guidance or advice from other agents proactively"""

        # Find appropriate agent for the domain
        helper_agent = self._find_best_helper_agent(domain)
        if not helper_agent:
            logger.warning(f"⚠️ No suitable advisor found for domain: {domain}")
            return None

        # Send guidance request
        return await self._send_help_request(
            target_agent=helper_agent,
            problem_type="guidance_request",
            problem_description=question,
            context=context or {},
            urgency="low",
        )

    def get_help_request_history(self) -> List[Dict[str, Any]]:
        """Get history of help requests made by this agent"""
        return [
            {
                "request_id": req.request_id,
                "target_agent": req.target_agent,
                "problem_type": req.problem_type,
                "description": req.problem_description,
                "urgency": req.urgency,
                "status": req.status,
                "created_at": req.created_at.isoformat(),
                "response_received": req.response is not None,
            }
            for req in self.help_requests.values()
        ]

    def _determine_help_request_type(
        self, problem_type: str, context: Dict[str, Any] = None
    ) -> HelpRequestType:
        """Determine the type of help request based on problem and context"""

        if context and context.get("advisor_request"):
            # If it's an advisor request, check if it's about capabilities or status
            question = context.get("question", "").lower()
            if any(
                word in question
                for word in ["what do you do", "what can you", "capabilities", "skills"]
            ):
                return HelpRequestType.CAPABILITY_QUESTION
            if any(
                word in question for word in ["what did you", "how did you", "status", "result"]
            ):
                return HelpRequestType.STATUS_INQUIRY

            return HelpRequestType.SKILL_GUIDANCE

        # For error-based help requests, usually troubleshooting
        if context and (
            "error" in context
            or problem_type in ["data_processing", "data_storage", "standardization"]
        ):
            return HelpRequestType.TROUBLESHOOTING

        # Default to skill guidance
        return HelpRequestType.SKILL_GUIDANCE

    def _map_problem_to_skill(self, problem_type: str) -> Optional[str]:
        """Map a problem type to a specific skill"""

        problem_skill_mapping = {
            "data_processing": "data-processing",
            "data_storage": "crud-operations",
            "data_retrieval": "database-management",
            "standardization": "batch-standardization",
            "metadata": "metadata-registration",
            "quality_assessment": "quality-assessment",
            "ord_registration": "ord-descriptor-creation",
            "dublin_core": "dublin-core-extraction",
        }

        return problem_skill_mapping.get(problem_type)

    def get_agent_operational_state(self) -> Dict[str, Any]:
        """Get comprehensive agent operational state for A2A status inquiries"""

        # Get task tracker state if available
        current_tasks = []
        if hasattr(self, "task_tracker") and self.task_tracker:
            current_tasks = [
                {
                    "task_id": task_id,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "created_at": task.created_at.isoformat(),
                    "progress": {
                        "completed_items": len(
                            [item for item in task.checklist if item.status == TaskStatus.COMPLETED]
                        ),
                        "total_items": len(task.checklist),
                        "pending_help_requests": len(
                            [item for item in task.checklist if item.help_request_id is not None]
                        ),
                    },
                }
                for task_id, task in self.task_tracker.tasks.items()
                if task.status in [TaskStatus.IN_PROGRESS, TaskStatus.WAITING_FOR_HELP]
            ]

        # Determine overall agent state
        if current_tasks:
            # Check if any tasks are waiting for help
            waiting_for_help = any(task["status"] == "waiting_for_help" for task in current_tasks)
            if waiting_for_help:
                agent_state = "waiting_for_help"
            else:
                agent_state = "processing"
        else:
            agent_state = "idle"

        # Get recent help requests
        recent_help_requests = [
            {
                "request_id": req.request_id,
                "target_agent": req.target_agent,
                "problem_type": req.problem_type,
                "status": req.status,
                "created_at": req.created_at.isoformat(),
                "urgency": req.urgency,
            }
            for req in self.help_requests.values()
            if (datetime.utcnow() - req.created_at).total_seconds() < 3600  # Last hour
        ]

        # Calculate error statistics
        total_errors = sum(self.error_counters.values())
        active_failures = {k: v for k, v in self.consecutive_failures.items() if v > 0}

        return {
            "agent_id": getattr(self, "agent_id", "unknown"),
            "agent_name": getattr(self, "agent_card", {}).get("name", "Unknown Agent"),
            "timestamp": datetime.utcnow().isoformat(),
            "operational_state": agent_state,
            "current_tasks": current_tasks,
            "task_summary": {
                "active_tasks": len(current_tasks),
                "tasks_waiting_for_help": len(
                    [t for t in current_tasks if t["status"] == "waiting_for_help"]
                ),
                "total_checklist_items": sum(t["progress"]["total_items"] for t in current_tasks),
                "completed_checklist_items": sum(
                    t["progress"]["completed_items"] for t in current_tasks
                ),
            },
            "help_activity": {
                "recent_help_requests": len(recent_help_requests),
                "help_requests_detail": recent_help_requests[-5:],  # Last 5 requests
                "total_error_count": total_errors,
                "active_failure_patterns": active_failures,
            },
            "capabilities": {
                "help_seeking_enabled": True,
                "trust_validation_active": hasattr(self, "trust_validator"),
                "task_tracking_active": hasattr(self, "task_tracker"),
                "ai_advisor_available": hasattr(self, "ai_advisor"),
            },
            "trust_context": {
                "agent_registry_known": len(self.agent_registry),
                "known_specialties": list(
                    set().union(*[info["specialties"] for info in self.agent_registry.values()])
                ),
            },
            "message_queue": {
                "queue_enabled": self.message_queue is not None,
                "queue_status": (
                    self.message_queue.get_queue_status() if self.message_queue else None
                ),
            },
        }

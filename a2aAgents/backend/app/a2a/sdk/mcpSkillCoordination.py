"""
MCP Extension for Intra-Agent Skill Coordination
Extends the Model Context Protocol for skill-to-skill communication within A2A agents
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import inspect
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SkillMessageType(Enum):
    """Extended MCP message types for skill coordination"""
    # Standard MCP types
    TOOL_CALL = "tools/call"
    RESOURCE_READ = "resources/read"
    PROMPT_GET = "prompts/get"

    # Extended types for skill coordination
    SKILL_REQUEST = "skills/request"
    SKILL_RESPONSE = "skills/response"
    SKILL_NOTIFICATION = "skills/notification"
    SKILL_STATUS = "skills/status"
    SKILL_COORDINATION = "skills/coordination"


class SkillPriority(Enum):
    """Message priority levels for skill coordination"""
    CRITICAL = 1    # Must be processed immediately
    HIGH = 2        # Process before normal messages
    NORMAL = 3      # Standard processing
    LOW = 4         # Process when queue is light
    BACKGROUND = 5  # Process when idle


class SkillState(Enum):
    """Skill execution states"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class SkillMessage:
    """Extended MCP message for skill-to-skill communication"""
    # Standard MCP fields
    jsonrpc: str = "2.0"
    method: str = ""
    id: Optional[Union[str, int]] = None
    params: Optional[Dict[str, Any]] = None

    # Extended fields for skill coordination
    message_type: SkillMessageType = SkillMessageType.SKILL_REQUEST
    sender_skill: str = ""
    receiver_skill: str = ""
    priority: SkillPriority = SkillPriority.NORMAL
    correlation_id: Optional[str] = None  # For request-response correlation
    reply_to: Optional[str] = None        # For response routing
    expires_at: Optional[datetime] = None # Message expiration
    context: Optional[Dict[str, Any]] = None  # Additional context

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class SkillCapability:
    """Describes what a skill can do"""
    skill_name: str
    tools: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    prompts: List[Dict[str, Any]]
    dependencies: List[str]  # Skills this skill depends on
    provides: List[str]      # Services this skill provides
    state: SkillState = SkillState.IDLE
    load_factor: float = 0.0 # Current load (0.0 = idle, 1.0 = fully busy)
    last_activity: Optional[datetime] = None


class SkillMessageQueue:
    """Priority-based message queue for skill coordination"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {priority: deque() for priority in SkillPriority}
        self.pending_responses = {}  # correlation_id -> Future
        self.message_history = deque(maxlen=500)
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_processed": 0,
            "messages_expired": 0,
            "average_processing_time": 0.0
        }

    async def enqueue(self, message: SkillMessage) -> bool:
        """Add message to appropriate priority queue"""

        # Check if queue is full
        total_messages = sum(len(q) for q in self.queues.values())
        if total_messages >= self.max_size:
            # Remove oldest low priority messages to make room
            if self.queues[SkillPriority.BACKGROUND]:
                self.queues[SkillPriority.BACKGROUND].popleft()
            elif self.queues[SkillPriority.LOW]:
                self.queues[SkillPriority.LOW].popleft()
            else:
                logger.warning("Message queue full, dropping message")
                return False

        # Check expiration
        if message.expires_at and datetime.utcnow() > message.expires_at:
            logger.debug(f"Message {message.id} expired before queuing")
            self.stats["messages_expired"] += 1
            return False

        # Add to appropriate queue
        self.queues[message.priority].append(message)
        self.message_history.append({
            "action": "enqueued",
            "message_id": message.id,
            "timestamp": datetime.utcnow(),
            "priority": message.priority.name
        })

        self.stats["messages_received"] += 1
        logger.debug(f"Enqueued message {message.id} with priority {message.priority.name}")
        return True

    async def dequeue(self) -> Optional[SkillMessage]:
        """Get next message by priority"""

        # Process by priority order
        for priority in SkillPriority:
            queue = self.queues[priority]

            while queue:
                message = queue.popleft()

                # Check if message expired
                if message.expires_at and datetime.utcnow() > message.expires_at:
                    logger.debug(f"Message {message.id} expired during processing")
                    self.stats["messages_expired"] += 1
                    continue

                self.message_history.append({
                    "action": "dequeued",
                    "message_id": message.id,
                    "timestamp": datetime.utcnow(),
                    "priority": priority.name
                })

                return message

        return None

    def add_pending_response(self, correlation_id: str, future: asyncio.Future):
        """Track pending response for correlation"""
        self.pending_responses[correlation_id] = future

    def complete_pending_response(self, correlation_id: str, result: Any):
        """Complete a pending response"""
        if correlation_id in self.pending_responses:
            future = self.pending_responses.pop(correlation_id)
            if not future.done():
                future.set_result(result)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        queue_sizes = {priority.name: len(queue) for priority, queue in self.queues.items()}
        return {
            **self.stats,
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "pending_responses": len(self.pending_responses)
        }


class SkillStateManager:
    """Manages state of skills within the agent"""

    def __init__(self):
        self.skills: Dict[str, SkillCapability] = {}
        self.skill_dependencies: Dict[str, List[str]] = {}
        self.skill_load_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.coordination_rules: Dict[str, Callable] = {}

    def register_skill(self, capability: SkillCapability):
        """Register a skill with its capabilities"""
        self.skills[capability.skill_name] = capability
        self.skill_dependencies[capability.skill_name] = capability.dependencies
        logger.info(f"Registered skill: {capability.skill_name}")

    def update_skill_state(self, skill_name: str, state: SkillState, load_factor: float = None):
        """Update skill state and load"""
        if skill_name in self.skills:
            self.skills[skill_name].state = state
            self.skills[skill_name].last_activity = datetime.utcnow()

            if load_factor is not None:
                self.skills[skill_name].load_factor = load_factor
                # Keep load history (last 100 readings)
                history = self.skill_load_history[skill_name]
                history.append((datetime.utcnow(), load_factor))
                if len(history) > 100:
                    history.pop(0)

    def get_available_skills(self, required_capability: str = None) -> List[SkillCapability]:
        """Get skills that are available and optionally provide a capability"""
        available = []

        for skill in self.skills.values():
            if skill.state in [SkillState.IDLE, SkillState.BUSY] and skill.load_factor < 0.9:
                if required_capability is None or required_capability in skill.provides:
                    available.append(skill)

        # Sort by load factor (least loaded first)
        return sorted(available, key=lambda s: s.load_factor)

    def get_skill_dependencies(self, skill_name: str) -> List[str]:
        """Get dependencies for a skill"""
        return self.skill_dependencies.get(skill_name, [])

    def can_skill_execute(self, skill_name: str) -> bool:
        """Check if skill can execute (dependencies available)"""
        if skill_name not in self.skills:
            return False

        dependencies = self.get_skill_dependencies(skill_name)
        for dep in dependencies:
            if dep not in self.skills or self.skills[dep].state == SkillState.OFFLINE:
                return False

        return True

    def add_coordination_rule(self, rule_name: str, rule_func: Callable):
        """Add a coordination rule for skill interaction"""
        self.coordination_rules[rule_name] = rule_func

    def apply_coordination_rules(self, message: SkillMessage) -> bool:
        """Apply coordination rules to determine if message should be processed"""
        for rule_name, rule_func in self.coordination_rules.items():
            try:
                if not rule_func(message, self.skills):
                    logger.debug(f"Message {message.id} blocked by rule {rule_name}")
                    return False
            except Exception as e:
                logger.warning(f"Error applying coordination rule {rule_name}: {e}")

        return True


class MCPSkillCoordinator:
    """Main coordinator for MCP-based skill communication"""

    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.message_queue = SkillMessageQueue()
        self.state_manager = SkillStateManager()
        self.message_handlers: Dict[SkillMessageType, Callable] = {}
        self.is_running = False
        self.processing_task = None

        # Setup default handlers
        self._setup_default_handlers()

        # Discover skills
        self._discover_skills()

        logger.info(f"MCP Skill Coordinator initialized for {getattr(agent_instance, 'agent_id', 'unknown')}")

    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers[SkillMessageType.SKILL_REQUEST] = self._handle_skill_request
        self.message_handlers[SkillMessageType.SKILL_RESPONSE] = self._handle_skill_response
        self.message_handlers[SkillMessageType.SKILL_NOTIFICATION] = self._handle_skill_notification
        self.message_handlers[SkillMessageType.SKILL_STATUS] = self._handle_skill_status
        self.message_handlers[SkillMessageType.TOOL_CALL] = self._handle_tool_call

    def _discover_skills(self):
        """Discover skills with MCP capabilities in the agent"""

        for attr_name in dir(self.agent):
            try:
                attr = getattr(self.agent, attr_name)
                if not callable(attr):
                    continue

                # Check for MCP decorators
                tools = []
                resources = []
                prompts = []
                dependencies = []
                provides = []

                if hasattr(attr, '_mcp_tool'):
                    tools.append(attr._mcp_tool)
                    provides.append(attr._mcp_tool['name'])

                if hasattr(attr, '_mcp_resource'):
                    resources.append(attr._mcp_resource)
                    provides.append(attr._mcp_resource['name'])

                if hasattr(attr, '_mcp_prompt'):
                    prompts.append(attr._mcp_prompt)
                    provides.append(attr._mcp_prompt['name'])

                # Check for skill dependencies (custom attribute)
                if hasattr(attr, '_skill_dependencies'):
                    dependencies = attr._skill_dependencies

                # Register if it has any MCP capabilities
                if tools or resources or prompts:
                    capability = SkillCapability(
                        skill_name=attr_name,
                        tools=tools,
                        resources=resources,
                        prompts=prompts,
                        dependencies=dependencies,
                        provides=provides
                    )
                    self.state_manager.register_skill(capability)

            except Exception as e:
                logger.warning(f"Error discovering skill {attr_name}: {e}")

    async def start(self):
        """Start the skill coordinator"""
        if self.is_running:
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._message_processing_loop())
        logger.info("MCP Skill Coordinator started")

    async def stop(self):
        """Stop the skill coordinator"""
        if not self.is_running:
            return

        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("MCP Skill Coordinator stopped")

    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.is_running:
            try:
                message = await self.message_queue.dequeue()
                if message:
                    await self._process_message(message)
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_message(self, message: SkillMessage):
        """Process a single skill message"""
        start_time = datetime.utcnow()

        try:
            # Apply coordination rules
            if not self.state_manager.apply_coordination_rules(message):
                return

            # Get handler for message type
            handler = self.message_handlers.get(message.message_type)
            if not handler:
                logger.warning(f"No handler for message type {message.message_type}")
                return

            # Process the message
            await handler(message)

            # Update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            stats = self.message_queue.stats
            total_processed = stats["messages_processed"] + 1
            avg_time = stats["average_processing_time"]
            stats["average_processing_time"] = ((avg_time * stats["messages_processed"]) + processing_time) / total_processed
            stats["messages_processed"] = total_processed

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")

    async def _handle_skill_request(self, message: SkillMessage):
        """Handle skill request message"""
        # Find the target skill
        target_skill = message.receiver_skill
        if target_skill not in self.state_manager.skills:
            logger.warning(f"Target skill {target_skill} not found")
            return

        # Get the skill method
        try:
            skill_method = getattr(self.agent, target_skill)

            # Extract parameters
            params = message.params or {}

            # Call the skill
            if inspect.iscoroutinefunction(skill_method):
                result = await skill_method(**params)
            else:
                result = skill_method(**params)

            # Send response
            response = SkillMessage(
                method="skills/response",
                message_type=SkillMessageType.SKILL_RESPONSE,
                sender_skill=target_skill,
                receiver_skill=message.sender_skill,
                correlation_id=message.correlation_id,
                reply_to=message.id,
                params={"result": result, "success": True}
            )

            await self.send_message(response)

        except Exception as e:
            logger.error(f"Error executing skill {target_skill}: {e}")

            # Send error response
            error_response = SkillMessage(
                method="skills/response",
                message_type=SkillMessageType.SKILL_RESPONSE,
                sender_skill=target_skill,
                receiver_skill=message.sender_skill,
                correlation_id=message.correlation_id,
                reply_to=message.id,
                params={"error": str(e), "success": False}
            )

            await self.send_message(error_response)

    async def _handle_skill_response(self, message: SkillMessage):
        """Handle skill response message"""
        correlation_id = message.correlation_id
        if correlation_id:
            result = message.params.get("result") if message.params.get("success") else message.params.get("error")
            self.message_queue.complete_pending_response(correlation_id, result)

    async def _handle_skill_notification(self, message: SkillMessage):
        """Handle skill notification message"""
        # Broadcast notification to interested skills
        logger.info(f"Skill notification from {message.sender_skill}: {message.params}")

    async def _handle_skill_status(self, message: SkillMessage):
        """Handle skill status update"""
        skill_name = message.sender_skill
        params = message.params or {}

        new_state = SkillState(params.get("state", "idle"))
        load_factor = params.get("load_factor", 0.0)

        self.state_manager.update_skill_state(skill_name, new_state, load_factor)

    async def _handle_tool_call(self, message: SkillMessage):
        """Handle MCP tool call within skill coordination"""
        await self._handle_skill_request(message)  # Treat as skill request

    async def send_message(self, message: SkillMessage) -> Optional[Any]:
        """Send a message to another skill"""

        # For requests, set up response waiting
        future = None
        if message.message_type == SkillMessageType.SKILL_REQUEST:
            future = asyncio.Future()
            self.message_queue.add_pending_response(message.correlation_id, future)

        # Enqueue the message
        success = await self.message_queue.enqueue(message)
        if not success:
            if future:
                future.set_exception(Exception("Failed to enqueue message"))
            return None

        self.message_queue.stats["messages_sent"] += 1

        # Wait for response if it's a request
        if future:
            try:
                return await asyncio.wait_for(future, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response to message {message.id}")
                return None

        return True

    async def call_skill(
        self,
        target_skill: str,
        method: str,
        params: Dict[str, Any] = None,
        priority: SkillPriority = SkillPriority.NORMAL,
        timeout: float = 30.0
    ) -> Any:
        """High-level interface to call another skill"""

        message = SkillMessage(
            method=method,
            message_type=SkillMessageType.SKILL_REQUEST,
            sender_skill="coordinator",  # Or get from call stack
            receiver_skill=target_skill,
            priority=priority,
            params=params,
            expires_at=datetime.utcnow() + timedelta(seconds=timeout)
        )

        return await self.send_message(message)

    def get_skill_capabilities(self) -> Dict[str, SkillCapability]:
        """Get all registered skill capabilities"""
        return self.state_manager.skills.copy()

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        return {
            "queue_stats": self.message_queue.get_stats(),
            "skills": {name: {
                "state": skill.state.value,
                "load_factor": skill.load_factor,
                "tools_count": len(skill.tools),
                "resources_count": len(skill.resources),
                "prompts_count": len(skill.prompts),
                "last_activity": skill.last_activity.isoformat() if skill.last_activity else None
            } for name, skill in self.state_manager.skills.items()},
            "coordination_active": self.is_running
        }


# Mixin to add skill coordination to agents
class MCPSkillCoordinationMixin:
    """Mixin to add MCP skill coordination to A2A agents"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_coordinator: Optional[MCPSkillCoordinator] = None

    def initialize_skill_coordinator(self):
        """Initialize MCP skill coordinator"""
        self.skill_coordinator = MCPSkillCoordinator(self)
        logger.info(f"MCP skill coordinator initialized for agent {getattr(self, 'agent_id', 'unknown')}")
        return self.skill_coordinator

    async def start_skill_coordinator(self):
        """Start the skill coordinator"""
        if not self.skill_coordinator:
            self.initialize_skill_coordinator()

        await self.skill_coordinator.start()

    async def stop_skill_coordinator(self):
        """Stop the skill coordinator"""
        if self.skill_coordinator:
            await self.skill_coordinator.stop()

    async def call_skill(self, target_skill: str, method: str, **kwargs) -> Any:
        """Convenience method to call another skill"""
        if not self.skill_coordinator:
            raise RuntimeError("Skill coordinator not initialized")

        return await self.skill_coordinator.call_skill(target_skill, method, kwargs)

    def get_skill_coordination_status(self) -> Dict[str, Any]:
        """Get skill coordination status"""
        if not self.skill_coordinator:
            return {"error": "Skill coordinator not initialized"}

        return self.skill_coordinator.get_coordination_stats()


# Decorators for skill dependencies and coordination
def skill_depends_on(*dependencies: str):
    """Decorator to specify skill dependencies"""
    def decorator(func):
        func._skill_dependencies = list(dependencies)
        return func
    return decorator


def skill_provides(*services: str):
    """Decorator to specify what services a skill provides"""
    def decorator(func):
        func._skill_provides = list(services)
        return func
    return decorator


def coordination_rule(rule_name: str):
    """Decorator to define a coordination rule"""
    def decorator(func):
        func._coordination_rule = rule_name
        return func
    return decorator

"""
A2A-compliant distributed task coordination system
Integrates with existing A2A task management, message protocol, and agent discovery
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

from ..a2a.sdk.types import A2AMessage, MessagePart, MessageRole, TaskStatus
from ..a2a.sdk.agentBase import A2AAgentBase
from ..a2a.core.telemetry import trace_async, add_span_attributes
from .task_persistence import TaskPersistenceManager, PersistedTask, TaskStatus as PersistTaskStatus
from .agentTaskTracker import AgentTaskTracker
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class TaskDistributionStrategy(str, Enum):
    """Task distribution strategies"""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    TRUST_BASED = "trust_based"
    LOCALITY_AWARE = "locality_aware"


class A2ADistributedTask:
    """A2A-compliant distributed task"""

    def __init__(
        self,
        task_id: str,
        source_agent_id: str,
        task_type: str,
        payload: Dict[str, Any],
        context_id: Optional[str] = None,
        required_capabilities: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.task_id = task_id
        self.source_agent_id = source_agent_id
        self.task_type = task_type
        self.payload = payload
        self.context_id = context_id or str(uuid4())
        self.required_capabilities = required_capabilities or []
        self.metadata = metadata or {}
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.assigned_agent_id: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def to_a2a_message(self, target_agent_id: str) -> A2AMessage:
        """Convert task to A2A message for distribution"""
        return A2AMessage(
            messageId=str(uuid4()),
            role=MessageRole.SYSTEM,
            parts=[
                MessagePart(
                    kind="task_assignment",
                    data={
                        "task_id": self.task_id,
                        "task_type": self.task_type,
                        "payload": self.payload,
                        "source_agent_id": self.source_agent_id,
                        "required_capabilities": self.required_capabilities,
                        "metadata": self.metadata
                    }
                )
            ],
            taskId=self.task_id,
            contextId=self.context_id
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "task_id": self.task_id,
            "source_agent_id": self.source_agent_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "context_id": self.context_id,
            "required_capabilities": self.required_capabilities,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "assigned_agent_id": self.assigned_agent_id,
            "result": self.result,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2ADistributedTask':
        """Deserialize from dictionary"""
        task = cls(
            task_id=data["task_id"],
            source_agent_id=data["source_agent_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            context_id=data["context_id"],
            required_capabilities=data["required_capabilities"],
            metadata=data["metadata"]
        )
        task.status = TaskStatus(data["status"])
        task.created_at = datetime.fromisoformat(data["created_at"])
        task.assigned_agent_id = data.get("assigned_agent_id")
        task.result = data.get("result")
        task.error = data.get("error")
        return task


class A2ADistributedCoordinator:
    """A2A-compliant distributed task coordinator"""

    def __init__(
        self,
        agent_base: A2AAgentBase,
        redis_config: RedisConfig = None,
        distribution_strategy: TaskDistributionStrategy = TaskDistributionStrategy.CAPABILITY_MATCH
    ):
        self.agent_base = agent_base
        self.redis_client = RedisClient(redis_config or RedisConfig())
        self.distribution_strategy = distribution_strategy

        # Integrate with existing A2A components
        self.task_persistence = TaskPersistenceManager()
        self.task_tracker = AgentTaskTracker(
            agent_base.agent_id,
            agent_base.name
        )

        # Distributed state
        self.active_tasks: Dict[str, A2ADistributedTask] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.running = False

        # Redis keys for A2A task coordination
        self.namespace = f"a2a:coordination:{agent_base.agent_id}"
        self.task_queue_key = f"{self.namespace}:tasks"
        self.task_data_key = f"{self.namespace}:task_data"
        self.agent_capabilities_key = "a2a:agents:capabilities"
        self.task_assignments_key = f"{self.namespace}:assignments"

        # Register A2A message handlers for coordination
        self._register_coordination_handlers()

    def _register_coordination_handlers(self):
        """Register message handlers for task coordination"""

        @self.agent_base.handlers.setdefault('task_assignment', lambda: None)
        async def handle_task_assignment(self, message: A2AMessage, context_id: str):
            """Handle incoming task assignment"""
            task_data = message.parts[0].data
            task = A2ADistributedTask(
                task_id=task_data["task_id"],
                source_agent_id=task_data["source_agent_id"],
                task_type=task_data["task_type"],
                payload=task_data["payload"],
                context_id=context_id,
                required_capabilities=task_data.get("required_capabilities", []),
                metadata=task_data.get("metadata", {})
            )

            return await self._accept_task_assignment(task)

        @self.agent_base.handlers.setdefault('task_result', lambda: None)
        async def handle_task_result(self, message: A2AMessage, context_id: str):
            """Handle task completion result"""
            result_data = message.parts[0].data
            task_id = result_data["task_id"]

            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus(result_data["status"])
                task.result = result_data.get("result")
                task.error = result_data.get("error")

                await self._handle_task_completion(task)

            return {"success": True, "task_id": task_id}

        @self.agent_base.handlers.setdefault('help_request', lambda: None)
        async def handle_help_request(self, message: A2AMessage, context_id: str):
            """Handle help request for task failure"""
            help_data = message.parts[0].data
            task_id = help_data["task_id"]

            return await self._provide_task_help(task_id, help_data)

    async def initialize(self):
        """Initialize the distributed coordinator"""
        await self.redis_client.initialize()
        await self.task_persistence.initialize()

        # Register agent capabilities in shared registry
        await self._register_agent_capabilities()

        # Start coordination background tasks
        self.running = True
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._heartbeat_loop())

        logger.info(f"A2A distributed coordinator initialized for {self.agent_base.agent_id}")

    async def shutdown(self):
        """Shutdown the coordinator"""
        self.running = False

        # Complete active tasks or reassign them
        for task_id, task in self.active_tasks.items():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                await self._reassign_task(task)

        await self.redis_client.close()
        logger.info(f"A2A distributed coordinator shut down for {self.agent_base.agent_id}")

    @trace_async("distribute_task")
    async def distribute_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: List[str] = None,
        context_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Distribute a task across the A2A network"""

        task_id = str(uuid4())

        # Create distributed task
        task = A2ADistributedTask(
            task_id=task_id,
            source_agent_id=self.agent_base.agent_id,
            task_type=task_type,
            payload=payload,
            context_id=context_id,
            required_capabilities=required_capabilities or [],
            metadata=metadata or {}
        )

        add_span_attributes({
            "task.id": task_id,
            "task.type": task_type,
            "task.capabilities": required_capabilities or [],
            "agent.id": self.agent_base.agent_id
        })

        # Store task locally and in distributed state
        self.active_tasks[task_id] = task
        await self.redis_client.hset(self.task_data_key, task_id, json.dumps(task.to_dict()))

        # Create local task record for A2A agent compatibility
        await self.agent_base.create_task(task_type, payload)

        # Create persistent task record
        persistent_task = PersistedTask(
            task_id=task_id,
            agent_id=self.agent_base.agent_id,
            task_type=task_type,
            payload=payload,
            status=PersistTaskStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata
        )
        await self.task_persistence.save_task(persistent_task)

        # Track in task tracker
        self.task_tracker.add_task(task_id, task_type, payload)

        # Find and assign to capable agent
        target_agent_id = await self._find_target_agent(task)
        if target_agent_id:
            await self._assign_task(task, target_agent_id)
        else:
            # Queue for later assignment
            await self.redis_client.lpush(self.task_queue_key, task_id)
            logger.info(f"Task {task_id} queued - no capable agent found")

        logger.info(f"Distributed task {task_id} of type {task_type}")
        return task_id

    async def _find_target_agent(self, task: A2ADistributedTask) -> Optional[str]:
        """Find the best agent to handle the task"""

        if self.distribution_strategy == TaskDistributionStrategy.CAPABILITY_MATCH:
            return await self._find_by_capabilities(task.required_capabilities)
        elif self.distribution_strategy == TaskDistributionStrategy.LOAD_BALANCED:
            return await self._find_least_loaded_agent(task.required_capabilities)
        elif self.distribution_strategy == TaskDistributionStrategy.TRUST_BASED:
            return await self._find_most_trusted_agent(task.required_capabilities)
        else:
            return await self._find_round_robin_agent(task.required_capabilities)

    async def _find_by_capabilities(self, required_capabilities: List[str]) -> Optional[str]:
        """Find agent with required capabilities"""
        try:
            # Get agent registry from A2A network
            from ..a2a.storage import get_distributed_storage

            registry = await get_distributed_storage()
            agents = await registry.discover_agents(
                capabilities=required_capabilities,
                exclude_agents=[self.agent_base.agent_id]
            )

            if agents and "agents" in agents and agents["agents"]:
                # Select agent with best capability match
                best_agent = max(
                    agents["agents"],
                    key=lambda a: len(set(a.get("capabilities", [])) & set(required_capabilities))
                )
                return best_agent["agent_id"]

            return None

        except Exception as e:
            logger.error(f"Error finding capable agent: {e}")
            return None

    async def _find_least_loaded_agent(self, required_capabilities: List[str]) -> Optional[str]:
        """Find least loaded agent with capabilities"""
        capable_agents = await self._find_by_capabilities(required_capabilities)
        if not capable_agents:
            return None

        # Get load metrics from agents
        # This would integrate with A2A monitoring system
        # For now, return the first capable agent
        return capable_agents

    async def _find_most_trusted_agent(self, required_capabilities: List[str]) -> Optional[str]:
        """Find most trusted agent with capabilities"""
        try:
            from ..a2a.storage import get_distributed_storage

            registry = await get_distributed_storage()
            agents = await registry.discover_agents(
                capabilities=required_capabilities,
                min_trust_score=0.8,
                exclude_agents=[self.agent_base.agent_id]
            )

            if agents and "agents" in agents and agents["agents"]:
                # Select agent with highest trust score
                best_agent = max(
                    agents["agents"],
                    key=lambda a: a.get("trust_score", 0.0)
                )
                return best_agent["agent_id"]

            return None

        except Exception as e:
            logger.error(f"Error finding trusted agent: {e}")
            return None

    async def _assign_task(self, task: A2ADistributedTask, target_agent_id: str):
        """Assign task to target agent via A2A message"""

        task.assigned_agent_id = target_agent_id
        task.status = TaskStatus.RUNNING

        # Update distributed state
        await self.redis_client.hset(
            self.task_assignments_key,
            task.task_id,
            target_agent_id
        )
        await self.redis_client.hset(
            self.task_data_key,
            task.task_id,
            json.dumps(task.to_dict())
        )

        # Create A2A message for task assignment
        message = task.to_a2a_message(target_agent_id)

        # Send via A2A agent's messaging system
        try:
            # Use agent's built-in message processing
            signed_request = await self.agent_base.send_signed_request(
                target_agent_id=target_agent_id,
                method="POST",
                path="/messages",
                body={"message": message.model_dump(), "contextId": task.context_id}
            )

            logger.info(f"Assigned task {task.task_id} to agent {target_agent_id}")

        except Exception as e:
            logger.error(f"Failed to assign task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

            # Trigger help-seeking mechanism
            await self._request_help_for_task(task, str(e))

    async def _accept_task_assignment(self, task: A2ADistributedTask) -> Dict[str, Any]:
        """Accept an incoming task assignment"""

        # Check if we can handle this task
        can_handle = await self._can_handle_task(task)
        if not can_handle:
            return {
                "success": False,
                "error": "Agent cannot handle task with required capabilities",
                "task_id": task.task_id
            }

        # Add to local tracking
        self.active_tasks[task.task_id] = task
        await self.agent_base.create_task(task.task_type, task.payload)

        # Create persistent record
        persistent_task = PersistedTask(
            task_id=task.task_id,
            agent_id=self.agent_base.agent_id,
            task_type=task.task_type,
            payload=task.payload,
            status=PersistTaskStatus.IN_PROGRESS,
            created_at=task.created_at,
            updated_at=datetime.utcnow(),
            metadata=task.metadata
        )
        await self.task_persistence.save_task(persistent_task)

        # Execute task asynchronously
        asyncio.create_task(self._execute_assigned_task(task))

        return {
            "success": True,
            "task_id": task.task_id,
            "assigned_to": self.agent_base.agent_id
        }

    async def _execute_assigned_task(self, task: A2ADistributedTask):
        """Execute an assigned task"""

        try:
            task.status = TaskStatus.RUNNING

            # Find handler for task type
            handler = self.agent_base.handlers.get(task.task_type)
            if not handler:
                # Try to find skill
                skill = self.agent_base.skills.get(task.task_type)
                if skill:
                    handler = getattr(self.agent_base, skill.method_name)

            if not handler:
                raise ValueError(f"No handler found for task type: {task.task_type}")

            # Execute with A2A agent's built-in retry and error handling
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload, task.metadata)
            else:
                result = handler(task.payload, task.metadata)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.result = result

            # Update local agent task
            await self.agent_base.update_task(task.task_id, TaskStatus.COMPLETED, result)

            # Update persistent task
            await self.task_persistence.update_task_status(
                task.task_id,
                PersistTaskStatus.COMPLETED
            )

            # Send result back to source agent
            await self._send_task_result(task)

        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {e}")

            task.status = TaskStatus.FAILED
            task.error = str(e)

            # Update local and persistent state
            await self.agent_base.update_task(task.task_id, TaskStatus.FAILED, error=str(e))
            await self.task_persistence.update_task_status(
                task.task_id,
                PersistTaskStatus.FAILED,
                str(e)
            )

            # Send failure result
            await self._send_task_result(task)

            # Request help via A2A help-seeking system
            await self._request_help_for_task(task, str(e))

    async def _send_task_result(self, task: A2ADistributedTask):
        """Send task result back to source agent"""

        result_message = A2AMessage(
            messageId=str(uuid4()),
            role=MessageRole.AGENT,
            parts=[
                MessagePart(
                    kind="task_result",
                    data={
                        "task_id": task.task_id,
                        "status": task.status.value,
                        "result": task.result,
                        "error": task.error,
                        "completed_by": self.agent_base.agent_id
                    }
                )
            ],
            taskId=task.task_id,
            contextId=task.context_id
        )

        try:
            # Send back to source agent
            signed_request = await self.agent_base.send_signed_request(
                target_agent_id=task.source_agent_id,
                method="POST",
                path="/messages",
                body={
                    "message": result_message.model_dump(),
                    "contextId": task.context_id
                }
            )

            logger.info(f"Sent result for task {task.task_id} to {task.source_agent_id}")

        except Exception as e:
            logger.error(f"Failed to send task result: {e}")

    async def _request_help_for_task(self, task: A2ADistributedTask, error: str):
        """Request help for failed task using A2A help-seeking system"""

        help_request_id = self.task_tracker.create_help_request(
            task_id=task.task_id,
            problem_type="execution_failure",
            problem_description=error,
            target_agent="agent_manager",  # or find specific helper
            urgency="medium"
        )

        help_message = A2AMessage(
            messageId=str(uuid4()),
            role=MessageRole.AGENT,
            parts=[
                MessagePart(
                    kind="help_request",
                    data={
                        "help_request_id": help_request_id,
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "error": error,
                        "requesting_agent": self.agent_base.agent_id,
                        "capabilities_needed": task.required_capabilities
                    }
                )
            ],
            contextId=task.context_id
        )

        try:
            # Send to agent manager or specific helper
            signed_request = await self.agent_base.send_signed_request(
                target_agent_id="agent_manager",
                method="POST",
                path="/messages",
                body={
                    "message": help_message.model_dump(),
                    "contextId": task.context_id
                }
            )

            logger.info(f"Requested help for task {task.task_id}")

        except Exception as e:
            logger.error(f"Failed to request help: {e}")

    async def _can_handle_task(self, task: A2ADistributedTask) -> bool:
        """Check if this agent can handle the task"""

        # Check required capabilities
        agent_capabilities = [cap.name for cap in self.agent_base.capabilities]

        if task.required_capabilities:
            has_capabilities = all(
                cap in agent_capabilities for cap in task.required_capabilities
            )
            if not has_capabilities:
                return False

        # Check if we have a handler for this task type
        has_handler = (
            task.task_type in self.agent_base.handlers or
            task.task_type in self.agent_base.skills
        )

        return has_handler

    async def _register_agent_capabilities(self):
        """Register agent capabilities in shared registry"""
        capabilities_data = {
            "agent_id": self.agent_base.agent_id,
            "capabilities": [cap.name for cap in self.agent_base.capabilities],
            "skills": list(self.agent_base.skills.keys()),
            "handlers": list(self.agent_base.handlers.keys()),
            "updated_at": datetime.utcnow().isoformat()
        }

        await self.redis_client.hset(
            self.agent_capabilities_key,
            self.agent_base.agent_id,
            json.dumps(capabilities_data)
        )

    async def _coordination_loop(self):
        """Background coordination loop"""
        while self.running:
            try:
                # Process queued tasks
                task_id = await self.redis_client.rpop(self.task_queue_key)
                if task_id:
                    task_data = await self.redis_client.hget(self.task_data_key, task_id)
                    if task_data:
                        task = A2ADistributedTask.from_dict(json.loads(task_data))
                        target_agent = await self._find_target_agent(task)
                        if target_agent:
                            await self._assign_task(task, target_agent)

                # Cleanup completed tasks
                await self._cleanup_completed_tasks()

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(30)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                await self._register_agent_capabilities()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks older than 1 hour"""
        cutoff = datetime.utcnow() - timedelta(hours=1)

        completed_tasks = []
        for task_id, task in list(self.active_tasks.items()):
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                task.created_at < cutoff):
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            # Remove from active tracking
            del self.active_tasks[task_id]

            # Clean up Redis data
            await self.redis_client.hdel(self.task_data_key, task_id)
            await self.redis_client.hdel(self.task_assignments_key, task_id)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        return {
            "agent_id": self.agent_base.agent_id,
            "active_tasks": len(self.active_tasks),
            "task_states": {
                status.value: len([t for t in self.active_tasks.values() if t.status == status])
                for status in TaskStatus
            },
            "distribution_strategy": self.distribution_strategy.value,
            "capabilities": [cap.name for cap in self.agent_base.capabilities],
            "registered_handlers": len(self.agent_base.handlers),
            "registered_skills": len(self.agent_base.skills)
        }


# Integration helper for existing A2A agents
async def initialize_a2a_coordination(agent: A2AAgentBase) -> A2ADistributedCoordinator:
    """Initialize A2A distributed coordination for an agent"""
    coordinator = A2ADistributedCoordinator(agent)
    await coordinator.initialize()
    return coordinator

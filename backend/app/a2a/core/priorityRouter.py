"""
A2A Priority-Based Message Router
Implements intelligent message routing based on priority, agent capabilities, and system load
Supports dynamic priority adjustment, preemption, and quality of service (QoS) guarantees
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq

from .messageQueue import MessagePriority
from .loadBalancer import AgentLoadBalancer, AgentInstance, get_load_balancer

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies"""

    PRIORITY_FIRST = "priority_first"  # Strict priority ordering
    WEIGHTED_FAIR = "weighted_fair"  # Fair queuing with priority weights
    DEADLINE_BASED = "deadline_based"  # EDF (Earliest Deadline First)
    ADAPTIVE = "adaptive"  # Dynamic strategy based on system state
    PREEMPTIVE = "preemptive"  # Can preempt lower priority tasks


class QoSLevel(Enum):
    """Quality of Service levels"""

    BEST_EFFORT = "best_effort"  # No guarantees
    GUARANTEED = "guaranteed"  # Guaranteed delivery and timing
    PREMIUM = "premium"  # Highest priority, dedicated resources


@dataclass
class RoutingPolicy:
    """Routing policy configuration"""

    strategy: RoutingStrategy = RoutingStrategy.PRIORITY_FIRST
    enable_preemption: bool = False
    max_queue_time: int = 300  # seconds
    priority_boost_interval: int = 60  # seconds
    enable_qos: bool = True
    enable_load_balancing: bool = True
    enable_deadline_routing: bool = True
    max_retries: int = 3


@dataclass
class RoutingRequest:
    """Request for routing a message"""

    message_id: str
    agent_id: str
    priority: MessagePriority
    context_id: str
    payload: Dict[str, Any]
    qos_level: QoSLevel = QoSLevel.BEST_EFFORT
    deadline: Optional[datetime] = None
    required_capabilities: List[str] = field(default_factory=list)
    preferred_instances: List[str] = field(default_factory=list)
    excluded_instances: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_priority_value(self) -> int:
        """Get numeric priority value (higher is more important)"""
        priority_map = {
            MessagePriority.LOW: 1,
            MessagePriority.MEDIUM: 2,
            MessagePriority.HIGH: 3,
            MessagePriority.CRITICAL: 4,
        }
        base_priority = priority_map.get(self.priority, 2)

        # QoS boost
        if self.qos_level == QoSLevel.PREMIUM:
            base_priority += 2
        elif self.qos_level == QoSLevel.GUARANTEED:
            base_priority += 1

        return base_priority


@dataclass
class RoutingDecision:
    """Result of routing decision"""

    request_id: str
    agent_id: str
    instance_id: str
    instance_url: str
    priority_score: float
    estimated_wait_time: float
    routing_reason: str
    alternative_routes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QueueMetrics:
    """Metrics for a priority queue"""

    queue_depth: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    messages_routed: int = 0
    messages_preempted: int = 0
    messages_expired: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PriorityRouter:
    """
    Priority-based message router for A2A agents
    Routes messages based on priority, deadlines, and system load
    """

    def __init__(self, policy: RoutingPolicy = None):
        self.policy = policy or RoutingPolicy()
        self.load_balancer: Optional[AgentLoadBalancer] = None

        # Priority queues per agent
        self.priority_queues: Dict[str, List[Tuple[float, RoutingRequest]]] = defaultdict(list)
        self.queue_metrics: Dict[str, QueueMetrics] = defaultdict(QueueMetrics)

        # Active routing decisions
        self.active_routes: Dict[str, RoutingDecision] = {}
        self.routing_history: deque = deque(maxlen=1000)

        # Preemption tracking
        self.preemptible_tasks: Dict[str, Set[str]] = defaultdict(set)
        self.preemption_count: Dict[str, int] = defaultdict(int)

        # QoS guarantees
        self.qos_reservations: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.routing_times: deque = deque(maxlen=100)
        self.success_count = 0
        self.failure_count = 0

        self._lock = asyncio.Lock()
        self._running = False
        self._processor_task = None

        logger.info(f"Initialized priority router with strategy: {self.policy.strategy.value}")

    async def initialize(self):
        """Initialize router and start background processing"""
        self.load_balancer = await get_load_balancer()
        self._running = True
        self._processor_task = asyncio.create_task(self._process_queues())
        logger.info("Priority router initialized")

    async def shutdown(self):
        """Shutdown router and cleanup resources"""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Priority router shutdown complete")

    async def route_message(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route a message based on priority and routing policy
        Returns routing decision with selected instance
        """
        start_time = time.time()

        try:
            async with self._lock:
                # Check for QoS reservations
                if request.qos_level == QoSLevel.PREMIUM:
                    decision = await self._route_premium_message(request)
                    if decision:
                        self.success_count += 1
                        self.routing_times.append(time.time() - start_time)
                        return decision

                # Apply routing strategy
                if self.policy.strategy == RoutingStrategy.PRIORITY_FIRST:
                    decision = await self._route_priority_first(request)
                elif self.policy.strategy == RoutingStrategy.WEIGHTED_FAIR:
                    decision = await self._route_weighted_fair(request)
                elif self.policy.strategy == RoutingStrategy.DEADLINE_BASED:
                    decision = await self._route_deadline_based(request)
                elif self.policy.strategy == RoutingStrategy.ADAPTIVE:
                    decision = await self._route_adaptive(request)
                elif self.policy.strategy == RoutingStrategy.PREEMPTIVE:
                    decision = await self._route_preemptive(request)
                else:
                    decision = await self._route_priority_first(request)

                if decision:
                    self.success_count += 1
                    self.active_routes[request.message_id] = decision
                    self.routing_history.append(
                        {
                            "timestamp": datetime.utcnow(),
                            "request_id": request.message_id,
                            "decision": decision,
                        }
                    )
                else:
                    self.failure_count += 1
                    logger.warning(f"Failed to route message {request.message_id}")

                self.routing_times.append(time.time() - start_time)
                return decision

        except Exception as e:
            logger.error(f"Error routing message {request.message_id}: {e}")
            self.failure_count += 1
            raise

    async def _route_priority_first(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Route based on strict priority ordering"""
        # Get available instances
        instances = await self._get_available_instances(request)
        if not instances:
            return None

        # Calculate priority score
        priority_score = self._calculate_priority_score(request)

        # Select instance with least queue depth for this priority
        best_instance = None
        min_queue_depth = float("inf")

        for instance in instances:
            queue_key = f"{request.agent_id}:{instance.instance_id}"
            queue_depth = len(
                [r for _, r in self.priority_queues[queue_key] if r.priority == request.priority]
            )

            if queue_depth < min_queue_depth:
                min_queue_depth = queue_depth
                best_instance = instance

        if not best_instance:
            return None

        # Add to priority queue
        queue_key = f"{request.agent_id}:{best_instance.instance_id}"
        heapq.heappush(
            self.priority_queues[queue_key], (-priority_score, request)  # Negative for max heap
        )

        # Update metrics
        metrics = self.queue_metrics[queue_key]
        metrics.queue_depth = len(self.priority_queues[queue_key])
        metrics.last_updated = datetime.utcnow()

        return RoutingDecision(
            request_id=request.message_id,
            agent_id=request.agent_id,
            instance_id=best_instance.instance_id,
            instance_url=best_instance.base_url,
            priority_score=priority_score,
            estimated_wait_time=min_queue_depth * 0.5,  # Rough estimate
            routing_reason=f"Priority-first routing to instance with {min_queue_depth} queued messages",
        )

    async def _route_weighted_fair(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Route using weighted fair queuing"""
        instances = await self._get_available_instances(request)
        if not instances:
            return None

        # Calculate weighted queue depths
        weighted_instances = []
        for instance in instances:
            queue_key = f"{request.agent_id}:{instance.instance_id}"

            # Calculate weighted depth based on priority distribution
            weighted_depth = 0.0
            for priority in MessagePriority:
                count = len(
                    [r for _, r in self.priority_queues[queue_key] if r.priority == priority]
                )
                weight = self._get_priority_weight(priority)
                weighted_depth += count * weight

            weighted_instances.append((weighted_depth, instance))

        # Select instance with lowest weighted queue depth
        weighted_instances.sort(key=lambda x: x[0])
        best_instance = weighted_instances[0][1]

        # Add to queue
        priority_score = self._calculate_priority_score(request)
        queue_key = f"{request.agent_id}:{best_instance.instance_id}"
        heapq.heappush(self.priority_queues[queue_key], (-priority_score, request))

        return RoutingDecision(
            request_id=request.message_id,
            agent_id=request.agent_id,
            instance_id=best_instance.instance_id,
            instance_url=best_instance.base_url,
            priority_score=priority_score,
            estimated_wait_time=weighted_instances[0][0] * 0.3,
            routing_reason=f"Weighted fair routing with depth {weighted_instances[0][0]:.2f}",
        )

    async def _route_deadline_based(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Route based on earliest deadline first (EDF)"""
        if not request.deadline:
            # Fall back to priority-based routing
            return await self._route_priority_first(request)

        instances = await self._get_available_instances(request)
        if not instances:
            return None

        # Calculate time until deadline
        time_to_deadline = (request.deadline - datetime.utcnow()).total_seconds()
        if time_to_deadline <= 0:
            logger.warning(f"Message {request.message_id} already past deadline")

        # Find instance that can meet the deadline
        best_instance = None
        best_score = float("-inf")

        for instance in instances:
            # Estimate processing time based on queue and response times
            queue_key = f"{request.agent_id}:{instance.instance_id}"
            queue_depth = len(self.priority_queues[queue_key])
            avg_response_time = (
                sum(instance.response_times) / len(instance.response_times)
                if instance.response_times
                else 1.0
            )

            estimated_wait = queue_depth * avg_response_time
            can_meet_deadline = estimated_wait < time_to_deadline

            # Score based on deadline slack and queue depth
            score = time_to_deadline - estimated_wait
            if can_meet_deadline and score > best_score:
                best_score = score
                best_instance = instance

        if not best_instance:
            # No instance can meet deadline, pick fastest
            best_instance = min(
                instances,
                key=lambda i: (
                    sum(i.response_times) / len(i.response_times)
                    if i.response_times
                    else float("inf")
                ),
            )

        # Priority score includes deadline urgency
        deadline_factor = max(0, min(1, time_to_deadline / 300))  # Normalize to 0-1
        priority_score = self._calculate_priority_score(request) * (2 - deadline_factor)

        queue_key = f"{request.agent_id}:{best_instance.instance_id}"
        heapq.heappush(self.priority_queues[queue_key], (-priority_score, request))

        return RoutingDecision(
            request_id=request.message_id,
            agent_id=request.agent_id,
            instance_id=best_instance.instance_id,
            instance_url=best_instance.base_url,
            priority_score=priority_score,
            estimated_wait_time=best_score if best_score > 0 else 0,
            routing_reason=f"Deadline-based routing, {time_to_deadline:.1f}s until deadline",
        )

    async def _route_adaptive(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Adaptively choose routing strategy based on system state"""
        # Analyze current system state
        total_queued = sum(len(q) for q in self.priority_queues.values())
        avg_wait_time = (
            sum(m.avg_wait_time for m in self.queue_metrics.values()) / len(self.queue_metrics)
            if self.queue_metrics
            else 0
        )

        # Choose strategy based on conditions
        if request.deadline and (request.deadline - datetime.utcnow()).total_seconds() < 60:
            # Urgent deadline - use deadline-based
            return await self._route_deadline_based(request)
        elif total_queued > 100 and avg_wait_time > 10:
            # High load - use weighted fair queuing
            return await self._route_weighted_fair(request)
        elif request.priority == MessagePriority.CRITICAL:
            # Critical priority - consider preemption
            if self.policy.enable_preemption:
                return await self._route_preemptive(request)
            else:
                return await self._route_priority_first(request)
        else:
            # Normal conditions - use priority-first
            return await self._route_priority_first(request)

    async def _route_preemptive(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Route with ability to preempt lower priority tasks"""
        instances = await self._get_available_instances(request)
        if not instances:
            return None

        request_priority = request.get_priority_value()

        # Look for preemption opportunities
        best_instance = None
        preempt_message_id = None

        for instance in instances:
            queue_key = f"{request.agent_id}:{instance.instance_id}"

            # Check if we can preempt any running task
            if queue_key in self.preemptible_tasks:
                for task_id in self.preemptible_tasks[queue_key]:
                    if task_id in self.active_routes:
                        self.active_routes[task_id]
                        # Find original request priority
                        for _, queued_req in self.priority_queues[queue_key]:
                            if queued_req.message_id == task_id:
                                if queued_req.get_priority_value() < request_priority:
                                    best_instance = instance
                                    preempt_message_id = task_id
                                    break
                        if best_instance:
                            break

            if best_instance:
                break

        if best_instance and preempt_message_id:
            # Perform preemption
            logger.info(f"Preempting message {preempt_message_id} for {request.message_id}")

            # Update metrics
            queue_key = f"{request.agent_id}:{best_instance.instance_id}"
            self.queue_metrics[queue_key].messages_preempted += 1
            self.preemption_count[request.message_id] += 1

            # Add to front of queue
            priority_score = self._calculate_priority_score(request) + 10  # Boost for preemption
            heapq.heappush(self.priority_queues[queue_key], (-priority_score, request))

            return RoutingDecision(
                request_id=request.message_id,
                agent_id=request.agent_id,
                instance_id=best_instance.instance_id,
                instance_url=best_instance.base_url,
                priority_score=priority_score,
                estimated_wait_time=0,  # Immediate due to preemption
                routing_reason=f"Preemptive routing, preempted {preempt_message_id}",
            )
        else:
            # No preemption possible, use priority-first
            return await self._route_priority_first(request)

    async def _route_premium_message(self, request: RoutingRequest) -> Optional[RoutingDecision]:
        """Route premium QoS messages with guaranteed resources"""
        # Premium messages get dedicated instance or reserved capacity
        instances = await self._get_available_instances(request)
        if not instances:
            return None

        # Find instance with most available capacity
        best_instance = None
        max_capacity = 0

        for instance in instances:
            available_capacity = instance.max_connections - instance.active_connections
            if available_capacity > max_capacity:
                max_capacity = available_capacity
                best_instance = instance

        if not best_instance:
            return None

        # Reserve capacity
        reservation_id = f"premium_{request.message_id}"
        self.qos_reservations[reservation_id] = {
            "instance_id": best_instance.instance_id,
            "reserved_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(minutes=5),
        }

        # Immediate routing for premium
        priority_score = 100.0  # Maximum priority

        return RoutingDecision(
            request_id=request.message_id,
            agent_id=request.agent_id,
            instance_id=best_instance.instance_id,
            instance_url=best_instance.base_url,
            priority_score=priority_score,
            estimated_wait_time=0,
            routing_reason="Premium QoS routing with reserved capacity",
        )

    async def _get_available_instances(self, request: RoutingRequest) -> List[AgentInstance]:
        """Get available instances for routing"""
        if not self.load_balancer:
            return []

        # Get all healthy instances for the agent
        all_instances = self.load_balancer._get_healthy_instances(request.agent_id)

        # Filter based on request constraints
        available = []
        for instance in all_instances:
            # Check excluded instances
            if instance.instance_id in request.excluded_instances:
                continue

            # Check required capabilities (simplified - assumes instance has all capabilities)
            if request.required_capabilities:
                # In real implementation, would check instance capabilities
                pass

            # Check if preferred instance
            if request.preferred_instances:
                if instance.instance_id not in request.preferred_instances:
                    continue

            available.append(instance)

        return available

    def _calculate_priority_score(self, request: RoutingRequest) -> float:
        """Calculate priority score for queue ordering"""
        base_score = request.get_priority_value() * 10

        # Age boost - older messages get priority boost
        age_seconds = (datetime.utcnow() - request.created_at).total_seconds()
        age_boost = min(age_seconds / self.policy.priority_boost_interval, 5)

        # QoS boost
        qos_boost = {QoSLevel.BEST_EFFORT: 0, QoSLevel.GUARANTEED: 5, QoSLevel.PREMIUM: 10}.get(
            request.qos_level, 0
        )

        # Deadline urgency
        deadline_boost = 0
        if request.deadline:
            time_to_deadline = (request.deadline - datetime.utcnow()).total_seconds()
            if time_to_deadline < 60:
                deadline_boost = 10
            elif time_to_deadline < 300:
                deadline_boost = 5

        return base_score + age_boost + qos_boost + deadline_boost

    def _get_priority_weight(self, priority: MessagePriority) -> float:
        """Get weight for priority level"""
        return {
            MessagePriority.LOW: 0.25,
            MessagePriority.MEDIUM: 1.0,
            MessagePriority.HIGH: 2.0,
            MessagePriority.CRITICAL: 4.0,
        }.get(priority, 1.0)

    async def _process_queues(self):
        """Background task to process priority queues"""
        while self._running:
            try:
                await asyncio.sleep(0.1)  # Process every 100ms

                # Process each queue
                for queue_key, queue in self.priority_queues.items():
                    if not queue:
                        continue

                    # Get highest priority item
                    if queue:
                        _, request = heapq.heappop(queue)

                        # Check if expired
                        if (
                            datetime.utcnow() - request.created_at
                        ).total_seconds() > self.policy.max_queue_time:
                            self.queue_metrics[queue_key].messages_expired += 1
                            logger.warning(f"Message {request.message_id} expired in queue")
                            continue

                        # Process the request (simplified - would actually send to instance)
                        self.queue_metrics[queue_key].messages_routed += 1

                        # Update wait time metrics
                        wait_time = (datetime.utcnow() - request.created_at).total_seconds()
                        metrics = self.queue_metrics[queue_key]
                        metrics.avg_wait_time = (
                            metrics.avg_wait_time * metrics.messages_routed + wait_time
                        ) / (metrics.messages_routed + 1)
                        metrics.max_wait_time = max(metrics.max_wait_time, wait_time)

                # Clean up expired reservations
                expired_reservations = []
                for res_id, reservation in self.qos_reservations.items():
                    if datetime.utcnow() > reservation["expires_at"]:
                        expired_reservations.append(res_id)

                for res_id in expired_reservations:
                    del self.qos_reservations[res_id]

            except Exception as e:
                logger.error(f"Error processing queues: {e}")

    def get_queue_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of priority queues"""
        if agent_id:
            # Get status for specific agent
            agent_queues = {
                k: v for k, v in self.priority_queues.items() if k.startswith(f"{agent_id}:")
            }
            agent_metrics = {
                k: v for k, v in self.queue_metrics.items() if k.startswith(f"{agent_id}:")
            }
        else:
            agent_queues = self.priority_queues
            agent_metrics = self.queue_metrics

        status = {
            "total_queued": sum(len(q) for q in agent_queues.values()),
            "active_routes": len(self.active_routes),
            "qos_reservations": len(self.qos_reservations),
            "queues": {},
        }

        for queue_key, queue in agent_queues.items():
            metrics = agent_metrics.get(queue_key, QueueMetrics())
            priority_breakdown = defaultdict(int)

            for _, request in queue:
                priority_breakdown[request.priority.value] += 1

            status["queues"][queue_key] = {
                "depth": len(queue),
                "priority_breakdown": dict(priority_breakdown),
                "metrics": {
                    "avg_wait_time": metrics.avg_wait_time,
                    "max_wait_time": metrics.max_wait_time,
                    "messages_routed": metrics.messages_routed,
                    "messages_preempted": metrics.messages_preempted,
                    "messages_expired": metrics.messages_expired,
                },
            }

        return status

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get overall routing metrics"""
        avg_routing_time = (
            sum(self.routing_times) / len(self.routing_times) if self.routing_times else 0
        )

        return {
            "strategy": self.policy.strategy.value,
            "total_routed": self.success_count,
            "total_failed": self.failure_count,
            "success_rate": (
                self.success_count / (self.success_count + self.failure_count)
                if (self.success_count + self.failure_count) > 0
                else 0
            ),
            "avg_routing_time_ms": avg_routing_time * 1000,
            "active_routes": len(self.active_routes),
            "total_preemptions": sum(self.preemption_count.values()),
            "qos_reservations": len(self.qos_reservations),
        }


# Global priority router instance
_priority_router: Optional[PriorityRouter] = None


async def get_priority_router() -> PriorityRouter:
    """Get or create the global priority router instance"""
    global _priority_router
    if _priority_router is None:
        _priority_router = PriorityRouter()
        await _priority_router.initialize()
    return _priority_router

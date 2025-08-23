"""
A2A Agent Load Balancer
Implements intelligent load distribution across multiple agent instances
Supports multiple algorithms: round-robin, weighted, least-connections, response-time based
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Available load balancing algorithms"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    DYNAMIC = "dynamic"  # AI-based dynamic selection


@dataclass
class AgentInstance:
    """Represents an agent instance in the load balancer pool"""

    agent_id: str
    instance_id: str
    base_url: str
    weight: int = 1  # For weighted algorithms
    max_connections: int = 100
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    last_health_check: datetime = field(default_factory=datetime.utcnow)

    # Runtime metrics
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Circuit breaker state
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_opened_at: Optional[datetime] = None


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer behavior"""

    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5  # seconds
    max_failures: int = 3  # failures before marking unhealthy
    circuit_breaker_timeout: int = 60  # seconds
    sticky_sessions: bool = False
    session_ttl: int = 3600  # seconds
    enable_retries: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class AgentLoadBalancer:
    """
    Intelligent load balancer for A2A agent instances
    Distributes requests across multiple agent instances based on various algorithms
    """

    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        self.agent_pools: Dict[str, List[AgentInstance]] = {}  # agent_id -> instances
        self.round_robin_counters: Dict[str, int] = {}
        self.session_affinity: Dict[str, str] = {}  # session_id -> instance_id
        self.ip_hash_cache: Dict[str, str] = {}  # ip -> instance_id

        # Performance tracking
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Health check task
        self._health_check_task = None
        self._is_running = False

        logger.info(f"Initialized load balancer with algorithm: {self.config.algorithm.value}")

    async def initialize(self):
        """Initialize load balancer and start health checks"""
        self._is_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer initialized and health checks started")

    async def shutdown(self):
        """Shutdown load balancer and cleanup resources"""
        self._is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Load balancer shutdown complete")

    def register_instance(self, agent_id: str, instance: AgentInstance):
        """Register an agent instance in the load balancer pool"""
        if agent_id not in self.agent_pools:
            self.agent_pools[agent_id] = []
            self.round_robin_counters[agent_id] = 0

        # Check if instance already exists
        existing = [i for i in self.agent_pools[agent_id] if i.instance_id == instance.instance_id]
        if not existing:
            self.agent_pools[agent_id].append(instance)
            logger.info(f"Registered instance {instance.instance_id} for agent {agent_id}")
        else:
            # Update existing instance
            idx = self.agent_pools[agent_id].index(existing[0])
            self.agent_pools[agent_id][idx] = instance
            logger.info(f"Updated instance {instance.instance_id} for agent {agent_id}")

    def unregister_instance(self, agent_id: str, instance_id: str):
        """Remove an agent instance from the load balancer pool"""
        if agent_id in self.agent_pools:
            self.agent_pools[agent_id] = [
                i for i in self.agent_pools[agent_id] if i.instance_id != instance_id
            ]
            logger.info(f"Unregistered instance {instance_id} for agent {agent_id}")

    async def select_instance(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentInstance]:
        """
        Select an agent instance based on the configured algorithm
        Returns None if no healthy instances are available
        """
        instances = self._get_healthy_instances(agent_id)
        if not instances:
            logger.warning(f"No healthy instances available for agent {agent_id}")
            return None

        # Check sticky sessions first
        if self.config.sticky_sessions and session_id:
            instance_id = self.session_affinity.get(session_id)
            if instance_id:
                instance = self._find_instance_by_id(agent_id, instance_id)
                if instance and instance.health_status == "healthy":
                    return instance

        # Select based on algorithm
        if self.config.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            instance = self._round_robin_select(agent_id, instances)
        elif self.config.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            instance = self._weighted_round_robin_select(agent_id, instances)
        elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            instance = self._least_connections_select(instances)
        elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            instance = self._least_response_time_select(instances)
        elif self.config.algorithm == LoadBalancingAlgorithm.RANDOM:
            instance = self._random_select(instances)
        elif self.config.algorithm == LoadBalancingAlgorithm.IP_HASH:
            instance = self._ip_hash_select(instances, client_ip)
        elif self.config.algorithm == LoadBalancingAlgorithm.DYNAMIC:
            instance = await self._dynamic_select(instances, request_context)
        else:
            instance = self._round_robin_select(agent_id, instances)

        # Update sticky session if enabled
        if instance and self.config.sticky_sessions and session_id:
            self.session_affinity[session_id] = instance.instance_id

        return instance

    def _get_healthy_instances(self, agent_id: str) -> List[AgentInstance]:
        """Get all healthy instances for an agent"""
        if agent_id not in self.agent_pools:
            return []

        return [
            instance
            for instance in self.agent_pools[agent_id]
            if instance.health_status in ["healthy", "degraded"] and not instance.circuit_open
        ]

    def _find_instance_by_id(self, agent_id: str, instance_id: str) -> Optional[AgentInstance]:
        """Find a specific instance by ID"""
        if agent_id not in self.agent_pools:
            return None

        for instance in self.agent_pools[agent_id]:
            if instance.instance_id == instance_id:
                return instance
        return None

    def _round_robin_select(self, agent_id: str, instances: List[AgentInstance]) -> AgentInstance:
        """Round-robin selection algorithm"""
        if not instances:
            return None

        counter = self.round_robin_counters.get(agent_id, 0)
        instance = instances[counter % len(instances)]
        self.round_robin_counters[agent_id] = counter + 1

        return instance

    def _weighted_round_robin_select(
        self, agent_id: str, instances: List[AgentInstance]
    ) -> AgentInstance:
        """Weighted round-robin selection algorithm"""
        if not instances:
            return None

        # Build weighted list
        weighted_instances = []
        for instance in instances:
            weighted_instances.extend([instance] * instance.weight)

        if not weighted_instances:
            return instances[0]

        counter = self.round_robin_counters.get(agent_id, 0)
        instance = weighted_instances[counter % len(weighted_instances)]
        self.round_robin_counters[agent_id] = counter + 1

        return instance

    def _least_connections_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with least active connections"""
        if not instances:
            return None

        return min(instances, key=lambda i: i.active_connections)

    def _least_response_time_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with lowest average response time"""
        if not instances:
            return None

        def avg_response_time(instance: AgentInstance) -> float:
            if not instance.response_times:
                return 0.0
            return statistics.mean(instance.response_times)

        return min(instances, key=avg_response_time)

    def _random_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Random selection algorithm"""
        if not instances:
            return None

        return secrets.choice(instances)

    def _ip_hash_select(
        self, instances: List[AgentInstance], client_ip: Optional[str]
    ) -> AgentInstance:
        """IP hash based selection for session persistence"""
        if not instances:
            return None

        if not client_ip:
            return self._random_select(instances)

        # Check cache first
        if client_ip in self.ip_hash_cache:
            instance_id = self.ip_hash_cache[client_ip]
            for instance in instances:
                if instance.instance_id == instance_id:
                    return instance

        # Hash IP to select instance
        ip_hash = hash(client_ip)
        instance = instances[ip_hash % len(instances)]

        # Cache the selection
        self.ip_hash_cache[client_ip] = instance.instance_id

        return instance

    async def _dynamic_select(
        self, instances: List[AgentInstance], request_context: Optional[Dict[str, Any]]
    ) -> AgentInstance:
        """
        AI-based dynamic selection considering multiple factors:
        - Current load
        - Response times
        - Error rates
        - Request complexity
        - Resource availability
        """
        if not instances:
            return None

        # Calculate scores for each instance
        scores = []
        for instance in instances:
            score = self._calculate_instance_score(instance, request_context)
            scores.append((instance, score))

        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select best instance
        return scores[0][0]

    def _calculate_instance_score(
        self, instance: AgentInstance, request_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate a score for an instance based on multiple factors"""
        score = 100.0

        # Factor 1: Active connections (normalized)
        connection_ratio = instance.active_connections / instance.max_connections
        score -= connection_ratio * 30  # Max penalty of 30

        # Factor 2: Response time
        if instance.response_times:
            avg_response_time = statistics.mean(instance.response_times)
            # Penalize if average > 1 second
            if avg_response_time > 1.0:
                score -= min(20, (avg_response_time - 1.0) * 10)

        # Factor 3: Error rate
        if instance.total_requests > 0:
            error_rate = instance.failed_requests / instance.total_requests
            score -= error_rate * 40  # Max penalty of 40

        # Factor 4: Health status
        if instance.health_status == "degraded":
            score -= 20

        # Factor 5: Weight (boost score based on weight)
        score += instance.weight * 5

        # Factor 6: Request complexity (if provided in context)
        if request_context and "complexity" in request_context:
            complexity = request_context["complexity"]
            # Prefer less loaded instances for complex requests
            if complexity == "high" and connection_ratio < 0.5:
                score += 10

        return max(0, score)

    async def record_request_start(self, agent_id: str, instance_id: str) -> bool:
        """Record the start of a request to an instance"""
        instance = self._find_instance_by_id(agent_id, instance_id)
        if instance:
            instance.active_connections += 1
            instance.total_requests += 1
            self.request_counts[f"{agent_id}:{instance_id}"] += 1
            return True
        return False

    async def record_request_end(
        self, agent_id: str, instance_id: str, response_time: float, success: bool = True
    ):
        """Record the end of a request to an instance"""
        instance = self._find_instance_by_id(agent_id, instance_id)
        if instance:
            instance.active_connections = max(0, instance.active_connections - 1)
            instance.response_times.append(response_time)

            if not success:
                instance.failed_requests += 1
                instance.consecutive_failures += 1
                self.error_counts[f"{agent_id}:{instance_id}"] += 1

                # Check if circuit breaker should open
                if instance.consecutive_failures >= self.config.max_failures:
                    instance.circuit_open = True
                    instance.circuit_opened_at = datetime.utcnow()
                    logger.warning(
                        f"Circuit breaker opened for instance {instance_id} "
                        f"after {instance.consecutive_failures} failures"
                    )
            else:
                instance.consecutive_failures = 0

    async def _health_check_loop(self):
        """Background task to perform health checks on all instances"""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for _, instances in self.agent_pools.items():
                    for instance in instances:
                        await self._check_instance_health(instance)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _check_instance_health(self, instance: AgentInstance):
        """Check health of a single instance"""
        try:
            # Check if circuit breaker should be reset
            if instance.circuit_open:
                if instance.circuit_opened_at:
                    elapsed = (datetime.utcnow() - instance.circuit_opened_at).total_seconds()
                    if elapsed >= self.config.circuit_breaker_timeout:
                        instance.circuit_open = False
                        instance.consecutive_failures = 0
                        logger.info(f"Circuit breaker reset for instance {instance.instance_id}")
                return

            # Perform health check (simplified - would be HTTP in real implementation)
            # This is a placeholder for actual health check logic
            instance.last_health_check = datetime.utcnow()

            # Update health status based on metrics
            if instance.failed_requests > 0 and instance.total_requests > 0:
                error_rate = instance.failed_requests / instance.total_requests
                if error_rate > 0.5:
                    instance.health_status = "unhealthy"
                elif error_rate > 0.2:
                    instance.health_status = "degraded"
                else:
                    instance.health_status = "healthy"

        except Exception as e:
            logger.error(f"Health check failed for instance {instance.instance_id}: {e}")
            instance.health_status = "unhealthy"

    def get_pool_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of all instances in a pool"""
        if agent_id not in self.agent_pools:
            return {"error": f"No pool found for agent {agent_id}"}

        instances_status = []
        for instance in self.agent_pools[agent_id]:
            avg_response_time = 0.0
            if instance.response_times:
                avg_response_time = statistics.mean(instance.response_times)

            error_rate = 0.0
            if instance.total_requests > 0:
                error_rate = instance.failed_requests / instance.total_requests

            instances_status.append(
                {
                    "instance_id": instance.instance_id,
                    "base_url": instance.base_url,
                    "health_status": instance.health_status,
                    "circuit_open": instance.circuit_open,
                    "active_connections": instance.active_connections,
                    "total_requests": instance.total_requests,
                    "failed_requests": instance.failed_requests,
                    "average_response_time": avg_response_time,
                    "error_rate": error_rate,
                    "weight": instance.weight,
                }
            )

        healthy_count = len(self._get_healthy_instances(agent_id))

        return {
            "agent_id": agent_id,
            "algorithm": self.config.algorithm.value,
            "total_instances": len(self.agent_pools[agent_id]),
            "healthy_instances": healthy_count,
            "instances": instances_status,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get overall load balancer metrics"""
        total_instances = sum(len(pool) for pool in self.agent_pools.values())
        total_healthy = sum(
            len(self._get_healthy_instances(agent_id)) for agent_id in self.agent_pools
        )

        return {
            "algorithm": self.config.algorithm.value,
            "total_agents": len(self.agent_pools),
            "total_instances": total_instances,
            "healthy_instances": total_healthy,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "sticky_sessions_active": len(self.session_affinity),
            "ip_hash_cache_size": len(self.ip_hash_cache),
        }


# Global load balancer instance
_load_balancer: Optional[AgentLoadBalancer] = None


async def get_load_balancer() -> AgentLoadBalancer:
    """Get or create the global load balancer instance"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = AgentLoadBalancer()
        await _load_balancer.initialize()
    return _load_balancer

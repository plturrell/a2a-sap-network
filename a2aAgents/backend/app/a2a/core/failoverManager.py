"""
A2A Failover Manager
Implements automatic failover, recovery, and high availability for agent instances
Supports multiple failover strategies and maintains service continuity
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import time
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random

from .loadBalancer import AgentLoadBalancer, AgentInstance, get_load_balancer
from .circuitBreaker import EnhancedCircuitBreaker, CircuitBreakerConfig, CircuitState
from .networkClient import NetworkClient, get_network_client

logger = logging.getLogger(__name__)


class FailoverStrategy(Enum):
    """Available failover strategies"""

    ACTIVE_PASSIVE = "active_passive"  # Single active, one or more passive
    ACTIVE_ACTIVE = "active_active"  # All instances active with load balancing
    PRIMARY_BACKUP = "primary_backup"  # Primary with multiple backups
    GEOGRAPHIC = "geographic"  # Failover based on geographic regions
    CASCADING = "cascading"  # Cascade through instance list


class InstanceRole(Enum):
    """Role of an instance in failover configuration"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"
    STANDBY = "standby"


class FailoverEvent(Enum):
    """Types of failover events"""

    INSTANCE_FAILURE = "instance_failure"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    MANUAL_FAILOVER = "manual_failover"
    NETWORK_PARTITION = "network_partition"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class FailoverConfig:
    """Configuration for failover behavior"""

    strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE
    health_check_interval: int = 10  # seconds
    failure_threshold: int = 3  # failures before failover
    recovery_threshold: int = 5  # successful checks before failback
    enable_auto_failback: bool = True
    failback_delay: int = 300  # seconds before attempting failback
    enable_cascading: bool = True  # Allow cascading failures
    max_failover_attempts: int = 3  # Maximum failover attempts
    enable_split_brain_detection: bool = True


@dataclass
class FailoverGroup:
    """Group of instances that can failover to each other"""

    group_id: str
    agent_id: str
    instances: List[AgentInstance]
    primary_instance_id: Optional[str] = None
    active_instance_ids: Set[str] = field(default_factory=set)
    strategy: FailoverStrategy = FailoverStrategy.ACTIVE_PASSIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverState:
    """Current state of a failover group"""

    group_id: str
    current_primary: Optional[str]
    active_instances: Set[str]
    failed_instances: Set[str]
    in_failover: bool = False
    last_failover: Optional[datetime] = None
    failover_count: int = 0
    consecutive_failures: int = 0


@dataclass
class FailoverDecision:
    """Result of failover decision making"""

    should_failover: bool
    target_instance_id: Optional[str]
    source_instance_id: Optional[str]
    reason: FailoverEvent
    confidence: float  # 0.0 to 1.0
    alternatives: List[str] = field(default_factory=list)


@dataclass
class RecoveryPlan:
    """Plan for recovering failed instances"""

    instance_id: str
    recovery_steps: List[str]
    estimated_recovery_time: int  # seconds
    can_auto_recover: bool
    requires_manual_intervention: bool


class FailoverManager:
    """
    Manages failover operations for A2A agents
    Ensures high availability and automatic recovery
    """

    def __init__(self, config: FailoverConfig = None):
        self.config = config or FailoverConfig()
        self.load_balancer: Optional[AgentLoadBalancer] = None
        self.network_client: Optional[NetworkClient] = None

        # Failover groups and states
        self.failover_groups: Dict[str, FailoverGroup] = {}
        self.failover_states: Dict[str, FailoverState] = {}

        # Circuit breakers per instance
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}

        # Health check tracking
        self.health_scores: Dict[str, float] = defaultdict(lambda: 100.0)
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Failover history and metrics
        self.failover_history: deque = deque(maxlen=1000)
        self.recovery_queue: List[RecoveryPlan] = []

        # Performance metrics
        self.failover_times: deque = deque(maxlen=100)
        self.successful_failovers = 0
        self.failed_failovers = 0

        # Split brain detection
        self.quorum_size: Dict[str, int] = {}
        self.witness_nodes: Dict[str, List[str]] = {}

        self._lock = asyncio.Lock()
        self._running = False
        self._monitor_task = None
        self._recovery_task = None

        # Instance URL registry
        self.instance_urls: Dict[str, str] = {}

        logger.info(f"Initialized failover manager with strategy: {self.config.strategy.value}")

    async def initialize(self):
        """Initialize failover manager and start monitoring"""
        self.load_balancer = await get_load_balancer()
        self.network_client = await get_network_client()
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_instances())
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        logger.info("Failover manager initialized")

    async def shutdown(self):
        """Shutdown failover manager"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._recovery_task:
            self._recovery_task.cancel()

        try:
            if self._monitor_task:
                await self._monitor_task
            if self._recovery_task:
                await self._recovery_task
        except asyncio.CancelledError:
            pass

        logger.info("Failover manager shutdown complete")

    async def create_failover_group(
        self,
        group_id: str,
        agent_id: str,
        instances: List[AgentInstance],
        strategy: FailoverStrategy = None,
    ) -> FailoverGroup:
        """Create a new failover group"""
        async with self._lock:
            if group_id in self.failover_groups:
                raise ValueError(f"Failover group {group_id} already exists")

            strategy = strategy or self.config.strategy

            # Create group
            group = FailoverGroup(
                group_id=group_id, agent_id=agent_id, instances=instances, strategy=strategy
            )

            # Initialize based on strategy
            if strategy == FailoverStrategy.ACTIVE_PASSIVE:
                # First instance is primary, others are passive
                if instances:
                    group.primary_instance_id = instances[0].instance_id
                    group.active_instance_ids.add(instances[0].instance_id)

            elif strategy == FailoverStrategy.ACTIVE_ACTIVE:
                # All instances are active
                group.active_instance_ids = {i.instance_id for i in instances}

            elif strategy == FailoverStrategy.PRIMARY_BACKUP:
                # First is primary, second is backup, rest are standby
                if instances:
                    group.primary_instance_id = instances[0].instance_id
                    group.active_instance_ids.add(instances[0].instance_id)

            # Store group and state
            self.failover_groups[group_id] = group
            self.failover_states[group_id] = FailoverState(
                group_id=group_id,
                current_primary=group.primary_instance_id,
                active_instances=group.active_instance_ids.copy(),
                failed_instances=set(),
            )

            # Initialize circuit breakers
            for instance in instances:
                cb_config = CircuitBreakerConfig(
                    failure_threshold=self.config.failure_threshold,
                    recovery_threshold=self.config.recovery_threshold,
                )
                self.circuit_breakers[instance.instance_id] = EnhancedCircuitBreaker(
                    name=f"failover_{instance.instance_id}", config=cb_config
                )

            # Calculate quorum size for split-brain detection
            if self.config.enable_split_brain_detection:
                self.quorum_size[group_id] = (len(instances) // 2) + 1

            logger.info(f"Created failover group {group_id} with {len(instances)} instances")
            return group

    async def trigger_failover(
        self, group_id: str, failed_instance_id: str, reason: FailoverEvent, force: bool = False
    ) -> Optional[str]:
        """
        Trigger failover for a failed instance
        Returns the new active instance ID if successful
        """
        start_time = time.time()

        async with self._lock:
            if group_id not in self.failover_groups:
                logger.error(f"Failover group {group_id} not found")
                return None

            group = self.failover_groups[group_id]
            state = self.failover_states[group_id]

            # Check if already in failover
            if state.in_failover and not force:
                logger.warning(f"Group {group_id} already in failover process")
                return None

            # Check failover attempts
            if state.consecutive_failures >= self.config.max_failover_attempts:
                logger.error(f"Max failover attempts reached for group {group_id}")
                self.failed_failovers += 1
                return None

            try:
                state.in_failover = True

                # Make failover decision
                decision = await self._make_failover_decision(
                    group, state, failed_instance_id, reason
                )

                if not decision.should_failover:
                    logger.info(f"Failover not needed for {failed_instance_id}")
                    state.in_failover = False
                    return None

                if not decision.target_instance_id:
                    logger.error(f"No suitable failover target for {failed_instance_id}")
                    state.consecutive_failures += 1
                    state.in_failover = False
                    self.failed_failovers += 1
                    return None

                # Execute failover
                success = await self._execute_failover(
                    group, state, failed_instance_id, decision.target_instance_id
                )

                if success:
                    # Update state
                    state.failed_instances.add(failed_instance_id)
                    state.active_instances.discard(failed_instance_id)
                    state.active_instances.add(decision.target_instance_id)

                    if state.current_primary == failed_instance_id:
                        state.current_primary = decision.target_instance_id

                    state.last_failover = datetime.utcnow()
                    state.failover_count += 1
                    state.consecutive_failures = 0

                    # Record failover
                    self.failover_history.append(
                        {
                            "timestamp": datetime.utcnow(),
                            "group_id": group_id,
                            "source": failed_instance_id,
                            "target": decision.target_instance_id,
                            "reason": reason.value,
                            "duration": time.time() - start_time,
                        }
                    )

                    self.failover_times.append(time.time() - start_time)
                    self.successful_failovers += 1

                    logger.info(
                        f"Successful failover in group {group_id}: "
                        f"{failed_instance_id} -> {decision.target_instance_id}"
                    )

                    # Create recovery plan for failed instance
                    recovery_plan = await self._create_recovery_plan(failed_instance_id, reason)
                    if recovery_plan:
                        self.recovery_queue.append(recovery_plan)

                    return decision.target_instance_id
                else:
                    state.consecutive_failures += 1
                    self.failed_failovers += 1
                    logger.error(f"Failover execution failed for group {group_id}")
                    return None

            finally:
                state.in_failover = False

    async def _make_failover_decision(
        self,
        group: FailoverGroup,
        state: FailoverState,
        failed_instance_id: str,
        reason: FailoverEvent,
    ) -> FailoverDecision:
        """Make intelligent failover decision"""
        # Find healthy instances
        healthy_instances = []
        for instance in group.instances:
            if (
                instance.instance_id != failed_instance_id
                and instance.instance_id not in state.failed_instances
                and self.health_scores.get(instance.instance_id, 0) > 50
            ):
                healthy_instances.append(instance)

        if not healthy_instances:
            return FailoverDecision(
                should_failover=True,
                target_instance_id=None,
                source_instance_id=failed_instance_id,
                reason=reason,
                confidence=1.0,
            )

        # Select target based on strategy
        target_instance = None

        if group.strategy == FailoverStrategy.ACTIVE_PASSIVE:
            # Choose next healthy instance in order
            target_instance = healthy_instances[0]

        elif group.strategy == FailoverStrategy.PRIMARY_BACKUP:
            # Prefer backup instances first
            for instance in healthy_instances:
                if instance.instance_id in group.metadata.get("backup_instances", []):
                    target_instance = instance
                    break
            if not target_instance:
                target_instance = healthy_instances[0]

        elif group.strategy == FailoverStrategy.GEOGRAPHIC:
            # Choose instance in different region
            failed_region = group.metadata.get("instance_regions", {}).get(failed_instance_id)
            for instance in healthy_instances:
                instance_region = group.metadata.get("instance_regions", {}).get(
                    instance.instance_id
                )
                if instance_region != failed_region:
                    target_instance = instance
                    break
            if not target_instance:
                target_instance = healthy_instances[0]

        elif group.strategy == FailoverStrategy.CASCADING:
            # Choose next in cascade order
            cascade_order = group.metadata.get("cascade_order", [])
            if cascade_order:
                for instance_id in cascade_order:
                    for instance in healthy_instances:
                        if instance.instance_id == instance_id:
                            target_instance = instance
                            break
                    if target_instance:
                        break
            if not target_instance:
                target_instance = healthy_instances[0]

        else:  # ACTIVE_ACTIVE or default
            # Choose instance with best health score
            target_instance = max(
                healthy_instances, key=lambda i: self.health_scores.get(i.instance_id, 0)
            )

        # Calculate confidence based on target health
        confidence = self.health_scores.get(target_instance.instance_id, 0) / 100.0

        return FailoverDecision(
            should_failover=True,
            target_instance_id=target_instance.instance_id if target_instance else None,
            source_instance_id=failed_instance_id,
            reason=reason,
            confidence=confidence,
            alternatives=[i.instance_id for i in healthy_instances if i != target_instance],
        )

    async def _execute_failover(
        self, group: FailoverGroup, state: FailoverState, source_id: str, target_id: str
    ) -> bool:
        """Execute the failover operation"""
        try:
            logger.info(f"Executing failover: {source_id} -> {target_id}")

            # Step 1: Verify target is healthy
            target_health = await self._check_instance_health(target_id)
            if not target_health:
                logger.error(f"Target instance {target_id} health check failed")
                return False

            # Step 2: Update load balancer (if active-passive)
            if group.strategy in [FailoverStrategy.ACTIVE_PASSIVE, FailoverStrategy.PRIMARY_BACKUP]:
                # Remove source from load balancer
                if self.load_balancer:
                    self.load_balancer.unregister_instance(group.agent_id, source_id)

                    # Ensure target is registered
                    target_instance = next(
                        (i for i in group.instances if i.instance_id == target_id), None
                    )
                    if target_instance:
                        self.load_balancer.register_instance(group.agent_id, target_instance)

            # Step 3: Update routing tables
            await self._update_routing_tables(group, source_id, target_id)

            # Step 4: Drain connections from source (if still accessible)
            # This would gracefully close existing connections

            # Step 5: Activate target
            # This might involve starting services, mounting storage, etc.

            # Step 6: Verify target is serving traffic
            await asyncio.sleep(2)  # Give time for activation
            final_check = await self._check_instance_health(target_id)

            return final_check

        except Exception as e:
            logger.error(f"Failover execution error: {e}")
            return False

    async def attempt_failback(self, group_id: str, recovered_instance_id: str) -> bool:
        """
        Attempt to fail back to a recovered instance
        Only if auto-failback is enabled and conditions are met
        """
        if not self.config.enable_auto_failback:
            logger.info(f"Auto-failback disabled for {recovered_instance_id}")
            return False

        async with self._lock:
            if group_id not in self.failover_groups:
                return False

            group = self.failover_groups[group_id]
            state = self.failover_states[group_id]

            # Check if instance was previously failed
            if recovered_instance_id not in state.failed_instances:
                return False

            # Check if enough time has passed since failover
            if state.last_failover:
                time_since_failover = (datetime.utcnow() - state.last_failover).total_seconds()
                if time_since_failover < self.config.failback_delay:
                    logger.info(
                        f"Failback delay not met for {recovered_instance_id} "
                        f"({time_since_failover:.0f}s < {self.config.failback_delay}s)"
                    )
                    return False

            # Check if instance is healthy enough
            health_score = self.health_scores.get(recovered_instance_id, 0)
            if health_score < 90:  # Require high health for failback
                logger.info(
                    f"Health score too low for failback: {recovered_instance_id} "
                    f"({health_score:.1f} < 90)"
                )
                return False

            # Execute failback (similar to failover but in reverse)
            current_primary = state.current_primary
            if current_primary and current_primary != recovered_instance_id:
                success = await self._execute_failover(
                    group, state, current_primary, recovered_instance_id
                )

                if success:
                    state.failed_instances.discard(recovered_instance_id)
                    state.active_instances.add(recovered_instance_id)
                    if group.primary_instance_id == recovered_instance_id:
                        state.current_primary = recovered_instance_id

                    logger.info(f"Successful failback to {recovered_instance_id}")
                    return True

            return False

    async def _monitor_instances(self):
        """Background task to monitor instance health"""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check all instances in all groups
                for group_id, group in self.failover_groups.items():
                    state = self.failover_states.get(group_id)
                    if not state:
                        continue

                    for instance in group.instances:
                        # Skip if instance is known to be failed
                        if instance.instance_id in state.failed_instances:
                            continue

                        # Perform health check
                        is_healthy = await self._check_instance_health(instance.instance_id)

                        # Update health score
                        self._update_health_score(instance.instance_id, is_healthy)

                        # Check if we need to trigger failover
                        health_score = self.health_scores.get(instance.instance_id, 100)
                        if health_score < 30 and instance.instance_id in state.active_instances:
                            logger.warning(
                                f"Instance {instance.instance_id} health degraded "
                                f"(score: {health_score:.1f})"
                            )

                            # Trigger failover if it's the primary or active
                            if (
                                instance.instance_id == state.current_primary
                                or group.strategy == FailoverStrategy.ACTIVE_ACTIVE
                            ):
                                await self.trigger_failover(
                                    group_id,
                                    instance.instance_id,
                                    FailoverEvent.HEALTH_CHECK_FAILURE,
                                )

            except Exception as e:
                logger.error(f"Error in instance monitoring: {e}")

    async def _recovery_loop(self):
        """Background task to attempt recovery of failed instances"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.recovery_queue:
                    continue

                # Process recovery plans
                completed_recoveries = []

                for plan in self.recovery_queue[:]:  # Copy to avoid modification during iteration
                    if plan.can_auto_recover:
                        recovered = await self._attempt_recovery(plan)
                        if recovered:
                            completed_recoveries.append(plan)

                            # Check if we can failback
                            for group_id, state in self.failover_states.items():
                                if plan.instance_id in state.failed_instances:
                                    await self.attempt_failback(group_id, plan.instance_id)

                # Remove completed recoveries
                for plan in completed_recoveries:
                    self.recovery_queue.remove(plan)

            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")

    async def _check_instance_health(self, instance_id: str) -> bool:
        """Check health of a specific instance"""
        try:
            # Check circuit breaker state
            cb = self.circuit_breakers.get(instance_id)
            if cb and cb.state == CircuitState.OPEN:
                return False

            if not self.network_client:
                logger.warning("Network client not available for health check")
                return False

            # Get instance URL
            instance_url = await self._get_instance_url(instance_id)
            if not instance_url:
                logger.warning(f"URL not found for instance {instance_id}")
                return False

            # Perform actual health check
            is_healthy = await self.network_client.health_check(instance_url, timeout=5)

            if is_healthy:
                logger.debug(f"Health check passed for {instance_id}")
            else:
                logger.warning(f"Health check failed for {instance_id}")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for {instance_id}: {e}")
            return False

    def _update_health_score(self, instance_id: str, is_healthy: bool):
        """Update health score based on health check result"""
        current_score = self.health_scores.get(instance_id, 100.0)

        if is_healthy:
            # Increase score (slower recovery)
            new_score = min(100.0, current_score + 5)
        else:
            # Decrease score (faster degradation)
            new_score = max(0.0, current_score - 10)

        self.health_scores[instance_id] = new_score
        self.health_history[instance_id].append(
            {"timestamp": datetime.utcnow(), "healthy": is_healthy, "score": new_score}
        )

    async def _create_recovery_plan(
        self, instance_id: str, failure_reason: FailoverEvent
    ) -> Optional[RecoveryPlan]:
        """Create a recovery plan for failed instance"""
        recovery_steps = []
        can_auto_recover = False

        if failure_reason == FailoverEvent.HEALTH_CHECK_FAILURE:
            recovery_steps = [
                "Wait for instance to stabilize",
                "Perform extended health checks",
                "Verify service availability",
                "Test with synthetic transactions",
            ]
            can_auto_recover = True

        elif failure_reason == FailoverEvent.CIRCUIT_BREAKER_OPEN:
            recovery_steps = [
                "Wait for circuit breaker timeout",
                "Perform gradual health checks",
                "Reset circuit breaker if healthy",
            ]
            can_auto_recover = True

        elif failure_reason == FailoverEvent.NETWORK_PARTITION:
            recovery_steps = [
                "Verify network connectivity",
                "Check for split-brain conditions",
                "Validate data consistency",
                "Rejoin cluster if safe",
            ]
            can_auto_recover = False  # Requires manual verification

        else:
            recovery_steps = [
                "Investigate failure cause",
                "Apply necessary fixes",
                "Perform comprehensive testing",
                "Gradual traffic ramp-up",
            ]
            can_auto_recover = False

        return RecoveryPlan(
            instance_id=instance_id,
            recovery_steps=recovery_steps,
            estimated_recovery_time=300,  # 5 minutes default
            can_auto_recover=can_auto_recover,
            requires_manual_intervention=not can_auto_recover,
        )

    async def _attempt_recovery(self, plan: RecoveryPlan) -> bool:
        """Attempt to recover a failed instance"""
        logger.info(f"Attempting recovery for instance {plan.instance_id}")

        try:
            # Execute recovery steps
            for step in plan.recovery_steps:
                logger.debug(f"Recovery step for {plan.instance_id}: {step}")
                success = await self._execute_recovery_step(plan.instance_id, step)
                if not success:
                    logger.warning(f"Recovery step failed: {step} for {plan.instance_id}")
                    return False

            # Verify instance is healthy
            is_healthy = await self._check_instance_health(plan.instance_id)

            if is_healthy:
                # Reset health score
                self.health_scores[plan.instance_id] = 80.0  # Start at 80%

                # Reset circuit breaker
                cb = self.circuit_breakers.get(plan.instance_id)
                if cb:
                    await cb.reset()

                logger.info(f"Successfully recovered instance {plan.instance_id}")
                return True
            else:
                logger.warning(f"Recovery verification failed for {plan.instance_id}")
                return False

        except Exception as e:
            logger.error(f"Recovery attempt failed for {plan.instance_id}: {e}")
            return False

    def get_failover_status(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current failover status"""
        if group_id:
            if group_id not in self.failover_groups:
                return {"error": f"Group {group_id} not found"}

            group = self.failover_groups[group_id]
            state = self.failover_states[group_id]

            return {
                "group_id": group_id,
                "strategy": group.strategy.value,
                "current_primary": state.current_primary,
                "active_instances": list(state.active_instances),
                "failed_instances": list(state.failed_instances),
                "in_failover": state.in_failover,
                "failover_count": state.failover_count,
                "last_failover": state.last_failover.isoformat() if state.last_failover else None,
                "health_scores": {
                    i.instance_id: self.health_scores.get(i.instance_id, 0) for i in group.instances
                },
            }
        else:
            # Return status for all groups
            return {
                group_id: self.get_failover_status(group_id) for group_id in self.failover_groups
            }

    def get_failover_metrics(self) -> Dict[str, Any]:
        """Get failover performance metrics"""
        avg_failover_time = (
            sum(self.failover_times) / len(self.failover_times) if self.failover_times else 0
        )

        return {
            "total_failovers": self.successful_failovers + self.failed_failovers,
            "successful_failovers": self.successful_failovers,
            "failed_failovers": self.failed_failovers,
            "success_rate": (
                self.successful_failovers / (self.successful_failovers + self.failed_failovers)
                if (self.successful_failovers + self.failed_failovers) > 0
                else 0
            ),
            "avg_failover_time_seconds": avg_failover_time,
            "recovery_queue_size": len(self.recovery_queue),
            "active_groups": len(self.failover_groups),
            "total_instances_monitored": sum(
                len(group.instances) for group in self.failover_groups.values()
            ),
        }

    async def _get_instance_url(self, instance_id: str) -> Optional[str]:
        """Get URL for a specific instance"""
        if instance_id in self.instance_urls:
            return self.instance_urls[instance_id]

        # Try to find instance in failover groups
        for group in self.failover_groups.values():
            for instance in group.instances:
                if instance.instance_id == instance_id:
                    url = instance.base_url
                    self.instance_urls[instance_id] = url
                    return url

        # Default to localhost-based URL for development
        port = 8000 + int(instance_id.split('_')[-1]) if '_' in instance_id else 8000
        url = f"http://localhost:{port}"
        self.instance_urls[instance_id] = url
        return url

    def register_instance_url(self, instance_id: str, url: str):
        """Register URL for an instance"""
        self.instance_urls[instance_id] = url
        logger.info(f"Registered instance {instance_id} at {url}")

    async def _update_routing_tables(self, group: FailoverGroup, source_id: str, target_id: str) -> bool:
        """Update routing tables to redirect traffic from source to target"""
        if not self.network_client:
            logger.warning("Network client not available for routing table updates")
            return False

        try:
            # Prepare routing update
            routing_update = {
                "failover_group": group.group_id,
                "source_instance": source_id,
                "target_instance": target_id,
                "timestamp": datetime.utcnow().isoformat(),
                "operation": "failover"
            }

            # Update routing on all healthy instances in the group
            success_count = 0
            for instance in group.instances:
                if instance.instance_id in [source_id, target_id]:
                    continue  # Skip the failover instances

                try:
                    instance_url = await self._get_instance_url(instance.instance_id)
                    if instance_url:
                        success = await self.network_client.update_routing_table(
                            instance_url, routing_update
                        )
                        if success:
                            success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update routing on {instance.instance_id}: {e}")
                    continue

            logger.info(f"Updated routing tables on {success_count} instances for failover {source_id} -> {target_id}")
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to update routing tables: {e}")
            return False

    async def _execute_recovery_step(self, instance_id: str, step: str) -> bool:
        """Execute a specific recovery step on an instance"""
        if not self.network_client:
            logger.warning("Network client not available for recovery step execution")
            return False

        try:
            instance_url = await self._get_instance_url(instance_id)
            if not instance_url:
                logger.error(f"URL not found for instance {instance_id}")
                return False

            # Map recovery step to parameters
            parameters = self._get_recovery_step_parameters(step)

            # Execute recovery step
            success = await self.network_client.execute_recovery_step(
                instance_url, step, parameters
            )

            if success:
                logger.debug(f"Recovery step '{step}' executed successfully on {instance_id}")
            else:
                logger.warning(f"Recovery step '{step}' failed on {instance_id}")

            return success

        except Exception as e:
            logger.error(f"Error executing recovery step '{step}' on {instance_id}: {e}")
            return False

    def _get_recovery_step_parameters(self, step: str) -> Dict[str, Any]:
        """Get parameters for a specific recovery step"""
        step_parameters = {
            "Wait for instance to stabilize": {
                "wait_time": 30,
                "check_interval": 5
            },
            "Perform extended health checks": {
                "check_count": 5,
                "check_interval": 10,
                "required_success_rate": 0.8
            },
            "Verify service availability": {
                "endpoints": ["/health", "/ready", "/metrics"],
                "timeout": 10
            },
            "Test with synthetic transactions": {
                "transaction_count": 10,
                "timeout": 30
            },
            "Wait for circuit breaker timeout": {
                "timeout": self.config.recovery_threshold * 30  # 30s per threshold
            },
            "Perform gradual health checks": {
                "initial_interval": 60,
                "success_threshold": 3
            },
            "Reset circuit breaker if healthy": {
                "health_check_count": 5
            },
            "Verify network connectivity": {
                "ping_targets": ["8.8.8.8", "1.1.1.1"],
                "timeout": 10
            },
            "Check for split-brain conditions": {
                "quorum_check": True,
                "witness_nodes": True
            },
            "Validate data consistency": {
                "consistency_checks": ["version", "checksum", "replica_count"],
                "timeout": 60
            },
            "Rejoin cluster if safe": {
                "safety_checks": True,
                "gradual_rejoin": True
            }
        }

        return step_parameters.get(step, {})


# Global failover manager instance
_failover_manager: Optional[FailoverManager] = None


async def get_failover_manager() -> FailoverManager:
    """Get or create the global failover manager instance"""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = FailoverManager()
        await _failover_manager.initialize()
    return _failover_manager

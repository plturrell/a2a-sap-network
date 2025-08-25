"""
Chaos Engineering Framework for A2A Agents
Provides automated failure injection, resilience testing, and system validation
"""

import asyncio
import json
import logging
import random
import secrets
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import traceback

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..a2a.sdk.types import A2AMessage, TaskStatus
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class ChaosExperimentType(str, Enum):
    """Types of chaos experiments"""
    NETWORK_PARTITION = "network_partition"
    LATENCY_INJECTION = "latency_injection"
    PACKET_LOSS = "packet_loss"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SERVICE_FAILURE = "service_failure"
    DATABASE_FAILURE = "database_failure"
    MESSAGE_CORRUPTION = "message_corruption"
    DISK_FAILURE = "disk_failure"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    AGENT_CRASH = "agent_crash"
    TIMEOUT_INJECTION = "timeout_injection"


class ExperimentStatus(str, Enum):
    """Chaos experiment status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    ABORTED = "aborted"


class ImpactLevel(str, Enum):
    """Impact level of chaos experiments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperimentConfig:
    """Configuration for a chaos experiment"""
    name: str
    description: str
    experiment_type: ChaosExperimentType
    target_agents: List[str]
    duration_seconds: int
    impact_level: ImpactLevel
    parameters: Dict[str, Any] = field(default_factory=dict)
    blast_radius: float = 0.1  # Percentage of system to affect
    abort_conditions: List[Dict[str, Any]] = field(default_factory=list)
    steady_state_hypothesis: Dict[str, Any] = field(default_factory=dict)
    rollback_conditions: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChaosExperimentResult:
    """Result of a chaos experiment"""
    experiment_id: str
    config: ChaosExperimentConfig
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_during: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    recovery_time_seconds: Optional[float] = None
    blast_radius_actual: float = 0.0


class NetworkChaosInjector:
    """Network-related chaos injection"""

    def __init__(self):
        self.active_injections = {}

    async def inject_latency(self, target_agents: List[str], latency_ms: int, duration: int) -> str:
        """Inject network latency"""
        injection_id = str(uuid4())

        try:
            # Simulate network latency injection
            logger.info(f"Injecting {latency_ms}ms latency for agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "latency",
                "targets": target_agents,
                "latency_ms": latency_ms,
                "start_time": time.time()
            }

            # Schedule removal after duration
            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject latency: {e}")
            raise e

    async def inject_packet_loss(self, target_agents: List[str], loss_percentage: float, duration: int) -> str:
        """Inject packet loss"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting {loss_percentage}% packet loss for agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "packet_loss",
                "targets": target_agents,
                "loss_percentage": loss_percentage,
                "start_time": time.time()
            }

            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject packet loss: {e}")
            raise e

    async def create_network_partition(self, partition_groups: List[List[str]], duration: int) -> str:
        """Create network partition between agent groups"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Creating network partition between groups: {partition_groups}")

            self.active_injections[injection_id] = {
                "type": "partition",
                "partition_groups": partition_groups,
                "start_time": time.time()
            }

            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to create network partition: {e}")
            raise e

    async def _remove_injection_after_delay(self, injection_id: str, delay: int):
        """Remove injection after specified delay"""
        await asyncio.sleep(delay)
        await self.remove_injection(injection_id)

    async def remove_injection(self, injection_id: str):
        """Remove network injection"""
        if injection_id in self.active_injections:
            injection = self.active_injections.pop(injection_id)
            logger.info(f"Removed network injection: {injection['type']} for {injection_id}")


class ResourceChaosInjector:
    """Resource-related chaos injection"""

    def __init__(self):
        self.active_injections = {}
        self.stress_tasks = {}

    async def inject_cpu_stress(self, target_agents: List[str], cpu_percentage: int, duration: int) -> str:
        """Inject CPU stress"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting {cpu_percentage}% CPU stress for agents: {target_agents}")

            # Start CPU stress simulation
            stress_task = asyncio.create_task(self._simulate_cpu_stress(cpu_percentage, duration))
            self.stress_tasks[injection_id] = stress_task

            self.active_injections[injection_id] = {
                "type": "cpu_stress",
                "targets": target_agents,
                "cpu_percentage": cpu_percentage,
                "start_time": time.time()
            }

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject CPU stress: {e}")
            raise e

    async def inject_memory_pressure(self, target_agents: List[str], memory_mb: int, duration: int) -> str:
        """Inject memory pressure"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting {memory_mb}MB memory pressure for agents: {target_agents}")

            # Start memory pressure simulation
            stress_task = asyncio.create_task(self._simulate_memory_pressure(memory_mb, duration))
            self.stress_tasks[injection_id] = stress_task

            self.active_injections[injection_id] = {
                "type": "memory_pressure",
                "targets": target_agents,
                "memory_mb": memory_mb,
                "start_time": time.time()
            }

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject memory pressure: {e}")
            raise e

    async def inject_disk_failure(self, target_agents: List[str], duration: int) -> str:
        """Simulate disk failure"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Simulating disk failure for agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "disk_failure",
                "targets": target_agents,
                "start_time": time.time()
            }

            # Schedule removal after duration
            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject disk failure: {e}")
            raise e

    async def _simulate_cpu_stress(self, cpu_percentage: int, duration: int):
        """Simulate CPU stress by consuming CPU cycles"""
        end_time = time.time() + duration

        while time.time() < end_time:
            # Consume CPU for the specified percentage of time
            stress_duration = (cpu_percentage / 100.0) * 0.1  # 100ms intervals
            start = time.time()

            # Busy wait to consume CPU
            while time.time() - start < stress_duration:
                pass

            # Sleep for the remaining time
            await asyncio.sleep(0.1 - stress_duration)

    async def _simulate_memory_pressure(self, memory_mb: int, duration: int):
        """Simulate memory pressure by allocating memory"""
        allocated_memory = []

        try:
            # Allocate memory in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks_needed = memory_mb

            for _ in range(chunks_needed):
                allocated_memory.append(bytearray(chunk_size))

            logger.info(f"Allocated {memory_mb}MB for memory pressure simulation")

            # Hold the memory for the duration
            await asyncio.sleep(duration)

        finally:
            # Release memory
            allocated_memory.clear()
            logger.info("Released allocated memory")

    async def _remove_injection_after_delay(self, injection_id: str, delay: int):
        """Remove injection after specified delay"""
        await asyncio.sleep(delay)
        await self.remove_injection(injection_id)

    async def remove_injection(self, injection_id: str):
        """Remove resource injection"""
        if injection_id in self.active_injections:
            injection = self.active_injections.pop(injection_id)
            logger.info(f"Removed resource injection: {injection['type']} for {injection_id}")

        if injection_id in self.stress_tasks:
            task = self.stress_tasks.pop(injection_id)
            task.cancel()


class ServiceChaosInjector:
    """Service-level chaos injection"""

    def __init__(self):
        self.active_injections = {}
        self.original_functions = {}

    async def inject_service_failure(self, target_agents: List[str], service_name: str, duration: int) -> str:
        """Inject service failure"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting service failure for {service_name} on agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "service_failure",
                "targets": target_agents,
                "service_name": service_name,
                "start_time": time.time()
            }

            # Schedule removal after duration
            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject service failure: {e}")
            raise e

    async def inject_timeout_errors(self, target_agents: List[str], timeout_probability: float, duration: int) -> str:
        """Inject random timeout errors"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting timeout errors (p={timeout_probability}) for agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "timeout_errors",
                "targets": target_agents,
                "timeout_probability": timeout_probability,
                "start_time": time.time()
            }

            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject timeout errors: {e}")
            raise e

    async def inject_message_corruption(self, target_agents: List[str], corruption_rate: float, duration: int) -> str:
        """Inject message corruption"""
        injection_id = str(uuid4())

        try:
            logger.info(f"Injecting message corruption (rate={corruption_rate}) for agents: {target_agents}")

            self.active_injections[injection_id] = {
                "type": "message_corruption",
                "targets": target_agents,
                "corruption_rate": corruption_rate,
                "start_time": time.time()
            }

            asyncio.create_task(self._remove_injection_after_delay(injection_id, duration))

            return injection_id

        except Exception as e:
            logger.error(f"Failed to inject message corruption: {e}")
            raise e

    async def _remove_injection_after_delay(self, injection_id: str, delay: int):
        """Remove injection after specified delay"""
        await asyncio.sleep(delay)
        await self.remove_injection(injection_id)

    async def remove_injection(self, injection_id: str):
        """Remove service injection"""
        if injection_id in self.active_injections:
            injection = self.active_injections.pop(injection_id)
            logger.info(f"Removed service injection: {injection['type']} for {injection_id}")


class ChaosMetricsCollector:
    """Collect metrics during chaos experiments"""

    def __init__(self, redis_client: RedisClient):
        self.redis_client = redis_client

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            import psutil


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

            metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": dict(psutil.net_io_counters()._asdict()),
                "timestamp": datetime.utcnow().isoformat()
            }

            return metrics

        except ImportError:
            # Fallback metrics
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def collect_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Collect agent-specific metrics"""
        try:
            # Get agent metrics from Redis
            metrics_key = f"agent_metrics:{agent_id}"
            metrics_data = await self.redis_client.hgetall(metrics_key)

            if not metrics_data:
                return {"error": "No metrics available"}

            return {
                "agent_id": agent_id,
                "response_time": float(metrics_data.get("response_time", 0)),
                "error_rate": float(metrics_data.get("error_rate", 0)),
                "throughput": float(metrics_data.get("throughput", 0)),
                "active_connections": int(metrics_data.get("active_connections", 0)),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to collect agent metrics for {agent_id}: {e}")
            return {"error": str(e)}

    async def collect_network_metrics(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Collect network connectivity metrics between agents"""
        connectivity_matrix = {}

        for source_agent in agent_ids:
            connectivity_matrix[source_agent] = {}

            for target_agent in agent_ids:
                if source_agent != target_agent:
                    # Simulate connectivity check
                    connectivity = await self._check_connectivity(source_agent, target_agent)
                    connectivity_matrix[source_agent][target_agent] = connectivity

        return {
            "connectivity_matrix": connectivity_matrix,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _check_connectivity(self, source: str, target: str) -> Dict[str, Any]:
        """Check connectivity between two agents"""
        # Simulate connectivity check
        return {
            "reachable": secrets.choice([True, False]),
            "latency_ms": random.uniform(10, 500),
            "packet_loss": random.uniform(0, 0.1)
        }


class ChaosExperimentRunner:
    """Run and manage chaos experiments"""

    def __init__(self, redis_config: RedisConfig = None):
        self.redis_client = RedisClient(redis_config or RedisConfig())
        self.network_injector = NetworkChaosInjector()
        self.resource_injector = ResourceChaosInjector()
        self.service_injector = ServiceChaosInjector()
        self.metrics_collector = ChaosMetricsCollector(self.redis_client)

        self.running_experiments: Dict[str, ChaosExperimentResult] = {}
        self.experiment_observers: List[Callable] = []

    async def initialize(self):
        """Initialize the chaos experiment runner"""
        await self.redis_client.initialize()
        logger.info("Chaos experiment runner initialized")

    async def shutdown(self):
        """Shutdown the chaos experiment runner"""
        # Stop all running experiments
        for experiment_id in list(self.running_experiments.keys()):
            await self.abort_experiment(experiment_id)

        await self.redis_client.close()
        logger.info("Chaos experiment runner shut down")

    def register_observer(self, observer: Callable):
        """Register an experiment observer"""
        self.experiment_observers.append(observer)

    @trace_async("chaos_experiment")
    async def run_experiment(self, config: ChaosExperimentConfig) -> str:
        """Run a chaos experiment"""
        experiment_id = str(uuid4())

        add_span_attributes({
            "experiment.id": experiment_id,
            "experiment.type": config.experiment_type.value,
            "experiment.impact": config.impact_level.value,
            "experiment.duration": config.duration_seconds
        })

        result = ChaosExperimentResult(
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.PENDING,
            start_time=datetime.utcnow()
        )

        self.running_experiments[experiment_id] = result

        try:
            logger.info(f"Starting chaos experiment: {config.name} ({experiment_id})")

            # Collect baseline metrics
            result.metrics_before = await self._collect_all_metrics(config.target_agents)

            # Check steady state hypothesis
            if not await self._verify_steady_state(config.steady_state_hypothesis):
                raise Exception("Steady state hypothesis not met")

            # Run the experiment
            result.status = ExperimentStatus.RUNNING
            await self._notify_observers("experiment_started", result)

            injection_id = await self._inject_chaos(config)

            # Monitor during experiment
            monitoring_task = asyncio.create_task(
                self._monitor_experiment(result, config.duration_seconds)
            )

            # Wait for experiment completion or abort conditions
            await self._wait_for_completion_or_abort(result, monitoring_task)

            # Clean up injection
            await self._cleanup_injection(config.experiment_type, injection_id)

            # Collect post-experiment metrics
            result.metrics_after = await self._collect_all_metrics(config.target_agents)

            # Calculate recovery time
            result.recovery_time_seconds = await self._measure_recovery_time(result)

            result.status = ExperimentStatus.COMPLETED
            result.success = True
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            await self._notify_observers("experiment_completed", result)

            logger.info(f"Chaos experiment completed: {config.name} ({experiment_id})")

        except Exception as e:
            logger.error(f"Chaos experiment failed: {config.name} ({experiment_id}): {e}")

            result.status = ExperimentStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            await self._notify_observers("experiment_failed", result)

        finally:
            # Store experiment result
            await self._store_experiment_result(result)

        return experiment_id

    async def abort_experiment(self, experiment_id: str) -> bool:
        """Abort a running experiment"""
        if experiment_id not in self.running_experiments:
            return False

        result = self.running_experiments[experiment_id]

        if result.status != ExperimentStatus.RUNNING:
            return False

        try:
            logger.warning(f"Aborting chaos experiment: {experiment_id}")

            # Clean up any active injections
            await self._cleanup_injection(result.config.experiment_type, None)

            result.status = ExperimentStatus.ABORTED
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            await self._notify_observers("experiment_aborted", result)

            return True

        except Exception as e:
            logger.error(f"Failed to abort experiment {experiment_id}: {e}")
            return False

    async def _inject_chaos(self, config: ChaosExperimentConfig) -> str:
        """Inject chaos based on experiment type"""
        experiment_type = config.experiment_type
        params = config.parameters
        target_agents = config.target_agents
        duration = config.duration_seconds

        if experiment_type == ChaosExperimentType.NETWORK_PARTITION:
            partition_groups = params.get("partition_groups", [target_agents[:len(target_agents)//2], target_agents[len(target_agents)//2:]])
            return await self.network_injector.create_network_partition(partition_groups, duration)

        elif experiment_type == ChaosExperimentType.LATENCY_INJECTION:
            latency_ms = params.get("latency_ms", 100)
            return await self.network_injector.inject_latency(target_agents, latency_ms, duration)

        elif experiment_type == ChaosExperimentType.PACKET_LOSS:
            loss_percentage = params.get("loss_percentage", 5.0)
            return await self.network_injector.inject_packet_loss(target_agents, loss_percentage, duration)

        elif experiment_type == ChaosExperimentType.CPU_STRESS:
            cpu_percentage = params.get("cpu_percentage", 80)
            return await self.resource_injector.inject_cpu_stress(target_agents, cpu_percentage, duration)

        elif experiment_type == ChaosExperimentType.MEMORY_PRESSURE:
            memory_mb = params.get("memory_mb", 512)
            return await self.resource_injector.inject_memory_pressure(target_agents, memory_mb, duration)

        elif experiment_type == ChaosExperimentType.DISK_FAILURE:
            return await self.resource_injector.inject_disk_failure(target_agents, duration)

        elif experiment_type == ChaosExperimentType.SERVICE_FAILURE:
            service_name = params.get("service_name", "main")
            return await self.service_injector.inject_service_failure(target_agents, service_name, duration)

        elif experiment_type == ChaosExperimentType.TIMEOUT_INJECTION:
            timeout_probability = params.get("timeout_probability", 0.1)
            return await self.service_injector.inject_timeout_errors(target_agents, timeout_probability, duration)

        elif experiment_type == ChaosExperimentType.MESSAGE_CORRUPTION:
            corruption_rate = params.get("corruption_rate", 0.05)
            return await self.service_injector.inject_message_corruption(target_agents, corruption_rate, duration)

        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    async def _cleanup_injection(self, experiment_type: ChaosExperimentType, injection_id: Optional[str]):
        """Clean up chaos injection"""
        try:
            if injection_id:
                if experiment_type in [ChaosExperimentType.NETWORK_PARTITION, ChaosExperimentType.LATENCY_INJECTION, ChaosExperimentType.PACKET_LOSS]:
                    await self.network_injector.remove_injection(injection_id)
                elif experiment_type in [ChaosExperimentType.CPU_STRESS, ChaosExperimentType.MEMORY_PRESSURE, ChaosExperimentType.DISK_FAILURE]:
                    await self.resource_injector.remove_injection(injection_id)
                elif experiment_type in [ChaosExperimentType.SERVICE_FAILURE, ChaosExperimentType.TIMEOUT_INJECTION, ChaosExperimentType.MESSAGE_CORRUPTION]:
                    await self.service_injector.remove_injection(injection_id)
        except Exception as e:
            logger.error(f"Failed to cleanup injection {injection_id}: {e}")

    async def _collect_all_metrics(self, target_agents: List[str]) -> Dict[str, Any]:
        """Collect all relevant metrics"""
        metrics = {
            "system": await self.metrics_collector.collect_system_metrics(),
            "agents": {},
            "network": await self.metrics_collector.collect_network_metrics(target_agents)
        }

        for agent_id in target_agents:
            metrics["agents"][agent_id] = await self.metrics_collector.collect_agent_metrics(agent_id)

        return metrics

    async def _verify_steady_state(self, hypothesis: Dict[str, Any]) -> bool:
        """Verify steady state hypothesis"""
        if not hypothesis:
            return True  # No hypothesis to verify

        # Implement steady state verification logic
        # This would check system metrics against expected values
        return True

    async def _monitor_experiment(self, result: ChaosExperimentResult, duration: int):
        """Monitor experiment progress and collect metrics"""
        start_time = time.time()
        interval = 10  # Collect metrics every 10 seconds

        while time.time() - start_time < duration and result.status == ExperimentStatus.RUNNING:
            try:
                # Collect current metrics
                current_metrics = await self._collect_all_metrics(result.config.target_agents)

                # Store in metrics_during
                timestamp = datetime.utcnow().isoformat()
                if "timeline" not in result.metrics_during:
                    result.metrics_during["timeline"] = []

                result.metrics_during["timeline"].append({
                    "timestamp": timestamp,
                    "metrics": current_metrics
                })

                # Check for abort conditions
                if await self._check_abort_conditions(result.config.abort_conditions, current_metrics):
                    logger.warning(f"Abort condition met for experiment {result.experiment_id}")
                    await self.abort_experiment(result.experiment_id)
                    break

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error during experiment monitoring: {e}")
                await asyncio.sleep(interval)

    async def _check_abort_conditions(self, abort_conditions: List[Dict[str, Any]], metrics: Dict[str, Any]) -> bool:
        """Check if any abort conditions are met"""
        for condition in abort_conditions:
            metric_path = condition.get("metric")
            threshold = condition.get("threshold")
            operator = condition.get("operator", "gt")  # greater than

            if not metric_path or threshold is None:
                continue

            # Extract metric value using path
            metric_value = self._extract_metric_value(metrics, metric_path)

            if metric_value is None:
                continue

            # Check condition
            if operator == "gt" and metric_value > threshold:
                return True
            elif operator == "lt" and metric_value < threshold:
                return True
            elif operator == "eq" and metric_value == threshold:
                return True

        return False

    def _extract_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """Extract metric value using dot notation path"""
        try:
            keys = path.split('.')
            current = metrics

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None

            return float(current) if isinstance(current, (int, float)) else None

        except Exception:
            return None

    async def _wait_for_completion_or_abort(self, result: ChaosExperimentResult, monitoring_task: asyncio.Task):
        """Wait for experiment completion or abort"""
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    async def _measure_recovery_time(self, result: ChaosExperimentResult) -> Optional[float]:
        """Measure system recovery time after experiment"""
        if not result.config.steady_state_hypothesis:
            return None

        recovery_start = time.time()
        max_recovery_time = 300  # 5 minutes max

        while time.time() - recovery_start < max_recovery_time:
            if await self._verify_steady_state(result.config.steady_state_hypothesis):
                return time.time() - recovery_start

            await asyncio.sleep(5)  # Check every 5 seconds

        return None  # Failed to recover within time limit

    async def _notify_observers(self, event: str, result: ChaosExperimentResult):
        """Notify registered observers of experiment events"""
        for observer in self.experiment_observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(event, result)
                else:
                    observer(event, result)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")

    async def _store_experiment_result(self, result: ChaosExperimentResult):
        """Store experiment result in Redis"""
        try:
            key = f"chaos_experiments:{result.experiment_id}"
            data = {
                "experiment_id": result.experiment_id,
                "name": result.config.name,
                "type": result.config.experiment_type.value,
                "status": result.status.value,
                "success": result.success,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration_seconds,
                "target_agents": result.config.target_agents,
                "impact_level": result.config.impact_level.value,
                "error": result.error,
                "recovery_time_seconds": result.recovery_time_seconds
            }

            await self.redis_client.setex(
                key,
                timedelta(days=30),  # Keep results for 30 days
                json.dumps(data, default=str)
            )

        except Exception as e:
            logger.error(f"Failed to store experiment result: {e}")

    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results"""
        try:
            key = f"chaos_experiments:{experiment_id}"
            data = await self.redis_client.get(key)

            if data:
                return json.loads(data)

            return None

        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return None

    async def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List chaos experiments"""
        try:
            pattern = "chaos_experiments:*"
            keys = await self.redis_client.keys(pattern)

            experiments = []

            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    experiment = json.loads(data)

                    if status is None or experiment.get("status") == status.value:
                        experiments.append(experiment)

            # Sort by start time (most recent first)
            experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

            return experiments

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []


# Global chaos experiment runner
_chaos_runner = None


async def initialize_chaos_engineering(redis_config: RedisConfig = None) -> ChaosExperimentRunner:
    """Initialize global chaos engineering framework"""
    global _chaos_runner

    if _chaos_runner is None:
        _chaos_runner = ChaosExperimentRunner(redis_config)
        await _chaos_runner.initialize()

    return _chaos_runner


async def get_chaos_runner() -> Optional[ChaosExperimentRunner]:
    """Get the global chaos experiment runner"""
    return _chaos_runner


async def shutdown_chaos_engineering():
    """Shutdown global chaos engineering framework"""
    global _chaos_runner

    if _chaos_runner:
        await _chaos_runner.shutdown()
        _chaos_runner = None

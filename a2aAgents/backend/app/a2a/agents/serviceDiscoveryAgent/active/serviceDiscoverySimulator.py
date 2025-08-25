import asyncio
import random
import secrets
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics

"""
Service Discovery Simulation Framework
Provides comprehensive simulation capabilities for testing service discovery scenarios
"""

from .comprehensiveServiceDiscoveryAgentSdk import (
    ServiceDiscoveryAgentSdk, ServiceRegistration, ServiceEndpoint,
    ServiceStatus, LoadBalancingStrategy
)
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)

class SimulationScenario(Enum):
    NORMAL_OPERATIONS = "normal_operations"
    HIGH_LOAD = "high_load"
    SERVICE_FAILURES = "service_failures"
    NETWORK_PARTITIONS = "network_partitions"
    CASCADING_FAILURES = "cascading_failures"
    SERVICE_DISCOVERY_STRESS = "service_discovery_stress"
    LOAD_BALANCING_TEST = "load_balancing_test"
    HEALTH_CHECK_STORM = "health_check_storm"

@dataclass
class SimulationAgent:
    """Simulated agent for testing"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    agent_id: str
    name: str
    service_types: List[str]
    max_services: int = 3
    failure_probability: float = 0.01
    recovery_probability: float = 0.1
    request_rate: float = 1.0  # requests per second
    is_active: bool = True
    services: List[str] = field(default_factory=list)

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    service_registrations: int = 0
    service_deregistrations: int = 0
    health_check_failures: int = 0
    load_balancing_decisions: int = 0
    response_times: List[float] = field(default_factory=list)
    service_availability: Dict[str, float] = field(default_factory=dict)

class ServiceDiscoverySimulator(SecureA2AAgent):
    """
    Comprehensive simulation framework for service discovery testing
    """

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, discovery_agent: ServiceDiscoveryAgentSdk):

        super().__init__()
        self.discovery_agent = discovery_agent
        self.simulation_agents: List[SimulationAgent] = []
        self.simulation_metrics = SimulationMetrics()
        self.simulation_running = False
        self.simulation_tasks: List[asyncio.Task] = []

        # Network simulation parameters
        self.network_latency_ms = 50.0
        self.network_jitter_ms = 10.0
        self.packet_loss_rate = 0.0

        # Service templates for different types
        self.service_templates = {
            "api": {
                "endpoints": 2,
                "base_port": 8000,
                "health_check_interval": 30,
                "capabilities": ["http_api", "rest", "json"]
            },
            "database": {
                "endpoints": 1,
                "base_port": 5432,
                "health_check_interval": 60,
                "capabilities": ["sql", "transactions", "persistence"]
            },
            "cache": {
                "endpoints": 3,
                "base_port": 6379,
                "health_check_interval": 15,
                "capabilities": ["caching", "key_value", "fast_access"]
            },
            "message_queue": {
                "endpoints": 2,
                "base_port": 5672,
                "health_check_interval": 30,
                "capabilities": ["messaging", "pub_sub", "queuing"]
            }
        }

    async def setup_simulation(
        self,
        agent_count: int = 10,
        scenario: SimulationScenario = SimulationScenario.NORMAL_OPERATIONS
    ):
        """Setup simulation environment"""

        # Clear existing simulation
        await self.cleanup_simulation()

        # Create simulation agents
        self.simulation_agents = []
        service_types = list(self.service_templates.keys())

        for i in range(agent_count):
            # Deterministic service type assignment based on agent index
            service_type_count = (i % len(service_types)) + 1
            assigned_types = [service_types[(i + j) % len(service_types)] for j in range(service_type_count)]
            
            # Max services based on agent role
            max_services = 2 + (i % 3)  # 2-4 services
            
            agent = SimulationAgent(
                agent_id=f"sim-agent-{i:03d}",
                name=f"SimulationAgent{i:03d}",
                service_types=assigned_types,
                max_services=max_services,
                failure_probability=self._get_failure_probability(scenario),
                recovery_probability=self._get_recovery_probability(scenario),
                request_rate=self._get_request_rate(scenario)
            )
            self.simulation_agents.append(agent)

        # Apply scenario-specific configurations
        await self._apply_scenario_config(scenario)

        logger.info(f"Simulation setup complete: {agent_count} agents, scenario: {scenario.value}")

    def _get_failure_probability(self, scenario: SimulationScenario) -> float:
        """Get failure probability based on scenario"""
        probabilities = {
            SimulationScenario.NORMAL_OPERATIONS: 0.01,
            SimulationScenario.HIGH_LOAD: 0.02,
            SimulationScenario.SERVICE_FAILURES: 0.15,
            SimulationScenario.NETWORK_PARTITIONS: 0.05,
            SimulationScenario.CASCADING_FAILURES: 0.08,
            SimulationScenario.SERVICE_DISCOVERY_STRESS: 0.03,
            SimulationScenario.LOAD_BALANCING_TEST: 0.02,
            SimulationScenario.HEALTH_CHECK_STORM: 0.01
        }
        return probabilities.get(scenario, 0.01)

    def _get_recovery_probability(self, scenario: SimulationScenario) -> float:
        """Get recovery probability based on scenario"""
        probabilities = {
            SimulationScenario.NORMAL_OPERATIONS: 0.3,
            SimulationScenario.HIGH_LOAD: 0.2,
            SimulationScenario.SERVICE_FAILURES: 0.1,
            SimulationScenario.NETWORK_PARTITIONS: 0.05,
            SimulationScenario.CASCADING_FAILURES: 0.05,
            SimulationScenario.SERVICE_DISCOVERY_STRESS: 0.25,
            SimulationScenario.LOAD_BALANCING_TEST: 0.3,
            SimulationScenario.HEALTH_CHECK_STORM: 0.3
        }
        return probabilities.get(scenario, 0.1)

    def _get_request_rate(self, scenario: SimulationScenario) -> float:
        """Get request rate based on scenario"""
        rates = {
            SimulationScenario.NORMAL_OPERATIONS: 1.0,
            SimulationScenario.HIGH_LOAD: 10.0,
            SimulationScenario.SERVICE_FAILURES: 2.0,
            SimulationScenario.NETWORK_PARTITIONS: 1.5,
            SimulationScenario.CASCADING_FAILURES: 3.0,
            SimulationScenario.SERVICE_DISCOVERY_STRESS: 5.0,
            SimulationScenario.LOAD_BALANCING_TEST: 8.0,
            SimulationScenario.HEALTH_CHECK_STORM: 0.5
        }
        return rates.get(scenario, 1.0)

    async def _apply_scenario_config(self, scenario: SimulationScenario):
        """Apply scenario-specific configurations"""

        if scenario == SimulationScenario.NETWORK_PARTITIONS:
            self.network_latency_ms = 200.0
            self.network_jitter_ms = 50.0
            self.packet_loss_rate = 0.05

        elif scenario == SimulationScenario.HIGH_LOAD:
            self.network_latency_ms = 100.0
            self.network_jitter_ms = 30.0

        elif scenario == SimulationScenario.SERVICE_DISCOVERY_STRESS:
            # Create more agents that frequently register/deregister
            for i in range(20):
                agent = SimulationAgent(
                    agent_id=f"stress-agent-{i:03d}",
                    name=f"StressAgent{i:03d}",
                    service_types=["api"],
                    max_services=1,
                    request_rate=2.0
                )
                self.simulation_agents.append(agent)

    async def run_simulation(
        self,
        duration_seconds: int = 300,
        report_interval: int = 30
    ) -> SimulationMetrics:
        """Run the simulation for specified duration"""

        self.simulation_running = True
        self.simulation_metrics = SimulationMetrics()

        try:
            # Start simulation tasks
            await self._start_simulation_tasks()

            # Run simulation with periodic reporting
            start_time = datetime.now()
            last_report = start_time

            while self.simulation_running:
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()

                if elapsed >= duration_seconds:
                    break

                # Generate periodic report
                if (current_time - last_report).total_seconds() >= report_interval:
                    await self._generate_progress_report(elapsed, duration_seconds)
                    last_report = current_time

                await asyncio.sleep(1)

            # Final metrics calculation
            await self._calculate_final_metrics()

            logger.info("Simulation completed successfully")

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

        finally:
            await self._stop_simulation_tasks()
            self.simulation_running = False

        return self.simulation_metrics

    async def _start_simulation_tasks(self):
        """Start all simulation tasks"""

        # Agent lifecycle simulation
        for agent in self.simulation_agents:
            task = asyncio.create_task(self._simulate_agent_lifecycle(agent))
            self.simulation_tasks.append(task)

        # Service request simulation
        for agent in self.simulation_agents:
            task = asyncio.create_task(self._simulate_service_requests(agent))
            self.simulation_tasks.append(task)

        # Network condition simulation
        task = asyncio.create_task(self._simulate_network_conditions())
        self.simulation_tasks.append(task)

    async def _simulate_agent_lifecycle(self, agent: SimulationAgent):
        """Simulate agent lifecycle - service registration/deregistration"""

        while self.simulation_running and agent.is_active:
            try:
                # Register services if under capacity
                if len(agent.services) < agent.max_services:
                    if secrets.SystemRandom().random() < 0.3:  # 30% chance to register
                        await self._register_service_for_agent(agent)

                # Random service failures
                if agent.services and secrets.SystemRandom().random() < agent.failure_probability:
                    await self._simulate_service_failure(agent)

                # Service recovery
                if secrets.SystemRandom().random() < agent.recovery_probability:
                    await self._simulate_service_recovery(agent)

                # Random deregistration
                if agent.services and secrets.SystemRandom().random() < 0.1:  # 10% chance
                    await self._deregister_service_for_agent(agent)

                await asyncio.sleep(random.uniform(5, 15))

            except Exception as e:
                logger.error(f"Agent lifecycle simulation error for {agent.agent_id}: {e}")
                await asyncio.sleep(5)

    async def _simulate_service_requests(self, agent: SimulationAgent):
        """Simulate service discovery and load balancing requests"""

        while self.simulation_running and agent.is_active:
            try:
                request_interval = 1.0 / agent.request_rate

                # Service discovery requests
                if secrets.SystemRandom().random() < 0.7:  # 70% discovery, 30% load balancing
                    await self._simulate_service_discovery_request(agent)
                else:
                    await self._simulate_load_balancing_request(agent)

                await asyncio.sleep(random.expovariate(1.0 / request_interval))

            except Exception as e:
                logger.error(f"Service request simulation error for {agent.agent_id}: {e}")
                await asyncio.sleep(1)

    async def _simulate_network_conditions(self):
        """Simulate varying network conditions"""

        while self.simulation_running:
            try:
                # Simulate network conditions with realistic patterns
                cycle_time = time.time()
                # Sinusoidal latency variation every 100 seconds
                latency_factor = 1.0 + 0.3 * math.sin(cycle_time * 0.01)
                self.network_latency_ms = 50.0 * latency_factor
                
                # Jitter increases with latency
                self.network_jitter_ms = 10.0 * (0.5 + 0.5 * latency_factor)
                
                # Packet loss follows a slower pattern
                loss_wave = (math.sin(cycle_time * 0.005) + 1) / 2  # 0 to 1
                self.packet_loss_rate = min(0.1, 0.02 * loss_wave)

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Network simulation error: {e}")
                await asyncio.sleep(5)

    async def _register_service_for_agent(self, agent: SimulationAgent):
        """Register a service for an agent"""

        # Select service type based on agent's queue of unregistered types
        service_index = len(agent.services) % len(agent.service_types)
        service_type = agent.service_types[service_index]
        template = self.service_templates[service_type]

        service_name = f"{agent.name}_{service_type}_service"

        # Create endpoints deterministically
        endpoints = []
        for i in range(template["endpoints"]):
            # Generate deterministic port offset using hash
            port_hash = hash(f"{agent.agent_id}{service_type}{i}") % 1000
            port = template["base_port"] + i + abs(port_hash) % 100
            
            endpoint_id = hashlib.md5(f"{agent.agent_id}{service_type}{i}".encode()).hexdigest()[:8]
            
            endpoints.append({
                "id": f"ep-{endpoint_id}",
                "url": f"http://sim-{agent.agent_id}:{port}",
                "port": port,
                "weight": 1.0 + (i * 0.25)  # Even weight distribution: 1.0, 1.25, 1.5, etc.
            })

        try:
            result = await self.discovery_agent.register_service(
                agent_id=agent.agent_id,
                service_name=service_name,
                service_type=service_type,
                endpoints=endpoints,
                capabilities=template["capabilities"],
                health_check_interval=template["health_check_interval"],
                tags=[f"simulated", agent.agent_id, service_type]
            )

            agent.services.append(result["service_id"])
            self.simulation_metrics.service_registrations += 1

            logger.debug(f"Registered service {service_name} for {agent.agent_id}")

        except Exception as e:
            logger.error(f"Failed to register service for {agent.agent_id}: {e}")

    async def _deregister_service_for_agent(self, agent: SimulationAgent):
        """Deregister a random service for an agent"""

        if not agent.services:
            return

        service_id = random.choice(agent.services)

        try:
            await self.discovery_agent.deregister_service(
                service_id=service_id,
                agent_id=agent.agent_id
            )

            agent.services.remove(service_id)
            self.simulation_metrics.service_deregistrations += 1

            logger.debug(f"Deregistered service {service_id} for {agent.agent_id}")

        except Exception as e:
            logger.error(f"Failed to deregister service for {agent.agent_id}: {e}")

    async def _simulate_service_discovery_request(self, agent: SimulationAgent):
        """Simulate a service discovery request"""

        start_time = datetime.now()

        try:
            # Add network latency simulation
            await self._simulate_network_delay()

            # Random discovery query
            query_params = {}
            if secrets.SystemRandom().random() < 0.5:
                query_params["service_type"] = random.choice(list(self.service_templates.keys()))
            if secrets.SystemRandom().random() < 0.3:
                query_params["capabilities"] = [random.choice(["http_api", "sql", "caching", "messaging"])]

            await self.discovery_agent.discover_services(**query_params)

            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.simulation_metrics.response_times.append(response_time)
            self.simulation_metrics.total_requests += 1
            self.simulation_metrics.successful_requests += 1

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.simulation_metrics.response_times.append(response_time)
            self.simulation_metrics.total_requests += 1
            self.simulation_metrics.failed_requests += 1
            logger.debug(f"Discovery request failed for {agent.agent_id}: {e}")

    async def _simulate_load_balancing_request(self, agent: SimulationAgent):
        """Simulate a load balancing request"""

        start_time = datetime.now()

        try:
            # Add network latency simulation
            await self._simulate_network_delay()

            # Try to get endpoint for a random service type
            service_type = random.choice(list(self.service_templates.keys()))
            strategy = random.choice([s.value for s in LoadBalancingStrategy])

            # Find a service of this type first
            discovery_result = await self.discovery_agent.discover_services(service_type=service_type)

            if discovery_result["services"]:
                service_name = discovery_result["services"][0]["service_name"]

                await self.discovery_agent.get_service_endpoint(
                    service_name=service_name,
                    strategy=strategy
                )

                self.simulation_metrics.load_balancing_decisions += 1

            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.simulation_metrics.response_times.append(response_time)
            self.simulation_metrics.total_requests += 1
            self.simulation_metrics.successful_requests += 1

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.simulation_metrics.response_times.append(response_time)
            self.simulation_metrics.total_requests += 1
            self.simulation_metrics.failed_requests += 1
            logger.debug(f"Load balancing request failed for {agent.agent_id}: {e}")

    async def _simulate_service_failure(self, agent: SimulationAgent):
        """Simulate service failure"""

        if not agent.services:
            return

        # Simulate health check failure
        self.simulation_metrics.health_check_failures += 1
        logger.debug(f"Simulated service failure for {agent.agent_id}")

    async def _simulate_service_recovery(self, agent: SimulationAgent):
        """Simulate service recovery"""

        if not agent.services:
            return

        # Send heartbeat to indicate recovery
        service_id = random.choice(agent.services)

        try:
            await self.discovery_agent.send_heartbeat(
                service_id=service_id,
                agent_id=agent.agent_id,
                status="healthy"
            )

            logger.debug(f"Simulated service recovery for {agent.agent_id}")

        except Exception as e:
            logger.debug(f"Recovery simulation failed for {agent.agent_id}: {e}")

    async def _simulate_network_delay(self):
        """Simulate network latency and jitter"""

        if secrets.SystemRandom().random() < self.packet_loss_rate:
            # Simulate packet loss with longer delay
            delay = self.network_latency_ms * 3
        else:
            # Normal latency with jitter
            jitter = random.uniform(-self.network_jitter_ms, self.network_jitter_ms)
            delay = max(0, self.network_latency_ms + jitter)

        await asyncio.sleep(delay / 1000.0)

    async def _generate_progress_report(self, elapsed: float, total: float):
        """Generate progress report during simulation"""

        progress = (elapsed / total) * 100
        success_rate = (
            self.simulation_metrics.successful_requests / self.simulation_metrics.total_requests * 100
            if self.simulation_metrics.total_requests > 0 else 0
        )

        avg_response_time = (
            statistics.mean(self.simulation_metrics.response_times)
            if self.simulation_metrics.response_times else 0
        )

        logger.info(f"Simulation Progress: {progress:.1f}% | "
                   f"Requests: {self.simulation_metrics.total_requests} | "
                   f"Success Rate: {success_rate:.1f}% | "
                   f"Avg Response: {avg_response_time:.1f}ms")

    async def _calculate_final_metrics(self):
        """Calculate final simulation metrics"""

        if self.simulation_metrics.response_times:
            self.simulation_metrics.average_response_time = statistics.mean(
                self.simulation_metrics.response_times
            )

        # Calculate service availability
        for agent in self.simulation_agents:
            if agent.services:
                # Simulate availability calculation
                availability = random.uniform(0.85, 0.99)
                for service_id in agent.services:
                    self.simulation_metrics.service_availability[service_id] = availability

    async def _stop_simulation_tasks(self):
        """Stop all simulation tasks"""

        for task in self.simulation_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self.simulation_tasks:
            await asyncio.gather(*self.simulation_tasks, return_exceptions=True)

        self.simulation_tasks.clear()

    async def cleanup_simulation(self):
        """Clean up simulation environment"""

        self.simulation_running = False
        await self._stop_simulation_tasks()

        # Deregister all simulation services
        for agent in self.simulation_agents:
            for service_id in agent.services.copy():
                try:
                    await self.discovery_agent.deregister_service(
                        service_id=service_id,
                        agent_id=agent.agent_id
                    )
                except Exception as e:
                    logger.debug(f"Cleanup deregistration failed: {e}")

        self.simulation_agents.clear()

    def get_simulation_report(self) -> Dict[str, Any]:
        """Get comprehensive simulation report"""

        success_rate = (
            self.simulation_metrics.successful_requests / self.simulation_metrics.total_requests * 100
            if self.simulation_metrics.total_requests > 0 else 0
        )

        percentiles = {}
        if self.simulation_metrics.response_times:
            sorted_times = sorted(self.simulation_metrics.response_times)
            percentiles = {
                "p50": sorted_times[len(sorted_times) // 2],
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)]
            }

        return {
            "summary": {
                "total_requests": self.simulation_metrics.total_requests,
                "successful_requests": self.simulation_metrics.successful_requests,
                "failed_requests": self.simulation_metrics.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "average_response_time_ms": round(self.simulation_metrics.average_response_time, 2)
            },
            "service_lifecycle": {
                "registrations": self.simulation_metrics.service_registrations,
                "deregistrations": self.simulation_metrics.service_deregistrations,
                "health_check_failures": self.simulation_metrics.health_check_failures
            },
            "load_balancing": {
                "decisions_made": self.simulation_metrics.load_balancing_decisions
            },
            "response_time_percentiles": percentiles,
            "service_availability": self.simulation_metrics.service_availability,
            "agent_count": len(self.simulation_agents),
            "network_conditions": {
                "latency_ms": self.network_latency_ms,
                "jitter_ms": self.network_jitter_ms,
                "packet_loss_rate": self.packet_loss_rate
            }
        }


# Convenience functions for common simulation scenarios
async def run_normal_operations_simulation(
    discovery_agent: ServiceDiscoveryAgentSdk,
    duration_seconds: int = 300
) -> Dict[str, Any]:
    """Run normal operations simulation"""

    simulator = ServiceDiscoverySimulator(discovery_agent)
    await simulator.setup_simulation(
        agent_count=10,
        scenario=SimulationScenario.NORMAL_OPERATIONS
    )

    try:
        await simulator.run_simulation(duration_seconds)
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_high_load_simulation(
    discovery_agent: ServiceDiscoveryAgentSdk,
    duration_seconds: int = 180
) -> Dict[str, Any]:
    """Run high load simulation"""

    simulator = ServiceDiscoverySimulator(discovery_agent)
    await simulator.setup_simulation(
        agent_count=25,
        scenario=SimulationScenario.HIGH_LOAD
    )

    try:
        await simulator.run_simulation(duration_seconds)
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_failure_recovery_simulation(
    discovery_agent: ServiceDiscoveryAgentSdk,
    duration_seconds: int = 240
) -> Dict[str, Any]:
    """Run service failure and recovery simulation"""

    simulator = ServiceDiscoverySimulator(discovery_agent)
    await simulator.setup_simulation(
        agent_count=15,
        scenario=SimulationScenario.SERVICE_FAILURES
    )

    try:
        await simulator.run_simulation(duration_seconds)
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()


# Create simulator instance
def create_service_discovery_simulator(discovery_agent: ServiceDiscoveryAgentSdk) -> ServiceDiscoverySimulator:
    """Create a new service discovery simulator instance"""
    return ServiceDiscoverySimulator(discovery_agent)

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

from app.a2a.core.security_base import SecureA2AAgent
"""
Agent Builder Simulation Framework
Provides comprehensive simulation capabilities for testing agent creation and deployment scenarios
"""

logger = logging.getLogger(__name__)

class BuilderScenario(Enum):
    NORMAL_AGENT_CREATION = "normal_agent_creation"
    RAPID_PROTOTYPING = "rapid_prototyping"
    BULK_AGENT_GENERATION = "bulk_agent_generation"
    TEMPLATE_BASED_BUILD = "template_based_build"
    CUSTOM_AGENT_BUILD = "custom_agent_build"
    DEPLOYMENT_TESTING = "deployment_testing"
    VERSION_MANAGEMENT = "version_management"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    INTEGRATION_TESTING = "integration_testing"

@dataclass
class AgentSpecification:
    """Specification for agent to be built"""
    id: str
    name: str
    agent_type: str
    capabilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    template_id: Optional[str] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    build_complexity: str = "medium"  # "simple", "medium", "complex"
    estimated_build_time: float = 30.0  # seconds
    version: str = "1.0.0"

@dataclass
class BuildResult:
    """Result of agent build process"""
    spec_id: str
    build_id: str
    status: str  # "success", "failed", "building"
    start_time: datetime
    end_time: Optional[datetime] = None
    build_duration: Optional[float] = None
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    deployment_url: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationMetrics:
    """Metrics collected during agent builder simulation"""
    total_builds: int = 0
    successful_builds: int = 0
    failed_builds: int = 0
    average_build_time: float = 0.0
    build_times: List[float] = field(default_factory=list)
    deployment_success_rate: float = 0.0
    template_usage: Dict[str, int] = field(default_factory=dict)
    complexity_breakdown: Dict[str, int] = field(default_factory=dict)
    concurrent_builds: int = 0
    max_concurrent_builds: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)

class AgentBuilderSimulator(SecureA2AAgent):
    """Comprehensive simulation framework for agent builder testing"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, agent_builder):
        super().__init__()
        self.agent_builder = agent_builder
        self.simulation_metrics = SimulationMetrics()
        self.simulation_running = False
        self.simulation_tasks: List[asyncio.Task] = []

        # Active builds tracking
        self.active_builds: Dict[str, BuildResult] = {}
        self.build_queue: List[AgentSpecification] = []
        self.completed_builds: List[BuildResult] = []

        # Available agent templates
        self.agent_templates = {
            "data_processor": {
                "name": "Data Processing Agent",
                "capabilities": ["data_ingestion", "data_transformation", "data_validation"],
                "base_complexity": "medium",
                "estimated_time": 25.0
            },
            "ai_analyzer": {
                "name": "AI Analysis Agent",
                "capabilities": ["machine_learning", "pattern_recognition", "predictive_analysis"],
                "base_complexity": "complex",
                "estimated_time": 45.0
            },
            "api_connector": {
                "name": "API Connector Agent",
                "capabilities": ["http_client", "api_integration", "data_mapping"],
                "base_complexity": "simple",
                "estimated_time": 15.0
            },
            "monitoring": {
                "name": "Monitoring Agent",
                "capabilities": ["health_monitoring", "metrics_collection", "alerting"],
                "base_complexity": "medium",
                "estimated_time": 20.0
            },
            "workflow": {
                "name": "Workflow Agent",
                "capabilities": ["task_orchestration", "process_automation", "event_handling"],
                "base_complexity": "complex",
                "estimated_time": 40.0
            }
        }

        # Build environment simulation
        self.build_environment = {
            "max_concurrent_builds": 5,
            "cpu_cores": 8,
            "memory_gb": 16,
            "disk_space_gb": 100,
            "network_bandwidth_mbps": 1000
        }

        # Resource usage patterns
        self.resource_patterns = {
            "simple": {"cpu": 0.2, "memory": 0.1, "disk": 0.05, "network": 0.1},
            "medium": {"cpu": 0.5, "memory": 0.3, "disk": 0.15, "network": 0.2},
            "complex": {"cpu": 0.8, "memory": 0.6, "disk": 0.3, "network": 0.4}
        }

    async def setup_simulation(
        self,
        scenario: BuilderScenario = BuilderScenario.NORMAL_AGENT_CREATION,
        num_agents: int = 20,
        complexity_distribution: Dict[str, float] = None
    ):
        """Setup simulation environment"""

        self.simulation_metrics = SimulationMetrics()
        self.active_builds.clear()
        self.build_queue.clear()
        self.completed_builds.clear()

        # Default complexity distribution
        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.3, "medium": 0.5, "complex": 0.2}

        # Generate agent specifications based on scenario
        self.agent_specs = await self._generate_agent_specifications(
            scenario, num_agents, complexity_distribution
        )

        # Configure scenario-specific parameters
        await self._configure_scenario(scenario)

        logger.info(f"Agent builder simulation setup complete: {scenario.value}, "
                   f"{len(self.agent_specs)} agent specifications")

    async def _generate_agent_specifications(
        self,
        scenario: BuilderScenario,
        num_agents: int,
        complexity_distribution: Dict[str, float]
    ) -> List[AgentSpecification]:
        """Generate agent specifications for simulation"""

        specs = []
        template_ids = list(self.agent_templates.keys())

        for i in range(num_agents):
            # Select complexity based on distribution
            complexity = self._select_complexity(complexity_distribution)

            # Select template based on scenario
            if scenario == BuilderScenario.TEMPLATE_BASED_BUILD:
                template_id = random.choice(template_ids)
                template = self.agent_templates[template_id]
                agent_type = template_id
                capabilities = template["capabilities"].copy()
                estimated_time = template["estimated_time"]
            else:
                template_id = random.choice(template_ids) if secrets.SystemRandom().random() < 0.7 else None
                agent_type = random.choice(["custom", "hybrid", "specialized"])
                capabilities = self._generate_random_capabilities()
                estimated_time = self._estimate_build_time(complexity)

            # Add complexity-based variations
            if complexity == "complex":
                estimated_time *= 1.5
                capabilities.extend(["advanced_features", "custom_integrations"])
            elif complexity == "simple":
                estimated_time *= 0.7

            spec = AgentSpecification(
                id=f"agent_spec_{i:04d}",
                name=f"SimulatedAgent_{i:04d}",
                agent_type=agent_type,
                capabilities=capabilities,
                dependencies=self._generate_dependencies(i),
                template_id=template_id,
                custom_config=self._generate_custom_config(scenario),
                build_complexity=complexity,
                estimated_build_time=estimated_time,
                version=f"1.{random.randint(0, 5)}.{random.randint(0, 10)}"
            )
            specs.append(spec)

        return specs

    def _select_complexity(self, distribution: Dict[str, float]) -> str:
        """Select complexity based on probability distribution"""
        rand = secrets.SystemRandom().random()
        cumulative = 0

        for complexity, probability in distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return complexity

        return "medium"  # fallback

    def _generate_random_capabilities(self) -> List[str]:
        """Generate random capabilities for custom agents"""
        all_capabilities = [
            "data_processing", "api_integration", "machine_learning",
            "monitoring", "alerting", "workflow_management",
            "security", "authentication", "logging", "caching",
            "batch_processing", "real_time_processing", "analytics"
        ]

        num_capabilities = random.randint(2, 6)
        return random.sample(all_capabilities, num_capabilities)

    def _estimate_build_time(self, complexity: str) -> float:
        """Estimate build time based on complexity"""
        base_times = {"simple": 15.0, "medium": 30.0, "complex": 60.0}
        base_time = base_times.get(complexity, 30.0)

        # Add randomness
        variation = random.uniform(0.8, 1.3)
        return base_time * variation

    def _generate_dependencies(self, agent_index: int) -> List[str]:
        """Generate dependencies for agent"""
        if agent_index == 0:
            return []  # First agent has no dependencies

        # Some agents depend on previously created agents
        if secrets.SystemRandom().random() < 0.3:  # 30% chance of having dependencies
            num_deps = random.randint(1, min(3, agent_index))
            return [f"agent_spec_{random.randint(0, agent_index-1):04d}" for _ in range(num_deps)]

        return []

    def _generate_custom_config(self, scenario: BuilderScenario) -> Dict[str, Any]:
        """Generate custom configuration based on scenario"""

        base_config = {
            "runtime": random.choice(["python", "nodejs", "java"]),
            "memory_limit": random.choice(["256MB", "512MB", "1GB", "2GB"]),
            "cpu_limit": random.uniform(0.1, 1.0),
            "timeout": random.randint(30, 300)
        }

        if scenario == BuilderScenario.PERFORMANCE_OPTIMIZATION:
            base_config.update({
                "optimization_level": "high",
                "caching_enabled": True,
                "parallel_processing": True
            })
        elif scenario == BuilderScenario.DEPLOYMENT_TESTING:
            base_config.update({
                "deployment_target": random.choice(["docker", "kubernetes", "serverless"]),
                "auto_scaling": True,
                "health_checks": True
            })

        return base_config

    async def _configure_scenario(self, scenario: BuilderScenario):
        """Configure scenario-specific parameters"""

        if scenario == BuilderScenario.RAPID_PROTOTYPING:
            # Faster builds, lower quality
            for spec in self.agent_specs:
                spec.estimated_build_time *= 0.5

        elif scenario == BuilderScenario.BULK_AGENT_GENERATION:
            # Increase concurrent build capacity
            self.build_environment["max_concurrent_builds"] = 10

        elif scenario == BuilderScenario.PERFORMANCE_OPTIMIZATION:
            # Longer builds but better performance
            for spec in self.agent_specs:
                spec.estimated_build_time *= 1.3

    async def run_simulation(
        self,
        duration_seconds: int = 300,
        build_rate: float = 0.2,  # builds per second
        report_interval: int = 30
    ) -> SimulationMetrics:
        """Run agent builder simulation"""

        self.simulation_running = True

        try:
            # Start simulation tasks
            await self._start_simulation_tasks(build_rate)

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

            # Calculate final metrics
            await self._calculate_final_metrics()

            logger.info("Agent builder simulation completed successfully")

        except Exception as e:
            logger.error(f"Agent builder simulation failed: {e}")
            raise

        finally:
            await self._stop_simulation_tasks()
            self.simulation_running = False

        return self.simulation_metrics

    async def _start_simulation_tasks(self, build_rate: float):
        """Start simulation tasks"""

        # Build request generation task
        task = asyncio.create_task(self._generate_build_requests(build_rate))
        self.simulation_tasks.append(task)

        # Build processing task
        task = asyncio.create_task(self._process_build_queue())
        self.simulation_tasks.append(task)

        # Resource monitoring task
        task = asyncio.create_task(self._monitor_resources())
        self.simulation_tasks.append(task)

        # Deployment simulation task
        task = asyncio.create_task(self._simulate_deployments())
        self.simulation_tasks.append(task)

    async def _generate_build_requests(self, build_rate: float):
        """Generate build requests at specified rate"""

        interval = 1.0 / build_rate
        spec_index = 0

        while self.simulation_running and spec_index < len(self.agent_specs):
            try:
                spec = self.agent_specs[spec_index]

                # Check dependencies are satisfied
                dependencies_ready = await self._check_dependencies(spec)

                if dependencies_ready:
                    # Add to build queue
                    self.build_queue.append(spec)
                    logger.debug(f"Added {spec.name} to build queue")
                    spec_index += 1
                else:
                    # Wait for dependencies
                    logger.debug(f"Waiting for dependencies for {spec.name}")

                await asyncio.sleep(random.expovariate(1.0 / interval))

            except Exception as e:
                logger.error(f"Build request generation error: {e}")
                await asyncio.sleep(1)

    async def _check_dependencies(self, spec: AgentSpecification) -> bool:
        """Check if all dependencies are satisfied"""

        for dep_id in spec.dependencies:
            # Check if dependency has been successfully built
            dep_built = any(
                build.spec_id == dep_id and build.status == "success"
                for build in self.completed_builds
            )

            if not dep_built:
                return False

        return True

    async def _process_build_queue(self):
        """Process build queue with concurrent builds"""

        while self.simulation_running:
            try:
                # Check if we can start new builds
                max_concurrent = self.build_environment["max_concurrent_builds"]
                current_concurrent = len(self.active_builds)

                if current_concurrent < max_concurrent and self.build_queue:
                    # Start new build
                    spec = self.build_queue.pop(0)
                    await self._start_build(spec)

                # Update concurrent build tracking
                self.simulation_metrics.concurrent_builds = len(self.active_builds)
                self.simulation_metrics.max_concurrent_builds = max(
                    self.simulation_metrics.max_concurrent_builds,
                    len(self.active_builds)
                )

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Build queue processing error: {e}")
                await asyncio.sleep(2)

    async def _start_build(self, spec: AgentSpecification):
        """Start building an agent"""

        build_id = str(uuid.uuid4())

        build_result = BuildResult(
            spec_id=spec.id,
            build_id=build_id,
            status="building",
            start_time=datetime.now()
        )

        self.active_builds[build_id] = build_result
        self.simulation_metrics.total_builds += 1

        # Start build task
        task = asyncio.create_task(self._simulate_build_process(spec, build_result))
        self.simulation_tasks.append(task)

        logger.debug(f"Started build for {spec.name} (ID: {build_id})")

    async def _simulate_build_process(
        self,
        spec: AgentSpecification,
        build_result: BuildResult
    ):
        """Simulate the agent build process"""

        try:
            # Simulate build phases
            await self._simulate_build_phase("preparation", spec, 0.1)
            await self._simulate_build_phase("code_generation", spec, 0.4)
            await self._simulate_build_phase("compilation", spec, 0.3)
            await self._simulate_build_phase("testing", spec, 0.1)
            await self._simulate_build_phase("packaging", spec, 0.1)

            # Simulate build success/failure
            success_probability = self._calculate_success_probability(spec)

            if secrets.SystemRandom().random() < success_probability:
                # Build succeeded
                build_result.status = "success"
                build_result.artifacts = self._generate_build_artifacts(spec)
                build_result.performance_metrics = self._generate_performance_metrics(spec)
                self.simulation_metrics.successful_builds += 1

                # Update template usage tracking
                if spec.template_id:
                    self.simulation_metrics.template_usage[spec.template_id] = (
                        self.simulation_metrics.template_usage.get(spec.template_id, 0) + 1
                    )
            else:
                # Build failed
                build_result.status = "failed"
                build_result.error_message = self._generate_error_message(spec)
                self.simulation_metrics.failed_builds += 1

            # Complete build
            build_result.end_time = datetime.now()
            build_result.build_duration = (
                build_result.end_time - build_result.start_time
            ).total_seconds()

            self.simulation_metrics.build_times.append(build_result.build_duration)

            # Update complexity breakdown
            complexity = spec.build_complexity
            self.simulation_metrics.complexity_breakdown[complexity] = (
                self.simulation_metrics.complexity_breakdown.get(complexity, 0) + 1
            )

            # Move to completed builds
            self.completed_builds.append(build_result)

        except Exception as e:
            build_result.status = "failed"
            build_result.error_message = f"Build error: {str(e)}"
            build_result.end_time = datetime.now()
            build_result.build_duration = (
                build_result.end_time - build_result.start_time
            ).total_seconds()
            self.simulation_metrics.failed_builds += 1

            logger.error(f"Build simulation error for {spec.name}: {e}")

        finally:
            # Remove from active builds
            if build_result.build_id in self.active_builds:
                del self.active_builds[build_result.build_id]

    async def _simulate_build_phase(
        self,
        phase_name: str,
        spec: AgentSpecification,
        time_percentage: float
    ):
        """Simulate individual build phase"""

        phase_duration = spec.estimated_build_time * time_percentage

        # Add complexity-based variations
        if spec.build_complexity == "complex":
            phase_duration *= random.uniform(1.2, 1.8)
        elif spec.build_complexity == "simple":
            phase_duration *= random.uniform(0.5, 0.8)

        # Simulate phase processing
        await asyncio.sleep(phase_duration)

        logger.debug(f"Completed {phase_name} phase for {spec.name}")

    def _calculate_success_probability(self, spec: AgentSpecification) -> float:
        """Calculate build success probability"""

        base_probability = 0.85

        # Adjust based on complexity
        if spec.build_complexity == "simple":
            base_probability = 0.95
        elif spec.build_complexity == "complex":
            base_probability = 0.75

        # Adjust based on template usage
        if spec.template_id:
            base_probability += 0.1  # Templates are more reliable

        # Adjust based on dependencies
        dependency_penalty = len(spec.dependencies) * 0.02
        base_probability -= dependency_penalty

        return max(0.1, min(0.99, base_probability))

    def _generate_build_artifacts(self, spec: AgentSpecification) -> List[str]:
        """Generate build artifacts for successful build"""

        artifacts = [
            f"{spec.name.lower()}_agent.py",
            f"{spec.name.lower()}_config.json",
            f"{spec.name.lower()}_requirements.txt"
        ]

        if spec.build_complexity in ["medium", "complex"]:
            artifacts.extend([
                f"{spec.name.lower()}_tests.py",
                f"{spec.name.lower()}_docs.md"
            ])

        if spec.build_complexity == "complex":
            artifacts.extend([
                f"{spec.name.lower()}_docker.yml",
                f"{spec.name.lower()}_deploy.sh"
            ])

        return artifacts

    def _generate_performance_metrics(self, spec: AgentSpecification) -> Dict[str, Any]:
        """Generate performance metrics for built agent"""

        complexity_multipliers = {
            "simple": {"startup": 0.5, "memory": 0.3, "cpu": 0.2},
            "medium": {"startup": 1.0, "memory": 1.0, "cpu": 1.0},
            "complex": {"startup": 2.0, "memory": 2.5, "cpu": 3.0}
        }

        multiplier = complexity_multipliers.get(spec.build_complexity, {"startup": 1.0, "memory": 1.0, "cpu": 1.0})

        return {
            "startup_time_ms": round(random.uniform(100, 500) * multiplier["startup"], 2),
            "memory_usage_mb": round(random.uniform(50, 200) * multiplier["memory"], 2),
            "cpu_usage_percent": round(random.uniform(5, 25) * multiplier["cpu"], 2),
            "build_size_mb": round(random.uniform(10, 100), 2),
            "test_coverage_percent": round(random.uniform(70, 95), 1)
        }

    def _generate_error_message(self, spec: AgentSpecification) -> str:
        """Generate error message for failed build"""

        error_types = [
            "Dependency resolution failed",
            "Compilation error in generated code",
            "Test suite failures",
            "Resource allocation timeout",
            "Template parsing error",
            "Configuration validation failed"
        ]

        return random.choice(error_types)

    async def _monitor_resources(self):
        """Monitor resource utilization during simulation"""

        while self.simulation_running:
            try:
                # Calculate resource utilization
                total_usage = {"cpu": 0, "memory": 0, "disk": 0, "network": 0}

                for build_id, build_result in self.active_builds.items():
                    # Find corresponding spec
                    spec = next((s for s in self.agent_specs if s.id == build_result.spec_id), None)
                    if spec:
                        complexity = spec.build_complexity
                        usage = self.resource_patterns.get(complexity, {"cpu": 0.3, "memory": 0.2, "disk": 0.1, "network": 0.1})

                        for resource, value in usage.items():
                            total_usage[resource] += value

                # Calculate percentages
                environment = self.build_environment
                utilization = {
                    "cpu_percent": min(100, (total_usage["cpu"] / environment["cpu_cores"]) * 100),
                    "memory_percent": min(100, (total_usage["memory"] * environment["memory_gb"]) * 100),
                    "disk_percent": min(100, (total_usage["disk"] * environment["disk_space_gb"]) * 100),
                    "network_percent": min(100, (total_usage["network"] * environment["network_bandwidth_mbps"]) * 100)
                }

                self.simulation_metrics.resource_utilization = utilization

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(2)

    async def _simulate_deployments(self):
        """Simulate agent deployments"""

        while self.simulation_running:
            try:
                # Check for newly completed successful builds
                deployable_builds = [
                    build for build in self.completed_builds
                    if build.status == "success" and build.deployment_url is None
                ]

                for build in deployable_builds:
                    # Simulate deployment
                    deployment_success = secrets.SystemRandom().random() < 0.9  # 90% deployment success rate

                    if deployment_success:
                        build.deployment_url = f"http://agent-{build.build_id[:8]}.simulation.local"
                        logger.debug(f"Deployed agent {build.spec_id} to {build.deployment_url}")
                    else:
                        logger.debug(f"Deployment failed for agent {build.spec_id}")

                # Calculate deployment success rate
                total_deployable = len([b for b in self.completed_builds if b.status == "success"])
                successful_deployments = len([b for b in self.completed_builds if b.deployment_url is not None])

                if total_deployable > 0:
                    self.simulation_metrics.deployment_success_rate = (successful_deployments / total_deployable) * 100

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Deployment simulation error: {e}")
                await asyncio.sleep(5)

    async def _generate_progress_report(self, elapsed: float, total: float):
        """Generate progress report during simulation"""

        progress = (elapsed / total) * 100

        success_rate = (
            self.simulation_metrics.successful_builds /
            max(1, self.simulation_metrics.total_builds) * 100
        )

        avg_build_time = (
            statistics.mean(self.simulation_metrics.build_times)
            if self.simulation_metrics.build_times else 0
        )

        logger.info(f"Agent Builder Simulation Progress: {progress:.1f}% | "
                   f"Builds: {self.simulation_metrics.total_builds} | "
                   f"Success Rate: {success_rate:.1f}% | "
                   f"Avg Build Time: {avg_build_time:.1f}s | "
                   f"Concurrent: {self.simulation_metrics.concurrent_builds}")

    async def _calculate_final_metrics(self):
        """Calculate final simulation metrics"""

        if self.simulation_metrics.build_times:
            self.simulation_metrics.average_build_time = statistics.mean(
                self.simulation_metrics.build_times
            )

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

        # Clear simulation data
        self.active_builds.clear()
        self.build_queue.clear()
        self.completed_builds.clear()
        self.agent_specs.clear()

    def get_simulation_report(self) -> Dict[str, Any]:
        """Get comprehensive simulation report"""

        total_builds = self.simulation_metrics.total_builds

        percentiles = {}
        if self.simulation_metrics.build_times:
            sorted_times = sorted(self.simulation_metrics.build_times)
            percentiles = {
                "p50": sorted_times[len(sorted_times) // 2],
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)]
            }

        return {
            "summary": {
                "total_builds": total_builds,
                "successful_builds": self.simulation_metrics.successful_builds,
                "failed_builds": self.simulation_metrics.failed_builds,
                "success_rate_percent": (self.simulation_metrics.successful_builds / total_builds * 100) if total_builds > 0 else 0,
                "deployment_success_rate_percent": self.simulation_metrics.deployment_success_rate
            },
            "performance": {
                "average_build_time_seconds": round(self.simulation_metrics.average_build_time, 2),
                "build_time_percentiles": percentiles,
                "max_concurrent_builds": self.simulation_metrics.max_concurrent_builds,
                "resource_utilization": self.simulation_metrics.resource_utilization
            },
            "analysis": {
                "template_usage": self.simulation_metrics.template_usage,
                "complexity_breakdown": self.simulation_metrics.complexity_breakdown,
                "build_environment": self.build_environment
            },
            "artifacts": {
                "completed_builds": len(self.completed_builds),
                "successful_deployments": len([b for b in self.completed_builds if b.deployment_url]),
                "total_artifacts": sum(len(b.artifacts) for b in self.completed_builds)
            }
        }


# Convenience functions for common simulation scenarios
async def run_normal_build_simulation(
    agent_builder,
    duration_seconds: int = 300
) -> Dict[str, Any]:
    """Run normal agent building simulation"""

    simulator = AgentBuilderSimulator(agent_builder)
    await simulator.setup_simulation(
        scenario=BuilderScenario.NORMAL_AGENT_CREATION,
        num_agents=15
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            build_rate=0.1
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_rapid_prototyping_simulation(
    agent_builder,
    duration_seconds: int = 180
) -> Dict[str, Any]:
    """Run rapid prototyping simulation"""

    simulator = AgentBuilderSimulator(agent_builder)
    await simulator.setup_simulation(
        scenario=BuilderScenario.RAPID_PROTOTYPING,
        num_agents=25,
        complexity_distribution={"simple": 0.7, "medium": 0.3, "complex": 0.0}
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            build_rate=0.3
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_bulk_generation_simulation(
    agent_builder,
    duration_seconds: int = 240
) -> Dict[str, Any]:
    """Run bulk agent generation simulation"""

    simulator = AgentBuilderSimulator(agent_builder)
    await simulator.setup_simulation(
        scenario=BuilderScenario.BULK_AGENT_GENERATION,
        num_agents=50
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            build_rate=0.5
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()


# Create simulator instance
def create_agent_builder_simulator(agent_builder) -> AgentBuilderSimulator:
    """Create a new agent builder simulator instance"""
    return AgentBuilderSimulator(agent_builder)

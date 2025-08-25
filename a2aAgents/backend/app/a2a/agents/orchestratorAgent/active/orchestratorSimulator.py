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
Orchestrator Simulation Framework
Provides comprehensive simulation capabilities for testing workflow orchestration scenarios
"""

from .comprehensiveOrchestratorAgentSdk import (
    OrchestratorAgentSdk, WorkflowDefinition, WorkflowTask,
    WorkflowStatus, TaskStatus, OrchestrationStrategy
)
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)

class SimulationScenario(Enum):
    NORMAL_WORKFLOW_EXECUTION = "normal_workflow_execution"
    HIGH_CONCURRENCY = "high_concurrency"
    AGENT_FAILURES = "agent_failures"
    COMPLEX_DEPENDENCIES = "complex_dependencies"
    CASCADING_WORKFLOWS = "cascading_workflows"
    RESOURCE_CONTENTION = "resource_contention"
    TIMEOUT_SCENARIOS = "timeout_scenarios"
    RECOVERY_TESTING = "recovery_testing"
    LOAD_BALANCING = "load_balancing"
    COORDINATION_STRESS = "coordination_stress"

@dataclass
class SimulatedAgent(SecureA2AAgent):
    """Simulated agent for orchestration testing"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    agent_id: str
    name: str
    capabilities: List[str]
    max_concurrent_tasks: int = 5
    current_tasks: int = 0
    failure_probability: float = 0.01
    response_time_ms: float = 100.0
    success_rate: float = 0.95
    is_available: bool = True
    load_factor: float = 1.0  # Affects response time

@dataclass
class WorkflowTemplate(SecureA2AAgent):
    """Template for generating test workflows"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    name: str
    task_count: int
    strategy: OrchestrationStrategy
    complexity: str  # "simple", "medium", "complex"
    dependency_ratio: float = 0.3  # Ratio of tasks with dependencies
    agent_types: List[str] = field(default_factory=list)

@dataclass
class SimulationMetrics(SecureA2AAgent):
    """Metrics collected during orchestration simulation"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    workflows_created: int = 0
    workflows_executed: int = 0
    workflows_completed: int = 0
    workflows_failed: int = 0
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    coordination_sessions: int = 0
    average_workflow_duration: float = 0.0
    average_task_duration: float = 0.0
    throughput_per_second: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    workflow_durations: List[float] = field(default_factory=list)
    task_durations: List[float] = field(default_factory=list)

class OrchestratorSimulator(SecureA2AAgent):
    """Comprehensive simulation framework for workflow orchestration testing"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, orchestrator_agent: OrchestratorAgentSdk):

        super().__init__()
        self.orchestrator_agent = orchestrator_agent
        self.simulated_agents: List[SimulatedAgent] = []
        self.simulation_metrics = SimulationMetrics()
        self.simulation_running = False
        self.simulation_tasks: List[asyncio.Task] = []

        # Workflow templates for different complexity levels
        self.workflow_templates = {
            "simple": WorkflowTemplate(
                name="SimpleWorkflow",
                task_count=3,
                strategy=OrchestrationStrategy.SEQUENTIAL,
                complexity="simple",
                agent_types=["data", "calc"]
            ),
            "medium": WorkflowTemplate(
                name="MediumWorkflow",
                task_count=8,
                strategy=OrchestrationStrategy.DAG,
                complexity="medium",
                dependency_ratio=0.4,
                agent_types=["data", "calc", "ai", "sql"]
            ),
            "complex": WorkflowTemplate(
                name="ComplexWorkflow",
                task_count=15,
                strategy=OrchestrationStrategy.DAG,
                complexity="complex",
                dependency_ratio=0.6,
                agent_types=["data", "calc", "ai", "sql", "qa", "validation"]
            )
        }

        # Agent type configurations
        self.agent_configs = {
            "data": {
                "capabilities": ["data_processing", "etl", "validation"],
                "response_time_ms": 200.0,
                "max_concurrent": 10
            },
            "calc": {
                "capabilities": ["mathematical_operations", "statistical_analysis"],
                "response_time_ms": 50.0,
                "max_concurrent": 15
            },
            "ai": {
                "capabilities": ["machine_learning", "inference", "training"],
                "response_time_ms": 500.0,
                "max_concurrent": 3
            },
            "sql": {
                "capabilities": ["database_operations", "query_execution"],
                "response_time_ms": 150.0,
                "max_concurrent": 8
            },
            "qa": {
                "capabilities": ["quality_assurance", "testing", "validation"],
                "response_time_ms": 300.0,
                "max_concurrent": 5
            },
            "validation": {
                "capabilities": ["data_validation", "compliance_check"],
                "response_time_ms": 100.0,
                "max_concurrent": 12
            }
        }

    async def setup_simulation(
        self,
        agent_count_per_type: int = 3,
        scenario: SimulationScenario = SimulationScenario.NORMAL_WORKFLOW_EXECUTION
    ):
        """Setup simulation environment with agents"""

        # Clear existing simulation
        await self.cleanup_simulation()

        # Create simulated agents for each type
        self.simulated_agents = []

        for agent_type, config in self.agent_configs.items():
            for i in range(agent_count_per_type):
                agent = SimulatedAgent(
                    agent_id=f"{agent_type}-agent-{i:02d}",
                    name=f"{agent_type.capitalize()} Agent {i:02d}",
                    capabilities=config["capabilities"],
                    max_concurrent_tasks=config["max_concurrent"],
                    response_time_ms=config["response_time_ms"],
                    failure_probability=self._get_failure_probability(scenario),
                    success_rate=self._get_success_rate(scenario)
                )
                self.simulated_agents.append(agent)

        # Apply scenario-specific configurations
        await self._apply_scenario_config(scenario)

        # Register agents with orchestrator (mock registration)
        for agent in self.simulated_agents:
            self.orchestrator_agent.available_agents[agent.agent_id] = {
                "name": agent.name,
                "capabilities": agent.capabilities,
                "status": "available" if agent.is_available else "unavailable",
                "max_concurrent": agent.max_concurrent_tasks,
                "current_load": agent.current_tasks
            }

        logger.info(f"Simulation setup complete: {len(self.simulated_agents)} agents, scenario: {scenario.value}")

    def _get_failure_probability(self, scenario: SimulationScenario) -> float:
        """Get failure probability based on scenario"""
        probabilities = {
            SimulationScenario.NORMAL_WORKFLOW_EXECUTION: 0.01,
            SimulationScenario.HIGH_CONCURRENCY: 0.02,
            SimulationScenario.AGENT_FAILURES: 0.15,
            SimulationScenario.COMPLEX_DEPENDENCIES: 0.03,
            SimulationScenario.CASCADING_WORKFLOWS: 0.05,
            SimulationScenario.RESOURCE_CONTENTION: 0.08,
            SimulationScenario.TIMEOUT_SCENARIOS: 0.12,
            SimulationScenario.RECOVERY_TESTING: 0.20,
            SimulationScenario.LOAD_BALANCING: 0.02,
            SimulationScenario.COORDINATION_STRESS: 0.04
        }
        return probabilities.get(scenario, 0.01)

    def _get_success_rate(self, scenario: SimulationScenario) -> float:
        """Get success rate based on scenario"""
        rates = {
            SimulationScenario.NORMAL_WORKFLOW_EXECUTION: 0.98,
            SimulationScenario.HIGH_CONCURRENCY: 0.95,
            SimulationScenario.AGENT_FAILURES: 0.80,
            SimulationScenario.COMPLEX_DEPENDENCIES: 0.92,
            SimulationScenario.CASCADING_WORKFLOWS: 0.88,
            SimulationScenario.RESOURCE_CONTENTION: 0.85,
            SimulationScenario.TIMEOUT_SCENARIOS: 0.75,
            SimulationScenario.RECOVERY_TESTING: 0.70,
            SimulationScenario.LOAD_BALANCING: 0.96,
            SimulationScenario.COORDINATION_STRESS: 0.90
        }
        return rates.get(scenario, 0.95)

    async def _apply_scenario_config(self, scenario: SimulationScenario):
        """Apply scenario-specific configurations"""

        if scenario == SimulationScenario.AGENT_FAILURES:
            # Make some agents unavailable
            for agent in random.sample(self.simulated_agents, k=len(self.simulated_agents) // 4):
                agent.is_available = False
                agent.failure_probability = 0.3

        elif scenario == SimulationScenario.HIGH_CONCURRENCY:
            # Reduce max concurrent tasks to create contention
            for agent in self.simulated_agents:
                agent.max_concurrent_tasks = max(1, agent.max_concurrent_tasks // 2)

        elif scenario == SimulationScenario.RESOURCE_CONTENTION:
            # Increase load factors
            for agent in self.simulated_agents:
                agent.load_factor = random.uniform(1.5, 3.0)
                agent.response_time_ms *= agent.load_factor

        elif scenario == SimulationScenario.TIMEOUT_SCENARIOS:
            # Increase response times for some agents
            for agent in random.sample(self.simulated_agents, k=len(self.simulated_agents) // 3):
                agent.response_time_ms *= random.uniform(5.0, 10.0)

    async def run_simulation(
        self,
        duration_seconds: int = 300,
        workflow_generation_rate: float = 0.5,  # workflows per second
        report_interval: int = 30
    ) -> SimulationMetrics:
        """Run the orchestration simulation"""

        self.simulation_running = True
        self.simulation_metrics = SimulationMetrics()

        try:
            # Start simulation tasks
            await self._start_simulation_tasks(workflow_generation_rate)

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

            logger.info("Orchestration simulation completed successfully")

        except Exception as e:
            logger.error(f"Orchestration simulation failed: {e}")
            raise

        finally:
            await self._stop_simulation_tasks()
            self.simulation_running = False

        return self.simulation_metrics

    async def _start_simulation_tasks(self, workflow_generation_rate: float):
        """Start all simulation tasks"""

        # Workflow generation task
        task = asyncio.create_task(self._generate_workflows_task(workflow_generation_rate))
        self.simulation_tasks.append(task)

        # Agent status simulation task
        task = asyncio.create_task(self._simulate_agent_status_changes())
        self.simulation_tasks.append(task)

        # Resource utilization monitoring task
        task = asyncio.create_task(self._monitor_resource_utilization())
        self.simulation_tasks.append(task)

        # Coordination session simulation
        task = asyncio.create_task(self._simulate_coordination_sessions())
        self.simulation_tasks.append(task)

    async def _generate_workflows_task(self, generation_rate: float):
        """Generate workflows at specified rate"""

        interval = 1.0 / generation_rate

        while self.simulation_running:
            try:
                # Randomly select workflow template
                template_name = random.choice(list(self.workflow_templates.keys()))
                template = self.workflow_templates[template_name]

                # Generate and execute workflow
                workflow_id = await self._generate_and_execute_workflow(template)

                if workflow_id:
                    self.simulation_metrics.workflows_created += 1
                    self.simulation_metrics.workflows_executed += 1

                # Wait for next generation
                await asyncio.sleep(random.expovariate(1.0 / interval))

            except Exception as e:
                logger.error(f"Workflow generation error: {e}")
                await asyncio.sleep(1)

    async def _generate_and_execute_workflow(self, template: WorkflowTemplate) -> Optional[str]:
        """Generate and execute a workflow based on template"""

        try:
            # Generate tasks for workflow
            tasks = await self._generate_workflow_tasks(template)

            # Create workflow
            result = await self.orchestrator_agent.create_workflow(
                workflow_name=f"{template.name}_{datetime.now().strftime('%H%M%S')}_{random.randint(100, 999)}",
                description=f"Simulated {template.complexity} workflow",
                tasks=tasks,
                strategy=template.strategy.value,
                timeout_minutes=30
            )

            workflow_id = result["workflow_id"]

            # Execute workflow
            await self.orchestrator_agent.execute_workflow(
                workflow_id=workflow_id,
                execution_context={"simulation": True, "template": template.name}
            )

            # Start monitoring task for this workflow
            asyncio.create_task(self._monitor_workflow_execution(workflow_id))

            return workflow_id

        except Exception as e:
            logger.error(f"Failed to generate/execute workflow: {e}")
            return None

    async def _generate_workflow_tasks(self, template: WorkflowTemplate) -> List[Dict[str, Any]]:
        """Generate tasks for a workflow based on template"""

        tasks = []
        available_agent_types = template.agent_types

        for i in range(template.task_count):
            # Select agent type and specific agent
            agent_type = random.choice(available_agent_types)
            suitable_agents = [
                agent for agent in self.simulated_agents
                if agent.agent_id.startswith(agent_type) and agent.is_available
            ]

            if not suitable_agents:
                # Fallback to any available agent
                suitable_agents = [agent for agent in self.simulated_agents if agent.is_available]

            if not suitable_agents:
                continue  # Skip if no agents available

            selected_agent = random.choice(suitable_agents)

            # Create task
            task = {
                "id": f"task-{i:03d}",
                "name": f"SimTask_{i:03d}_{agent_type}",
                "agent_id": selected_agent.agent_id,
                "action": f"simulate_{random.choice(['process', 'analyze', 'transform', 'validate'])}",
                "parameters": {
                    "data_size": random.randint(100, 10000),
                    "complexity": template.complexity,
                    "priority": random.choice(["low", "medium", "high"])
                },
                "timeout_seconds": random.randint(60, 300),
                "max_retries": random.randint(1, 3)
            }

            # Add dependencies based on template
            if i > 0 and secrets.SystemRandom().random() < template.dependency_ratio:
                # Add dependency on previous task(s)
                num_deps = min(i, random.randint(1, 3))
                dependencies = random.sample(
                    [f"task-{j:03d}" for j in range(i)],
                    num_deps
                )
                task["dependencies"] = dependencies

            tasks.append(task)

        return tasks

    async def _monitor_workflow_execution(self, workflow_id: str):
        """Monitor individual workflow execution"""

        start_time = datetime.now()

        try:
            while self.simulation_running:
                status = await self.orchestrator_agent.get_workflow_status(workflow_id)

                if status["status"] in ["completed", "failed", "cancelled"]:
                    # Workflow finished
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()

                    self.simulation_metrics.workflow_durations.append(duration)

                    if status["status"] == "completed":
                        self.simulation_metrics.workflows_completed += 1
                    else:
                        self.simulation_metrics.workflows_failed += 1

                    # Update task metrics
                    for task in status["tasks"]:
                        self.simulation_metrics.total_tasks += 1
                        if task["status"] == "completed":
                            self.simulation_metrics.tasks_completed += 1
                        elif task["status"] == "failed":
                            self.simulation_metrics.tasks_failed += 1

                    break

                await asyncio.sleep(5)  # Check every 5 seconds

        except Exception as e:
            logger.error(f"Workflow monitoring error for {workflow_id}: {e}")

    async def _simulate_agent_status_changes(self):
        """Simulate dynamic agent status changes"""

        while self.simulation_running:
            try:
                # Randomly change agent status
                agent = random.choice(self.simulated_agents)

                if secrets.SystemRandom().random() < agent.failure_probability:
                    if agent.is_available:
                        agent.is_available = False
                        logger.debug(f"Simulated failure for agent {agent.agent_id}")
                elif not agent.is_available and secrets.SystemRandom().random() < 0.3:  # 30% recovery chance
                    agent.is_available = True
                    logger.debug(f"Simulated recovery for agent {agent.agent_id}")

                # Simulate load changes
                if agent.is_available:
                    # Randomly adjust current task count
                    delta = random.randint(-2, 3)
                    agent.current_tasks = max(0, min(agent.max_concurrent_tasks, agent.current_tasks + delta))

                    # Adjust response time based on load
                    base_response_time = self.agent_configs[agent.agent_id.split('-')[0]]["response_time_ms"]
                    load_multiplier = 1.0 + (agent.current_tasks / agent.max_concurrent_tasks)
                    agent.response_time_ms = base_response_time * load_multiplier * agent.load_factor

                await asyncio.sleep(random.uniform(10, 30))

            except Exception as e:
                logger.error(f"Agent status simulation error: {e}")
                await asyncio.sleep(5)

    async def _monitor_resource_utilization(self):
        """Monitor resource utilization across agents"""

        while self.simulation_running:
            try:
                # Calculate utilization per agent type
                agent_type_utilization = {}

                for agent_type in self.agent_configs.keys():
                    agents_of_type = [
                        agent for agent in self.simulated_agents
                        if agent.agent_id.startswith(agent_type)
                    ]

                    if agents_of_type:
                        total_capacity = sum(agent.max_concurrent_tasks for agent in agents_of_type)
                        current_usage = sum(agent.current_tasks for agent in agents_of_type)
                        utilization = (current_usage / total_capacity) if total_capacity > 0 else 0
                        agent_type_utilization[agent_type] = utilization

                self.simulation_metrics.resource_utilization = agent_type_utilization

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)

    async def _simulate_coordination_sessions(self):
        """Simulate agent coordination sessions"""

        while self.simulation_running:
            try:
                # Randomly create coordination sessions
                if secrets.SystemRandom().random() < 0.1:  # 10% chance per cycle
                    agents = random.sample(
                        [agent.agent_id for agent in self.simulated_agents if agent.is_available],
                        k=min(random.randint(2, 5), len([a for a in self.simulated_agents if a.is_available]))
                    )

                    if len(agents) >= 2:
                        coordination_plan = {
                            "steps": [
                                {
                                    "id": f"coord-step-{i}",
                                    "action": "coordinate",
                                    "agents": agents,
                                    "objective": f"simulation_coordination_{i}"
                                }
                                for i in range(random.randint(1, 3))
                            ]
                        }

                        await self.orchestrator_agent.coordinate_agents(
                            coordination_plan=coordination_plan,
                            agents=agents,
                            objective="Simulation coordination test"
                        )

                        self.simulation_metrics.coordination_sessions += 1

                await asyncio.sleep(random.uniform(30, 60))

            except Exception as e:
                logger.error(f"Coordination simulation error: {e}")
                await asyncio.sleep(10)

    async def _generate_progress_report(self, elapsed: float, total: float):
        """Generate progress report during simulation"""

        progress = (elapsed / total) * 100

        success_rate = (
            self.simulation_metrics.workflows_completed /
            max(1, self.simulation_metrics.workflows_executed) * 100
        )

        task_success_rate = (
            self.simulation_metrics.tasks_completed /
            max(1, self.simulation_metrics.total_tasks) * 100
        )

        avg_duration = (
            statistics.mean(self.simulation_metrics.workflow_durations)
            if self.simulation_metrics.workflow_durations else 0
        )

        logger.info(f"Simulation Progress: {progress:.1f}% | "
                   f"Workflows: {self.simulation_metrics.workflows_executed} | "
                   f"Success Rate: {success_rate:.1f}% | "
                   f"Task Success: {task_success_rate:.1f}% | "
                   f"Avg Duration: {avg_duration:.1f}s")

    async def _calculate_final_metrics(self):
        """Calculate final simulation metrics"""

        if self.simulation_metrics.workflow_durations:
            self.simulation_metrics.average_workflow_duration = statistics.mean(
                self.simulation_metrics.workflow_durations
            )

        if self.simulation_metrics.task_durations:
            self.simulation_metrics.average_task_duration = statistics.mean(
                self.simulation_metrics.task_durations
            )

        # Calculate throughput
        total_time = max(self.simulation_metrics.workflow_durations) if self.simulation_metrics.workflow_durations else 1
        self.simulation_metrics.throughput_per_second = (
            self.simulation_metrics.workflows_completed / total_time
        )

        # Calculate error rates
        self.simulation_metrics.error_rates = {
            "workflow_failure_rate": (
                self.simulation_metrics.workflows_failed /
                max(1, self.simulation_metrics.workflows_executed) * 100
            ),
            "task_failure_rate": (
                self.simulation_metrics.tasks_failed /
                max(1, self.simulation_metrics.total_tasks) * 100
            )
        }

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

        # Clear agent registry
        self.orchestrator_agent.available_agents.clear()

        self.simulated_agents.clear()

    def get_simulation_report(self) -> Dict[str, Any]:
        """Get comprehensive simulation report"""

        percentiles = {}
        if self.simulation_metrics.workflow_durations:
            sorted_durations = sorted(self.simulation_metrics.workflow_durations)
            percentiles = {
                "p50": sorted_durations[len(sorted_durations) // 2],
                "p95": sorted_durations[int(len(sorted_durations) * 0.95)],
                "p99": sorted_durations[int(len(sorted_durations) * 0.99)]
            }

        return {
            "summary": {
                "workflows_created": self.simulation_metrics.workflows_created,
                "workflows_executed": self.simulation_metrics.workflows_executed,
                "workflows_completed": self.simulation_metrics.workflows_completed,
                "workflows_failed": self.simulation_metrics.workflows_failed,
                "total_tasks": self.simulation_metrics.total_tasks,
                "tasks_completed": self.simulation_metrics.tasks_completed,
                "tasks_failed": self.simulation_metrics.tasks_failed,
                "coordination_sessions": self.simulation_metrics.coordination_sessions
            },
            "performance": {
                "average_workflow_duration_seconds": round(self.simulation_metrics.average_workflow_duration, 2),
                "average_task_duration_seconds": round(self.simulation_metrics.average_task_duration, 2),
                "throughput_per_second": round(self.simulation_metrics.throughput_per_second, 3),
                "workflow_duration_percentiles": percentiles
            },
            "reliability": {
                "workflow_success_rate_percent": round(
                    (self.simulation_metrics.workflows_completed /
                     max(1, self.simulation_metrics.workflows_executed)) * 100, 2
                ),
                "task_success_rate_percent": round(
                    (self.simulation_metrics.tasks_completed /
                     max(1, self.simulation_metrics.total_tasks)) * 100, 2
                ),
                "error_rates": self.simulation_metrics.error_rates
            },
            "resource_utilization": self.simulation_metrics.resource_utilization,
            "agent_count": len(self.simulated_agents)
        }


# Convenience functions for common simulation scenarios
async def run_normal_orchestration_simulation(
    orchestrator_agent: OrchestratorAgentSdk,
    duration_seconds: int = 300
) -> Dict[str, Any]:
    """Run normal orchestration simulation"""

    simulator = OrchestratorSimulator(orchestrator_agent)
    await simulator.setup_simulation(
        agent_count_per_type=3,
        scenario=SimulationScenario.NORMAL_WORKFLOW_EXECUTION
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            workflow_generation_rate=0.3
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_high_concurrency_simulation(
    orchestrator_agent: OrchestratorAgentSdk,
    duration_seconds: int = 180
) -> Dict[str, Any]:
    """Run high concurrency simulation"""

    simulator = OrchestratorSimulator(orchestrator_agent)
    await simulator.setup_simulation(
        agent_count_per_type=2,
        scenario=SimulationScenario.HIGH_CONCURRENCY
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            workflow_generation_rate=1.0
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_failure_recovery_simulation(
    orchestrator_agent: OrchestratorAgentSdk,
    duration_seconds: int = 240
) -> Dict[str, Any]:
    """Run agent failure and recovery simulation"""

    simulator = OrchestratorSimulator(orchestrator_agent)
    await simulator.setup_simulation(
        agent_count_per_type=4,
        scenario=SimulationScenario.AGENT_FAILURES
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            workflow_generation_rate=0.5
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_complex_workflow_simulation(
    orchestrator_agent: OrchestratorAgentSdk,
    duration_seconds: int = 360
) -> Dict[str, Any]:
    """Run complex workflow with dependencies simulation"""

    simulator = OrchestratorSimulator(orchestrator_agent)
    await simulator.setup_simulation(
        agent_count_per_type=5,
        scenario=SimulationScenario.COMPLEX_DEPENDENCIES
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            workflow_generation_rate=0.2
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()


# Create simulator instance
def create_orchestrator_simulator(orchestrator_agent: OrchestratorAgentSdk) -> OrchestratorSimulator:
    """Create a new orchestrator simulator instance"""
    return OrchestratorSimulator(orchestrator_agent)

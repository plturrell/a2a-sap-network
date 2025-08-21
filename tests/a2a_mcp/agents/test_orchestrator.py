"""
A2A Test Orchestration Agent
Intelligent agent for managing test execution workflows and optimization
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..tools.test_executor import TestExecutor, TestSuite, TestSuiteBuilder, TestReporter, TestResult, TestStatus

logger = logging.getLogger(__name__)

class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

class WorkflowStatus(Enum):
    """Test workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TestWorkflow:
    """Test execution workflow definition."""
    id: str
    name: str
    suites: List[TestSuite]
    priority: TestPriority
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 600
    retry_count: int = 1
    parallel: bool = True
    coverage_required: bool = False
    status: WorkflowStatus = WorkflowStatus.PENDING
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@dataclass
class TestAgent:
    """Individual test execution agent."""
    id: str
    name: str
    capabilities: List[str]
    max_concurrent: int = 2
    current_load: int = 0
    status: str = "idle"
    assigned_workflows: List[str] = field(default_factory=list)

class TestOrchestrator:
    """Intelligent test orchestration agent with workflow management."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
        self.executor = TestExecutor(test_root)
        self.suite_builder = TestSuiteBuilder(test_root)
        self.reporter = TestReporter(test_root)
        
        # Workflow management
        self.workflows: Dict[str, TestWorkflow] = {}
        self.workflow_queue: List[str] = []
        self.running_workflows: Set[str] = set()
        
        # Agent management
        self.agents: Dict[str, TestAgent] = {}
        self._initialize_agents()
        
        # Execution history and optimization
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configuration
        self.max_parallel_workflows = 3
        self.default_timeout = 600
    
    def _initialize_agents(self):
        """Initialize test execution agents."""
        self.agents = {
            "unit_agent": TestAgent(
                id="unit_agent",
                name="Unit Test Agent",
                capabilities=["unit", "python", "javascript"],
                max_concurrent=3
            ),
            "integration_agent": TestAgent(
                id="integration_agent", 
                name="Integration Test Agent",
                capabilities=["integration", "python", "javascript", "database"],
                max_concurrent=2
            ),
            "e2e_agent": TestAgent(
                id="e2e_agent",
                name="E2E Test Agent", 
                capabilities=["e2e", "ui", "cypress", "selenium"],
                max_concurrent=1
            ),
            "contract_agent": TestAgent(
                id="contract_agent",
                name="Smart Contract Test Agent",
                capabilities=["contracts", "solidity", "forge", "blockchain"],
                max_concurrent=2
            ),
            "performance_agent": TestAgent(
                id="performance_agent",
                name="Performance Test Agent",
                capabilities=["performance", "load", "stress", "benchmark"],
                max_concurrent=1
            ),
            "security_agent": TestAgent(
                id="security_agent",
                name="Security Test Agent",
                capabilities=["security", "authentication", "authorization", "blockchain"],
                max_concurrent=2
            )
        }
    
    async def create_workflow(
        self,
        name: str,
        test_type: str = "all",
        module: str = "all",
        priority: TestPriority = TestPriority.MEDIUM,
        **kwargs
    ) -> str:
        """Create a new test workflow."""
        workflow_id = f"workflow_{len(self.workflows) + 1}_{int(datetime.now().timestamp())}"
        
        # Build test suites
        suites = self.suite_builder.build_suites(test_type, module)
        
        if not suites:
            raise ValueError(f"No test suites found for type: {test_type}, module: {module}")
        
        # Create workflow
        workflow = TestWorkflow(
            id=workflow_id,
            name=name,
            suites=suites,
            priority=priority,
            timeout=kwargs.get("timeout", self.default_timeout),
            retry_count=kwargs.get("retry_count", 1),
            parallel=kwargs.get("parallel", True),
            coverage_required=kwargs.get("coverage", False),
            dependencies=kwargs.get("dependencies", [])
        )
        
        self.workflows[workflow_id] = workflow
        self.workflow_queue.append(workflow_id)
        
        logger.info(f"Created workflow {workflow_id}: {name} with {len(suites)} suites")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> TestWorkflow:
        """Execute a specific test workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PENDING:
            raise ValueError(f"Workflow {workflow_id} is not in pending status")
        
        # Check dependencies
        for dep_id in workflow.dependencies:
            if dep_id in self.workflows:
                dep_workflow = self.workflows[dep_id]
                if dep_workflow.status != WorkflowStatus.COMPLETED:
                    raise ValueError(f"Dependency workflow {dep_id} not completed")
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = datetime.now()
        self.running_workflows.add(workflow_id)
        
        try:
            # Execute all suites in the workflow
            all_results = []
            
            if workflow.parallel and len(workflow.suites) > 1:
                # Execute suites in parallel
                tasks = []
                for suite in workflow.suites:
                    # Assign to appropriate agent
                    agent = self._assign_agent(suite)
                    if agent:
                        task = asyncio.create_task(
                            self._execute_suite_with_agent(suite, agent, workflow)
                        )
                        tasks.append(task)
                    else:
                        # Execute directly if no agent available
                        task = asyncio.create_task(
                            self.executor.execute_test_suite(
                                suite, 
                                parallel=workflow.parallel,
                                timeout=workflow.timeout,
                                coverage=workflow.coverage_required
                            )
                        )
                        tasks.append(task)
                
                suite_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Flatten results
                for results in suite_results:
                    if isinstance(results, Exception):
                        logger.error(f"Suite execution failed: {results}")
                        continue
                    all_results.extend(results)
            else:
                # Execute suites sequentially
                for suite in workflow.suites:
                    agent = self._assign_agent(suite)
                    if agent:
                        results = await self._execute_suite_with_agent(suite, agent, workflow)
                    else:
                        results = await self.executor.execute_test_suite(
                            suite,
                            parallel=False,
                            timeout=workflow.timeout,
                            coverage=workflow.coverage_required
                        )
                    all_results.extend(results)
            
            workflow.results = all_results
            workflow.end_time = datetime.now()
            
            # Determine workflow status
            if any(r.status == TestStatus.FAILED for r in all_results):
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED
            
            # Record execution history
            self._record_execution(workflow)
            
            logger.info(f"Workflow {workflow_id} completed with status: {workflow.status.value}")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed with error: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
        
        finally:
            self.running_workflows.discard(workflow_id)
            self._release_agents(workflow_id)
        
        return workflow
    
    async def execute_all_pending(self) -> List[TestWorkflow]:
        """Execute all pending workflows based on priority."""
        # Sort workflows by priority
        pending_workflows = [
            wf for wf in self.workflows.values() 
            if wf.status == WorkflowStatus.PENDING
        ]
        
        # Sort by priority (critical first)
        priority_order = {
            TestPriority.CRITICAL: 0,
            TestPriority.HIGH: 1,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 3
        }
        
        pending_workflows.sort(key=lambda w: priority_order[w.priority])
        
        executed_workflows = []
        
        for workflow in pending_workflows:
            if len(self.running_workflows) >= self.max_parallel_workflows:
                logger.info("Maximum parallel workflows reached, queuing remaining")
                break
            
            try:
                executed_workflow = await self.execute_workflow(workflow.id)
                executed_workflows.append(executed_workflow)
            except Exception as e:
                logger.error(f"Failed to execute workflow {workflow.id}: {e}")
        
        return executed_workflows
    
    def _assign_agent(self, suite: TestSuite) -> Optional[TestAgent]:
        """Assign an appropriate agent to execute a test suite."""
        # Find agents capable of handling this suite type
        capable_agents = [
            agent for agent in self.agents.values()
            if suite.type in agent.capabilities and agent.current_load < agent.max_concurrent
        ]
        
        if not capable_agents:
            return None
        
        # Select agent with lowest current load
        selected_agent = min(capable_agents, key=lambda a: a.current_load)
        selected_agent.current_load += 1
        selected_agent.status = "busy"
        
        return selected_agent
    
    async def _execute_suite_with_agent(
        self, 
        suite: TestSuite, 
        agent: TestAgent, 
        workflow: TestWorkflow
    ) -> List[TestResult]:
        """Execute a test suite using a specific agent."""
        logger.info(f"Agent {agent.name} executing suite {suite.name}")
        
        try:
            agent.assigned_workflows.append(workflow.id)
            
            results = await self.executor.execute_test_suite(
                suite,
                parallel=workflow.parallel,
                timeout=workflow.timeout,
                coverage=workflow.coverage_required
            )
            
            return results
        
        finally:
            agent.current_load = max(0, agent.current_load - 1)
            if agent.current_load == 0:
                agent.status = "idle"
            
            if workflow.id in agent.assigned_workflows:
                agent.assigned_workflows.remove(workflow.id)
    
    def _release_agents(self, workflow_id: str):
        """Release agents assigned to a workflow."""
        for agent in self.agents.values():
            if workflow_id in agent.assigned_workflows:
                agent.assigned_workflows.remove(workflow_id)
                agent.current_load = max(0, agent.current_load - 1)
                if agent.current_load == 0:
                    agent.status = "idle"
    
    def _record_execution(self, workflow: TestWorkflow):
        """Record workflow execution for optimization."""
        execution_record = {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "duration": (workflow.end_time - workflow.start_time).total_seconds() if workflow.start_time and workflow.end_time else 0,
            "status": workflow.status.value,
            "suite_count": len(workflow.suites),
            "test_count": len(workflow.results),
            "passed_count": len([r for r in workflow.results if r.status == TestStatus.PASSED]),
            "failed_count": len([r for r in workflow.results if r.status == TestStatus.FAILED]),
            "coverage_enabled": workflow.coverage_required,
            "parallel_enabled": workflow.parallel
        }
        
        self.execution_history.append(execution_record)
        
        # Update performance metrics
        self._update_performance_metrics(execution_record)
    
    def _update_performance_metrics(self, execution_record: Dict[str, Any]):
        """Update performance metrics based on execution history."""
        if "average_duration" not in self.performance_metrics:
            self.performance_metrics["average_duration"] = 0
            self.performance_metrics["total_executions"] = 0
            self.performance_metrics["success_rate"] = 0
        
        # Update averages
        total = self.performance_metrics["total_executions"]
        current_avg = self.performance_metrics["average_duration"]
        new_duration = execution_record["duration"]
        
        self.performance_metrics["average_duration"] = (
            (current_avg * total + new_duration) / (total + 1)
        )
        self.performance_metrics["total_executions"] = total + 1
        
        # Update success rate
        successful_executions = len([
            r for r in self.execution_history 
            if r["status"] == WorkflowStatus.COMPLETED.value
        ])
        self.performance_metrics["success_rate"] = (
            successful_executions / self.performance_metrics["total_executions"]
        )
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed status of a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        status = {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "priority": workflow.priority.value,
            "suite_count": len(workflow.suites),
            "test_count": len(workflow.results),
            "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "duration": (
                (workflow.end_time - workflow.start_time).total_seconds() 
                if workflow.start_time and workflow.end_time 
                else None
            )
        }
        
        if workflow.results:
            status.update({
                "passed": len([r for r in workflow.results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in workflow.results if r.status == TestStatus.FAILED]),
                "errors": len([r for r in workflow.results if r.status == TestStatus.ERROR]),
                "skipped": len([r for r in workflow.results if r.status == TestStatus.SKIPPED])
            })
        
        return status
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all test agents."""
        return {
            agent_id: {
                "name": agent.name,
                "status": agent.status,
                "capabilities": agent.capabilities,
                "current_load": agent.current_load,
                "max_concurrent": agent.max_concurrent,
                "assigned_workflows": agent.assigned_workflows
            }
            for agent_id, agent in self.agents.items()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add current status
        metrics.update({
            "total_workflows": len(self.workflows),
            "pending_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.PENDING]),
            "running_workflows": len(self.running_workflows),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.FAILED])
        })
        
        return metrics
    
    def optimize_execution_strategy(self) -> Dict[str, Any]:
        """Analyze execution history and suggest optimizations."""
        if len(self.execution_history) < 5:
            return {"recommendations": ["Insufficient execution history for optimization"]}
        
        recommendations = []
        
        # Analyze parallel vs sequential performance
        parallel_executions = [r for r in self.execution_history if r.get("parallel_enabled")]
        sequential_executions = [r for r in self.execution_history if not r.get("parallel_enabled")]
        
        if parallel_executions and sequential_executions:
            avg_parallel = sum(r["duration"] for r in parallel_executions) / len(parallel_executions)
            avg_sequential = sum(r["duration"] for r in sequential_executions) / len(sequential_executions)
            
            if avg_parallel < avg_sequential * 0.8:
                recommendations.append("Enable parallel execution for better performance")
            elif avg_sequential < avg_parallel * 0.8:
                recommendations.append("Consider sequential execution for reliability")
        
        # Analyze failure patterns
        failed_executions = [r for r in self.execution_history if r["status"] == WorkflowStatus.FAILED.value]
        if len(failed_executions) > len(self.execution_history) * 0.2:
            recommendations.append("High failure rate detected - review test stability")
        
        # Analyze coverage impact
        coverage_executions = [r for r in self.execution_history if r.get("coverage_enabled")]
        no_coverage_executions = [r for r in self.execution_history if not r.get("coverage_enabled")]
        
        if coverage_executions and no_coverage_executions:
            avg_coverage = sum(r["duration"] for r in coverage_executions) / len(coverage_executions)
            avg_no_coverage = sum(r["duration"] for r in no_coverage_executions) / len(no_coverage_executions)
            
            overhead = ((avg_coverage - avg_no_coverage) / avg_no_coverage) * 100
            if overhead > 50:
                recommendations.append(f"Coverage adds {overhead:.1f}% overhead - consider selective coverage")
        
        return {
            "recommendations": recommendations,
            "metrics": self.performance_metrics,
            "execution_count": len(self.execution_history)
        }
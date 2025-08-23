"""
Comprehensive Orchestrator Agent SDK - Agent 15
Multi-agent workflow orchestration and coordination system
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# SDK and Framework imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

from app.a2a.core.ai_intelligence import (
    AIIntelligenceFramework, AIIntelligenceConfig,
    create_ai_intelligence_framework, create_enhanced_agent_config
)

from app.a2a.sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from app.a2a.sdk.mcpSkillCoordination import (
    skill_depends_on, skill_provides, coordination_rule
)

from app.a2a.sdk.mixins import (
    PerformanceMonitorMixin, SecurityHardenedMixin,
    TelemetryMixin
)

from app.a2a.core.workflowContext import workflowContextManager
from app.a2a.core.circuitBreaker import EnhancedCircuitBreaker
from app.a2a.core.trustManager import sign_a2a_message, verify_a2a_message

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class OrchestrationStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    DAG = "dag"  # Directed Acyclic Graph
    PIPELINE = "pipeline"

@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    id: str
    name: str
    agent_id: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    strategy: OrchestrationStrategy
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_minutes: int = 60

class OrchestratorAgentSdk(
    A2AAgentBase,
    PerformanceMonitorMixin,
    SecurityHardenedMixin,
    TelemetryMixin
):
    """
    Comprehensive Orchestrator Agent for multi-agent workflow coordination
    """
    
    def __init__(self):
        super().__init__(
            agent_id=create_agent_id("orchestrator-agent"),
            name="Orchestrator Agent",
            description="Multi-agent workflow orchestration and coordination system",
            version="1.0.0"
        )
        
        # Initialize AI Intelligence Framework
        self.ai_framework = create_ai_intelligence_framework(
            create_enhanced_agent_config("orchestrator")
        )
        
        # Workflow management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Agent registry
        self.available_agents: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers for agent communication
        self.agent_circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        
        logger.info("OrchestratorAgent initialized")

    @a2a_skill(
        name="workflow_creation",
        description="Create and define new workflows",
        version="1.0.0"
    )
    @mcp_tool(
        name="create_workflow",
        description="Create a new workflow definition with tasks and dependencies"
    )
    async def create_workflow(
        self,
        workflow_name: str,
        description: str,
        tasks: List[Dict[str, Any]],
        strategy: str = "sequential",
        timeout_minutes: int = 60,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new workflow definition
        """
        try:
            workflow_id = str(uuid.uuid4())
            
            # Convert task dictionaries to WorkflowTask objects
            workflow_tasks = []
            for task_data in tasks:
                task = WorkflowTask(
                    id=task_data.get("id", str(uuid.uuid4())),
                    name=task_data["name"],
                    agent_id=task_data["agent_id"],
                    action=task_data["action"],
                    parameters=task_data.get("parameters", {}),
                    dependencies=task_data.get("dependencies", []),
                    timeout_seconds=task_data.get("timeout_seconds", 300),
                    max_retries=task_data.get("max_retries", 3)
                )
                workflow_tasks.append(task)
            
            # Create workflow definition
            workflow = WorkflowDefinition(
                id=workflow_id,
                name=workflow_name,
                description=description,
                tasks=workflow_tasks,
                strategy=OrchestrationStrategy(strategy),
                timeout_minutes=timeout_minutes,
                metadata=metadata or {}
            )
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow)
            if not validation_result["valid"]:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")
            
            # Store workflow
            self.workflows[workflow_id] = workflow
            
            logger.info(f"Created workflow: {workflow_name} ({workflow_id})")
            
            return {
                "workflow_id": workflow_id,
                "status": "created",
                "task_count": len(workflow_tasks),
                "validation": validation_result
            }
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise

    @a2a_skill(
        name="workflow_execution",
        description="Execute workflows with various orchestration strategies",
        version="1.0.0"
    )
    @mcp_tool(
        name="execute_workflow",
        description="Execute a workflow with the specified strategy"
    )
    async def execute_workflow(
        self,
        workflow_id: str,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            if workflow.status == WorkflowStatus.RUNNING:
                raise ValueError(f"Workflow {workflow_id} is already running")
            
            # Update workflow status
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Start execution task
            execution_task = asyncio.create_task(
                self._execute_workflow_async(workflow, execution_context or {})
            )
            
            self.active_executions[workflow_id] = execution_task
            
            logger.info(f"Started workflow execution: {workflow_id}")
            
            return {
                "workflow_id": workflow_id,
                "status": "started",
                "started_at": workflow.started_at.isoformat(),
                "strategy": workflow.strategy.value
            }
            
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise

    @a2a_skill(
        name="workflow_monitoring",
        description="Monitor workflow execution status and progress",
        version="1.0.0"
    )
    @mcp_tool(
        name="get_workflow_status",
        description="Get current status of a workflow execution"
    )
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get workflow execution status
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            # Calculate progress
            total_tasks = len(workflow.tasks)
            completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.FAILED)
            running_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.RUNNING)
            
            progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Get execution duration
            execution_duration = None
            if workflow.started_at:
                end_time = workflow.completed_at or datetime.now()
                execution_duration = (end_time - workflow.started_at).total_seconds()
            
            return {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
                "progress": {
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "running_tasks": running_tasks,
                    "total_tasks": total_tasks,
                    "percentage": progress_percentage
                },
                "timing": {
                    "created_at": workflow.created_at.isoformat(),
                    "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                    "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                    "duration_seconds": execution_duration
                },
                "tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "agent_id": task.agent_id,
                        "status": task.status.value,
                        "error": task.error
                    }
                    for task in workflow.tasks
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow status {workflow_id}: {e}")
            raise

    @a2a_skill(
        name="agent_coordination",
        description="Coordinate communication between multiple agents",
        version="1.0.0"
    )
    @mcp_tool(
        name="coordinate_agents",
        description="Coordinate multi-agent collaboration for complex tasks"
    )
    async def coordinate_agents(
        self,
        coordination_plan: Dict[str, Any],
        agents: List[str],
        objective: str
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for complex objective
        """
        try:
            coordination_id = str(uuid.uuid4())
            
            # Create coordination session
            coordination_session = {
                "id": coordination_id,
                "objective": objective,
                "agents": agents,
                "plan": coordination_plan,
                "status": "active",
                "created_at": datetime.now(),
                "messages": [],
                "results": {}
            }
            
            # Execute coordination plan
            results = await self._execute_coordination_plan(
                coordination_session, coordination_plan
            )
            
            logger.info(f"Completed agent coordination: {coordination_id}")
            
            return {
                "coordination_id": coordination_id,
                "status": "completed",
                "participating_agents": agents,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to coordinate agents: {e}")
            raise

    @a2a_skill(
        name="workflow_templates",
        description="Manage reusable workflow templates",
        version="1.0.0"
    )
    @mcp_tool(
        name="create_workflow_template",
        description="Create a reusable workflow template"
    )
    async def create_workflow_template(
        self,
        template_name: str,
        description: str,
        template_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a reusable workflow template
        """
        try:
            template_id = str(uuid.uuid4())
            
            template = {
                "id": template_id,
                "name": template_name,
                "description": description,
                "definition": template_definition,
                "created_at": datetime.now(),
                "usage_count": 0
            }
            
            # Store template (in production, use persistent storage)
            # For now, store in memory
            if not hasattr(self, 'workflow_templates'):
                self.workflow_templates = {}
            
            self.workflow_templates[template_id] = template
            
            logger.info(f"Created workflow template: {template_name} ({template_id})")
            
            return {
                "template_id": template_id,
                "status": "created",
                "name": template_name
            }
            
        except Exception as e:
            logger.error(f"Failed to create workflow template: {e}")
            raise

    async def _validate_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Validate workflow definition
        """
        errors = []
        warnings = []
        
        try:
            # Check for circular dependencies
            if self._has_circular_dependencies(workflow.tasks):
                errors.append("Workflow contains circular dependencies")
            
            # Validate agent availability
            for task in workflow.tasks:
                if task.agent_id not in self.available_agents:
                    warnings.append(f"Agent {task.agent_id} not in registry")
            
            # Validate task dependencies
            task_ids = {task.id for task in workflow.tasks}
            for task in workflow.tasks:
                for dep in task.dependencies:
                    if dep not in task_ids:
                        errors.append(f"Task {task.id} depends on non-existent task {dep}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": warnings
            }

    def _has_circular_dependencies(self, tasks: List[WorkflowTask]) -> bool:
        """
        Check for circular dependencies using DFS
        """
        # Build dependency graph
        graph = {task.id: task.dependencies for task in tasks}
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_id in graph:
            if has_cycle(task_id):
                return True
                
        return False

    async def _execute_workflow_async(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> None:
        """
        Asynchronous workflow execution
        """
        try:
            # Execute based on strategy
            if workflow.strategy == OrchestrationStrategy.SEQUENTIAL:
                await self._execute_sequential(workflow, context)
            elif workflow.strategy == OrchestrationStrategy.PARALLEL:
                await self._execute_parallel(workflow, context)
            elif workflow.strategy == OrchestrationStrategy.DAG:
                await self._execute_dag(workflow, context)
            elif workflow.strategy == OrchestrationStrategy.PIPELINE:
                await self._execute_pipeline(workflow, context)
            else:
                raise ValueError(f"Unsupported strategy: {workflow.strategy}")
            
            # Update workflow status
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            # Record execution history
            self._record_execution_history(workflow)
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            logger.error(f"Workflow {workflow.id} failed: {e}")
        finally:
            # Clean up active execution
            if workflow.id in self.active_executions:
                del self.active_executions[workflow.id]

    async def _execute_sequential(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute tasks sequentially
        """
        for task in workflow.tasks:
            await self._execute_task(task, context)
            if task.status == TaskStatus.FAILED:
                raise Exception(f"Task {task.name} failed: {task.error}")

    async def _execute_parallel(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute tasks in parallel
        """
        tasks = [self._execute_task(task, context) for task in workflow.tasks]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_dag(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute tasks based on DAG dependencies
        """
        # Topological sort
        sorted_tasks = self._topological_sort(workflow.tasks)
        
        # Execute in dependency order
        for task in sorted_tasks:
            # Wait for dependencies
            for dep_id in task.dependencies:
                dep_task = next((t for t in workflow.tasks if t.id == dep_id), None)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    raise Exception(f"Dependency {dep_id} not completed for task {task.name}")
            
            await self._execute_task(task, context)

    async def _execute_pipeline(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute tasks as a pipeline, passing results between stages
        """
        pipeline_context = context.copy()
        
        for task in workflow.tasks:
            # Execute task with accumulated context
            await self._execute_task(task, pipeline_context)
            
            # Add task result to context for next task
            if task.result:
                pipeline_context[f"task_{task.id}_result"] = task.result

    async def _execute_task(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute individual task
        """
        try:
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.now()
            
            # Prepare message for target agent
            message = A2AMessage(
                sender_id=self.agent_id,
                recipient_id=task.agent_id,
                content={
                    "action": task.action,
                    "parameters": task.parameters,
                    "context": context
                },
                message_type="task_execution",
                role=MessageRole.USER
            )
            
            # Execute with timeout and retries
            result = await self._execute_with_retry(task, message)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()
            logger.error(f"Task {task.name} failed: {e}")

    async def _execute_with_retry(
        self,
        task: WorkflowTask,
        message: A2AMessage
    ) -> Dict[str, Any]:
        """
        Execute task with retry logic
        """
        for attempt in range(task.max_retries + 1):
            try:
                # Use circuit breaker for agent communication
                circuit_breaker = self._get_agent_circuit_breaker(task.agent_id)
                
                async def execute_call():
                    return await self.send_message_and_wait_for_response(
                        message,
                        timeout=task.timeout_seconds
                    )
                
                result = await circuit_breaker.call(execute_call)
                return result
                
            except Exception as e:
                task.retry_count = attempt
                if attempt < task.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise e

    def _get_agent_circuit_breaker(self, agent_id: str) -> EnhancedCircuitBreaker:
        """
        Get or create circuit breaker for agent
        """
        if agent_id not in self.agent_circuit_breakers:
            self.agent_circuit_breakers[agent_id] = EnhancedCircuitBreaker(
                name=f"agent_{agent_id}",
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=Exception
            )
        return self.agent_circuit_breakers[agent_id]

    def _topological_sort(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """
        Topological sort for DAG execution
        """
        # Build graph
        graph = {task.id: task for task in tasks}
        in_degree = {task.id: 0 for task in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                in_degree[task.id] += 1
        
        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            task_id = queue.pop(0)
            result.append(graph[task_id])
            
            # Update in_degree for dependent tasks
            for task in tasks:
                if task_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return result

    async def _execute_coordination_plan(
        self,
        session: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agent coordination plan
        """
        # Simplified coordination execution
        # In production, implement sophisticated coordination protocols
        results = {}
        
        for step in plan.get("steps", []):
            step_results = await self._execute_coordination_step(session, step)
            results[step["id"]] = step_results
        
        return results

    async def _execute_coordination_step(
        self,
        session: Dict[str, Any],
        step: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute individual coordination step
        """
        # Placeholder for coordination step execution
        return {"status": "completed", "step_id": step["id"]}

    def _record_execution_history(self, workflow: WorkflowDefinition) -> None:
        """
        Record workflow execution in history
        """
        history_entry = {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "task_count": len(workflow.tasks),
            "success_rate": sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED) / len(workflow.tasks)
        }
        
        self.execution_history.append(history_entry)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

# Create singleton instance
orchestrator_agent = OrchestratorAgentSdk()

def get_orchestrator_agent() -> OrchestratorAgentSdk:
    """Get the singleton orchestrator agent instance"""
    return orchestrator_agent
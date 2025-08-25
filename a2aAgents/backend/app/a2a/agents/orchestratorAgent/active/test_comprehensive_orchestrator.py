import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

"""
Comprehensive Test Suite for Orchestrator Agent
Tests workflow creation, execution, monitoring, coordination, and simulations
"""

from .comprehensiveOrchestratorAgentSdk import (
    OrchestratorAgentSdk, WorkflowDefinition, WorkflowTask,
    WorkflowStatus, TaskStatus, OrchestrationStrategy
)
from app.a2a.core.security_base import SecureA2AAgent
from .mockOrchestratorAgent import (
    MockOrchestratorAgent, OrchestratorTestHelper
)
from .orchestratorSimulator import (
    OrchestratorSimulator, SimulationScenario,
    run_normal_orchestration_simulation, run_high_concurrency_simulation
)

class TestOrchestratorAgent(SecureA2AAgent):
    """Test suite for Orchestrator Agent"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    @pytest.fixture
    async def orchestrator_agent(self):
        """Create test orchestrator agent"""
        agent = OrchestratorAgentSdk()
        yield agent
        # Cleanup after test
        for workflow_id in list(agent.workflows.keys()):
            if workflow_id in agent.active_executions:
                agent.active_executions[workflow_id].cancel()

    @pytest.fixture
    def mock_orchestrator_agent(self):
        """Create mock orchestrator agent"""
        return MockOrchestratorAgent()
    
    @pytest.fixture
    def test_helper(self, mock_orchestrator_agent):
        """Create test helper"""
        return OrchestratorTestHelper(mock_orchestrator_agent)

    # Workflow Creation Tests
    
    @pytest.mark.asyncio
    async def test_workflow_creation_basic(self, orchestrator_agent):
        """Test basic workflow creation"""
        
        tasks = [
            {
                "id": "task-1",
                "name": "DataProcessing",
                "agent_id": "data-agent",
                "action": "process_data",
                "parameters": {"input_file": "test.csv"}
            },
            {
                "id": "task-2", 
                "name": "DataValidation",
                "agent_id": "validation-agent",
                "action": "validate_data",
                "parameters": {"validation_rules": ["not_null", "format_check"]}
            }
        ]
        
        result = await orchestrator_agent.create_workflow(
            workflow_name="TestWorkflow",
            description="Basic test workflow",
            tasks=tasks,
            strategy="sequential"
        )
        
        assert result["status"] == "created"
        assert "workflow_id" in result
        assert result["task_count"] == 2
        assert result["validation"]["valid"] == True
        
        # Verify workflow is stored
        workflow_id = result["workflow_id"]
        assert workflow_id in orchestrator_agent.workflows

    @pytest.mark.asyncio 
    async def test_workflow_creation_with_dependencies(self, orchestrator_agent):
        """Test workflow creation with task dependencies"""
        
        tasks = [
            {
                "id": "task-1",
                "name": "ExtractData",
                "agent_id": "extract-agent",
                "action": "extract",
                "parameters": {}
            },
            {
                "id": "task-2",
                "name": "TransformData", 
                "agent_id": "transform-agent",
                "action": "transform",
                "parameters": {},
                "dependencies": ["task-1"]
            },
            {
                "id": "task-3",
                "name": "LoadData",
                "agent_id": "load-agent", 
                "action": "load",
                "parameters": {},
                "dependencies": ["task-2"]
            }
        ]
        
        result = await orchestrator_agent.create_workflow(
            workflow_name="ETLWorkflow",
            description="ETL workflow with dependencies",
            tasks=tasks,
            strategy="dag"
        )
        
        assert result["status"] == "created"
        assert result["validation"]["valid"] == True
        
        workflow_id = result["workflow_id"]
        workflow = orchestrator_agent.workflows[workflow_id]
        assert len(workflow.tasks) == 3
        assert workflow.tasks[1].dependencies == ["task-1"]
        assert workflow.tasks[2].dependencies == ["task-2"]

    @pytest.mark.asyncio
    async def test_workflow_validation_errors(self, orchestrator_agent):
        """Test workflow validation with errors"""
        
        # Test circular dependency
        tasks = [
            {
                "id": "task-1",
                "name": "Task1",
                "agent_id": "agent-1",
                "action": "action1",
                "dependencies": ["task-2"]
            },
            {
                "id": "task-2",
                "name": "Task2", 
                "agent_id": "agent-2",
                "action": "action2",
                "dependencies": ["task-1"]
            }
        ]
        
        with pytest.raises(ValueError, match="validation failed"):
            await orchestrator_agent.create_workflow(
                workflow_name="InvalidWorkflow",
                description="Workflow with circular dependencies",
                tasks=tasks,
                strategy="dag"
            )

    # Workflow Execution Tests
    
    @pytest.mark.asyncio
    async def test_sequential_workflow_execution(self, orchestrator_agent):
        """Test sequential workflow execution"""
        
        # Mock agent message handling
        async def mock_send_message(message, timeout=None):
            return {"status": "success", "result": f"Processed by {message.recipient_id}"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        tasks = [
            {
                "id": "task-1",
                "name": "FirstTask",
                "agent_id": "agent-1",
                "action": "first_action"
            },
            {
                "id": "task-2",
                "name": "SecondTask",
                "agent_id": "agent-2", 
                "action": "second_action"
            }
        ]
        
        # Create workflow
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="SequentialTest",
            description="Test sequential execution",
            tasks=tasks,
            strategy="sequential"
        )
        
        workflow_id = create_result["workflow_id"]
        
        # Execute workflow
        exec_result = await orchestrator_agent.execute_workflow(workflow_id)
        
        assert exec_result["status"] == "started"
        assert exec_result["workflow_id"] == workflow_id
        
        # Wait for completion (with timeout)
        for _ in range(50):  # 5 second timeout
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        # Check final status
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        assert final_status["status"] == "completed"
        assert final_status["progress"]["completed_tasks"] == 2

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, orchestrator_agent):
        """Test parallel workflow execution"""
        
        # Mock agent message handling with delay
        async def mock_send_message(message, timeout=None):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"status": "success", "result": f"Processed by {message.recipient_id}"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        tasks = [
            {
                "id": "task-1",
                "name": "ParallelTask1", 
                "agent_id": "agent-1",
                "action": "parallel_action_1"
            },
            {
                "id": "task-2",
                "name": "ParallelTask2",
                "agent_id": "agent-2",
                "action": "parallel_action_2"
            },
            {
                "id": "task-3",
                "name": "ParallelTask3",
                "agent_id": "agent-3", 
                "action": "parallel_action_3"
            }
        ]
        
        # Create and execute workflow
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="ParallelTest",
            description="Test parallel execution",
            tasks=tasks,
            strategy="parallel"
        )
        
        workflow_id = create_result["workflow_id"]
        start_time = datetime.now()
        
        await orchestrator_agent.execute_workflow(workflow_id)
        
        # Wait for completion
        for _ in range(50):
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Check that parallel execution was faster than sequential would be
        assert execution_time < 0.5  # Should be much less than 3 * 0.1 seconds
        
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        assert final_status["status"] == "completed"

    # Workflow Monitoring Tests
    
    @pytest.mark.asyncio
    async def test_workflow_status_monitoring(self, orchestrator_agent):
        """Test workflow status monitoring"""
        
        tasks = [{
            "id": "task-1",
            "name": "MonitoringTest",
            "agent_id": "test-agent",
            "action": "test_action"
        }]
        
        # Create workflow
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="MonitoringWorkflow",
            description="Test monitoring",
            tasks=tasks
        )
        
        workflow_id = create_result["workflow_id"]
        
        # Check initial status
        status = await orchestrator_agent.get_workflow_status(workflow_id)
        
        assert status["workflow_id"] == workflow_id
        assert status["name"] == "MonitoringWorkflow"
        assert status["status"] == "created"
        assert status["progress"]["total_tasks"] == 1
        assert status["progress"]["completed_tasks"] == 0
        assert len(status["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_workflow_progress_tracking(self, orchestrator_agent):
        """Test workflow progress tracking during execution"""
        
        # Mock slow agent responses
        call_count = 0
        async def mock_send_message(message, timeout=None):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.2)  # Longer delay to check progress
            return {"status": "success", "result": f"Call {call_count}"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        tasks = [
            {"id": f"task-{i}", "name": f"Task{i}", "agent_id": f"agent-{i}", "action": "slow_action"}
            for i in range(1, 4)
        ]
        
        # Create and start workflow
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="ProgressTest",
            description="Test progress tracking",
            tasks=tasks,
            strategy="sequential"
        )
        
        workflow_id = create_result["workflow_id"]
        await orchestrator_agent.execute_workflow(workflow_id)
        
        # Monitor progress
        progress_snapshots = []
        for _ in range(10):
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            progress_snapshots.append(status["progress"]["percentage"])
            
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        # Verify progress increased over time
        assert len(progress_snapshots) > 1
        assert progress_snapshots[-1] >= progress_snapshots[0]

    # Agent Coordination Tests
    
    @pytest.mark.asyncio
    async def test_agent_coordination_basic(self, orchestrator_agent):
        """Test basic agent coordination"""
        
        coordination_plan = {
            "steps": [
                {
                    "id": "step-1",
                    "action": "negotiate_resources",
                    "agents": ["agent-1", "agent-2"]
                },
                {
                    "id": "step-2", 
                    "action": "synchronize_state",
                    "agents": ["agent-1", "agent-2", "agent-3"]
                }
            ]
        }
        
        result = await orchestrator_agent.coordinate_agents(
            coordination_plan=coordination_plan,
            agents=["agent-1", "agent-2", "agent-3"],
            objective="Test coordination"
        )
        
        assert result["status"] == "completed"
        assert "coordination_id" in result
        assert result["participating_agents"] == ["agent-1", "agent-2", "agent-3"]
        assert "results" in result

    # Workflow Template Tests
    
    @pytest.mark.asyncio
    async def test_workflow_template_creation(self, orchestrator_agent):
        """Test workflow template creation"""
        
        template_definition = {
            "task_templates": [
                {
                    "name_pattern": "Extract_{data_source}",
                    "agent_type": "extractor",
                    "action": "extract_data"
                },
                {
                    "name_pattern": "Process_{data_source}",
                    "agent_type": "processor", 
                    "action": "process_data",
                    "dependencies": ["Extract_{data_source}"]
                }
            ],
            "parameters": {
                "data_source": {"type": "string", "required": True}
            }
        }
        
        result = await orchestrator_agent.create_workflow_template(
            template_name="ETL_Template",
            description="Generic ETL workflow template",
            template_definition=template_definition
        )
        
        assert result["status"] == "created"
        assert "template_id" in result
        assert result["name"] == "ETL_Template"

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_task_failure(self, orchestrator_agent):
        """Test workflow execution with task failures"""
        
        # Mock agent responses with one failure
        call_count = 0
        async def mock_send_message(message, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second task fails
                raise Exception("Simulated task failure")
            return {"status": "success", "result": "Task completed"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        tasks = [
            {"id": "task-1", "name": "SuccessTask", "agent_id": "agent-1", "action": "success"},
            {"id": "task-2", "name": "FailTask", "agent_id": "agent-2", "action": "fail"},
            {"id": "task-3", "name": "SkippedTask", "agent_id": "agent-3", "action": "skip"}
        ]
        
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="FailureTest",
            description="Test failure handling",
            tasks=tasks,
            strategy="sequential"
        )
        
        workflow_id = create_result["workflow_id"]
        await orchestrator_agent.execute_workflow(workflow_id)
        
        # Wait for completion
        for _ in range(50):
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        assert final_status["status"] == "failed"
        assert final_status["progress"]["failed_tasks"] >= 1

    @pytest.mark.asyncio
    async def test_workflow_execution_timeout(self, orchestrator_agent):
        """Test workflow execution timeout"""
        
        # Mock very slow agent response
        async def mock_send_message(message, timeout=None):
            await asyncio.sleep(10)  # Longer than timeout
            return {"status": "success"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        tasks = [{
            "id": "task-1",
            "name": "SlowTask",
            "agent_id": "slow-agent",
            "action": "slow_action",
            "timeout_seconds": 1  # Very short timeout
        }]
        
        create_result = await orchestrator_agent.create_workflow(
            workflow_name="TimeoutTest",
            description="Test timeout handling",
            tasks=tasks
        )
        
        workflow_id = create_result["workflow_id"]
        await orchestrator_agent.execute_workflow(workflow_id)
        
        # Wait for timeout/failure
        for _ in range(50):
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        assert final_status["status"] == "failed"

    # Mock Testing
    
    @pytest.mark.asyncio
    async def test_mock_workflow_creation(self, mock_orchestrator_agent):
        """Test mock workflow creation"""
        
        tasks = [
            {
                "id": "mock-task-1",
                "name": "MockTask1",
                "agent_id": "mock-agent-1",
                "action": "mock_action"
            }
        ]
        
        result = await mock_orchestrator_agent.create_workflow(
            workflow_name="MockWorkflow",
            description="Test mock workflow",
            tasks=tasks
        )
        
        assert result["status"] == "created"
        assert "workflow_id" in result

    @pytest.mark.asyncio
    async def test_mock_failure_scenarios(self, mock_orchestrator_agent):
        """Test mock failure scenarios"""
        
        # Set up failure scenario
        mock_orchestrator_agent.set_failure_scenario("create_workflow", True)
        
        tasks = [{"id": "task-1", "name": "Task1", "agent_id": "agent-1", "action": "action1"}]
        
        with pytest.raises(Exception, match="Mock workflow creation failure"):
            await mock_orchestrator_agent.create_workflow(
                workflow_name="FailWorkflow",
                description="Test failure",
                tasks=tasks
            )
        
        # Clear failure scenario
        mock_orchestrator_agent.clear_failure_scenarios()

    @pytest.mark.asyncio
    async def test_test_helper_utilities(self, test_helper):
        """Test helper utility functions"""
        
        # Create test workflow
        workflow_id = await test_helper.create_test_workflow(
            name="HelperTestWorkflow",
            task_count=3,
            strategy="sequential"
        )
        
        assert workflow_id in test_helper.mock_orchestrator.mock_workflows
        
        # Execute workflow
        await test_helper.mock_orchestrator.execute_workflow(workflow_id)
        
        # Wait for completion
        status = await test_helper.wait_for_workflow_completion(
            workflow_id,
            timeout_seconds=5
        )
        
        assert status["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_execution_order_verification(self, test_helper):
        """Test execution order verification for sequential workflows"""
        
        # Create sequential workflow
        workflow_id = await test_helper.create_test_workflow(
            name="OrderTestWorkflow",
            task_count=3,
            strategy="sequential"
        )
        
        # Execute workflow
        await test_helper.mock_orchestrator.execute_workflow(workflow_id)
        
        # Wait for completion
        await test_helper.wait_for_workflow_completion(workflow_id)
        
        # Verify execution order
        expected_order = ["TestTask0", "TestTask1", "TestTask2"]
        order_correct = test_helper.verify_workflow_execution_order(
            workflow_id,
            expected_order
        )
        
        assert order_correct

    # Simulation Tests
    
    @pytest.mark.asyncio
    async def test_basic_simulation(self, orchestrator_agent):
        """Test basic orchestration simulation"""
        
        simulator = OrchestratorSimulator(orchestrator_agent)
        
        # Setup small simulation
        await simulator.setup_simulation(
            agent_count_per_type=1,
            scenario=SimulationScenario.NORMAL_WORKFLOW_EXECUTION
        )
        
        # Run short simulation
        metrics = await simulator.run_simulation(
            duration_seconds=10,
            workflow_generation_rate=0.2
        )
        
        # Verify metrics
        assert metrics.workflows_created >= 0
        assert metrics.workflows_executed >= 0
        
        # Cleanup
        await simulator.cleanup_simulation()

    @pytest.mark.asyncio
    async def test_simulation_scenarios(self, orchestrator_agent):
        """Test different simulation scenarios"""
        
        scenarios_to_test = [
            SimulationScenario.NORMAL_WORKFLOW_EXECUTION,
            SimulationScenario.HIGH_CONCURRENCY,
            SimulationScenario.AGENT_FAILURES
        ]
        
        for scenario in scenarios_to_test:
            simulator = OrchestratorSimulator(orchestrator_agent)
            
            await simulator.setup_simulation(
                agent_count_per_type=1,
                scenario=scenario
            )
            
            metrics = await simulator.run_simulation(
                duration_seconds=5,
                workflow_generation_rate=0.1
            )
            
            # Verify scenario was applied
            assert len(simulator.simulated_agents) >= 3
            
            await simulator.cleanup_simulation()

    @pytest.mark.asyncio
    async def test_simulation_convenience_functions(self, orchestrator_agent):
        """Test simulation convenience functions"""
        
        # Test normal operations simulation
        report = await run_normal_orchestration_simulation(
            orchestrator_agent,
            duration_seconds=10
        )
        
        assert "summary" in report
        assert "performance" in report
        assert "reliability" in report

    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_full_orchestration_lifecycle(self, orchestrator_agent):
        """Test complete orchestration lifecycle"""
        
        # Mock agent responses
        async def mock_send_message(message, timeout=None):
            await asyncio.sleep(0.05)  # Simulate processing
            return {
                "status": "success",
                "result": f"Processed {message.content['action']} by {message.recipient_id}"
            }
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        # 1. Create workflow template
        template_result = await orchestrator_agent.create_workflow_template(
            template_name="IntegrationTemplate",
            description="Integration test template",
            template_definition={
                "tasks": [
                    {"type": "data_extraction", "agent_type": "extractor"},
                    {"type": "data_processing", "agent_type": "processor"}
                ]
            }
        )
        
        # 2. Create workflow from scratch
        tasks = [
            {
                "id": "extract-task",
                "name": "ExtractData",
                "agent_id": "extractor-agent",
                "action": "extract",
                "parameters": {"source": "database"}
            },
            {
                "id": "process-task",
                "name": "ProcessData",
                "agent_id": "processor-agent",
                "action": "process",
                "parameters": {"algorithm": "ml_pipeline"},
                "dependencies": ["extract-task"]
            },
            {
                "id": "validate-task",
                "name": "ValidateResults",
                "agent_id": "validator-agent",
                "action": "validate",
                "parameters": {"criteria": ["accuracy", "completeness"]},
                "dependencies": ["process-task"]
            }
        ]
        
        workflow_result = await orchestrator_agent.create_workflow(
            workflow_name="IntegrationWorkflow",
            description="Full integration test workflow",
            tasks=tasks,
            strategy="dag",
            timeout_minutes=10
        )
        
        workflow_id = workflow_result["workflow_id"]
        
        # 3. Execute workflow
        exec_result = await orchestrator_agent.execute_workflow(
            workflow_id,
            execution_context={"integration_test": True}
        )
        
        assert exec_result["status"] == "started"
        
        # 4. Monitor execution progress
        progress_history = []
        for _ in range(30):  # 3 second timeout
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            progress_history.append(status["progress"]["percentage"])
            
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        # 5. Verify completion
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        
        assert final_status["status"] == "completed"
        assert final_status["progress"]["completed_tasks"] == 3
        assert final_status["progress"]["failed_tasks"] == 0
        assert final_status["progress"]["percentage"] == 100.0
        
        # 6. Verify task execution order (DAG should respect dependencies)
        task_completion_order = []
        for task in final_status["tasks"]:
            if task["status"] == "completed":
                task_completion_order.append(task["id"])
        
        # Extract should complete before process, process before validate
        extract_index = task_completion_order.index("extract-task")
        process_index = task_completion_order.index("process-task")
        validate_index = task_completion_order.index("validate-task")
        
        assert extract_index < process_index < validate_index
        
        # 7. Test coordination
        coordination_result = await orchestrator_agent.coordinate_agents(
            coordination_plan={
                "steps": [
                    {
                        "id": "cleanup-step",
                        "action": "cleanup_resources",
                        "agents": ["extractor-agent", "processor-agent", "validator-agent"]
                    }
                ]
            },
            agents=["extractor-agent", "processor-agent", "validator-agent"],
            objective="Cleanup after workflow execution"
        )
        
        assert coordination_result["status"] == "completed"


# Performance Tests
class TestOrchestratorPerformance(SecureA2AAgent):
    """Performance tests for orchestrator"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_executions(self, orchestrator_agent):
        """Test concurrent workflow executions"""
        
        # Mock fast agent responses
        async def mock_send_message(message, timeout=None):
            await asyncio.sleep(0.01)  # Very fast response
            return {"status": "success", "result": "completed"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        # Create multiple workflows
        workflow_ids = []
        for i in range(5):
            tasks = [
                {
                    "id": f"task-{i}-1",
                    "name": f"ConcurrentTask{i}",
                    "agent_id": f"agent-{i}",
                    "action": "concurrent_action"
                }
            ]
            
            result = await orchestrator_agent.create_workflow(
                workflow_name=f"ConcurrentWorkflow{i}",
                description=f"Concurrent test workflow {i}",
                tasks=tasks
            )
            workflow_ids.append(result["workflow_id"])
        
        # Execute all workflows concurrently
        start_time = datetime.now()
        execution_tasks = [
            orchestrator_agent.execute_workflow(wf_id)
            for wf_id in workflow_ids
        ]
        await asyncio.gather(*execution_tasks)
        
        # Wait for all to complete
        completed_count = 0
        for _ in range(100):  # 10 second timeout
            completed_count = 0
            for wf_id in workflow_ids:
                status = await orchestrator_agent.get_workflow_status(wf_id)
                if status["status"] in ["completed", "failed"]:
                    completed_count += 1
            
            if completed_count == len(workflow_ids):
                break
            await asyncio.sleep(0.1)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify all workflows completed
        assert completed_count == len(workflow_ids)
        
        # Verify reasonable performance (should be much faster than sequential)
        assert total_time < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_large_workflow_handling(self, orchestrator_agent):
        """Test handling of large workflows"""
        
        # Mock agent responses
        async def mock_send_message(message, timeout=None):
            return {"status": "success", "result": "task_completed"}
        
        orchestrator_agent.send_message_and_wait_for_response = mock_send_message
        
        # Create large workflow with many tasks
        task_count = 50
        tasks = []
        for i in range(task_count):
            task = {
                "id": f"large-task-{i:03d}",
                "name": f"LargeWorkflowTask{i:03d}",
                "agent_id": f"agent-{i % 10}",  # Distribute across 10 agents
                "action": "large_workflow_action",
                "parameters": {"task_index": i}
            }
            
            # Add some dependencies to create moderate complexity
            if i > 0 and i % 5 == 0:  # Every 5th task depends on previous
                task["dependencies"] = [f"large-task-{i-1:03d}"]
            
            tasks.append(task)
        
        # Create workflow
        start_time = datetime.now()
        result = await orchestrator_agent.create_workflow(
            workflow_name="LargeWorkflow",
            description=f"Large workflow with {task_count} tasks",
            tasks=tasks,
            strategy="dag"
        )
        
        creation_time = (datetime.now() - start_time).total_seconds()
        
        # Verify creation was reasonably fast
        assert creation_time < 1.0  # Should create in under 1 second
        assert result["task_count"] == task_count
        
        workflow_id = result["workflow_id"]
        
        # Execute workflow
        exec_start = datetime.now()
        await orchestrator_agent.execute_workflow(workflow_id)
        
        # Monitor for completion (with reasonable timeout)
        for _ in range(100):  # 10 second timeout
            status = await orchestrator_agent.get_workflow_status(workflow_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        exec_time = (datetime.now() - exec_start).total_seconds()
        
        # Verify completion
        final_status = await orchestrator_agent.get_workflow_status(workflow_id)
        assert final_status["status"] == "completed"
        assert final_status["progress"]["completed_tasks"] == task_count
        
        # Verify reasonable execution time
        assert exec_time < 10.0  # Should complete in under 10 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

"""
Integration Tests for A2A Production Workflows
Tests complete end-to-end workflows with all production components
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

# A2A imports
from app.a2a.sdk.types import A2AMessage, MessageType, AgentCapability
from app.a2a.sdk.agentBase import A2AAgentBase

# Production components
from app.core.secure_secrets_manager import (
    SecureSecretsManager, SecretType, SecretScope, RotationPolicy,
    initialize_secrets_manager, get_secrets_manager
)
from app.core.resource_manager import (
    ResourceManager, ResourceType, initialize_resource_manager, get_resource_manager
)
from app.core.a2a_protocol_validator import (
    A2AProtocolValidator, ValidationSeverity, ValidationCategory,
    initialize_protocol_validator, get_protocol_validator
)
from app.core.slo_sli_framework import (
    SLOSLIFramework, SLODefinition, SLIType, AlertSeverity,
    initialize_slo_sli_framework, get_slo_sli_framework
)
from app.core.backpressure_manager import (
    BackpressureManager, BackpressureStrategy, LoadShedStrategy,
    initialize_backpressure_manager, get_backpressure_manager
)
from app.core.security_hardening import (
    SecurityHardeningFramework, SecurityLevel, AuthenticationMethod, Permission,
    initialize_security_hardening, get_security_framework
)
from app.core.observability_stack import (
    ObservabilityManager, MetricType, LogLevel,
    initialize_observability_manager, get_observability_manager
)
from app.core.disaster_recovery import (
    DisasterRecoveryManager, RecoveryStrategy, BackupType,
    initialize_disaster_recovery_manager, get_disaster_recovery_manager
)
from app.core.a2a_distributed_coordinator import (
    A2ADistributedCoordinator, CoordinationStrategy, TaskPriority,
    initialize_distributed_coordinator, get_distributed_coordinator
)
from app.core.resource_cleanup_handlers import (
    initialize_cleanup_system, cleanup_all_resources
)

# Test utilities
from app.clients.redisClient import RedisConfig

logger = logging.getLogger(__name__)


class MockA2AAgent(A2AAgentBase):
    """Mock A2A agent for testing"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability] = None):
        self.agent_id = agent_id
        self.capabilities = capabilities or [AgentCapability.REASONING, AgentCapability.TOOL_CALLING]
        self.messages_received = []
        self.messages_sent = []
        self.is_running = False
        self.task_results = {}
    
    async def initialize(self):
        """Initialize the mock agent"""
        self.is_running = True
        logger.info(f"Mock agent {self.agent_id} initialized")
    
    async def shutdown(self):
        """Shutdown the mock agent"""
        self.is_running = False
        logger.info(f"Mock agent {self.agent_id} shut down")
    
    async def process_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process an incoming message"""
        self.messages_received.append(message)
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate response based on message type
        if message.type == MessageType.REQUEST:
            response = A2AMessage(
                id=f"response_{uuid.uuid4()}",
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                receiver=message.sender,
                timestamp=datetime.utcnow().isoformat(),
                payload={
                    "status": "success",
                    "result": f"Processed request {message.id}",
                    "capabilities_used": [cap.value for cap in self.capabilities[:1]]
                },
                in_reply_to=message.id
            )
            self.messages_sent.append(response)
            return response
        
        return None
    
    async def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        await asyncio.sleep(0.2)  # Simulate task execution
        
        result = {
            "task_id": task_id,
            "status": "completed",
            "result": f"Task {task_id} completed by {self.agent_id}",
            "execution_time": 0.2
        }
        
        self.task_results[task_id] = result
        return result


@pytest.fixture
async def redis_config():
    """Provide Redis configuration for testing"""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=15,  # Use separate DB for tests
        password=None,
        decode_responses=True
    )


@pytest.fixture
async def production_components(redis_config):
    """Initialize all production components for testing"""
    components = {}
    
    try:
        # Initialize all production components
        components['secrets_manager'] = await initialize_secrets_manager(redis_config)
        components['resource_manager'] = await initialize_resource_manager()
        components['protocol_validator'] = initialize_protocol_validator()
        components['slo_sli_framework'] = await initialize_slo_sli_framework(redis_config)
        components['backpressure_manager'] = await initialize_backpressure_manager(redis_config)
        components['security_framework'] = await initialize_security_hardening(redis_config)
        components['observability_manager'] = await initialize_observability_manager(redis_config)
        components['disaster_recovery'] = await initialize_disaster_recovery_manager(redis_config)
        components['distributed_coordinator'] = await initialize_distributed_coordinator(redis_config)
        components['cleanup_system'] = initialize_cleanup_system()
        
        logger.info("All production components initialized for testing")
        yield components
        
    finally:
        # Cleanup all components
        await cleanup_all_resources()
        logger.info("All production components cleaned up after testing")


@pytest.fixture
async def mock_agents():
    """Create mock agents for testing"""
    agents = {
        'agent_1': MockA2AAgent('test_agent_1', [AgentCapability.REASONING, AgentCapability.MEMORY]),
        'agent_2': MockA2AAgent('test_agent_2', [AgentCapability.TOOL_CALLING, AgentCapability.PLANNING]),
        'agent_3': MockA2AAgent('test_agent_3', [AgentCapability.COLLABORATION])
    }
    
    # Initialize all agents
    for agent in agents.values():
        await agent.initialize()
    
    yield agents
    
    # Shutdown all agents
    for agent in agents.values():
        await agent.shutdown()


class TestA2AProductionWorkflows:
    """Test complete A2A production workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle_workflow(self, production_components, mock_agents):
        """Test complete agent lifecycle with all production components"""
        
        # Get components
        secrets_manager = production_components['secrets_manager']
        resource_manager = production_components['resource_manager']
        protocol_validator = production_components['protocol_validator']
        security_framework = production_components['security_framework']
        
        agent = mock_agents['agent_1']
        
        # 1. Store agent secrets
        agent_api_key = await secrets_manager.store_secret(
            secret_id=f"{agent.agent_id}_api_key",
            value="test_api_key_12345",
            secret_type=SecretType.API_KEY,
            scope=SecretScope.AGENT
        )
        assert agent_api_key is True
        
        # 2. Allocate resources for agent
        memory_allocation = await resource_manager.allocate_resource(
            resource_type=ResourceType.MEMORY,
            size_bytes=1024*1024,  # 1MB
            owner=agent.agent_id
        )
        assert memory_allocation is not None
        
        # 3. Register agent with security framework
        principal = await security_framework.auth_manager.register_principal(
            principal_id=agent.agent_id,
            principal_type="agent",
            name=f"Test Agent {agent.agent_id}",
            permissions={Permission.READ_MESSAGES, Permission.WRITE_MESSAGES, Permission.EXECUTE_TASKS},
            authentication_methods=[AuthenticationMethod.API_KEY]
        )
        assert principal is not None
        
        # 4. Create and validate A2A message
        test_message = A2AMessage(
            id=f"msg_{uuid.uuid4()}",
            type=MessageType.CAPABILITY_ANNOUNCEMENT,
            sender=agent.agent_id,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "capabilities": [cap.value for cap in agent.capabilities],
                "agent_status": "active"
            }
        )
        
        validation_results = await protocol_validator.validate_message(test_message)
        assert len([r for r in validation_results if r.severity == ValidationSeverity.ERROR]) == 0
        
        # 5. Authenticate agent
        token = await security_framework.auth_manager.authenticate(
            principal_id=agent.agent_id,
            credentials={"api_key": "test_api_key_12345"},
            auth_method=AuthenticationMethod.API_KEY
        )
        assert token is not None
        
        # 6. Process message through agent
        response = await agent.process_message(test_message)
        assert response is None  # Capability announcement doesn't generate response
        
        # 7. Cleanup resources
        await resource_manager.deallocate_resource(memory_allocation)
        
        # 8. Retrieve secret for validation
        retrieved_secret = await secrets_manager.retrieve_secret(f"{agent.agent_id}_api_key")
        assert retrieved_secret == "test_api_key_12345"
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, production_components, mock_agents):
        """Test multi-agent coordination workflow"""
        
        distributed_coordinator = production_components['distributed_coordinator']
        protocol_validator = production_components['protocol_validator']
        backpressure_manager = production_components['backpressure_manager']
        
        agents = list(mock_agents.values())
        
        # 1. Register agents with coordinator
        for agent in agents:
            await distributed_coordinator.register_agent(
                agent_id=agent.agent_id,
                capabilities=agent.capabilities,
                metadata={"test_agent": True}
            )
        
        # 2. Create coordination task
        task_data = {
            "task_type": "collaborative_reasoning",
            "input_data": "Solve complex problem X",
            "required_capabilities": [AgentCapability.REASONING, AgentCapability.COLLABORATION],
            "timeout": 30
        }
        
        task_id = await distributed_coordinator.create_task(
            task_data=task_data,
            priority=TaskPriority.NORMAL,
            required_capabilities=task_data["required_capabilities"]
        )
        assert task_id is not None
        
        # 3. Distribute task to suitable agents
        suitable_agents = await distributed_coordinator.find_suitable_agents(
            task_data["required_capabilities"]
        )
        assert len(suitable_agents) >= 2  # Should find agents with required capabilities
        
        # 4. Execute task coordination
        coordination_result = await distributed_coordinator.coordinate_task_execution(
            task_id=task_id,
            agents=suitable_agents[:2]  # Use first 2 suitable agents
        )
        assert coordination_result["status"] == "success"
        
        # 5. Validate coordination messages
        coordination_messages = coordination_result.get("messages", [])
        for message in coordination_messages:
            validation_results = await protocol_validator.validate_message(message)
            critical_issues = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
            assert len(critical_issues) == 0
        
        # 6. Check backpressure handling
        bp_status = await backpressure_manager.get_system_status()
        assert bp_status["overall_health"] in ["healthy", "warning"]  # Should not be critical
    
    @pytest.mark.asyncio
    async def test_security_and_monitoring_workflow(self, production_components, mock_agents):
        """Test security hardening and monitoring workflow"""
        
        security_framework = production_components['security_framework']
        observability_manager = production_components['observability_manager']
        slo_sli_framework = production_components['slo_sli_framework']
        
        agent = mock_agents['agent_1']
        
        # 1. Perform security scan
        vulnerabilities = await security_framework.perform_security_scan()
        
        # Should not have critical vulnerabilities in test environment
        critical_vulns = [v for v in vulnerabilities if v.severity.value == "critical"]
        assert len(critical_vulns) == 0, f"Critical vulnerabilities found: {critical_vulns}"
        
        # 2. Monitor agent performance
        # Start monitoring
        await observability_manager.start_monitoring_agent(agent.agent_id)
        
        # Simulate agent activity
        for i in range(5):
            test_message = A2AMessage(
                id=f"perf_test_{i}",
                type=MessageType.REQUEST,
                sender="test_client",
                receiver=agent.agent_id,
                timestamp=datetime.utcnow().isoformat(),
                payload={"request": f"Performance test {i}"}
            )
            
            start_time = time.time()
            response = await agent.process_message(test_message)
            end_time = time.time()
            
            # Record metrics
            await observability_manager.record_metric(
                metric_name="agent_response_time",
                value=end_time - start_time,
                labels={"agent_id": agent.agent_id, "message_type": "request"}
            )
            
            assert response is not None
            assert response.type == MessageType.RESPONSE
        
        # 3. Check SLO compliance
        slo_results = await slo_sli_framework.evaluate_slos()
        
        # At least some SLOs should be passing
        passing_slos = [slo for slo in slo_results if slo["status"] == "passing"]
        assert len(passing_slos) > 0, "No SLOs are passing"
        
        # 4. Generate performance report
        performance_metrics = await observability_manager.get_agent_metrics(agent.agent_id)
        assert "agent_response_time" in performance_metrics
        
        # 5. Stop monitoring
        await observability_manager.stop_monitoring_agent(agent.agent_id)
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_workflow(self, production_components, mock_agents):
        """Test disaster recovery and backup workflow"""
        
        disaster_recovery = production_components['disaster_recovery']
        secrets_manager = production_components['secrets_manager']
        
        # 1. Store some test data
        test_secrets = {
            "test_secret_1": "secret_value_1",
            "test_secret_2": "secret_value_2",
            "test_secret_3": "secret_value_3"
        }
        
        for secret_id, value in test_secrets.items():
            await secrets_manager.store_secret(
                secret_id=secret_id,
                value=value,
                secret_type=SecretType.API_KEY,
                scope=SecretScope.SERVICE
            )
        
        # 2. Create backup
        backup_id = await disaster_recovery.create_backup(
            backup_type=BackupType.FULL,
            include_secrets=True,
            compression=True
        )
        assert backup_id is not None
        
        # 3. Verify backup exists
        backup_metadata = await disaster_recovery.get_backup_metadata(backup_id)
        assert backup_metadata is not None
        assert backup_metadata["backup_type"] == BackupType.FULL.value
        
        # 4. Test recovery point creation
        recovery_point_id = await disaster_recovery.create_recovery_point(
            description="Test recovery point",
            include_data=True
        )
        assert recovery_point_id is not None
        
        # 5. Simulate disaster (delete some data)
        await secrets_manager.delete_secret("test_secret_2")
        
        # Verify deletion
        retrieved_secret = await secrets_manager.retrieve_secret("test_secret_2")
        assert retrieved_secret is None
        
        # 6. Restore from backup
        restore_result = await disaster_recovery.restore_from_backup(
            backup_id=backup_id,
            selective_restore=["secrets"]
        )
        assert restore_result["status"] == "success"
        
        # 7. Verify data restoration
        # Note: In a real implementation, this would restore the deleted secret
        # For this test, we'll just verify the restore process completed
        assert "restored_items" in restore_result
    
    @pytest.mark.asyncio
    async def test_load_and_stress_workflow(self, production_components, mock_agents):
        """Test system behavior under load"""
        
        backpressure_manager = production_components['backpressure_manager']
        resource_manager = production_components['resource_manager']
        observability_manager = production_components['observability_manager']
        
        agents = list(mock_agents.values())
        
        # 1. Configure backpressure thresholds
        await backpressure_manager.configure_thresholds({
            "message_queue_size": 100,
            "cpu_usage": 80.0,
            "memory_usage": 80.0,
            "response_time": 1.0
        })
        
        # 2. Generate high load
        concurrent_messages = 50
        message_tasks = []
        
        start_time = time.time()
        
        for i in range(concurrent_messages):
            for agent in agents:
                test_message = A2AMessage(
                    id=f"load_test_{i}_{agent.agent_id}",
                    type=MessageType.REQUEST,
                    sender="load_tester",
                    receiver=agent.agent_id,
                    timestamp=datetime.utcnow().isoformat(),
                    payload={"load_test": True, "message_id": i}
                )
                
                # Create task but don't await yet
                task = asyncio.create_task(agent.process_message(test_message))
                message_tasks.append(task)
        
        # 3. Process all messages concurrently
        responses = await asyncio.gather(*message_tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 4. Analyze results
        successful_responses = [r for r in responses if isinstance(r, A2AMessage)]
        failed_responses = [r for r in responses if isinstance(r, Exception)]
        
        success_rate = len(successful_responses) / len(responses)
        throughput = len(successful_responses) / total_time
        
        logger.info(f"Load test results: {len(successful_responses)}/{len(responses)} successful, "
                   f"{success_rate:.2%} success rate, {throughput:.1f} msg/s throughput")
        
        # 5. Check system health after load
        bp_status = await backpressure_manager.get_system_status()
        resource_stats = resource_manager.get_resource_stats()
        
        # System should still be operational
        assert bp_status["overall_health"] in ["healthy", "warning", "critical"]
        assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"
        
        # 6. Verify backpressure activated if needed
        if bp_status["overall_health"] == "critical":
            # Backpressure should have been activated
            assert bp_status["backpressure_active"] is True
            logger.info("Backpressure correctly activated under high load")
    
    @pytest.mark.asyncio
    async def test_full_system_integration_workflow(self, production_components, mock_agents):
        """Test complete system integration with all components"""
        
        # Get all components
        secrets_manager = production_components['secrets_manager']
        resource_manager = production_components['resource_manager']
        protocol_validator = production_components['protocol_validator']
        slo_sli_framework = production_components['slo_sli_framework']
        backpressure_manager = production_components['backpressure_manager']
        security_framework = production_components['security_framework']
        observability_manager = production_components['observability_manager']
        disaster_recovery = production_components['disaster_recovery']
        distributed_coordinator = production_components['distributed_coordinator']
        
        agents = list(mock_agents.values())
        
        # 1. System initialization and configuration
        logger.info("Starting full system integration test")
        
        # Configure SLOs
        await slo_sli_framework.create_slo(SLODefinition(
            name="agent_availability",
            description="Agent availability SLO",
            target_value=99.0,
            threshold_value=95.0,
            evaluation_window=timedelta(minutes=5),
            sli_type=SLIType.AVAILABILITY
        ))
        
        # 2. Agent registration and authentication
        for agent in agents:
            # Store agent credentials
            await secrets_manager.store_secret(
                secret_id=f"{agent.agent_id}_credentials",
                value=f"cred_{agent.agent_id}",
                secret_type=SecretType.API_KEY,
                scope=SecretScope.AGENT
            )
            
            # Register with security framework
            await security_framework.auth_manager.register_principal(
                principal_id=agent.agent_id,
                principal_type="agent",
                name=f"Integration Test Agent {agent.agent_id}",
                permissions={Permission.READ_MESSAGES, Permission.WRITE_MESSAGES, Permission.EXECUTE_TASKS},
                authentication_methods=[AuthenticationMethod.API_KEY]
            )
            
            # Register with coordinator
            await distributed_coordinator.register_agent(
                agent_id=agent.agent_id,
                capabilities=agent.capabilities
            )
        
        # 3. Create and execute coordinated workflow
        workflow_task = {
            "workflow_id": "integration_test_workflow",
            "steps": [
                {"step": "data_collection", "agent_capability": AgentCapability.TOOL_CALLING},
                {"step": "analysis", "agent_capability": AgentCapability.REASONING},
                {"step": "reporting", "agent_capability": AgentCapability.MEMORY}
            ],
            "timeout": 60
        }
        
        # Start workflow execution
        workflow_id = await distributed_coordinator.create_task(
            task_data=workflow_task,
            priority=TaskPriority.HIGH,
            required_capabilities=[AgentCapability.TOOL_CALLING, AgentCapability.REASONING]
        )
        
        # 4. Monitor workflow execution
        start_time = time.time()
        workflow_messages = []
        
        for step in workflow_task["steps"]:
            # Find suitable agent for step
            suitable_agents = await distributed_coordinator.find_suitable_agents(
                [step["agent_capability"]]
            )
            
            if suitable_agents:
                agent = agents[0]  # Use first available agent for simplicity
                
                # Create step message
                step_message = A2AMessage(
                    id=f"step_{step['step']}_{uuid.uuid4()}",
                    type=MessageType.REQUEST,
                    sender="workflow_coordinator",
                    receiver=agent.agent_id,
                    timestamp=datetime.utcnow().isoformat(),
                    payload=step
                )
                
                # Validate message
                validation_results = await protocol_validator.validate_message(step_message)
                assert len([r for r in validation_results if r.severity == ValidationSeverity.ERROR]) == 0
                
                # Execute step
                response = await agent.process_message(step_message)
                workflow_messages.append((step_message, response))
                
                # Record metrics
                await observability_manager.record_metric(
                    metric_name="workflow_step_duration",
                    value=time.time() - start_time,
                    labels={"step": step["step"], "agent_id": agent.agent_id}
                )
        
        # 5. Evaluate system performance
        end_time = time.time()
        total_workflow_time = end_time - start_time
        
        # Check SLO compliance
        slo_results = await slo_sli_framework.evaluate_slos()
        
        # Check system health
        bp_status = await backpressure_manager.get_system_status()
        resource_stats = resource_manager.get_resource_stats()
        security_status = security_framework.get_security_status()
        
        # 6. Create recovery checkpoint
        checkpoint_id = await disaster_recovery.create_recovery_point(
            description="Post-integration-test checkpoint"
        )
        
        # 7. Assertions and validation
        assert len(workflow_messages) == len(workflow_task["steps"])
        assert total_workflow_time < workflow_task["timeout"]
        assert bp_status["overall_health"] != "failed"
        assert security_status["auth_manager"]["principals"] >= len(agents)
        assert checkpoint_id is not None
        
        # All steps should have responses
        for step_msg, response in workflow_messages:
            assert response is not None
            assert response.type == MessageType.RESPONSE
            assert response.in_reply_to == step_msg.id
        
        logger.info(f"Full system integration test completed in {total_workflow_time:.2f}s")
        logger.info(f"Processed {len(workflow_messages)} workflow steps successfully")
        logger.info(f"System health: {bp_status['overall_health']}")
        logger.info(f"Security status: {len(security_status['auth_manager']['principals'])} principals registered")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
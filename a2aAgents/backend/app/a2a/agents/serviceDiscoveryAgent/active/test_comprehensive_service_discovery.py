import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

"""
Comprehensive Test Suite for Service Discovery Agent
Tests service registration, discovery, load balancing, health monitoring, and simulations
"""

from .comprehensiveServiceDiscoveryAgentSdk import (
    ServiceDiscoveryAgentSdk, ServiceRegistration, ServiceEndpoint,
    ServiceStatus, ServiceQuery, HealthCheckResult, LoadBalancingStrategy
)
from app.a2a.core.security_base import SecureA2AAgent
from .mockServiceDiscoveryAgent import (
    MockServiceDiscoveryAgent, ServiceDiscoveryTestHelper
)
from .serviceDiscoverySimulator import (
    ServiceDiscoverySimulator, SimulationScenario,
    run_normal_operations_simulation, run_high_load_simulation
)

class TestServiceDiscoveryAgent(SecureA2AAgent):
    """Test suite for Service Discovery Agent"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    @pytest.fixture
    async def discovery_agent(self):
        """Create test discovery agent"""
        agent = ServiceDiscoveryAgentSdk()
        yield agent
        # Cleanup after test
        await agent.cleanup_expired_services()

    @pytest.fixture
    def mock_discovery_agent(self):
        """Create mock discovery agent"""
        return MockServiceDiscoveryAgent()

    @pytest.fixture
    def test_helper(self, mock_discovery_agent):
        """Create test helper"""
        return ServiceDiscoveryTestHelper(mock_discovery_agent)

    # Service Registration Tests

    @pytest.mark.asyncio
    async def test_service_registration_basic(self, discovery_agent):
        """Test basic service registration"""

        endpoints = [
            {
                "id": "ep-1",
                "url": "http://localhost:8001",
                "port": 8001
            }
        ]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="TestService",
            service_type="api",
            endpoints=endpoints,
            capabilities=["http", "json"],
            tags=["test"]
        )

        assert result["status"] == "registered"
        assert "service_id" in result
        assert result["endpoints_count"] == 1

        # Verify service is in registry
        service_id = result["service_id"]
        assert service_id in discovery_agent.service_registry

    @pytest.mark.asyncio
    async def test_service_registration_multiple_endpoints(self, discovery_agent):
        """Test service registration with multiple endpoints"""

        endpoints = [
            {"id": "ep-1", "url": "http://localhost:8001", "port": 8001, "weight": 1.0},
            {"id": "ep-2", "url": "http://localhost:8002", "port": 8002, "weight": 1.5}
        ]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="MultiEndpointService",
            service_type="api",
            endpoints=endpoints
        )

        assert result["endpoints_count"] == 2

        service_id = result["service_id"]
        registration = discovery_agent.service_registry[service_id]
        assert len(registration.endpoints) == 2
        assert registration.endpoints[1].weight == 1.5

    @pytest.mark.asyncio
    async def test_service_registration_with_health_check(self, discovery_agent):
        """Test service registration with health check configuration"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="HealthCheckedService",
            service_type="api",
            endpoints=endpoints,
            health_check_url="http://localhost:8001/health",
            health_check_interval=30
        )

        assert result["health_monitoring"] == True

        service_id = result["service_id"]
        registration = discovery_agent.service_registry[service_id]
        assert registration.health_check_url == "http://localhost:8001/health"
        assert registration.health_check_interval == 30

    # Service Discovery Tests

    @pytest.mark.asyncio
    async def test_service_discovery_by_name(self, discovery_agent):
        """Test service discovery by service name"""

        # Register test service
        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        reg_result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="DiscoveryTestService",
            service_type="api",
            endpoints=endpoints,
            capabilities=["http", "json"]
        )

        # Discover service by name
        result = await discovery_agent.discover_services(
            service_name="DiscoveryTestService"
        )

        assert result["total_found"] == 1
        assert len(result["services"]) == 1
        assert result["services"][0]["service_name"] == "DiscoveryTestService"
        assert result["services"][0]["service_id"] == reg_result["service_id"]

    @pytest.mark.asyncio
    async def test_service_discovery_by_type(self, discovery_agent):
        """Test service discovery by service type"""

        # Register multiple services of same type
        for i in range(3):
            endpoints = [{"id": f"ep-{i}", "url": f"http://localhost:{8001+i}", "port": 8001+i}]

            await discovery_agent.register_service(
                agent_id=f"test-agent-{i}",
                service_name=f"TypeTestService{i}",
                service_type="database",
                endpoints=endpoints
            )

        # Discover services by type
        result = await discovery_agent.discover_services(service_type="database")

        assert result["total_found"] == 3
        for service in result["services"]:
            assert service["service_type"] == "database"

    @pytest.mark.asyncio
    async def test_service_discovery_by_capabilities(self, discovery_agent):
        """Test service discovery by capabilities"""

        # Register services with different capabilities
        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        await discovery_agent.register_service(
            agent_id="test-agent-1",
            service_name="CapabilityService1",
            service_type="api",
            endpoints=endpoints,
            capabilities=["http", "json", "auth"]
        )

        await discovery_agent.register_service(
            agent_id="test-agent-2",
            service_name="CapabilityService2",
            service_type="api",
            endpoints=endpoints,
            capabilities=["http", "xml"]
        )

        # Discover services with specific capabilities
        result = await discovery_agent.discover_services(
            capabilities=["json", "auth"]
        )

        assert result["total_found"] == 1
        assert result["services"][0]["service_name"] == "CapabilityService1"

    # Load Balancing Tests

    @pytest.mark.asyncio
    async def test_load_balancing_health_based(self, discovery_agent):
        """Test health-based load balancing"""

        # Register service with multiple endpoints
        endpoints = [
            {"id": "ep-1", "url": "http://localhost:8001", "port": 8001},
            {"id": "ep-2", "url": "http://localhost:8002", "port": 8002}
        ]

        await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="LoadBalanceService",
            service_type="api",
            endpoints=endpoints
        )

        # Manually set different health metrics
        service_id = list(discovery_agent.service_registry.keys())[0]
        registration = discovery_agent.service_registry[service_id]
        registration.endpoints[0].response_time_ms = 50.0
        registration.endpoints[0].success_rate = 0.95
        registration.endpoints[1].response_time_ms = 200.0
        registration.endpoints[1].success_rate = 0.80

        # Get endpoint using health-based strategy
        result = await discovery_agent.get_service_endpoint(
            service_name="LoadBalanceService",
            strategy="health_based"
        )

        assert result["strategy_used"] == "health_based"
        assert result["endpoint"]["id"] == "ep-1"  # Should select better performing endpoint

    @pytest.mark.asyncio
    async def test_load_balancing_round_robin(self, discovery_agent):
        """Test round-robin load balancing"""

        endpoints = [
            {"id": "ep-1", "url": "http://localhost:8001", "port": 8001},
            {"id": "ep-2", "url": "http://localhost:8002", "port": 8002},
            {"id": "ep-3", "url": "http://localhost:8003", "port": 8003}
        ]

        await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="RoundRobinService",
            service_type="api",
            endpoints=endpoints
        )

        # Make multiple requests and verify round-robin
        selected_endpoints = []
        for _ in range(6):  # 2 full rounds
            result = await discovery_agent.get_service_endpoint(
                service_name="RoundRobinService",
                strategy="round_robin"
            )
            selected_endpoints.append(result["endpoint"]["id"])

        # Verify round-robin pattern
        assert selected_endpoints[:3] == ["ep-1", "ep-2", "ep-3"]
        assert selected_endpoints[3:] == ["ep-1", "ep-2", "ep-3"]

    # Health Monitoring Tests

    @pytest.mark.asyncio
    async def test_health_monitoring_basic(self, discovery_agent):
        """Test basic health monitoring"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="HealthService",
            service_type="api",
            endpoints=endpoints
        )

        service_id = result["service_id"]

        # Get health status
        health_result = await discovery_agent.get_service_health(service_id=service_id)

        assert health_result["total_services"] == 1
        assert len(health_result["services"]) == 1
        assert health_result["services"][0]["service_id"] == service_id

    @pytest.mark.asyncio
    async def test_heartbeat_processing(self, discovery_agent):
        """Test heartbeat processing"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="HeartbeatService",
            service_type="api",
            endpoints=endpoints,
            ttl_seconds=60
        )

        service_id = result["service_id"]

        # Send heartbeat
        heartbeat_result = await discovery_agent.send_heartbeat(
            service_id=service_id,
            agent_id="test-agent",
            status="healthy"
        )

        assert heartbeat_result["status"] == "heartbeat_received"
        assert "next_heartbeat_due" in heartbeat_result

    # Service Lifecycle Tests

    @pytest.mark.asyncio
    async def test_service_deregistration(self, discovery_agent):
        """Test service deregistration"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="DeregisterService",
            service_type="api",
            endpoints=endpoints
        )

        service_id = result["service_id"]

        # Verify service exists
        assert service_id in discovery_agent.service_registry

        # Deregister service
        dereg_result = await discovery_agent.deregister_service(
            service_id=service_id,
            agent_id="test-agent"
        )

        assert dereg_result["status"] == "deregistered"
        assert dereg_result["service_id"] == service_id

        # Verify service is removed
        assert service_id not in discovery_agent.service_registry

    @pytest.mark.asyncio
    async def test_service_ttl_expiration(self, discovery_agent):
        """Test service TTL expiration"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="TTLService",
            service_type="api",
            endpoints=endpoints,
            ttl_seconds=1  # Very short TTL
        )

        service_id = result["service_id"]

        # Verify service exists
        assert service_id in discovery_agent.service_registry

        # Wait for TTL to expire
        await asyncio.sleep(2)

        # Trigger cleanup
        await discovery_agent.cleanup_expired_services()

        # Verify service is removed
        assert service_id not in discovery_agent.service_registry

    # Mock Testing

    @pytest.mark.asyncio
    async def test_mock_service_registration(self, mock_discovery_agent):
        """Test mock service registration"""

        endpoints = [{"id": "ep-1", "url": "http://mock:8001", "port": 8001}]

        result = await mock_discovery_agent.register_service(
            agent_id="mock-agent",
            service_name="MockService",
            service_type="api",
            endpoints=endpoints
        )

        assert result["status"] == "registered"
        assert "service_id" in result

    @pytest.mark.asyncio
    async def test_mock_failure_scenarios(self, mock_discovery_agent):
        """Test mock failure scenarios"""

        # Set up failure scenario
        mock_discovery_agent.set_failure_scenario("register_service", True)

        endpoints = [{"id": "ep-1", "url": "http://mock:8001", "port": 8001}]

        with pytest.raises(Exception, match="Mock registration failure"):
            await mock_discovery_agent.register_service(
                agent_id="mock-agent",
                service_name="FailService",
                service_type="api",
                endpoints=endpoints
            )

        # Clear failure scenario
        mock_discovery_agent.clear_failure_scenarios()

    @pytest.mark.asyncio
    async def test_test_helper_utilities(self, test_helper):
        """Test helper utility functions"""

        # Create test service
        service_id = await test_helper.create_test_service(
            service_name="HelperTestService",
            service_type="test",
            endpoint_count=2
        )

        assert service_id in test_helper.mock_agent.mock_services

        # Simulate failure and recovery
        await test_helper.simulate_service_failure(service_id)
        service = test_helper.mock_agent.mock_services[service_id]
        assert service.status == ServiceStatus.UNHEALTHY

        await test_helper.simulate_service_recovery(service_id)
        assert service.status == ServiceStatus.HEALTHY

    # Simulation Tests

    @pytest.mark.asyncio
    async def test_basic_simulation(self, discovery_agent):
        """Test basic simulation framework"""

        simulator = ServiceDiscoverySimulator(discovery_agent)

        # Setup small simulation
        await simulator.setup_simulation(
            agent_count=3,
            scenario=SimulationScenario.NORMAL_OPERATIONS
        )

        # Run short simulation
        metrics = await simulator.run_simulation(duration_seconds=10)

        # Verify metrics
        assert metrics.total_requests >= 0
        assert metrics.service_registrations >= 0

        # Cleanup
        await simulator.cleanup_simulation()

    @pytest.mark.asyncio
    async def test_simulation_scenarios(self, discovery_agent):
        """Test different simulation scenarios"""

        scenarios_to_test = [
            SimulationScenario.NORMAL_OPERATIONS,
            SimulationScenario.HIGH_LOAD,
            SimulationScenario.SERVICE_FAILURES
        ]

        for scenario in scenarios_to_test:
            simulator = ServiceDiscoverySimulator(discovery_agent)

            await simulator.setup_simulation(
                agent_count=2,
                scenario=scenario
            )

            await simulator.run_simulation(duration_seconds=5)

            # Verify scenario was applied
            assert len(simulator.simulation_agents) >= 2

            await simulator.cleanup_simulation()

    @pytest.mark.asyncio
    async def test_simulation_convenience_functions(self, discovery_agent):
        """Test simulation convenience functions"""

        # Test normal operations simulation
        report = await run_normal_operations_simulation(
            discovery_agent,
            duration_seconds=10
        )

        assert "summary" in report
        assert "service_lifecycle" in report
        assert "load_balancing" in report

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_duplicate_service_id_handling(self, discovery_agent):
        """Test handling of potential duplicate service scenarios"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        # Register service
        result1 = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="UniqueService",
            service_type="api",
            endpoints=endpoints
        )

        # Register another service with same name (should get different ID)
        result2 = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="UniqueService",
            service_type="api",
            endpoints=endpoints
        )

        assert result1["service_id"] != result2["service_id"]

    @pytest.mark.asyncio
    async def test_unauthorized_deregistration(self, discovery_agent):
        """Test unauthorized deregistration attempt"""

        endpoints = [{"id": "ep-1", "url": "http://localhost:8001", "port": 8001}]

        result = await discovery_agent.register_service(
            agent_id="test-agent",
            service_name="AuthService",
            service_type="api",
            endpoints=endpoints
        )

        service_id = result["service_id"]

        # Try to deregister with wrong agent ID
        with pytest.raises(ValueError, match="not authorized"):
            await discovery_agent.deregister_service(
                service_id=service_id,
                agent_id="wrong-agent"
            )

    @pytest.mark.asyncio
    async def test_nonexistent_service_operations(self, discovery_agent):
        """Test operations on nonexistent services"""

        fake_service_id = "fake-service-id"

        # Test deregistration of nonexistent service
        with pytest.raises(ValueError, match="not found"):
            await discovery_agent.deregister_service(
                service_id=fake_service_id,
                agent_id="test-agent"
            )

        # Test heartbeat for nonexistent service
        with pytest.raises(ValueError, match="not found"):
            await discovery_agent.send_heartbeat(
                service_id=fake_service_id,
                agent_id="test-agent"
            )

    # Integration Tests

    @pytest.mark.asyncio
    async def test_full_lifecycle_integration(self, discovery_agent):
        """Test complete service lifecycle integration"""

        endpoints = [
            {"id": "ep-1", "url": "http://localhost:8001", "port": 8001},
            {"id": "ep-2", "url": "http://localhost:8002", "port": 8002}
        ]

        # 1. Register service
        reg_result = await discovery_agent.register_service(
            agent_id="integration-agent",
            service_name="IntegrationService",
            service_type="api",
            endpoints=endpoints,
            capabilities=["http", "json"],
            health_check_url="http://localhost:8001/health",
            health_check_interval=30,
            tags=["integration", "test"]
        )

        service_id = reg_result["service_id"]

        # 2. Discover service
        discovery_result = await discovery_agent.discover_services(
            service_name="IntegrationService"
        )

        assert discovery_result["total_found"] == 1
        assert discovery_result["services"][0]["service_id"] == service_id

        # 3. Get endpoint via load balancing
        endpoint_result = await discovery_agent.get_service_endpoint(
            service_name="IntegrationService",
            strategy="health_based"
        )

        assert endpoint_result["service_id"] == service_id
        assert endpoint_result["endpoint"]["id"] in ["ep-1", "ep-2"]

        # 4. Check health
        health_result = await discovery_agent.get_service_health(service_id=service_id)

        assert health_result["total_services"] == 1
        assert health_result["services"][0]["service_id"] == service_id

        # 5. Send heartbeat
        heartbeat_result = await discovery_agent.send_heartbeat(
            service_id=service_id,
            agent_id="integration-agent",
            status="healthy"
        )

        assert heartbeat_result["status"] == "heartbeat_received"

        # 6. Deregister service
        dereg_result = await discovery_agent.deregister_service(
            service_id=service_id,
            agent_id="integration-agent"
        )

        assert dereg_result["status"] == "deregistered"

        # 7. Verify service is gone
        discovery_result = await discovery_agent.discover_services(
            service_name="IntegrationService"
        )

        assert discovery_result["total_found"] == 0


# Performance Tests
class TestServiceDiscoveryPerformance(SecureA2AAgent):

        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning

    """Performance tests for service discovery"""

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, discovery_agent):
        """Test concurrent service registrations"""

        async def register_service(i):
            endpoints = [{"id": f"ep-{i}", "url": f"http://localhost:{8000+i}", "port": 8000+i}]
            return await discovery_agent.register_service(
                agent_id=f"perf-agent-{i}",
                service_name=f"PerfService{i}",
                service_type="api",
                endpoints=endpoints
            )

        # Register 10 services concurrently
        tasks = [register_service(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all registrations succeeded
        assert len(results) == 10
        for result in results:
            assert result["status"] == "registered"

        # Verify all services are in registry
        assert len(discovery_agent.service_registry) == 10

    @pytest.mark.asyncio
    async def test_concurrent_discovery_requests(self, discovery_agent):
        """Test concurrent discovery requests"""

        # First register some services
        for i in range(5):
            endpoints = [{"id": f"ep-{i}", "url": f"http://localhost:{8000+i}", "port": 8000+i}]
            await discovery_agent.register_service(
                agent_id=f"disco-agent-{i}",
                service_name=f"DiscoService{i}",
                service_type="api",
                endpoints=endpoints
            )

        # Perform concurrent discovery requests
        async def discovery_request():
            return await discovery_agent.discover_services(service_type="api")

        tasks = [discovery_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # Verify all requests succeeded and returned consistent results
        assert len(results) == 20
        for result in results:
            assert result["total_found"] == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

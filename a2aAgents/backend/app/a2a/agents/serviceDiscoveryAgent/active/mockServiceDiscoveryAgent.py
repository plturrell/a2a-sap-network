import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from unittest.mock import Mock, AsyncMock
import logging

from .comprehensiveServiceDiscoveryAgentSdk import (
    ServiceDiscoveryAgentSdk, ServiceRegistration, ServiceEndpoint,
    ServiceStatus, ServiceQuery, HealthCheckResult, LoadBalancingStrategy
)
from app.a2a.core.security_base import SecureA2AAgent

"""
Mock Service Discovery Agent for Testing
Provides mock implementations for isolated testing
"""

logger = logging.getLogger(__name__)

class MockServiceDiscoveryAgent(SecureA2AAgent):
    """Mock implementation of Service Discovery Agent for testing"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self):
        
        super().__init__()
        self.mock_services = {}
        self.mock_health_results = {}
        self.mock_load_balancer_calls = []
        self.mock_heartbeats = {}
        self.failure_scenarios = {}
        
        # Pre-populate with test services
        self._populate_test_services()
    
    def _populate_test_services(self):
        """Populate mock registry with test services"""
        
        # Mock Agent Manager Service
        agent_manager_endpoints = [
            ServiceEndpoint(
                id="am-ep-1",
                url="http://localhost:8001",
                protocol="http",
                port=8001,
                weight=1.0,
                response_time_ms=50.0,
                success_rate=0.95
            ),
            ServiceEndpoint(
                id="am-ep-2", 
                url="http://localhost:8002",
                protocol="http",
                port=8002,
                weight=1.0,
                response_time_ms=75.0,
                success_rate=0.90
            )
        ]
        
        agent_manager_service = ServiceRegistration(
            service_id="agent-manager-001",
            agent_id="agent-manager",
            service_name="AgentManager",
            service_type="coordination",
            version="1.0.0",
            endpoints=agent_manager_endpoints,
            capabilities=["agent_lifecycle", "agent_coordination", "resource_management"],
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8001/health",
            tags=["core", "management"]
        )
        
        # Mock Reasoning Agent Service
        reasoning_endpoints = [
            ServiceEndpoint(
                id="ra-ep-1",
                url="http://localhost:8003",
                protocol="http", 
                port=8003,
                weight=1.5,
                response_time_ms=120.0,
                success_rate=0.98
            )
        ]
        
        reasoning_service = ServiceRegistration(
            service_id="reasoning-agent-001",
            agent_id="reasoning-agent",
            service_name="ReasoningAgent",
            service_type="intelligence",
            version="1.0.0",
            endpoints=reasoning_endpoints,
            capabilities=["logical_reasoning", "problem_solving", "inference"],
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8003/health",
            tags=["intelligence", "core"]
        )
        
        # Mock SQL Agent Service (Degraded)
        sql_endpoints = [
            ServiceEndpoint(
                id="sql-ep-1",
                url="http://localhost:8004",
                protocol="http",
                port=8004,
                weight=1.0,
                response_time_ms=200.0,
                success_rate=0.70  # Degraded performance
            )
        ]
        
        sql_service = ServiceRegistration(
            service_id="sql-agent-001",
            agent_id="sql-agent",
            service_name="SqlAgent",
            service_type="data",
            version="1.0.0",
            endpoints=sql_endpoints,
            capabilities=["database_query", "data_analysis", "sql_execution"],
            status=ServiceStatus.DEGRADED,
            health_check_url="http://localhost:8004/health",
            tags=["data", "database"]
        )
        
        # Store mock services
        self.mock_services = {
            "agent-manager-001": agent_manager_service,
            "reasoning-agent-001": reasoning_service,
            "sql-agent-001": sql_service
        }

    async def register_service(
        self,
        agent_id: str,
        service_name: str,
        service_type: str,
        endpoints: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Mock service registration"""
        
        if self.failure_scenarios.get("register_service"):
            raise Exception("Mock registration failure")
        
        service_id = f"mock-{str(uuid.uuid4())[:8]}"
        
        # Convert endpoints
        service_endpoints = []
        for ep_data in endpoints:
            endpoint = ServiceEndpoint(
                id=ep_data.get("id", str(uuid.uuid4())),
                url=ep_data["url"],
                protocol=ep_data.get("protocol", "http"),
                port=ep_data.get("port", 8000),
                weight=ep_data.get("weight", 1.0)
            )
            service_endpoints.append(endpoint)
        
        # Create mock registration
        registration = ServiceRegistration(
            service_id=service_id,
            agent_id=agent_id,
            service_name=service_name,
            service_type=service_type,
            version=kwargs.get("version", "1.0.0"),
            endpoints=service_endpoints,
            capabilities=kwargs.get("capabilities", []),
            status=ServiceStatus.HEALTHY
        )
        
        self.mock_services[service_id] = registration
        
        logger.info(f"Mock registered service: {service_name} ({service_id})")
        
        return {
            "service_id": service_id,
            "status": "registered",
            "endpoints_count": len(service_endpoints),
            "health_monitoring": False
        }

    async def discover_services(
        self,
        service_name: Optional[str] = None,
        service_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock service discovery"""
        
        if self.failure_scenarios.get("discover_services"):
            raise Exception("Mock discovery failure")
        
        matching_services = []
        
        for registration in self.mock_services.values():
            # Apply filters
            if service_name and registration.service_name != service_name:
                continue
            if service_type and registration.service_type != service_type:
                continue
            if capabilities:
                if not all(cap in registration.capabilities for cap in capabilities):
                    continue
            
            # Convert to response format
            service_info = {
                "service_id": registration.service_id,
                "agent_id": registration.agent_id,
                "service_name": registration.service_name,
                "service_type": registration.service_type,
                "version": registration.version,
                "status": registration.status.value,
                "capabilities": registration.capabilities,
                "tags": registration.tags,
                "endpoints": [
                    {
                        "id": ep.id,
                        "url": ep.url,
                        "protocol": ep.protocol,
                        "port": ep.port,
                        "response_time_ms": ep.response_time_ms,
                        "success_rate": ep.success_rate
                    }
                    for ep in registration.endpoints
                ]
            }
            matching_services.append(service_info)
        
        logger.info(f"Mock discovered {len(matching_services)} services")
        
        return {
            "services": matching_services,
            "total_found": len(matching_services),
            "query": {
                "service_name": service_name,
                "service_type": service_type,
                "capabilities": capabilities
            }
        }

    async def get_service_endpoint(
        self,
        service_name: str,
        strategy: str = "health_based",
        **kwargs
    ) -> Dict[str, Any]:
        """Mock load balancing endpoint selection"""
        
        if self.failure_scenarios.get("get_service_endpoint"):
            raise Exception("Mock endpoint selection failure")
        
        # Track load balancer calls
        self.mock_load_balancer_calls.append({
            "service_name": service_name,
            "strategy": strategy,
            "timestamp": datetime.now()
        })
        
        # Find service by name
        matching_service = None
        for service in self.mock_services.values():
            if service.service_name == service_name:
                matching_service = service
                break
        
        if not matching_service:
            raise ValueError(f"Mock service not found: {service_name}")
        
        if not matching_service.endpoints:
            raise ValueError(f"Mock service has no endpoints: {service_name}")
        
        # Simple selection logic for mock
        if strategy == "health_based":
            selected_endpoint = max(
                matching_service.endpoints,
                key=lambda ep: ep.success_rate / (ep.response_time_ms + 1)
            )
        else:
            selected_endpoint = matching_service.endpoints[0]
        
        logger.info(f"Mock selected endpoint {selected_endpoint.id} for {service_name}")
        
        return {
            "service_id": matching_service.service_id,
            "endpoint": {
                "id": selected_endpoint.id,
                "url": selected_endpoint.url,
                "protocol": selected_endpoint.protocol,
                "port": selected_endpoint.port,
                "response_time_ms": selected_endpoint.response_time_ms,
                "success_rate": selected_endpoint.success_rate
            },
            "strategy_used": strategy,
            "total_available": len(matching_service.endpoints)
        }

    async def get_service_health(
        self,
        service_id: Optional[str] = None,
        service_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock health monitoring"""
        
        if self.failure_scenarios.get("get_service_health"):
            raise Exception("Mock health check failure")
        
        if service_id:
            services = [self.mock_services.get(service_id)]
            services = [s for s in services if s is not None]
        elif service_name:
            services = [
                s for s in self.mock_services.values()
                if s.service_name == service_name
            ]
        else:
            services = list(self.mock_services.values())
        
        results = []
        for service in services:
            health_info = {
                "service_id": service.service_id,
                "service_name": service.service_name,
                "agent_id": service.agent_id,
                "status": service.status.value,
                "last_heartbeat": service.last_heartbeat.isoformat(),
                "endpoints": [
                    {
                        "endpoint_id": ep.id,
                        "url": ep.url,
                        "response_time_ms": ep.response_time_ms,
                        "success_rate": ep.success_rate,
                        "current_connections": ep.current_connections,
                        "max_connections": ep.max_connections
                    }
                    for ep in service.endpoints
                ]
            }
            results.append(health_info)
        
        return {
            "services": results,
            "total_services": len(results),
            "timestamp": datetime.now().isoformat()
        }

    async def send_heartbeat(
        self,
        service_id: str,
        agent_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock heartbeat processing"""
        
        if self.failure_scenarios.get("send_heartbeat"):
            raise Exception("Mock heartbeat failure")
        
        if service_id not in self.mock_services:
            raise ValueError(f"Mock service not found: {service_id}")
        
        service = self.mock_services[service_id]
        service.last_heartbeat = datetime.now()
        
        # Track heartbeat
        self.mock_heartbeats[service_id] = {
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "service_name": service.service_name
        }
        
        logger.debug(f"Mock heartbeat for service: {service.service_name}")
        
        return {
            "service_id": service_id,
            "status": "heartbeat_received",
            "next_heartbeat_due": (datetime.now() + timedelta(seconds=300)).isoformat()
        }

    async def deregister_service(
        self,
        service_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """Mock service deregistration"""
        
        if self.failure_scenarios.get("deregister_service"):
            raise Exception("Mock deregistration failure")
        
        if service_id not in self.mock_services:
            raise ValueError(f"Mock service not found: {service_id}")
        
        service = self.mock_services[service_id]
        service_name = service.service_name
        
        del self.mock_services[service_id]
        
        logger.info(f"Mock deregistered service: {service_name}")
        
        return {
            "service_id": service_id,
            "status": "deregistered",
            "service_name": service_name
        }

    def set_failure_scenario(self, method: str, should_fail: bool = True):
        """Set up failure scenarios for testing"""
        self.failure_scenarios[method] = should_fail

    def clear_failure_scenarios(self):
        """Clear all failure scenarios"""
        self.failure_scenarios.clear()

    def get_mock_statistics(self) -> Dict[str, Any]:
        """Get mock usage statistics for test verification"""
        return {
            "registered_services": len(self.mock_services),
            "load_balancer_calls": len(self.mock_load_balancer_calls),
            "heartbeats_received": len(self.mock_heartbeats),
            "services_by_type": {
                service_type: len([
                    s for s in self.mock_services.values()
                    if s.service_type == service_type
                ])
                for service_type in set(s.service_type for s in self.mock_services.values())
            },
            "services_by_status": {
                status.value: len([
                    s for s in self.mock_services.values()
                    if s.status == status
                ])
                for status in ServiceStatus
            }
        }

    def reset_mock_data(self):
        """Reset all mock data to initial state"""
        self.mock_services.clear()
        self.mock_health_results.clear()
        self.mock_load_balancer_calls.clear()
        self.mock_heartbeats.clear()
        self.failure_scenarios.clear()
        self._populate_test_services()


# Test utilities and fixtures
class ServiceDiscoveryTestHelper(SecureA2AAgent):
    """Helper class for service discovery testing"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, mock_agent: MockServiceDiscoveryAgent):
        
        super().__init__()
        self.mock_agent = mock_agent
    
    async def create_test_service(
        self,
        service_name: str,
        service_type: str = "test",
        agent_id: str = "test-agent",
        endpoint_count: int = 1,
        status: ServiceStatus = ServiceStatus.HEALTHY
    ) -> str:
        """Create a test service for testing"""
        
        endpoints = []
        for i in range(endpoint_count):
            endpoints.append({
                "id": f"test-ep-{i}",
                "url": f"http://localhost:{8000 + i}",
                "port": 8000 + i
            })
        
        result = await self.mock_agent.register_service(
            agent_id=agent_id,
            service_name=service_name,
            service_type=service_type,
            endpoints=endpoints
        )
        
        # Update status if needed
        service_id = result["service_id"]
        if status != ServiceStatus.HEALTHY:
            self.mock_agent.mock_services[service_id].status = status
        
        return service_id
    
    async def simulate_service_failure(self, service_id: str):
        """Simulate service failure"""
        if service_id in self.mock_agent.mock_services:
            service = self.mock_agent.mock_services[service_id]
            service.status = ServiceStatus.UNHEALTHY
            for endpoint in service.endpoints:
                endpoint.success_rate = 0.0
                endpoint.response_time_ms = 5000.0
    
    async def simulate_service_recovery(self, service_id: str):
        """Simulate service recovery"""
        if service_id in self.mock_agent.mock_services:
            service = self.mock_agent.mock_services[service_id]
            service.status = ServiceStatus.HEALTHY
            for endpoint in service.endpoints:
                endpoint.success_rate = 0.95
                endpoint.response_time_ms = 100.0
    
    def verify_load_balancing(self, service_name: str, expected_strategy: str) -> bool:
        """Verify load balancing calls"""
        calls = [
            call for call in self.mock_agent.mock_load_balancer_calls
            if call["service_name"] == service_name and call["strategy"] == expected_strategy
        ]
        return len(calls) > 0


# Create mock instance for testing
mock_service_discovery_agent = MockServiceDiscoveryAgent()
test_helper = ServiceDiscoveryTestHelper(mock_service_discovery_agent)

def get_mock_service_discovery_agent() -> MockServiceDiscoveryAgent:
    """Get mock service discovery agent for testing"""
    return mock_service_discovery_agent

def get_service_discovery_test_helper() -> ServiceDiscoveryTestHelper:
    """Get test helper for service discovery testing"""
    return test_helper

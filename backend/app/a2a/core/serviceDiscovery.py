"""
Real Service Discovery System for A2A Network
Replaces fallback mechanisms with proper service registry integration
"""

import asyncio
import logging
import os
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""

    service_name: str
    agent_id: str
    endpoint_url: str
    capabilities: List[str]
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class ServiceDiscovery:
    """
    Real service discovery implementation using A2A network registry
    No fallback to mock implementations - fails with proper errors
    """

    def __init__(self):
        self.registry_url = os.getenv("A2A_REGISTRY_URL")
        if not self.registry_url:
            raise ValueError(
                "A2A_REGISTRY_URL environment variable is required for service discovery"
            )

        self.local_cache: Dict[str, ServiceEndpoint] = {}
        self.cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes
        self.client_timeout = 30.0

        # Health check configuration
        self.health_check_interval = 60  # seconds
        self.unhealthy_threshold = 3  # failed checks before marking unhealthy
        self.service_health_status: Dict[str, int] = {}  # Track consecutive failures

        logger.info(f"Service Discovery initialized with registry: {self.registry_url}")

    async def discover_service(
        self, service_type: str, capabilities: Optional[List[str]] = None
    ) -> List[ServiceEndpoint]:
        """
        Discover services by type and capabilities
        Returns real services from A2A network registry - no fallbacks
        """
        try:
            # Check cache first
            cache_key = f"{service_type}:{':'.join(capabilities or [])}"
            cached_services = self._get_cached_services(cache_key)
            if cached_services:
                logger.debug(f"Returning {len(cached_services)} cached services for {service_type}")
                return cached_services

            # Query real A2A registry
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                query_params = {"service_type": service_type, "active": True}

                if capabilities:
                    query_params["capabilities"] = ",".join(capabilities)

                response = await client.get(
                    f"{self.registry_url}/api/v1/agents/search", params=query_params
                )
                response.raise_for_status()

                agents_data = response.json()
                services = []

                for agent_data in agents_data.get("agents", []):
                    service = ServiceEndpoint(
                        service_name=agent_data["name"],
                        agent_id=agent_data["agent_id"],
                        endpoint_url=agent_data["endpoint"],
                        capabilities=agent_data.get("capabilities", []),
                        health_status="unknown",
                        metadata=agent_data.get("metadata", {}),
                    )
                    services.append(service)

                # Cache the results
                self._cache_services(cache_key, services)

                logger.info(f"Discovered {len(services)} services for {service_type}")
                return services

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"Service discovery failed with HTTP {e.response.status_code}: {e.response.text}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Service registry unreachable: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Service discovery failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def get_service_endpoint(
        self, service_type: str, agent_id: Optional[str] = None
    ) -> ServiceEndpoint:
        """
        Get a specific service endpoint
        Returns real service endpoint - no fallbacks to localhost or defaults
        """
        services = await self.discover_service(service_type)

        if not services:
            error_msg = f"No {service_type} services available in A2A network"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if agent_id:
            # Find specific agent
            for service in services:
                if service.agent_id == agent_id:
                    return service

            error_msg = f"Agent {agent_id} not found in {service_type} services"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Return first healthy service
        healthy_services = [s for s in services if s.health_status == "healthy"]
        if healthy_services:
            return healthy_services[0]

        # If no healthy services, try to find any service but log warning
        logger.warning(f"No healthy {service_type} services found, using first available")
        return services[0]

    async def health_check_service(self, service: ServiceEndpoint) -> bool:
        """
        Perform health check on a service endpoint
        Returns real health status - no mock responses
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service.endpoint_url}/health")

                if response.status_code == 200:
                    health_data = response.json()
                    service.health_status = health_data.get("status", "healthy")
                    service.last_health_check = datetime.utcnow()

                    # Reset failure count on success
                    self.service_health_status[service.agent_id] = 0

                    return service.health_status == "healthy"
                else:
                    service.health_status = "unhealthy"
                    self._record_health_failure(service.agent_id)
                    return False

        except Exception as e:
            logger.warning(f"Health check failed for {service.agent_id}: {e}")
            service.health_status = "unhealthy"
            service.last_health_check = datetime.utcnow()
            self._record_health_failure(service.agent_id)
            return False

    async def register_service(self, service: ServiceEndpoint) -> bool:
        """
        Register a service with the A2A network registry
        Real registration - no mock implementations
        """
        try:
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                registration_data = {
                    "agent_id": service.agent_id,
                    "name": service.service_name,
                    "endpoint": service.endpoint_url,
                    "capabilities": service.capabilities,
                    "metadata": service.metadata or {},
                }

                response = await client.post(
                    f"{self.registry_url}/api/v1/agents/register", json=registration_data
                )
                response.raise_for_status()

                logger.info(f"Successfully registered service {service.agent_id}")
                return True

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"Service registration failed with HTTP {e.response.status_code}: {e.response.text}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Service registration failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def deregister_service(self, agent_id: str) -> bool:
        """
        Deregister a service from the A2A network registry
        """
        try:
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.delete(f"{self.registry_url}/api/v1/agents/{agent_id}")
                response.raise_for_status()

                # Remove from local cache
                self._remove_from_cache(agent_id)

                logger.info(f"Successfully deregistered service {agent_id}")
                return True

        except Exception as e:
            error_msg = f"Service deregistration failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def start_health_monitoring(self):
        """
        Start periodic health monitoring of discovered services
        """
        logger.info("Starting service health monitoring")

        while True:
            try:
                # Health check all cached services
                for service in self.local_cache.values():
                    if isinstance(service, ServiceEndpoint):
                        await self.health_check_service(service)

                # Clean up unhealthy services from cache
                self._cleanup_unhealthy_services()

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    def _get_cached_services(self, cache_key: str) -> Optional[List[ServiceEndpoint]]:
        """Get services from cache if not expired"""
        if cache_key not in self.local_cache:
            return None

        cached_data = self.local_cache[cache_key]
        if isinstance(cached_data, dict) and "timestamp" in cached_data:
            if datetime.utcnow() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["services"]

        return None

    def _cache_services(self, cache_key: str, services: List[ServiceEndpoint]):
        """Cache discovered services"""
        self.local_cache[cache_key] = {"services": services, "timestamp": datetime.utcnow()}

        # Also cache individual services by agent_id
        for service in services:
            self.local_cache[service.agent_id] = service

    def _record_health_failure(self, agent_id: str):
        """Record a health check failure"""
        self.service_health_status[agent_id] = self.service_health_status.get(agent_id, 0) + 1

        if self.service_health_status[agent_id] >= self.unhealthy_threshold:
            logger.warning(
                f"Service {agent_id} marked as unhealthy after {self.unhealthy_threshold} failures"
            )

    def _cleanup_unhealthy_services(self):
        """Remove consistently unhealthy services from cache"""
        unhealthy_agents = [
            agent_id
            for agent_id, failure_count in self.service_health_status.items()
            if failure_count >= self.unhealthy_threshold
        ]

        for agent_id in unhealthy_agents:
            self._remove_from_cache(agent_id)

    def _remove_from_cache(self, agent_id: str):
        """Remove service from cache"""
        # Remove individual service
        if agent_id in self.local_cache:
            del self.local_cache[agent_id]

        # Remove from cached service lists
        keys_to_remove = []
        for key, cached_data in self.local_cache.items():
            if isinstance(cached_data, dict) and "services" in cached_data:
                cached_data["services"] = [
                    s for s in cached_data["services"] if s.agent_id != agent_id
                ]
                if not cached_data["services"]:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.local_cache[key]


# Singleton instance
service_discovery = ServiceDiscovery()


async def discover_qa_agents() -> List[ServiceEndpoint]:
    """Discover QA validation agents in the network"""
    return await service_discovery.discover_service(
        service_type="qa_validation", capabilities=["question_analysis", "validation"]
    )


async def discover_data_managers() -> List[ServiceEndpoint]:
    """Discover data manager agents in the network"""
    return await service_discovery.discover_service(
        service_type="data_manager", capabilities=["data_retrieval", "search"]
    )


async def discover_reasoning_engines() -> List[ServiceEndpoint]:
    """Discover reasoning engine agents in the network"""
    return await service_discovery.discover_service(
        service_type="reasoning_engine", capabilities=["logical_reasoning", "inference"]
    )


async def discover_synthesis_agents() -> List[ServiceEndpoint]:
    """Discover answer synthesis agents in the network"""
    return await service_discovery.discover_service(
        service_type="answer_synthesis", capabilities=["synthesis", "aggregation"]
    )

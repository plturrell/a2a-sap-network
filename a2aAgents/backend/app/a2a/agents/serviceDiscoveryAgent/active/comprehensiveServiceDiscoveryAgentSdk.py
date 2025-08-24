"""
Comprehensive Service Discovery Agent SDK - Agent 17
Dynamic service registry and agent discovery system
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
from collections import defaultdict
import sqlite3
import aiosqlite

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
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    OFFLINE = "offline"

class DiscoveryProtocol(Enum):
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    REGISTRY = "registry"
    PEER_TO_PEER = "peer_to_peer"
    HYBRID = "hybrid"

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    HEALTH_BASED = "health_based"
    GEOGRAPHIC = "geographic"

@dataclass
class ServiceEndpoint:
    """Individual service endpoint information"""
    id: str
    url: str
    protocol: str = "a2a"  # A2A blockchain protocol only
    port: int = 0  # A2A Protocol: No ports - blockchain messaging only
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceRegistration:
    """Complete service registration information"""
    service_id: str
    agent_id: str
    service_name: str
    service_type: str
    version: str
    endpoints: List[ServiceEndpoint]
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    health_check_url: Optional[str] = None
    health_check_interval: int = 30  # seconds
    tags: List[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 300  # Time to live
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceQuery:
    """Service discovery query"""
    service_name: Optional[str] = None
    service_type: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: Optional[ServiceStatus] = None
    location: Optional[str] = None
    max_response_time: Optional[float] = None
    min_success_rate: Optional[float] = None

@dataclass
class HealthCheckResult:
    """Health check result"""
    service_id: str
    endpoint_id: str
    status: ServiceStatus
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class ServiceDiscoveryAgentSdk(SecureA2AAgent,
    PerformanceMonitorMixin,
    SecurityHardenedMixin,
    TelemetryMixin
):
    """
    Comprehensive Service Discovery Agent for dynamic service registry and agent discovery
    """
    
    def __init__(self):
        # Create agent config
        from app.a2a.sdk.types import AgentConfig
        config = AgentConfig(
            agent_id=create_agent_id("service-discovery-agent"),
            name="Service Discovery Agent",
            description="Dynamic service registry and agent discovery system",
            version="1.0.0"
        )
        
        super().__init__(config)
        
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        # Initialize AI Intelligence Framework
        self.ai_framework = create_ai_intelligence_framework(
            create_enhanced_agent_config("service_discovery")
        )
        
        # Service registry
        self.service_registry: Dict[str, ServiceRegistration] = {}
        self.service_index: Dict[str, Set[str]] = defaultdict(set)  # For fast lookups
        
        # Health monitoring
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        
        # Load balancing
        self.load_balancer_state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Discovery protocols
        self.discovery_protocols: Set[DiscoveryProtocol] = {DiscoveryProtocol.REGISTRY}
        
        # Circuit breakers for service health checks
        self.health_check_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        
        # Database for persistence
        self.db_path = "service_discovery.db"
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
        
        logger.info("ServiceDiscoveryAgent initialized")

    async def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS service_registrations (
                    service_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    service_name TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    capabilities TEXT,
                    dependencies TEXT,
                    status TEXT NOT NULL,
                    health_check_url TEXT,
                    health_check_interval INTEGER,
                    tags TEXT,
                    registered_at TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    ttl_seconds INTEGER,
                    metadata TEXT,
                    endpoints TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    endpoint_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    error_message TEXT
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_service_name ON service_registrations(service_name);
                CREATE INDEX IF NOT EXISTS idx_service_type ON service_registrations(service_type);
                CREATE INDEX IF NOT EXISTS idx_status ON service_registrations(status);
                CREATE INDEX IF NOT EXISTS idx_health_service ON health_history(service_id);
            """)
            
            await db.commit()

    @a2a_skill(
        name="service_registration",
        description="Register and manage service endpoints",
        version="1.0.0"
    )
    @mcp_tool(
        name="register_service",
        description="Register a new service in the discovery registry"
    )
    async def register_service(
        self,
        agent_id: str,
        service_name: str,
        service_type: str,
        endpoints: List[Dict[str, Any]],
        version: str = "1.0.0",
        capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        health_check_url: Optional[str] = None,
        health_check_interval: int = 30,
        tags: Optional[List[str]] = None,
        ttl_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new service in the discovery registry
        """
        try:
            service_id = str(uuid.uuid4())
            
            # Convert endpoint dictionaries to ServiceEndpoint objects
            service_endpoints = []
            for ep_data in endpoints:
                endpoint = ServiceEndpoint(
                    id=ep_data.get("id", str(uuid.uuid4())),
                    url=ep_data["url"],
                    protocol=ep_data.get("protocol", "http"),
                    port=ep_data.get("port", 8000),
                    weight=ep_data.get("weight", 1.0),
                    max_connections=ep_data.get("max_connections", 100),
                    metadata=ep_data.get("metadata", {})
                )
                service_endpoints.append(endpoint)
            
            # Create service registration
            registration = ServiceRegistration(
                service_id=service_id,
                agent_id=agent_id,
                service_name=service_name,
                service_type=service_type,
                version=version,
                endpoints=service_endpoints,
                capabilities=capabilities or [],
                dependencies=dependencies or [],
                health_check_url=health_check_url,
                health_check_interval=health_check_interval,
                tags=tags or [],
                ttl_seconds=ttl_seconds,
                metadata=metadata or {}
            )
            
            # Store in registry
            self.service_registry[service_id] = registration
            
            # Update indexes
            self._update_service_index(registration)
            
            # Persist to database
            await self._persist_service_registration(registration)
            
            # Start health monitoring if health check URL provided
            if health_check_url:
                await self._start_health_monitoring(service_id)
            
            logger.info(f"Registered service: {service_name} ({service_id}) for agent {agent_id}")
            
            return {
                "service_id": service_id,
                "status": "registered",
                "endpoints_count": len(service_endpoints),
                "health_monitoring": health_check_url is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            raise

    @a2a_skill(
        name="service_discovery",
        description="Discover and query available services",
        version="1.0.0"
    )
    @mcp_tool(
        name="discover_services",
        description="Discover services based on query criteria"
    )
    async def discover_services(
        self,
        service_name: Optional[str] = None,
        service_type: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        include_unhealthy: bool = False,
        max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Discover services based on query criteria
        """
        try:
            query = ServiceQuery(
                service_name=service_name,
                service_type=service_type,
                capabilities=capabilities or [],
                tags=tags or [],
                status=ServiceStatus(status) if status else None
            )
            
            # Find matching services
            matching_services = await self._query_services(query, include_unhealthy)
            
            # Limit results
            if len(matching_services) > max_results:
                matching_services = matching_services[:max_results]
            
            # Convert to response format
            results = []
            for registration in matching_services:
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
                            "weight": ep.weight,
                            "response_time_ms": ep.response_time_ms,
                            "success_rate": ep.success_rate
                        }
                        for ep in registration.endpoints
                    ],
                    "metadata": registration.metadata
                }
                results.append(service_info)
            
            logger.info(f"Discovered {len(results)} services matching query")
            
            return {
                "services": results,
                "total_found": len(results),
                "query": {
                    "service_name": service_name,
                    "service_type": service_type,
                    "capabilities": capabilities,
                    "tags": tags,
                    "status": status
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            raise

    @a2a_skill(
        name="load_balancing",
        description="Provide load balancing for service endpoints",
        version="1.0.0"
    )
    @mcp_tool(
        name="get_service_endpoint",
        description="Get optimal service endpoint using load balancing"
    )
    async def get_service_endpoint(
        self,
        service_name: str,
        strategy: str = "health_based",
        exclude_unhealthy: bool = True,
        client_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimal service endpoint using load balancing strategy
        """
        try:
            # Find services by name
            matching_services = await self._query_services(
                ServiceQuery(service_name=service_name),
                include_unhealthy=not exclude_unhealthy
            )
            
            if not matching_services:
                raise ValueError(f"No services found with name: {service_name}")
            
            # Collect all endpoints from matching services
            all_endpoints = []
            for service in matching_services:
                for endpoint in service.endpoints:
                    all_endpoints.append((service, endpoint))
            
            if not all_endpoints:
                raise ValueError(f"No endpoints available for service: {service_name}")
            
            # Apply load balancing strategy
            selected_service, selected_endpoint = await self._apply_load_balancing(
                all_endpoints, LoadBalancingStrategy(strategy), client_location
            )
            
            # Update connection count
            selected_endpoint.current_connections += 1
            
            logger.info(f"Selected endpoint {selected_endpoint.id} for service {service_name}")
            
            return {
                "service_id": selected_service.service_id,
                "endpoint": {
                    "id": selected_endpoint.id,
                    "url": selected_endpoint.url,
                    "protocol": selected_endpoint.protocol,
                    "port": selected_endpoint.port,
                    "response_time_ms": selected_endpoint.response_time_ms,
                    "success_rate": selected_endpoint.success_rate
                },
                "strategy_used": strategy,
                "total_available": len(all_endpoints)
            }
            
        except Exception as e:
            logger.error(f"Failed to get service endpoint: {e}")
            raise

    @a2a_skill(
        name="health_monitoring",
        description="Monitor service health and availability",
        version="1.0.0"
    )
    @mcp_tool(
        name="get_service_health",
        description="Get health status and metrics for services"
    )
    async def get_service_health(
        self,
        service_id: Optional[str] = None,
        service_name: Optional[str] = None,
        include_history: bool = False,
        history_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get health status and metrics for services
        """
        try:
            if service_id:
                services = [self.service_registry.get(service_id)]
                services = [s for s in services if s is not None]
            elif service_name:
                services = [
                    s for s in self.service_registry.values()
                    if s.service_name == service_name
                ]
            else:
                services = list(self.service_registry.values())
            
            results = []
            for service in services:
                health_info = {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "agent_id": service.agent_id,
                    "status": service.status.value,
                    "last_heartbeat": service.last_heartbeat.isoformat(),
                    "endpoints": []
                }
                
                # Add endpoint health information
                for endpoint in service.endpoints:
                    endpoint_health = {
                        "endpoint_id": endpoint.id,
                        "url": endpoint.url,
                        "response_time_ms": endpoint.response_time_ms,
                        "success_rate": endpoint.success_rate,
                        "current_connections": endpoint.current_connections,
                        "max_connections": endpoint.max_connections,
                        "last_health_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
                    }
                    health_info["endpoints"].append(endpoint_health)
                
                # Add history if requested
                if include_history:
                    since = datetime.now() - timedelta(hours=history_hours)
                    history = [
                        {
                            "timestamp": h.timestamp.isoformat(),
                            "status": h.status.value,
                            "response_time_ms": h.response_time_ms,
                            "endpoint_id": h.endpoint_id,
                            "error_message": h.error_message
                        }
                        for h in self.health_history[service.service_id]
                        if h.timestamp >= since
                    ]
                    health_info["history"] = history
                
                results.append(health_info)
            
            return {
                "services": results,
                "total_services": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service health: {e}")
            raise

    @a2a_skill(
        name="service_lifecycle",
        description="Manage service lifecycle operations",
        version="1.0.0"
    )
    @mcp_tool(
        name="deregister_service",
        description="Deregister a service from the discovery registry"
    )
    async def deregister_service(
        self,
        service_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Deregister a service from the discovery registry
        """
        try:
            if service_id not in self.service_registry:
                raise ValueError(f"Service {service_id} not found")
            
            registration = self.service_registry[service_id]
            
            # Verify ownership
            if registration.agent_id != agent_id:
                raise ValueError(f"Agent {agent_id} not authorized to deregister service {service_id}")
            
            # Stop health monitoring
            if service_id in self.health_check_tasks:
                self.health_check_tasks[service_id].cancel()
                del self.health_check_tasks[service_id]
            
            # Remove from registry and indexes
            del self.service_registry[service_id]
            self._remove_from_service_index(registration)
            
            # Remove from database
            await self._remove_service_from_database(service_id)
            
            logger.info(f"Deregistered service: {registration.service_name} ({service_id})")
            
            return {
                "service_id": service_id,
                "status": "deregistered",
                "service_name": registration.service_name
            }
            
        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            raise

    @a2a_skill(
        name="heartbeat_management",
        description="Manage service heartbeats and TTL",
        version="1.0.0"
    )
    @mcp_tool(
        name="send_heartbeat",
        description="Send heartbeat to maintain service registration"
    )
    async def send_heartbeat(
        self,
        service_id: str,
        agent_id: str,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send heartbeat to maintain service registration
        """
        try:
            if service_id not in self.service_registry:
                raise ValueError(f"Service {service_id} not found")
            
            registration = self.service_registry[service_id]
            
            # Verify ownership
            if registration.agent_id != agent_id:
                raise ValueError(f"Agent {agent_id} not authorized to send heartbeat for service {service_id}")
            
            # Update heartbeat
            registration.last_heartbeat = datetime.now()
            
            # Update status if provided
            if status:
                registration.status = ServiceStatus(status)
            
            # Update metadata if provided
            if metadata:
                registration.metadata.update(metadata)
            
            # Persist update
            await self._persist_service_registration(registration)
            
            logger.debug(f"Heartbeat received for service: {registration.service_name} ({service_id})")
            
            return {
                "service_id": service_id,
                "status": "heartbeat_received",
                "next_heartbeat_due": (datetime.now() + timedelta(seconds=registration.ttl_seconds)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process heartbeat: {e}")
            raise

    async def _query_services(
        self,
        query: ServiceQuery,
        include_unhealthy: bool = False
    ) -> List[ServiceRegistration]:
        """
        Query services based on criteria
        """
        matching_services = []
        
        for registration in self.service_registry.values():
            # Skip unhealthy services if requested
            if not include_unhealthy and registration.status != ServiceStatus.HEALTHY:
                continue
            
            # Apply filters
            if query.service_name and registration.service_name != query.service_name:
                continue
            
            if query.service_type and registration.service_type != query.service_type:
                continue
            
            if query.status and registration.status != query.status:
                continue
            
            # Check capabilities
            if query.capabilities:
                if not all(cap in registration.capabilities for cap in query.capabilities):
                    continue
            
            # Check tags
            if query.tags:
                if not all(tag in registration.tags for tag in query.tags):
                    continue
            
            matching_services.append(registration)
        
        return matching_services

    async def _apply_load_balancing(
        self,
        endpoints: List[Tuple[ServiceRegistration, ServiceEndpoint]],
        strategy: LoadBalancingStrategy,
        client_location: Optional[str] = None
    ) -> Tuple[ServiceRegistration, ServiceEndpoint]:
        """
        Apply load balancing strategy to select endpoint
        """
        if not endpoints:
            raise ValueError("No endpoints available")
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            service_name = endpoints[0][0].service_name
            current_index = self.load_balancer_state[service_name].get("round_robin_index", 0)
            selected = endpoints[current_index % len(endpoints)]
            self.load_balancer_state[service_name]["round_robin_index"] = (current_index + 1) % len(endpoints)
            return selected
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select endpoint with least current connections
            return min(endpoints, key=lambda x: x[1].current_connections)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED:
            # Weighted selection based on endpoint weight
            total_weight = sum(ep[1].weight for ep in endpoints)
            import random
            r = random.uniform(0, total_weight)
            cumulative = 0
            for service, endpoint in endpoints:
                cumulative += endpoint.weight
                if r <= cumulative:
                    return service, endpoint
            return endpoints[-1]  # Fallback
        
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            # Select based on health metrics (response time and success rate)
            def health_score(ep):
                # Lower response time and higher success rate = better score
                return ep[1].success_rate / (ep[1].response_time_ms + 1)
            
            return max(endpoints, key=health_score)
        
        else:
            # Default to first available
            return endpoints[0]

    async def _start_health_monitoring(self, service_id: str):
        """
        Start health monitoring for a service
        """
        if service_id in self.health_check_tasks:
            return  # Already monitoring
        
        registration = self.service_registry.get(service_id)
        if not registration or not registration.health_check_url:
            return
        
        # Create health check task
        task = asyncio.create_task(
            self._health_check_loop(service_id)
        )
        self.health_check_tasks[service_id] = task

    async def _health_check_loop(self, service_id: str):
        """
        Continuous health check loop for a service
        """
        while service_id in self.service_registry:
            try:
                registration = self.service_registry[service_id]
                
                # Perform health checks on all endpoints
                for endpoint in registration.endpoints:
                    health_result = await self._perform_health_check(registration, endpoint)
                    
                    # Update endpoint metrics
                    endpoint.response_time_ms = health_result.response_time_ms
                    endpoint.last_health_check = health_result.timestamp
                    
                    # Update success rate (rolling average)
                    if health_result.status == ServiceStatus.HEALTHY:
                        endpoint.success_rate = min(1.0, endpoint.success_rate * 0.9 + 0.1)
                    else:
                        endpoint.success_rate = max(0.0, endpoint.success_rate * 0.9)
                    
                    # Store health history
                    self.health_history[service_id].append(health_result)
                    
                    # Limit history size
                    if len(self.health_history[service_id]) > 1000:
                        self.health_history[service_id] = self.health_history[service_id][-500:]
                
                # Update overall service status
                healthy_endpoints = sum(
                    1 for ep in registration.endpoints
                    if ep.success_rate > 0.5
                )
                
                if healthy_endpoints == 0:
                    registration.status = ServiceStatus.UNHEALTHY
                elif healthy_endpoints == len(registration.endpoints):
                    registration.status = ServiceStatus.HEALTHY
                else:
                    registration.status = ServiceStatus.DEGRADED
                
                # Wait for next check
                await asyncio.sleep(registration.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error for service {service_id}: {e}")
                await asyncio.sleep(10)  # Shorter retry interval on error

    async def _perform_health_check(
        self,
        registration: ServiceRegistration,
        endpoint: ServiceEndpoint
    ) -> HealthCheckResult:
        """
        Perform health check on a specific endpoint
        """
        start_time = time.time()
        
        try:
            # Get or create circuit breaker for this endpoint
            breaker_key = f"{registration.service_id}_{endpoint.id}"
            if breaker_key not in self.health_check_breakers:
                self.health_check_breakers[breaker_key] = EnhancedCircuitBreaker(
                    name=breaker_key,
                    failure_threshold=3,
                    recovery_timeout=30
                )
            
            circuit_breaker = self.health_check_breakers[breaker_key]
            
            # Perform health check through circuit breaker
            async def health_check_call():
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    health_url = registration.health_check_url or f"{endpoint.url}/health"
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise Exception(f"Health check failed with status {response.status}")
            
            result = await circuit_breaker.call(health_check_call)
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service_id=registration.service_id,
                endpoint_id=endpoint.id,
                status=ServiceStatus.HEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=result if isinstance(result, dict) else {}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service_id=registration.service_id,
                endpoint_id=endpoint.id,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    def _update_service_index(self, registration: ServiceRegistration):
        """
        Update service indexes for fast lookups
        """
        service_id = registration.service_id
        
        # Index by service name
        self.service_index[f"name:{registration.service_name}"].add(service_id)
        
        # Index by service type
        self.service_index[f"type:{registration.service_type}"].add(service_id)
        
        # Index by capabilities
        for capability in registration.capabilities:
            self.service_index[f"capability:{capability}"].add(service_id)
        
        # Index by tags
        for tag in registration.tags:
            self.service_index[f"tag:{tag}"].add(service_id)

    def _remove_from_service_index(self, registration: ServiceRegistration):
        """
        Remove service from indexes
        """
        service_id = registration.service_id
        
        # Remove from all indexes
        for index_set in self.service_index.values():
            index_set.discard(service_id)

    async def _persist_service_registration(self, registration: ServiceRegistration):
        """
        Persist service registration to database
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO service_registrations 
                (service_id, agent_id, service_name, service_type, version, capabilities,
                 dependencies, status, health_check_url, health_check_interval, tags,
                 registered_at, last_heartbeat, ttl_seconds, metadata, endpoints)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                registration.service_id,
                registration.agent_id,
                registration.service_name,
                registration.service_type,
                registration.version,
                json.dumps(registration.capabilities),
                json.dumps(registration.dependencies),
                registration.status.value,
                registration.health_check_url,
                registration.health_check_interval,
                json.dumps(registration.tags),
                registration.registered_at.isoformat(),
                registration.last_heartbeat.isoformat(),
                registration.ttl_seconds,
                json.dumps(registration.metadata),
                json.dumps([{
                    "id": ep.id,
                    "url": ep.url,
                    "protocol": ep.protocol,
                    "port": ep.port,
                    "weight": ep.weight,
                    "max_connections": ep.max_connections,
                    "metadata": ep.metadata
                } for ep in registration.endpoints])
            ))
            await db.commit()

    async def _remove_service_from_database(self, service_id: str):
        """
        Remove service from database
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM service_registrations WHERE service_id = ?", (service_id,))
            await db.execute("DELETE FROM health_history WHERE service_id = ?", (service_id,))
            await db.commit()

    async def cleanup_expired_services(self):
        """
        Clean up expired services based on TTL
        """
        current_time = datetime.now()
        expired_services = []
        
        for service_id, registration in self.service_registry.items():
            time_since_heartbeat = (current_time - registration.last_heartbeat).total_seconds()
            if time_since_heartbeat > registration.ttl_seconds:
                expired_services.append(service_id)
        
        for service_id in expired_services:
            registration = self.service_registry[service_id]
            logger.info(f"Cleaning up expired service: {registration.service_name} ({service_id})")
            
            # Stop health monitoring
            if service_id in self.health_check_tasks:
                self.health_check_tasks[service_id].cancel()
                del self.health_check_tasks[service_id]
            
            # Remove from registry
            del self.service_registry[service_id]
            self._remove_from_service_index(registration)
            
            # Remove from database
            await self._remove_service_from_database(service_id)

# Create singleton instance
service_discovery_agent = ServiceDiscoveryAgentSdk()

def get_service_discovery_agent() -> ServiceDiscoveryAgentSdk:
    """Get the singleton service discovery agent instance"""
    return service_discovery_agent

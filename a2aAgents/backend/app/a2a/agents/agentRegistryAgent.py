"""
Agent Registry Agent - Central registry for agent discovery and management
Handles agent registration, discovery requests, and network topology management
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from ..sdk.agentBase import A2AAgentBase, AgentConfig
from ..sdk import a2a_handler, a2a_skill
from ..sdk.types import A2AMessage, MessageRole
from ..core.agentDiscovery import AgentProfile, DiscoveryRequest, DiscoveryResult
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)


class AgentRegistryAgent(SecureA2AAgent):
    """
    Central Agent Registry for the A2A Network
    Manages agent registration, discovery, and network health monitoring
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Security features are initialized by SecureA2AAgent base class


        # Agent registry storage
        self.registered_agents: Dict[str, AgentProfile] = {}
        self.agent_health_status: Dict[str, Dict[str, Any]] = {}
        self.registration_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.discovery_requests: List[Dict[str, Any]] = []
        self.network_statistics = {
            "total_agents": 0,
            "active_agents": 0,
            "discovery_requests_24h": 0,
            "avg_response_time": 0.0
        }

        # Health monitoring
        self.health_check_interval = 300  # 5 minutes
        self.health_check_task = None

        logger.info(f"Agent Registry Agent initialized: {self.agent_id}")

    async def start(self):
        """Start the agent registry services"""
        await super().start()

        # Start health monitoring
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())

        logger.info("🏢 Agent Registry Agent started - ready to manage network")

    async def stop(self):
        """Stop the agent registry services"""
        if self.health_check_task:
            self.health_check_task.cancel()

        await super().stop()
        logger.info("Agent Registry Agent stopped")

    @a2a_handler("AGENT_REGISTRATION")
    async def handle_agent_registration(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent registration requests"""
        try:
            # Extract registration data
            registration_data = None
            for part in message.parts:
                if part.kind == "data" and part.data and "agent_profile" in part.data:
                    registration_data = part.data
                    break

            if not registration_data:
                return {"status": "error", "error": "No registration data provided"}

            # Create agent profile
            profile_data = registration_data["agent_profile"]
            agent_profile = AgentProfile(
                agent_id=profile_data["agent_id"],
                name=profile_data["name"],
                endpoint=profile_data["endpoint"],
                capabilities=profile_data["capabilities"],
                performance_metrics=profile_data.get("performance_metrics", {}),
                trust_score=profile_data.get("trust_score", 0.8),
                reputation=profile_data.get("reputation", 100),
                specializations=profile_data.get("specializations", []),
                metadata=profile_data.get("metadata", {}),
                blockchain_address=profile_data.get("blockchain_address"),
                version=profile_data.get("version", "1.0.0")
            )

            # Register the agent
            self.registered_agents[agent_profile.agent_id] = agent_profile

            # Initialize health status
            self.agent_health_status[agent_profile.agent_id] = {
                "status": "unknown",
                "last_check": datetime.utcnow(),
                "response_time": 0.0,
                "consecutive_failures": 0
            }

            # Record registration
            registration_record = {
                "agent_id": agent_profile.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "capabilities": agent_profile.capabilities,
                "endpoint": agent_profile.endpoint
            }
            self.registration_history.append(registration_record)

            # Update network statistics
            self._update_network_statistics()

            logger.info(f"✅ Registered agent: {agent_profile.agent_id} with {len(agent_profile.capabilities)} capabilities")

            return {
                "status": "success",
                "agent_id": agent_profile.agent_id,
                "registered_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return {"status": "error", "error": str(e)}

    @a2a_handler("CAPABILITY_REGISTRATION")
    async def handle_capability_registration(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle capability registration/update requests"""
        return await self.handle_agent_registration(message, context_id)

    @a2a_handler("AGENT_REGISTRY_QUERY")
    async def handle_registry_query(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle registry queries for agent lists"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Extract query parameters
            query_data = None
            for part in message.parts:
                if part.kind == "data" and part.data and "operation" in part.data:
                    query_data = part.data
                    break

            if not query_data:
                return {"status": "error", "error": "No query data provided"}

            operation = query_data.get("operation")

            if operation == "list_active_agents":
                scope = query_data.get("scope", "local_network")
                include_performance = query_data.get("include_performance", False)
                include_reputation = query_data.get("include_reputation", False)

                agents_data = []
                for agent_profile in self.registered_agents.values():
                    agent_data = {
                        "agent_id": agent_profile.agent_id,
                        "name": agent_profile.name,
                        "endpoint": agent_profile.endpoint,
                        "capabilities": agent_profile.capabilities,
                        "last_seen": agent_profile.last_seen.isoformat(),
                        "availability_score": agent_profile.availability_score,
                        "specializations": agent_profile.specializations,
                        "metadata": agent_profile.metadata,
                        "blockchain_address": agent_profile.blockchain_address,
                        "version": agent_profile.version
                    }

                    if include_performance:
                        agent_data["performance_metrics"] = agent_profile.performance_metrics
                        agent_data["response_time_avg"] = agent_profile.response_time_avg
                        agent_data["success_rate"] = agent_profile.success_rate

                    if include_reputation:
                        agent_data["trust_score"] = agent_profile.trust_score
                        agent_data["reputation"] = agent_profile.reputation

                    agents_data.append(agent_data)

                response_time = asyncio.get_event_loop().time() - start_time

                # Record discovery request
                self.discovery_requests.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "operation": operation,
                    "scope": scope,
                    "agents_returned": len(agents_data),
                    "response_time": response_time
                })

                logger.info(f"🔍 Registry query processed: {len(agents_data)} agents returned in {response_time:.3f}s")

                return {
                    "status": "success",
                    "agents": agents_data,
                    "total_count": len(agents_data),
                    "response_time": response_time
                }

            else:
                return {"status": "error", "error": f"Unknown operation: {operation}"}

        except Exception as e:
            logger.error(f"Registry query failed: {e}")
            return {"status": "error", "error": str(e)}

    @a2a_handler("SERVICE_DISCOVERY")
    async def handle_service_discovery(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle service discovery requests"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Extract discovery request
            discovery_data = None
            for part in message.parts:
                if part.kind == "data" and part.data and "operation" in part.data:
                    discovery_data = part.data
                    break

            if not discovery_data:
                return {"status": "error", "error": "No discovery data provided"}

            if discovery_data.get("operation") == "discover_agents":
                service_type = discovery_data.get("service_type")
                capabilities = discovery_data.get("capabilities", [])
                active_only = discovery_data.get("active_only", True)

                # Filter agents based on criteria
                matching_agents = []

                for agent_profile in self.registered_agents.values():
                    # Check if agent has required capabilities
                    if capabilities and not all(cap in agent_profile.capabilities for cap in capabilities):
                        continue

                    # Check if agent is active (if required)
                    if active_only:
                        health_info = self.agent_health_status.get(agent_profile.agent_id, {})
                        if health_info.get("status") == "unhealthy":
                            continue

                    # Check service type matching (if specified)
                    if service_type:
                        agent_type = agent_profile.metadata.get("agent_type", "").lower()
                        if service_type.lower() not in agent_type and service_type not in agent_profile.capabilities:
                            continue

                    matching_agents.append({
                        "agent_id": agent_profile.agent_id,
                        "name": agent_profile.name,
                        "endpoint": agent_profile.endpoint,
                        "capabilities": agent_profile.capabilities,
                        "trust_score": agent_profile.trust_score,
                        "availability_score": agent_profile.availability_score,
                        "metadata": agent_profile.metadata
                    })

                response_time = asyncio.get_event_loop().time() - start_time

                logger.info(f"🎯 Service discovery: {len(matching_agents)} agents found for {service_type} in {response_time:.3f}s")

                return {
                    "status": "success",
                    "agents": matching_agents,
                    "total_found": len(matching_agents),
                    "response_time": response_time
                }

            else:
                return {"status": "error", "error": "Unknown discovery operation"}

        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return {"status": "error", "error": str(e)}

    @a2a_handler("AGENT_DEREGISTRATION")
    async def handle_agent_deregistration(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent deregistration requests"""
        try:
            # Extract deregistration data
            deregistration_data = None
            for part in message.parts:
                if part.kind == "data" and part.data and "agent_id" in part.data:
                    deregistration_data = part.data
                    break

            if not deregistration_data:
                return {"status": "error", "error": "No deregistration data provided"}

            agent_id = deregistration_data["agent_id"]

            # Remove agent from registry
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]

            # Remove health status
            if agent_id in self.agent_health_status:
                del self.agent_health_status[agent_id]

            # Update network statistics
            self._update_network_statistics()

            logger.info(f"✅ Deregistered agent: {agent_id}")

            return {
                "status": "success",
                "agent_id": agent_id,
                "deregistered_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent deregistration failed: {e}")
            return {"status": "error", "error": str(e)}

    @a2a_handler("PERFORMANCE_UPDATE")
    async def handle_performance_update(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent performance metric updates"""
        try:
            # Extract performance data
            performance_data = None
            for part in message.parts:
                if part.kind == "data" and part.data and "agent_id" in part.data:
                    performance_data = part.data
                    break

            if not performance_data:
                return {"status": "error", "error": "No performance data provided"}

            agent_id = performance_data["agent_id"]
            metrics = performance_data.get("metrics", {})

            # Update agent performance metrics
            if agent_id in self.registered_agents:
                agent_profile = self.registered_agents[agent_id]
                agent_profile.performance_metrics.update(metrics)
                agent_profile.last_seen = datetime.utcnow()

                # Update derived metrics
                if "success_rate" in metrics:
                    agent_profile.success_rate = metrics["success_rate"]
                if "avg_processing_time" in metrics:
                    agent_profile.response_time_avg = metrics["avg_processing_time"]
                if "uptime" in metrics:
                    agent_profile.availability_score = metrics["uptime"]

                logger.debug(f"📊 Updated performance metrics for {agent_id}")

                return {"status": "success", "agent_id": agent_id}
            else:
                return {"status": "error", "error": f"Agent {agent_id} not found in registry"}

        except Exception as e:
            logger.error(f"Performance update failed: {e}")
            return {"status": "error", "error": str(e)}

    @a2a_skill("network_statistics")
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        # Update statistics
        self._update_network_statistics()

        # Calculate additional insights
        capabilities_distribution = {}
        trust_levels = {"high": 0, "medium": 0, "low": 0}

        for agent_profile in self.registered_agents.values():
            # Capability distribution
            for cap in agent_profile.capabilities:
                capabilities_distribution[cap] = capabilities_distribution.get(cap, 0) + 1

            # Trust level distribution
            if agent_profile.trust_score >= 0.8:
                trust_levels["high"] += 1
            elif agent_profile.trust_score >= 0.5:
                trust_levels["medium"] += 1
            else:
                trust_levels["low"] += 1

        # Recent activity
        recent_discoveries = [
            req for req in self.discovery_requests
            if datetime.fromisoformat(req["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
        ]

        return {
            "network_health": "excellent" if trust_levels["high"] >= len(self.registered_agents) * 0.7 else "good",
            "statistics": self.network_statistics,
            "capabilities_distribution": capabilities_distribution,
            "trust_distribution": trust_levels,
            "recent_activity": {
                "discoveries_24h": len(recent_discoveries),
                "avg_discovery_time": sum(req["response_time"] for req in recent_discoveries) / max(len(recent_discoveries), 1),
                "registrations_24h": len([
                    reg for reg in self.registration_history
                    if datetime.fromisoformat(reg["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
                ])
            }
        }

    def _update_network_statistics(self):
        """Update network statistics"""
        self.network_statistics.update({
            "total_agents": len(self.registered_agents),
            "active_agents": len([
                agent_id for agent_id, health in self.agent_health_status.items()
                if health.get("status") != "unhealthy"
            ]),
            "discovery_requests_24h": len([
                req for req in self.discovery_requests
                if datetime.fromisoformat(req["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
            ])
        })

        # Calculate average response time
        recent_requests = [
            req for req in self.discovery_requests
            if datetime.fromisoformat(req["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
        ]

        if recent_requests:
            self.network_statistics["avg_response_time"] = sum(
                req["response_time"] for req in recent_requests
            ) / len(recent_requests)

    async def _health_monitoring_loop(self):
        """Background health monitoring of registered agents"""
        logger.info("🔍 Starting agent health monitoring loop")

        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check health of all registered agents
                for agent_id, agent_profile in self.registered_agents.items():
                    await self._check_agent_health(agent_id, agent_profile)

                # Clean up old discovery requests (keep last 1000)
                if len(self.discovery_requests) > 1000:
                    self.discovery_requests = self.discovery_requests[-1000:]

                # Clean up old registration history (keep last 500)
                if len(self.registration_history) > 500:
                    self.registration_history = self.registration_history[-500:]

                logger.debug(f"Health monitoring cycle complete: {len(self.registered_agents)} agents checked")

            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _check_agent_health(self, agent_id: str, agent_profile: AgentProfile):
        """Check health of a specific agent"""
        try:
            start_time = asyncio.get_event_loop().time()

            # A2A Protocol Compliance: Send health check via blockchain messaging
            from ..core.networkClient import A2ANetworkClient

            network_client = A2ANetworkClient(agent_id=self.agent_id)

            health_request_data = {
                "operation": "health_check",
                "timestamp": datetime.utcnow().isoformat(),
                "from_registry": True,
                "registry_id": self.agent_id
            }

            try:
                # Send health check via A2A blockchain messaging
                response = await network_client.send_a2a_message(
                    to_agent=agent_id,
                    message=health_request_data,
                    message_type="HEALTH_CHECK"
                )

                health_info = self.agent_health_status[agent_id]
                response_time = asyncio.get_event_loop().time() - start_time

                if response and response.get("status") == "healthy":
                    # Agent responded and is healthy
                    health_info["status"] = "healthy"
                    health_info["consecutive_failures"] = 0
                    agent_profile.availability_score = min(agent_profile.availability_score + 0.01, 1.0)
                    agent_profile.last_seen = datetime.utcnow()
                    agent_profile.response_time_avg = (agent_profile.response_time_avg + response_time) / 2
                else:
                    # Agent didn't respond properly
                    health_info["consecutive_failures"] += 1

                    if health_info["consecutive_failures"] >= 3:
                        health_info["status"] = "unhealthy"
                        agent_profile.availability_score = max(agent_profile.availability_score - 0.1, 0.0)
                    else:
                        health_info["status"] = "degraded"

                health_info["last_check"] = datetime.utcnow()
                health_info["response_time"] = response_time

            except Exception as health_error:
                # Health check failed due to communication error
                health_info = self.agent_health_status[agent_id]
                health_info["consecutive_failures"] += 1
                health_info["status"] = "unhealthy" if health_info["consecutive_failures"] >= 3 else "degraded"
                health_info["last_check"] = datetime.utcnow()
                health_info["response_time"] = asyncio.get_event_loop().time() - start_time

                logger.warning(f"Health check communication failed for {agent_id}: {health_error}")

        except Exception as e:
            logger.warning(f"Health check failed for {agent_id}: {e}")
            health_info = self.agent_health_status[agent_id]
            health_info["consecutive_failures"] += 1
            health_info["status"] = "unhealthy" if health_info["consecutive_failures"] >= 3 else "degraded"


# Factory function for creating the agent registry
async def create_agent_registry(
    agent_id: str = "agent_registry",
    name: str = "Agent Registry",
    base_url: str = None
) -> AgentRegistryAgent:
    """Create and initialize an Agent Registry Agent"""

    # A2A Protocol Compliance: No hardcoded defaults - require explicit configuration
    if not base_url:
        raise ValueError("base_url is required for Agent Registry - no localhost fallbacks allowed")

    config = AgentConfig(
        agent_id=agent_id,
        name=name,
        description="Central registry for agent discovery and network management",
        version="1.0.0",
        base_url=base_url,
        enable_telemetry=True,
        enable_request_signing=False,
        a2a_protocol_only=True,
        blockchain_capabilities=["agent_registry", "service_discovery", "network_management"]
    )

    registry_agent = AgentRegistryAgent(config)

    logger.info(f"✅ Agent Registry created: {agent_id}")
    return registry_agent

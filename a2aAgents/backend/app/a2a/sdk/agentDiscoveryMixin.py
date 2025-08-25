"""
Agent Discovery Mixin for A2A Agents
Provides standardized agent discovery capabilities to all A2A agents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.agentDiscovery import (
    AgentDiscoveryEngine,
    DiscoveryRequest,
    DiscoveryResult,
    AgentProfile,
    AgentCapabilityType,
    DiscoveryScope,
    discover_agents_by_capability,
    find_best_agent_for_task,
    get_network_topology
)

logger = logging.getLogger(__name__)


class AgentDiscoveryMixin:
    """
    Mixin class that adds agent discovery capabilities to any A2A agent
    Provides intelligent agent discovery and network topology insights
    """

    def __init__(self):
        super().__init__()
        self.discovery_engine = None
        self.agent_profile = None
        self.discovery_enabled = True

        # Discovery analytics
        self.discovery_metrics = {
            "total_discoveries": 0,
            "successful_collaborations": 0,
            "failed_discoveries": 0,
            "avg_discovery_time": 0.0
        }

    async def initialize_discovery_engine(self):
        """Initialize the agent discovery engine"""
        if not self.discovery_enabled:
            logger.info("Agent discovery disabled")
            return

        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            # Get blockchain client if available for enhanced discovery
            blockchain_client = getattr(self, 'blockchain_client', None)

            # Initialize discovery engine - let it fail if there's a real issue
            self.discovery_engine = AgentDiscoveryEngine(agent_id)

            # Create agent profile for this agent
            await self._create_agent_profile()

            # Register with blockchain registry
            if self.agent_profile:
                success = await self.discovery_engine.register_agent_capabilities(self.agent_profile)
                if success:
                    logger.info(f"✅ Agent discovery initialized for {agent_id}")
                else:
                    logger.warning(f"⚠️ Failed to register capabilities for {agent_id}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize discovery engine: {e}")
            self.discovery_enabled = False

    async def _create_agent_profile(self):
        """Create comprehensive agent profile for discovery"""
        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            agent_name = getattr(self, 'agent_card', {}).get('name', agent_id)
            endpoint = getattr(self, 'base_url', '') or getattr(self, 'endpoint', '')

            # Extract capabilities from skills
            capabilities = []
            if hasattr(self, 'skills') and self.skills:
                # Handle both dictionary and list formats
                if hasattr(self.skills, 'keys'):
                    capabilities.extend(list(self.skills.keys()))
                elif isinstance(self.skills, list):
                    # For list format, extract names from skill objects
                    for skill in self.skills:
                        skill_name = getattr(skill, 'name', str(skill))
                        capabilities.append(skill_name)

            # Add blockchain capabilities if available
            if hasattr(self, 'blockchain_capabilities') and self.blockchain_capabilities:
                capabilities.extend(self.blockchain_capabilities)

            # Extract specializations from agent metadata
            specializations = []
            if hasattr(self, 'agent_card') and self.agent_card:
                specializations = self.agent_card.get('specializations', [])

            # Performance metrics
            performance_metrics = {
                "uptime": getattr(self, 'uptime_percentage', 0.95),
                "avg_processing_time": getattr(self, 'avg_processing_time', 2.0),
                "success_rate": getattr(self, 'success_rate', 0.9),
                "load_capacity": getattr(self, 'max_concurrent_tasks', 10)
            }

            # Create agent profile
            self.agent_profile = AgentProfile(
                agent_id=agent_id,
                name=agent_name,
                endpoint=endpoint,
                capabilities=list(set(capabilities)),  # Remove duplicates
                performance_metrics=performance_metrics,
                trust_score=getattr(self, 'trust_score', 0.8),
                reputation=getattr(self, 'reputation', 100),
                specializations=specializations,
                metadata={
                    "version": getattr(self, 'version', '1.0.0'),
                    "created_at": datetime.utcnow().isoformat(),
                    "agent_type": self.__class__.__name__,
                    "blockchain_enabled": getattr(self, 'blockchain_enabled', False)
                },
                blockchain_address=getattr(self, 'blockchain_address', None)
            )

            logger.info(f"📝 Created agent profile for {agent_id}: {len(capabilities)} capabilities")

        except Exception as e:
            logger.error(f"Failed to create agent profile: {e}")
            self.agent_profile = None

    async def discover_agents(self, request: DiscoveryRequest) -> List[DiscoveryResult]:
        """
        Discover agents matching the request criteria

        Args:
            request: Discovery request specification

        Returns:
            List of discovery results sorted by relevance
        """
        if not self.discovery_enabled or not self.discovery_engine:
            logger.warning("Agent discovery not available")
            return []

        start_time = asyncio.get_event_loop().time()

        try:
            results = await self.discovery_engine.discover_agents(request)

            # Update metrics
            discovery_time = asyncio.get_event_loop().time() - start_time
            self.discovery_metrics["total_discoveries"] += 1

            if results:
                self.discovery_metrics["successful_collaborations"] += 1
            else:
                self.discovery_metrics["failed_discoveries"] += 1

            # Update average discovery time
            total_discoveries = self.discovery_metrics["total_discoveries"]
            current_avg = self.discovery_metrics["avg_discovery_time"]
            self.discovery_metrics["avg_discovery_time"] = (
                (current_avg * (total_discoveries - 1) + discovery_time) / total_discoveries
            )

            logger.info(f"🔍 Discovery completed in {discovery_time:.2f}s: {len(results)} agents found")
            return results

        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            self.discovery_metrics["failed_discoveries"] += 1
            return []

    async def find_agents_by_capability(self, capabilities: List[str], max_results: int = 5) -> List[DiscoveryResult]:
        """
        Quick discovery of agents by specific capabilities

        Args:
            capabilities: List of required capabilities
            max_results: Maximum number of results to return

        Returns:
            List of discovery results
        """
        request = DiscoveryRequest(
            requesting_agent=getattr(self, 'agent_id', 'unknown'),
            required_capabilities=capabilities,
            max_results=max_results,
            scope=DiscoveryScope.LOCAL_NETWORK
        )

        return await self.discover_agents(request)

    async def find_best_collaborator(self, task_type: str, task_context: Dict[str, Any] = None) -> Optional[DiscoveryResult]:
        """
        Find the best agent for collaboration on a specific task

        Args:
            task_type: Type of task requiring collaboration
            task_context: Additional context about the task

        Returns:
            Best matching agent or None if no suitable agent found
        """
        if not self.discovery_enabled:
            return None

        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            result = await find_best_agent_for_task(agent_id, task_type, task_context or {})

            if result:
                logger.info(f"🤝 Found best collaborator for {task_type}: {result.agent_profile.agent_id} (score: {result.match_score:.2f})")
            else:
                logger.warning(f"⚠️ No suitable collaborator found for {task_type}")

            return result

        except Exception as e:
            logger.error(f"Failed to find collaborator: {e}")
            return None

    async def discover_specialized_agents(self, specialization: str, min_trust: float = 0.7) -> List[DiscoveryResult]:
        """
        Discover agents with specific specializations

        Args:
            specialization: Required specialization
            min_trust: Minimum trust score required

        Returns:
            List of specialized agents
        """
        request = DiscoveryRequest(
            requesting_agent=getattr(self, 'agent_id', 'unknown'),
            required_capabilities=[],  # We'll filter by specialization
            min_trust_score=min_trust,
            metadata_filters={"specialization": specialization},
            prefer_specialized=True,
            max_results=10
        )

        return await self.discover_agents(request)

    async def get_network_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive overview of the agent network

        Returns:
            Network topology and health information
        """
        if not self.discovery_enabled:
            return {"error": "Discovery not enabled"}

        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            topology = await get_network_topology(agent_id)

            # Add discovery metrics
            topology["discovery_metrics"] = self.discovery_metrics.copy()

            logger.info(f"📊 Network overview: {topology['total_agents']} agents, health: {topology['network_health']}")
            return topology

        except Exception as e:
            logger.error(f"Failed to get network overview: {e}")
            return {"error": str(e)}

    async def update_agent_performance(self, metrics: Dict[str, float]):
        """
        Update this agent's performance metrics in the registry

        Args:
            metrics: Performance metrics to update
        """
        if not self.discovery_enabled or not self.discovery_engine:
            return

        try:
            agent_id = getattr(self, 'agent_id', 'unknown')
            success = await self.discovery_engine.update_agent_performance(agent_id, metrics)

            if success:
                logger.debug(f"✅ Updated performance metrics for {agent_id}")

                # Update local profile
                if self.agent_profile:
                    self.agent_profile.performance_metrics.update(metrics)
            else:
                logger.warning(f"⚠️ Failed to update performance metrics for {agent_id}")

        except Exception as e:
            logger.error(f"Performance update failed: {e}")

    async def discover_cross_chain_agents(self, target_chain: str = None) -> List[DiscoveryResult]:
        """
        Discover agents capable of cross-chain operations

        Args:
            target_chain: Specific blockchain to target (optional)

        Returns:
            List of cross-chain capable agents
        """
        capabilities = ["cross_chain", "blockchain_operations"]
        metadata_filters = {}

        if target_chain:
            metadata_filters["supported_chains"] = {"operator": "contains", "value": target_chain}

        request = DiscoveryRequest(
            requesting_agent=getattr(self, 'agent_id', 'unknown'),
            required_capabilities=capabilities,
            metadata_filters=metadata_filters,
            scope=DiscoveryScope.CROSS_CHAIN,
            max_results=10,
            min_trust_score=0.8  # Higher trust for cross-chain operations
        )

        return await self.discover_agents(request)

    async def find_ai_reasoning_agents(self, reasoning_type: str = None) -> List[DiscoveryResult]:
        """
        Discover agents with AI reasoning capabilities

        Args:
            reasoning_type: Specific type of reasoning needed (optional)

        Returns:
            List of AI-capable agents
        """
        capabilities = ["ai_reasoning"]
        metadata_filters = {}

        if reasoning_type:
            metadata_filters["reasoning_types"] = {"operator": "contains", "value": reasoning_type}

        request = DiscoveryRequest(
            requesting_agent=getattr(self, 'agent_id', 'unknown'),
            required_capabilities=capabilities,
            optional_capabilities=["synthesis", "quality_assessment"],
            metadata_filters=metadata_filters,
            prefer_specialized=True,
            max_results=5
        )

        return await self.discover_agents(request)

    def get_discovery_metrics(self) -> Dict[str, Any]:
        """
        Get discovery performance metrics

        Returns:
            Discovery metrics and analytics
        """
        base_metrics = self.discovery_metrics.copy()

        if self.discovery_engine:
            base_metrics.update(self.discovery_engine.get_discovery_analytics())

        return base_metrics

    async def broadcast_capability_update(self, new_capabilities: List[str]):
        """
        Broadcast capability updates to the network

        Args:
            new_capabilities: List of new capabilities to announce
        """
        if not self.discovery_enabled or not self.agent_profile:
            return

        try:
            # Update local profile
            self.agent_profile.capabilities.extend(new_capabilities)
            self.agent_profile.capabilities = list(set(self.agent_profile.capabilities))  # Remove duplicates

            # Register updated capabilities
            success = await self.discovery_engine.register_agent_capabilities(self.agent_profile)

            if success:
                logger.info(f"📢 Broadcasted capability update: {new_capabilities}")
            else:
                logger.warning(f"⚠️ Failed to broadcast capability update")

        except Exception as e:
            logger.error(f"Capability broadcast failed: {e}")

    async def shutdown_discovery_engine(self):
        """Gracefully shutdown the discovery engine"""
        if self.discovery_engine:
            try:
                # Final performance update
                if hasattr(self, 'get_performance_metrics'):
                    final_metrics = await self.get_performance_metrics()
                    await self.update_agent_performance(final_metrics)

                logger.info("🔄 Discovery engine shutdown complete")

            except Exception as e:
                logger.warning(f"Discovery engine shutdown error: {e}")


# Convenience functions for agents

async def quick_agent_discovery(agent_id: str, required_capabilities: List[str]) -> List[str]:
    """
    Quick function to get agent IDs with specific capabilities

    Args:
        agent_id: Requesting agent ID
        required_capabilities: List of required capabilities

    Returns:
        List of agent IDs that match the criteria
    """
    results = await discover_agents_by_capability(agent_id, required_capabilities)
    return [result.agent_profile.agent_id for result in results]


async def find_backup_agents(agent_id: str, primary_agent_id: str, capabilities: List[str]) -> List[str]:
    """
    Find backup agents with similar capabilities

    Args:
        agent_id: Requesting agent ID
        primary_agent_id: Primary agent to find backups for
        capabilities: Required capabilities

    Returns:
        List of backup agent IDs
    """
    results = await discover_agents_by_capability(agent_id, capabilities, max_results=10)

    # Exclude the primary agent
    backup_agents = [
        result.agent_profile.agent_id
        for result in results
        if result.agent_profile.agent_id != primary_agent_id
    ]

    return backup_agents[:5]  # Return top 5 backups

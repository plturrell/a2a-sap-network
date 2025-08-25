"""
Advanced Agent Discovery System for A2A Network
Provides intelligent agent discovery through blockchain registry with AI-enhanced capabilities
"""

import asyncio
import logging
import json
import os
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# A2A Protocol Compliance: Required imports
from .networkClient import A2ANetworkClient
from ..sdk.blockchainIntegration import BlockchainIntegrationMixin

logger = logging.getLogger(__name__)

# A2A Protocol Compliance: AI is required - no fallbacks allowed
try:
    from app.clients.grokClient import GrokClient
    AI_DISCOVERY_AVAILABLE = True
except ImportError:
    try:
        from .grokClient import GrokClient
        AI_DISCOVERY_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "GrokClient is required for A2A protocol compliance in agent discovery. "
            "All agent discovery must use AI intelligence - no fallbacks allowed."
        )


class AgentCapabilityType(Enum):
    """Types of agent capabilities for discovery"""
    DATA_PROCESSING = "data_processing"
    FINANCIAL_ANALYSIS = "financial_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    STANDARDIZATION = "standardization"
    METADATA_MANAGEMENT = "metadata_management"
    AI_REASONING = "ai_reasoning"
    BLOCKCHAIN_OPERATIONS = "blockchain_operations"
    CROSS_CHAIN = "cross_chain"
    SECURITY = "security"
    MONITORING = "monitoring"
    SYNTHESIS = "synthesis"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"


class DiscoveryScope(Enum):
    """Scope of agent discovery"""
    LOCAL_NETWORK = "local_network"
    CROSS_CHAIN = "cross_chain"
    GLOBAL_NETWORK = "global_network"
    TRUSTED_ONLY = "trusted_only"


@dataclass
class AgentProfile:
    """Comprehensive agent profile for discovery"""
    agent_id: str
    name: str
    endpoint: str
    capabilities: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trust_score: float = 0.0
    reputation: int = 0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    availability_score: float = 1.0
    response_time_avg: float = 0.0
    success_rate: float = 1.0
    specializations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    blockchain_address: Optional[str] = None
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics,
            "trust_score": self.trust_score,
            "reputation": self.reputation,
            "last_seen": self.last_seen.isoformat(),
            "availability_score": self.availability_score,
            "response_time_avg": self.response_time_avg,
            "success_rate": self.success_rate,
            "specializations": self.specializations,
            "metadata": self.metadata,
            "blockchain_address": self.blockchain_address,
            "version": self.version
        }


@dataclass
class DiscoveryRequest:
    """Agent discovery request specification"""
    requesting_agent: str
    required_capabilities: List[str]
    optional_capabilities: List[str] = field(default_factory=list)
    min_trust_score: float = 0.0
    min_reputation: int = 0
    max_response_time: float = 5.0
    min_availability: float = 0.8
    scope: DiscoveryScope = DiscoveryScope.LOCAL_NETWORK
    max_results: int = 10
    prefer_specialized: bool = True
    exclude_agents: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Agent discovery result with scoring"""
    agent_profile: AgentProfile
    match_score: float
    capability_match: float
    performance_score: float
    trust_factor: float
    availability_factor: float
    recommendation_reason: str
    ai_analysis: Optional[str] = None


class AgentDiscoveryEngine:
    """
    Advanced Agent Discovery Engine with AI-Enhanced Matching
    Provides intelligent agent discovery through blockchain registry
    """

    def __init__(self, agent_id: str, blockchain_client=None):
        self.agent_id = agent_id
        self.blockchain_client = blockchain_client  # Optional blockchain client

        # A2ANetworkClient requires blockchain_client
        if blockchain_client:
            self.network_client = A2ANetworkClient(agent_id=agent_id, blockchain_client=blockchain_client)
        else:
            # Create a minimal blockchain client if none provided
            from ..sdk.blockchain.web3Client import A2ABlockchainClient
            import os
            rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
            blockchain_client = A2ABlockchainClient(rpc_url)
            self.network_client = A2ANetworkClient(agent_id=agent_id, blockchain_client=blockchain_client)

        # Initialize AI for intelligent discovery
        try:
            self.grok_client = GrokClient()
            logger.info(f"✅ AI-enhanced discovery initialized for {agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize GrokClient: {e}")
            raise RuntimeError(f"Failed to initialize AI for agent discovery: {e}")

        # Agent registry cache
        self.agent_registry: Dict[str, AgentProfile] = {}
        self.last_registry_update = datetime.min
        self.registry_cache_ttl = timedelta(minutes=5)

        # Discovery analytics
        self.discovery_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, Dict[str, float]] = {}

        # Blockchain integration for verification
        self.blockchain_integration = None

    async def initialize_blockchain_integration(self):
        """Initialize blockchain integration for agent verification"""
        try:
            # This would be injected by the agent using this discovery engine
            logger.info("Blockchain integration ready for agent discovery")
        except Exception as e:
            logger.warning(f"Blockchain integration not available: {e}")

    async def discover_agents(self, request: DiscoveryRequest) -> List[DiscoveryResult]:
        """
        Discover agents matching the request criteria with AI-enhanced ranking
        """
        logger.info(f"🔍 Starting agent discovery for {request.requesting_agent}")
        logger.info(f"   Required capabilities: {request.required_capabilities}")
        logger.info(f"   Scope: {request.scope.value}")

        # Update agent registry from blockchain
        await self._update_agent_registry(request.scope)

        # Filter agents based on basic criteria
        candidate_agents = await self._filter_candidates(request)

        # AI-enhanced scoring and ranking
        discovery_results = await self._ai_rank_agents(request, candidate_agents)

        # Record discovery analytics
        await self._record_discovery_analytics(request, discovery_results)

        logger.info(f"✅ Discovery complete: {len(discovery_results)} agents found")
        return discovery_results[:request.max_results]

    async def _update_agent_registry(self, scope: DiscoveryScope):
        """Update agent registry from blockchain"""
        if datetime.utcnow() - self.last_registry_update < self.registry_cache_ttl:
            logger.debug("Using cached agent registry")
            return

        try:
            # Query blockchain registry for active agents
            registry_request = {
                "operation": "list_active_agents",
                "scope": scope.value,
                "include_performance": True,
                "include_reputation": True,
                "timestamp": datetime.utcnow().isoformat()
            }

            response = await self.network_client.send_a2a_message(
                to_agent="agent_registry",
                message=registry_request,
                message_type="AGENT_REGISTRY_QUERY"
            )

            if response and response.get("agents"):
                agents_data = response["agents"]

                # Update registry with fresh data
                for agent_data in agents_data:
                    agent_profile = AgentProfile(
                        agent_id=agent_data["agent_id"],
                        name=agent_data["name"],
                        endpoint=agent_data["endpoint"],
                        capabilities=agent_data.get("capabilities", []),
                        performance_metrics=agent_data.get("performance_metrics", {}),
                        trust_score=agent_data.get("trust_score", 0.0),
                        reputation=agent_data.get("reputation", 0),
                        last_seen=datetime.fromisoformat(agent_data.get("last_seen", datetime.utcnow().isoformat())),
                        availability_score=agent_data.get("availability_score", 1.0),
                        response_time_avg=agent_data.get("response_time_avg", 0.0),
                        success_rate=agent_data.get("success_rate", 1.0),
                        specializations=agent_data.get("specializations", []),
                        metadata=agent_data.get("metadata", {}),
                        blockchain_address=agent_data.get("blockchain_address"),
                        version=agent_data.get("version", "1.0.0")
                    )

                    self.agent_registry[agent_profile.agent_id] = agent_profile

                self.last_registry_update = datetime.utcnow()
                logger.info(f"✅ Updated agent registry: {len(self.agent_registry)} agents")
            else:
                logger.warning("No agents returned from blockchain registry")

        except Exception as e:
            logger.error(f"Failed to update agent registry: {e}")
            raise

    async def _filter_candidates(self, request: DiscoveryRequest) -> List[AgentProfile]:
        """Filter agents based on basic criteria"""
        candidates = []

        for agent_profile in self.agent_registry.values():
            # Skip excluded agents
            if agent_profile.agent_id in request.exclude_agents:
                continue

            # Skip self
            if agent_profile.agent_id == request.requesting_agent:
                continue

            # Check required capabilities
            if not all(cap in agent_profile.capabilities for cap in request.required_capabilities):
                continue

            # Check trust score
            if agent_profile.trust_score < request.min_trust_score:
                continue

            # Check reputation
            if agent_profile.reputation < request.min_reputation:
                continue

            # Check availability
            if agent_profile.availability_score < request.min_availability:
                continue

            # Check response time
            if agent_profile.response_time_avg > request.max_response_time:
                continue

            # Check metadata filters
            if request.metadata_filters:
                if not self._matches_metadata_filters(agent_profile.metadata, request.metadata_filters):
                    continue

            candidates.append(agent_profile)

        logger.info(f"🔎 Filtered to {len(candidates)} candidate agents")
        return candidates

    def _matches_metadata_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if agent metadata matches filters"""
        for key, expected_value in filters.items():
            if key not in metadata:
                return False

            actual_value = metadata[key]

            # Handle different comparison types
            if isinstance(expected_value, dict) and "operator" in expected_value:
                operator = expected_value["operator"]
                value = expected_value["value"]

                if operator == "gte" and actual_value < value:
                    return False
                elif operator == "lte" and actual_value > value:
                    return False
                elif operator == "eq" and actual_value != value:
                    return False
                elif operator == "contains" and value not in str(actual_value):
                    return False
            else:
                if actual_value != expected_value:
                    return False

        return True

    async def _ai_rank_agents(self, request: DiscoveryRequest, candidates: List[AgentProfile]) -> List[DiscoveryResult]:
        """Use AI to intelligently rank and score agent candidates"""
        if not candidates:
            return []

        try:
            # Prepare context for AI analysis
            ranking_context = {
                "requesting_agent": request.requesting_agent,
                "discovery_request": {
                    "required_capabilities": request.required_capabilities,
                    "optional_capabilities": request.optional_capabilities,
                    "prefer_specialized": request.prefer_specialized,
                    "scope": request.scope.value
                },
                "candidates": [agent.to_dict() for agent in candidates],
                "discovery_history": self.discovery_history[-5:] if self.discovery_history else []
            }

            ai_prompt = f"""
            As an AI agent discovery expert, analyze these agent candidates and provide intelligent ranking:

            Context: {json.dumps(ranking_context, indent=2)}

            For each candidate, consider:
            1. Capability match (required vs optional capabilities)
            2. Performance metrics and reliability
            3. Trust score and reputation
            4. Specialization alignment
            5. Historical success patterns
            6. Current availability and response time
            7. Agent synergy and collaboration potential

            Provide detailed scoring and ranking as JSON:
            {{
                "rankings": [
                    {{
                        "agent_id": "agent_id",
                        "match_score": 0.95,
                        "capability_match": 0.9,
                        "performance_score": 0.8,
                        "trust_factor": 0.85,
                        "availability_factor": 0.9,
                        "recommendation_reason": "Detailed explanation of why this agent is ranked here",
                        "ai_analysis": "In-depth analysis of agent suitability"
                    }}
                ],
                "discovery_insights": "Overall insights about the agent ecosystem for this request"
            }}

            Sort by match_score descending.
            """

            ai_response = await self.grok_client.generate_response(ai_prompt)

            if ai_response and "rankings" in ai_response:
                discovery_results = []

                for ranking in ai_response["rankings"]:
                    agent_id = ranking["agent_id"]

                    # Find the corresponding agent profile
                    agent_profile = next((agent for agent in candidates if agent.agent_id == agent_id), None)
                    if not agent_profile:
                        continue

                    result = DiscoveryResult(
                        agent_profile=agent_profile,
                        match_score=ranking.get("match_score", 0.0),
                        capability_match=ranking.get("capability_match", 0.0),
                        performance_score=ranking.get("performance_score", 0.0),
                        trust_factor=ranking.get("trust_factor", 0.0),
                        availability_factor=ranking.get("availability_factor", 0.0),
                        recommendation_reason=ranking.get("recommendation_reason", "AI recommendation"),
                        ai_analysis=ranking.get("ai_analysis")
                    )

                    discovery_results.append(result)

                logger.info(f"🤖 AI ranked {len(discovery_results)} agents with intelligence")
                return discovery_results

        except Exception as e:
            # A2A Protocol Compliance: AI ranking is required - no fallbacks allowed
            logger.error(f"AI ranking failed: {e}")
            raise RuntimeError(f"AI ranking is required for A2A protocol compliance: {e}")


    async def _record_discovery_analytics(self, request: DiscoveryRequest, results: List[DiscoveryResult]):
        """Record discovery analytics for learning"""
        analytics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "requesting_agent": request.requesting_agent,
            "required_capabilities": request.required_capabilities,
            "scope": request.scope.value,
            "candidates_found": len(results),
            "top_match_score": results[0].match_score if results else 0.0,
            "discovery_successful": len(results) > 0
        }

        self.discovery_history.append(analytics_record)

        # Keep only recent history
        if len(self.discovery_history) > 100:
            self.discovery_history = self.discovery_history[-100:]

    async def register_agent_capabilities(self, agent_profile: AgentProfile) -> bool:
        """Register agent capabilities with the blockchain registry"""
        try:
            registration_data = {
                "operation": "register_capabilities",
                "agent_profile": agent_profile.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

            response = await self.network_client.send_a2a_message(
                to_agent="agent_registry",
                message=registration_data,
                message_type="CAPABILITY_REGISTRATION"
            )

            if response and response.get("status") == "success":
                logger.info(f"✅ Registered capabilities for {agent_profile.agent_id}")
                return True
            else:
                logger.error(f"Failed to register capabilities: {response}")
                return False

        except Exception as e:
            logger.error(f"Capability registration failed: {e}")
            return False

    async def update_agent_performance(self, agent_id: str, metrics: Dict[str, float]) -> bool:
        """Update agent performance metrics"""
        try:
            update_data = {
                "operation": "update_performance",
                "agent_id": agent_id,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

            response = await self.network_client.send_a2a_message(
                to_agent="agent_registry",
                message=update_data,
                message_type="PERFORMANCE_UPDATE"
            )

            return response and response.get("status") == "success"

        except Exception as e:
            logger.error(f"Performance update failed: {e}")
            return False

    def get_discovery_analytics(self) -> Dict[str, Any]:
        """Get discovery analytics and insights"""
        if not self.discovery_history:
            return {"total_discoveries": 0}

        recent_discoveries = [d for d in self.discovery_history if
                            datetime.fromisoformat(d["timestamp"]) > datetime.utcnow() - timedelta(hours=24)]

        return {
            "total_discoveries": len(self.discovery_history),
            "recent_discoveries_24h": len(recent_discoveries),
            "success_rate": len([d for d in self.discovery_history if d["discovery_successful"]]) / len(self.discovery_history),
            "avg_candidates_found": sum(d["candidates_found"] for d in self.discovery_history) / len(self.discovery_history),
            "most_requested_capabilities": self._get_top_capabilities(),
            "registry_size": len(self.agent_registry),
            "last_registry_update": self.last_registry_update.isoformat() if self.last_registry_update != datetime.min else None
        }

    def _get_top_capabilities(self) -> List[Tuple[str, int]]:
        """Get most frequently requested capabilities"""
        capability_counts = {}

        for discovery in self.discovery_history:
            for cap in discovery.get("required_capabilities", []):
                capability_counts[cap] = capability_counts.get(cap, 0) + 1

        return sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)[:10]


# Convenience functions for agent discovery

async def discover_agents_by_capability(agent_id: str, capabilities: List[str], max_results: int = 5) -> List[DiscoveryResult]:
    """Quick agent discovery by capabilities"""
    discovery_engine = AgentDiscoveryEngine(agent_id)

    request = DiscoveryRequest(
        requesting_agent=agent_id,
        required_capabilities=capabilities,
        max_results=max_results
    )

    return await discovery_engine.discover_agents(request)


async def find_best_agent_for_task(agent_id: str, task_type: str, task_context: Dict[str, Any]) -> Optional[DiscoveryResult]:
    """Find the best agent for a specific task"""
    discovery_engine = AgentDiscoveryEngine(agent_id)

    # Map task types to capabilities
    task_capability_mapping = {
        "data_processing": ["data_processing", "standardization"],
        "financial_analysis": ["financial_analysis", "data_processing"],
        "quality_assessment": ["quality_assessment", "ai_reasoning"],
        "synthesis": ["synthesis", "ai_reasoning"],
        "blockchain_ops": ["blockchain_operations", "security"],
        "cross_chain": ["cross_chain", "blockchain_operations"]
    }

    required_caps = task_capability_mapping.get(task_type, [task_type])

    request = DiscoveryRequest(
        requesting_agent=agent_id,
        required_capabilities=required_caps,
        metadata_filters=task_context,
        max_results=1,
        prefer_specialized=True
    )

    results = await discovery_engine.discover_agents(request)
    return results[0] if results else None


async def get_network_topology(agent_id: str) -> Dict[str, Any]:
    """Get comprehensive network topology information"""
    discovery_engine = AgentDiscoveryEngine(agent_id)
    await discovery_engine._update_agent_registry(DiscoveryScope.GLOBAL_NETWORK)

    topology = {
        "total_agents": len(discovery_engine.agent_registry),
        "capabilities_distribution": {},
        "trust_distribution": {"high": 0, "medium": 0, "low": 0},
        "availability_distribution": {"high": 0, "medium": 0, "low": 0},
        "agent_specializations": {},
        "network_health": "unknown"
    }

    for agent in discovery_engine.agent_registry.values():
        # Capability distribution
        for cap in agent.capabilities:
            topology["capabilities_distribution"][cap] = topology["capabilities_distribution"].get(cap, 0) + 1

        # Trust distribution
        if agent.trust_score >= 0.8:
            topology["trust_distribution"]["high"] += 1
        elif agent.trust_score >= 0.5:
            topology["trust_distribution"]["medium"] += 1
        else:
            topology["trust_distribution"]["low"] += 1

        # Availability distribution
        if agent.availability_score >= 0.9:
            topology["availability_distribution"]["high"] += 1
        elif agent.availability_score >= 0.7:
            topology["availability_distribution"]["medium"] += 1
        else:
            topology["availability_distribution"]["low"] += 1

        # Specializations
        for spec in agent.specializations:
            topology["agent_specializations"][spec] = topology["agent_specializations"].get(spec, 0) + 1

    # Calculate network health
    high_trust_ratio = topology["trust_distribution"]["high"] / max(topology["total_agents"], 1)
    high_availability_ratio = topology["availability_distribution"]["high"] / max(topology["total_agents"], 1)

    if high_trust_ratio >= 0.7 and high_availability_ratio >= 0.8:
        topology["network_health"] = "excellent"
    elif high_trust_ratio >= 0.5 and high_availability_ratio >= 0.6:
        topology["network_health"] = "good"
    elif high_trust_ratio >= 0.3 and high_availability_ratio >= 0.4:
        topology["network_health"] = "fair"
    else:
        topology["network_health"] = "poor"

    return topology
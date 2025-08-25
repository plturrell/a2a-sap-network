"""
Collaborative Intelligence Framework for A2A Agents
Part of Phase 1: Core AI Framework

This module provides advanced multi-agent collaboration, knowledge sharing,
consensus mechanisms, and swarm intelligence capabilities.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import asyncio
from abc import ABC, abstractmethod
import networkx as nx
from uuid import uuid4

logger = logging.getLogger(__name__)


class CollaborationType(str, Enum):
    """Types of collaboration"""

    CONSENSUS = "consensus"
    DELEGATION = "delegation"
    COORDINATION = "coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    SWARM = "swarm"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"


class ConsensusAlgorithmType(str, Enum):
    """Consensus algorithms"""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    RAFT = "raft"
    PROOF_OF_STAKE = "proof_of_stake"
    PRACTICAL_BFT = "practical_bft"


class SwarmBehaviorType(str, Enum):
    """Swarm intelligence behaviors"""

    FLOCKING = "flocking"
    FORAGING = "foraging"
    CLUSTERING = "clustering"
    EXPLORATION = "exploration"
    DIVISION_OF_LABOR = "division_of_labor"


@dataclass
class Agent:
    """Represents an agent in the collaborative network"""

    agent_id: str
    name: str
    capabilities: List[str]
    trust_score: float = 1.0
    reputation: float = 1.0
    workload: float = 0.0
    specializations: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationRequest:
    """Represents a collaboration request"""

    request_id: str
    requester_id: str
    task_description: str
    required_capabilities: List[str]
    collaboration_type: CollaborationType
    deadline: Optional[datetime] = None
    priority: float = 0.5
    resources_offered: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CollaborationResult:
    """Represents collaboration result"""

    collaboration_id: str
    participants: List[str]
    result: Any
    success: bool
    confidence: float
    duration: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborativeIntelligenceFramework:
    """
    Advanced collaborative intelligence framework
    Provides multi-agent coordination, consensus, and swarm intelligence
    """

    def __init__(self, agent_id: str, network_config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.network_config = network_config or {}

        # Agent registry and network
        self.agents = {}
        self.collaboration_network = nx.DiGraph()

        # Collaboration management
        self.active_collaborations = {}
        self.collaboration_history = deque(maxlen=1000)
        self.pending_requests = {}

        # Consensus mechanisms
        self.consensus_algorithms = {
            ConsensusAlgorithmType.MAJORITY_VOTE: MajorityVoteConsensus(),
            ConsensusAlgorithmType.WEIGHTED_VOTE: WeightedVoteConsensus(),
            ConsensusAlgorithmType.BYZANTINE_FAULT_TOLERANT: ByzantineFaultTolerantConsensus(),
            ConsensusAlgorithmType.RAFT: RaftConsensus(),
            ConsensusAlgorithmType.PROOF_OF_STAKE: ProofOfStakeConsensus(),
        }

        # Swarm intelligence
        self.swarm_behaviors = {
            SwarmBehaviorType.FLOCKING: FlockingBehavior(),
            SwarmBehaviorType.FORAGING: ForagingBehavior(),
            SwarmBehaviorType.CLUSTERING: ClusteringBehavior(),
            SwarmBehaviorType.EXPLORATION: ExplorationBehavior(),
            SwarmBehaviorType.DIVISION_OF_LABOR: DivisionOfLaborBehavior(),
        }

        # Knowledge sharing
        self.knowledge_cache = {}
        self.sharing_policies = {}

        # Performance tracking
        self.collaboration_metrics = defaultdict(list)

        logger.info(f"Initialized collaborative intelligence framework for agent {agent_id}")

    def _default_success_rate(self):
        """Default success rate structure for agent success tracking"""
        return {'successes': 0, 'attempts': 0}

    async def register_agent(self, agent: Agent) -> bool:
        """
        Register an agent in the collaborative network

        Args:
            agent: Agent to register

        Returns:
            Success status
        """
        self.agents[agent.agent_id] = agent

        # Add to collaboration network
        self.collaboration_network.add_node(
            agent.agent_id,
            capabilities=agent.capabilities,
            trust_score=agent.trust_score,
            reputation=agent.reputation,
            specializations=agent.specializations,
        )

        # Create initial connections based on capabilities
        await self._create_initial_connections(agent)

        logger.info(f"Registered agent {agent.agent_id} with capabilities: {agent.capabilities}")
        return True

    async def request_collaboration(self, request: CollaborationRequest) -> Dict[str, Any]:
        """
        Request collaboration from other agents

        Args:
            request: Collaboration request details

        Returns:
            Collaboration response
        """
        # Find suitable agents
        suitable_agents = await self._find_suitable_agents(request)

        if not suitable_agents:
            return {"success": False, "reason": "No suitable agents found", "suitable_agents": []}

        # Create collaboration session
        collaboration_id = str(uuid4())

        # Send requests to selected agents
        responses = await self._send_collaboration_requests(
            collaboration_id, request, suitable_agents
        )

        # Select participants based on responses
        participants = await self._select_participants(responses, request)

        if participants:
            self.active_collaborations[collaboration_id] = {
                "request": request,
                "participants": participants,
                "status": "active",
                "created_at": datetime.utcnow(),
            }

            return {
                "success": True,
                "collaboration_id": collaboration_id,
                "participants": [p["agent_id"] for p in participants],
                "estimated_duration": self._estimate_collaboration_duration(request, participants),
            }
        else:
            return {
                "success": False,
                "reason": "No agents accepted collaboration",
                "responses": responses,
            }

    async def execute_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any]
    ) -> CollaborationResult:
        """
        Execute a collaboration

        Args:
            collaboration_id: ID of the collaboration
            task_data: Task data to process

        Returns:
            Collaboration result
        """
        if collaboration_id not in self.active_collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")

        collaboration = self.active_collaborations[collaboration_id]
        request = collaboration["request"]
        participants = collaboration["participants"]

        start_time = datetime.utcnow()

        try:
            # Execute based on collaboration type
            if request.collaboration_type == CollaborationType.CONSENSUS:
                result = await self._execute_consensus_collaboration(
                    collaboration_id, task_data, participants
                )
            elif request.collaboration_type == CollaborationType.SWARM:
                result = await self._execute_swarm_collaboration(
                    collaboration_id, task_data, participants
                )
            elif request.collaboration_type == CollaborationType.DELEGATION:
                result = await self._execute_delegation_collaboration(
                    collaboration_id, task_data, participants
                )
            elif request.collaboration_type == CollaborationType.COORDINATION:
                result = await self._execute_coordination_collaboration(
                    collaboration_id, task_data, participants
                )
            else:
                result = await self._execute_default_collaboration(
                    collaboration_id, task_data, participants
                )

            duration = (datetime.utcnow() - start_time).total_seconds()

            # Create result
            collaboration_result = CollaborationResult(
                collaboration_id=collaboration_id,
                participants=[p["agent_id"] for p in participants],
                result=result,
                success=True,
                confidence=result.get("confidence", 0.8),
                duration=duration,
            )

            # Update metrics
            await self._update_collaboration_metrics(collaboration_result)

            # Clean up
            del self.active_collaborations[collaboration_id]
            self.collaboration_history.append(collaboration_result)

            return collaboration_result

        except Exception as e:
            logger.error(f"Collaboration {collaboration_id} failed: {e}")

            duration = (datetime.utcnow() - start_time).total_seconds()

            collaboration_result = CollaborationResult(
                collaboration_id=collaboration_id,
                participants=[p["agent_id"] for p in participants],
                result={"error": str(e)},
                success=False,
                confidence=0.0,
                duration=duration,
            )

            return collaboration_result

    async def reach_consensus(
        self,
        agents: List[str],
        proposals: List[Any],
        algorithm: ConsensusAlgorithmType = ConsensusAlgorithmType.MAJORITY_VOTE,
    ) -> Dict[str, Any]:
        """
        Reach consensus among agents

        Args:
            agents: List of agent IDs
            proposals: List of proposals to vote on
            algorithm: Consensus algorithm to use

        Returns:
            Consensus result
        """
        consensus_impl = self.consensus_algorithms.get(algorithm)
        if not consensus_impl:
            raise ValueError(f"Unknown consensus algorithm: {algorithm}")

        # Collect votes from agents
        votes = await self._collect_votes(agents, proposals)

        # Apply consensus algorithm
        result = await consensus_impl.reach_consensus(votes, agents, self.agents)

        # Record consensus
        self.collaboration_metrics["consensus"].append(
            {
                "algorithm": algorithm.value,
                "participants": len(agents),
                "proposals": len(proposals),
                "result": result,
                "timestamp": datetime.utcnow(),
            }
        )

        return result

    async def execute_swarm_behavior(
        self, agents: List[str], behavior: SwarmBehaviorType, environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute swarm intelligence behavior

        Args:
            agents: List of agent IDs
            behavior: Swarm behavior to execute
            environment: Environment parameters

        Returns:
            Swarm behavior result
        """
        swarm_impl = self.swarm_behaviors.get(behavior)
        if not swarm_impl:
            raise ValueError(f"Unknown swarm behavior: {behavior}")

        # Execute swarm behavior
        result = await swarm_impl.execute(agents, environment, self.agents)

        # Record swarm activity
        self.collaboration_metrics["swarm"].append(
            {
                "behavior": behavior.value,
                "participants": len(agents),
                "environment": environment,
                "result": result,
                "timestamp": datetime.utcnow(),
            }
        )

        return result

    async def share_knowledge(
        self,
        target_agents: List[str],
        knowledge: Dict[str, Any],
        sharing_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Share knowledge with other agents

        Args:
            target_agents: List of target agent IDs
            knowledge: Knowledge to share
            sharing_policy: Policy for knowledge sharing

        Returns:
            Sharing result
        """
        # Apply sharing policy
        filtered_knowledge = await self._apply_sharing_policy(knowledge, sharing_policy)

        # Share with target agents
        sharing_results = {}

        for agent_id in target_agents:
            if agent_id in self.agents:
                success = await self._send_knowledge(agent_id, filtered_knowledge)
                sharing_results[agent_id] = success

        # Update knowledge cache
        knowledge_id = str(uuid4())
        self.knowledge_cache[knowledge_id] = {
            "knowledge": filtered_knowledge,
            "shared_with": target_agents,
            "timestamp": datetime.utcnow(),
        }

        return {
            "knowledge_id": knowledge_id,
            "shared_with": list(sharing_results.keys()),
            "successful_shares": sum(sharing_results.values()),
            "total_targets": len(target_agents),
        }

    async def update_trust_scores(self, performance_data: Dict[str, float]):
        """
        Update trust scores based on performance

        Args:
            performance_data: Agent performance data
        """
        for agent_id, performance in performance_data.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]

                # Update performance history
                agent.performance_history.append(performance)
                if len(agent.performance_history) > 100:
                    agent.performance_history.pop(0)

                # Calculate new trust score
                recent_performance = agent.performance_history[-10:]
                avg_performance = np.mean(recent_performance)
                consistency = 1.0 - np.std(recent_performance)

                # Update trust score (weighted average)
                new_trust = avg_performance * 0.7 + consistency * 0.3
                agent.trust_score = agent.trust_score * 0.8 + new_trust * 0.2

                # Update reputation
                agent.reputation = agent.reputation * 0.9 + avg_performance * 0.1

                # Update network edge weights
                await self._update_network_connections(agent_id)

    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration statistics"""
        stats = {
            "total_agents": len(self.agents),
            "active_collaborations": len(self.active_collaborations),
            "total_collaborations": len(self.collaboration_history),
            "network_density": nx.density(self.collaboration_network),
            "average_trust_score": np.mean([agent.trust_score for agent in self.agents.values()]),
            "collaboration_success_rate": self._calculate_success_rate(),
            "most_active_agents": self._get_most_active_agents(),
            "collaboration_types": self._get_collaboration_type_stats(),
            "consensus_efficiency": self._calculate_consensus_efficiency(),
            "swarm_effectiveness": self._calculate_swarm_effectiveness(),
        }

        return stats

    async def _find_suitable_agents(self, request: CollaborationRequest) -> List[Agent]:
        """Find agents suitable for collaboration request"""
        suitable_agents = []

        for agent in self.agents.values():
            if agent.agent_id == request.requester_id:
                continue  # Skip requester

            # Check capability match
            capability_match = any(
                cap in agent.capabilities for cap in request.required_capabilities
            )

            if capability_match:
                # Check availability (workload)
                if agent.workload < 0.8:  # Not overloaded
                    # Check trust score
                    if agent.trust_score > 0.5:
                        suitable_agents.append(agent)

        # Sort by suitability score
        def get_agent_suitability_score(a):
            return self._calculate_suitability_score(a, request)

        suitable_agents.sort(key=get_agent_suitability_score, reverse=True)

        return suitable_agents[:10]  # Limit to top 10

    def _calculate_suitability_score(self, agent: Agent, request: CollaborationRequest) -> float:
        """Calculate how suitable an agent is for a request"""
        score = 0.0

        # Capability match score
        matching_caps = sum(1 for cap in request.required_capabilities if cap in agent.capabilities)
        capability_score = matching_caps / len(request.required_capabilities)
        score += capability_score * 0.4

        # Trust score
        score += agent.trust_score * 0.3

        # Availability score (inverse of workload)
        availability_score = 1.0 - agent.workload
        score += availability_score * 0.2

        # Reputation score
        score += agent.reputation * 0.1

        return score

    async def _send_collaboration_requests(
        self, collaboration_id: str, request: CollaborationRequest, agents: List[Agent]
    ) -> List[Dict[str, Any]]:
        """Send collaboration requests to agents"""
        responses = []

        for agent in agents:
            # Simulate sending request and getting response
            # In real implementation, would send over network
            response = await self._simulate_agent_response(agent, request)
            response["agent_id"] = agent.agent_id
            responses.append(response)

        return responses

    async def _simulate_agent_response(
        self, agent: Agent, request: CollaborationRequest
    ) -> Dict[str, Any]:
        """Simulate agent response to collaboration request"""
        # Simple heuristic for acceptance
        suitability = self._calculate_suitability_score(agent, request)

        # Probability of acceptance based on suitability and workload
        acceptance_prob = suitability * (1.0 - agent.workload)

        accepts = np.random.random() < acceptance_prob

        return {
            "accepts": accepts,
            "confidence": suitability,
            "estimated_effort": np.random.uniform(0.1, 0.5),
            "conditions": [] if accepts else ["overloaded", "low_match"],
        }

    async def _select_participants(
        self, responses: List[Dict[str, Any]], request: CollaborationRequest
    ) -> List[Dict[str, Any]]:
        """Select participants from responses"""
        # Filter accepting agents
        accepting = [r for r in responses if r["accepts"]]

        if not accepting:
            return []

        # Sort by confidence
        def get_confidence_score(x):
            return x["confidence"]

        accepting.sort(key=get_confidence_score, reverse=True)

        # Select based on collaboration type
        if request.collaboration_type == CollaborationType.CONSENSUS:
            # Need odd number for tie-breaking
            target_count = min(5, len(accepting))
            if target_count % 2 == 0 and target_count > 1:
                target_count -= 1
        else:
            # Select top performers
            target_count = min(3, len(accepting))

        return accepting[:target_count]

    async def _execute_consensus_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute consensus-based collaboration"""
        # Each participant generates a proposal
        proposals = []
        for participant in participants:
            proposal = await self._generate_proposal(participant, task_data)
            proposals.append(proposal)

        # Reach consensus
        agent_ids = [p["agent_id"] for p in participants]
        consensus_result = await self.reach_consensus(
            agent_ids, proposals, ConsensusAlgorithmType.WEIGHTED_VOTE
        )

        return {
            "type": "consensus",
            "proposals": proposals,
            "consensus": consensus_result,
            "confidence": consensus_result.get("confidence", 0.8),
        }

    async def _execute_swarm_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute swarm-based collaboration"""
        agent_ids = [p["agent_id"] for p in participants]

        # Determine swarm behavior based on task
        behavior = self._determine_swarm_behavior(task_data)

        # Execute swarm behavior
        result = await self.execute_swarm_behavior(agent_ids, behavior, task_data)

        return {
            "type": "swarm",
            "behavior": behavior.value,
            "result": result,
            "confidence": result.get("confidence", 0.7),
        }

    async def _execute_delegation_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute delegation-based collaboration"""
        # Select best participant as delegate
        def get_participant_confidence(p):
            return p["confidence"]

        delegate = max(participants, key=get_participant_confidence)

        # Delegate task execution
        result = await self._delegate_task(delegate, task_data)

        return {
            "type": "delegation",
            "delegate": delegate["agent_id"],
            "result": result,
            "confidence": delegate["confidence"],
        }

    async def _execute_coordination_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute coordination-based collaboration"""
        # Divide task among participants
        subtasks = await self._divide_task(task_data, participants)

        # Execute subtasks in parallel
        subtask_results = await asyncio.gather(
            *[
                self._execute_subtask(participant, subtask)
                for participant, subtask in zip(participants, subtasks)
            ]
        )

        # Combine results
        combined_result = await self._combine_results(subtask_results)

        return {
            "type": "coordination",
            "subtasks": len(subtasks),
            "subtask_results": subtask_results,
            "combined_result": combined_result,
            "confidence": np.mean([r.get("confidence", 0.5) for r in subtask_results]),
        }

    async def _execute_default_collaboration(
        self, collaboration_id: str, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute default collaboration (simple aggregation)"""
        # All participants work on the same task
        results = await asyncio.gather(
            *[self._execute_individual_task(participant, task_data) for participant in participants]
        )

        # Aggregate results
        aggregated = await self._aggregate_results(results)

        return {
            "type": "aggregation",
            "individual_results": results,
            "aggregated_result": aggregated,
            "confidence": np.mean([r.get("confidence", 0.5) for r in results]),
        }

    # Helper methods for simulation
    async def _generate_proposal(
        self, participant: Dict[str, Any], task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a proposal from a participant"""
        return {
            "agent_id": participant["agent_id"],
            "proposal": f"proposal_for_{task_data.get('task_id', 'unknown')}",
            "confidence": np.random.uniform(0.6, 0.9),
        }

    async def _delegate_task(
        self, delegate: Dict[str, Any], task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate task to an agent"""
        return {
            "result": f"delegated_result_from_{delegate['agent_id']}",
            "confidence": delegate["confidence"],
        }

    async def _divide_task(
        self, task_data: Dict[str, Any], participants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Divide task into subtasks"""
        return [{"subtask_id": f"subtask_{i}", "data": task_data} for i in range(len(participants))]

    async def _execute_subtask(
        self, participant: Dict[str, Any], subtask: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a subtask"""
        return {
            "subtask_id": subtask["subtask_id"],
            "result": f"result_from_{participant['agent_id']}",
            "confidence": np.random.uniform(0.5, 0.8),
        }

    async def _combine_results(self, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine subtask results"""
        return {"combined": "all_subtasks_combined", "subtask_count": len(subtask_results)}

    async def _execute_individual_task(
        self, participant: Dict[str, Any], task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task individually"""
        return {
            "agent_id": participant["agent_id"],
            "result": f"individual_result_from_{participant['agent_id']}",
            "confidence": np.random.uniform(0.4, 0.7),
        }

    async def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual results"""
        return {"aggregated": "all_results_aggregated", "result_count": len(results)}

    def _determine_swarm_behavior(self, task_data: Dict[str, Any]) -> SwarmBehaviorType:
        """Determine appropriate swarm behavior for task"""
        task_type = task_data.get("type", "unknown")

        if "search" in task_type or "explore" in task_type:
            return SwarmBehaviorType.EXPLORATION
        elif "cluster" in task_type or "group" in task_type:
            return SwarmBehaviorType.CLUSTERING
        elif "forage" in task_type or "collect" in task_type:
            return SwarmBehaviorType.FORAGING
        else:
            return SwarmBehaviorType.FLOCKING  # Default

    # Additional helper methods would go here...

    def _calculate_success_rate(self) -> float:
        """Calculate collaboration success rate"""
        if not self.collaboration_history:
            return 0.0

        successful = sum(1 for collab in self.collaboration_history if collab.success)
        return successful / len(self.collaboration_history)

    def _get_most_active_agents(self) -> List[str]:
        """Get most active agents"""
        activity_counts = defaultdict(int)

        for collab in self.collaboration_history:
            for participant in collab.participants:
                activity_counts[participant] += 1

        def get_activity_count(x):
            return x[1]

        sorted_agents = sorted(activity_counts.items(), key=get_activity_count, reverse=True)
        return [agent_id for agent_id, _ in sorted_agents[:5]]

    def _get_collaboration_type_stats(self) -> Dict[str, int]:
        """Get statistics by collaboration type"""
        type_counts = defaultdict(int)

        for collab in self.collaboration_history:
            collab_type = collab.metadata.get("type", "unknown")
            type_counts[collab_type] += 1

        return dict(type_counts)

    def _calculate_consensus_efficiency(self) -> float:
        """Calculate consensus efficiency"""
        consensus_metrics = self.collaboration_metrics.get("consensus", [])
        if not consensus_metrics:
            return 0.0

        return np.mean([m["result"].get("efficiency", 0.0) for m in consensus_metrics])

    def _calculate_swarm_effectiveness(self) -> float:
        """Calculate swarm effectiveness"""
        swarm_metrics = self.collaboration_metrics.get("swarm", [])
        if not swarm_metrics:
            return 0.0

        return np.mean([m["result"].get("effectiveness", 0.0) for m in swarm_metrics])


# Consensus Algorithm Implementations
class ConsensusAlgorithm(ABC):
    """Base class for consensus algorithms"""

    @abstractmethod
    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        """Reach consensus based on votes"""


class MajorityVoteConsensus(ConsensusAlgorithm):
    """Simple majority vote consensus"""

    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        """Reach consensus using majority vote"""
        vote_counts = defaultdict(int)

        for agent_id, vote in votes.items():
            vote_counts[str(vote)] += 1

        # Find majority
        def get_vote_count(x):
            return x[1]

        majority_vote = max(vote_counts.items(), key=get_vote_count)
        total_votes = sum(vote_counts.values())

        return {
            "consensus": majority_vote[0],
            "vote_count": majority_vote[1],
            "total_votes": total_votes,
            "confidence": majority_vote[1] / total_votes,
            "algorithm": "majority_vote",
        }


class WeightedVoteConsensus(ConsensusAlgorithm):
    """Weighted vote consensus based on trust scores"""

    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        """Reach consensus using weighted votes"""
        weighted_votes = defaultdict(float)
        total_weight = 0.0

        for agent_id, vote in votes.items():
            if agent_id in agent_registry:
                weight = agent_registry[agent_id].trust_score
                weighted_votes[str(vote)] += weight
                total_weight += weight

        # Find weighted majority
        if weighted_votes:
            def get_weighted_vote_count(x):
                return x[1]

            consensus_vote = max(weighted_votes.items(), key=get_weighted_vote_count)
            confidence = consensus_vote[1] / total_weight if total_weight > 0 else 0.0

            return {
                "consensus": consensus_vote[0],
                "weighted_score": consensus_vote[1],
                "total_weight": total_weight,
                "confidence": confidence,
                "algorithm": "weighted_vote",
            }

        return {"consensus": None, "confidence": 0.0, "algorithm": "weighted_vote"}


# Simplified implementations for other consensus algorithms
class ByzantineFaultTolerantConsensus(ConsensusAlgorithm):
    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified BFT - in practice would be much more complex
        return {
            "consensus": "bft_result",
            "confidence": 0.9,
            "algorithm": "byzantine_fault_tolerant",
        }


class RaftConsensus(ConsensusAlgorithm):
    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified Raft - in practice would implement leader election, log replication
        return {"consensus": "raft_result", "confidence": 0.95, "algorithm": "raft"}


class ProofOfStakeConsensus(ConsensusAlgorithm):
    async def reach_consensus(
        self, votes: Dict[str, Any], agents: List[str], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified PoS - in practice would implement stake-based selection
        return {"consensus": "pos_result", "confidence": 0.85, "algorithm": "proof_of_stake"}


# Swarm Behavior Implementations
class SwarmBehavior(ABC):
    """Base class for swarm behaviors"""

    @abstractmethod
    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        """Execute swarm behavior"""


class FlockingBehavior(SwarmBehavior):
    """Flocking behavior implementation"""

    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified flocking - separation, alignment, cohesion
        return {
            "behavior": "flocking",
            "participants": len(agents),
            "formation": "v_formation",
            "efficiency": 0.8,
            "confidence": 0.9,
        }


class ForagingBehavior(SwarmBehavior):
    """Foraging behavior implementation"""

    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified foraging - search and exploitation
        return {
            "behavior": "foraging",
            "participants": len(agents),
            "resources_found": np.secrets.randbelow(5, 20),
            "efficiency": 0.75,
            "confidence": 0.85,
        }


class ClusteringBehavior(SwarmBehavior):
    """Clustering behavior implementation"""

    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified clustering
        num_clusters = max(1, len(agents) // 3)
        return {
            "behavior": "clustering",
            "participants": len(agents),
            "clusters_formed": num_clusters,
            "efficiency": 0.7,
            "confidence": 0.8,
        }


class ExplorationBehavior(SwarmBehavior):
    """Exploration behavior implementation"""

    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified exploration
        return {
            "behavior": "exploration",
            "participants": len(agents),
            "area_covered": len(agents) * 10,
            "discoveries": np.secrets.randbelow(1, 5),
            "efficiency": 0.85,
            "confidence": 0.9,
        }


class DivisionOfLaborBehavior(SwarmBehavior):
    """Division of labor behavior implementation"""

    async def execute(
        self, agents: List[str], environment: Dict[str, Any], agent_registry: Dict[str, Agent]
    ) -> Dict[str, Any]:
        # Simplified division of labor
        specializations = len(
            set(
                spec
                for agent_id in agents
                if agent_id in agent_registry
                for spec in agent_registry[agent_id].specializations
            )
        )

        return {
            "behavior": "division_of_labor",
            "participants": len(agents),
            "specializations": specializations,
            "efficiency": min(1.0, specializations / len(agents)),
            "confidence": 0.8,
        }


    # Missing method implementations (stubs for now)
    async def _create_initial_connections(self, agent: Agent):
        """Create initial network connections for new agent"""
        logger.info(f"Creating initial connections for agent {agent.agent_id}")

        try:
            # Get agent capabilities
            capabilities = getattr(agent, 'capabilities', [])
            if isinstance(capabilities, str):
                capabilities = [capabilities]

            # Find agents with complementary capabilities
            compatible_agents = []
            for existing_agent_id, existing_agent in self.agent_registry.items():
                if existing_agent_id == agent.agent_id:
                    continue

                existing_capabilities = getattr(existing_agent, 'capabilities', [])
                if isinstance(existing_capabilities, str):
                    existing_capabilities = [existing_capabilities]

                # Check for capability overlap or complementary skills
                compatibility_score = self._calculate_compatibility_score(
                    capabilities, existing_capabilities
                )

                if compatibility_score > 0.3:  # Threshold for initial connection
                    compatible_agents.append({
                        'agent_id': existing_agent_id,
                        'score': compatibility_score,
                        'capabilities': existing_capabilities
                    })

            # Sort by compatibility and create connections to top agents
            def get_agent_compatibility_score(x):
                return x['score']

            compatible_agents.sort(key=get_agent_compatibility_score, reverse=True)
            max_initial_connections = min(5, len(compatible_agents))  # Limit initial connections

            for i in range(max_initial_connections):
                target_agent = compatible_agents[i]

                # Create bidirectional connection
                if agent.agent_id not in self.network_connections:
                    self.network_connections[agent.agent_id] = {}
                if target_agent['agent_id'] not in self.network_connections:
                    self.network_connections[target_agent['agent_id']] = {}

                # Set initial trust score based on compatibility
                initial_trust = min(0.7, target_agent['score'] + 0.2)

                self.network_connections[agent.agent_id][target_agent['agent_id']] = {
                    'trust_score': initial_trust,
                    'interaction_count': 0,
                    'success_rate': 0.5,  # Start neutral
                    'last_interaction': datetime.utcnow(),
                    'connection_strength': target_agent['score']
                }

                self.network_connections[target_agent['agent_id']][agent.agent_id] = {
                    'trust_score': initial_trust,
                    'interaction_count': 0,
                    'success_rate': 0.5,
                    'last_interaction': datetime.utcnow(),
                    'connection_strength': target_agent['score']
                }

                logger.info(f"Created connection: {agent.agent_id} <-> {target_agent['agent_id']} (trust: {initial_trust:.2f})")

            logger.info(f"Created {max_initial_connections} initial connections for agent {agent.agent_id}")

        except Exception as e:
            logger.error(f"Failed to create initial connections for {agent.agent_id}: {e}")

    def _calculate_compatibility_score(self, caps1: List[str], caps2: List[str]) -> float:
        """Calculate compatibility score between two sets of capabilities"""
        if not caps1 or not caps2:
            return 0.0

        # Convert to sets for easier comparison
        set1 = set(caps1)
        set2 = set(caps2)

        # Calculate Jaccard similarity for overlapping capabilities
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union

        # Add bonus for complementary capabilities
        complementary_bonus = 0.0
        complementary_pairs = [
            ('data_processing', 'data_analysis'),
            ('reasoning', 'validation'),
            ('synthesis', 'evaluation'),
            ('monitoring', 'alerting'),
            ('collection', 'storage'),
            ('computation', 'visualization')
        ]

        for cap1, cap2 in complementary_pairs:
            if (cap1 in set1 and cap2 in set2) or (cap2 in set1 and cap1 in set2):
                complementary_bonus += 0.2

        return min(1.0, jaccard_similarity + complementary_bonus)

    def _estimate_collaboration_duration(self, request: CollaborationRequest, participants: List[Dict[str, Any]]) -> float:
        """Estimate collaboration duration"""
        base_duration = 60.0  # Base 1 minute
        complexity_factor = len(participants) * 0.5
        return base_duration + complexity_factor

    async def _update_collaboration_metrics(self, result: CollaborationResult):
        """Update collaboration performance metrics"""
        logger.info(f"Updating metrics for collaboration {result.collaboration_id}")

        try:
            # Ensure metrics storage exists
            if not hasattr(self, 'collaboration_metrics'):
                self.collaboration_metrics = {
                    'total_collaborations': 0,
                    'successful_collaborations': 0,
                    'average_duration': 0.0,
                    'average_satisfaction': 0.0,
                    'participant_performance': defaultdict(list),
                    'collaboration_types': defaultdict(int),
                    'timeline': []
                }

            # Update overall metrics
            self.collaboration_metrics['total_collaborations'] += 1

            if result.success:
                self.collaboration_metrics['successful_collaborations'] += 1

            # Update duration metrics
            if result.duration is not None:
                current_avg = self.collaboration_metrics['average_duration']
                total_count = self.collaboration_metrics['total_collaborations']

                # Calculate rolling average
                self.collaboration_metrics['average_duration'] = (
                    (current_avg * (total_count - 1) + result.duration) / total_count
                )

            # Update satisfaction metrics
            if hasattr(result, 'satisfaction_score') and result.satisfaction_score is not None:
                current_avg_satisfaction = self.collaboration_metrics['average_satisfaction']
                total_count = self.collaboration_metrics['total_collaborations']

                self.collaboration_metrics['average_satisfaction'] = (
                    (current_avg_satisfaction * (total_count - 1) + result.satisfaction_score) / total_count
                )

            # Track participant performance
            for participant_id in result.participants:
                performance_data = {
                    'collaboration_id': result.collaboration_id,
                    'success': result.success,
                    'duration': result.duration,
                    'timestamp': datetime.utcnow(),
                    'satisfaction': getattr(result, 'satisfaction_score', 0.5)
                }

                self.collaboration_metrics['participant_performance'][participant_id].append(performance_data)

                # Keep only last 100 collaborations per participant
                if len(self.collaboration_metrics['participant_performance'][participant_id]) > 100:
                    self.collaboration_metrics['participant_performance'][participant_id] = (
                        self.collaboration_metrics['participant_performance'][participant_id][-100:]
                    )

                # Update network connections based on performance
                await self._update_connection_performance(participant_id, result.success)

            # Track collaboration types
            collab_type = getattr(result, 'collaboration_type', 'general')
            self.collaboration_metrics['collaboration_types'][collab_type] += 1

            # Add to timeline
            timeline_entry = {
                'timestamp': datetime.utcnow(),
                'collaboration_id': result.collaboration_id,
                'success': result.success,
                'participants': len(result.participants),
                'duration': result.duration,
                'type': collab_type
            }

            self.collaboration_metrics['timeline'].append(timeline_entry)

            # Keep only last 1000 timeline entries
            if len(self.collaboration_metrics['timeline']) > 1000:
                self.collaboration_metrics['timeline'] = self.collaboration_metrics['timeline'][-1000:]

            # Calculate and log current success rate
            success_rate = (
                self.collaboration_metrics['successful_collaborations'] /
                self.collaboration_metrics['total_collaborations']
            ) if self.collaboration_metrics['total_collaborations'] > 0 else 0.0

            logger.info(
                f"Collaboration metrics updated - "
                f"Total: {self.collaboration_metrics['total_collaborations']}, "
                f"Success rate: {success_rate:.2%}, "
                f"Avg duration: {self.collaboration_metrics['average_duration']:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to update collaboration metrics for {result.collaboration_id}: {e}")

    async def _update_connection_performance(self, agent_id: str, success: bool):
        """Update network connection performance based on collaboration result"""
        try:
            if agent_id in self.network_connections:
                for connected_agent_id in self.network_connections[agent_id]:
                    connection = self.network_connections[agent_id][connected_agent_id]

                    # Update interaction count
                    connection['interaction_count'] += 1

                    # Update success rate with exponential moving average
                    alpha = 0.1  # Learning rate
                    current_success_rate = connection.get('success_rate', 0.5)
                    new_success_rate = (
                        alpha * (1.0 if success else 0.0) +
                        (1 - alpha) * current_success_rate
                    )
                    connection['success_rate'] = new_success_rate

                    # Update trust score based on success rate
                    trust_adjustment = 0.05 if success else -0.03
                    connection['trust_score'] = max(0.0, min(1.0,
                        connection['trust_score'] + trust_adjustment
                    ))

                    connection['last_interaction'] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Failed to update connection performance for {agent_id}: {e}")

    async def _collect_votes(self, agents: List[str], proposals: List[Any]) -> Dict[str, Any]:
        """Collect votes from agents on proposals"""
        votes = {}
        for agent_id in agents:
            if proposals:
                # Simulate vote - in practice would query agent
                votes[agent_id] = proposals[0]
        return votes

    async def _apply_sharing_policy(self, knowledge: Dict[str, Any], policy: Optional[str]) -> Dict[str, Any]:
        """Apply knowledge sharing policy"""
        # Simple policy - return as-is for now
        return knowledge

    async def _send_knowledge(self, agent_id: str, knowledge: Dict[str, Any]) -> bool:
        """Send knowledge to target agent"""
        logger.info(f"Sending knowledge to agent {agent_id}")

        try:
            # Validate target agent exists
            if agent_id not in self.agent_registry:
                logger.warning(f"Target agent {agent_id} not found in registry")
                return False

            target_agent = self.agent_registry[agent_id]

            # Prepare knowledge message
            knowledge_message = {
                'type': 'knowledge_transfer',
                'source_agent': self.agent_id,
                'target_agent': agent_id,
                'timestamp': datetime.utcnow().isoformat(),
                'knowledge_id': f"knowledge_{datetime.utcnow().timestamp()}",
                'content': knowledge,
                'metadata': {
                    'knowledge_type': knowledge.get('type', 'general'),
                    'confidence_score': knowledge.get('confidence', 0.8),
                    'relevance_score': knowledge.get('relevance', 0.7),
                    'expiry': knowledge.get('expiry'),
                    'priority': knowledge.get('priority', 'normal')
                }
            }

            # Check if target agent has knowledge handling capability
            target_capabilities = getattr(target_agent, 'capabilities', [])
            if isinstance(target_capabilities, str):
                target_capabilities = [target_capabilities]

            knowledge_capable = any(cap in ['knowledge_processing', 'learning', 'reasoning', 'analysis']
                                  for cap in target_capabilities)

            if not knowledge_capable:
                logger.warning(f"Agent {agent_id} may not be capable of processing knowledge")

            # Try to send via direct method call if agent has receive_knowledge method
            if hasattr(target_agent, 'receive_knowledge'):
                try:
                    result = await target_agent.receive_knowledge(knowledge_message)
                    if result:
                        logger.info(f"Knowledge successfully sent to {agent_id} via direct method")
                        await self._update_knowledge_transfer_metrics(agent_id, True, 'direct_method')
                        return True
                except Exception as e:
                    logger.warning(f"Direct knowledge transfer failed: {e}")

            # Fallback: Store in agent's knowledge queue
            if not hasattr(self, 'knowledge_queues'):
                self.knowledge_queues = defaultdict(list)

            self.knowledge_queues[agent_id].append(knowledge_message)

            # Limit queue size
            if len(self.knowledge_queues[agent_id]) > 100:
                self.knowledge_queues[agent_id] = self.knowledge_queues[agent_id][-100:]

            # Try to notify agent of pending knowledge
            await self._notify_agent_of_pending_knowledge(agent_id)

            logger.info(f"Knowledge queued for agent {agent_id}")
            await self._update_knowledge_transfer_metrics(agent_id, True, 'queued')

            return True

        except Exception as e:
            logger.error(f"Failed to send knowledge to agent {agent_id}: {e}")
            await self._update_knowledge_transfer_metrics(agent_id, False, 'error')
            return False

    async def _notify_agent_of_pending_knowledge(self, agent_id: str):
        """Notify agent that knowledge is available"""
        try:
            target_agent = self.agent_registry.get(agent_id)
            if target_agent and hasattr(target_agent, 'notify_pending_knowledge'):
                await target_agent.notify_pending_knowledge()
        except Exception as e:
            logger.debug(f"Failed to notify agent {agent_id} of pending knowledge: {e}")

    async def _update_knowledge_transfer_metrics(self, agent_id: str, success: bool, method: str):
        """Update knowledge transfer metrics"""
        try:
            if not hasattr(self, 'knowledge_transfer_metrics'):
                self.knowledge_transfer_metrics = {
                    'total_transfers': 0,
                    'successful_transfers': 0,
                    'transfer_methods': defaultdict(int),
                    'agent_success_rates': defaultdict(self._default_success_rate)
                }

            self.knowledge_transfer_metrics['total_transfers'] += 1
            if success:
                self.knowledge_transfer_metrics['successful_transfers'] += 1
                self.knowledge_transfer_metrics['agent_success_rates'][agent_id]['successes'] += 1

            self.knowledge_transfer_metrics['agent_success_rates'][agent_id]['attempts'] += 1
            self.knowledge_transfer_metrics['transfer_methods'][method] += 1

        except Exception as e:
            logger.error(f"Failed to update knowledge transfer metrics: {e}")

    async def _update_network_connections(self, agent_id: str):
        """Update network connections based on trust scores"""
        logger.info(f"Updating network connections for agent {agent_id}")

        try:
            if agent_id not in self.network_connections:
                logger.warning(f"No network connections found for agent {agent_id}")
                return

            current_connections = self.network_connections[agent_id].copy()

            # Remove low-trust connections
            connections_to_remove = []
            for connected_agent_id, connection_data in current_connections.items():
                trust_score = connection_data.get('trust_score', 0.5)

                # Remove connections with very low trust
                if trust_score < 0.2:
                    connections_to_remove.append(connected_agent_id)
                    logger.info(f"Removing low-trust connection: {agent_id} -> {connected_agent_id} (trust: {trust_score:.2f})")

            # Remove bidirectional low-trust connections
            for connected_agent_id in connections_to_remove:
                # Remove from current agent's connections
                if connected_agent_id in self.network_connections[agent_id]:
                    del self.network_connections[agent_id][connected_agent_id]

                # Remove reverse connection
                if (connected_agent_id in self.network_connections and
                    agent_id in self.network_connections[connected_agent_id]):
                    del self.network_connections[connected_agent_id][agent_id]

            # Find potential new connections based on successful collaborations
            await self._discover_new_connections(agent_id)

            # Optimize existing connections by adjusting trust scores
            await self._optimize_connection_strengths(agent_id)

            # Update connection topology for better network efficiency
            await self._optimize_network_topology(agent_id)

            # Log network statistics
            connection_count = len(self.network_connections.get(agent_id, {}))
            avg_trust = self._calculate_average_trust(agent_id)

            logger.info(f"Network updated for {agent_id}: {connection_count} connections, avg trust: {avg_trust:.2f}")

        except Exception as e:
            logger.error(f"Failed to update network connections for {agent_id}: {e}")

    async def _discover_new_connections(self, agent_id: str):
        """Discover and create new beneficial connections"""
        try:
            # Find agents that frequently collaborate with our connections
            potential_connections = {}

            current_connections = self.network_connections.get(agent_id, {})

            # Look at second-degree connections (friends of friends)
            for connected_agent_id in current_connections.keys():
                if connected_agent_id in self.network_connections:
                    for second_degree_agent in self.network_connections[connected_agent_id]:
                        if (second_degree_agent != agent_id and
                            second_degree_agent not in current_connections):

                            # Calculate potential value of this connection
                            trust_via_connection = (
                                current_connections[connected_agent_id]['trust_score'] *
                                self.network_connections[connected_agent_id][second_degree_agent]['trust_score']
                            )

                            if second_degree_agent not in potential_connections:
                                potential_connections[second_degree_agent] = {'score': 0, 'recommendations': 0}

                            potential_connections[second_degree_agent]['score'] += trust_via_connection
                            potential_connections[second_degree_agent]['recommendations'] += 1

            # Create new connections for highly recommended agents
            for potential_agent_id, data in potential_connections.items():
                avg_recommendation_score = data['score'] / data['recommendations']

                if avg_recommendation_score > 0.6 and data['recommendations'] >= 2:
                    await self._create_new_connection(agent_id, potential_agent_id, avg_recommendation_score * 0.7)
                    logger.info(f"Created new connection: {agent_id} <-> {potential_agent_id} (recommended)")

        except Exception as e:
            logger.error(f"Failed to discover new connections for {agent_id}: {e}")

    async def _create_new_connection(self, agent1_id: str, agent2_id: str, initial_trust: float):
        """Create a new bidirectional connection between agents"""
        try:
            # Initialize connection data
            if agent1_id not in self.network_connections:
                self.network_connections[agent1_id] = {}
            if agent2_id not in self.network_connections:
                self.network_connections[agent2_id] = {}

            connection_data = {
                'trust_score': initial_trust,
                'interaction_count': 0,
                'success_rate': 0.5,
                'last_interaction': datetime.utcnow(),
                'connection_strength': initial_trust,
                'connection_type': 'discovered'
            }

            # Create bidirectional connection
            self.network_connections[agent1_id][agent2_id] = connection_data.copy()
            self.network_connections[agent2_id][agent1_id] = connection_data.copy()

        except Exception as e:
            logger.error(f"Failed to create connection between {agent1_id} and {agent2_id}: {e}")

    async def _optimize_connection_strengths(self, agent_id: str):
        """Optimize connection strengths based on recent performance"""
        try:
            if agent_id not in self.network_connections:
                return

            for connected_agent_id, connection in self.network_connections[agent_id].items():
                # Decay trust over time for inactive connections
                days_since_interaction = (
                    datetime.utcnow() - connection.get('last_interaction', datetime.utcnow())
                ).days

                if days_since_interaction > 7:  # No interaction for a week
                    decay_factor = max(0.9, 1.0 - (days_since_interaction * 0.01))
                    connection['trust_score'] *= decay_factor

                # Boost trust for frequently successful connections
                interaction_count = connection.get('interaction_count', 0)
                success_rate = connection.get('success_rate', 0.5)

                if interaction_count > 10 and success_rate > 0.8:
                    connection['trust_score'] = min(1.0, connection['trust_score'] * 1.05)

        except Exception as e:
            logger.error(f"Failed to optimize connection strengths for {agent_id}: {e}")

    async def _optimize_network_topology(self, agent_id: str):
        """Optimize network topology for better information flow"""
        try:
            if agent_id not in self.network_connections:
                return

            connections = self.network_connections[agent_id]

            # If agent has too many connections, prune the weakest ones
            max_connections = 20  # Reasonable limit for network efficiency

            if len(connections) > max_connections:
                # Sort connections by combined trust and activity score
                def get_connection_score(x):
                    return (
                        x[1].get('trust_score', 0) * 0.6 +
                        min(1.0, x[1].get('interaction_count', 0) / 100.0) * 0.4
                    )

                sorted_connections = sorted(
                    connections.items(),
                    key=get_connection_score,
                    reverse=True
                )

                # Keep only top connections
                connections_to_keep = dict(sorted_connections[:max_connections])
                connections_to_remove = set(connections.keys()) - set(connections_to_keep.keys())

                # Remove excess connections
                for connection_to_remove in connections_to_remove:
                    del self.network_connections[agent_id][connection_to_remove]

                    # Remove reverse connection
                    if (connection_to_remove in self.network_connections and
                        agent_id in self.network_connections[connection_to_remove]):
                        del self.network_connections[connection_to_remove][agent_id]

                logger.info(f"Pruned {len(connections_to_remove)} excess connections for {agent_id}")

        except Exception as e:
            logger.error(f"Failed to optimize network topology for {agent_id}: {e}")

    def _calculate_average_trust(self, agent_id: str) -> float:
        """Calculate average trust score for an agent's connections"""
        if agent_id not in self.network_connections or not self.network_connections[agent_id]:
            return 0.0

        trust_scores = [conn.get('trust_score', 0.5) for conn in self.network_connections[agent_id].values()]
        return sum(trust_scores) / len(trust_scores)


# Utility functions
def create_collaborative_intelligence_framework(
    agent_id: str,
) -> CollaborativeIntelligenceFramework:
    """Factory function to create collaborative intelligence framework"""
    return CollaborativeIntelligenceFramework(agent_id)


async def register_agent_in_network(
    framework: CollaborativeIntelligenceFramework, agent_id: str, capabilities: List[str]
) -> bool:
    """Register an agent in the collaborative network"""
    agent = Agent(agent_id=agent_id, name=f"Agent {agent_id}", capabilities=capabilities)
    return await framework.register_agent(agent)

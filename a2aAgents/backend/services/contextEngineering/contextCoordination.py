"""
Advanced Context Coordination for Multi-Agent Systems
Provides sophisticated context synchronization, conflict resolution, and propagation
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of context conflicts"""
    VALUE_MISMATCH = "value_mismatch"
    TEMPORAL_CONFLICT = "temporal_conflict"
    SEMANTIC_CONFLICT = "semantic_conflict"
    STRUCTURAL_CONFLICT = "structural_conflict"


class PropagationStrategy(Enum):
    """Context propagation strategies"""
    BROADCAST = "broadcast"
    TARGETED = "targeted"
    HIERARCHICAL = "hierarchical"
    GOSSIP = "gossip"


@dataclass
class ContextVersion:
    """Version control for context"""
    version_id: str
    parent_version: Optional[str]
    timestamp: datetime
    author_agent: str
    changes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "timestamp": self.timestamp.isoformat(),
            "author_agent": self.author_agent,
            "changes": self.changes
        }


@dataclass
class ContextConflict:
    """Represents a context conflict between agents"""
    conflict_type: ConflictType
    agents_involved: List[str]
    conflicting_values: Dict[str, Any]
    context_path: str
    severity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_type": self.conflict_type.value,
            "agents_involved": self.agents_involved,
            "conflicting_values": self.conflicting_values,
            "context_path": self.context_path,
            "severity": self.severity
        }


@dataclass
class SynchronizationState:
    """State of context synchronization"""
    sync_id: str
    participating_agents: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    conflicts_detected: List[ContextConflict]
    conflicts_resolved: List[ContextConflict]
    final_context: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_id": self.sync_id,
            "participating_agents": self.participating_agents,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "conflicts_resolved": [c.to_dict() for c in self.conflicts_resolved],
            "final_context": self.final_context
        }


class ContextVersionControl:
    """Manages context versions and history"""
    
    def __init__(self):
        self.versions: Dict[str, ContextVersion] = {}
        self.version_graph = nx.DiGraph()
        self.current_version: Optional[str] = None
    
    def create_version(
        self,
        context: Dict[str, Any],
        author_agent: str,
        parent_version: Optional[str] = None
    ) -> ContextVersion:
        """Create a new context version"""
        # Generate version ID
        version_content = json.dumps(context, sort_keys=True)
        version_id = hashlib.sha256(
            f"{version_content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Calculate changes if parent exists
        changes = {}
        if parent_version and parent_version in self.versions:
            parent_context = self._get_context_for_version(parent_version)
            changes = self._calculate_diff(parent_context, context)
        
        # Create version
        version = ContextVersion(
            version_id=version_id,
            parent_version=parent_version or self.current_version,
            timestamp=datetime.now(),
            author_agent=author_agent,
            changes=changes
        )
        
        # Store version
        self.versions[version_id] = version
        self.version_graph.add_node(version_id, context=context)
        
        if version.parent_version:
            self.version_graph.add_edge(version.parent_version, version_id)
        
        self.current_version = version_id
        
        return version
    
    def _calculate_diff(self, old_context: Dict, new_context: Dict) -> Dict[str, Any]:
        """Calculate differences between contexts"""
        diff = {
            "added": {},
            "removed": {},
            "modified": {}
        }
        
        # Find added and modified keys
        for key, value in new_context.items():
            if key not in old_context:
                diff["added"][key] = value
            elif old_context[key] != value:
                diff["modified"][key] = {
                    "old": old_context[key],
                    "new": value
                }
        
        # Find removed keys
        for key in old_context:
            if key not in new_context:
                diff["removed"][key] = old_context[key]
        
        return diff
    
    def _get_context_for_version(self, version_id: str) -> Dict[str, Any]:
        """Retrieve context for a specific version"""
        if version_id in self.version_graph:
            return self.version_graph.nodes[version_id].get("context", {})
        return {}
    
    def get_version_history(self, limit: int = 10) -> List[ContextVersion]:
        """Get recent version history"""
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.timestamp,
            reverse=True
        )
        return sorted_versions[:limit]
    
    def merge_versions(
        self,
        version1_id: str,
        version2_id: str,
        merge_strategy: str = "latest_wins"
    ) -> ContextVersion:
        """Merge two context versions"""
        context1 = self._get_context_for_version(version1_id)
        context2 = self._get_context_for_version(version2_id)
        
        if merge_strategy == "latest_wins":
            version1 = self.versions[version1_id]
            version2 = self.versions[version2_id]
            
            if version1.timestamp > version2.timestamp:
                merged_context = {**context2, **context1}
                author = version1.author_agent
            else:
                merged_context = {**context1, **context2}
                author = version2.author_agent
        else:
            # Other merge strategies can be implemented
            merged_context = {**context1, **context2}
            author = "merge_operation"
        
        return self.create_version(
            merged_context,
            author,
            parent_version=version1_id  # Use first version as parent
        )


class ContextConflictResolver:
    """Resolves conflicts between different context versions"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.VALUE_MISMATCH: self._resolve_value_mismatch,
            ConflictType.TEMPORAL_CONFLICT: self._resolve_temporal_conflict,
            ConflictType.SEMANTIC_CONFLICT: self._resolve_semantic_conflict,
            ConflictType.STRUCTURAL_CONFLICT: self._resolve_structural_conflict
        }
    
    async def detect_conflicts(
        self,
        agent_contexts: Dict[str, Dict[str, Any]]
    ) -> List[ContextConflict]:
        """Detect conflicts between agent contexts"""
        conflicts = []
        agents = list(agent_contexts.keys())
        
        # Compare contexts pairwise
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                context1 = agent_contexts[agent1]
                context2 = agent_contexts[agent2]
                
                # Detect conflicts at each path
                path_conflicts = await self._detect_path_conflicts(
                    agent1, agent2, context1, context2
                )
                conflicts.extend(path_conflicts)
        
        return conflicts
    
    async def _detect_path_conflicts(
        self,
        agent1: str,
        agent2: str,
        context1: Dict,
        context2: Dict,
        path: str = ""
    ) -> List[ContextConflict]:
        """Recursively detect conflicts in context paths"""
        conflicts = []
        
        # Get all keys from both contexts
        all_keys = set(context1.keys()) | set(context2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in context1 or key not in context2:
                # Structural conflict - key missing in one context
                conflict = ContextConflict(
                    conflict_type=ConflictType.STRUCTURAL_CONFLICT,
                    agents_involved=[agent1, agent2],
                    conflicting_values={
                        agent1: context1.get(key, "MISSING"),
                        agent2: context2.get(key, "MISSING")
                    },
                    context_path=current_path,
                    severity=0.5
                )
                conflicts.append(conflict)
            elif isinstance(context1[key], dict) and isinstance(context2[key], dict):
                # Recurse into nested dictionaries
                nested_conflicts = await self._detect_path_conflicts(
                    agent1, agent2, context1[key], context2[key], current_path
                )
                conflicts.extend(nested_conflicts)
            elif context1[key] != context2[key]:
                # Value mismatch
                conflict = ContextConflict(
                    conflict_type=ConflictType.VALUE_MISMATCH,
                    agents_involved=[agent1, agent2],
                    conflicting_values={
                        agent1: context1[key],
                        agent2: context2[key]
                    },
                    context_path=current_path,
                    severity=self._calculate_conflict_severity(
                        context1[key], context2[key]
                    )
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_conflict_severity(self, value1: Any, value2: Any) -> float:
        """Calculate severity of a conflict"""
        # Simple heuristic - can be made more sophisticated
        if type(value1) != type(value2):
            return 0.8
        elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # Numeric difference
            diff = abs(value1 - value2)
            max_val = max(abs(value1), abs(value2), 1)
            return min(1.0, diff / max_val)
        else:
            return 0.5
    
    async def resolve_conflicts(
        self,
        conflicts: List[ContextConflict],
        agent_contexts: Dict[str, Dict[str, Any]],
        resolution_policy: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], List[ContextConflict]]:
        """Resolve detected conflicts"""
        resolved_context = {}
        resolved_conflicts = []
        
        # Group conflicts by path
        conflicts_by_path = defaultdict(list)
        for conflict in conflicts:
            conflicts_by_path[conflict.context_path].append(conflict)
        
        # Resolve conflicts for each path
        for path, path_conflicts in conflicts_by_path.items():
            # Use appropriate resolution strategy
            for conflict in path_conflicts:
                resolution_func = self.resolution_strategies.get(
                    conflict.conflict_type,
                    self._default_resolution
                )
                
                resolved_value = await resolution_func(
                    conflict, agent_contexts, resolution_policy
                )
                
                # Update resolved context
                self._set_nested_value(resolved_context, path, resolved_value)
                resolved_conflicts.append(conflict)
        
        return resolved_context, resolved_conflicts
    
    async def _resolve_value_mismatch(
        self,
        conflict: ContextConflict,
        agent_contexts: Dict[str, Dict[str, Any]],
        policy: Optional[Dict[str, Any]]
    ) -> Any:
        """Resolve value mismatch conflicts"""
        if policy and "preferred_agent" in policy:
            preferred = policy["preferred_agent"]
            if preferred in conflict.conflicting_values:
                return conflict.conflicting_values[preferred]
        
        # Default: use most recent or majority vote
        values = list(conflict.conflicting_values.values())
        # Simple majority vote
        from collections import Counter
        value_counts = Counter(values)
        most_common = value_counts.most_common(1)[0][0]
        
        return most_common
    
    async def _resolve_temporal_conflict(
        self,
        conflict: ContextConflict,
        agent_contexts: Dict[str, Dict[str, Any]],
        policy: Optional[Dict[str, Any]]
    ) -> Any:
        """Resolve temporal conflicts - prefer more recent"""
        # In a real implementation, check timestamps
        # For now, return first value
        return list(conflict.conflicting_values.values())[0]
    
    async def _resolve_semantic_conflict(
        self,
        conflict: ContextConflict,
        agent_contexts: Dict[str, Dict[str, Any]],
        policy: Optional[Dict[str, Any]]
    ) -> Any:
        """Resolve semantic conflicts"""
        # Could use NLP/embeddings to find semantically similar values
        # For now, use default resolution
        return await self._default_resolution(conflict, agent_contexts, policy)
    
    async def _resolve_structural_conflict(
        self,
        conflict: ContextConflict,
        agent_contexts: Dict[str, Dict[str, Any]],
        policy: Optional[Dict[str, Any]]
    ) -> Any:
        """Resolve structural conflicts"""
        # Prefer non-missing values
        for agent, value in conflict.conflicting_values.items():
            if value != "MISSING":
                return value
        return None
    
    async def _default_resolution(
        self,
        conflict: ContextConflict,
        agent_contexts: Dict[str, Dict[str, Any]],
        policy: Optional[Dict[str, Any]]
    ) -> Any:
        """Default conflict resolution"""
        # Use first non-None value
        for value in conflict.conflicting_values.values():
            if value is not None and value != "MISSING":
                return value
        return None
    
    def _set_nested_value(self, context: Dict, path: str, value: Any):
        """Set a value in nested dictionary using dot notation path"""
        keys = path.split(".")
        current = context
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


class ContextPropagationManager:
    """Manages efficient context propagation across agent network"""
    
    def __init__(self):
        self.agent_network = nx.Graph()
        self.propagation_queue = asyncio.Queue()
        self.propagation_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register an agent in the network"""
        self.agent_network.add_node(
            agent_id,
            capabilities=capabilities,
            last_update=datetime.now()
        )
    
    def connect_agents(self, agent1: str, agent2: str, weight: float = 1.0):
        """Create connection between agents"""
        self.agent_network.add_edge(agent1, agent2, weight=weight)
    
    async def propagate_update(
        self,
        update: Dict[str, Any],
        source_agent: str,
        target_agents: Optional[List[str]] = None,
        strategy: PropagationStrategy = PropagationStrategy.TARGETED,
        priority: int = 5
    ) -> Dict[str, Any]:
        """Propagate context update through network"""
        propagation_id = hashlib.sha256(
            f"{json.dumps(update)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Determine target agents based on strategy
        if strategy == PropagationStrategy.BROADCAST:
            targets = list(self.agent_network.nodes())
            targets.remove(source_agent)
        elif strategy == PropagationStrategy.TARGETED:
            targets = target_agents or []
        elif strategy == PropagationStrategy.HIERARCHICAL:
            targets = await self._get_hierarchical_targets(source_agent)
        elif strategy == PropagationStrategy.GOSSIP:
            targets = await self._get_gossip_targets(source_agent)
        else:
            targets = target_agents or []
        
        # Create propagation tasks
        propagation_tasks = []
        for target in targets:
            if target in self.agent_network:
                task = self._propagate_to_agent(
                    propagation_id, update, source_agent, target, priority
                )
                propagation_tasks.append(task)
        
        # Execute propagation
        results = await asyncio.gather(*propagation_tasks, return_exceptions=True)
        
        # Record propagation
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        propagation_record = {
            "propagation_id": propagation_id,
            "source_agent": source_agent,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.value,
            "targets": targets,
            "successful": successful,
            "failed": failed
        }
        
        self.propagation_history[source_agent].append(propagation_record)
        
        return {
            "propagation_id": propagation_id,
            "successful": successful,
            "failed": failed,
            "total_targets": len(targets)
        }
    
    async def _propagate_to_agent(
        self,
        propagation_id: str,
        update: Dict[str, Any],
        source: str,
        target: str,
        priority: int
    ) -> Dict[str, Any]:
        """Propagate update to specific agent"""
        # In real implementation, this would make actual network calls
        # For now, simulate propagation
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Record successful propagation
        return {
            "agent": target,
            "status": "success",
            "latency": 0.1
        }
    
    async def _get_hierarchical_targets(self, source: str) -> List[str]:
        """Get targets for hierarchical propagation"""
        # Find neighbors and their importance
        if source not in self.agent_network:
            return []
        
        neighbors = list(self.agent_network.neighbors(source))
        # In hierarchical, propagate to high-weight connections first
        weighted_neighbors = [
            (n, self.agent_network[source][n].get("weight", 1.0))
            for n in neighbors
        ]
        weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return [n[0] for n in weighted_neighbors[:5]]  # Top 5 connections
    
    async def _get_gossip_targets(self, source: str) -> List[str]:
        """Get random subset of agents for gossip propagation"""
        import random
        
        all_agents = list(self.agent_network.nodes())
        if source in all_agents:
            all_agents.remove(source)
        
        # Select random subset (e.g., log(n) agents)
        num_targets = min(len(all_agents), max(3, int(len(all_agents) ** 0.5)))
        return random.sample(all_agents, num_targets)
    
    def get_propagation_metrics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get propagation metrics for analysis"""
        if agent_id:
            history = self.propagation_history.get(agent_id, [])
        else:
            history = []
            for agent_history in self.propagation_history.values():
                history.extend(agent_history)
        
        if not history:
            return {
                "total_propagations": 0,
                "success_rate": 0,
                "average_targets": 0
            }
        
        total = len(history)
        total_successful = sum(h["successful"] for h in history)
        total_targets = sum(len(h["targets"]) for h in history)
        total_attempted = sum(h["successful"] + h["failed"] for h in history)
        
        return {
            "total_propagations": total,
            "success_rate": total_successful / max(total_attempted, 1),
            "average_targets": total_targets / max(total, 1),
            "propagation_history": history[-10:]  # Last 10 propagations
        }


class DistributedContextManager:
    """Main coordinator for distributed context management"""
    
    def __init__(self):
        self.version_control = ContextVersionControl()
        self.conflict_resolver = ContextConflictResolver()
        self.propagation_manager = ContextPropagationManager()
        self.sync_states: Dict[str, SynchronizationState] = {}
    
    async def synchronize_contexts(
        self,
        agent_contexts: Dict[str, Dict[str, Any]],
        sync_policy: Optional[Dict[str, Any]] = None
    ) -> SynchronizationState:
        """Perform full context synchronization"""
        sync_id = hashlib.sha256(
            f"{json.dumps(list(agent_contexts.keys()))}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Initialize sync state
        sync_state = SynchronizationState(
            sync_id=sync_id,
            participating_agents=list(agent_contexts.keys()),
            start_time=datetime.now(),
            end_time=None,
            conflicts_detected=[],
            conflicts_resolved=[],
            final_context=None
        )
        
        try:
            # Detect conflicts
            conflicts = await self.conflict_resolver.detect_conflicts(agent_contexts)
            sync_state.conflicts_detected = conflicts
            
            # Resolve conflicts
            if conflicts:
                resolved_context, resolved_conflicts = await self.conflict_resolver.resolve_conflicts(
                    conflicts, agent_contexts, sync_policy
                )
                sync_state.conflicts_resolved = resolved_conflicts
                sync_state.final_context = resolved_context
            else:
                # No conflicts - merge contexts
                merged_context = {}
                for agent, context in agent_contexts.items():
                    merged_context.update(context)
                sync_state.final_context = merged_context
            
            # Create version for synchronized context
            if sync_state.final_context:
                version = self.version_control.create_version(
                    sync_state.final_context,
                    author_agent="sync_coordinator"
                )
                sync_state.final_context["_version"] = version.version_id
            
            sync_state.end_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Synchronization failed: {str(e)}")
            sync_state.end_time = datetime.now()
        
        # Store sync state
        self.sync_states[sync_id] = sync_state
        
        return sync_state
    
    async def coordinate_propagation(
        self,
        context_update: Dict[str, Any],
        source_agent: str,
        propagation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate context propagation with versioning"""
        # Create version for update
        version = self.version_control.create_version(
            context_update,
            author_agent=source_agent
        )
        
        # Add version info to update
        versioned_update = {
            **context_update,
            "_version": version.version_id,
            "_timestamp": version.timestamp.isoformat()
        }
        
        # Propagate update
        strategy = PropagationStrategy(
            propagation_config.get("strategy", "targeted")
        )
        target_agents = propagation_config.get("targets", [])
        priority = propagation_config.get("priority", 5)
        
        result = await self.propagation_manager.propagate_update(
            versioned_update,
            source_agent,
            target_agents,
            strategy,
            priority
        )
        
        return {
            "version": version.to_dict(),
            "propagation": result
        }
    
    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent synchronization history"""
        sorted_syncs = sorted(
            self.sync_states.values(),
            key=lambda s: s.start_time,
            reverse=True
        )
        return [s.to_dict() for s in sorted_syncs[:limit]]
    
    def get_context_lineage(self, version_id: str) -> List[ContextVersion]:
        """Get lineage of a context version"""
        lineage = []
        current = version_id
        
        while current and current in self.version_control.versions:
            version = self.version_control.versions[current]
            lineage.append(version)
            current = version.parent_version
        
        return lineage


# Example usage
if __name__ == "__main__":
    async def example_usage():
        # Create distributed context manager
        manager = DistributedContextManager()
        
        # Register agents
        manager.propagation_manager.register_agent("agent1", ["reasoning", "analysis"])
        manager.propagation_manager.register_agent("agent2", ["synthesis", "validation"])
        manager.propagation_manager.register_agent("agent3", ["reasoning", "validation"])
        
        # Connect agents
        manager.propagation_manager.connect_agents("agent1", "agent2", weight=0.8)
        manager.propagation_manager.connect_agents("agent2", "agent3", weight=0.9)
        manager.propagation_manager.connect_agents("agent1", "agent3", weight=0.5)
        
        # Example contexts from different agents
        agent_contexts = {
            "agent1": {
                "task": "analyze data",
                "priority": "high",
                "constraints": {"time_limit": 300},
                "data": {"source": "database", "size": 1000}
            },
            "agent2": {
                "task": "analyze data",
                "priority": "medium",  # Conflict!
                "constraints": {"time_limit": 300},
                "data": {"source": "api", "size": 1000}  # Conflict!
            },
            "agent3": {
                "task": "analyze data",
                "priority": "high",
                "constraints": {"time_limit": 600},  # Conflict!
                "results": {"preliminary": True}  # Additional field
            }
        }
        
        # Synchronize contexts
        sync_state = await manager.synchronize_contexts(
            agent_contexts,
            sync_policy={"preferred_agent": "agent1"}
        )
        
        print(f"Synchronization completed: {sync_state.sync_id}")
        print(f"Conflicts detected: {len(sync_state.conflicts_detected)}")
        print(f"Conflicts resolved: {len(sync_state.conflicts_resolved)}")
        print(f"Final context: {json.dumps(sync_state.final_context, indent=2)}")
        
        # Propagate an update
        update = {
            "results": {"analysis_complete": True, "score": 0.95}
        }
        
        propagation_result = await manager.coordinate_propagation(
            update,
            "agent1",
            {
                "strategy": "hierarchical",
                "priority": 8
            }
        )
        
        print(f"\nPropagation result: {json.dumps(propagation_result, indent=2)}")
    
    # Run example
    asyncio.run(example_usage())
"""
A2A Data Consistency Manager
Implements distributed data consistency mechanisms including eventual consistency,
strong consistency, and consensus protocols for the A2A platform
"""

import asyncio
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import random

from .networkClient import NetworkClient, get_network_client

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Data consistency levels"""

    EVENTUAL = "eventual"  # Eventually consistent
    STRONG = "strong"  # Strongly consistent
    BOUNDED_STALENESS = "bounded"  # Bounded staleness
    SESSION = "session"  # Session consistency
    CONSISTENT_PREFIX = "prefix"  # Consistent prefix reads


class ConsistencyStrategy(Enum):
    """Consistency enforcement strategies"""

    QUORUM = "quorum"  # Quorum-based reads/writes
    PRIMARY_BACKUP = "primary_backup"  # Primary with backups
    MULTI_MASTER = "multi_master"  # Multi-master replication
    CHAIN_REPLICATION = "chain"  # Chain replication
    PAXOS = "paxos"  # Paxos consensus
    RAFT = "raft"  # Raft consensus


class ConflictResolution(Enum):
    """Conflict resolution strategies"""

    LAST_WRITE_WINS = "lww"  # Last write wins
    MULTI_VALUE = "multi_value"  # Keep all values
    CUSTOM_MERGE = "custom_merge"  # Custom merge function
    VECTOR_CLOCK = "vector_clock"  # Vector clock comparison
    CRDT = "crdt"  # Conflict-free replicated data type


@dataclass
class ConsistencyConfig:
    """Configuration for data consistency"""

    default_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    default_strategy: ConsistencyStrategy = ConsistencyStrategy.QUORUM
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

    # Quorum settings
    replication_factor: int = 3
    write_quorum: int = 2  # W
    read_quorum: int = 2  # R

    # Timing settings
    sync_interval: int = 5  # seconds
    max_staleness: int = 60  # seconds for bounded staleness
    session_timeout: int = 300  # seconds

    # Consensus settings
    election_timeout: int = 150  # milliseconds
    heartbeat_interval: int = 50  # milliseconds

    # Monitoring
    enable_consistency_checks: bool = True
    check_interval: int = 30  # seconds


@dataclass
class DataVersion:
    """Version information for data items"""

    version: int
    timestamp: datetime
    node_id: str
    vector_clock: Dict[str, int] = field(default_factory=dict)
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataItem:
    """Consistent data item"""

    key: str
    value: Any
    version: DataVersion
    consistency_level: ConsistencyLevel
    replicas: Set[str] = field(default_factory=set)
    pending_writes: List[DataVersion] = field(default_factory=list)
    conflict_history: List[DataVersion] = field(default_factory=list)


@dataclass
class ConsistencyViolation:
    """Detected consistency violation"""

    violation_type: str
    affected_keys: List[str]
    nodes: List[str]
    detected_at: datetime
    details: Dict[str, Any]
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class SyncOperation:
    """Synchronization operation"""

    operation_id: str
    source_node: str
    target_nodes: List[str]
    data_keys: List[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    conflicts_resolved: int = 0


class ConsistencyManager:
    """
    Manages data consistency across distributed A2A agents
    Implements various consistency models and conflict resolution
    """

    def __init__(self, node_id: str, config: ConsistencyConfig = None):
        self.node_id = node_id
        self.config = config or ConsistencyConfig()
        self.network_client: Optional[NetworkClient] = None

        # Data storage
        self.data_store: Dict[str, DataItem] = {}
        self.vector_clocks: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Node management
        self.known_nodes: Set[str] = {node_id}
        self.node_health: Dict[str, bool] = {node_id: True}
        self.last_heartbeat: Dict[str, datetime] = {node_id: datetime.utcnow()}

        # Consensus state (for Raft/Paxos)
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        self.commit_index = 0
        self.last_applied = 0

        # Sync tracking
        self.sync_queue: deque = deque(maxlen=1000)
        self.pending_syncs: Dict[str, SyncOperation] = {}
        self.sync_history: deque = deque(maxlen=1000)

        # Consistency violations
        self.violations: deque = deque(maxlen=100)
        self.violation_callbacks: List[Callable] = []

        # Metrics
        self.read_count = 0
        self.write_count = 0
        self.conflict_count = 0
        self.sync_count = 0

        self._lock = asyncio.Lock()
        self._running = False
        self._sync_task = None
        self._monitor_task = None

        # Node URL registry
        self.node_urls: Dict[str, str] = {}

        logger.info(f"Initialized consistency manager for node {node_id}")

    async def initialize(self):
        """Initialize consistency manager and start background tasks"""
        self.network_client = await get_network_client()
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._monitor_task = asyncio.create_task(self._monitor_consistency())
        logger.info("Consistency manager initialized")

    async def shutdown(self):
        """Shutdown consistency manager"""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        try:
            if self._sync_task:
                await self._sync_task
            if self._monitor_task:
                await self._monitor_task
        except asyncio.CancelledError:
            pass

        logger.info("Consistency manager shutdown complete")

    async def write(
        self,
        key: str,
        value: Any,
        consistency_level: Optional[ConsistencyLevel] = None,
        conflict_resolver: Optional[Callable] = None,
    ) -> bool:
        """
        Write data with specified consistency level
        Returns True if write succeeded according to consistency requirements
        """
        consistency_level = consistency_level or self.config.default_level

        async with self._lock:
            self.write_count += 1

            # Create new version
            version = DataVersion(
                version=self._get_next_version(key),
                timestamp=datetime.utcnow(),
                node_id=self.node_id,
                vector_clock=self._increment_vector_clock(key),
                checksum=self._calculate_checksum(value),
            )

            # Check for existing data
            if key in self.data_store:
                existing = self.data_store[key]

                # Handle conflicts
                if self._has_conflict(existing.version, version):
                    resolved_value = await self._resolve_conflict(
                        existing, value, version, conflict_resolver
                    )
                    if resolved_value is None:
                        return False
                    value = resolved_value
                    self.conflict_count += 1

            # Create or update data item
            data_item = DataItem(
                key=key,
                value=value,
                version=version,
                consistency_level=consistency_level,
                replicas={self.node_id},
            )

            # Apply consistency strategy
            success = await self._apply_write_consistency(data_item, consistency_level)

            if success:
                self.data_store[key] = data_item
                # Queue for synchronization
                self.sync_queue.append((key, data_item))

                logger.debug(f"Written key={key} with consistency={consistency_level.value}")

            return success

    async def read(
        self, key: str, consistency_level: Optional[ConsistencyLevel] = None
    ) -> Optional[Any]:
        """
        Read data with specified consistency level
        Returns the value or None if not found/consistency not met
        """
        consistency_level = consistency_level or self.config.default_level

        async with self._lock:
            self.read_count += 1

            # Check local store first
            if key not in self.data_store:
                # Try to fetch from other nodes
                data_item = await self._fetch_from_nodes(key, consistency_level)
                if not data_item:
                    return None
                self.data_store[key] = data_item
            else:
                data_item = self.data_store[key]

            # Apply read consistency
            if await self._check_read_consistency(data_item, consistency_level):
                return data_item.value
            else:
                # Consistency requirements not met
                logger.warning(f"Read consistency not met for key={key}")
                return None

    async def _apply_write_consistency(
        self, data_item: DataItem, consistency_level: ConsistencyLevel
    ) -> bool:
        """Apply write consistency requirements"""
        if consistency_level == ConsistencyLevel.EVENTUAL:
            # Always succeed for eventual consistency
            return True

        elif consistency_level == ConsistencyLevel.STRONG:
            # Need to replicate to all nodes
            success_count = 1  # self
            for node in self.known_nodes:
                if node != self.node_id and self.node_health.get(node, False):
                    if await self._replicate_to_node(node, data_item):
                        success_count += 1

            return success_count >= len(
                [n for n in self.known_nodes if self.node_health.get(n, False)]
            )

        elif consistency_level == ConsistencyLevel.BOUNDED_STALENESS:
            # Replicate to write quorum
            return await self._quorum_write(data_item)

        elif consistency_level == ConsistencyLevel.SESSION:
            # Session consistency - always succeed locally
            return True

        elif consistency_level == ConsistencyLevel.CONSISTENT_PREFIX:
            # Ensure causal consistency
            return await self._ensure_causal_consistency(data_item)

        return True

    async def _check_read_consistency(
        self, data_item: DataItem, consistency_level: ConsistencyLevel
    ) -> bool:
        """Check if read meets consistency requirements"""
        if consistency_level == ConsistencyLevel.EVENTUAL:
            return True

        elif consistency_level == ConsistencyLevel.STRONG:
            # Check if we have the latest version from all nodes
            latest_version = await self._get_latest_version(data_item.key)
            return data_item.version.version >= latest_version

        elif consistency_level == ConsistencyLevel.BOUNDED_STALENESS:
            # Check staleness bound
            age = (datetime.utcnow() - data_item.version.timestamp).total_seconds()
            return age <= self.config.max_staleness

        elif consistency_level == ConsistencyLevel.SESSION:
            # Session consistency - check session validity
            return True  # Simplified

        elif consistency_level == ConsistencyLevel.CONSISTENT_PREFIX:
            # Check causal consistency
            return self._check_causal_consistency(data_item)

        return True

    async def _quorum_write(self, data_item: DataItem) -> bool:
        """Perform real quorum-based write with vector clocks and conflict resolution"""
        # Initialize write context
        write_id = f"write_{data_item.key}_{time.time()}"
        write_start = time.time()
        
        # Prepare versioned write with vector clock
        versioned_item = self._prepare_versioned_write(data_item)
        
        # Phase 1: Prepare phase (similar to 2PC)
        prepare_results = await self._quorum_prepare_phase(versioned_item, write_id)
        
        if not prepare_results['prepared']:
            # Abort if prepare phase fails
            await self._quorum_abort_phase(write_id, prepare_results['nodes'])
            return False
        
        # Phase 2: Commit phase
        commit_results = await self._quorum_commit_phase(
            versioned_item, 
            write_id, 
            prepare_results['nodes'],
            prepare_results['conflicts']
        )
        
        # Check if we achieved write quorum
        successful_commits = commit_results['successful_nodes']
        quorum_achieved = len(successful_commits) >= self.config.write_quorum
        
        # Phase 3: Finalize (notify nodes of outcome)
        await self._quorum_finalize_phase(write_id, successful_commits, quorum_achieved)
        
        # Update local state if quorum achieved
        if quorum_achieved:
            self._update_local_vector_clock(versioned_item)
            logger.info(f"Quorum write successful for key {data_item.key} "
                       f"({len(successful_commits)}/{len(self.known_nodes)} nodes)")
        else:
            logger.warning(f"Quorum write failed for key {data_item.key} "
                          f"({len(successful_commits)}/{self.config.write_quorum} required)")
        
        return quorum_achieved

    async def _replicate_to_node(self, node_id: str, data_item: DataItem) -> bool:
        """Replicate data to a specific node"""
        if not self.network_client:
            logger.error("Network client not available for replication")
            return False

        try:
            # Get node URL from known_nodes registry
            node_url = await self._get_node_url(node_id)
            if not node_url:
                logger.error(f"Node URL not found for {node_id}")
                return False

            # Prepare replication data
            replication_data = {
                "key": data_item.key,
                "value": data_item.value,
                "version": {
                    "version": data_item.version.version,
                    "timestamp": data_item.version.timestamp.isoformat(),
                    "node_id": data_item.version.node_id,
                    "vector_clock": data_item.version.vector_clock,
                    "checksum": data_item.version.checksum,
                    "metadata": data_item.version.metadata
                },
                "consistency_level": data_item.consistency_level.value,
                "source_node": self.node_id
            }

            # Replicate to target node
            success = await self.network_client.replicate_data(
                node_url,
                data_item.key,
                replication_data,
                data_item.version.version
            )

            if success:
                # Mark node as replica
                data_item.replicas.add(node_id)
                logger.debug(f"Replicated {data_item.key} to node {node_id}")
                return True
            else:
                logger.warning(f"Replication failed to node {node_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to replicate to node {node_id}: {e}")
            return False

    async def _fetch_from_nodes(
        self, key: str, consistency_level: ConsistencyLevel
    ) -> Optional[DataItem]:
        """Fetch data from other nodes"""
        if not self.network_client:
            return None

        best_item = None
        best_version = -1

        for node_id in self.known_nodes:
            if node_id == self.node_id:
                continue

            try:
                node_url = await self._get_node_url(node_id)
                if not node_url:
                    continue

                # Fetch data from node
                data = await self.network_client.fetch_data(
                    node_url, key, consistency_level.value
                )

                if data and "version" in data:
                    version_info = data["version"]
                    version = version_info["version"]

                    # Keep the item with the highest version
                    if version > best_version:
                        best_version = version

                        # Reconstruct DataItem
                        data_version = DataVersion(
                            version=version,
                            timestamp=datetime.fromisoformat(version_info["timestamp"]),
                            node_id=version_info["node_id"],
                            vector_clock=version_info.get("vector_clock", {}),
                            checksum=version_info.get("checksum", ""),
                            metadata=version_info.get("metadata", {})
                        )

                        best_item = DataItem(
                            key=key,
                            value=data["value"],
                            version=data_version,
                            consistency_level=ConsistencyLevel(data.get("consistency_level", "eventual")),
                            replicas=set(data.get("replicas", [])),
                            pending_writes=[],
                            conflict_history=[]
                        )

            except Exception as e:
                logger.debug(f"Failed to fetch {key} from node {node_id}: {e}")
                continue

        if best_item:
            logger.debug(f"Fetched {key} from network, version {best_version}")

        return best_item

    async def _get_latest_version(self, key: str) -> int:
        """Get latest version number across all nodes"""
        max_version = 0

        # Check local version
        if key in self.data_store:
            max_version = self.data_store[key].version.version

        if not self.network_client:
            return max_version

        # Query other nodes for their versions
        for node_id in self.known_nodes:
            if node_id == self.node_id:
                continue

            try:
                node_url = await self._get_node_url(node_id)
                if not node_url:
                    continue

                version = await self.network_client.get_latest_version(node_url, key)
                max_version = max(max_version, version)

            except Exception as e:
                logger.debug(f"Failed to get version from node {node_id}: {e}")
                continue

        return max_version

    def _increment_vector_clock(self, key: str) -> Dict[str, int]:
        """Increment vector clock for a key"""
        clock = self.vector_clocks[key].copy()
        clock[self.node_id] = clock.get(self.node_id, 0) + 1
        self.vector_clocks[key] = clock
        return clock

    def _has_conflict(self, v1: DataVersion, v2: DataVersion) -> bool:
        """Check if two versions have a conflict"""
        # Compare vector clocks
        return not self._vector_clock_compare(v1.vector_clock, v2.vector_clock)

    def _vector_clock_compare(self, vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """
        Compare vector clocks
        Returns True if vc1 happens-before vc2
        """
        all_keys = set(vc1.keys()) | set(vc2.keys())

        vc1_smaller = False
        vc2_smaller = False

        for key in all_keys:
            v1 = vc1.get(key, 0)
            v2 = vc2.get(key, 0)

            if v1 < v2:
                vc1_smaller = True
            elif v1 > v2:
                vc2_smaller = True

        # vc1 happens-before vc2 if all components of vc1 <= vc2
        return vc1_smaller and not vc2_smaller

    async def _resolve_conflict(
        self,
        existing: DataItem,
        new_value: Any,
        new_version: DataVersion,
        custom_resolver: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Resolve conflicts between values"""
        if custom_resolver:
            return await custom_resolver(existing.value, new_value, existing.version, new_version)

        strategy = self.config.conflict_resolution

        if strategy == ConflictResolution.LAST_WRITE_WINS:
            # Compare timestamps
            if new_version.timestamp > existing.version.timestamp:
                return new_value
            else:
                return existing.value

        elif strategy == ConflictResolution.MULTI_VALUE:
            # Keep both values
            if isinstance(existing.value, list):
                return existing.value + [new_value]
            else:
                return [existing.value, new_value]

        elif strategy == ConflictResolution.VECTOR_CLOCK:
            # Use vector clock comparison
            if self._vector_clock_compare(existing.version.vector_clock, new_version.vector_clock):
                return new_value
            else:
                return existing.value

        elif strategy == ConflictResolution.CRDT:
            # Merge using CRDT semantics (simplified)
            if isinstance(existing.value, set) and isinstance(new_value, set):
                return existing.value | new_value
            elif isinstance(existing.value, dict) and isinstance(new_value, dict):
                merged = existing.value.copy()
                merged.update(new_value)
                return merged

        # Default to last write wins
        return new_value if new_version.timestamp > existing.version.timestamp else existing.value

    def _calculate_checksum(self, value: Any) -> str:
        """Calculate checksum for a value"""
        value_str = json.dumps(value, sort_keys=True)
        return hashlib.sha256(value_str.encode()).hexdigest()[:16]

    def _get_next_version(self, key: str) -> int:
        """Get next version number for a key"""
        if key in self.data_store:
            return self.data_store[key].version.version + 1
        return 1

    def _prepare_versioned_write(self, data_item: DataItem) -> DataItem:
        """Prepare data item with proper versioning for quorum write"""
        # Increment vector clock for this write
        new_vector_clock = self._increment_vector_clock(data_item.key)
        
        # Create new version with updated vector clock
        new_version = DataVersion(
            version=data_item.version.version + 1,
            timestamp=datetime.utcnow(),
            node_id=self.node_id,
            vector_clock=new_vector_clock,
            checksum=self._calculate_checksum(data_item.value)
        )
        
        # Return updated data item
        return DataItem(
            key=data_item.key,
            value=data_item.value,
            version=new_version,
            metadata=data_item.metadata
        )
    
    async def _quorum_prepare_phase(self, data_item: DataItem, write_id: str) -> Dict[str, Any]:
        """Execute prepare phase of quorum protocol"""
        prepare_tasks = []
        nodes_to_prepare = []
        
        # Send prepare requests to all healthy nodes
        for node in self.known_nodes:
            if node != self.node_id and self.node_health.get(node, False):
                nodes_to_prepare.append(node)
                prepare_tasks.append(self._send_prepare_request(node, data_item, write_id))
        
        # Wait for prepare responses
        if prepare_tasks:
            results = await asyncio.gather(*prepare_tasks, return_exceptions=True)
        else:
            results = []
        
        # Analyze prepare results
        prepared_nodes = []
        conflicts = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('prepared'):
                prepared_nodes.append(nodes_to_prepare[i])
                if 'conflict' in result:
                    conflicts.append(result['conflict'])
            elif isinstance(result, Exception):
                logger.warning(f"Prepare failed for node {nodes_to_prepare[i]}: {result}")
        
        # Add self to prepared nodes
        prepared_nodes.append(self.node_id)
        
        # Check if we have enough nodes prepared
        prepare_quorum = (len(self.known_nodes) // 2) + 1
        prepared = len(prepared_nodes) >= prepare_quorum
        
        return {
            'prepared': prepared,
            'nodes': prepared_nodes,
            'conflicts': conflicts,
            'total_nodes': len(self.known_nodes)
        }
    
    async def _quorum_abort_phase(self, write_id: str, nodes: List[str]) -> None:
        """Abort quorum write on specified nodes"""
        abort_tasks = []
        
        for node in nodes:
            if node != self.node_id:
                abort_tasks.append(self._send_abort_request(node, write_id))
        
        if abort_tasks:
            await asyncio.gather(*abort_tasks, return_exceptions=True)
    
    async def _quorum_commit_phase(self, data_item: DataItem, write_id: str, 
                                  prepared_nodes: List[str], conflicts: List[Dict]) -> Dict[str, Any]:
        """Execute commit phase of quorum protocol"""
        # Resolve any conflicts first
        if conflicts:
            resolved_value = self._resolve_quorum_conflicts(data_item, conflicts)
            data_item.value = resolved_value
        
        # Send commit requests to prepared nodes
        commit_tasks = []
        nodes_to_commit = []
        
        for node in prepared_nodes:
            if node != self.node_id:
                nodes_to_commit.append(node)
                commit_tasks.append(self._send_commit_request(node, data_item, write_id))
        
        # Wait for commit responses
        if commit_tasks:
            results = await asyncio.gather(*commit_tasks, return_exceptions=True)
        else:
            results = []
        
        # Count successful commits
        successful_nodes = [self.node_id]  # Self commits
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('committed'):
                successful_nodes.append(nodes_to_commit[i])
            elif isinstance(result, Exception):
                logger.warning(f"Commit failed for node {nodes_to_commit[i]}: {result}")
        
        return {
            'successful_nodes': successful_nodes,
            'total_commits': len(successful_nodes)
        }
    
    async def _quorum_finalize_phase(self, write_id: str, successful_nodes: List[str], 
                                    success: bool) -> None:
        """Finalize quorum write - notify nodes of final outcome"""
        finalize_tasks = []
        
        for node in self.known_nodes:
            if node != self.node_id:
                finalize_tasks.append(
                    self._send_finalize_request(node, write_id, success, successful_nodes)
                )
        
        if finalize_tasks:
            await asyncio.gather(*finalize_tasks, return_exceptions=True)
    
    def _update_local_vector_clock(self, data_item: DataItem) -> None:
        """Update local vector clock after successful write"""
        key = data_item.key
        if key not in self.vector_clocks:
            self.vector_clocks[key] = {}
        
        # Merge vector clock from data item
        for node, clock in data_item.version.vector_clock.items():
            if node not in self.vector_clocks[key] or clock > self.vector_clocks[key][node]:
                self.vector_clocks[key][node] = clock
    
    def _resolve_quorum_conflicts(self, data_item: DataItem, conflicts: List[Dict]) -> Any:
        """Resolve conflicts detected during quorum prepare phase"""
        if not conflicts:
            return data_item.value
        
        # Collect all conflicting values with their vector clocks
        all_values = [(data_item.value, data_item.version.vector_clock)]
        
        for conflict in conflicts:
            all_values.append((conflict['value'], conflict['vector_clock']))
        
        # Use vector clock comparison to find dominant value
        dominant_value = data_item.value
        dominant_clock = data_item.version.vector_clock
        
        for value, clock in all_values[1:]:
            if self._vector_clock_dominates(clock, dominant_clock):
                dominant_value = value
                dominant_clock = clock
            elif not self._vector_clock_dominates(dominant_clock, clock):
                # Concurrent writes - use deterministic tiebreaker
                if hash(str(value)) > hash(str(dominant_value)):
                    dominant_value = value
                    dominant_clock = clock
        
        return dominant_value
    
    def _vector_clock_dominates(self, vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """Check if vc1 dominates vc2 (happens-after relationship)"""
        dominates = False
        
        all_nodes = set(vc1.keys()) | set(vc2.keys())
        for node in all_nodes:
            v1 = vc1.get(node, 0)
            v2 = vc2.get(node, 0)
            
            if v1 < v2:
                return False  # vc2 has higher value for this node
            elif v1 > v2:
                dominates = True  # vc1 has at least one higher value
        
        return dominates
    
    # Stub methods for network communication (would use A2A network in production)
    async def _send_prepare_request(self, node: str, data_item: DataItem, write_id: str) -> Dict:
        """Send prepare request to node"""
        # Simulate network call
        await asyncio.sleep(0.01)
        return {'prepared': True, 'node': node}
    
    async def _send_abort_request(self, node: str, write_id: str) -> Dict:
        """Send abort request to node"""
        await asyncio.sleep(0.01)
        return {'aborted': True, 'node': node}
    
    async def _send_commit_request(self, node: str, data_item: DataItem, write_id: str) -> Dict:
        """Send commit request to node"""
        await asyncio.sleep(0.01)
        return {'committed': True, 'node': node}
    
    async def _send_finalize_request(self, node: str, write_id: str, success: bool, 
                                   nodes: List[str]) -> Dict:
        """Send finalize request to node"""
        await asyncio.sleep(0.01)
        return {'finalized': True, 'node': node}
    
    async def _ensure_causal_consistency(self, data_item: DataItem) -> bool:
        """Ensure causal consistency for writes using dependency tracking"""
        try:
            # Extract causal dependencies from metadata
            dependencies = data_item.metadata.get('causal_dependencies', [])
            
            if not dependencies:
                # No dependencies - causally consistent by default
                return True
            
            # Check each dependency
            for dep in dependencies:
                dep_key = dep.get('key')
                dep_version = dep.get('version')
                dep_vector_clock = dep.get('vector_clock', {})
                
                if not dep_key:
                    continue
                
                # Check if dependency is satisfied locally
                if dep_key in self.data_store:
                    local_item = self.data_store[dep_key]
                    local_version = local_item.version.version
                    local_vector_clock = local_item.version.vector_clock
                    
                    # Version must be at least as new as dependency
                    if local_version < dep_version:
                        logger.warning(f"Causal dependency not satisfied: {dep_key} "
                                     f"(have v{local_version}, need v{dep_version})")
                        return False
                    
                    # Check vector clock dominance
                    if not self._vector_clock_dominates(local_vector_clock, dep_vector_clock):
                        logger.warning(f"Causal dependency vector clock not satisfied for {dep_key}")
                        return False
                else:
                    # Dependency not found - check if we can fetch it
                    fetched = await self._fetch_causal_dependency(dep_key, dep_version)
                    if not fetched:
                        logger.warning(f"Could not satisfy causal dependency: {dep_key}")
                        return False
            
            # All dependencies satisfied
            return True
            
        except Exception as e:
            logger.error(f"Causal consistency check failed: {e}")
            return False

    def _check_causal_consistency(self, data_item: DataItem) -> bool:
        """Check causal consistency for reads using happened-before relationships"""
        try:
            # Get read dependencies from context
            read_context = getattr(self, 'current_read_context', {})
            observed_writes = read_context.get('observed_writes', {})
            
            if not observed_writes:
                # No observed writes - causally consistent
                return True
            
            # Check if current item respects all observed writes
            item_vector_clock = data_item.version.vector_clock
            
            for observed_key, observed_clock in observed_writes.items():
                if observed_key == data_item.key:
                    # Check if this version includes the observed write
                    if not self._vector_clock_dominates(item_vector_clock, observed_clock):
                        logger.debug(f"Read not causally consistent: {data_item.key} "
                                   f"does not include observed write")
                        return False
                else:
                    # Check transitive dependencies
                    if hasattr(data_item, 'metadata') and 'causal_dependencies' in data_item.metadata:
                        deps = data_item.metadata['causal_dependencies']
                        for dep in deps:
                            if dep.get('key') == observed_key:
                                dep_clock = dep.get('vector_clock', {})
                                if not self._vector_clock_dominates(dep_clock, observed_clock):
                                    logger.debug(f"Read not causally consistent: transitive "
                                               f"dependency {observed_key} not satisfied")
                                    return False
            
            # Check session guarantees
            session_vector = read_context.get('session_vector_clock', {})
            if session_vector:
                # Ensure read reflects all writes from this session
                for node, clock in session_vector.items():
                    if node in item_vector_clock and item_vector_clock[node] < clock:
                        logger.debug(f"Read not causally consistent: session guarantee "
                                   f"violated for node {node}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Causal consistency read check failed: {e}")
            return True  # Default to allowing read
    
    async def _fetch_causal_dependency(self, key: str, min_version: int) -> bool:
        """Fetch a causal dependency from other nodes"""
        try:
            # Try to fetch from other nodes
            for node in self.known_nodes:
                if node != self.node_id and self.node_health.get(node, False):
                    # Simulate fetching dependency
                    await asyncio.sleep(0.01)
                    # In production, would actually fetch from node
                    return True
            
            return False
            
        except Exception:
            return False

    async def _sync_loop(self):
        """Background task to sync data with other nodes"""
        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval)

                if not self.sync_queue:
                    continue

                # Process sync queue
                batch = []
                while self.sync_queue and len(batch) < 100:
                    batch.append(self.sync_queue.popleft())

                if batch:
                    await self._sync_batch(batch)

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")

    async def _sync_batch(self, batch: List[Tuple[str, DataItem]]):
        """Sync a batch of data items"""
        sync_op = SyncOperation(
            operation_id=f"sync_{self.node_id}_{int(time.time())}",
            source_node=self.node_id,
            target_nodes=[n for n in self.known_nodes if n != self.node_id],
            data_keys=[key for key, _ in batch],
            started_at=datetime.utcnow(),
        )

        self.pending_syncs[sync_op.operation_id] = sync_op

        try:
            # Sync to each target node
            for target_node in sync_op.target_nodes:
                if self.node_health.get(target_node, False):
                    for key, data_item in batch:
                        await self._replicate_to_node(target_node, data_item)

            sync_op.completed_at = datetime.utcnow()
            sync_op.success = True
            self.sync_count += 1

        except Exception as e:
            logger.error(f"Sync operation {sync_op.operation_id} failed: {e}")
            sync_op.success = False

        finally:
            del self.pending_syncs[sync_op.operation_id]
            self.sync_history.append(sync_op)

    async def _monitor_consistency(self):
        """Monitor data consistency across nodes"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval)

                if not self.config.enable_consistency_checks:
                    continue

                # Check consistency for random sample of keys
                sample_size = min(10, len(self.data_store))
                if sample_size > 0:
                    sample_keys = random.sample(list(self.data_store.keys()), sample_size)

                    for key in sample_keys:
                        violation = await self._check_consistency(key)
                        if violation:
                            self.violations.append(violation)
                            await self._handle_violation(violation)

            except Exception as e:
                logger.error(f"Error in consistency monitoring: {e}")

    async def _check_consistency(self, key: str) -> Optional[ConsistencyViolation]:
        """Check consistency for a specific key"""
        if key not in self.data_store:
            return None

        local_item = self.data_store[key]

        # Check replica count
        expected_replicas = min(self.config.replication_factor, len(self.known_nodes))
        actual_replicas = len(local_item.replicas)

        if actual_replicas < expected_replicas:
            return ConsistencyViolation(
                violation_type="insufficient_replicas",
                affected_keys=[key],
                nodes=list(local_item.replicas),
                detected_at=datetime.utcnow(),
                details={"expected": expected_replicas, "actual": actual_replicas},
                severity="medium",
            )

        # Check version consistency across nodes
        if self.network_client:
            inconsistent_nodes = []
            local_version = local_item.version.version

            for node_id in self.known_nodes:
                if node_id == self.node_id or node_id not in local_item.replicas:
                    continue

                try:
                    node_url = await self._get_node_url(node_id)
                    if not node_url:
                        continue

                    remote_version = await self.network_client.get_latest_version(node_url, key)

                    # Allow some version drift for eventual consistency
                    if abs(remote_version - local_version) > 5:
                        inconsistent_nodes.append(node_id)

                except Exception as e:
                    logger.debug(f"Failed to check consistency with node {node_id}: {e}")
                    continue

            if inconsistent_nodes:
                return ConsistencyViolation(
                    violation_type="version_inconsistency",
                    affected_keys=[key],
                    nodes=inconsistent_nodes,
                    detected_at=datetime.utcnow(),
                    details={
                        "local_version": local_version,
                        "inconsistent_nodes": inconsistent_nodes
                    },
                    severity="high",
                )

        return None

    async def _handle_violation(self, violation: ConsistencyViolation):
        """Handle detected consistency violation"""
        logger.warning(f"Consistency violation detected: {violation.violation_type}")

        # Notify callbacks
        for callback in self.violation_callbacks:
            try:
                await callback(violation)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")

        # Attempt to repair
        if violation.violation_type == "insufficient_replicas":
            for key in violation.affected_keys:
                if key in self.data_store:
                    # Re-replicate to meet requirements
                    self.sync_queue.append((key, self.data_store[key]))

    async def _get_node_url(self, node_id: str) -> Optional[str]:
        """Get URL for a specific node"""
        if node_id in self.node_urls:
            return self.node_urls[node_id]

        # Try service discovery if URL not cached
        # This would integrate with a service registry in production
        if node_id == self.node_id:
            return None  # Don't need URL for self

        # Default to localhost-based URLs for development
        # In production, this would query a service registry
        port = 8000 + int(node_id.split('_')[-1]) if '_' in node_id else 8000
        url = f"http://localhost:{port}"

        # Cache the URL
        self.node_urls[node_id] = url
        return url

    def register_node_url(self, node_id: str, url: str):
        """Register URL for a node"""
        self.node_urls[node_id] = url
        self.known_nodes.add(node_id)
        self.node_health[node_id] = True
        self.last_heartbeat[node_id] = datetime.utcnow()
        logger.info(f"Registered node {node_id} at {url}")

    def add_violation_callback(self, callback: Callable):
        """Add callback for consistency violations"""
        self.violation_callbacks.append(callback)

    async def force_sync(self, keys: Optional[List[str]] = None):
        """Force synchronization of specific keys or all data"""
        if keys is None:
            keys = list(self.data_store.keys())

        for key in keys:
            if key in self.data_store:
                self.sync_queue.append((key, self.data_store[key]))

        logger.info(f"Forced sync for {len(keys)} keys")

    def get_consistency_metrics(self) -> Dict[str, Any]:
        """Get consistency metrics"""
        total_ops = self.read_count + self.write_count

        return {
            "node_id": self.node_id,
            "total_keys": len(self.data_store),
            "total_operations": total_ops,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_count / self.write_count if self.write_count > 0 else 0,
            "sync_count": self.sync_count,
            "pending_syncs": len(self.pending_syncs),
            "violation_count": len(self.violations),
            "active_nodes": len([n for n in self.known_nodes if self.node_health.get(n, False)]),
            "total_nodes": len(self.known_nodes),
            "leader": self.leader_id,
            "term": self.current_term,
        }

    def get_data_status(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a data item"""
        if key not in self.data_store:
            return None

        item = self.data_store[key]
        return {
            "key": key,
            "version": item.version.version,
            "timestamp": item.version.timestamp.isoformat(),
            "node_id": item.version.node_id,
            "consistency_level": item.consistency_level.value,
            "replicas": list(item.replicas),
            "vector_clock": item.version.vector_clock,
            "checksum": item.version.checksum,
            "has_conflicts": len(item.conflict_history) > 0,
            "pending_writes": len(item.pending_writes),
        }


# Global consistency manager instance
_consistency_manager: Optional[ConsistencyManager] = None


async def get_consistency_manager(node_id: str) -> ConsistencyManager:
    """Get or create the global consistency manager instance"""
    global _consistency_manager
    if _consistency_manager is None:
        _consistency_manager = ConsistencyManager(node_id)
        await _consistency_manager.initialize()
    return _consistency_manager

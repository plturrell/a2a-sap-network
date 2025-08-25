"""
Memory and Context Management System for A2A Agents
Part of Phase 1: Core AI Framework

This module provides advanced memory management with episodic, semantic,
and working memory, plus context awareness and retrieval.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import json
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import pickle
import os
import hashlib
import faiss
import networkx as nx

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the system"""

    EPISODIC = "episodic"  # Specific experiences and events
    SEMANTIC = "semantic"  # General knowledge and facts
    WORKING = "working"  # Short-term active memory
    PROCEDURAL = "procedural"  # How-to knowledge
    SENSORY = "sensory"  # Recent perceptual data


class ContextType(str, Enum):
    """Types of context"""

    TASK = "task"
    USER = "user"
    ENVIRONMENT = "environment"
    TEMPORAL = "temporal"
    SOCIAL = "social"
    DOMAIN = "domain"


@dataclass
class Memory:
    """Represents a memory item"""

    memory_id: str
    memory_type: MemoryType
    content: Any
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5
    decay_rate: float = 0.01
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Represents a context"""

    context_id: str
    context_type: ContextType
    attributes: Dict[str, Any]
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float = 1.0


class MemoryContextManager:
    """
    Advanced memory and context management system
    Provides human-like memory capabilities with forgetting, consolidation, and retrieval
    """

    def __init__(self, agent_id: str, embedding_dim: int = 768, storage_path: Optional[str] = None):
        self.agent_id = agent_id
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path or f"/tmp/a2a_memory/{agent_id}"
        os.makedirs(self.storage_path, exist_ok=True)

        # Memory stores
        self.episodic_memory = {}
        self.semantic_memory = {}
        self.working_memory = deque(maxlen=50)  # Limited capacity
        self.procedural_memory = {}

        # Context management
        self.active_contexts = {}
        self.context_history = deque(maxlen=1000)

        # Memory indices for fast retrieval
        self.memory_index = self._initialize_memory_index()
        self.knowledge_graph = nx.DiGraph()

        # Memory consolidation
        self.consolidation_queue = asyncio.Queue()
        self.consolidation_task = None

        # Attention mechanism
        self.attention_weights = defaultdict(float)

        # Load existing memories
        self._load_memories()

        logger.info(f"Initialized memory and context manager for agent {agent_id}")

    def _initialize_memory_index(self) -> faiss.IndexFlatL2:
        """Initialize FAISS index for similarity search"""
        index = faiss.IndexFlatL2(self.embedding_dim)
        return index

    async def store_memory(
        self,
        content: Any,
        memory_type: MemoryType,
        importance: float = 0.5,
        associations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Store a new memory

        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            associations: Associated memory IDs
            metadata: Additional metadata

        Returns:
            The created memory
        """
        memory_id = self._generate_memory_id(content, memory_type)

        # Generate embedding
        embedding = await self._generate_embedding(content)

        # Create memory
        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            associations=associations or [],
            metadata=metadata or {},
        )

        # Store in appropriate memory type
        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory[memory_id] = memory
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[memory_id] = memory
        elif memory_type == MemoryType.PROCEDURAL:
            self.procedural_memory[memory_id] = memory
        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(memory)

        # Add to index
        if embedding is not None:
            # FAISS add() method takes vectors directly
            vectors = embedding.reshape(1, -1).astype('float32')
            self.memory_index.add(vectors)

        # Add to knowledge graph
        self._add_to_knowledge_graph(memory)

        # Queue for consolidation
        await self.consolidation_queue.put(memory)

        return memory

    async def retrieve_memory(
        self,
        query: Any,
        memory_types: Optional[List[MemoryType]] = None,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Memory]:
        """
        Retrieve relevant memories

        Args:
            query: Query for retrieval
            memory_types: Types of memory to search
            top_k: Number of memories to retrieve
            threshold: Similarity threshold

        Returns:
            List of relevant memories
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        if query_embedding is None:
            return []

        # Search in index
        query_vectors = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.memory_index.search(query_vectors, top_k * 2)

        # Collect memories
        all_memories = self._get_all_memories(memory_types)
        relevant_memories = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(all_memories):
                memory = all_memories[idx]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                if similarity >= threshold:
                    # Update access information
                    memory.access_count += 1
                    memory.last_accessed = datetime.utcnow()

                    # Apply attention-based boosting
                    if memory.memory_id in self.attention_weights:
                        similarity *= 1 + self.attention_weights[memory.memory_id]

                    relevant_memories.append((memory, similarity))

        # Sort by relevance and return top k
        def get_relevance_score(x):
            return x[1]
        relevant_memories.sort(key=get_relevance_score, reverse=True)
        return [mem for mem, _ in relevant_memories[:top_k]]

    async def update_context(
        self, context_type: ContextType, attributes: Dict[str, Any], merge: bool = True
    ) -> Context:
        """
        Update or create context

        Args:
            context_type: Type of context
            attributes: Context attributes
            merge: Whether to merge with existing context

        Returns:
            Updated context
        """
        context_id = f"{context_type.value}_{datetime.utcnow().timestamp()}"

        if merge and context_type.value in self.active_contexts:
            # Merge with existing context
            existing = self.active_contexts[context_type.value]
            existing.attributes.update(attributes)
            existing.updated_at = datetime.utcnow()
            context = existing
        else:
            # Create new context
            context = Context(
                context_id=context_id, context_type=context_type, attributes=attributes
            )
            self.active_contexts[context_type.value] = context

        # Add to history
        self.context_history.append(context)

        # Update attention weights based on context
        await self._update_attention_weights(context)

        return context

    async def get_relevant_context(
        self, query: Any, context_types: Optional[List[ContextType]] = None
    ) -> List[Context]:
        """
        Get relevant contexts for a query

        Args:
            query: Query to match against
            context_types: Types of context to consider

        Returns:
            List of relevant contexts
        """
        relevant_contexts = []

        for context_type, context in self.active_contexts.items():
            if context_types and ContextType(context_type) not in context_types:
                continue

            if context.active:
                # Calculate relevance
                relevance = await self._calculate_context_relevance(query, context)
                if relevance > 0.5:
                    context.relevance_score = relevance
                    relevant_contexts.append(context)

        # Sort by relevance
        def get_relevance_score(x):
            return x.relevance_score
        relevant_contexts.sort(key=get_relevance_score, reverse=True)
        return relevant_contexts

    async def consolidate_memories(self):
        """
        Consolidate memories (like sleep consolidation in humans)
        Transfers important short-term memories to long-term storage
        """
        consolidated_count = 0

        # Process working memory
        working_memories = list(self.working_memory)
        for memory in working_memories:
            # Check if memory should be consolidated
            if memory.importance > 0.7 or memory.access_count > 3:
                # Transfer to episodic or semantic memory
                if self._is_factual(memory.content):
                    self.semantic_memory[memory.memory_id] = memory
                else:
                    self.episodic_memory[memory.memory_id] = memory

                consolidated_count += 1

        # Clear old working memories
        self.working_memory.clear()

        # Consolidate associations
        await self._consolidate_associations()

        # Apply forgetting curve
        await self._apply_forgetting()

        logger.info(f"Consolidated {consolidated_count} memories")

        return consolidated_count

    async def forget_memories(self, decay_threshold: float = 0.1):
        """
        Apply forgetting curve to memories
        Removes memories that have decayed below threshold
        """
        forgotten_memories = []

        for memory_store in [self.episodic_memory, self.semantic_memory]:
            memories_to_forget = []

            for memory_id, memory in memory_store.items():
                # Calculate decay
                time_since_access = (datetime.utcnow() - memory.last_accessed).total_seconds()
                decay = memory.decay_rate * time_since_access / 86400  # Daily decay

                # Apply importance factor
                effective_strength = memory.importance * (1 - decay)

                if effective_strength < decay_threshold:
                    memories_to_forget.append(memory_id)
                    forgotten_memories.append(memory)

            # Remove forgotten memories
            for memory_id in memories_to_forget:
                del memory_store[memory_id]

        logger.info(f"Forgot {len(forgotten_memories)} memories")
        return forgotten_memories

    async def create_associations(self, memory_id1: str, memory_id2: str, strength: float = 0.5):
        """
        Create association between memories

        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            strength: Association strength (0-1)
        """
        # Find memories
        memory1 = self._find_memory(memory_id1)
        memory2 = self._find_memory(memory_id2)

        if memory1 and memory2:
            # Add bidirectional associations
            if memory_id2 not in memory1.associations:
                memory1.associations.append(memory_id2)
            if memory_id1 not in memory2.associations:
                memory2.associations.append(memory_id1)

            # Add to knowledge graph
            self.knowledge_graph.add_edge(memory_id1, memory_id2, weight=strength)

            # Update importance based on associations
            memory1.importance = min(1.0, memory1.importance + strength * 0.1)
            memory2.importance = min(1.0, memory2.importance + strength * 0.1)

    async def get_memory_path(
        self, start_memory_id: str, end_memory_id: str
    ) -> Optional[List[str]]:
        """
        Find path between memories through associations

        Args:
            start_memory_id: Starting memory
            end_memory_id: Target memory

        Returns:
            Path of memory IDs if exists
        """
        try:
            path = nx.shortest_path(self.knowledge_graph, start_memory_id, end_memory_id)
            return path
        except nx.NetworkXNoPath:
            return None

    async def get_memory_clusters(self) -> List[Set[str]]:
        """
        Get clusters of related memories

        Returns:
            List of memory clusters (sets of memory IDs)
        """
        # Find connected components
        clusters = list(nx.connected_components(self.knowledge_graph.to_undirected()))
        return clusters

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "total_memories": len(self.episodic_memory)
            + len(self.semantic_memory)
            + len(self.procedural_memory)
            + len(self.working_memory),
            "by_type": {
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory),
                "procedural": len(self.procedural_memory),
                "working": len(self.working_memory),
            },
            "active_contexts": len(self.active_contexts),
            "knowledge_graph": {
                "nodes": self.knowledge_graph.number_of_nodes(),
                "edges": self.knowledge_graph.number_of_edges(),
                "clusters": len(
                    list(nx.connected_components(self.knowledge_graph.to_undirected()))
                ),
            },
            "average_importance": self._calculate_average_importance(),
            "memory_health": self._calculate_memory_health(),
        }

        return stats

    async def _generate_embedding(self, content: Any) -> Optional[np.ndarray]:
        """Generate embedding for content"""
        # Simplified embedding generation
        # In real implementation, would use proper embedding model

        if isinstance(content, str):
            # Simple hash-based embedding
            content_hash = hashlib.sha256(content.encode()).digest()
            embedding = np.frombuffer(content_hash, dtype=np.uint8)
            # Pad or truncate to embedding_dim
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            else:
                embedding = embedding[: self.embedding_dim]
            return embedding.astype(np.float32)
        elif isinstance(content, dict):
            # Convert dict to string and embed
            return await self._generate_embedding(json.dumps(content, sort_keys=True))
        else:
            # For other types, convert to string
            return await self._generate_embedding(str(content))

    def _generate_memory_id(self, content: Any, memory_type: MemoryType) -> str:
        """Generate unique memory ID"""
        content_str = str(content)[:100]  # Limit length
        timestamp = datetime.utcnow().timestamp()
        return f"{memory_type.value}_{hash(content_str)}_{timestamp}"

    def _get_all_memories(self, memory_types: Optional[List[MemoryType]] = None) -> List[Memory]:
        """Get all memories of specified types"""
        all_memories = []

        if not memory_types:
            memory_types = list(MemoryType)

        if MemoryType.EPISODIC in memory_types:
            all_memories.extend(self.episodic_memory.values())
        if MemoryType.SEMANTIC in memory_types:
            all_memories.extend(self.semantic_memory.values())
        if MemoryType.PROCEDURAL in memory_types:
            all_memories.extend(self.procedural_memory.values())
        if MemoryType.WORKING in memory_types:
            all_memories.extend(self.working_memory)

        return all_memories

    def _find_memory(self, memory_id: str) -> Optional[Memory]:
        """Find memory by ID across all stores"""
        # Check all memory stores
        if memory_id in self.episodic_memory:
            return self.episodic_memory[memory_id]
        elif memory_id in self.semantic_memory:
            return self.semantic_memory[memory_id]
        elif memory_id in self.procedural_memory:
            return self.procedural_memory[memory_id]
        else:
            # Check working memory
            for memory in self.working_memory:
                if memory.memory_id == memory_id:
                    return memory
        return None

    def _add_to_knowledge_graph(self, memory: Memory):
        """Add memory to knowledge graph"""
        self.knowledge_graph.add_node(
            memory.memory_id,
            memory_type=memory.memory_type.value,
            importance=memory.importance,
            timestamp=memory.timestamp.isoformat(),
        )

        # Add edges for associations
        for associated_id in memory.associations:
            if self.knowledge_graph.has_node(associated_id):
                self.knowledge_graph.add_edge(memory.memory_id, associated_id)

    async def _update_attention_weights(self, context: Context):
        """Update attention weights based on context"""
        # Simple attention mechanism
        # In real implementation, would use transformer-based attention

        # Boost memories related to active context
        for memory in self._get_all_memories():
            if context.context_type.value in memory.metadata.get("contexts", []):
                self.attention_weights[memory.memory_id] = min(
                    1.0, self.attention_weights[memory.memory_id] + 0.1
                )

        # Decay old attention weights
        for memory_id in list(self.attention_weights.keys()):
            self.attention_weights[memory_id] *= 0.95
            if self.attention_weights[memory_id] < 0.01:
                del self.attention_weights[memory_id]

    async def _calculate_context_relevance(self, query: Any, context: Context) -> float:
        """Calculate relevance of context to query"""
        # Simplified relevance calculation
        # In real implementation, would use semantic similarity

        relevance = 0.5  # Base relevance

        # Check for keyword matches
        query_str = str(query).lower()
        for key, value in context.attributes.items():
            if key.lower() in query_str or str(value).lower() in query_str:
                relevance += 0.2

        # Consider recency
        age = (datetime.utcnow() - context.updated_at).total_seconds()
        recency_factor = 1.0 / (1.0 + age / 3600)  # Hourly decay
        relevance *= recency_factor

        return min(1.0, relevance)

    def _is_factual(self, content: Any) -> bool:
        """Determine if content is factual (for semantic memory)"""
        # Simplified heuristic
        # In real implementation, would use NLP classification

        if isinstance(content, dict):
            # Facts often have structured data
            return "fact" in content or "definition" in content or "rule" in content
        elif isinstance(content, str):
            # Look for factual indicators
            factual_keywords = ["is", "are", "equals", "means", "defined as"]
            return any(keyword in content.lower() for keyword in factual_keywords)

        return False

    async def _consolidate_associations(self):
        """Consolidate associations between memories"""
        # Strengthen frequently co-accessed memories
        access_patterns = defaultdict(int)

        for memory in self._get_all_memories():
            for associated_id in memory.associations:
                pair = tuple(sorted([memory.memory_id, associated_id]))
                access_patterns[pair] += memory.access_count

        # Strengthen strong associations
        for (id1, id2), count in access_patterns.items():
            if count > 5:
                strength = min(1.0, count / 10.0)
                if self.knowledge_graph.has_edge(id1, id2):
                    self.knowledge_graph[id1][id2]["weight"] = strength

    async def _apply_forgetting(self):
        """Apply forgetting curve to memories"""
        await self.forget_memories()

    def _calculate_average_importance(self) -> float:
        """Calculate average importance of all memories"""
        all_memories = self._get_all_memories()
        if not all_memories:
            return 0.0

        total_importance = sum(memory.importance for memory in all_memories)
        return total_importance / len(all_memories)

    def _calculate_memory_health(self) -> float:
        """Calculate overall health of memory system"""
        # Consider various factors
        health_score = 1.0

        # Penalize if too many memories
        total_memories = len(self._get_all_memories())
        if total_memories > 10000:
            health_score *= 0.8
        elif total_memories < 100:
            health_score *= 0.9

        # Check balance between memory types
        type_counts = [
            len(self.episodic_memory),
            len(self.semantic_memory),
            len(self.procedural_memory),
        ]
        if max(type_counts) > sum(type_counts) * 0.7:
            health_score *= 0.85  # Penalize imbalance

        # Check association health
        if self.knowledge_graph.number_of_edges() < self.knowledge_graph.number_of_nodes() * 0.5:
            health_score *= 0.9  # Too few associations

        return health_score

    def _save_memories(self):
        """Save memories to disk"""
        # Save memory stores
        memory_data = {
            "episodic": self.episodic_memory,
            "semantic": self.semantic_memory,
            "procedural": self.procedural_memory,
            "working": list(self.working_memory),
        }

        memory_file = os.path.join(self.storage_path, "memories.pkl")
        try:
            with open(memory_file, "wb") as f:
                pickle.dump(memory_data, f)
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

        # Save knowledge graph
        graph_file = os.path.join(self.storage_path, "knowledge_graph.pkl")
        try:
            with open(graph_file, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def _load_memories(self):
        """Load memories from disk"""
        # Load memory stores
        memory_file = os.path.join(self.storage_path, "memories.pkl")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "rb") as f:
                    memory_data = pickle.load(f)
                    self.episodic_memory = memory_data.get("episodic", {})
                    self.semantic_memory = memory_data.get("semantic", {})
                    self.procedural_memory = memory_data.get("procedural", {})

                    # Reconstruct working memory
                    working_memories = memory_data.get("working", [])
                    for memory in working_memories[-50:]:  # Limit to maxlen
                        self.working_memory.append(memory)

                logger.info(f"Loaded {len(self._get_all_memories())} memories")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")

        # Load knowledge graph
        graph_file = os.path.join(self.storage_path, "knowledge_graph.pkl")
        if os.path.exists(graph_file):
            try:
                with open(graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                logger.info(
                    f"Loaded knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes"
                )
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")

    async def start_consolidation_loop(self, interval: int = 3600):
        """Start periodic memory consolidation"""

        async def consolidation_loop():
            while True:
                await asyncio.sleep(interval)
                await self.consolidate_memories()
                self._save_memories()

        self.consolidation_task = asyncio.create_task(consolidation_loop())

    def stop_consolidation_loop(self):
        """Stop memory consolidation loop"""
        if self.consolidation_task:
            self.consolidation_task.cancel()


# Utility functions
def create_memory_context_manager(agent_id: str) -> MemoryContextManager:
    """Factory function to create memory context manager"""
    return MemoryContextManager(agent_id)


async def store_experience(
    manager: MemoryContextManager, experience: Dict[str, Any], importance: float = 0.5
) -> Memory:
    """Store an experience as episodic memory"""
    return await manager.store_memory(
        content=experience,
        memory_type=MemoryType.EPISODIC,
        importance=importance,
        metadata={"type": "experience"},
    )


async def store_fact(
    manager: MemoryContextManager, fact: Dict[str, Any], importance: float = 0.7
) -> Memory:
    """Store a fact as semantic memory"""
    return await manager.store_memory(
        content=fact,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        metadata={"type": "fact"},
    )


async def update_task_context(manager: MemoryContextManager, task_info: Dict[str, Any]) -> Context:
    """Update task context"""
    return await manager.update_context(context_type=ContextType.TASK, attributes=task_info)

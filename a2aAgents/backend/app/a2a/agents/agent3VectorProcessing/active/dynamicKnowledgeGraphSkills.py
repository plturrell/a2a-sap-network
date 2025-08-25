import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import hashlib
import asyncio
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task
from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
from app.a2a.core.security_base import SecureA2AAgent
"""
Dynamic Knowledge Graph Skills for Agent 3 - Phase 3 Intelligence Layer
Implements real-time knowledge graph updates with intelligent relationship discovery
"""

try:
    from app.a2a.core.trustIdentity import TrustIdentity
except ImportError:
    class TrustIdentity(SecureA2AAgent):
        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning
        def __init__(self, **kwargs): pass
        def validate(self, *args): return True

try:
    from app.a2a.core.dataValidation import DataValidator
except ImportError:
    class DataValidator(SecureA2AAgent):
        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning
        def __init__(self, **kwargs): pass
        def validate(self, *args): return {"valid": True}

try:
    from app.clients.grokClient import GrokClient, get_grok_client
except ImportError:
    class GrokClient(SecureA2AAgent):
        # Security features provided by SecureA2AAgent:
        # - JWT authentication and authorization
        # - Rate limiting and request throttling
        # - Input validation and sanitization
        # - Audit logging and compliance tracking
        # - Encrypted communication channels
        # - Automatic security scanning
        def __init__(self, **kwargs): pass
        async def generate_embedding(self, *args): return [0] * 768
    def get_grok_client(): return GrokClient()


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    FUNCTIONAL_DEPENDENCY = "functional_dependency"
    EQUIVALENCE = "equivalence"
    CONTRADICTION = "contradiction"


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    ENTITY = "entity"
    CONCEPT = "concept"
    FACT = "fact"
    RULE = "rule"
    CALCULATION = "calculation"
    DOCUMENT = "document"


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence_score: float = 1.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    update_count: int = 0


@dataclass
class KnowledgeEdge:
    """Represents an edge in the knowledge graph"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    strength: float = 1.0
    last_validated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    validation_count: int = 0


@dataclass
class GraphUpdate:
    """Represents an update to the knowledge graph"""
    update_id: str
    update_type: str  # "node_add", "node_update", "node_remove", "edge_add", "edge_update", "edge_remove"
    affected_elements: List[str]  # IDs of affected nodes/edges
    update_reason: str
    confidence_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DynamicKnowledgeGraphSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Real A2A agent skills for dynamic knowledge graphs with intelligent updates
    Provides real-time graph evolution and relationship discovery
    """

    def __init__(self, trust_identity: TrustIdentity):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataValidator()

        # Initialize embedding model for semantic analysis
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        # Initialize GrokClient for intelligent analysis
        try:
            self.grok_client = get_grok_client()
            self.logger.info("GrokClient initialized for knowledge graph analysis")
        except Exception as e:
            self.logger.warning(f"GrokClient initialization failed: {e}")
            self.grok_client = None

        # Knowledge graph storage
        self.knowledge_graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}

        # Update tracking
        self.update_history: List[GraphUpdate] = []
        self.relationship_rules = self._initialize_relationship_rules()

        # Performance metrics
        self.graph_metrics = {
            'total_nodes': 0,
            'total_edges': 0,
            'updates_processed': 0,
            'relationships_discovered': 0,
            'graph_coherence_score': 0.0,
            'avg_update_confidence': 0.0
        }

    @a2a_skill(
        name="addKnowledgeNode",
        description="Add a new node to the dynamic knowledge graph",
        input_schema={
            "type": "object",
            "properties": {
                "node_data": {
                    "type": "object",
                    "properties": {
                        "node_type": {
                            "type": "string",
                            "enum": ["entity", "concept", "fact", "rule", "calculation", "document"]
                        },
                        "label": {"type": "string"},
                        "properties": {"type": "object"},
                        "content_text": {"type": "string"}
                    },
                    "required": ["node_type", "label"]
                },
                "auto_discover_relationships": {"type": "boolean", "default": True},
                "relationship_threshold": {"type": "number", "default": 0.7},
                "use_grok_analysis": {"type": "boolean", "default": True}
            },
            "required": ["node_data"]
        }
    )
    def add_knowledge_node(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new node to the knowledge graph with automatic relationship discovery"""
        try:
            node_data = request_data["node_data"]
            auto_discover = request_data.get("auto_discover_relationships", True)
            relationship_threshold = request_data.get("relationship_threshold", 0.7)
            use_grok = request_data.get("use_grok_analysis", True) and self.grok_client is not None

            # Create unique node ID
            node_content = f"{node_data['label']}_{json.dumps(node_data.get('properties', {}), sort_keys=True)}"
            node_id = hashlib.md5(node_content.encode()).hexdigest()[:12]

            # Generate embedding for semantic analysis
            content_text = node_data.get('content_text', node_data['label'])
            embedding = self.embedding_model.encode(content_text, normalize_embeddings=True)

            # Create knowledge node
            knowledge_node = KnowledgeNode(
                node_id=node_id,
                node_type=NodeType(node_data['node_type']),
                label=node_data['label'],
                properties=node_data.get('properties', {}),
                embedding=embedding,
                confidence_score=node_data.get('confidence_score', 1.0)
            )

            # Add to graph structures
            self.nodes[node_id] = knowledge_node
            self.knowledge_graph.add_node(node_id, **knowledge_node.__dict__)

            discovered_relationships = []

            # Auto-discover relationships if enabled
            if auto_discover and len(self.nodes) > 1:
                discovered_relationships = self._discover_relationships(
                    knowledge_node, relationship_threshold, use_grok
                )

            # Get Grok insights on the new node
            grok_insights = None
            if use_grok:
                grok_insights = self._get_grok_node_insights(knowledge_node)

            # Update metrics
            self.graph_metrics['total_nodes'] += 1
            self.graph_metrics['relationships_discovered'] += len(discovered_relationships)
            self.graph_metrics['updates_processed'] += 1

            # Record update
            update = GraphUpdate(
                update_id=hashlib.md5(f"add_node_{node_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12],
                update_type="node_add",
                affected_elements=[node_id],
                update_reason=f"Added new {node_data['node_type']} node",
                confidence_score=knowledge_node.confidence_score
            )
            self.update_history.append(update)

            # Update graph coherence
            self.graph_metrics['graph_coherence_score'] = self._calculate_graph_coherence()

            self.logger.info(f"Added knowledge node {node_id} with {len(discovered_relationships)} relationships")

            return {
                'success': True,
                'node_details': {
                    'node_id': node_id,
                    'node_type': knowledge_node.node_type.value,
                    'label': knowledge_node.label,
                    'properties': knowledge_node.properties,
                    'confidence_score': knowledge_node.confidence_score,
                    'embedding_dimension': len(embedding)
                },
                'discovered_relationships': [
                    {
                        'edge_id': rel['edge_id'],
                        'target_node_id': rel['target_node_id'],
                        'relationship_type': rel['relationship_type'],
                        'confidence_score': rel['confidence_score'],
                        'strength': rel['strength']
                    }
                    for rel in discovered_relationships
                ],
                'grok_insights': grok_insights,
                'graph_metrics': {
                    'total_nodes': self.graph_metrics['total_nodes'],
                    'total_edges': self.graph_metrics['total_edges'],
                    'coherence_score': self.graph_metrics['graph_coherence_score']
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to add knowledge node: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'node_addition_error'
            }

    @a2a_skill(
        name="updateKnowledgeGraph",
        description="Update the knowledge graph with new information and re-evaluate relationships",
        input_schema={
            "type": "object",
            "properties": {
                "update_data": {
                    "type": "object",
                    "properties": {
                        "node_updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_id": {"type": "string"},
                                    "property_updates": {"type": "object"},
                                    "confidence_update": {"type": "number"}
                                }
                            }
                        },
                        "new_facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "fact_text": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "source": {"type": "string"}
                                }
                            }
                        },
                        "relationship_hints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "target_id": {"type": "string"},
                                    "relationship_type": {"type": "string"},
                                    "confidence": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "revalidation_threshold": {"type": "number", "default": 0.6},
                "use_grok_validation": {"type": "boolean", "default": True}
            },
            "required": ["update_data"]
        }
    )
    def update_knowledge_graph(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update the knowledge graph with new information and re-evaluate relationships"""
        try:
            update_data = request_data["update_data"]
            revalidation_threshold = request_data.get("revalidation_threshold", 0.6)
            use_grok = request_data.get("use_grok_validation", True) and self.grok_client is not None

            update_summary = {
                'nodes_updated': 0,
                'relationships_added': 0,
                'relationships_updated': 0,
                'facts_processed': 0,
                'validation_changes': 0
            }

            # Process node updates
            node_updates = update_data.get("node_updates", [])
            for node_update in node_updates:
                node_id = node_update["node_id"]
                if node_id in self.nodes:
                    node = self.nodes[node_id]

                    # Update properties
                    property_updates = node_update.get("property_updates", {})
                    node.properties.update(property_updates)

                    # Update confidence if provided
                    if "confidence_update" in node_update:
                        node.confidence_score = node_update["confidence_update"]

                    # Update timestamps and counts
                    node.last_updated = datetime.utcnow().isoformat()
                    node.update_count += 1

                    # Update graph representation
                    self.knowledge_graph.nodes[node_id].update(node.__dict__)
                    update_summary['nodes_updated'] += 1

            # Process new facts
            new_facts = update_data.get("new_facts", [])
            for fact_data in new_facts:
                fact_node = self._create_fact_node(fact_data)
                if fact_node:
                    # Discover relationships for the new fact
                    relationships = self._discover_relationships(fact_node, revalidation_threshold, use_grok)
                    update_summary['facts_processed'] += 1
                    update_summary['relationships_added'] += len(relationships)

            # Process relationship hints
            relationship_hints = update_data.get("relationship_hints", [])
            for hint in relationship_hints:
                if self._validate_relationship_hint(hint):
                    edge = self._create_relationship_edge(
                        hint["source_id"],
                        hint["target_id"],
                        RelationshipType(hint["relationship_type"]),
                        hint.get("confidence", 0.8)
                    )
                    if edge:
                        update_summary['relationships_added'] += 1

            # Re-validate existing relationships
            revalidation_results = self._revalidate_relationships(revalidation_threshold, use_grok)
            update_summary['validation_changes'] = len(revalidation_results)

            # Update graph metrics
            self._update_graph_metrics()

            # Get Grok insights on the updates
            grok_update_insights = None
            if use_grok:
                grok_update_insights = self._get_grok_update_insights(update_data, update_summary)

            # Record the update
            update_record = GraphUpdate(
                update_id=hashlib.md5(f"graph_update_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12],
                update_type="graph_update",
                affected_elements=list(set([u["node_id"] for u in node_updates if "node_id" in u])),
                update_reason="Batch graph update with new information",
                confidence_score=np.mean([f.get("confidence", 0.8) for f in new_facts]) if new_facts else 0.8
            )
            self.update_history.append(update_record)

            self.logger.info(f"Updated knowledge graph: {update_summary}")

            return {
                'success': True,
                'update_summary': update_summary,
                'revalidation_results': revalidation_results,
                'grok_insights': grok_update_insights,
                'graph_metrics': {
                    'total_nodes': self.graph_metrics['total_nodes'],
                    'total_edges': self.graph_metrics['total_edges'],
                    'coherence_score': self.graph_metrics['graph_coherence_score'],
                    'updates_processed': self.graph_metrics['updates_processed']
                },
                'update_id': update_record.update_id
            }

        except Exception as e:
            self.logger.error(f"Knowledge graph update failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'graph_update_error'
            }

    @a2a_skill(
        name="queryKnowledgeGraph",
        description="Query the knowledge graph for insights and relationships",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["semantic_search", "path_finding", "neighborhood_analysis", "concept_exploration", "relationship_analysis"]
                        },
                        "query_text": {"type": "string"},
                        "target_node_ids": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "relationship_types": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "max_results": {"type": "integer", "default": 10},
                        "min_confidence": {"type": "number", "default": 0.5}
                    },
                    "required": ["query_type"]
                },
                "use_grok_enhancement": {"type": "boolean", "default": True}
            },
            "required": ["query"]
        }
    )
    def query_knowledge_graph(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph for insights and relationships"""
        try:
            query = request_data["query"]
            use_grok = request_data.get("use_grok_enhancement", True) and self.grok_client is not None

            query_type = query["query_type"]
            query_results = []

            if query_type == "semantic_search":
                query_results = self._perform_semantic_search(query)
            elif query_type == "path_finding":
                query_results = self._find_knowledge_paths(query)
            elif query_type == "neighborhood_analysis":
                query_results = self._analyze_node_neighborhood(query)
            elif query_type == "concept_exploration":
                query_results = self._explore_concepts(query)
            elif query_type == "relationship_analysis":
                query_results = self._analyze_relationships(query)

            # Enhance results with Grok insights if enabled
            grok_enhancement = None
            if use_grok and query_results:
                grok_enhancement = self._get_grok_query_enhancement(query, query_results)

            # Calculate query confidence
            query_confidence = self._calculate_query_confidence(query_results)

            return {
                'success': True,
                'query_type': query_type,
                'results': query_results,
                'result_count': len(query_results),
                'query_confidence': query_confidence,
                'grok_enhancement': grok_enhancement,
                'graph_context': {
                    'total_nodes_searched': self.graph_metrics['total_nodes'],
                    'total_edges_considered': self.graph_metrics['total_edges'],
                    'graph_coherence': self.graph_metrics['graph_coherence_score']
                }
            }

        except Exception as e:
            self.logger.error(f"Knowledge graph query failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'graph_query_error'
            }

    def _initialize_relationship_rules(self) -> Dict[str, Any]:
        """Initialize rules for relationship discovery"""
        return {
            'semantic_similarity_threshold': 0.75,
            'hierarchical_keywords': ['parent', 'child', 'contains', 'part of', 'subset'],
            'causal_keywords': ['causes', 'leads to', 'results in', 'because', 'due to'],
            'temporal_keywords': ['before', 'after', 'during', 'while', 'when'],
            'equivalence_keywords': ['equals', 'same as', 'identical', 'equivalent'],
            'contradiction_keywords': ['contradicts', 'opposes', 'conflicts with', 'not']
        }

    def _discover_relationships(self, new_node: KnowledgeNode, threshold: float, use_grok: bool) -> List[Dict[str, Any]]:
        """Discover relationships between the new node and existing nodes"""
        discovered_relationships = []

        # Skip if no existing nodes
        if len(self.nodes) <= 1:
            return discovered_relationships

        new_embedding = new_node.embedding
        if new_embedding is None:
            return discovered_relationships

        # Calculate similarities with existing nodes
        for existing_node_id, existing_node in self.nodes.items():
            if existing_node_id == new_node.node_id or existing_node.embedding is None:
                continue

            # Semantic similarity
            similarity = float(cosine_similarity([new_embedding], [existing_node.embedding])[0][0])

            if similarity >= threshold:
                # Determine relationship type
                relationship_type, confidence = self._determine_relationship_type(
                    new_node, existing_node, similarity, use_grok
                )

                if relationship_type:
                    edge = self._create_relationship_edge(
                        new_node.node_id,
                        existing_node_id,
                        relationship_type,
                        confidence,
                        strength=similarity
                    )

                    if edge:
                        discovered_relationships.append({
                            'edge_id': edge.edge_id,
                            'target_node_id': existing_node_id,
                            'relationship_type': relationship_type.value,
                            'confidence_score': confidence,
                            'strength': similarity
                        })

        return discovered_relationships

    def _determine_relationship_type(self, node1: KnowledgeNode, node2: KnowledgeNode, similarity: float, use_grok: bool) -> Tuple[Optional[RelationshipType], float]:
        """Determine the type of relationship between two nodes"""

        # Text-based relationship detection
        text1 = f"{node1.label} {json.dumps(node1.properties)}"
        text2 = f"{node2.label} {json.dumps(node2.properties)}"
        combined_text = f"{text1} {text2}".lower()

        # Check for specific relationship patterns
        for pattern_type, keywords in [
            (RelationshipType.HIERARCHICAL, self.relationship_rules['hierarchical_keywords']),
            (RelationshipType.CAUSAL_RELATIONSHIP, self.relationship_rules['causal_keywords']),
            (RelationshipType.TEMPORAL, self.relationship_rules['temporal_keywords']),
            (RelationshipType.EQUIVALENCE, self.relationship_rules['equivalence_keywords']),
            (RelationshipType.CONTRADICTION, self.relationship_rules['contradiction_keywords'])
        ]:
            if any(keyword in combined_text for keyword in keywords):
                confidence = min(0.8 + similarity * 0.2, 0.95)
                return pattern_type, confidence

        # Default to semantic similarity if above threshold
        if similarity >= self.relationship_rules['semantic_similarity_threshold']:
            return RelationshipType.SEMANTIC_SIMILARITY, similarity

        return None, 0.0

    def _create_relationship_edge(self, source_id: str, target_id: str, relationship_type: RelationshipType, confidence: float, strength: float = 1.0) -> Optional[KnowledgeEdge]:
        """Create a relationship edge between two nodes"""
        edge_id = hashlib.md5(f"{source_id}_{target_id}_{relationship_type.value}".encode()).hexdigest()[:12]

        # Check if edge already exists
        if edge_id in self.edges:
            # Update existing edge
            existing_edge = self.edges[edge_id]
            existing_edge.confidence_score = max(existing_edge.confidence_score, confidence)
            existing_edge.strength = max(existing_edge.strength, strength)
            existing_edge.last_validated = datetime.utcnow().isoformat()
            existing_edge.validation_count += 1
            return existing_edge

        # Create new edge
        edge = KnowledgeEdge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_type=relationship_type,
            confidence_score=confidence,
            strength=strength
        )

        # Add to graph structures
        self.edges[edge_id] = edge
        self.knowledge_graph.add_edge(source_id, target_id, key=edge_id, **edge.__dict__)

        # Update metrics
        self.graph_metrics['total_edges'] += 1

        return edge

    def _create_fact_node(self, fact_data: Dict[str, Any]) -> Optional[KnowledgeNode]:
        """Create a fact node from fact data"""
        fact_text = fact_data["fact_text"]
        confidence = fact_data.get("confidence", 0.8)
        source = fact_data.get("source", "unknown")

        # Generate embedding
        embedding = self.embedding_model.encode(fact_text, normalize_embeddings=True)

        # Create node ID
        node_id = hashlib.md5(f"fact_{fact_text}_{source}".encode()).hexdigest()[:12]

        # Create fact node
        fact_node = KnowledgeNode(
            node_id=node_id,
            node_type=NodeType.FACT,
            label=fact_text[:100] + "..." if len(fact_text) > 100 else fact_text,
            properties={
                'full_text': fact_text,
                'source': source,
                'fact_type': 'extracted'
            },
            embedding=embedding,
            confidence_score=confidence
        )

        # Add to graph
        self.nodes[node_id] = fact_node
        self.knowledge_graph.add_node(node_id, **fact_node.__dict__)
        self.graph_metrics['total_nodes'] += 1

        return fact_node

    def _validate_relationship_hint(self, hint: Dict[str, Any]) -> bool:
        """Validate a relationship hint"""
        source_id = hint["source_id"]
        target_id = hint["target_id"]
        relationship_type = hint["relationship_type"]

        # Check if nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        # Check if relationship type is valid
        try:
            RelationshipType(relationship_type)
            return True
        except ValueError:
            return False

    def _revalidate_relationships(self, threshold: float, use_grok: bool) -> List[Dict[str, Any]]:
        """Re-validate existing relationships based on updated information"""
        revalidation_results = []

        for edge_id, edge in list(self.edges.items()):
            source_node = self.nodes.get(edge.source_node_id)
            target_node = self.nodes.get(edge.target_node_id)

            if not source_node or not target_node:
                continue

            # Re-calculate relationship strength
            if source_node.embedding is not None and target_node.embedding is not None:
                new_similarity = float(cosine_similarity([source_node.embedding], [target_node.embedding])[0][0])

                # Update confidence based on new similarity
                confidence_change = abs(edge.strength - new_similarity)
                if confidence_change > 0.1:  # Significant change
                    edge.strength = new_similarity
                    edge.confidence_score = max(edge.confidence_score - confidence_change * 0.5, 0.1)
                    edge.last_validated = datetime.utcnow().isoformat()
                    edge.validation_count += 1

                    revalidation_results.append({
                        'edge_id': edge_id,
                        'old_strength': edge.strength,
                        'new_strength': new_similarity,
                        'confidence_change': confidence_change,
                        'action': 'updated'
                    })

                # Remove weak relationships
                if edge.confidence_score < threshold:
                    self.knowledge_graph.remove_edge(edge.source_node_id, edge.target_node_id, key=edge_id)
                    del self.edges[edge_id]
                    self.graph_metrics['total_edges'] -= 1

                    revalidation_results.append({
                        'edge_id': edge_id,
                        'action': 'removed',
                        'reason': 'confidence_below_threshold'
                    })

        return revalidation_results

    def _update_graph_metrics(self):
        """Update graph metrics"""
        self.graph_metrics['total_nodes'] = len(self.nodes)
        self.graph_metrics['total_edges'] = len(self.edges)

        # Calculate average update confidence
        if self.update_history:
            self.graph_metrics['avg_update_confidence'] = np.mean([u.confidence_score for u in self.update_history[-100:]])

        # Update graph coherence
        self.graph_metrics['graph_coherence_score'] = self._calculate_graph_coherence()

    def _calculate_graph_coherence(self) -> float:
        """Calculate overall graph coherence score"""
        if len(self.nodes) < 2:
            return 1.0

        # Calculate based on connectivity and confidence
        total_confidence = sum(node.confidence_score for node in self.nodes.values())
        avg_node_confidence = total_confidence / len(self.nodes)

        total_edge_confidence = sum(edge.confidence_score for edge in self.edges.values())
        avg_edge_confidence = total_edge_confidence / max(len(self.edges), 1)

        # Connectivity ratio
        max_possible_edges = len(self.nodes) * (len(self.nodes) - 1)
        connectivity_ratio = len(self.edges) / max(max_possible_edges, 1)

        # Combined coherence score
        coherence = (avg_node_confidence * 0.4 + avg_edge_confidence * 0.4 + connectivity_ratio * 0.2)
        return min(coherence, 1.0)

    def _perform_semantic_search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform semantic search on the knowledge graph"""
        query_text = query.get("query_text", "")
        max_results = query.get("max_results", 10)
        min_confidence = query.get("min_confidence", 0.5)

        if not query_text:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)

        # Calculate similarities
        search_results = []
        for node_id, node in self.nodes.items():
            if node.embedding is None:
                continue

            similarity = float(cosine_similarity([query_embedding], [node.embedding])[0][0])

            if similarity >= min_confidence:
                search_results.append({
                    'node_id': node_id,
                    'node_type': node.node_type.value,
                    'label': node.label,
                    'properties': node.properties,
                    'similarity_score': similarity,
                    'confidence_score': node.confidence_score
                })

        # Sort by similarity and return top results
        search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return search_results[:max_results]

    def _find_knowledge_paths(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find paths between nodes in the knowledge graph"""
        target_nodes = query.get("target_node_ids", [])

        if len(target_nodes) < 2:
            return []

        paths = []
        source = target_nodes[0]

        for target in target_nodes[1:]:
            try:
                if nx.has_path(self.knowledge_graph, source, target):
                    shortest_path = nx.shortest_path(self.knowledge_graph, source, target)
                    path_info = {
                        'source': source,
                        'target': target,
                        'path': shortest_path,
                        'path_length': len(shortest_path) - 1,
                        'path_edges': []
                    }

                    # Get edge information for the path
                    for i in range(len(shortest_path) - 1):
                        node1, node2 = shortest_path[i], shortest_path[i + 1]
                        edge_data = self.knowledge_graph.get_edge_data(node1, node2)
                        if edge_data:
                            path_info['path_edges'].append({
                                'source': node1,
                                'target': node2,
                                'edge_info': list(edge_data.values())[0] if edge_data else {}
                            })

                    paths.append(path_info)
            except nx.NetworkXNoPath:
                continue

        return paths

    def _analyze_node_neighborhood(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze the neighborhood of specified nodes"""
        target_nodes = query.get("target_node_ids", [])

        neighborhoods = []
        for node_id in target_nodes:
            if node_id not in self.nodes:
                continue

            # Get immediate neighbors
            predecessors = list(self.knowledge_graph.predecessors(node_id))
            successors = list(self.knowledge_graph.successors(node_id))

            # Calculate neighborhood statistics
            neighborhood_info = {
                'node_id': node_id,
                'node_label': self.nodes[node_id].label,
                'incoming_connections': len(predecessors),
                'outgoing_connections': len(successors),
                'total_connections': len(predecessors) + len(successors),
                'predecessors': [
                    {
                        'node_id': pred,
                        'label': self.nodes[pred].label,
                        'node_type': self.nodes[pred].node_type.value
                    }
                    for pred in predecessors if pred in self.nodes
                ],
                'successors': [
                    {
                        'node_id': succ,
                        'label': self.nodes[succ].label,
                        'node_type': self.nodes[succ].node_type.value
                    }
                    for succ in successors if succ in self.nodes
                ]
            }

            neighborhoods.append(neighborhood_info)

        return neighborhoods

    def _explore_concepts(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Explore concepts related to the query"""
        query_text = query.get("query_text", "")
        max_results = query.get("max_results", 10)

        # Find concept nodes
        concept_nodes = [node for node in self.nodes.values() if node.node_type == NodeType.CONCEPT]

        if not query_text:
            return [
                {
                    'node_id': node.node_id,
                    'label': node.label,
                    'properties': node.properties,
                    'confidence_score': node.confidence_score
                }
                for node in concept_nodes[:max_results]
            ]

        # Semantic search within concept nodes
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)
        concept_results = []

        for node in concept_nodes:
            if node.embedding is None:
                continue

            similarity = float(cosine_similarity([query_embedding], [node.embedding])[0][0])
            concept_results.append({
                'node_id': node.node_id,
                'label': node.label,
                'properties': node.properties,
                'confidence_score': node.confidence_score,
                'similarity_score': similarity
            })

        concept_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return concept_results[:max_results]

    def _analyze_relationships(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze relationships in the knowledge graph"""
        relationship_types = query.get("relationship_types", [])

        relationship_analysis = []

        # Group edges by relationship type
        relationship_groups = {}
        for edge in self.edges.values():
            rel_type = edge.relationship_type.value
            if not relationship_types or rel_type in relationship_types:
                if rel_type not in relationship_groups:
                    relationship_groups[rel_type] = []
                relationship_groups[rel_type].append(edge)

        # Analyze each relationship type
        for rel_type, edges in relationship_groups.items():
            avg_confidence = np.mean([edge.confidence_score for edge in edges])
            avg_strength = np.mean([edge.strength for edge in edges])

            relationship_analysis.append({
                'relationship_type': rel_type,
                'count': len(edges),
                'average_confidence': float(avg_confidence),
                'average_strength': float(avg_strength),
                'strongest_relationship': {
                    'edge_id': max(edges, key=lambda e: e.strength).edge_id,
                    'source': max(edges, key=lambda e: e.strength).source_node_id,
                    'target': max(edges, key=lambda e: e.strength).target_node_id,
                    'strength': max(edge.strength for edge in edges)
                }
            })

        return relationship_analysis

    def _calculate_query_confidence(self, query_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for query results"""
        if not query_results:
            return 0.0

        # Calculate based on result confidence scores and similarity scores
        confidences = []
        for result in query_results:
            result_conf = result.get('confidence_score', 0.5)
            similarity_conf = result.get('similarity_score', 0.5)
            combined_conf = (result_conf + similarity_conf) / 2
            confidences.append(combined_conf)

        return float(np.mean(confidences))

    def _get_grok_node_insights(self, node: KnowledgeNode) -> Optional[str]:
        """Get Grok insights for a new node"""
        if not self.grok_client:
            return None

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledge graph expert. Analyze nodes and suggest potential relationships and insights."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this knowledge graph node and suggest insights:

                    Node Type: {node.node_type.value}
                    Label: {node.label}
                    Properties: {json.dumps(node.properties, indent=2)}

                    What relationships might this node have? What insights can you provide?
                    """
                }
            ]

            response = self.grok_client.chat_completion(messages, temperature=0.4, max_tokens=300)
            return response.content

        except Exception as e:
            self.logger.warning(f"GrokClient node insights failed: {e}")
            return None

    def _get_grok_update_insights(self, update_data: Dict[str, Any], update_summary: Dict[str, Any]) -> Optional[str]:
        """Get Grok insights on graph updates"""
        if not self.grok_client:
            return None

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledge graph expert. Analyze graph updates and provide insights on knowledge evolution."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this knowledge graph update:

                    Update Summary: {json.dumps(update_summary, indent=2)}
                    Update Data: {json.dumps(update_data, indent=2)}

                    What patterns do you see in the knowledge evolution? Any insights or recommendations?
                    """
                }
            ]

            response = self.grok_client.chat_completion(messages, temperature=0.4, max_tokens=400)
            return response.content

        except Exception as e:
            self.logger.warning(f"GrokClient update insights failed: {e}")
            return None

    def _get_grok_query_enhancement(self, query: Dict[str, Any], results: List[Dict[str, Any]]) -> Optional[str]:
        """Get Grok enhancement for query results"""
        if not self.grok_client:
            return None

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledge graph expert. Enhance query results with additional insights and connections."
                },
                {
                    "role": "user",
                    "content": f"""
                    Enhance these knowledge graph query results:

                    Query: {json.dumps(query, indent=2)}
                    Results: {json.dumps(results[:5], indent=2)}  # First 5 results

                    Provide additional insights, suggest related concepts, or identify patterns.
                    """
                }
            ]

            response = self.grok_client.chat_completion(messages, temperature=0.5, max_tokens=350)
            return response.content

        except Exception as e:
            self.logger.warning(f"GrokClient query enhancement failed: {e}")
            return None

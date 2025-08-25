"""
Knowledge Representation System
Advanced knowledge graphs, ontologies, and semantic reasoning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import uuid
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    FACT = "fact"
    RULE = "rule"
    CONCEPT = "concept"
    RELATION = "relation"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"

class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    IMPLIES = "implies"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    EQUIVALENT = "equivalent"

@dataclass
class KnowledgeNode:
    node_id: str
    content: str
    knowledge_type: KnowledgeType
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    source: Optional[str] = None

@dataclass
class KnowledgeRelation:
    relation_id: str
    source_node: str
    target_node: str
    relation_type: RelationType
    strength: float
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ReasoningPath:
    path_id: str
    nodes: List[str]
    relations: List[str]
    confidence: float
    path_length: int
    reasoning_type: str
    explanation: str

class KnowledgeGraph:
    """Advanced knowledge graph with semantic reasoning capabilities"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes = {}  # node_id -> KnowledgeNode
        self.relations = {}  # relation_id -> KnowledgeRelation
        self.concept_hierarchy = nx.DiGraph()
        self.inference_cache = {}

        # Indexing for fast retrieval
        self.type_index = defaultdict(set)
        self.content_index = {}
        self.relation_index = defaultdict(list)

    def add_knowledge_node(self, node: KnowledgeNode) -> str:
        """Add knowledge node to graph"""

        self.nodes[node.node_id] = node

        # Add to NetworkX graph
        self.graph.add_node(
            node.node_id,
            content=node.content,
            knowledge_type=node.knowledge_type.value,
            confidence=node.confidence,
            properties=node.properties
        )

        # Update indexes
        self.type_index[node.knowledge_type].add(node.node_id)
        self.content_index[node.content.lower()] = node.node_id

        logger.debug(f"Added knowledge node: {node.content}")
        return node.node_id

    def add_knowledge_relation(self, relation: KnowledgeRelation) -> str:
        """Add knowledge relation to graph"""

        self.relations[relation.relation_id] = relation

        # Add to NetworkX graph
        self.graph.add_edge(
            relation.source_node,
            relation.target_node,
            relation_id=relation.relation_id,
            relation_type=relation.relation_type.value,
            strength=relation.strength,
            confidence=relation.confidence,
            properties=relation.properties
        )

        # Update indexes
        self.relation_index[relation.relation_type].append(relation.relation_id)

        logger.debug(f"Added relation: {relation.source_node} -> {relation.target_node} ({relation.relation_type.value})")
        return relation.relation_id

    def find_nodes_by_content(self, query: str, similarity_threshold: float = 0.7) -> List[KnowledgeNode]:
        """Find nodes by content similarity"""

        matching_nodes = []
        query_lower = query.lower()

        for node in self.nodes.values():
            content_lower = node.content.lower()

            # Exact match
            if query_lower == content_lower:
                matching_nodes.append(node)
                continue

            # Substring match
            if query_lower in content_lower or content_lower in query_lower:
                matching_nodes.append(node)
                continue

            # Word overlap similarity
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())

            if query_words and content_words:
                overlap = len(query_words.intersection(content_words))
                total = len(query_words.union(content_words))
                similarity = overlap / total

                if similarity >= similarity_threshold:
                    matching_nodes.append(node)

        return sorted(matching_nodes, key=lambda n: n.confidence, reverse=True)

    def find_nodes_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeNode]:
        """Find nodes by knowledge type"""

        node_ids = self.type_index[knowledge_type]
        return [self.nodes[node_id] for node_id in node_ids]

    def get_related_nodes(self, node_id: str, relation_types: List[RelationType] = None,
                         max_depth: int = 2) -> Dict[str, List[KnowledgeNode]]:
        """Get nodes related to given node"""

        if node_id not in self.nodes:
            return {}

        related_nodes = defaultdict(list)
        visited = set()

        def explore_relations(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get outgoing edges
            for successor in self.graph.successors(current_id):
                edge_data = self.graph[current_id][successor]

                for edge_key, edge_attrs in edge_data.items():
                    relation_type_str = edge_attrs.get('relation_type')

                    if relation_type_str:
                        try:
                            relation_type = RelationType(relation_type_str)

                            if not relation_types or relation_type in relation_types:
                                if successor in self.nodes:
                                    related_nodes[relation_type_str].append(self.nodes[successor])

                                if depth < max_depth:
                                    explore_relations(successor, depth + 1)
                        except ValueError:
                            continue

        explore_relations(node_id, 0)
        return dict(related_nodes)

    def find_reasoning_paths(self, source_node: str, target_node: str,
                           max_length: int = 5) -> List[ReasoningPath]:
        """Find reasoning paths between nodes"""

        if source_node not in self.nodes or target_node not in self.nodes:
            return []

        reasoning_paths = []

        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, source_node, target_node, cutoff=max_length
            ))

            for path in paths:
                path_relations = []
                path_confidence = 1.0
                reasoning_steps = []

                # Analyze path relations
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]

                    # Get edge data
                    edge_data = self.graph[current_node][next_node]

                    for edge_key, edge_attrs in edge_data.items():
                        relation_id = edge_attrs.get('relation_id')
                        if relation_id in self.relations:
                            relation = self.relations[relation_id]
                            path_relations.append(relation_id)
                            path_confidence *= relation.confidence

                            reasoning_steps.append(
                                f"{self.nodes[current_node].content} {relation.relation_type.value} {self.nodes[next_node].content}"
                            )
                            break

                if path_relations:
                    reasoning_path = ReasoningPath(
                        path_id=str(uuid.uuid4())[:8],
                        nodes=path,
                        relations=path_relations,
                        confidence=path_confidence,
                        path_length=len(path),
                        reasoning_type="deductive_path",
                        explanation=" → ".join(reasoning_steps)
                    )
                    reasoning_paths.append(reasoning_path)

        except nx.NetworkXNoPath:
            pass

        return sorted(reasoning_paths, key=lambda p: p.confidence, reverse=True)

    def infer_new_relations(self, confidence_threshold: float = 0.6) -> List[KnowledgeRelation]:
        """Infer new relations using graph patterns"""

        new_relations = []

        # Transitivity inference (A->B, B->C => A->C)
        transitive_relations = [RelationType.IS_A, RelationType.PART_OF, RelationType.IMPLIES]

        for relation_type in transitive_relations:
            relation_ids = self.relation_index[relation_type]

            for rel_id in relation_ids:
                relation = self.relations[rel_id]

                # Find relations that start where this one ends
                for other_rel_id in relation_ids:
                    if other_rel_id == rel_id:
                        continue

                    other_relation = self.relations[other_rel_id]

                    if relation.target_node == other_relation.source_node:
                        # Potential transitive relation
                        inferred_confidence = relation.confidence * other_relation.confidence * 0.8

                        if inferred_confidence >= confidence_threshold:
                            # Check if relation already exists
                            if not self._relation_exists(relation.source_node, other_relation.target_node, relation_type):
                                new_relation = KnowledgeRelation(
                                    relation_id=str(uuid.uuid4())[:8],
                                    source_node=relation.source_node,
                                    target_node=other_relation.target_node,
                                    relation_type=relation_type,
                                    strength=0.7,
                                    confidence=inferred_confidence,
                                    properties={"inferred": True, "inference_type": "transitivity"},
                                    evidence=[rel_id, other_rel_id]
                                )
                                new_relations.append(new_relation)

        # Similarity-based inference
        similarity_relations = self._infer_similarity_relations(confidence_threshold)
        new_relations.extend(similarity_relations)

        return new_relations[:50]  # Limit to prevent explosion

    def _relation_exists(self, source: str, target: str, relation_type: RelationType) -> bool:
        """Check if relation already exists"""

        if not self.graph.has_edge(source, target):
            return False

        edge_data = self.graph[source][target]

        for edge_attrs in edge_data.values():
            if edge_attrs.get('relation_type') == relation_type.value:
                return True

        return False

    def _infer_similarity_relations(self, confidence_threshold: float) -> List[KnowledgeRelation]:
        """Infer similarity relations based on shared properties"""

        similarity_relations = []
        concept_nodes = self.find_nodes_by_type(KnowledgeType.CONCEPT)

        for i, node1 in enumerate(concept_nodes):
            for node2 in concept_nodes[i+1:]:
                # Calculate similarity
                similarity_score = self._calculate_node_similarity(node1, node2)

                if similarity_score >= confidence_threshold:
                    # Create bidirectional similarity relations
                    rel1 = KnowledgeRelation(
                        relation_id=str(uuid.uuid4())[:8],
                        source_node=node1.node_id,
                        target_node=node2.node_id,
                        relation_type=RelationType.SIMILAR_TO,
                        strength=similarity_score,
                        confidence=similarity_score * 0.9,
                        properties={"inferred": True, "similarity_score": similarity_score}
                    )

                    rel2 = KnowledgeRelation(
                        relation_id=str(uuid.uuid4())[:8],
                        source_node=node2.node_id,
                        target_node=node1.node_id,
                        relation_type=RelationType.SIMILAR_TO,
                        strength=similarity_score,
                        confidence=similarity_score * 0.9,
                        properties={"inferred": True, "similarity_score": similarity_score}
                    )

                    similarity_relations.extend([rel1, rel2])

        return similarity_relations

    def _calculate_node_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Calculate similarity between two nodes"""

        # Content similarity
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())

        if not words1 or not words2:
            content_similarity = 0.0
        else:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            content_similarity = overlap / total

        # Property similarity
        props1 = set(node1.properties.keys())
        props2 = set(node2.properties.keys())

        if not props1 or not props2:
            property_similarity = 0.0
        else:
            prop_overlap = len(props1.intersection(props2))
            prop_total = len(props1.union(props2))
            property_similarity = prop_overlap / prop_total

        # Combined similarity
        return (content_similarity * 0.7) + (property_similarity * 0.3)

    def query_knowledge(self, query: str, query_type: str = "semantic") -> Dict[str, Any]:
        """Query knowledge graph with different strategies"""

        if query_type == "semantic":
            return self._semantic_query(query)
        elif query_type == "pattern":
            return self._pattern_query(query)
        elif query_type == "reasoning":
            return self._reasoning_query(query)
        else:
            return self._simple_query(query)

    def _semantic_query(self, query: str) -> Dict[str, Any]:
        """Semantic query using content similarity"""

        matching_nodes = self.find_nodes_by_content(query, similarity_threshold=0.5)

        results = {
            "query": query,
            "query_type": "semantic",
            "direct_matches": [],
            "related_knowledge": [],
            "reasoning_paths": []
        }

        for node in matching_nodes[:5]:  # Top 5 matches
            results["direct_matches"].append({
                "node_id": node.node_id,
                "content": node.content,
                "confidence": node.confidence,
                "type": node.knowledge_type.value
            })

            # Get related knowledge
            related = self.get_related_nodes(node.node_id, max_depth=2)
            for relation_type, related_nodes in related.items():
                for related_node in related_nodes[:3]:  # Top 3 per relation
                    results["related_knowledge"].append({
                        "relation": relation_type,
                        "node": {
                            "content": related_node.content,
                            "confidence": related_node.confidence
                        }
                    })

        return results

    def _reasoning_query(self, query: str) -> Dict[str, Any]:
        """Reasoning query to find logical connections"""

        # Parse query for potential source/target concepts
        query_words = query.lower().split()

        # Find potential source and target nodes
        source_candidates = []
        target_candidates = []

        for word in query_words:
            matching_nodes = self.find_nodes_by_content(word, similarity_threshold=0.8)
            if matching_nodes:
                if not source_candidates:
                    source_candidates = matching_nodes[:2]
                else:
                    target_candidates = matching_nodes[:2]

        results = {
            "query": query,
            "query_type": "reasoning",
            "reasoning_paths": [],
            "inferences": []
        }

        # Find reasoning paths between candidates
        for source in source_candidates:
            for target in target_candidates:
                if source.node_id != target.node_id:
                    paths = self.find_reasoning_paths(source.node_id, target.node_id)

                    for path in paths[:3]:  # Top 3 paths
                        results["reasoning_paths"].append({
                            "source": source.content,
                            "target": target.content,
                            "confidence": path.confidence,
                            "explanation": path.explanation,
                            "path_length": path.path_length
                        })

        return results


class OntologyManager:
    """Manages ontologies and concept hierarchies"""

    def __init__(self):
        self.concept_hierarchy = nx.DiGraph()
        self.concept_properties = {}
        self.domain_ontologies = {}

    def add_concept(self, concept_name: str, parent_concept: str = None,
                   properties: Dict[str, Any] = None) -> str:
        """Add concept to ontology"""

        concept_id = f"concept_{uuid.uuid4().hex[:8]}"

        self.concept_hierarchy.add_node(concept_id, name=concept_name)

        if properties:
            self.concept_properties[concept_id] = properties

        if parent_concept:
            parent_id = self._find_concept_id(parent_concept)
            if parent_id:
                self.concept_hierarchy.add_edge(parent_id, concept_id, relation="is_a")

        return concept_id

    def _find_concept_id(self, concept_name: str) -> Optional[str]:
        """Find concept ID by name"""

        for node_id, data in self.concept_hierarchy.nodes(data=True):
            if data.get("name") == concept_name:
                return node_id
        return None

    def get_concept_ancestors(self, concept_name: str) -> List[str]:
        """Get all ancestor concepts"""

        concept_id = self._find_concept_id(concept_name)
        if not concept_id:
            return []

        ancestors = []
        for ancestor_id in nx.ancestors(self.concept_hierarchy, concept_id):
            ancestor_name = self.concept_hierarchy.nodes[ancestor_id].get("name")
            if ancestor_name:
                ancestors.append(ancestor_name)

        return ancestors

    def get_concept_descendants(self, concept_name: str) -> List[str]:
        """Get all descendant concepts"""

        concept_id = self._find_concept_id(concept_name)
        if not concept_id:
            return []

        descendants = []
        for descendant_id in nx.descendants(self.concept_hierarchy, concept_id):
            descendant_name = self.concept_hierarchy.nodes[descendant_id].get("name")
            if descendant_name:
                descendants.append(descendant_name)

        return descendants


class KnowledgeRepresentationSystem:
    """Main knowledge representation system"""

    def __init__(self, storage_path: str = None):
        self.knowledge_graph = KnowledgeGraph()
        self.ontology_manager = OntologyManager()
        self.storage_path = Path(storage_path) if storage_path else Path("knowledge_storage")
        self.storage_path.mkdir(exist_ok=True)

        # Performance tracking
        self.query_cache = {}
        self.inference_cache = {}

    async def add_reasoning_knowledge(self, reasoning_chain: List[Dict[str, Any]],
                                    question: str, answer: str, confidence: float) -> List[str]:
        """Extract and add knowledge from reasoning chain"""

        added_nodes = []

        # Add question as a concept node
        question_node = KnowledgeNode(
            node_id=f"question_{uuid.uuid4().hex[:8]}",
            content=question,
            knowledge_type=KnowledgeType.CONCEPT,
            confidence=0.8,
            properties={"type": "question"}
        )

        question_id = self.knowledge_graph.add_knowledge_node(question_node)
        added_nodes.append(question_id)

        # Add answer as a fact node
        answer_node = KnowledgeNode(
            node_id=f"answer_{uuid.uuid4().hex[:8]}",
            content=answer,
            knowledge_type=KnowledgeType.FACT,
            confidence=confidence,
            properties={"type": "answer", "reasoning_confidence": confidence}
        )

        answer_id = self.knowledge_graph.add_knowledge_node(answer_node)
        added_nodes.append(answer_id)

        # Create relation between question and answer
        qa_relation = KnowledgeRelation(
            relation_id=f"qa_rel_{uuid.uuid4().hex[:8]}",
            source_node=question_id,
            target_node=answer_id,
            relation_type=RelationType.IMPLIES,
            strength=confidence,
            confidence=confidence,
            properties={"type": "question_answer"}
        )

        self.knowledge_graph.add_knowledge_relation(qa_relation)

        # Extract knowledge from reasoning steps
        previous_node_id = question_id

        for i, step in enumerate(reasoning_chain):
            if "inference" in step:
                inference = step["inference"]

                # Create node for reasoning step
                step_node = KnowledgeNode(
                    node_id=f"step_{uuid.uuid4().hex[:8]}",
                    content=inference.conclusion,
                    knowledge_type=KnowledgeType.RULE,
                    confidence=inference.confidence,
                    properties={
                        "premise": inference.premise,
                        "inference_rule": inference.inference_rule,
                        "step_number": i
                    }
                )

                step_id = self.knowledge_graph.add_knowledge_node(step_node)
                added_nodes.append(step_id)

                # Create relation from previous step
                step_relation = KnowledgeRelation(
                    relation_id=f"step_rel_{uuid.uuid4().hex[:8]}",
                    source_node=previous_node_id,
                    target_node=step_id,
                    relation_type=RelationType.IMPLIES,
                    strength=inference.confidence,
                    confidence=inference.confidence,
                    properties={"reasoning_step": True}
                )

                self.knowledge_graph.add_knowledge_relation(step_relation)
                previous_node_id = step_id

        # Connect final step to answer
        if previous_node_id != question_id:
            final_relation = KnowledgeRelation(
                relation_id=f"final_rel_{uuid.uuid4().hex[:8]}",
                source_node=previous_node_id,
                target_node=answer_id,
                relation_type=RelationType.IMPLIES,
                strength=confidence,
                confidence=confidence,
                properties={"final_step": True}
            )

            self.knowledge_graph.add_knowledge_relation(final_relation)

        logger.info(f"Added {len(added_nodes)} knowledge nodes from reasoning chain")
        return added_nodes

    async def query_relevant_knowledge(self, question: str,
                                     max_results: int = 10) -> Dict[str, Any]:
        """Query knowledge relevant to a question"""

        # Check cache first
        cache_key = f"query_{hash(question)}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Perform semantic query
        results = self.knowledge_graph.query_knowledge(question, query_type="semantic")

        # Enhance with reasoning paths
        reasoning_results = self.knowledge_graph.query_knowledge(question, query_type="reasoning")
        results["reasoning_paths"].extend(reasoning_results.get("reasoning_paths", []))

        # Limit results
        results["direct_matches"] = results["direct_matches"][:max_results]
        results["related_knowledge"] = results["related_knowledge"][:max_results * 2]
        results["reasoning_paths"] = results["reasoning_paths"][:max_results]

        # Cache results
        self.query_cache[cache_key] = results

        return results

    async def save_knowledge(self, filepath: str = None) -> str:
        """Save knowledge representation to file"""

        if not filepath:
            filepath = self.storage_path / f"knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        knowledge_data = {
            "nodes": self.knowledge_graph.nodes,
            "relations": self.knowledge_graph.relations,
            "graph": nx.node_link_data(self.knowledge_graph.graph),
            "concept_hierarchy": nx.node_link_data(self.ontology_manager.concept_hierarchy),
            "concept_properties": self.ontology_manager.concept_properties,
            "saved_at": datetime.utcnow().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(knowledge_data, f)

        logger.info(f"Knowledge saved to {filepath}")
        return str(filepath)

    async def load_knowledge(self, filepath: str) -> bool:
        """Load knowledge representation from file"""

        try:
            with open(filepath, 'rb') as f:
                knowledge_data = pickle.load(f)

            # Restore nodes and relations
            self.knowledge_graph.nodes = knowledge_data["nodes"]
            self.knowledge_graph.relations = knowledge_data["relations"]

            # Restore graphs
            self.knowledge_graph.graph = nx.node_link_graph(knowledge_data["graph"])
            self.ontology_manager.concept_hierarchy = nx.node_link_graph(knowledge_data["concept_hierarchy"])
            self.ontology_manager.concept_properties = knowledge_data["concept_properties"]

            # Rebuild indexes
            self._rebuild_indexes()

            logger.info(f"Knowledge loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load knowledge from {filepath}: {e}")
            return False

    def _rebuild_indexes(self):
        """Rebuild search indexes after loading"""

        # Rebuild type index
        self.knowledge_graph.type_index.clear()
        for node_id, node in self.knowledge_graph.nodes.items():
            self.knowledge_graph.type_index[node.knowledge_type].add(node_id)

        # Rebuild content index
        self.knowledge_graph.content_index.clear()
        for node_id, node in self.knowledge_graph.nodes.items():
            self.knowledge_graph.content_index[node.content.lower()] = node_id

        # Rebuild relation index
        self.knowledge_graph.relation_index.clear()
        for relation_id, relation in self.knowledge_graph.relations.items():
            self.knowledge_graph.relation_index[relation.relation_type].append(relation_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get knowledge system statistics"""

        return {
            "knowledge_graph": {
                "nodes": len(self.knowledge_graph.nodes),
                "relations": len(self.knowledge_graph.relations),
                "graph_density": nx.density(self.knowledge_graph.graph),
                "connected_components": nx.number_weakly_connected_components(self.knowledge_graph.graph)
            },
            "ontology": {
                "concepts": len(self.ontology_manager.concept_hierarchy.nodes),
                "hierarchy_depth": self._calculate_hierarchy_depth(),
                "concept_properties": len(self.ontology_manager.concept_properties)
            },
            "performance": {
                "query_cache_size": len(self.query_cache),
                "inference_cache_size": len(self.inference_cache)
            },
            "node_types": {
                knowledge_type.value: len(nodes)
                for knowledge_type, nodes in self.knowledge_graph.type_index.items()
            }
        }

    def _calculate_hierarchy_depth(self) -> int:
        """Calculate maximum depth of concept hierarchy"""

        if not self.ontology_manager.concept_hierarchy.nodes:
            return 0

        # Find root nodes (nodes with no predecessors)
        root_nodes = [n for n in self.ontology_manager.concept_hierarchy.nodes()
                      if self.ontology_manager.concept_hierarchy.in_degree(n) == 0]

        max_depth = 0
        for root in root_nodes:
            try:
                depth = nx.dag_longest_path_length(
                    self.ontology_manager.concept_hierarchy.subgraph(
                        nx.descendants(self.ontology_manager.concept_hierarchy, root) | {root}
                    )
                )
                max_depth = max(max_depth, depth)
            except:
                continue

        return max_depth


# Factory function
def create_knowledge_representation_system(storage_path: str = None) -> KnowledgeRepresentationSystem:
    """Create configured knowledge representation system"""

    system = KnowledgeRepresentationSystem(storage_path)
    logger.info("Created knowledge representation system")

    return system

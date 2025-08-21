"""
Vector Processing Agent - A2A Microservice
Agent 3: Stores vectors and enables similarity search
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass
import hashlib

import sys
sys.path.append('../../shared')

import sys
import os
# Add the shared directory to Python path for a2aCommon imports
shared_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, os.path.abspath(shared_path))

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    entity_id: str
    entity_type: str
    similarity_score: float
    metadata: Dict[str, Any]


class MockVectorDB:
    """Mock vector database for development"""
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
        self.graph = {"nodes": [], "edges": []}
    
    async def store_vector(self, entity_id: str, vector: List[float], metadata: Dict[str, Any]):
        """Store vector with metadata"""
        self.vectors[entity_id] = np.array(vector)
        self.metadata[entity_id] = metadata
        return True
    
    async def search_similar(self, query_vector: List[float], top_k: int = 10) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        query = np.array(query_vector)
        results = []
        
        for entity_id, vector in self.vectors.items():
            # Cosine similarity
            similarity = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector))
            results.append(VectorSearchResult(
                entity_id=entity_id,
                entity_type=self.metadata[entity_id].get("entity_type", "unknown"),
                similarity_score=float(similarity),
                metadata=self.metadata[entity_id]
            ))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    async def add_to_graph(self, nodes: List[Dict], edges: List[Dict]):
        """Add nodes and edges to knowledge graph"""
        self.graph["nodes"].extend(nodes)
        self.graph["edges"].extend(edges)
        return True


class VectorProcessingAgent(A2AAgentBase):
    """
    Agent 3: Vector Processing Agent
    Stores embeddings and enables vector search
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, vector_db_config: Dict[str, Any]):
        super().__init__(
            agent_id="vector_processing_agent_3",
            name="Vector Processing Agent",
            description="A2A v0.2.9 compliant agent for vector storage and similarity search",
            version="1.0.0",
            base_url=base_url
        )
        
        self.agent_manager_url = agent_manager_url
        self.vector_db_config = vector_db_config
        self.vector_db_connected = False
        
        # Initialize processing stats
        self.processing_stats = {
            "total_processed": 0,
            "vectors_stored": 0,
            "graph_nodes_added": 0,
            "graph_edges_added": 0,
            "searches_performed": 0
        }
        
        logger.info(f"Initialized A2A {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """Initialize agent resources and vector database"""
        logger.info("Initializing Vector Processing Agent...")
        
        # Initialize vector database
        if self.vector_db_config.get("use_mock", True):
            logger.info("Using mock vector database")
            self.vector_db = MockVectorDB()
            self.vector_db_connected = True
        else:
            # In production, initialize actual HANA connection
            logger.warning("HANA integration not implemented, using mock")
            self.vector_db = MockVectorDB()
            self.vector_db_connected = True
        
        # Initialize A2A trust identity
        await self._initialize_trust_identity()
        
        self.is_ready = True
        logger.info("Vector Processing Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            registration = {
                "agent_id": self.agent_id,
                "name": self.name,
                "base_url": self.base_url,
                "capabilities": {
                    "vector_storage": True,
                    "similarity_search": True,
                    "knowledge_graph": True,
                    "supported_dimensions": [384, 768, 1536],
                    "max_vectors": 1000000,
                    "index_types": ["flat", "hnsw"]
                },
                "handlers": [h.name for h in self.handlers.values()],
                "skills": [s.name for s in self.skills.values()]
            }
            
            logger.info(f"Registered with A2A network at {self.agent_manager_url}")
            self.is_registered = True
            
        except Exception as e:
            logger.error(f"Failed to register with A2A network: {e}")
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
    
    @a2a_handler("store_vectors", "Store AI-prepared vectors in database")
    async def handle_vector_storage_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for vector storage requests"""
        try:
            # Extract AI-prepared data from A2A message
            ai_prepared_data = self._extract_ai_prepared_data(message)
            
            if not ai_prepared_data:
                return create_error_response(400, "No AI-prepared data found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("vector_storage", {
                "context_id": context_id,
                "data_types": list(ai_prepared_data.keys()),
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_vector_storage(task_id, ai_prepared_data, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "data_types": list(ai_prepared_data.keys()),
                "message": "Vector storage started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling vector storage request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("store_embedding", "Store a single embedding")
    async def store_embedding(self, entity_id: str, embedding: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Store a single embedding in the vector database"""
        try:
            vector = embedding.get("vector", [])
            if not vector:
                return False
            
            # Add embedding metadata
            full_metadata = {
                **metadata,
                "embedding_model": embedding.get("model", "unknown"),
                "embedding_dimension": embedding.get("dimension", len(vector)),
                "stored_timestamp": datetime.utcnow().isoformat()
            }
            
            success = await self.vector_db.store_vector(entity_id, vector, full_metadata)
            if success:
                self.processing_stats["vectors_stored"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    @a2a_skill("search_similar", "Search for similar vectors")
    async def search_similar_vectors(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in the database"""
        try:
            results = await self.vector_db.search_similar(query_vector, top_k)
            
            # Apply filters if provided
            if filters:
                results = [r for r in results if self._match_filters(r.metadata, filters)]
            
            self.processing_stats["searches_performed"] += 1
            
            # Convert to dict format
            return [
                {
                    "entity_id": r.entity_id,
                    "entity_type": r.entity_type,
                    "similarity_score": r.similarity_score,
                    "metadata": r.metadata
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching similar vectors: {e}")
            return []
    
    @a2a_skill("build_knowledge_graph", "Build knowledge graph from relationships")
    async def build_knowledge_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build knowledge graph from entities and relationships"""
        try:
            # Create nodes
            nodes = []
            for entity in entities:
                node = {
                    "id": entity.get("id", str(hash(str(entity)))),
                    "type": entity.get("entity_type", "unknown"),
                    "label": entity.get("name", entity.get("id", "unnamed")),
                    "properties": {k: v for k, v in entity.items() if k not in ["embedding", "id"]}
                }
                nodes.append(node)
            
            # Create edges
            edges = []
            for rel in relationships:
                edge = {
                    "id": f"{rel['source_id']}_{rel['target_id']}_{rel['relationship_type']}",
                    "source": rel["source_id"],
                    "target": rel["target_id"],
                    "type": rel["relationship_type"],
                    "properties": rel.get("attributes", {})
                }
                edges.append(edge)
            
            # Store in graph database
            await self.vector_db.add_to_graph(nodes, edges)
            
            self.processing_stats["graph_nodes_added"] += len(nodes)
            self.processing_stats["graph_edges_added"] += len(edges)
            
            return {
                "nodes_added": len(nodes),
                "edges_added": len(edges),
                "total_nodes": self.processing_stats["graph_nodes_added"],
                "total_edges": self.processing_stats["graph_edges_added"]
            }
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {"error": str(e)}
    
    async def _process_vector_storage(self, task_id: str, data: Dict[str, Any], context_id: str):
        """Process vector storage asynchronously"""
        try:
            storage_results = {}
            
            # Extract relationships if present
            relationships = data.pop("relationships", [])
            
            # Process each entity type
            for entity_type, entities in data.items():
                if isinstance(entities, list) and len(entities) > 0:
                    logger.info(f"Storing {len(entities)} {entity_type} vectors")
                    
                    stored_count = 0
                    for entity in entities:
                        if "embedding" in entity:
                            entity_id = entity.get("id", str(hash(str(entity))))
                            metadata = {
                                "entity_type": entity_type,
                                "context_id": context_id,
                                **{k: v for k, v in entity.items() if k not in ["embedding", "id"]}
                            }
                            
                            success = await self.store_embedding(
                                entity_id,
                                entity["embedding"],
                                metadata
                            )
                            
                            if success:
                                stored_count += 1
                    
                    storage_results[entity_type] = {
                        "total": len(entities),
                        "stored": stored_count
                    }
            
            # Build knowledge graph if relationships exist
            if relationships:
                all_entities = []
                for entity_type, entities in data.items():
                    if isinstance(entities, list):
                        for e in entities:
                            e["entity_type"] = entity_type
                        all_entities.extend(entities)
                
                graph_result = await self.build_knowledge_graph(all_entities, relationships)
                storage_results["knowledge_graph"] = graph_result
            
            # Update stats
            self.processing_stats["total_processed"] += 1
            
            # Update task status
            await self.update_task_status(task_id, "completed", {
                "storage_results": storage_results,
                "total_vectors_stored": self.processing_stats["vectors_stored"]
            })
            
            # Send completion notification (in production, send to interested agents)
            logger.info(f"Vector storage completed for context {context_id}")
            
        except Exception as e:
            logger.error(f"Error processing vector storage: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    def _extract_ai_prepared_data(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract AI-prepared data from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('ai_prepared_data', content.get('data', None))
        return None
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filters"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def _initialize_trust_identity(self) -> None:
        """Initialize agent's trust identity for A2A network"""
        logger.info(f"Trust identity initialization placeholder for {self.agent_id}")
        pass
    
    async def create_task(self, task_type: str, metadata: Dict[str, Any]) -> str:
        """Create and track a new task"""
        import uuid
        task_id = str(uuid.uuid4())
        
        if not hasattr(self, 'tasks'):
            self.tasks = {}
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        return task_id
    
    async def update_task_status(self, task_id: str, status: str, update_data: Dict[str, Any] = None):
        """Update task status and metadata"""
        if not hasattr(self, 'tasks'):
            self.tasks = {}
            
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
            
            if update_data:
                self.tasks[task_id]["metadata"].update(update_data)
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Vector Processing Agent...")
        self.is_ready = False
        self.vector_db_connected = False
        logger.info("Vector Processing Agent shutdown complete")
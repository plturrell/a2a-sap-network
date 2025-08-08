"""
SAP HANA Vector Engine Ingestion & Knowledge Graph Agent - SDK Version
Agent 3: Enhanced with A2A SDK for simplified development and maintenance
"""

import asyncio
import uuid
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from pydantic import BaseModel, Field
import logging
import hashlib

# SAP HANA Cloud integration
try:
    from langchain_hana import HanaDB, HanaInternalEmbeddings
    from langchain_hana.vectorstores import DistanceStrategy
    from hdbcli import dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    DistanceStrategy = None
    logging.warning("SAP HANA Cloud integration not available. Install langchain-hana package.")

# Sentence Transformers for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence Transformers not available. Install sentence-transformers package.")

# NetworkX for real graph operations
try:
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    logging.warning("NetworkX not available. Install networkx package.")

from ..sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)
from ..sdk.utils import create_success_response, create_error_response
from ..core.workflow_context import workflow_context_manager
from ..core.workflow_monitor import workflow_monitor
from ..security.smart_contract_trust import sign_a2a_message
from ..security.delegation_contracts import DelegationAction
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

logger = logging.getLogger(__name__)


class VectorDocument(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: List[float]
    created_at: datetime
    entity_type: str
    source_agent: str


class KnowledgeGraphNode(BaseModel):
    node_id: str
    entity_id: str
    entity_type: str
    properties: Dict[str, Any]
    vector_reference: str
    created_at: datetime
    updated_at: datetime


class KnowledgeGraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence_score: float
    created_at: datetime


class SearchResult(BaseModel):
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    entity_type: str


class VectorProcessingAgentSDK(A2AAgentBase):
    """
    Agent 3: SAP HANA Vector Engine Ingestion & Knowledge Graph Agent
    SDK Version - Enhanced vector processing and knowledge graph management
    """
    
    def __init__(self, base_url: str, hana_config: Dict[str, Any]):
        super().__init__(
            agent_id="vector_processing_agent_3",
            name="SAP HANA Vector Engine Ingestion Agent",
            description="A2A v0.2.9 compliant agent for vector processing and knowledge graph management",
            version="3.0.0",  # SDK version
            base_url=base_url
        )
        
        self.hana_config = hana_config
        self.vector_store = None
        self.hana_connection = None
        self.embedding_model = None
        self.knowledge_graph = nx.DiGraph() if GRAPH_AVAILABLE else {}
        self._graph_fallback = not GRAPH_AVAILABLE
        
        # Prometheus metrics
        self.tasks_completed = Counter('a2a_agent_tasks_completed_total', 'Total completed tasks', ['agent_id', 'task_type'])
        self.tasks_failed = Counter('a2a_agent_tasks_failed_total', 'Total failed tasks', ['agent_id', 'task_type'])
        self.processing_time = Histogram('a2a_agent_processing_time_seconds', 'Task processing time', ['agent_id', 'task_type'])
        self.queue_depth = Gauge('a2a_agent_queue_depth', 'Current queue depth', ['agent_id'])
        self.skills_count = Gauge('a2a_agent_skills_count', 'Number of skills available', ['agent_id'])
        
        # Set initial metrics
        self.queue_depth.labels(agent_id=self.agent_id).set(0)
        self.skills_count.labels(agent_id=self.agent_id).set(5)  # 5 main skills
        
        # Start metrics server
        self._start_metrics_server()
        
        self.processing_stats = {
            "total_processed": 0,
            "vectors_ingested": 0,
            "knowledge_graph_nodes": 0,
            "search_queries": 0
        }
        
        logger.info(f"Initialized {self.name} with SDK v3.0.0")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = int(os.environ.get('PROMETHEUS_PORT', '8004'))
            start_http_server(port)
            logger.info(f"Started Prometheus metrics server on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Vector Processing Agent resources...")
        
        # Initialize vector storage
        storage_path = os.getenv("VECTOR_AGENT_STORAGE_PATH", "/tmp/vector_agent_state")
        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        
        # Initialize HANA connection if available
        if HANA_AVAILABLE:
            await self._initialize_hana_connection()
        else:
            logger.warning("HANA integration not available, using fallback storage")
            await self._initialize_fallback_storage()
        
        # Load existing state
        await self._load_agent_state()
        
        logger.info("Vector Processing Agent initialization complete")
    
    @a2a_handler("vector_ingestion")
    async def handle_vector_ingestion(self, message: A2AMessage) -> Dict[str, Any]:
        """Main handler for vector ingestion requests"""
        start_time = time.time()
        
        try:
            # Extract AI-ready entity data from message
            ai_entity_data = self._extract_ai_entity_data(message)
            if not ai_entity_data:
                return create_error_response("No valid AI entity data found in message")
            
            # Process vector ingestion
            ingestion_result = await self.ingest_ai_entity(
                ai_entity_data=ai_entity_data,
                context_id=message.conversation_id
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='vector_ingestion').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='vector_ingestion').observe(time.time() - start_time)
            
            return create_success_response(ingestion_result)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='vector_ingestion').inc()
            logger.error(f"Vector ingestion failed: {e}")
            return create_error_response(f"Vector ingestion failed: {str(e)}")
    
    @a2a_handler("semantic_search")
    async def handle_semantic_search(self, message: A2AMessage) -> Dict[str, Any]:
        """Handler for semantic search requests"""
        start_time = time.time()
        
        try:
            # Verify message trust if trust system is enabled
            if self.trust_identity:
                trust_verification = await verify_a2a_message(
                    message.dict() if hasattr(message, 'dict') else message,
                    self.agent_id
                )
                
                if not trust_verification["valid"]:
                    logger.warning(f"Trust verification failed: {trust_verification['error']}")
                    return create_error_response(f"Trust verification failed: {trust_verification['error']}")
                
                # Add sender to trusted agents if verification passed
                sender_id = trust_verification.get("signer_id")
                if sender_id:
                    self.trusted_agents.add(sender_id)
            
            # Extract search parameters
            search_params = self._extract_search_params(message)
            if not search_params.get('query'):
                return create_error_response("No search query provided")
            
            # Perform semantic search
            search_results = await self.semantic_search(
                query=search_params['query'],
                filters=search_params.get('filters', {}),
                limit=search_params.get('limit', 10)
            )
            
            # Record success metrics
            self.tasks_completed.labels(agent_id=self.agent_id, task_type='semantic_search').inc()
            self.processing_time.labels(agent_id=self.agent_id, task_type='semantic_search').observe(time.time() - start_time)
            
            response_data = {
                "search_results": search_results,
                "query": search_params['query'],
                "result_count": len(search_results)
            }
            
            # Sign response if trust system is enabled
            if self.trust_identity:
                response_data = await sign_a2a_message(response_data, self.agent_id)
            
            return create_success_response(response_data)
            
        except Exception as e:
            # Record failure metrics
            self.tasks_failed.labels(agent_id=self.agent_id, task_type='semantic_search').inc()
            logger.error(f"Semantic search failed: {e}")
            return create_error_response(f"Semantic search failed: {str(e)}")
    
    @a2a_skill("vector_storage")
    async def vector_storage_skill(self, ai_entity: Dict[str, Any]) -> str:
        """Store vector embeddings in HANA Vector Engine"""
        
        # Create vector document
        vector_doc = VectorDocument(
            doc_id=str(uuid.uuid4()),
            content=self._create_document_content(ai_entity),
            metadata=self._extract_metadata(ai_entity),
            embeddings=ai_entity.get('vector_representation', {}).get('vector_embedding', []),
            created_at=datetime.utcnow(),
            entity_type=ai_entity.get('original_entity', {}).get('entity_type', 'unknown'),
            source_agent=ai_entity.get('processing_metadata', {}).get('agent_version', 'unknown')
        )
        
        # Store in HANA or fallback
        if self.vector_store:
            doc_id = await self._store_in_hana(vector_doc)
        else:
            doc_id = await self._store_in_fallback(vector_doc)
        
        self.processing_stats["vectors_ingested"] += 1
        return doc_id
    
    @a2a_skill("knowledge_graph_creation")
    async def knowledge_graph_creation_skill(self, ai_entity: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph nodes and edges"""
        
        entity_id = ai_entity.get('entity_id')
        
        # Create knowledge graph node
        kg_node = KnowledgeGraphNode(
            node_id=str(uuid.uuid4()),
            entity_id=entity_id,
            entity_type=ai_entity.get('original_entity', {}).get('entity_type', 'unknown'),
            properties=self._extract_node_properties(ai_entity),
            vector_reference=ai_entity.get('vector_representation', {}).get('entity_id', ''),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Create edges from relationships
        kg_edges = []
        for relationship in ai_entity.get('entity_relationships', []):
            edge = KnowledgeGraphEdge(
                edge_id=str(uuid.uuid4()),
                source_node_id=kg_node.node_id,
                target_node_id=relationship.get('target_entity'),
                relationship_type=relationship.get('relationship_type'),
                properties=relationship.get('properties', {}),
                confidence_score=relationship.get('confidence_score', 0.8),
                created_at=datetime.utcnow()
            )
            kg_edges.append(edge)
        
        # Store in knowledge graph (real graph database or HANA)
        if self.hana_connection:
            await self._store_graph_in_hana(kg_node, kg_edges)
        elif GRAPH_AVAILABLE and not self._graph_fallback:
            await self._store_graph_in_networkx(kg_node, kg_edges)
        else:
            # Fallback to dictionary storage
            self.knowledge_graph[kg_node.node_id] = {
                "node": kg_node.dict(),
                "edges": [edge.dict() for edge in kg_edges]
            }
        
        self.processing_stats["knowledge_graph_nodes"] += 1
        
        return {
            "node_id": kg_node.node_id,
            "entity_id": entity_id,
            "edges_created": len(kg_edges)
        }
    
    @a2a_skill("similarity_search")
    async def similarity_search_skill(self, query_vector: List[float], filters: Dict[str, Any] = None, limit: int = 10) -> List[SearchResult]:
        """Perform similarity search using vector embeddings"""
        
        if self.vector_store and HANA_AVAILABLE:
            results = await self._search_hana(query_vector, filters, limit)
        else:
            results = await self._search_fallback(query_vector, filters, limit)
        
        self.processing_stats["search_queries"] += 1
        return results
    
    @a2a_skill("knowledge_graph_query")
    async def knowledge_graph_query_skill(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query knowledge graph for entity relationships"""
        
        # Find node by entity_id
        target_node = None
        target_node_id = None
        
        for node_id, data in self.knowledge_graph.items():
            if data["node"]["entity_id"] == entity_id:
                target_node = data["node"]
                target_node_id = node_id
                break
        
        if not target_node:
            return {"error": f"Entity {entity_id} not found in knowledge graph"}
        
        # Traverse relationships
        relationships = []
        visited = set()
        
        def traverse(node_id: str, depth: int):
            if depth >= max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            node_data = self.knowledge_graph.get(node_id, {})
            
            for edge in node_data.get("edges", []):
                relationships.append({
                    "source": node_data["node"]["entity_id"],
                    "target": edge["target_node_id"],
                    "relationship": edge["relationship_type"],
                    "confidence": edge["confidence_score"],
                    "depth": depth
                })
                
                # Recursively traverse
                traverse(edge["target_node_id"], depth + 1)
        
        traverse(target_node_id, 0)
        
        return {
            "entity_id": entity_id,
            "relationships": relationships,
            "relationship_count": len(relationships)
        }
    
    @a2a_task(
        task_type="vector_entity_ingestion",
        description="Complete vector ingestion workflow for AI entities",
        timeout=300,
        retry_attempts=2
    )
    async def ingest_ai_entity(self, ai_entity_data: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Complete vector ingestion workflow"""
        
        entity_id = ai_entity_data.get('entity_id')
        
        try:
            # Stage 1: Vector storage
            vector_doc_id = await self.execute_skill("vector_storage", ai_entity_data)
            
            # Stage 2: Knowledge graph creation
            kg_result = await self.execute_skill("knowledge_graph_creation", ai_entity_data)
            
            # Stage 3: Validate ingestion
            validation_result = await self._validate_ingestion(vector_doc_id, kg_result)
            
            self.processing_stats["total_processed"] += 1
            
            return {
                "ingestion_successful": True,
                "entity_id": entity_id,
                "vector_doc_id": vector_doc_id,
                "knowledge_graph_node": kg_result["node_id"],
                "validation_result": validation_result,
                "context_id": context_id
            }
            
        except Exception as e:
            logger.error(f"Vector entity ingestion failed: {e}")
            return {
                "ingestion_successful": False,
                "entity_id": entity_id,
                "error": str(e)
            }
    
    async def semantic_search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        
        # Generate query vector (placeholder - would use proper embedding model)
        query_vector = await self._generate_query_vector(query)
        
        # Perform similarity search
        search_results = await self.execute_skill("similarity_search", query_vector, filters, limit)
        
        # Convert to dict format
        results = []
        for result in search_results:
            results.append({
                "doc_id": result.doc_id,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata,
                "entity_type": result.entity_type
            })
        
        return results
    
    def _extract_ai_entity_data(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract AI entity data from message"""
        ai_entity_data = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                ai_entity_data.update(part.data)
            elif part.kind == "file" and part.file:
                ai_entity_data["file"] = part.file
        
        return ai_entity_data
    
    def _extract_search_params(self, message: A2AMessage) -> Dict[str, Any]:
        """Extract search parameters from message"""
        search_params = {}
        
        for part in message.parts:
            if part.kind == "data" and part.data:
                search_params.update(part.data)
        
        return search_params
    
    def _create_document_content(self, ai_entity: Dict[str, Any]) -> str:
        """Create document content from AI entity"""
        components = []
        
        # Add semantic description
        semantic = ai_entity.get('semantic_enrichment', {})
        if semantic.get('semantic_description'):
            components.append(semantic['semantic_description'])
        
        # Add domain terminology
        terminology = semantic.get('domain_terminology', [])
        if terminology:
            components.append(" ".join(terminology))
        
        # Add business context
        business_context = semantic.get('business_context', {})
        if business_context.get('primary_function'):
            components.append(business_context['primary_function'])
        
        return " ".join(filter(None, components))
    
    def _extract_metadata(self, ai_entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from AI entity"""
        metadata = {
            "entity_id": ai_entity.get('entity_id'),
            "entity_type": ai_entity.get('original_entity', {}).get('entity_type'),
            "ai_readiness_score": ai_entity.get('ai_readiness_score'),
            "source_agent": ai_entity.get('processing_metadata', {}).get('agent_version')
        }
        
        # Add semantic tags
        vector_rep = ai_entity.get('vector_representation', {})
        if vector_rep.get('semantic_tags'):
            metadata["semantic_tags"] = vector_rep['semantic_tags']
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _extract_node_properties(self, ai_entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties for knowledge graph node"""
        properties = {}
        
        # Add core properties
        original_entity = ai_entity.get('original_entity', {})
        properties.update(original_entity)
        
        # Add AI metrics
        properties['ai_readiness_score'] = ai_entity.get('ai_readiness_score')
        
        # Add quality metrics
        quality_metrics = ai_entity.get('quality_metrics', {})
        properties.update({f"quality_{k}": v for k, v in quality_metrics.items()})
        
        return properties
    
    async def _generate_query_vector(self, query: str) -> List[float]:
        """Generate vector for query using real embedding model"""
        try:
            if self.embedding_model:
                # Use real sentence transformer model
                embedding = self.embedding_model.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return embedding.tolist()
            else:
                # Fallback to hash-based approach when model not available
                logger.warning("Using hash-based query embedding fallback")
                return self._generate_hash_embedding(query, 384)
                
        except Exception as e:
            logger.error(f"Error generating query vector: {e}")
            return self._generate_hash_embedding(query, 384)
            
    def _generate_hash_embedding(self, text: str, dimension: int) -> List[float]:
        """Generate deterministic hash-based embedding as fallback"""
        import hashlib
        import struct
        
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        
        for i in range(0, min(len(text_hash), dimension//8), 8):
            chunk = text_hash[i:i+8].ljust(8, b'\x00')
            value = struct.unpack('d', chunk)[0]
            normalized = (value % 2.0) - 1.0
            embedding.append(normalized)
        
        while len(embedding) < dimension:
            embedding.append(0.0)
        
        return embedding[:dimension]
    
    async def _initialize_hana_connection(self):
        """Initialize HANA vector store connection"""
        try:
            if HANA_AVAILABLE and self.hana_config:
                # Initialize real HANA connection
                connection = dbapi.connect(
                    address=self.hana_config.get("address", "localhost"),
                    port=self.hana_config.get("port", 30015),
                    user=self.hana_config.get("user"),
                    password=self.hana_config.get("password"),
                    databaseName=self.hana_config.get("databaseName", "SYSTEMDB")
                )
                self.hana_connection = connection
                
                # Initialize HANA Vector Store
                self.vector_store = HanaDB(
                    connection=connection,
                    table_name="A2A_VECTORS",
                    content_column="CONTENT",
                    metadata_column="METADATA",
                    vector_column="VECTOR_EMBEDDING",
                    distance_strategy=DistanceStrategy.COSINE
                )
                
                # Create vector table if not exists
                await self._create_hana_vector_tables()
                
                logger.info("✅ HANA Vector Engine connection initialized")
            else:
                logger.warning("HANA config missing or HANA not available")
                await self._initialize_fallback_storage()
                
        except Exception as e:
            logger.error(f"Failed to initialize HANA connection: {e}")
            await self._initialize_fallback_storage()
            
        # Initialize embedding model
        await self._initialize_embedding_model()
    
    async def _initialize_fallback_storage(self):
        """Initialize fallback storage when HANA is not available"""
        logger.info("Using fallback vector storage")
        self.vector_store = None
        
        # Initialize embedding model even for fallback
        await self._initialize_embedding_model()
        
    async def _initialize_embedding_model(self):
        """Initialize sentence transformer model for real embeddings"""
        try:
            if EMBEDDINGS_AVAILABLE:
                # Use financial domain optimized model or general purpose
                model_name = "all-MiniLM-L6-v2"  # 384 dimensions, fast inference
                logger.info(f"Loading embedding model {model_name}...")
                self.embedding_model = SentenceTransformer(model_name)
                self._embedding_dim = 384
                logger.info("✅ Real embedding model initialized")
            else:
                logger.warning("Sentence transformers not available, using hash fallback")
                self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
            
    async def _create_hana_vector_tables(self):
        """Create HANA vector tables if they don't exist"""
        try:
            cursor = self.hana_connection.cursor()
            
            # Create vector storage table
            cursor.execute("""
                CREATE COLUMN TABLE IF NOT EXISTS A2A_VECTORS (
                    DOC_ID NVARCHAR(255) PRIMARY KEY,
                    CONTENT NCLOB,
                    METADATA NCLOB,
                    VECTOR_EMBEDDING REAL_VECTOR(384),
                    ENTITY_TYPE NVARCHAR(100),
                    SOURCE_AGENT NVARCHAR(100),
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create graph tables
            cursor.execute("""
                CREATE COLUMN TABLE IF NOT EXISTS A2A_GRAPH_NODES (
                    NODE_ID NVARCHAR(255) PRIMARY KEY,
                    ENTITY_ID NVARCHAR(255),
                    ENTITY_TYPE NVARCHAR(100),
                    PROPERTIES NCLOB,
                    VECTOR_REFERENCE NVARCHAR(255),
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE COLUMN TABLE IF NOT EXISTS A2A_GRAPH_EDGES (
                    EDGE_ID NVARCHAR(255) PRIMARY KEY,
                    SOURCE_NODE_ID NVARCHAR(255),
                    TARGET_NODE_ID NVARCHAR(255),
                    RELATIONSHIP_TYPE NVARCHAR(100),
                    PROPERTIES NCLOB,
                    CONFIDENCE_SCORE DOUBLE,
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.close()
            logger.info("✅ HANA vector tables created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create HANA vector tables: {e}")
            raise
    
    async def _store_in_hana(self, vector_doc: VectorDocument) -> str:
        """Store vector document in HANA vector database"""
        try:
            cursor = self.hana_connection.cursor()
            
            # Insert vector document
            cursor.execute("""
                INSERT INTO A2A_VECTORS (
                    DOC_ID, CONTENT, METADATA, VECTOR_EMBEDDING, 
                    ENTITY_TYPE, SOURCE_AGENT
                ) VALUES (?, ?, ?, TO_REAL_VECTOR(?), ?, ?)
            """, (
                vector_doc.doc_id,
                vector_doc.content,
                json.dumps(vector_doc.metadata),
                str(vector_doc.embeddings),  # HANA expects string representation
                vector_doc.entity_type,
                vector_doc.source_agent
            ))
            
            cursor.close()
            logger.info(f"✅ Stored vector document {vector_doc.doc_id} in HANA")
            return vector_doc.doc_id
            
        except Exception as e:
            logger.error(f"Failed to store vector in HANA: {e}")
            # Fallback to file storage
            return await self._store_in_fallback(vector_doc)
    
    async def _store_in_fallback(self, vector_doc: VectorDocument) -> str:
        """Store vector document in fallback storage"""
        doc_file = os.path.join(self.storage_path, f"vector_{vector_doc.doc_id}.json")
        with open(doc_file, 'w') as f:
            json.dump(vector_doc.dict(), f, default=str, indent=2)
        return vector_doc.doc_id
    
    async def _search_hana(self, query_vector: List[float], filters: Dict[str, Any], limit: int) -> List[SearchResult]:
        """Search in HANA vector store using real vector similarity"""
        try:
            cursor = self.hana_connection.cursor()
            
            # Build filter conditions
            where_conditions = []
            params = [str(query_vector), limit]
            
            if filters:
                for key, value in filters.items():
                    if key == 'entity_type':
                        where_conditions.append("ENTITY_TYPE = ?")
                        params.insert(-1, value)
                    elif key == 'source_agent':
                        where_conditions.append("SOURCE_AGENT = ?")
                        params.insert(-1, value)
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Execute vector similarity search
            query_sql = f"""
                SELECT DOC_ID, CONTENT, METADATA, ENTITY_TYPE,
                       COSINE_SIMILARITY(VECTOR_EMBEDDING, TO_REAL_VECTOR(?)) as SIMILARITY_SCORE
                FROM A2A_VECTORS
                {where_clause}
                ORDER BY SIMILARITY_SCORE DESC
                LIMIT ?
            """
            
            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            
            search_results = []
            for row in results:
                doc_id, content, metadata_json, entity_type, score = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                search_results.append(SearchResult(
                    doc_id=doc_id,
                    score=float(score),
                    content=content,
                    metadata=metadata,
                    entity_type=entity_type
                ))
            
            cursor.close()
            logger.info(f"✅ HANA vector search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"HANA vector search failed: {e}")
            # Fallback to file-based search
            return await self._search_fallback(query_vector, filters, limit)
    
    async def _search_fallback(self, query_vector: List[float], filters: Dict[str, Any], limit: int) -> List[SearchResult]:
        """Search in fallback storage"""
        results = []
        
        # Simple cosine similarity search in fallback storage
        for filename in os.listdir(self.storage_path):
            if filename.startswith('vector_') and filename.endswith('.json'):
                try:
                    doc_path = os.path.join(self.storage_path, filename)
                    with open(doc_path, 'r') as f:
                        doc_data = json.load(f)
                    
                    # Calculate similarity (simplified)
                    doc_embedding = doc_data.get('embeddings', [])
                    if doc_embedding:
                        similarity = self._cosine_similarity(query_vector, doc_embedding)
                        
                        # Apply filters
                        if self._passes_filters(doc_data.get('metadata', {}), filters):
                            results.append(SearchResult(
                                doc_id=doc_data['doc_id'],
                                score=similarity,
                                content=doc_data['content'],
                                metadata=doc_data['metadata'],
                                entity_type=doc_data['entity_type']
                            ))
                except Exception as e:
                    logger.warning(f"Error processing vector document {filename}: {e}")
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes the given filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True
    
    async def _validate_ingestion(self, vector_doc_id: str, kg_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the ingestion process"""
        validation = {
            "vector_stored": vector_doc_id is not None,
            "knowledge_graph_created": kg_result.get("node_id") is not None,
            "edges_created": kg_result.get("edges_created", 0) > 0
        }
        
        validation["overall_success"] = all(validation.values())
        return validation
    
    async def _load_agent_state(self):
        """Load existing agent state from storage"""
        try:
            # Load knowledge graph state
            kg_file = os.path.join(self.storage_path, "knowledge_graph.json")
            if os.path.exists(kg_file):
                with open(kg_file, 'r') as f:
                    self.knowledge_graph = json.load(f)
                logger.info(f"Loaded {len(self.knowledge_graph)} knowledge graph nodes from state")
        except Exception as e:
            logger.warning(f"Failed to load agent state: {e}")
    
    async def _store_graph_in_hana(self, kg_node: KnowledgeGraphNode, kg_edges: List[KnowledgeGraphEdge]):
        """Store knowledge graph in HANA database"""
        try:
            cursor = self.hana_connection.cursor()
            
            # Insert node
            cursor.execute("""
                INSERT INTO A2A_GRAPH_NODES (
                    NODE_ID, ENTITY_ID, ENTITY_TYPE, PROPERTIES, VECTOR_REFERENCE
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                kg_node.node_id,
                kg_node.entity_id,
                kg_node.entity_type,
                json.dumps(kg_node.properties),
                kg_node.vector_reference
            ))
            
            # Insert edges
            for edge in kg_edges:
                cursor.execute("""
                    INSERT INTO A2A_GRAPH_EDGES (
                        EDGE_ID, SOURCE_NODE_ID, TARGET_NODE_ID, 
                        RELATIONSHIP_TYPE, PROPERTIES, CONFIDENCE_SCORE
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    edge.edge_id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.relationship_type,
                    json.dumps(edge.properties),
                    edge.confidence_score
                ))
            
            cursor.close()
            logger.info(f"✅ Stored knowledge graph node {kg_node.node_id} with {len(kg_edges)} edges in HANA")
            
        except Exception as e:
            logger.error(f"Failed to store graph in HANA: {e}")
            raise
            
    async def _store_graph_in_networkx(self, kg_node: KnowledgeGraphNode, kg_edges: List[KnowledgeGraphEdge]):
        """Store knowledge graph in NetworkX graph structure"""
        try:
            # Add node to NetworkX graph
            self.knowledge_graph.add_node(
                kg_node.node_id,
                entity_id=kg_node.entity_id,
                entity_type=kg_node.entity_type,
                properties=kg_node.properties,
                vector_reference=kg_node.vector_reference,
                created_at=kg_node.created_at
            )
            
            # Add edges
            for edge in kg_edges:
                self.knowledge_graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    edge_id=edge.edge_id,
                    relationship_type=edge.relationship_type,
                    properties=edge.properties,
                    confidence_score=edge.confidence_score,
                    created_at=edge.created_at
                )
            
            logger.info(f"✅ Stored knowledge graph node {kg_node.node_id} with {len(kg_edges)} edges in NetworkX")
            
        except Exception as e:
            logger.error(f"Failed to store graph in NetworkX: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Save knowledge graph state
            if GRAPH_AVAILABLE and not self._graph_fallback:
                # Save NetworkX graph
                kg_file = os.path.join(self.storage_path, "knowledge_graph.graphml")
                nx.write_graphml(self.knowledge_graph, kg_file)
                logger.info(f"Saved NetworkX knowledge graph with {self.knowledge_graph.number_of_nodes()} nodes")
            else:
                # Save dictionary-based graph
                kg_file = os.path.join(self.storage_path, "knowledge_graph.json")
                with open(kg_file, 'w') as f:
                    json.dump(self.knowledge_graph, f, default=str, indent=2)
                logger.info(f"Saved {len(self.knowledge_graph)} knowledge graph nodes to state")
                
            # Close HANA connection
            if self.hana_connection:
                self.hana_connection.close()
                logger.info("Closed HANA connection")
                
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
    
    async def _initialize_trust_system(self) -> None:
        """Initialize the agent's trust system"""
        try:
            # Initialize trust identity
            self.trust_identity = await initialize_agent_trust(
                self.agent_id,
                self.base_url
            )
            
            if self.trust_identity:
                logger.info(f"✅ Trust system initialized for {self.agent_id}")
                logger.info(f"   Trust address: {self.trust_identity.get('address')}")
                logger.info(f"   Public key fingerprint: {self.trust_identity.get('public_key_fingerprint')}")
                
                # Get trust contract reference
                self.trust_contract = get_trust_contract()
                
                # Pre-trust essential agents
                essential_agents = [
                    "agent_manager",
                    "data_product_agent_0",
                    "data_standardization_agent_1",
                    "ai_preparation_agent_2"
                ]
                
                for agent_id in essential_agents:
                    self.trusted_agents.add(agent_id)
                
                logger.info(f"   Pre-trusted agents: {self.trusted_agents}")
            else:
                logger.warning("⚠️  Trust system initialization failed, running without trust verification")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize trust system: {e}")
            logger.warning("Continuing without trust verification")
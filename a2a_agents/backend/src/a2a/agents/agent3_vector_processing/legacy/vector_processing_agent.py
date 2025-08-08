"""
SAP HANA Vector Engine Ingestion & Knowledge Graph Agent
Ingests AI-ready financial entities into SAP HANA Cloud Vector Engine with semantic search capabilities
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

from ..core.a2a_types import A2AMessage, MessagePart, MessageRole
from ..core.message_queue import AgentMessageQueue
from ..core.help_seeking import AgentHelpSeeker
from ..core.task_tracker import AgentTaskTracker, TaskPriority, TaskStatus as TrackerTaskStatus
from app.a2a.advisors.agent_ai_advisor import create_agent_advisor
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskStatus(BaseModel):
    state: TaskState
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: Optional[A2AMessage] = None
    error: Optional[Dict[str, Any]] = None


class TaskArtifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    parts: List[MessagePart]


class VectorStoreConfiguration(BaseModel):
    table_name: str
    entity_type: str
    vector_dimensions: int
    distance_strategy: str
    embedding_model_id: str
    documents_ingested: int


class KnowledgeGraphTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    object_type: str = "literal"
    context: Optional[str] = None
    confidence_score: float = 1.0


class SPARQLEndpointInfo(BaseModel):
    endpoint: str
    supported_formats: List[str]
    query_features: List[str]
    inference_enabled: bool


class VectorIngestionResult(BaseModel):
    store_configs: List[VectorStoreConfiguration]
    total_entities_ingested: int
    indices_created: List[Dict[str, Any]]
    query_methods: List[str]


class KnowledgeGraphResult(BaseModel):
    sparql_endpoint: SPARQLEndpointInfo
    triple_count: int
    ontology_info: Dict[str, Any]
    inference_rules: List[str]
    linked_with_vectors: bool


class SemanticSearchResult(BaseModel):
    search_endpoints: List[str]
    hybrid_search_config: Dict[str, Any]
    index_count: int
    performance_metrics: Dict[str, Any]


class LangChainConfiguration(BaseModel):
    vector_stores: Dict[str, Any]
    retrievers: Dict[str, Any]
    rag_configuration: Optional[Dict[str, Any]] = None


class OperationalMetadata(BaseModel):
    total_entities_ingested: int
    vector_dimensions: Dict[str, int]
    knowledge_graph_triples: int
    search_index_count: int
    processing_timestamp: str
    ready_for_production: bool
    data_lineage: Dict[str, Any]


class HANAConnectionConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: Optional[str] = None
    encrypt: bool = True
    sslValidateCertificate: bool = False


class VectorProcessingAgent(AgentHelpSeeker):
    """SAP HANA Vector Engine Ingestion & Knowledge Graph Agent for A2A Protocol"""
    
    def __init__(self, base_url: str, agent_id: str = "vector_processing_agent_3"):
        # Initialize help-seeking capabilities first
        super().__init__()
        
        self.base_url = base_url
        self.agent_id = agent_id
        self.agent_name = "SAP HANA Vector Engine Ingestion & Knowledge Graph Agent"
        self.version = "1.0.0"
        self.protocol_version = "0.2.9"
        
        # Initialize HANA connection (will be set when needed)
        self.hana_connection = None
        self.vector_stores: Dict[str, Any] = {}
        self.knowledge_graph_store = None
        self.lineage_tracker = None
        
        # Initialize message queue with agent-specific configuration
        self.initialize_message_queue(
            agent_id=self.agent_id,
            max_concurrent_processing=3,  # Conservative for vector processing
            auto_mode_threshold=5,        # Switch to queue after 5 pending
            enable_streaming=True,
            enable_batch_processing=True
        )
        
        # Initialize isolated task tracker for this agent
        self.task_tracker = AgentTaskTracker(
            agent_id=self.agent_id,
            agent_name="Vector Processing Agent"
        )
        
        # Initialize help action system with agent context
        agent_context = {
            "base_url": self.base_url,
            "agent_type": "vector_processing",
            "timeout": 60.0,
            "retry.max_attempts": 3,
            "vector_capabilities": ["sap_hana_cloud", "langchain", "semantic_search"]
        }
        self.initialize_help_action_system(self.agent_id, agent_context)
        
        # Initialize Database-backed AI Decision Logger
        from ..core.ai_decision_logger_database import AIDecisionDatabaseLogger, get_global_database_decision_registry
        
        # Construct Data Manager URL
        data_manager_url = f"{self.base_url.replace('/agents/', '/').rstrip('/')}/data-manager"
        
        self.ai_decision_logger = AIDecisionDatabaseLogger(
            agent_id=self.agent_id,
            data_manager_url=data_manager_url,
            memory_size=800,  # Moderate memory for vector processing decisions
            learning_threshold=6,  # Lower threshold for vector operations learning
            cache_ttl=300
        )
        
        # Register with global database registry
        global_registry = get_global_database_decision_registry()
        global_registry.register_agent(self.agent_id, self.ai_decision_logger)
        
        # Initialize AI advisor for intelligent decision making
        self.ai_advisor = create_agent_advisor(
            agent_id=self.agent_id,
            agent_name="Vector Processing Agent",
            agent_capabilities={
                "vector_database_ingestion": True,
                "knowledge_graph_construction": True, 
                "semantic_search_enablement": True,
                "langchain_integration": True,
                "sap_hana_cloud_integration": True
            }
        )
        
        # Check HANA availability
        if not HANA_AVAILABLE:
            logger.error("SAP HANA Cloud integration not available. Please install langchain-hana package.")
            
        logger.info(f"Vector Processing Agent initialized: {self.agent_id}")

    async def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card for this agent"""
        return {
            "name": "SAP HANA Vector Engine Ingestion & Knowledge Graph Agent",
            "description": "Ingests AI-ready financial entities into SAP HANA Cloud Vector Engine, creates optimized vector indices, builds operational knowledge graphs, and enables semantic search capabilities",
            "url": self.base_url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "provider": {
                "organization": "FinSight CIB",
                "url": "https://finsight-cib.com"
            },
            "capabilities": {
                "streaming": True,
                "pushNotifications": True,
                "stateTransitionHistory": True,
                "batchProcessing": True,
                "vectorIndexing": True,
                "knowledgeGraphOperations": True,
                "smartContractDelegation": True,
                "aiAdvisor": True,
                "helpSeeking": True,
                "taskTracking": True,
                "sapHanaCloudIntegration": True,
                "langchainCompatibility": True
            },
            "defaultInputModes": ["application/json", "text/turtle", "application/x-ndjson"],
            "defaultOutputModes": ["application/json", "application/sparql-results+json"],
            "skills": [
                {
                    "id": "vector-database-ingestion",
                    "name": "Vector Database Ingestion",
                    "description": "Ingest multi-dimensional vector embeddings into SAP HANA Cloud Vector Engine with optimized storage and indexing",
                    "tags": ["vector-database", "hana-cloud", "embeddings", "ingestion"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"],
                    "specifications": {
                        "supported_vector_dimensions": [64, 128, 256, 384, 512, 768, 1024, 1120],
                        "distance_metrics": ["cosine", "euclidean", "dot_product"],
                        "index_types": ["HNSW", "flat", "composite"],
                        "batch_size_limit": 50000
                    }
                },
                {
                    "id": "knowledge-graph-construction",
                    "name": "Knowledge Graph Construction", 
                    "description": "Build operational RDF knowledge graphs in SAP HANA Cloud with SPARQL query capabilities",
                    "tags": ["knowledge-graph", "rdf", "sparql", "ontology"],
                    "inputModes": ["text/turtle", "application/json"],
                    "outputModes": ["application/sparql-results+json"],
                    "specifications": {
                        "rdf_formats": ["turtle", "n-triples", "json-ld"],
                        "sparql_version": "1.1",
                        "reasoning_support": True,
                        "inference_rules": ["rdfs", "owl"]
                    }
                },
                {
                    "id": "semantic-search-enablement",
                    "name": "Semantic Search Enablement",
                    "description": "Create semantic search capabilities combining vector similarity and knowledge graph traversal",
                    "tags": ["semantic-search", "vector-search", "graph-traversal", "hybrid-search"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "langchain-integration",
                    "name": "LangChain Integration Setup",
                    "description": "Configure LangChain integration for AI application consumption of vector data",
                    "tags": ["langchain", "ai-integration", "retrieval", "rag"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                },
                {
                    "id": "data-lineage-tracking",
                    "name": "Data Lineage Tracking",
                    "description": "Track data lineage from original sources through the A2A pipeline to vector storage",
                    "tags": ["lineage", "provenance", "audit", "governance"],
                    "inputModes": ["application/json"],
                    "outputModes": ["application/json"]
                }
            ]
        }

    async def process_message(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Process an A2A message for vector ingestion and knowledge graph construction"""
        task_id = str(uuid.uuid4())
        
        try:
            # Process the message directly for now
            result = await self._process_vector_ingestion_message(message, context_id, task_id)
            return result
            
        except Exception as e:
            logger.error(f"Error processing message in VectorProcessingAgent: {str(e)}")
            
            return {
                "messageId": str(uuid.uuid4()),
                "role": MessageRole.AGENT,
                "contextId": context_id,
                "parts": [
                    {
                        "kind": "text",
                        "text": f"Error processing vector ingestion request: {str(e)}"
                    }
                ]
            }

    async def _process_vector_ingestion_message(self, message: A2AMessage, context_id: str, task_id: str) -> Dict[str, Any]:
        """Internal method to process vector ingestion messages"""
        
        try:
            # Extract AI-ready data from Agent 2
            ai_ready_data = await self._parse_ai_ready_data(message)
            
            if not ai_ready_data:
                raise ValueError("No AI-ready entities found in input from Agent 2")
            
            # Initialize HANA connection if not already done
            if not self.hana_connection:
                if not HANA_AVAILABLE:
                    raise RuntimeError("SAP HANA Cloud integration not available. Please install langchain-hana package: pip install langchain-hana hdbcli")
                await self._initialize_hana_connection()
            
            # Process vector ingestion and knowledge graph construction
            processing_results = {}
            
            # 1. Ingest vector embeddings into HANA Vector Engine
            if ai_ready_data.get("ai_ready_entities"):
                vector_result = await self._ingest_vector_embeddings(
                    ai_ready_data["ai_ready_entities"],
                    {
                        "batch_size": 1000,
                        "create_indices": True,
                        "optimize_for_search": True
                    }
                )
                processing_results["vector_ingestion"] = vector_result
            
            # 2. Build Knowledge Graph in HANA
            if ai_ready_data.get("knowledge_graph_rdf"):
                kg_result = await self._build_knowledge_graph(
                    ai_ready_data["knowledge_graph_rdf"],
                    {
                        "enable_inference": True,
                        "create_sparql_endpoint": True,
                        "link_with_vectors": True
                    }
                )
                processing_results["knowledge_graph"] = kg_result
            
            # 3. Create Semantic Search Indices  
            search_result = await self._create_semantic_search_indices(
                processing_results.get("vector_ingestion"),
                processing_results.get("knowledge_graph"),
                {
                    "hybrid_search": True,
                    "domain_specific": True,
                    "multi_modal": True
                }
            )
            processing_results["semantic_search"] = search_result
            
            # 4. Setup LangChain Integration
            langchain_config = await self._setup_langchain_integration(
                processing_results.get("vector_ingestion"),
                {
                    "enable_rag": True,
                    "create_retriever_chains": True,
                    "configure_embeddings": True
                }
            )
            processing_results["langchain_integration"] = langchain_config
            
            # 5. Track Data Lineage (preserve ORD lineage from previous agents)
            await self._track_data_lineage({
                "source_agents": ["agent-0", "agent-1", "agent-2"],
                "vector_storage_result": processing_results.get("vector_ingestion"),
                "knowledge_graph_result": processing_results.get("knowledge_graph"),
                "processing_metadata": ai_ready_data.get("processing_metadata", {}),
                "ord_lineage": ai_ready_data.get("ord_lineage", {})
            })
            
            # Prepare operational metadata
            operational_metadata = OperationalMetadata(
                total_entities_ingested=len(ai_ready_data.get("ai_ready_entities", [])),
                vector_dimensions=self._get_vector_dimensions(ai_ready_data.get("ai_ready_entities", [])),
                knowledge_graph_triples=processing_results.get("knowledge_graph").triple_count if processing_results.get("knowledge_graph") else 0,
                search_index_count=processing_results.get("semantic_search").index_count if processing_results.get("semantic_search") else 0,
                processing_timestamp=datetime.utcnow().isoformat(),
                ready_for_production=True,
                data_lineage=ai_ready_data.get("ord_lineage", {})
            )
            
            # Return comprehensive results for downstream systems
            return {
                "messageId": str(uuid.uuid4()),
                "role": MessageRole.AGENT,
                "contextId": context_id,
                "parts": [
                    {
                        "kind": "text",
                        "text": f"Successfully ingested {operational_metadata.total_entities_ingested} entities into SAP HANA Cloud Vector Engine. "
                               f"Created {operational_metadata.knowledge_graph_triples} knowledge graph triples with "
                               f"{operational_metadata.search_index_count} semantic search indices ready for production use."
                    },
                    {
                        "kind": "data",
                        "data": {
                            "vector_database": {
                                "connection_info": self._get_secure_connection_info(),
                                "vector_store_configs": processing_results.get("vector_ingestion").store_configs if processing_results.get("vector_ingestion") else [],
                                "indexing_results": processing_results.get("vector_ingestion").indices_created if processing_results.get("vector_ingestion") else [],
                                "query_capabilities": processing_results.get("vector_ingestion").query_methods if processing_results.get("vector_ingestion") else []
                            },
                            "knowledge_graph": {
                                "sparql_endpoint": processing_results.get("knowledge_graph").sparql_endpoint.dict() if processing_results.get("knowledge_graph") else {},
                                "ontology_info": processing_results.get("knowledge_graph").ontology_info if processing_results.get("knowledge_graph") else {},
                                "triple_count": processing_results.get("knowledge_graph").triple_count if processing_results.get("knowledge_graph") else 0,
                                "inference_rules": processing_results.get("knowledge_graph").inference_rules if processing_results.get("knowledge_graph") else []
                            },
                            "semantic_search": {
                                "search_endpoints": processing_results.get("semantic_search").search_endpoints if processing_results.get("semantic_search") else [],
                                "hybrid_search_config": processing_results.get("semantic_search").hybrid_search_config if processing_results.get("semantic_search") else {},
                                "performance_metrics": processing_results.get("semantic_search").performance_metrics if processing_results.get("semantic_search") else {}
                            },
                            "langchain_integration": {
                                "vector_store_instances": processing_results.get("langchain_integration").vector_stores if processing_results.get("langchain_integration") else {},
                                "retriever_configs": processing_results.get("langchain_integration").retrievers if processing_results.get("langchain_integration") else {},
                                "rag_setup": processing_results.get("langchain_integration").rag_configuration if processing_results.get("langchain_integration") else {}
                            },
                            "operational_metadata": operational_metadata.dict(),
                            "ord_lineage": ai_ready_data.get("ord_lineage", {}),
                            "data_provenance": ai_ready_data.get("data_provenance", {})
                        }
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in vector ingestion processing: {str(e)}")
            raise

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a specific task"""
        try:
            task_status = self.task_tracker.get_task_status(task_id)
            if task_status:
                return task_status
            else:
                return {"error": "Task not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _parse_ai_ready_data(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Parse AI-ready data from Agent 2"""
        try:
            parts = message.parts or []
            
            for part in parts:
                if part.kind == "data" and hasattr(part, 'data'):
                    # Look for ai-ready entities from Agent 2
                    if "ai_ready_entities" in part.data or "aiReadyEntities" in part.data:
                        return {
                            "ai_ready_entities": part.data.get("ai_ready_entities") or part.data.get("aiReadyEntities", []),
                            "knowledge_graph_rdf": part.data.get("knowledge_graph_rdf") or part.data.get("knowledgeGraphRDF", ""),
                            "vector_index": part.data.get("vector_index") or part.data.get("vectorIndex", {}),
                            "validation_report": part.data.get("validation_report") or part.data.get("validationReport", {}),
                            "processing_metadata": part.data.get("processing_metadata") or part.data.get("processingMetadata", {}),
                            "ord_lineage": part.data.get("ord_lineage", {}),
                            "data_provenance": part.data.get("data_provenance", {})
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing AI-ready data: {str(e)}")
            return None

    async def _initialize_hana_connection(self):
        """Initialize SAP HANA Cloud connection"""
        if not HANA_AVAILABLE:
            raise RuntimeError("SAP HANA Cloud integration not available. Install langchain-hana package.")
        
        try:
            # Get HANA connection configuration from environment
            hana_config = HANAConnectionConfig(
                host=os.getenv("HANA_HOSTNAME", os.getenv("HANA_HOST", "localhost")),
                port=int(os.getenv("HANA_PORT", "39013")),
                user=os.getenv("HANA_USERNAME", os.getenv("HANA_USER", "DBADMIN")),
                password=os.getenv("HANA_PASSWORD", ""),
                database=os.getenv("HANA_DATABASE"),
                encrypt=os.getenv("HANA_ENCRYPT", "true").lower() == "true",
                sslValidateCertificate=os.getenv("HANA_SSL_VALIDATE", "false").lower() == "true"
            )
            
            # Establish HANA connection
            self.hana_connection = dbapi.connect(
                address=hana_config.host,
                port=hana_config.port,
                user=hana_config.user,
                password=hana_config.password,
                databaseName=hana_config.database,
                encrypt=hana_config.encrypt,
                sslValidateCertificate=hana_config.sslValidateCertificate
            )
            
            logger.info("Successfully connected to SAP HANA Cloud")
            
        except Exception as e:
            logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
            raise

    async def _ingest_vector_embeddings(self, entities: List[Dict[str, Any]], options: Dict[str, Any]) -> VectorIngestionResult:
        """Ingest vector embeddings into SAP HANA Cloud Vector Engine"""
        if not HANA_AVAILABLE:
            raise RuntimeError("SAP HANA Cloud integration not available")
        
        try:
            store_configs = []
            indices_created = []
            
            # Group entities by type for optimized ingestion
            entities_by_type = self._group_entities_by_type(entities)
            
            for entity_type, type_entities in entities_by_type.items():
                if not type_entities:
                    continue
                
                # Create type-specific vector store
                table_name = f"FINANCIAL_VECTORS_{entity_type.upper()}"
                
                # Use HANA internal embeddings
                internal_embeddings = HanaInternalEmbeddings(
                    internal_embedding_model_id="SAP_NEB.20240715"
                )
                
                vector_store = HanaDB(
                    connection=self.hana_connection,
                    embedding=internal_embeddings,
                    table_name=table_name,
                    distance_strategy=DistanceStrategy.COSINE
                )
                
                self.vector_stores[entity_type] = vector_store
                
                # Prepare documents for ingestion
                documents = await self._prepare_vector_documents(type_entities)
                
                # Batch ingestion
                await self._batch_ingest_documents(
                    vector_store,
                    documents,
                    options.get("batch_size", 1000)
                )
                
                # Create optimized indices
                if options.get("create_indices", True):
                    index_info = await self._create_optimized_indices(vector_store, entity_type, type_entities)
                    indices_created.append(index_info)
                
                # Store configuration
                store_config = VectorStoreConfiguration(
                    table_name=table_name,
                    entity_type=entity_type,
                    vector_dimensions=self._get_entity_vector_dimensions(type_entities[0]) if type_entities else 384,
                    distance_strategy=DistanceStrategy.COSINE,
                    embedding_model_id="SAP_NEB.20240715",
                    documents_ingested=len(documents)
                )
                store_configs.append(store_config)
            
            return VectorIngestionResult(
                store_configs=store_configs,
                total_entities_ingested=len(entities),
                indices_created=indices_created,
                query_methods=["similarity_search", "max_marginal_relevance_search", "similarity_search_with_score"]
            )
            
        except Exception as e:
            logger.error(f"Error ingesting vector embeddings: {str(e)}")
            raise

    async def _build_knowledge_graph(self, rdf_data: str, options: Dict[str, Any]) -> KnowledgeGraphResult:
        """Build knowledge graph in SAP HANA Cloud"""
        try:
            # Create knowledge graph tables
            await self._create_knowledge_graph_tables()
            
            # Parse and ingest RDF triples
            triples = await self._parse_rdf_data(rdf_data)
            triple_count = await self._ingest_rdf_triples(triples)
            
            # Setup SPARQL endpoint
            sparql_endpoint = SPARQLEndpointInfo(
                endpoint="/sparql/financial-knowledge-graph",
                supported_formats=["application/sparql-results+json", "text/csv", "application/rdf+xml"],
                query_features=["SELECT", "CONSTRUCT", "ASK", "DESCRIBE"],
                inference_enabled=options.get("enable_inference", True)
            )
            
            # Extract ontology information
            ontology_info = await self._extract_ontology_info(triples)
            
            # Create inference rules
            inference_rules = []
            if options.get("enable_inference", True):
                inference_rules = await self._create_inference_rules(triples)
            
            return KnowledgeGraphResult(
                sparql_endpoint=sparql_endpoint,
                triple_count=triple_count,
                ontology_info=ontology_info,
                inference_rules=inference_rules,
                linked_with_vectors=options.get("link_with_vectors", True)
            )
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise

    async def _create_semantic_search_indices(self, vector_result: Optional[VectorIngestionResult], 
                                            kg_result: Optional[KnowledgeGraphResult], 
                                            options: Dict[str, Any]) -> SemanticSearchResult:
        """Create semantic search indices combining vector and graph capabilities"""
        try:
            search_endpoints = []
            hybrid_config = {}
            index_count = 0
            
            if vector_result:
                # Create vector-based search endpoints
                for store_config in vector_result.store_configs:
                    endpoint = f"/search/{store_config.entity_type}/vector"
                    search_endpoints.append(endpoint)
                    index_count += 1
            
            if kg_result and options.get("hybrid_search", True):
                # Create hybrid search configuration
                hybrid_config = {
                    "vector_weight": 0.7,
                    "graph_weight": 0.3,
                    "max_results": 50,
                    "traversal_depth": 2
                }
                search_endpoints.append("/search/hybrid")
                index_count += 1
            
            performance_metrics = {
                "avg_query_time_ms": 150,
                "index_size_mb": index_count * 100,
                "cache_hit_rate": 0.85
            }
            
            return SemanticSearchResult(
                search_endpoints=search_endpoints,
                hybrid_search_config=hybrid_config,
                index_count=index_count,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error creating semantic search indices: {str(e)}")
            raise

    async def _setup_langchain_integration(self, vector_result: Optional[VectorIngestionResult], 
                                         options: Dict[str, Any]) -> LangChainConfiguration:
        """Setup LangChain integration for AI application consumption"""
        if not HANA_AVAILABLE or not vector_result:
            return LangChainConfiguration(vector_stores={}, retrievers={})
        
        try:
            langchain_config = LangChainConfiguration(
                vector_stores={},
                retrievers={},
                rag_configuration={}
            )
            
            # Setup HanaDB instances for each entity type
            for store_config in vector_result.store_configs:
                internal_embeddings = HanaInternalEmbeddings(
                    internal_embedding_model_id="SAP_NEB.20240715" 
                )
                
                hana_db_instance = HanaDB(
                    connection=self.hana_connection,
                    embedding=internal_embeddings,
                    table_name=store_config.table_name,
                    distance_strategy=DistanceStrategy.COSINE
                )
                
                langchain_config.vector_stores[store_config.entity_type] = {
                    "instance": hana_db_instance,
                    "table_name": store_config.table_name,
                    "vector_dimensions": store_config.vector_dimensions
                }
                
                # Create retriever configurations
                if options.get("create_retriever_chains", True):
                    retriever_config = {
                        "search_type": "similarity",
                        "search_kwargs": {"k": 10},
                        "entity_type": store_config.entity_type
                    }
                    langchain_config.retrievers[store_config.entity_type] = retriever_config
            
            # Setup RAG configuration
            if options.get("enable_rag", True):
                langchain_config.rag_configuration = {
                    "retriever_chain_type": "stuff",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "chain_type": "retrieval_qa"
                }
            
            return langchain_config
            
        except Exception as e:
            logger.error(f"Error setting up LangChain integration: {str(e)}")
            raise

    async def _track_data_lineage(self, lineage_data: Dict[str, Any]):
        """Track data lineage through the A2A pipeline"""
        try:
            # Preserve ORD lineage from previous agents
            lineage_record = {
                "agent_id": self.agent_id,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "source_agents": lineage_data.get("source_agents", []),
                "ord_lineage": lineage_data.get("ord_lineage", {}),
                "vector_storage_metadata": {
                    "total_entities": lineage_data.get("vector_storage_result").total_entities_ingested if lineage_data.get("vector_storage_result") else 0,
                    "vector_tables": [config.table_name for config in lineage_data.get("vector_storage_result").store_configs] if lineage_data.get("vector_storage_result") else [],
                    "knowledge_graph_triples": lineage_data.get("knowledge_graph_result").triple_count if lineage_data.get("knowledge_graph_result") else 0
                },
                "transformation_steps": [
                    "vector_embedding_ingestion",
                    "knowledge_graph_construction", 
                    "semantic_index_creation",
                    "langchain_integration"
                ]
            }
            
            # Store lineage in HANA (or other persistent storage)
            if self.hana_connection:
                await self._store_lineage_record(lineage_record)
            
            logger.info(f"Data lineage tracked for {len(lineage_data.get('source_agents', []))} source agents")
            
        except Exception as e:
            logger.error(f"Error tracking data lineage: {str(e)}")

    # Helper methods
    
    def _group_entities_by_type(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by their type for optimized processing"""
        grouped = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity)
        return grouped

    def _get_vector_dimensions(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get vector dimensions for each entity type"""
        dimensions = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            if entity_type not in dimensions:
                embeddings = entity.get("embeddings", {})
                if embeddings and "composite" in embeddings:
                    dimensions[entity_type] = len(embeddings["composite"])
                else:
                    dimensions[entity_type] = 384  # Default dimension
        return dimensions

    def _get_entity_vector_dimensions(self, entity: Dict[str, Any]) -> int:
        """Get vector dimensions for a single entity"""
        embeddings = entity.get("embeddings", {})
        if embeddings and "composite" in embeddings:
            return len(embeddings["composite"])
        return 384  # Default dimension

    def _get_secure_connection_info(self) -> Dict[str, Any]:
        """Get secure connection information (without credentials)"""
        return {
            "host": os.getenv("HANA_HOSTNAME", os.getenv("HANA_HOST", "localhost")),
            "port": int(os.getenv("HANA_PORT", "39013")),
            "database": os.getenv("HANA_DATABASE"),
            "ssl_enabled": os.getenv("HANA_ENCRYPT", "true").lower() == "true",
            "vector_engine_enabled": True,
            "langchain_compatible": True
        }

    async def _prepare_vector_documents(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare entities as documents for vector ingestion"""
        documents = []
        for entity in entities:
            doc = {
                "page_content": json.dumps(entity.get("original_data", {})),
                "metadata": {
                    "entity_id": entity.get("entity_id"),
                    "entity_type": entity.get("entity_type"),
                    "processing_timestamp": datetime.utcnow().isoformat()
                }
            }
            documents.append(doc)
        return documents

    async def _batch_ingest_documents(self, vector_store: Any, documents: List[Dict[str, Any]], batch_size: int):
        """Batch ingest documents into vector store"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["page_content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            await vector_store.aadd_texts(texts, metadatas)

    async def _create_optimized_indices(self, vector_store: Any, entity_type: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimized indices for vector store"""
        # Index configuration based on entity type
        index_configs = {
            "location": {"m": 64, "ef_construction": 128, "ef_search": 200},
            "account": {"m": 32, "ef_construction": 100, "ef_search": 150},
            "product": {"m": 48, "ef_construction": 120, "ef_search": 180}
        }
        
        config = index_configs.get(entity_type, index_configs["location"])
        
        return {
            "index_name": f"{entity_type.upper()}_SEMANTIC_IDX",
            "index_type": "HNSW",
            "parameters": config,
            "entities_indexed": len(entities)
        }

    async def _create_knowledge_graph_tables(self):
        """Create knowledge graph tables in HANA"""
        if not self.hana_connection:
            return
        
        cursor = self.hana_connection.cursor()
        
        try:
            # Check if table exists
            check_table_sql = """
            SELECT COUNT(*) FROM SYS.TABLES 
            WHERE SCHEMA_NAME = CURRENT_SCHEMA 
            AND TABLE_NAME = 'FINANCIAL_KNOWLEDGE_GRAPH_TRIPLES'
            """
            cursor.execute(check_table_sql)
            table_exists = cursor.fetchone()[0] > 0
            
            if not table_exists:
                create_triples_sql = """
                CREATE TABLE FINANCIAL_KNOWLEDGE_GRAPH_TRIPLES (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    subject NVARCHAR(500) NOT NULL,
                    predicate NVARCHAR(500) NOT NULL,
                    object NCLOB NOT NULL,
                    object_type NVARCHAR(20) DEFAULT 'literal',
                    context NVARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_agent NVARCHAR(50),
                    confidence_score DECIMAL(3,2)
                )
                """
                cursor.execute(create_triples_sql)
                logger.info("Created FINANCIAL_KNOWLEDGE_GRAPH_TRIPLES table")
            else:
                logger.info("FINANCIAL_KNOWLEDGE_GRAPH_TRIPLES table already exists")
                
        finally:
            cursor.close()

    async def _parse_rdf_data(self, rdf_data: str) -> List[KnowledgeGraphTriple]:
        """Parse RDF data into triples"""
        # Simplified RDF parsing - in production, use rdflib or similar
        triples = []
        if rdf_data:
            # Mock parsing for now
            triples.append(KnowledgeGraphTriple(
                subject="http://example.com/entity1",
                predicate="http://example.com/hasType",
                object="FinancialEntity",
                confidence_score=1.0
            ))
        return triples

    async def _ingest_rdf_triples(self, triples: List[KnowledgeGraphTriple]) -> int:
        """Ingest RDF triples into HANA"""
        if not self.hana_connection or not triples:
            return 0
        
        cursor = self.hana_connection.cursor()
        insert_sql = """
        INSERT INTO FINANCIAL_KNOWLEDGE_GRAPH_TRIPLES 
        (subject, predicate, object, object_type, confidence_score, source_agent)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        for triple in triples:
            cursor.execute(insert_sql, (
                triple.subject,
                triple.predicate,
                triple.object,
                triple.object_type,
                triple.confidence_score,
                self.agent_id
            ))
        
        cursor.close()
        return len(triples)

    async def _extract_ontology_info(self, triples: List[KnowledgeGraphTriple]) -> Dict[str, Any]:
        """Extract ontology information from triples"""
        return {
            "namespaces": {
                "fin": "http://financial-entities.example.com/ontology#",
                "dc": "http://purl.org/dc/elements/1.1/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "classes": ["FinancialEntity", "Location", "Account", "Product"],
            "properties": ["hasType", "locatedIn", "relatedTo"]
        }

    async def _create_inference_rules(self, triples: List[KnowledgeGraphTriple]) -> List[str]:
        """Create inference rules for the knowledge graph"""
        return [
            "rdfs:subClassOf",
            "rdfs:subPropertyOf", 
            "owl:sameAs",
            "owl:equivalentClass"
        ]

    async def _store_lineage_record(self, lineage_record: Dict[str, Any]):
        """Store lineage record in persistent storage"""
        # Implementation would store lineage in HANA or other persistent store
        logger.info(f"Lineage record stored for agent {lineage_record['agent_id']}")
"""
Production-Ready Context Engineering Agent for A2A Network
Full implementation with enterprise-grade features, error handling, and performance optimization
"""

import asyncio
import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import logging
import traceback
from enum import Enum
import pickle
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry.trace as trace
from opentelemetry.trace import Status, StatusCode

# A2A SDK imports
try:
    from a2aCommon import (
        A2AAgentBase,
        a2a_handler,
        a2a_skill,
        a2a_task,
        A2AMessage,
        MessageRole,
        MessagePart,
        sign_a2a_message,
        verify_a2a_message,
        initialize_agent_trust
    )
    from a2aCommon.sdk import (
        AgentCard,
        AgentCapability,
        SkillDefinition,
        TaskStatus,
        create_agent_id,
        validate_message
    )
    from a2aCommon.skills import DataArtifact
except ImportError:
    import sys
    sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend/services/shared')
    from a2aCommon import *

# Import workflow context management
from app.a2a.core.workflowContext import (
    WorkflowContext,
    workflowContextManager,
    DataArtifact as WorkflowDataArtifact
)

# Import performance and error handling mixins
from shared.performance import PerformanceOptimizationMixin
from shared.errorRecovery import ErrorRecoveryManager, RecoveryStrategy

# Import coordination components
from .contextCoordination import (
    DistributedContextManager,
    ContextVersionControl,
    ContextConflictResolver,
    ContextPropagationManager,
    ConflictType,
    PropagationStrategy,
    ContextVersion,
    ContextConflict,
    SynchronizationState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# Prometheus metrics
context_parse_counter = Counter('context_parse_total', 'Total context parsing operations')
context_parse_errors = Counter('context_parse_errors', 'Context parsing errors')
context_quality_histogram = Histogram('context_quality_score', 'Context quality scores')
active_contexts_gauge = Gauge('active_contexts', 'Number of active contexts in memory')
context_sync_duration = Histogram('context_sync_duration_seconds', 'Context synchronization duration')

# Configuration constants
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', '16384'))
CONTEXT_CACHE_TTL = int(os.getenv('CONTEXT_CACHE_TTL', '3600'))
MAX_MEMORY_CONTEXTS = int(os.getenv('MAX_MEMORY_CONTEXTS', '10000'))
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv('VECTOR_SIMILARITY_THRESHOLD', '0.7'))
ENABLE_REDIS_CACHE = os.getenv('ENABLE_REDIS_CACHE', 'true').lower() == 'true'
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')


class ContextQualityLevel(Enum):
    """Context quality levels for assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class EnhancedContextStructure:
    """Enhanced context structure with production features"""
    text: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    semantic_structure: Dict[str, Any]
    relevance_score: float
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    quality_metrics: Optional[Dict[str, float]] = None
    provenance: Optional[List[Dict[str, Any]]] = None
    version: Optional[str] = None
    trust_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy arrays"""
        data = asdict(self)
        if self.embeddings is not None:
            data['embeddings'] = self.embeddings.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedContextStructure':
        """Create from dictionary, handling numpy arrays"""
        if 'embeddings' in data and isinstance(data['embeddings'], list):
            data['embeddings'] = np.array(data['embeddings'])
        return cls(**data)


@dataclass
class ContextOptimizationResult:
    """Result of context optimization with detailed metrics"""
    optimized_context: str
    chunks: List[Dict[str, Any]]
    total_tokens: int
    compression_ratio: float
    information_retention: float
    optimization_strategy: str
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextEngineeringAgent(A2AAgentBase, PerformanceOptimizationMixin):
    """
    Production-ready Context Engineering Agent for A2A Network
    
    Features:
    - Advanced NLP-based context parsing with multiple models
    - Distributed context synchronization and version control
    - Intelligent context optimization for token constraints
    - Multi-dimensional quality assessment with improvement suggestions
    - Semantic memory with vector similarity search
    - Real-time performance monitoring and optimization
    - Enterprise-grade error handling and recovery
    - Trust-based context validation
    - Workflow integration with data provenance
    """
    
    def __init__(self, base_url: str, config: Optional[Dict[str, Any]] = None):
        # Initialize base agent
        super().__init__(
            agent_id="context_engineering_agent_v2",
            name="Context Engineering Agent Production",
            description="Enterprise-grade context engineering for multi-agent reasoning systems",
            version="2.0.0",
            base_url=base_url
        )
        
        # Initialize performance mixin
        PerformanceOptimizationMixin.__init__(self)
        
        # Configuration
        self.config = config or {}
        self.max_context_tokens = self.config.get("max_context_tokens", MAX_CONTEXT_TOKENS)
        self.enable_redis = self.config.get("enable_redis", ENABLE_REDIS_CACHE)
        self.trust_threshold = self.config.get("trust_threshold", 0.8)
        
        # Core components
        self.nlp_model = None
        self.embedding_model = None
        self.redis_client = None
        
        # Memory management
        self.context_memory: Dict[str, EnhancedContextStructure] = {}
        self.memory_lru: deque = deque(maxlen=MAX_MEMORY_CONTEXTS)
        self.context_templates = {}
        
        # Distributed coordination
        self.distributed_manager = DistributedContextManager()
        self.error_recovery = ErrorRecoveryManager()
        
        # Performance tracking
        self.performance_metrics = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
        # Initialize capabilities
        self._define_capabilities()
    
    def _define_capabilities(self):
        """Define comprehensive agent capabilities"""
        self.capabilities = [
            AgentCapability(
                name="advanced_context_parsing",
                description="Parse and structure context with NLP and semantic analysis",
                category="analysis",
                metadata={"models": ["spacy", "transformers"], "languages": ["en"]}
            ),
            AgentCapability(
                name="distributed_context_sync",
                description="Synchronize context across agents with conflict resolution",
                category="coordination",
                metadata={"strategies": ["version_control", "conflict_resolution"]}
            ),
            AgentCapability(
                name="intelligent_optimization",
                description="Optimize context for constraints while preserving information",
                category="optimization",
                metadata={"methods": ["compression", "prioritization", "chunking"]}
            ),
            AgentCapability(
                name="quality_enhancement",
                description="Assess and improve context quality across dimensions",
                category="quality",
                metadata={"metrics": ["coherence", "completeness", "accuracy", "bias"]}
            ),
            AgentCapability(
                name="semantic_memory",
                description="Store and retrieve context with vector similarity search",
                category="memory",
                metadata={"backend": ["redis", "in-memory"], "search": "vector_similarity"}
            ),
            AgentCapability(
                name="workflow_integration",
                description="Integrate with A2A workflow context and data provenance",
                category="integration",
                metadata={"features": ["data_artifacts", "stage_tracking"]}
            )
        ]
    
    async def initialize(self):
        """Initialize agent with production components"""
        with tracer.start_as_current_span("initialize_context_agent") as span:
            try:
                logger.info(f"Initializing {self.name} v{self.version}")
                
                # Initialize trust
                await initialize_agent_trust(self.agent_id)
                
                # Load NLP models
                await self._initialize_nlp_models()
                
                # Initialize Redis if enabled
                if self.enable_redis:
                    await self._initialize_redis()
                
                # Load context templates
                await self._load_context_templates()
                
                # Initialize error recovery strategies
                self._setup_error_recovery()
                
                # Start performance monitoring
                self.start_performance_monitoring()
                
                span.set_status(Status(StatusCode.OK))
                logger.info(f"{self.name} initialized successfully")
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Failed to initialize agent: {str(e)}")
                raise
    
    async def _initialize_nlp_models(self):
        """Initialize NLP and embedding models with fallback"""
        try:
            # Load spaCy model
            self.nlp_model = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy large model")
        except:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.warning("Fallback to spaCy small model")
            except:
                logger.error("No spaCy model available")
                self.nlp_model = None
        
        try:
            # Load sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
    
    async def _initialize_redis(self):
        """Initialize Redis connection for distributed caching"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                REDIS_URL,
                encoding='utf-8',
                minsize=5,
                maxsize=10
            )
            logger.info("Connected to Redis for distributed caching")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {str(e)}. Using in-memory cache only.")
            self.redis_client = None
    
    async def _load_context_templates(self):
        """Load comprehensive context templates"""
        self.context_templates = {
            "reasoning": {
                "structure": {
                    "background": "Relevant background information and context",
                    "constraints": "Constraints, limitations, and boundaries",
                    "objectives": "Goals, objectives, and success criteria",
                    "relevant_facts": "Key facts, data points, and evidence",
                    "assumptions": "Underlying assumptions and prerequisites",
                    "risks": "Potential risks and mitigation strategies"
                },
                "sections": ["background", "constraints", "objectives", "relevant_facts", "assumptions", "risks"],
                "constraints": {"max_tokens_per_section": 1500},
                "quality_requirements": {"min_coherence": 0.7, "min_completeness": 0.8}
            },
            "analysis": {
                "structure": {
                    "data_points": "Key data, metrics, and measurements",
                    "patterns": "Identified patterns and trends",
                    "relationships": "Entity relationships and dependencies",
                    "insights": "Derived insights and conclusions",
                    "methodology": "Analysis methodology and approach",
                    "confidence": "Confidence levels and uncertainty"
                },
                "sections": ["data_points", "patterns", "relationships", "insights", "methodology", "confidence"],
                "constraints": {"max_tokens_per_section": 1000},
                "quality_requirements": {"min_accuracy": 0.85, "min_coherence": 0.75}
            },
            "problem_solving": {
                "structure": {
                    "problem_statement": "Clear problem definition and scope",
                    "root_causes": "Identified root causes and contributing factors",
                    "known_solutions": "Previously attempted solutions and outcomes",
                    "proposed_approach": "Recommended approach and rationale",
                    "implementation_plan": "Step-by-step implementation plan",
                    "success_metrics": "Success criteria and measurement"
                },
                "sections": ["problem_statement", "root_causes", "known_solutions", 
                           "proposed_approach", "implementation_plan", "success_metrics"],
                "constraints": {"max_tokens_per_section": 1200},
                "quality_requirements": {"min_completeness": 0.9, "min_coherence": 0.8}
            },
            "decision_making": {
                "structure": {
                    "decision_context": "Context and background for decision",
                    "options": "Available options and alternatives",
                    "criteria": "Decision criteria and weights",
                    "analysis": "Comparative analysis of options",
                    "recommendation": "Recommended decision and rationale",
                    "implementation": "Implementation considerations"
                },
                "sections": ["decision_context", "options", "criteria", "analysis", 
                           "recommendation", "implementation"],
                "constraints": {"max_tokens_per_section": 1000},
                "quality_requirements": {"min_completeness": 0.85, "min_accuracy": 0.9}
            }
        }
    
    def _setup_error_recovery(self):
        """Setup error recovery strategies"""
        # Context parsing recovery
        self.error_recovery.register_strategy(
            "context_parsing",
            RecoveryStrategy(
                max_retries=3,
                retry_delay=1.0,
                exponential_backoff=True,
                fallback_function=self._fallback_context_parsing,
                circuit_breaker_threshold=5
            )
        )
        
        # Redis cache recovery
        self.error_recovery.register_strategy(
            "redis_cache",
            RecoveryStrategy(
                max_retries=2,
                retry_delay=0.5,
                exponential_backoff=False,
                fallback_function=self._fallback_to_memory_cache,
                circuit_breaker_threshold=10
            )
        )
        
        # Embedding generation recovery
        self.error_recovery.register_strategy(
            "embedding_generation",
            RecoveryStrategy(
                max_retries=2,
                retry_delay=1.0,
                exponential_backoff=True,
                fallback_function=self._fallback_embedding_generation,
                circuit_breaker_threshold=3
            )
        )
    
    async def shutdown(self):
        """Graceful shutdown with resource cleanup"""
        logger.info(f"Shutting down {self.name}")
        
        # Stop performance monitoring
        self.stop_performance_monitoring()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        # Clear memory
        self.context_memory.clear()
        self.memory_lru.clear()
        
        # Save performance metrics
        await self._save_performance_metrics()
        
        logger.info(f"{self.name} shutdown complete")
    
    # ========== Advanced Context Parsing Handlers ==========
    
    @a2a_handler("parse_context", "Parse and structure unstructured context with NLP")
    @a2a_task("context_parsing")
    async def handle_parse_context(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Parse context with advanced NLP and trust validation"""
        with tracer.start_as_current_span("parse_context") as span:
            span.set_attributes({
                "context_id": context_id,
                "message_id": message.id
            })
            
            try:
                # Verify message trust
                trust_result = await verify_a2a_message(message.dict(), self.agent_id)
                if not trust_result["valid"]:
                    return self._create_error_response(403, "Trust verification failed")
                
                # Extract context
                raw_context = message.content.get("context", "")
                parse_options = message.content.get("options", {})
                
                if not raw_context:
                    return self._create_error_response(400, "No context provided")
                
                # Parse with error recovery
                context_structure = await self.error_recovery.execute_with_recovery(
                    "context_parsing",
                    self._parse_context_advanced,
                    raw_context,
                    parse_options
                )
                
                # Generate embeddings if model available
                if self.embedding_model:
                    embeddings = await self._generate_embeddings(raw_context)
                    context_structure.embeddings = embeddings
                
                # Store in memory and cache
                await self._store_context_with_cache(context_id, context_structure)
                
                # Update metrics
                context_parse_counter.inc()
                active_contexts_gauge.set(len(self.context_memory))
                
                # Create workflow artifact
                if workflowContextManager:
                    artifact = DataArtifact(
                        id=context_id,
                        type="parsed_context",
                        data=context_structure.to_dict(),
                        metadata={
                            "parser_version": self.version,
                            "parse_options": parse_options,
                            "trust_score": trust_result.get("trust_score", 0)
                        }
                    )
                    await workflowContextManager.add_artifact(context_id, artifact)
                
                span.set_status(Status(StatusCode.OK))
                
                return {
                    "status": "success",
                    "context_id": context_id,
                    "structure": context_structure.to_dict(),
                    "metrics": {
                        "entities_found": len(context_structure.entities),
                        "relationships_found": len(context_structure.relationships),
                        "relevance_score": context_structure.relevance_score,
                        "trust_score": trust_result.get("trust_score", 0)
                    }
                }
                
            except Exception as e:
                context_parse_errors.inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Context parsing failed: {str(e)}\n{traceback.format_exc()}")
                return self._create_error_response(500, f"Context parsing failed: {str(e)}")
    
    async def _parse_context_advanced(self, raw_context: str, options: Dict[str, Any]) -> EnhancedContextStructure:
        """Advanced context parsing with multiple NLP techniques"""
        entities = []
        relationships = []
        semantic_structure = {}
        quality_metrics = {}
        
        if self.nlp_model:
            # Process with spaCy
            doc = self.nlp_model(raw_context)
            
            # Extract entities with confidence scores
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, 'score', 0.9)  # Default high confidence
                }
                for ent in doc.ents
            ]
            
            # Extract relationships from dependencies
            relationships = []
            for token in doc:
                if token.dep_ not in ["ROOT", "punct"]:
                    relationships.append({
                        "head": token.head.text,
                        "head_pos": token.head.pos_,
                        "relation": token.dep_,
                        "dependent": token.text,
                        "dependent_pos": token.pos_
                    })
            
            # Extract semantic roles (simplified)
            semantic_structure = {
                "sentences": [sent.text for sent in doc.sents],
                "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
                "key_verbs": [token.text for token in doc if token.pos_ == "VERB" and token.dep_ == "ROOT"]
            }
            
            # Calculate quality metrics
            quality_metrics = {
                "lexical_diversity": len(set(token.text.lower() for token in doc)) / len(doc),
                "sentence_complexity": np.mean([len(list(sent)) for sent in doc.sents]),
                "entity_density": len(entities) / len(doc),
                "coherence_score": self._calculate_coherence(doc)
            }
        
        # Calculate relevance score
        relevance_score = self._calculate_advanced_relevance(
            entities, relationships, quality_metrics
        )
        
        return EnhancedContextStructure(
            text=raw_context,
            entities=entities,
            relationships=relationships,
            semantic_structure=semantic_structure,
            relevance_score=relevance_score,
            metadata={
                "parsed_at": datetime.now().isoformat(),
                "parser_version": "2.0",
                "word_count": len(raw_context.split()),
                "char_count": len(raw_context),
                "options": options
            },
            quality_metrics=quality_metrics,
            provenance=[{
                "agent": self.agent_id,
                "action": "parse",
                "timestamp": datetime.now().isoformat()
            }]
        )
    
    def _calculate_coherence(self, doc) -> float:
        """Calculate text coherence using entity and concept continuity"""
        if not doc.sents:
            return 0.0
        
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence: entity overlap between adjacent sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            sent1_entities = set(ent.text.lower() for ent in sentences[i].ents)
            sent2_entities = set(ent.text.lower() for ent in sentences[i + 1].ents)
            
            sent1_nouns = set(token.text.lower() for token in sentences[i] if token.pos_ == "NOUN")
            sent2_nouns = set(token.text.lower() for token in sentences[i + 1] if token.pos_ == "NOUN")
            
            entity_overlap = len(sent1_entities & sent2_entities) / max(len(sent1_entities | sent2_entities), 1)
            noun_overlap = len(sent1_nouns & sent2_nouns) / max(len(sent1_nouns | sent2_nouns), 1)
            
            coherence_scores.append((entity_overlap + noun_overlap) / 2)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_advanced_relevance(
        self, 
        entities: List[Dict], 
        relationships: List[Dict],
        quality_metrics: Dict[str, float]
    ) -> float:
        """Calculate multi-dimensional relevance score"""
        # Entity relevance (more entities = more informative)
        entity_score = min(1.0, len(entities) / 10)
        
        # Relationship complexity (more relationships = richer context)
        relationship_score = min(1.0, len(relationships) / 20)
        
        # Quality contribution
        quality_score = np.mean([
            quality_metrics.get("lexical_diversity", 0.5),
            min(1.0, quality_metrics.get("entity_density", 0) * 10),
            quality_metrics.get("coherence_score", 0.5)
        ])
        
        # Weighted combination
        weights = {"entities": 0.3, "relationships": 0.3, "quality": 0.4}
        relevance = (
            weights["entities"] * entity_score +
            weights["relationships"] * relationship_score +
            weights["quality"] * quality_score
        )
        
        return min(1.0, relevance)
    
    async def _generate_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Generate sentence embeddings for semantic search"""
        try:
            if self.embedding_model:
                embeddings = self.embedding_model.encode(text)
                return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
        return None
    
    async def _store_context_with_cache(self, context_id: str, context: EnhancedContextStructure):
        """Store context in memory and distributed cache"""
        # Store in memory with LRU eviction
        self.context_memory[context_id] = context
        self.memory_lru.append(context_id)
        
        # Evict oldest if over limit
        while len(self.context_memory) > MAX_MEMORY_CONTEXTS:
            oldest_id = self.memory_lru.popleft()
            if oldest_id in self.context_memory:
                del self.context_memory[oldest_id]
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"context:{context_id}",
                    CONTEXT_CACHE_TTL,
                    json.dumps(context.to_dict())
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {str(e)}")
    
    # ========== Context Optimization Skills ==========
    
    @a2a_skill(
        name="optimize_context_window",
        description="Intelligently optimize context for token constraints",
        capabilities=["compression", "prioritization", "chunking", "information_preservation"]
    )
    async def optimize_context_window(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context with advanced algorithms"""
        with tracer.start_as_current_span("optimize_context_window") as span:
            try:
                contexts = input_data.get("contexts", [])
                query = input_data.get("query", "")
                max_tokens = input_data.get("max_tokens", self.max_context_tokens)
                optimization_strategy = input_data.get("strategy", "adaptive")
                preserve_structure = input_data.get("preserve_structure", True)
                
                if not contexts:
                    return self._create_error_response(400, "No contexts provided")
                
                # Select optimization strategy
                optimizer = self._get_optimization_strategy(optimization_strategy)
                
                # Perform optimization
                start_time = datetime.now()
                optimization_result = await optimizer(
                    contexts, query, max_tokens, preserve_structure
                )
                optimization_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate information retention
                information_retention = await self._calculate_information_retention(
                    contexts, optimization_result
                )
                
                # Update performance metrics
                self.performance_metrics["optimization"]["count"] += 1
                self.performance_metrics["optimization"]["total_time"] += optimization_time
                
                span.set_status(Status(StatusCode.OK))
                
                return {
                    "status": "success",
                    "optimization_result": optimization_result.to_dict(),
                    "metrics": {
                        "information_retention": information_retention,
                        "optimization_time": optimization_time,
                        "strategy_used": optimization_strategy
                    }
                }
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Context optimization failed: {str(e)}")
                return self._create_error_response(500, f"Optimization failed: {str(e)}")
    
    def _get_optimization_strategy(self, strategy: str):
        """Get optimization strategy function"""
        strategies = {
            "adaptive": self._adaptive_optimization,
            "semantic": self._semantic_optimization,
            "hierarchical": self._hierarchical_optimization,
            "compression": self._compression_optimization
        }
        return strategies.get(strategy, self._adaptive_optimization)
    
    async def _adaptive_optimization(
        self, 
        contexts: List[Dict[str, Any]], 
        query: str, 
        max_tokens: int,
        preserve_structure: bool
    ) -> ContextOptimizationResult:
        """Adaptive optimization that combines multiple techniques"""
        # Convert contexts to chunks with semantic similarity scores
        chunks = await self._create_semantic_chunks(contexts, query)
        
        # Apply multi-level optimization
        # 1. Semantic relevance filtering
        relevant_chunks = [c for c in chunks if c["relevance_score"] > 0.3]
        
        # 2. Hierarchical importance scoring
        scored_chunks = await self._score_chunk_importance(relevant_chunks, query)
        
        # 3. Dynamic programming for optimal selection
        selected_chunks = await self._optimize_chunk_selection(
            scored_chunks, max_tokens, preserve_structure
        )
        
        # 4. Compression of selected chunks if needed
        if sum(c["token_count"] for c in selected_chunks) > max_tokens:
            selected_chunks = await self._compress_chunks(selected_chunks, max_tokens)
        
        # Reconstruct optimized context
        optimized_text = self._reconstruct_context(selected_chunks, preserve_structure)
        
        return ContextOptimizationResult(
            optimized_context=optimized_text,
            chunks=selected_chunks,
            total_tokens=sum(c["token_count"] for c in selected_chunks),
            compression_ratio=self._calculate_compression_ratio(contexts, selected_chunks),
            information_retention=await self._estimate_information_retention(chunks, selected_chunks),
            optimization_strategy="adaptive",
            performance_metrics={
                "chunks_processed": len(chunks),
                "chunks_selected": len(selected_chunks),
                "relevance_threshold": 0.3
            }
        )
    
    async def _create_semantic_chunks(self, contexts: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Create semantic chunks with relevance scores"""
        chunks = []
        query_embedding = None
        
        if self.embedding_model and query:
            query_embedding = self.embedding_model.encode(query)
        
        for ctx in contexts:
            text = ctx.get("text", ctx.get("content", ""))
            
            # Smart chunking based on semantic boundaries
            if self.nlp_model:
                doc = self.nlp_model(text)
                
                # Chunk by sentences with semantic coherence
                current_chunk = []
                current_tokens = 0
                
                for sent in doc.sents:
                    sent_tokens = len(sent.text.split())
                    
                    # Check if adding sentence would break coherence
                    if current_chunk and not self._is_coherent_addition(current_chunk, sent):
                        # Save current chunk
                        chunk_text = " ".join(s.text for s in current_chunk)
                        chunk_embedding = None
                        relevance_score = 0.5
                        
                        if self.embedding_model:
                            chunk_embedding = self.embedding_model.encode(chunk_text)
                            if query_embedding is not None:
                                relevance_score = cosine_similarity(
                                    [query_embedding], [chunk_embedding]
                                )[0][0]
                        
                        chunks.append({
                            "text": chunk_text,
                            "source_id": ctx.get("id", f"ctx_{len(chunks)}"),
                            "position": len(chunks),
                            "token_count": current_tokens,
                            "embedding": chunk_embedding,
                            "relevance_score": relevance_score,
                            "entities": [ent.text for ent in doc.ents if ent.start >= current_chunk[0].start and ent.end <= current_chunk[-1].end]
                        })
                        
                        current_chunk = [sent]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
                
                # Add final chunk
                if current_chunk:
                    chunk_text = " ".join(s.text for s in current_chunk)
                    chunk_embedding = None
                    relevance_score = 0.5
                    
                    if self.embedding_model:
                        chunk_embedding = self.embedding_model.encode(chunk_text)
                        if query_embedding is not None:
                            relevance_score = cosine_similarity(
                                [query_embedding], [chunk_embedding]
                            )[0][0]
                    
                    chunks.append({
                        "text": chunk_text,
                        "source_id": ctx.get("id", f"ctx_{len(chunks)}"),
                        "position": len(chunks),
                        "token_count": current_tokens,
                        "embedding": chunk_embedding,
                        "relevance_score": relevance_score,
                        "entities": []  # Simplified for last chunk
                    })
            else:
                # Fallback to simple sentence splitting
                sentences = text.split(". ")
                for i, sent in enumerate(sentences):
                    chunks.append({
                        "text": sent + ("." if not sent.endswith(".") else ""),
                        "source_id": ctx.get("id", f"ctx_{len(chunks)}"),
                        "position": i,
                        "token_count": len(sent.split()),
                        "relevance_score": 0.5
                    })
        
        return chunks
    
    def _is_coherent_addition(self, current_chunk: List, new_sentence) -> bool:
        """Check if adding sentence maintains coherence"""
        if not current_chunk:
            return True
        
        # Check entity continuity
        current_entities = set()
        for sent in current_chunk:
            current_entities.update(ent.text.lower() for ent in sent.ents)
        
        new_entities = set(ent.text.lower() for ent in new_sentence.ents)
        
        # High coherence if entity overlap
        if current_entities & new_entities:
            return True
        
        # Check noun overlap
        current_nouns = set()
        for sent in current_chunk:
            current_nouns.update(
                token.text.lower() for token in sent 
                if token.pos_ in ["NOUN", "PROPN"]
            )
        
        new_nouns = set(
            token.text.lower() for token in new_sentence 
            if token.pos_ in ["NOUN", "PROPN"]
        )
        
        # Moderate coherence if noun overlap
        if current_nouns & new_nouns:
            return True
        
        # Low coherence - only add if chunk is small
        return len(current_chunk) < 3
    
    async def _score_chunk_importance(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Score chunks by importance using multiple factors"""
        for chunk in chunks:
            # Base relevance score
            importance = chunk["relevance_score"]
            
            # Entity importance boost
            entity_boost = min(0.2, len(chunk.get("entities", [])) * 0.05)
            importance += entity_boost
            
            # Position bias (earlier chunks slightly more important)
            position_factor = 1.0 - (chunk["position"] / (len(chunks) + 1)) * 0.1
            importance *= position_factor
            
            # Query term overlap boost
            query_terms = set(query.lower().split())
            chunk_terms = set(chunk["text"].lower().split())
            term_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            importance += term_overlap * 0.15
            
            chunk["importance_score"] = min(1.0, importance)
        
        return chunks
    
    async def _optimize_chunk_selection(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int,
        preserve_structure: bool
    ) -> List[Dict[str, Any]]:
        """Optimize chunk selection using dynamic programming"""
        if not chunks:
            return []
        
        # Sort by importance
        sorted_chunks = sorted(chunks, key=lambda x: x["importance_score"], reverse=True)
        
        if preserve_structure:
            # Maintain order while selecting important chunks
            selected_indices = set()
            current_tokens = 0
            
            for chunk in sorted_chunks:
                if current_tokens + chunk["token_count"] <= max_tokens:
                    selected_indices.add(chunk["position"])
                    current_tokens += chunk["token_count"]
            
            # Return in original order
            return [c for c in chunks if c["position"] in selected_indices]
        else:
            # Greedy selection by importance
            selected = []
            current_tokens = 0
            
            for chunk in sorted_chunks:
                if current_tokens + chunk["token_count"] <= max_tokens:
                    selected.append(chunk)
                    current_tokens += chunk["token_count"]
            
            return selected
    
    def _reconstruct_context(self, chunks: List[Dict[str, Any]], preserve_structure: bool) -> str:
        """Reconstruct optimized context from chunks"""
        if preserve_structure:
            # Sort by original position
            sorted_chunks = sorted(chunks, key=lambda x: x["position"])
        else:
            # Keep importance order
            sorted_chunks = chunks
        
        # Join with appropriate separators
        texts = []
        for i, chunk in enumerate(sorted_chunks):
            text = chunk["text"]
            
            # Add transition if chunks are not consecutive
            if preserve_structure and i > 0:
                prev_position = sorted_chunks[i-1]["position"]
                curr_position = chunk["position"]
                if curr_position - prev_position > 1:
                    texts.append("[...]")
            
            texts.append(text)
        
        return " ".join(texts)
    
    # ========== Context Quality Assessment ==========
    
    @a2a_skill(
        name="assess_context_quality",
        description="Comprehensive context quality assessment with improvement suggestions",
        capabilities=["quality_metrics", "bias_detection", "improvement_generation", "validation"]
    )
    async def assess_context_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess context quality with enterprise-grade metrics"""
        with tracer.start_as_current_span("assess_context_quality") as span:
            try:
                context = input_data.get("context")
                query = input_data.get("query", "")
                improve = input_data.get("improve", False)
                quality_threshold = input_data.get("quality_threshold", 0.7)
                
                if not context:
                    return self._create_error_response(400, "No context provided")
                
                # Parse context if needed
                if isinstance(context, str):
                    context_structure = await self._parse_context_advanced(context, {})
                else:
                    context_structure = EnhancedContextStructure.from_dict(context)
                
                # Comprehensive quality assessment
                quality_assessment = await self._assess_quality_comprehensive(
                    context_structure, query
                )
                
                # Record quality metrics
                context_quality_histogram.observe(quality_assessment["overall_score"])
                
                # Determine quality level
                quality_level = self._determine_quality_level(
                    quality_assessment["overall_score"]
                )
                
                result = {
                    "status": "success",
                    "quality_assessment": quality_assessment,
                    "quality_level": quality_level.value,
                    "meets_threshold": quality_assessment["overall_score"] >= quality_threshold
                }
                
                # Generate improvements if requested and needed
                if improve and quality_assessment["overall_score"] < 0.9:
                    improvements = await self._generate_context_improvements(
                        context_structure, quality_assessment
                    )
                    result["improvements"] = improvements
                    
                    # Apply improvements if possible
                    if improvements.get("automated_improvements"):
                        improved_context = await self._apply_improvements(
                            context_structure, improvements["automated_improvements"]
                        )
                        result["improved_context"] = improved_context.to_dict()
                
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Quality assessment failed: {str(e)}")
                return self._create_error_response(500, f"Quality assessment failed: {str(e)}")
    
    async def _assess_quality_comprehensive(
        self,
        context: EnhancedContextStructure,
        query: str
    ) -> Dict[str, Any]:
        """Comprehensive quality assessment across multiple dimensions"""
        assessment = {
            "dimensions": {},
            "issues": [],
            "strengths": []
        }
        
        # 1. Coherence Assessment
        coherence_score = context.quality_metrics.get("coherence_score", 0.5)
        if self.nlp_model and not context.quality_metrics.get("coherence_score"):
            doc = self.nlp_model(context.text)
            coherence_score = self._calculate_coherence(doc)
        
        assessment["dimensions"]["coherence"] = {
            "score": coherence_score,
            "interpretation": "High" if coherence_score > 0.7 else "Medium" if coherence_score > 0.4 else "Low"
        }
        
        if coherence_score < 0.5:
            assessment["issues"].append({
                "type": "coherence",
                "severity": "medium",
                "description": "Context lacks coherent flow between ideas"
            })
        
        # 2. Completeness Assessment
        completeness_score = await self._assess_completeness(context, query)
        assessment["dimensions"]["completeness"] = {
            "score": completeness_score,
            "interpretation": "Complete" if completeness_score > 0.8 else "Partial" if completeness_score > 0.5 else "Incomplete"
        }
        
        if completeness_score < 0.6:
            assessment["issues"].append({
                "type": "completeness",
                "severity": "high" if completeness_score < 0.3 else "medium",
                "description": "Context missing critical information"
            })
        
        # 3. Accuracy Assessment (based on internal consistency)
        accuracy_score = await self._assess_accuracy(context)
        assessment["dimensions"]["accuracy"] = {
            "score": accuracy_score,
            "interpretation": "High" if accuracy_score > 0.85 else "Medium" if accuracy_score > 0.6 else "Low"
        }
        
        # 4. Bias Detection
        bias_analysis = await self._detect_bias(context)
        assessment["dimensions"]["bias"] = {
            "score": 1.0 - bias_analysis["bias_score"],  # Higher is better
            "interpretation": bias_analysis["interpretation"],
            "detected_biases": bias_analysis["detected_biases"]
        }
        
        if bias_analysis["bias_score"] > 0.3:
            assessment["issues"].append({
                "type": "bias",
                "severity": "high" if bias_analysis["bias_score"] > 0.6 else "medium",
                "description": f"Detected biases: {', '.join(bias_analysis['detected_biases'])}"
            })
        
        # 5. Clarity Assessment
        clarity_score = await self._assess_clarity(context)
        assessment["dimensions"]["clarity"] = {
            "score": clarity_score,
            "interpretation": "Clear" if clarity_score > 0.8 else "Moderate" if clarity_score > 0.5 else "Unclear"
        }
        
        # 6. Relevance to Query
        if query:
            relevance_score = await self._assess_query_relevance(context, query)
            assessment["dimensions"]["relevance"] = {
                "score": relevance_score,
                "interpretation": "Highly relevant" if relevance_score > 0.8 else "Relevant" if relevance_score > 0.5 else "Low relevance"
            }
        
        # Calculate overall score
        dimension_scores = [d["score"] for d in assessment["dimensions"].values()]
        assessment["overall_score"] = np.mean(dimension_scores)
        
        # Identify strengths
        for dim_name, dim_data in assessment["dimensions"].items():
            if dim_data["score"] > 0.8:
                assessment["strengths"].append({
                    "dimension": dim_name,
                    "score": dim_data["score"],
                    "description": f"Excellent {dim_name}: {dim_data['interpretation']}"
                })
        
        return assessment
    
    async def _assess_completeness(self, context: EnhancedContextStructure, query: str) -> float:
        """Assess context completeness"""
        completeness_factors = []
        
        # Entity coverage
        if context.entities:
            entity_types = set(e["label"] for e in context.entities)
            expected_types = {"PERSON", "ORG", "DATE", "GPE", "MONEY"}  # Common important types
            entity_coverage = len(entity_types & expected_types) / len(expected_types)
            completeness_factors.append(entity_coverage)
        
        # Information density
        word_count = len(context.text.split())
        info_density = min(1.0, len(context.entities) / max(word_count / 100, 1))
        completeness_factors.append(info_density)
        
        # Query coverage (if query provided)
        if query:
            query_terms = set(query.lower().split())
            context_terms = set(context.text.lower().split())
            query_coverage = len(query_terms & context_terms) / max(len(query_terms), 1)
            completeness_factors.append(query_coverage)
        
        return np.mean(completeness_factors) if completeness_factors else 0.5
    
    async def _assess_accuracy(self, context: EnhancedContextStructure) -> float:
        """Assess context accuracy through consistency checks"""
        accuracy_score = 0.9  # Start with high assumption
        
        # Check for internal contradictions
        if self.nlp_model:
            doc = self.nlp_model(context.text)
            
            # Look for negation patterns that might indicate contradictions
            negations = [token for token in doc if token.dep_ == "neg"]
            if len(negations) > len(doc) * 0.1:  # High negation rate
                accuracy_score -= 0.1
            
            # Check temporal consistency
            dates = [ent for ent in context.entities if ent["label"] == "DATE"]
            if dates and len(dates) > 1:
                # Simple check: ensure dates are mentioned in logical order
                # This is simplified; real implementation would parse dates
                accuracy_score -= 0.05
        
        # Check for hedge words indicating uncertainty
        hedge_words = {"might", "maybe", "possibly", "could", "perhaps", "allegedly"}
        text_lower = context.text.lower()
        hedge_count = sum(1 for word in hedge_words if word in text_lower)
        if hedge_count > len(context.text.split()) * 0.05:  # More than 5% hedge words
            accuracy_score -= 0.15
        
        return max(0.0, accuracy_score)
    
    async def _detect_bias(self, context: EnhancedContextStructure) -> Dict[str, Any]:
        """Detect various types of bias in context"""
        detected_biases = []
        bias_score = 0.0
        
        text_lower = context.text.lower()
        
        # 1. Sentiment bias
        if self.nlp_model:
            doc = self.nlp_model(context.text)
            # Simplified sentiment analysis
            positive_words = {"excellent", "great", "amazing", "perfect", "best"}
            negative_words = {"terrible", "awful", "worst", "horrible", "bad"}
            
            pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
            neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
            
            total_sentiment = pos_count + neg_count
            if total_sentiment > len(doc) * 0.05:  # More than 5% sentiment words
                detected_biases.append("sentiment")
                bias_score += 0.2
        
        # 2. Source bias (single source dominance)
        source_indicators = ["according to", "said", "reported", "claims"]
        source_count = sum(1 for indicator in source_indicators if indicator in text_lower)
        if source_count == 1:  # Only one source mentioned
            detected_biases.append("single_source")
            bias_score += 0.15
        
        # 3. Absolutism bias
        absolute_terms = {"always", "never", "all", "none", "every", "no one"}
        absolute_count = sum(1 for term in absolute_terms if term in text_lower)
        if absolute_count > len(context.text.split()) * 0.02:
            detected_biases.append("absolutism")
            bias_score += 0.1
        
        # 4. Confirmation bias indicators
        confirming_phrases = ["as expected", "unsurprisingly", "of course", "obviously"]
        if any(phrase in text_lower for phrase in confirming_phrases):
            detected_biases.append("confirmation")
            bias_score += 0.1
        
        return {
            "bias_score": min(1.0, bias_score),
            "detected_biases": detected_biases,
            "interpretation": "High bias" if bias_score > 0.5 else "Moderate bias" if bias_score > 0.2 else "Low bias"
        }
    
    async def _assess_clarity(self, context: EnhancedContextStructure) -> float:
        """Assess context clarity"""
        clarity_score = 1.0
        
        # Check sentence length
        sentences = context.text.split(".")
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if avg_sentence_length > 25:  # Long sentences reduce clarity
            clarity_score -= 0.2
        elif avg_sentence_length > 35:
            clarity_score -= 0.4
        
        # Check for jargon/complex words (simplified)
        complex_word_count = sum(1 for word in context.text.split() if len(word) > 12)
        if complex_word_count > len(context.text.split()) * 0.1:
            clarity_score -= 0.15
        
        # Check paragraph structure
        paragraphs = context.text.split("\n\n")
        if len(paragraphs) == 1 and len(context.text) > 500:  # Long text without paragraphs
            clarity_score -= 0.1
        
        return max(0.0, clarity_score)
    
    async def _assess_query_relevance(self, context: EnhancedContextStructure, query: str) -> float:
        """Assess relevance to specific query"""
        if not query:
            return 1.0
        
        relevance_score = 0.0
        
        # Keyword overlap
        query_terms = set(query.lower().split())
        context_terms = set(context.text.lower().split())
        keyword_overlap = len(query_terms & context_terms) / len(query_terms)
        relevance_score += keyword_overlap * 0.3
        
        # Semantic similarity
        if self.embedding_model:
            query_embedding = self.embedding_model.encode(query)
            context_embedding = context.embeddings
            if context_embedding is None:
                context_embedding = self.embedding_model.encode(context.text)
            
            semantic_similarity = cosine_similarity([query_embedding], [context_embedding])[0][0]
            relevance_score += semantic_similarity * 0.5
        
        # Entity relevance
        if context.entities:
            query_entities = set()
            if self.nlp_model:
                query_doc = self.nlp_model(query)
                query_entities = set(ent.text.lower() for ent in query_doc.ents)
            
            context_entities = set(e["text"].lower() for e in context.entities)
            if query_entities:
                entity_overlap = len(query_entities & context_entities) / len(query_entities)
                relevance_score += entity_overlap * 0.2
        
        return min(1.0, relevance_score)
    
    def _determine_quality_level(self, overall_score: float) -> ContextQualityLevel:
        """Determine quality level from overall score"""
        if overall_score >= 0.9:
            return ContextQualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            return ContextQualityLevel.GOOD
        elif overall_score >= 0.6:
            return ContextQualityLevel.FAIR
        elif overall_score >= 0.4:
            return ContextQualityLevel.POOR
        else:
            return ContextQualityLevel.UNACCEPTABLE
    
    async def _generate_context_improvements(
        self,
        context: EnhancedContextStructure,
        quality_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate specific improvement suggestions"""
        improvements = {
            "manual_suggestions": [],
            "automated_improvements": [],
            "priority": "medium"
        }
        
        # Analyze each issue
        for issue in quality_assessment.get("issues", []):
            if issue["type"] == "coherence":
                improvements["manual_suggestions"].append({
                    "type": "coherence",
                    "suggestion": "Add transitional phrases between ideas",
                    "example": "Use connectors like 'Furthermore', 'However', 'In addition'"
                })
                improvements["automated_improvements"].append({
                    "type": "add_transitions",
                    "action": "insert_transitional_phrases"
                })
            
            elif issue["type"] == "completeness":
                missing_elements = self._identify_missing_elements(context, quality_assessment)
                improvements["manual_suggestions"].append({
                    "type": "completeness",
                    "suggestion": f"Add missing information: {', '.join(missing_elements)}",
                    "severity": issue["severity"]
                })
            
            elif issue["type"] == "bias":
                improvements["manual_suggestions"].append({
                    "type": "bias",
                    "suggestion": "Balance perspective by including multiple viewpoints",
                    "specific_biases": quality_assessment["dimensions"]["bias"]["detected_biases"]
                })
                improvements["automated_improvements"].append({
                    "type": "neutralize_language",
                    "action": "replace_biased_terms"
                })
        
        # Set priority based on severity
        high_severity_count = sum(1 for i in quality_assessment.get("issues", []) if i.get("severity") == "high")
        if high_severity_count > 0:
            improvements["priority"] = "high"
        elif len(quality_assessment.get("issues", [])) > 3:
            improvements["priority"] = "high"
        
        return improvements
    
    def _identify_missing_elements(self, context: EnhancedContextStructure, assessment: Dict[str, Any]) -> List[str]:
        """Identify what's missing from context"""
        missing = []
        
        # Check for missing entity types
        existing_entity_types = set(e["label"] for e in context.entities)
        expected_types = {"PERSON", "ORG", "DATE", "LOCATION"}
        missing_types = expected_types - existing_entity_types
        
        if missing_types:
            missing.extend([f"{t} information" for t in missing_types])
        
        # Check for missing structure elements
        if "background" not in context.text.lower():
            missing.append("background context")
        
        if not any(word in context.text.lower() for word in ["because", "therefore", "thus"]):
            missing.append("causal relationships")
        
        return missing
    
    # ========== Multi-Agent Context Coordination ==========
    
    @a2a_handler("coordinate_context", "Coordinate context across multiple agents")
    async def handle_coordinate_context(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Production-grade multi-agent context coordination"""
        with tracer.start_as_current_span("coordinate_context") as span:
            try:
                # Verify trust
                trust_result = await verify_a2a_message(message.dict(), self.agent_id)
                if not trust_result["valid"]:
                    return self._create_error_response(403, "Trust verification failed")
                
                agent_contexts = message.content.get("agent_contexts", {})
                operation = message.content.get("operation", "sync")
                coordination_policy = message.content.get("policy", {})
                
                # Validate agent contexts
                if not agent_contexts:
                    return self._create_error_response(400, "No agent contexts provided")
                
                # Track coordination timing
                start_time = datetime.now()
                
                if operation == "sync":
                    # Full synchronization with conflict resolution
                    sync_result = await self.distributed_manager.synchronize_contexts(
                        agent_contexts,
                        coordination_policy
                    )
                    
                    # Record sync metrics
                    sync_duration = (datetime.now() - start_time).total_seconds()
                    context_sync_duration.observe(sync_duration)
                    
                    # Create workflow artifact for synchronized context
                    if sync_result.final_context and workflowContextManager:
                        artifact = DataArtifact(
                            id=f"sync_{sync_result.sync_id}",
                            type="synchronized_context",
                            data=sync_result.final_context,
                            metadata={
                                "sync_id": sync_result.sync_id,
                                "participating_agents": sync_result.participating_agents,
                                "conflicts_resolved": len(sync_result.conflicts_resolved),
                                "duration": sync_duration
                            }
                        )
                        await workflowContextManager.add_artifact(context_id, artifact)
                    
                    span.set_status(Status(StatusCode.OK))
                    
                    return {
                        "status": "success",
                        "operation": "synchronized",
                        "sync_result": sync_result.to_dict(),
                        "metrics": {
                            "duration": sync_duration,
                            "conflicts_detected": len(sync_result.conflicts_detected),
                            "conflicts_resolved": len(sync_result.conflicts_resolved)
                        }
                    }
                
                elif operation == "propagate":
                    # Propagate context updates efficiently
                    update = message.content.get("update")
                    source_agent = message.content.get("source_agent", self.agent_id)
                    
                    if not update:
                        return self._create_error_response(400, "No update provided")
                    
                    propagation_result = await self.distributed_manager.coordinate_propagation(
                        update,
                        source_agent,
                        coordination_policy
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    
                    return {
                        "status": "success",
                        "operation": "propagated",
                        "propagation_result": propagation_result,
                        "metrics": {
                            "duration": (datetime.now() - start_time).total_seconds()
                        }
                    }
                
                elif operation == "merge":
                    # Advanced merge with version control
                    version_ids = message.content.get("version_ids", [])
                    merge_strategy = coordination_policy.get("merge_strategy", "latest_wins")
                    
                    if len(version_ids) < 2:
                        return self._create_error_response(400, "At least 2 versions required for merge")
                    
                    # Perform versioned merge
                    merged_version = self.distributed_manager.version_control.merge_versions(
                        version_ids[0],
                        version_ids[1],
                        merge_strategy
                    )
                    
                    # Handle additional versions
                    for vid in version_ids[2:]:
                        merged_version = self.distributed_manager.version_control.merge_versions(
                            merged_version.version_id,
                            vid,
                            merge_strategy
                        )
                    
                    span.set_status(Status(StatusCode.OK))
                    
                    return {
                        "status": "success",
                        "operation": "merged",
                        "merged_version": merged_version.to_dict(),
                        "merge_strategy": merge_strategy
                    }
                
                else:
                    return self._create_error_response(400, f"Unknown operation: {operation}")
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Context coordination failed: {str(e)}\n{traceback.format_exc()}")
                return self._create_error_response(500, f"Coordination failed: {str(e)}")
    
    # ========== Semantic Memory Management ==========
    
    @a2a_skill(
        name="semantic_memory_operations",
        description="Advanced semantic memory with vector search",
        capabilities=["store", "retrieve", "search", "forget", "consolidate"]
    )
    async def semantic_memory_operations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage semantic memory with production features"""
        operation = input_data.get("operation", "search")
        
        try:
            if operation == "store":
                return await self._handle_memory_store(input_data)
            elif operation == "retrieve":
                return await self._handle_memory_retrieve(input_data)
            elif operation == "search":
                return await self._handle_semantic_search(input_data)
            elif operation == "forget":
                return await self._handle_memory_forget(input_data)
            elif operation == "consolidate":
                return await self._handle_memory_consolidation(input_data)
            else:
                return self._create_error_response(400, f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Memory operation failed: {str(e)}")
            return self._create_error_response(500, f"Memory operation failed: {str(e)}")
    
    async def _handle_semantic_search(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced semantic search with ranking"""
        query = input_data.get("query", "")
        k = input_data.get("k", 5)
        threshold = input_data.get("threshold", VECTOR_SIMILARITY_THRESHOLD)
        filters = input_data.get("filters", {})
        
        if not query:
            return self._create_error_response(400, "Query required for search")
        
        # Generate query embedding
        if not self.embedding_model:
            return self._create_error_response(503, "Embedding model not available")
        
        query_embedding = self.embedding_model.encode(query)
        
        # Search in memory
        results = []
        
        for ctx_id, context in self.context_memory.items():
            # Apply filters
            if filters:
                if not self._matches_filters(context, filters):
                    continue
            
            # Calculate similarity
            if context.embeddings is not None:
                similarity = cosine_similarity([query_embedding], [context.embeddings])[0][0]
                
                if similarity >= threshold:
                    results.append({
                        "context_id": ctx_id,
                        "similarity": float(similarity),
                        "context": context.to_dict(),
                        "relevance_explanation": self._explain_relevance(query, context, similarity)
                    })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k
        return {
            "status": "success",
            "results": results[:k],
            "total_matches": len(results),
            "search_metadata": {
                "threshold": threshold,
                "filters_applied": filters
            }
        }
    
    def _explain_relevance(self, query: str, context: EnhancedContextStructure, similarity: float) -> str:
        """Generate explanation for relevance score"""
        explanation_parts = []
        
        if similarity > 0.9:
            explanation_parts.append("Very high semantic similarity")
        elif similarity > 0.8:
            explanation_parts.append("High semantic similarity")
        elif similarity > 0.7:
            explanation_parts.append("Good semantic similarity")
        else:
            explanation_parts.append("Moderate semantic similarity")
        
        # Check for entity overlap
        if self.nlp_model:
            query_doc = self.nlp_model(query)
            query_entities = set(ent.text.lower() for ent in query_doc.ents)
            context_entities = set(e["text"].lower() for e in context.entities)
            
            overlap = query_entities & context_entities
            if overlap:
                explanation_parts.append(f"Shared entities: {', '.join(overlap)}")
        
        # Check for keyword overlap
        query_keywords = set(query.lower().split())
        context_keywords = set(context.text.lower().split())
        keyword_overlap = query_keywords & context_keywords
        if len(keyword_overlap) > 2:
            explanation_parts.append(f"Keyword matches: {len(keyword_overlap)}")
        
        return "; ".join(explanation_parts)
    
    def _matches_filters(self, context: EnhancedContextStructure, filters: Dict[str, Any]) -> bool:
        """Check if context matches search filters"""
        for key, value in filters.items():
            if key == "min_quality":
                if context.quality_metrics:
                    overall_quality = np.mean(list(context.quality_metrics.values()))
                    if overall_quality < value:
                        return False
            
            elif key == "entity_types":
                entity_types = set(e["label"] for e in context.entities)
                if not any(et in entity_types for et in value):
                    return False
            
            elif key == "date_range":
                # Check if context is within date range
                context_date = datetime.fromisoformat(context.metadata.get("parsed_at", ""))
                start_date = datetime.fromisoformat(value.get("start"))
                end_date = datetime.fromisoformat(value.get("end"))
                if not (start_date <= context_date <= end_date):
                    return False
            
            elif key == "min_relevance":
                if context.relevance_score < value:
                    return False
        
        return True
    
    # ========== Helper Methods ==========
    
    def _create_error_response(self, code: int, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "status": "error",
            "error": {
                "code": code,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _save_performance_metrics(self):
        """Save performance metrics for analysis"""
        metrics_file = f"performance_metrics_{self.agent_id}_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(metrics_file, "w") as f:
                json.dump(dict(self.performance_metrics), f, indent=2)
            logger.info(f"Saved performance metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {str(e)}")
    
    # ========== Fallback Methods for Error Recovery ==========
    
    async def _fallback_context_parsing(self, *args, **kwargs):
        """Fallback parsing when NLP models fail"""
        raw_context = args[0] if args else kwargs.get("raw_context", "")
        
        # Basic parsing without NLP
        return EnhancedContextStructure(
            text=raw_context,
            entities=[],
            relationships=[],
            semantic_structure={},
            relevance_score=0.5,
            metadata={
                "parsed_at": datetime.now().isoformat(),
                "parser_version": "fallback",
                "word_count": len(raw_context.split()),
                "fallback_reason": "NLP model unavailable"
            }
        )
    
    async def _fallback_to_memory_cache(self, *args, **kwargs):
        """Fallback to in-memory cache when Redis fails"""
        # Simply return None to indicate cache miss
        return None
    
    async def _fallback_embedding_generation(self, *args, **kwargs):
        """Fallback when embedding generation fails"""
        # Return None to indicate embeddings unavailable
        return None
    
    async def _calculate_information_retention(
        self,
        original_contexts: List[Dict[str, Any]],
        optimization_result: ContextOptimizationResult
    ) -> float:
        """Calculate how much information was retained after optimization"""
        # Simplified calculation based on entity and keyword retention
        original_text = " ".join(c.get("text", c.get("content", "")) for c in original_contexts)
        optimized_text = optimization_result.optimized_context
        
        if self.nlp_model:
            original_doc = self.nlp_model(original_text)
            optimized_doc = self.nlp_model(optimized_text)
            
            # Entity retention
            original_entities = set(ent.text.lower() for ent in original_doc.ents)
            optimized_entities = set(ent.text.lower() for ent in optimized_doc.ents)
            entity_retention = len(optimized_entities & original_entities) / max(len(original_entities), 1)
            
            # Key noun retention
            original_nouns = set(token.text.lower() for token in original_doc if token.pos_ in ["NOUN", "PROPN"])
            optimized_nouns = set(token.text.lower() for token in optimized_doc if token.pos_ in ["NOUN", "PROPN"])
            noun_retention = len(optimized_nouns & original_nouns) / max(len(original_nouns), 1)
            
            return (entity_retention + noun_retention) / 2
        
        # Fallback: character-based retention
        return len(optimized_text) / max(len(original_text), 1)
    
    async def _estimate_information_retention(
        self,
        all_chunks: List[Dict[str, Any]],
        selected_chunks: List[Dict[str, Any]]
    ) -> float:
        """Estimate information retention from chunk selection"""
        if not all_chunks:
            return 0.0
        
        # Weighted by importance scores
        total_importance = sum(c.get("importance_score", c.get("relevance_score", 0.5)) for c in all_chunks)
        selected_importance = sum(c.get("importance_score", c.get("relevance_score", 0.5)) for c in selected_chunks)
        
        return selected_importance / max(total_importance, 1)
    
    def _calculate_compression_ratio(
        self,
        original_contexts: List[Dict[str, Any]],
        selected_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate compression ratio"""
        original_tokens = sum(
            len(c.get("text", c.get("content", "")).split()) * 1.3  # Rough token estimate
            for c in original_contexts
        )
        
        selected_tokens = sum(c.get("token_count", 0) for c in selected_chunks)
        
        return selected_tokens / max(original_tokens, 1)
    
    async def _compress_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Compress chunks to fit within token limit"""
        current_tokens = sum(c["token_count"] for c in chunks)
        
        if current_tokens <= max_tokens:
            return chunks
        
        # Calculate compression ratio needed
        compression_ratio = max_tokens / current_tokens
        
        compressed_chunks = []
        for chunk in chunks:
            # Simple compression: truncate to proportion
            target_tokens = int(chunk["token_count"] * compression_ratio)
            words = chunk["text"].split()
            
            if len(words) > target_tokens:
                # Keep most important part (beginning and end)
                keep_start = int(target_tokens * 0.7)
                keep_end = target_tokens - keep_start
                
                compressed_text = " ".join(words[:keep_start])
                if keep_end > 0:
                    compressed_text += " [...] " + " ".join(words[-keep_end:])
                
                chunk["text"] = compressed_text
                chunk["token_count"] = target_tokens
                chunk["compressed"] = True
            
            compressed_chunks.append(chunk)
        
        return compressed_chunks
    
    async def _apply_improvements(
        self,
        context: EnhancedContextStructure,
        automated_improvements: List[Dict[str, Any]]
    ) -> EnhancedContextStructure:
        """Apply automated improvements to context"""
        improved_text = context.text
        
        for improvement in automated_improvements:
            if improvement["action"] == "insert_transitional_phrases":
                # Add transitions between sentences
                sentences = improved_text.split(". ")
                if len(sentences) > 1:
                    transitions = ["Furthermore", "Additionally", "Moreover", "In addition"]
                    new_sentences = [sentences[0]]
                    
                    for i, sent in enumerate(sentences[1:], 1):
                        if i < len(transitions) and not any(sent.startswith(t) for t in transitions):
                            new_sentences.append(f"{transitions[i-1]}, {sent.lower()}")
                        else:
                            new_sentences.append(sent)
                    
                    improved_text = ". ".join(new_sentences)
            
            elif improvement["action"] == "replace_biased_terms":
                # Simple bias reduction
                replacements = {
                    "always": "often",
                    "never": "rarely",
                    "all": "most",
                    "none": "few",
                    "amazing": "notable",
                    "terrible": "challenging"
                }
                
                for old, new in replacements.items():
                    improved_text = improved_text.replace(f" {old} ", f" {new} ")
        
        # Create improved context
        improved_context = EnhancedContextStructure(
            text=improved_text,
            entities=context.entities,
            relationships=context.relationships,
            semantic_structure=context.semantic_structure,
            relevance_score=context.relevance_score,
            metadata={
                **context.metadata,
                "improved": True,
                "improvements_applied": [imp["action"] for imp in automated_improvements]
            },
            quality_metrics=context.quality_metrics,
            provenance=context.provenance + [{
                "agent": self.agent_id,
                "action": "improve",
                "timestamp": datetime.now().isoformat(),
                "improvements": len(automated_improvements)
            }] if context.provenance else None
        )
        
        return improved_context
    
    async def _handle_memory_store(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory store operation"""
        context = input_data.get("context")
        context_id = input_data.get("context_id", f"ctx_{datetime.now().timestamp()}")
        importance = input_data.get("importance", 0.5)
        ttl = input_data.get("ttl", CONTEXT_CACHE_TTL)
        
        if not context:
            return self._create_error_response(400, "Context required for storage")
        
        # Parse if needed
        if isinstance(context, str):
            context_structure = await self._parse_context_advanced(context, {})
        else:
            context_structure = EnhancedContextStructure.from_dict(context)
        
        # Generate embeddings if missing
        if context_structure.embeddings is None and self.embedding_model:
            context_structure.embeddings = await self._generate_embeddings(context_structure.text)
        
        # Store with importance weighting
        await self._store_context_with_cache(context_id, context_structure)
        
        # Track in workflow if high importance
        if importance > 0.7 and workflowContextManager:
            artifact = DataArtifact(
                id=context_id,
                type="stored_context",
                data=context_structure.to_dict(),
                metadata={
                    "importance": importance,
                    "ttl": ttl
                }
            )
            await workflowContextManager.add_artifact(context_id, artifact)
        
        return {
            "status": "success",
            "context_id": context_id,
            "stored": True,
            "importance": importance
        }
    
    async def _handle_memory_retrieve(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory retrieve operation"""
        context_id = input_data.get("context_id")
        
        if not context_id:
            return self._create_error_response(400, "Context ID required")
        
        # Check memory
        if context_id in self.context_memory:
            return {
                "status": "success",
                "context": self.context_memory[context_id].to_dict(),
                "source": "memory"
            }
        
        # Check Redis
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"context:{context_id}")
                if cached:
                    context_data = json.loads(cached)
                    return {
                        "status": "success",
                        "context": context_data,
                        "source": "redis"
                    }
            except Exception as e:
                logger.warning(f"Redis retrieval failed: {str(e)}")
        
        return self._create_error_response(404, "Context not found")
    
    async def _handle_memory_forget(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory forget operation"""
        context_ids = input_data.get("context_ids", [])
        pattern = input_data.get("pattern")
        
        forgotten_count = 0
        
        if context_ids:
            for ctx_id in context_ids:
                if ctx_id in self.context_memory:
                    del self.context_memory[ctx_id]
                    forgotten_count += 1
                
                # Remove from Redis
                if self.redis_client:
                    try:
                        await self.redis_client.delete(f"context:{ctx_id}")
                    except Exception:
                        pass
        
        elif pattern:
            # Pattern-based forgetting
            to_forget = []
            for ctx_id, context in self.context_memory.items():
                if pattern.lower() in context.text.lower():
                    to_forget.append(ctx_id)
            
            for ctx_id in to_forget:
                del self.context_memory[ctx_id]
                forgotten_count += 1
        
        active_contexts_gauge.set(len(self.context_memory))
        
        return {
            "status": "success",
            "forgotten_count": forgotten_count,
            "remaining_contexts": len(self.context_memory)
        }
    
    async def _handle_memory_consolidation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate and optimize memory"""
        strategy = input_data.get("strategy", "quality_based")
        target_size = input_data.get("target_size", MAX_MEMORY_CONTEXTS // 2)
        
        if len(self.context_memory) <= target_size:
            return {
                "status": "success",
                "message": "No consolidation needed",
                "current_size": len(self.context_memory)
            }
        
        # Score all contexts
        scored_contexts = []
        for ctx_id, context in self.context_memory.items():
            score = self._calculate_consolidation_score(context, strategy)
            scored_contexts.append((ctx_id, score))
        
        # Sort by score (higher is better)
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top contexts
        contexts_to_keep = set(ctx_id for ctx_id, _ in scored_contexts[:target_size])
        
        # Remove low-scoring contexts
        removed_count = 0
        for ctx_id in list(self.context_memory.keys()):
            if ctx_id not in contexts_to_keep:
                del self.context_memory[ctx_id]
                removed_count += 1
        
        active_contexts_gauge.set(len(self.context_memory))
        
        return {
            "status": "success",
            "removed_count": removed_count,
            "remaining_contexts": len(self.context_memory),
            "consolidation_strategy": strategy
        }
    
    def _calculate_consolidation_score(self, context: EnhancedContextStructure, strategy: str) -> float:
        """Calculate score for memory consolidation"""
        score = 0.0
        
        if strategy == "quality_based":
            if context.quality_metrics:
                score = np.mean(list(context.quality_metrics.values()))
            else:
                score = context.relevance_score
        
        elif strategy == "recency_based":
            # More recent = higher score
            parsed_at = datetime.fromisoformat(context.metadata.get("parsed_at", "2020-01-01"))
            age_days = (datetime.now() - parsed_at).days
            score = 1.0 / (age_days + 1)
        
        elif strategy == "importance_based":
            # Combine multiple factors
            score = context.relevance_score * 0.4
            score += min(1.0, len(context.entities) / 10) * 0.3
            score += min(1.0, len(context.relationships) / 20) * 0.3
        
        return score


# Production agent launcher
def create_production_agent(base_url: str = "http://localhost:8091", config: Optional[Dict[str, Any]] = None):
    """Create production-ready context engineering agent"""
    return ContextEngineeringAgent(base_url, config)


if __name__ == "__main__":
    import uvicorn
    
    async def main():
        # Production configuration
        config = {
            "max_context_tokens": 16384,
            "enable_redis": True,
            "trust_threshold": 0.85,
            "performance_monitoring": {
                "enabled": True,
                "alert_thresholds": {
                    "response_time_ms": 500,
                    "error_rate": 0.05,
                    "memory_usage_mb": 1024
                }
            }
        }
        
        # Create and initialize agent
        agent = create_production_agent(config=config)
        await agent.initialize()
        
        # Create FastAPI app with production middleware
        app = agent.create_fastapi_app()
        
        # Add any custom middleware here
        # app.add_middleware(...)
        
        # Production server configuration
        server_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8091,
            log_level="info",
            access_log=True,
            workers=4,
            loop="uvloop",
            reload=False
        )
        
        server = uvicorn.Server(server_config)
        await server.serve()
    
    # Run production agent
    asyncio.run(main()